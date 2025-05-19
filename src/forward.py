import asyncio
import random
from urllib.parse import urljoin
from starlette.background import BackgroundTask
from typing import Any, AsyncGenerator, Union
import aiohttp
from aiohttp_socks import ProxyConnector
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import orjson
from pydantic import BaseModel
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status, Request
from config import Config, logger, cfg_obj
from const import CHAT_COMPLETION_ROUTE, MODELS_ROUTE
from soph import fetch_models, get_new_api_key
from utils import fake_useragent


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动事件
    await forwarder.build_client()
    logger.info("SophNet forwarding launched")

    yield  # 应用运行

    # 关闭事件
    await forwarder.close()
    logger.info("SophNet forwarding closed")


# 使用lifespan上下文管理器创建应用
app = FastAPI(title="sophnet-forwarding", version="0.1.0", lifespan=lifespan)

# CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 消息模型
class ChatMessage(BaseModel):
    role: str
    content: str
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float | None = 1.0
    top_p: float | None = 1.0
    n: int | None = 1
    stream: bool | None = False
    stop: Union[str, list[str]] | None = None
    max_tokens: int | None = None
    presence_penalty: float | None = 0.0
    frequency_penalty: float | None = 0.0
    logit_bias: dict[str, float] | None = None
    user: str | None = None


def apply_field_aliases(data: dict[str, Any]) -> dict:
    """应用字段别名映射规则，转换字段名称

    Args:
        data (dict): 原始请求数据

    Returns:
        dict: 转换后的请求数据
    """
    if not cfg_obj.field_aliases or not isinstance(data, dict):
        return data

    result: dict[str, Any] = {}
    for key, value in data.items():
        # 如果当前字段有映射规则，使用目标字段名
        target_key = cfg_obj.field_aliases.get(key, key)

        # 递归处理嵌套字典
        if isinstance(value, dict):
            result[target_key] = apply_field_aliases(value)
        # 递归处理列表中的每个字典
        elif isinstance(value, list):
            result[target_key] = [apply_field_aliases(item) for item in value]
        else:
            result[target_key] = value

    return result


class ChatForwarder:
    """OpenAI聊天完成API转发器"""

    MAX_RETRIES = 1  # Max retries (e.g., 1 means 1 original attempt + 1 retry)

    # 添加IP跟踪字典
    request_ips = {}

    def __init__(self, config: Config):
        self.cfg = config.model_copy()
        self.client = None
        self.timeout = aiohttp.ClientTimeout(connect=self.cfg.timeout)
        # 保存当前请求的前端key，用于在处理响应时切换API key
        self.inbound_key: str | None = None

    async def build_client(self):
        """异步构建HTTP客户端"""
        # 如果使用SOCKS代理
        if self.cfg.proxy and "socks" in self.cfg.proxy.lower():
            connector = ProxyConnector.from_url(
                self.cfg.proxy,
                limit=self.cfg.connection_limit,
                limit_per_host=self.cfg.connection_limit_per_host,
                force_close=self.cfg.force_close,
                enable_cleanup_closed=True,
                ssl=False,
            )
        else:
            # 原有的HTTP代理配置
            connector = aiohttp.TCPConnector(
                limit=self.cfg.connection_limit,
                limit_per_host=self.cfg.connection_limit_per_host,
                force_close=self.cfg.force_close,
                enable_cleanup_closed=True,
                ssl=False,
                keepalive_timeout=None
                if self.cfg.force_close
                else self.cfg.keepalive_timeout,
            )

        self.client = aiohttp.ClientSession(
            connector=connector,
            timeout=self.timeout,
            json_serialize=lambda x: orjson.dumps(x).decode("utf-8"),
        )

    async def close(self):
        """关闭HTTP客户端连接"""
        if self.client:
            await self.client.close()

    def validate_request_ip(self, ip: str | None):
        """验证请求IP是否允许访问
        Args:
            ip (str | None): 客户端IP地址, 如果为 None 且黑/白名单不为空, 那么访问不允许
        Raises:
            HTTPException: 如果IP不在白名单内或在黑名单内, 抛出403错误
        Note:
            黑名单优先
        """
        if self.cfg.validate_ip:
            # 黑名单优先
            if (
                not ip
                or (self.cfg.ip_black_list and ip in self.cfg.ip_black_list)
                or (self.cfg.ip_whitelist and ip not in self.cfg.ip_whitelist)
            ):
                logger.info(f"IP {ip} access denied")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Forbidden Error",
                )

    def get_client_ip(self, request: Request) -> str | None:
        """从请求中获取客户端IP地址
        Args:
            request (Request): FastAPI请求对象
        Returns:
            str | None: 客户端IP地址
        """
        # 添加转发代理头部支持
        for header in ["x-real-ip", "x-forwarded-for"]:
            ip = request.headers.get(header)
            if ip:
                # 如果是多个IP的列表，取第一个
                return ip.split(",", maxsplit=1)[0].strip()

        # 回退到客户端直连IP
        return request.client.host if request.client else None

    async def handle_authorization(self, auth_header: str) -> str:
        """处理授权信息
        Args:
            auth_header (str): 授权头部信息
        Returns:
            str: 处理后的授权头部信息
        """
        if not auth_header:
            logger.warning("no auth proviced")
            return auth_header
        if not auth_header.startswith("Bearer "):
            logger.warning("not supported auth type")
            return auth_header

        # 假设auth_header格式为"Bearer YOUR_API_KEY"
        bearer_prefix = "Bearer "
        self.inbound_key = auth_header[len(bearer_prefix) :]
        if self.inbound_key in self.cfg.api_keys:
            # 临时处理，始终使用新的token
            new_key = await get_new_api_key(self.inbound_key, self.client)
            if new_key:
                return bearer_prefix + new_key

        return auth_header

    async def send_request(
        self, method: str, url: str, headers: dict, data: bytes | None = None
    ):
        """发送请求到目标服务器

        Args:
            method (str): HTTP方法
            url (str): 请求URL
            headers (dict): 请求头
            data (bytes, optional): 请求体数据

        Returns:
            aiohttp.ClientResponse: 响应对象
        """
        assert self.client
        try:
            # 在发送请求前获取当前的IP地址
            request_id = random.randint(1, 1000000)
            # 发送请求
            response = await self.client.request(
                method=method,
                url=url,
                data=data,
                headers=headers,
                proxy=self.cfg.proxy,
                allow_redirects=True,
            )

            # 获取请求使用的IP地址
            if self.cfg.proxy:
                # 发起一个请求来获取当前IP
                try:
                    async with self.client.get(
                        "https://api.ipify.org",
                        proxy=self.cfg.proxy,
                        timeout=5,  # type: ignore
                    ) as ip_response:
                        if ip_response.status == 200:
                            current_ip = await ip_response.text()
                            # 记录请求ID和IP
                            self.request_ips[request_id] = current_ip
                            logger.info(
                                f"Req ID {request_id}, IP: {current_ip}"
                            )
                except Exception as e:
                    logger.error(f"Failed to get proxied ip: {str(e)}")

            return response
        except (
            aiohttp.ServerTimeoutError,
            aiohttp.ServerConnectionError,
            aiohttp.ServerDisconnectedError,
            asyncio.TimeoutError,
        ) as e:
            error_msg = (
                f"Failed to connect to {self.cfg.api_host} : {type(e).__name__}: {e}"
            )
            logger.error(error_msg)
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=error_msg,
            )
        except Exception as e:
            error_msg = f"Error in handling request: {type(e).__name__}: {e}"
            logger.error(error_msg)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_msg,
            )

    async def aiter_bytes(
        self, response: aiohttp.ClientResponse
    ) -> AsyncGenerator[bytes, Any]:
        try:
            # This non-streaming path in aiter_bytes should ideally not be hit if reverse_proxy
            # only passes successful (status 200, text/event-stream) responses to it.
            if not response.headers.get("content-type", "").startswith(
                "text/event-stream"
            ):
                content = await response.read()
                yield content
                return

            # 使用更大的缓冲区读取流
            buffer_size = 64 * 1024  # 64KB
            # Expected path: Processing a live, successful stream
            async for chunk in response.content.iter_chunked(buffer_size):
                yield chunk
        except Exception as e:
            logger.error(f"Error in aiter_bytes response processing: {str(e)}")
            raise  # Re-raise to ensure stream termination if error is critical

    async def prepare_request_info(self, request: Request) -> dict:
        """准备请求信息

        Args:
            request (Request): FastAPI请求对象

        Returns:
            dict: 请求信息
        """
        # 使用自定义目标路径
        url = urljoin(self.cfg.api_host, self.cfg.forward_chat_path)

        # 提取请求头
        headers = dict(request.headers)

        # 处理授权
        auth_header = headers.get("authorization")
        if auth_header:
            headers["authorization"] = await self.handle_authorization(auth_header)

        headers["Accept"] = "*/*"
        headers["Accept-Encoding"] = "gzip, deflate, br"  # 启用压缩以减少网络流量

        # 移除不需要转发的头信息
        headers_to_remove = ["host", "content-length"]
        for header in headers_to_remove:
            if header in headers:
                del headers[header]

        return {
            "method": request.method,
            "url": url,
            "headers": headers,
        }

    async def reverse_proxy(self, request: Request) -> StreamingResponse:
        """反向代理处理程序

        Args:
            request (Request): FastAPI请求对象

        Returns:
            StreamingResponse: 流式响应
        """
        assert self.client
        client_ip = self.get_client_ip(request)
        self.validate_request_ip(client_ip)

        # 创建请求ID用于跟踪
        request_id = random.randint(1, 1000000)
        logger.info(f"Req from {client_ip}, ID: {request_id}")

        original_data = await request.body()
        data: bytes | None = None  # Declare data here to be used in loop

        if original_data:
            try:
                payload_dict = orjson.loads(original_data)

                # 如果未设置模型和参数，添加默认值
                if "model" not in payload_dict:
                    payload_dict["model"] = self.cfg.default_model
                if "temperature" not in payload_dict:
                    payload_dict["temperature"] = self.cfg.default_temperature
                if "max_tokens" not in payload_dict:
                    payload_dict["max_tokens"] = self.cfg.default_max_tokens

                # 应用字段别名
                if self.cfg.field_aliases:
                    transformed_dict = apply_field_aliases(payload_dict)
                    payload_dict = transformed_dict

                # 序列化回bytes
                data = orjson.dumps(payload_dict)
            except Exception as e:
                logger.error(f"Error handling request body: {e}")
                data = original_data
        else:
            data = original_data

        # 防检测机制，使用随机的User-Agent和请求头
        if data:
            try:
                payload_dict = orjson.loads(data)
                headers = payload_dict.get("headers", {})

                # 设置随机请求头
                headers["User-Agent"] = fake_useragent()
                headers["Accept"] = "application/json, text/plain, */*"
                headers["Accept-Language"] = f"en-US,en;q=0.{random.randint(7, 9)}"
                headers["Accept-Encoding"] = "gzip, deflate, br"
                # 根据配置决定是否强制关闭连接
                if self.cfg.force_close:
                    headers["Connection"] = "close"
                else:
                    headers["Connection"] = "keep-alive"

                # 随机生成IP地址
                random_ip = f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}"

                # 随机生成Referer
                referers = [
                    "https://dify.app/",
                    "https://chat.openai.com/",
                    "https://platform.openai.com/",
                    "https://www.google.com/",
                    "https://www.bing.com/",
                ]
                headers["Referer"] = random.choice(referers)

                # 添加随机Origin
                origins = [
                    "https://dify.app",
                    "https://chat.openai.com",
                    "https://platform.openai.com",
                    "https://www.google.com",
                    "https://www.bing.com",
                ]
                headers["Origin"] = random.choice(origins)

                # 添加代理IP相关头
                headers["X-Forwarded-For"] = random_ip
                headers["Client-IP"] = random_ip
                headers["x-real-ip"] = random_ip
                # 使用当前代理池所指向的IP，而不是随机生成的IP
                if self.cfg.proxy:
                    try:
                        # 尝试获取当前代理的真实IP
                        async with self.client.get(
                            "https://api.ipify.org",
                            proxy=self.cfg.proxy,
                            timeout=5,  # type: ignore
                        ) as ip_response:
                            if ip_response.status == 200:
                                proxy_ip = await ip_response.text()
                                headers["X-Forwarded-For"] = proxy_ip
                                headers["Client-IP"] = proxy_ip
                                headers["x-real-ip"] = proxy_ip
                                logger.info(f"Use proxied ip in header: {proxy_ip}")
                    except Exception as e:
                        logger.error(
                            f"Failed to fetch proxied ip: {str(e)}"
                        )

                # 更新payload并重新序列化
                payload_dict["headers"] = headers
                data = orjson.dumps(payload_dict)
            except Exception as e:
                logger.error(f"Error in preparing headers: {e}")
        # 记录最终发送到后端的数据
        if data:
            try:
                logger.debug(
                    f"PROXY -> BACKEND (final body decoded): {data.decode('utf-8')}"
                )
            except UnicodeDecodeError:
                logger.debug(f"PROXY -> BACKEND (final body bytes): {data!r}")

        for attempt in range(self.MAX_RETRIES + 1):
            self.current_request_frontend_key = None  # Reset for each attempt

            request_info = await self.prepare_request_info(request)

            auth_header_for_log = request_info["headers"].get("authorization", "None")
            logger.debug(
                f"Attempt {attempt + 1}/{self.MAX_RETRIES + 1} using Authorization: {auth_header_for_log[:25]}..."
            )

            target_response = await self.send_request(
                method=request_info["method"],
                url=request_info["url"],
                headers=request_info["headers"],
                data=data,
            )

            if target_response.status == 200 and target_response.headers.get(
                "content-type", ""
            ).startswith("text/event-stream"):
                logger.debug(
                    f"Attempt {attempt + 1}: Successful stream response (status {target_response.status}). Forwarding."
                )
                return StreamingResponse(
                    self.aiter_bytes(target_response),
                    status_code=target_response.status,
                    media_type=target_response.headers.get("content-type"),
                    background=BackgroundTask(target_response.release),
                )

            response_content = await target_response.read()

            logger.debug(
                f"Attempt {attempt + 1}: Forwarding this response (status: {target_response.status}). No further retries for this error type or max attempts reached."
            )

            # 添加对429错误的专门日志记录
            if target_response.status == 429:
                try:
                    error_message = response_content.decode("utf-8")
                    logger.warning(
                        f"ErrCode: 429 Too Many Requests, URL: {request_info['url']}, Msg: {error_message}"
                    )

                    # 获得此时的ip地址
                    async with self.client.get(
                        "https://api.ipify.org", proxy=self.cfg.proxy, timeout=5 # type: ignore
                    ) as ip_response:
                        if ip_response.status == 200:
                            current_ip = await ip_response.text()
                            logger.warning(
                                f"ErrCode: 429 Too Many Requests, {current_ip=}"
                            )
                except Exception as e:
                    logger.warning(
                        f"ErrCode: 429 Too Many Requests, {str(e)}"
                    )

            async def _final_content_streamer(content_bytes, original_resp_to_release):
                try:
                    yield content_bytes
                finally:
                    if original_resp_to_release:
                        await original_resp_to_release.release()

            final_response_to_send = StreamingResponse(
                _final_content_streamer(response_content, target_response),
                status_code=target_response.status,
                media_type=target_response.headers.get("content-type"),
            )
            break

        return final_response_to_send # type: ignore


# 创建转发器，传入自定义转发路径
forwarder = ChatForwarder(cfg_obj)


@app.get("/health")
def health_check():
    """健康检查接口"""
    return {"status": "ok"}


# 注册转发路由
app.add_route(
    CHAT_COMPLETION_ROUTE,
    forwarder.reverse_proxy,
    methods=["POST"],
)


# 添加模型列表处理函数
@app.get(MODELS_ROUTE)
async def list_models(request: Request):
    """获取可用的模型列表

    Args:
        request (Request): FastAPI请求对象

    Returns:
        JSONResponse: 模型列表
    """
    client_ip = forwarder.get_client_ip(request)
    forwarder.validate_request_ip(client_ip)

    # 获取模型列表，总是使用匿名token
    models = await fetch_models()

    # 返回JSON响应
    return JSONResponse(content=models)


# 添加一个新的路由来查看IP使用情况
@app.get("/proxy-stats")
def proxy_stats():
    """返回代理使用统计信息"""
    return {
        "total_requests": len(forwarder.request_ips),
        "unique_ips": len(set(forwarder.request_ips.values())),
        "recent_ips": dict(list(forwarder.request_ips.items())[-10:]),  # 最近10条记录
    }
