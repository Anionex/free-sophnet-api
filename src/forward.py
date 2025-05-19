import asyncio
import random
from starlette.background import BackgroundTask
from typing import Any, AsyncGenerator, Union
import aiohttp
from aiohttp_socks import ProxyConnector
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import orjson
from pydantic import BaseModel
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status, Request
from config import (
    CURRENT_KEY_MAPPING,
    API_KEYS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    FIELD_ALIASES,
    FORCE_CLOSE,
    IP_BLACKLIST,
    IP_WHITELIST,
    MODELS_ROUTE,
    TIMEOUT,
    VALIDATE_IP,
    logger,
    BASE_URL,
    PROXY_URL_PATH,
    PROXY,
    ROUTE_PREFIX,
    CHAT_COMPLETION_ROUTE,
    CONNECTION_LIMIT,
    CONNECTION_LIMIT_PER_HOST,
    KEEPALIVE_TIMEOUT,
    WORKERS,
)
from soph import get_new_api_key
from utils import fake_useragent


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动事件
    await forwarder.build_client()
    logger.info("OpenAI 聊天代理服务已启动")
    logger.info(f"转发目标: {BASE_URL}")
    logger.info(f"转发路径: {PROXY_URL_PATH}")
    if FIELD_ALIASES:
        logger.info(f"字段别名: {FIELD_ALIASES}")
    if PROXY:
        logger.info(f"使用代理: {PROXY}")
    logger.info(f"监听路径: {ROUTE_PREFIX}{CHAT_COMPLETION_ROUTE}")
    logger.info(
        f"连接池配置: 总连接数={CONNECTION_LIMIT}, 每主机连接数={CONNECTION_LIMIT_PER_HOST}, 连接保活={KEEPALIVE_TIMEOUT}秒"
    )
    logger.info(f"工作进程数: {WORKERS}")

    # 初始化所有API keys映射
    for frontend_key, actual_keys in API_KEYS.items():
        if isinstance(actual_keys, list) and actual_keys:
            CURRENT_KEY_MAPPING[frontend_key] = actual_keys[0]
            logger.info(f"初始化API key映射: {frontend_key} -> {actual_keys[0][:5]}***")

    yield  # 应用运行

    # 关闭事件
    await forwarder.close()
    logger.info("OpenAI 聊天代理服务已关闭")


# 使用lifespan上下文管理器创建应用
app = FastAPI(title="openai-chat-proxy", version="0.1.0", lifespan=lifespan)

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


def apply_field_aliases(data: dict) -> dict:
    """应用字段别名映射规则，转换字段名称

    Args:
        data (dict): 原始请求数据

    Returns:
        dict: 转换后的请求数据
    """
    if not FIELD_ALIASES or not isinstance(data, dict):
        return data

    result = {}
    for key, value in data.items():
        # 如果当前字段有映射规则，使用目标字段名
        target_key = FIELD_ALIASES.get(key, key)

        # 递归处理嵌套字典
        if isinstance(value, dict):
            result[target_key] = apply_field_aliases(value)
        # 递归处理列表中的每个字典
        elif isinstance(value, list):
            result[target_key] = [
                apply_field_aliases(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            result[target_key] = value

    return result


class ChatForwarder:
    """OpenAI聊天完成API转发器"""

    MAX_RETRIES = 1  # Max retries (e.g., 1 means 1 original attempt + 1 retry)

    # 添加IP跟踪字典
    request_ips = {}

    def __init__(self, base_url: str, proxy_url_path: str, proxy=None):
        """初始化转发器

        Args:
            base_url (str): 转发目标的基础URL
            proxy_url_path (str): 转发目标的路径
            proxy (str, optional): 代理服务器地址
        """
        self.base_url = base_url
        self.proxy_url_path = proxy_url_path
        self.proxy = proxy
        self.client = None
        self.timeout = aiohttp.ClientTimeout(connect=TIMEOUT)
        # 保存当前请求的前端key，用于在处理响应时切换API key
        self.current_request_frontend_key = None

    async def build_client(self):
        """异步构建HTTP客户端"""
        # 如果使用SOCKS代理
        if self.proxy and "socks" in self.proxy.lower():
            connector = ProxyConnector.from_url(
                self.proxy,
                **{
                    "limit": CONNECTION_LIMIT,
                    "limit_per_host": CONNECTION_LIMIT_PER_HOST,
                    "force_close": FORCE_CLOSE,
                    "enable_cleanup_closed": True,
                    "ssl": False,
                },
            )
        else:
            # 原有的HTTP代理配置
            connector_args = {
                "limit": CONNECTION_LIMIT,
                "limit_per_host": CONNECTION_LIMIT_PER_HOST,
                "force_close": FORCE_CLOSE,
                "enable_cleanup_closed": True,
                "ssl": False,
            }

            if not FORCE_CLOSE:
                connector_args["keepalive_timeout"] = KEEPALIVE_TIMEOUT

            connector = aiohttp.TCPConnector(**connector_args)

        self.client = aiohttp.ClientSession(
            connector=connector, timeout=self.timeout, json_serialize=orjson.dumps
        )

    async def close(self):
        """关闭HTTP客户端连接"""
        if self.client:
            await self.client.close()

    def validate_request_ip(self, ip: str):
        """验证请求IP是否允许访问
        Args:
            ip (str): 客户端IP地址
        Raises:
            HTTPException: 如果IP不在白名单内或在黑名单内，抛出403错误
        """
        if VALIDATE_IP:
            if (IP_WHITELIST and ip not in IP_WHITELIST) or (
                IP_BLACKLIST and ip in IP_BLACKLIST
            ):
                logger.warning(f"IP {ip} 未授权访问")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Forbidden Error",
                )

    def get_client_ip(self, request: Request) -> str:
        """从请求中获取客户端IP地址
        Args:
            request (Request): FastAPI请求对象
        Returns:
            str: 客户端IP地址
        """
        # 添加转发代理头部支持
        for header in ["x-real-ip", "x-forwarded-for"]:
            ip = request.headers.get(header)
            if ip:
                # 如果是多个IP的列表，取第一个
                return ip.split(",")[0].strip()

        # 回退到客户端直连IP
        return request.client.host

    async def handle_authorization(self, auth_header: str) -> str:
        """处理授权信息
        Args:
            auth_header (str): 授权头部信息
        Returns:
            str: 处理后的授权头部信息
        """
        if not auth_header:
            logger.warning("没有提供授权信息")
            return auth_header

        # 假设auth_header格式为"Bearer YOUR_API_KEY"
        bearer_prefix = "Bearer "
        self.current_request_frontend_key = auth_header[len(bearer_prefix) :]
        if self.current_request_frontend_key in API_KEYS:
            # 临时处理，始终使用新的token

            new_key_token_str = await get_new_api_key(
                self.current_request_frontend_key, self.client
            )
            return bearer_prefix + new_key_token_str

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
        try:
            # 在发送请求前获取当前的IP地址
            request_id = random.randint(1, 1000000)
            # 发送请求
            response = await self.client.request(
                method=method,
                url=url,
                data=data,
                headers=headers,
                proxy=self.proxy,
                allow_redirects=True,
            )

            # 获取请求使用的IP地址
            if self.proxy:
                # 发起一个请求来获取当前IP
                try:
                    async with self.client.get(
                        "https://api.ipify.org", proxy=self.proxy, timeout=5
                    ) as ip_response:
                        if ip_response.status == 200:
                            current_ip = await ip_response.text()
                            # 记录请求ID和IP
                            self.request_ips[request_id] = current_ip
                            logger.info(
                                f"请求 ID {request_id} 使用代理IP: {current_ip}"
                            )
                except Exception as e:
                    logger.error(f"获取代理IP失败: {str(e)}")

            return response
        except (
            aiohttp.ServerTimeoutError,
            aiohttp.ServerConnectionError,
            aiohttp.ServerDisconnectedError,
            asyncio.TimeoutError,
        ) as e:
            error_msg = f"连接到 {self.base_url} 失败: {type(e).__name__}: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=error_msg,
            )
        except Exception as e:
            error_msg = f"请求处理出错: {type(e).__name__}: {str(e)}"
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
        url = f"{self.base_url}{self.proxy_url_path}"

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
        client_ip = self.get_client_ip(request)
        self.validate_request_ip(client_ip)

        # 创建请求ID用于跟踪
        request_id = random.randint(1, 1000000)
        logger.info(f"处理来自 {client_ip} 的新请求，ID: {request_id}")

        original_data = await request.body()
        data: bytes | None = None  # Declare data here to be used in loop

        if original_data:
            try:
                payload_dict = orjson.loads(original_data)

                # 如果未设置模型和参数，添加默认值
                if "model" not in payload_dict:
                    payload_dict["model"] = DEFAULT_MODEL
                if "temperature" not in payload_dict:
                    payload_dict["temperature"] = DEFAULT_TEMPERATURE
                if "max_tokens" not in payload_dict:
                    payload_dict["max_tokens"] = DEFAULT_MAX_TOKENS

                # 应用字段别名
                if FIELD_ALIASES:
                    transformed_dict = apply_field_aliases(payload_dict)
                    payload_dict = transformed_dict

                # 序列化回bytes
                data = orjson.dumps(payload_dict)
            except Exception as e:
                logger.error(f"处理请求数据时出错: {e}")
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
                if FORCE_CLOSE:
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
                # 使用当前代理池所指向的IP，而不是随机生成的IP
                if self.proxy:
                    try:
                        # 尝试获取当前代理的真实IP
                        async with self.client.get(
                            "https://api.ipify.org", proxy=self.proxy, timeout=5
                        ) as ip_response:
                            if ip_response.status == 200:
                                proxy_ip = await ip_response.text()
                                headers["X-Forwarded-For"] = proxy_ip
                                headers["Client-IP"] = proxy_ip
                                headers["x-real-ip"] = proxy_ip
                                logger.info(f"使用代理IP设置请求头: {proxy_ip}")
                            else:
                                # 如果获取失败，仍使用随机IP
                                headers["X-Forwarded-For"] = random_ip
                                headers["Client-IP"] = random_ip
                                headers["x-real-ip"] = random_ip
                    except Exception as e:
                        logger.error(f"获取代理IP失败: {str(e)}，使用随机IP")
                        headers["X-Forwarded-For"] = random_ip
                        headers["Client-IP"] = random_ip
                        headers["x-real-ip"] = random_ip
                else:
                    # 没有使用代理时，使用随机IP
                    headers["X-Forwarded-For"] = random_ip
                    headers["Client-IP"] = random_ip
                    headers["x-real-ip"] = random_ip

                # 更新payload并重新序列化
                payload_dict["headers"] = headers
                data = orjson.dumps(payload_dict)
            except Exception as e:
                logger.error(f"设置随机请求头时出错: {e}")
        # 记录最终发送到后端的数据
        if data:
            try:
                logger.debug(
                    f"PROXY -> BACKEND (final body decoded): {data.decode('utf-8')}"
                )
            except UnicodeDecodeError:
                logger.debug(f"PROXY -> BACKEND (final body bytes): {data!r}")

        final_response_to_send = None

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
                        f"收到429 Too Many Requests错误，URL: {request_info['url']}, 错误信息: {error_message}"
                    )

                    # 获得此时的ip地址
                    async with self.client.get(
                        "https://api.ipify.org", proxy=self.proxy, timeout=5
                    ) as ip_response:
                        if ip_response.status == 200:
                            current_ip = await ip_response.text()
                            logger.info(
                                f"收到429 Too Many Requests错误，此时的IP地址: {current_ip}"
                            )
                except Exception as e:
                    logger.warning(
                        f"收到429 Too Many Requests错误，但无法解析错误信息: {str(e)}"
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

        return final_response_to_send


# 创建转发器，传入自定义转发路径
forwarder = ChatForwarder(BASE_URL, PROXY_URL_PATH, PROXY)


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
