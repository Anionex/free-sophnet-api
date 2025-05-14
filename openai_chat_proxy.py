import asyncio
import os
from typing import AsyncGenerator, Dict, List, Optional, Union, Any

import aiohttp
import orjson
import uvicorn
from fastapi import FastAPI, Request, HTTPException, status
from starlette.background import BackgroundTask
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from loguru import logger
import yaml
from pathlib import Path

# 设置默认配置
DEFAULT_CONFIG = {
    "timeout": 30,
    "base_url": "https://api.openai.com",
    "route_prefix": "/",
    "proxy": None,
    "proxy_url_path": "/v1/chat/completions",  # 新增: 自定义转发目标路径
    "field_aliases": {                         # 新增: 字段别名映射
        # "原始字段名": "目标字段名"
        # 例如: "model": "model_id"
    },
    "api_keys": {},
    "log_requests": False,
    "log_responses": False,
    "ip_whitelist": [],
    "ip_blacklist": []
}

# 加载配置文件
config_file_path = Path("openai-chat-proxy-config.yaml")
if config_file_path.exists():
    with open(config_file_path, encoding='utf-8') as file:
        config = yaml.safe_load(file)
    # 合并默认配置和文件配置
    for key, value in DEFAULT_CONFIG.items():
        if key not in config:
            config[key] = value
else:
    config = DEFAULT_CONFIG

# 实际配置参数
TIMEOUT = config["timeout"]
BASE_URL = config["base_url"]
ROUTE_PREFIX = config["route_prefix"]
PROXY = config["proxy"]
PROXY_URL_PATH = config["proxy_url_path"]  # 新增: 获取自定义转发目标路径
FIELD_ALIASES = config["field_aliases"]      # 新增: 获取字段别名映射
API_KEYS = config["api_keys"]
LOG_REQUESTS = config["log_requests"]
LOG_RESPONSES = config["log_responses"]
IP_WHITELIST = config["ip_whitelist"]
IP_BLACKLIST = config["ip_blacklist"]

# 检查是否启用IP验证
VALIDATE_IP = bool(IP_WHITELIST or IP_BLACKLIST)

# 聊天完成接口路径
CHAT_COMPLETION_ROUTE = "/v1/chat/completions"


# FastAPI应用
app = FastAPI(title="openai-chat-proxy", version="0.1.0")

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
    name: Optional[str] = None
    
# 聊天完成请求模型
class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None

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
    
    async def build_client(self):
        """异步构建HTTP客户端"""
        connector = aiohttp.TCPConnector(limit=500, limit_per_host=0, force_close=False)
        self.client = aiohttp.ClientSession(connector=connector, timeout=self.timeout)
    
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
            if (IP_WHITELIST and ip not in IP_WHITELIST) or (IP_BLACKLIST and ip in IP_BLACKLIST):
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
        x_real_ip = request.headers.get("x-real-ip")
        if x_real_ip:
            return x_real_ip
        else:
            client_host = request.client.host
            return client_host
    
    def handle_authorization(self, auth_header: str) -> str:
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
        if auth_header.startswith(bearer_prefix):
            key = auth_header[len(bearer_prefix):]
            
            # 转换前端密钥到实际API密钥
            if key in API_KEYS:
                actual_key = API_KEYS[key]
                return bearer_prefix + actual_key
        
        return auth_header
    
    async def send_request(self, method: str, url: str, headers: dict, data: bytes = None):
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
            return await self.client.request(
                method=method,
                url=url,
                data=data,
                headers=headers,
                proxy=self.proxy,
            )
        except (
            aiohttp.ServerTimeoutError,
            aiohttp.ServerConnectionError,
            aiohttp.ServerDisconnectedError,
            asyncio.TimeoutError
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
    
    async def aiter_bytes(self, response: aiohttp.ClientResponse) -> AsyncGenerator[bytes, Any]:
        """异步迭代响应中的字节
        
        Args:
            response (aiohttp.ClientResponse): 响应对象
            
        Yields:
            bytes: 响应字节数据
        """
        try:
            async for chunk in response.content.iter_any():
                if LOG_RESPONSES:
                    try:
                        logger.debug(f"响应数据: {chunk.decode('utf-8')}")
                    except:
                        pass
                yield chunk
        except Exception as e:
            logger.error(f"流式响应处理错误: {str(e)}")
    
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
            headers["authorization"] = self.handle_authorization(auth_header)
        
        # 移除不需要转发的头信息
        headers_to_remove = ['host', 'content-length']
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
        # 验证IP
        client_ip = self.get_client_ip(request)
        self.validate_request_ip(client_ip)
        
        # 获取请求体数据
        original_data = await request.body()
        
        # 应用字段别名映射（如果有）
        if original_data and FIELD_ALIASES:
            try:
                # 解析请求体为JSON
                payload_dict = orjson.loads(original_data)
                
                # 应用字段别名映射
                transformed_dict = apply_field_aliases(payload_dict)
                
                # 转换回bytes
                data = orjson.dumps(transformed_dict)
                
                # 日志记录
                if LOG_REQUESTS:
                    logger.info(f"原始请求数据: {payload_dict}")
                    logger.info(f"转换后请求数据: {transformed_dict}")
            except Exception as e:
                logger.error(f"处理请求数据时出错: {e}")
                data = original_data
        else:
            data = original_data
            if data and LOG_REQUESTS:
                try:
                    payload = orjson.loads(data)
                    logger.info(f"请求数据: {payload}")
                except:
                    logger.info(f"请求数据: [无法解析JSON]")
        
        # 准备请求信息
        request_info = await self.prepare_request_info(request)
        
        # 发送请求到目标服务器
        response = await self.send_request(
            method=request_info["method"],
            url=request_info["url"],
            headers=request_info["headers"],
            data=data
        )
        
        # 返回流式响应
        return StreamingResponse(
            self.aiter_bytes(response),
            status_code=response.status,
            media_type=response.headers.get("content-type"),
            background=BackgroundTask(response.release),
        )

# 创建转发器，传入自定义转发路径
forwarder = ChatForwarder(BASE_URL, PROXY_URL_PATH, PROXY)

@app.on_event("startup")
async def startup():
    """应用启动事件处理程序"""
    await forwarder.build_client()
    logger.info(f"OpenAI 聊天代理服务已启动")
    logger.info(f"转发目标: {BASE_URL}")
    logger.info(f"转发路径: {PROXY_URL_PATH}")
    if FIELD_ALIASES:
        logger.info(f"字段别名: {FIELD_ALIASES}")
    if PROXY:
        logger.info(f"使用代理: {PROXY}")
    logger.info(f"监听路径: {ROUTE_PREFIX}{CHAT_COMPLETION_ROUTE}")

@app.on_event("shutdown")
async def shutdown():
    """应用关闭事件处理程序"""
    await forwarder.close()
    logger.info("OpenAI 聊天代理服务已关闭")

@app.get("/healthz")
def health_check():
    """健康检查接口"""
    return {"status": "ok"}

# 注册转发路由
app.add_route(
    CHAT_COMPLETION_ROUTE,
    forwarder.reverse_proxy,
    methods=["POST"],
)

# 主函数
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 