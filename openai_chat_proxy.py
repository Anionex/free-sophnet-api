import asyncio
from typing import AsyncGenerator, Dict, List, Optional, Union, Any
import random
from datetime import datetime
import platform
import multiprocessing
import time

import aiohttp
import orjson
import uvicorn
from fastapi import FastAPI, Request, HTTPException, status
from starlette.background import BackgroundTask
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from loguru import logger
import yaml
from pathlib import Path
from contextlib import asynccontextmanager

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
    "ip_whitelist": [],
    "ip_blacklist": [],
    "connection_limit": 1000,               # 新增: HTTP连接池连接数上限
    "connection_limit_per_host": 100,       # 新增: 每个主机的连接数上限
    "keepalive_timeout": 60,                # 新增: 连接保持活跃时间(秒)
    "workers": None,                        # 新增: 工作进程数，默认为CPU核心数
    "log_level": "INFO",                    # 新增: 日志级别
    "enable_function_call": True,           # 新增: 是否启用函数调用功能
    "function_call_timeout": 60             # 新增: 函数调用超时时间(秒)
}

# 加载配置文件
config_file_path = Path("config.yml")
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
PROXY_URL_PATH = config["proxy_url_path"]
FIELD_ALIASES = config["field_aliases"]
API_KEYS = config["api_keys"]
IP_WHITELIST = config["ip_whitelist"]
IP_BLACKLIST = config["ip_blacklist"]
DEFAULT_TEMPERATURE = config.get("default_temperature", 1.0)
DEFAULT_MAX_TOKENS = config.get("default_max_tokens", 2048)
DEFAULT_MODEL = config.get("default_model", "gpt-3.5-turbo")
# 新增配置参数
CONNECTION_LIMIT = config["connection_limit"]
CONNECTION_LIMIT_PER_HOST = config["connection_limit_per_host"]
KEEPALIVE_TIMEOUT = config["keepalive_timeout"]
WORKERS = config["workers"] or multiprocessing.cpu_count()
LOG_LEVEL = config["log_level"]
ENABLE_FUNCTION_CALL = config.get("enable_function_call", True)
FUNCTION_CALL_TIMEOUT = config.get("function_call_timeout", 60)

# 设置日志级别
logger.remove()
logger.add(
    "log/openai_chat_proxy_detailed.log",
    level=LOG_LEVEL,
    rotation="10 MB",
    compression="zip",
    retention="1 week"
)
logger.add(lambda msg: print(msg, end=""), level=LOG_LEVEL)

# 检查是否启用IP验证
VALIDATE_IP = bool(IP_WHITELIST or IP_BLACKLIST)

# 聊天完成接口路径
CHAT_COMPLETION_ROUTE = "/v1/chat/completions"

# 添加模型接口常量
MODELS_ROUTE = "/v1/models"
SOPHNET_MODELS_API = "https://www.sophnet.com/api/public/playground/models?projectUuid=Ar79PWUQUAhjJOja2orHs"
MODELS_CACHE_TTL = 300  # 缓存模型列表的时间（秒）

# 添加模型缓存和最后更新时间
models_cache = None
models_last_updated = 0

# 键值映射记录，存储前端key到当前使用的实际key的映射
CURRENT_KEY_MAPPING = {}

# 添加从sophnet获取匿名token的功能
async def get_anonymous_token(client_session: aiohttp.ClientSession) -> Optional[str]:
    """异步获取匿名token
    
    Args:
        client_session: aiohttp会话
        
    Returns:
        str: 获取的token，如果失败则返回None
    """
    url = "https://sophnet.com/api/sys/login/anonymous"
    
    # 添加随机参数避免缓存
    if random.random() > 0.7:
        cachebuster = int(datetime.now().timestamp() * 1000)
        url = f"{url}?_cb={cachebuster}"
    
    # 准备请求头
    headers = {
        "User-Agent": f"Mozilla/5.0 ({platform.system()} NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{random.randint(90, 110)}.0.{random.randint(4000, 5000)}.{random.randint(10, 200)} Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Referer": "https://sophnet.com/",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "Pragma": "no-cache",
        "Cache-Control": "no-cache"
    }
    
    try:
        async with client_session.get(url, headers=headers, timeout=10) as response:
            if response.status == 200:
                data = await response.json()
                if data["status"] == 0 and "result" in data:
                    token = data["result"]["anonymousToken"]
                    return f"anon-{token}"
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logger.error(f"获取匿名token时出错: {e}")
        await asyncio.sleep(random.uniform(0.5, 1.0))  # 减少错误后等待时间
    
    return None

# 获取新的API key函数
async def get_new_api_key(frontend_key: str, client_session: Optional[aiohttp.ClientSession] = None) -> str:
    """获取新的API key
    
    当检测到特定错误码时，获取一个新的API key来替换当前使用的key
    
    Args:
        frontend_key (str): 前端使用的key
        client_session: aiohttp会话，如果没有提供会创建新的
        
    Returns:
        str: 新的API key
    """
    # 动态获取新key
    logger.info(f"为 {frontend_key} 动态获取新的API key")
    # 创建一个临时会话或使用提供的会话
    close_session = False
    if client_session is None:
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=20)
        client_session = aiohttp.ClientSession(connector=connector)
        close_session = True
    
    try:
        # 尝试获取新token
        token = await get_anonymous_token(client_session)
        if token:
            logger.info(f"成功获取新token: {token[:10]}***")
            # 如果需要，更新映射
            CURRENT_KEY_MAPPING[frontend_key] = token
            return token
    finally:
        if close_session:
            await client_session.close()
    
    logger.warning(f"前端key {frontend_key} 没有对应的API key，且无法动态获取")
    return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动事件
    await forwarder.build_client()
    logger.info(f"OpenAI 聊天代理服务已启动")
    logger.info(f"转发目标: {BASE_URL}")
    logger.info(f"转发路径: {PROXY_URL_PATH}")
    if FIELD_ALIASES:
        logger.info(f"字段别名: {FIELD_ALIASES}")
    if PROXY:
        logger.info(f"使用代理: {PROXY}")
    logger.info(f"监听路径: {ROUTE_PREFIX}{CHAT_COMPLETION_ROUTE}")
    logger.info(f"连接池配置: 总连接数={CONNECTION_LIMIT}, 每主机连接数={CONNECTION_LIMIT_PER_HOST}, 连接保活={KEEPALIVE_TIMEOUT}秒")
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

# Function Call 相关模型
class FunctionDefinition(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
class ToolFunction(BaseModel):
    function: FunctionDefinition

class Tool(BaseModel):
    type: str = "function"
    function: FunctionDefinition

class FunctionCall(BaseModel):
    name: str
    arguments: str

# 聊天完成请求模型
class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    name: Optional[str] = None
    function_call: Optional[FunctionCall] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    
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
    functions: Optional[List[FunctionDefinition]] = None
    function_call: Optional[Union[str, Dict[str, str]]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None

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
        # 优化HTTP连接池
        connector = aiohttp.TCPConnector(
            limit=CONNECTION_LIMIT,
            limit_per_host=CONNECTION_LIMIT_PER_HOST,
            force_close=False,
            enable_cleanup_closed=True,
            keepalive_timeout=KEEPALIVE_TIMEOUT,
            ssl=False  # 禁用SSL验证提高性能，但请注意在生产环境中可能需要启用
        )
        self.client = aiohttp.ClientSession(
            connector=connector, 
            timeout=self.timeout,
            json_serialize=orjson.dumps  # 使用orjson序列化JSON数据
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
        if auth_header.startswith(bearer_prefix):
            key = auth_header[len(bearer_prefix):]
            self.current_request_frontend_key = key
            
            # 转换前端密钥到实际API密钥
            if key in API_KEYS:
                # 检查是否已经有一个正在使用的key
                if key in CURRENT_KEY_MAPPING:
                    actual_key = CURRENT_KEY_MAPPING[key]
                else:
                    # 如果没有，获取一个新key
                    actual_key = await get_new_api_key(key, self.client)
                    if actual_key is None:
                        # 如果无法获取新key，回退到原始配置
                        actual_key = API_KEYS[key]
                        if isinstance(actual_key, list) and actual_key:
                            actual_key = actual_key[0]
                
                return bearer_prefix + actual_key
            # 处理特殊的自动获取API key的情况
            else:
                return auth_header
        
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
                allow_redirects=True
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
        try:
            # This non-streaming path in aiter_bytes should ideally not be hit if reverse_proxy
            # only passes successful (status 200, text/event-stream) responses to it.
            if not response.headers.get("content-type", "").startswith("text/event-stream"):
                content = await response.read()
                # Log if it's a 40310, but key refresh is handled by reverse_proxy.
                if response.status == 200 or response.status == 403:
                    try:
                        json_content = orjson.loads(content)
                        if isinstance(json_content, dict) and json_content.get("status") == 40310:
                             logger.warning(f"aiter_bytes (non-stream): Encountered 40310. Key refresh is handled by reverse_proxy. Message: {json_content.get('message', '未知错误')}")
                    except Exception:
                        pass # Ignore parsing errors for this logging
                yield content
                return
            
            # 使用更大的缓冲区读取流
            buffer_size = 64 * 1024  # 64KB
            # Expected path: Processing a live, successful stream
            async for chunk in response.content.iter_chunked(buffer_size):
                # # 使用DEBUG级别记录响应数据
                # try:
                #     logger.debug(f"响应数据: {chunk.decode('utf-8')}")
                # except UnicodeDecodeError:
                #     logger.debug(f"响应数据 (bytes): {chunk}")
                  
                yield chunk
        except Exception as e:
            logger.error(f"Error in aiter_bytes response processing: {str(e)}")
            raise # Re-raise to ensure stream termination if error is critical
    
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
        client_ip = self.get_client_ip(request)
        self.validate_request_ip(client_ip)
        
        original_data = await request.body()
        data: Optional[bytes] = None # Declare data here to be used in loop
        
        # 使用DEBUG级别记录请求原始数据
        # if original_data:
        #     try:
        #         logger.debug(f"CLIENT -> PROXY (raw body decoded): {original_data.decode('utf-8')}")
        #     except UnicodeDecodeError:
        #         logger.debug(f"CLIENT -> PROXY (raw body bytes): {original_data!r}")

        if original_data:
            try:
                payload_dict = orjson.loads(original_data)
                
                # 检查是否包含function call相关字段
                has_function_fields = False
                if ENABLE_FUNCTION_CALL and (
                    'functions' in payload_dict or 
                    'function_call' in payload_dict or
                    'tools' in payload_dict or
                    'tool_choice' in payload_dict
                ):
                    has_function_fields = True
                    logger.debug("检测到function call/tools请求")
                
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
                    # 记录转换前后的数据
                    # logger.debug(f"原始请求数据: {payload_dict}")
                    # logger.debug(f"转换后请求数据: {transformed_dict}")
                    payload_dict = transformed_dict
                
                # 序列化回bytes
                data = orjson.dumps(payload_dict)
            except Exception as e:
                logger.error(f"处理请求数据时出错: {e}")
                data = original_data
        else:
            data = original_data

        # 记录最终发送到后端的数据
        if data:
            try:
                logger.debug(f"PROXY -> BACKEND (final body decoded): {data.decode('utf-8')}")
            except UnicodeDecodeError:
                logger.debug(f"PROXY -> BACKEND (final body bytes): {data!r}")

        final_response_to_send = None

        for attempt in range(self.MAX_RETRIES + 1):
            self.current_request_frontend_key = None # Reset for each attempt

            request_info = await self.prepare_request_info(request)
            
            auth_header_for_log = request_info["headers"].get("authorization", "None")
            logger.debug(f"Attempt {attempt + 1}/{self.MAX_RETRIES + 1} using Authorization: {auth_header_for_log[:25]}...")

            target_response = await self.send_request(
                method=request_info["method"],
                url=request_info["url"],
                headers=request_info["headers"],
                data=data
            )

            if target_response.status == 200 and \
               target_response.headers.get("content-type", "").startswith("text/event-stream"):
                logger.debug(f"Attempt {attempt + 1}: Successful stream response (status {target_response.status}). Forwarding.")
                return StreamingResponse(
                    self.aiter_bytes(target_response),
                    status_code=target_response.status,
                    media_type=target_response.headers.get("content-type"),
                    background=BackgroundTask(target_response.release),
                )

            response_content = await target_response.read()
            
            # 检查是否包含function call响应并记录日志
            if target_response.status == 200:
                try:
                    json_content = orjson.loads(response_content)
                    if isinstance(json_content, dict):
                        if json_content.get("choices") and isinstance(json_content["choices"], list):
                            for choice in json_content["choices"]:
                                if isinstance(choice, dict):
                                    message = choice.get("message", {})
                                    if message.get("function_call") or message.get("tool_calls"):
                                        logger.debug(f"检测到function call/tools响应: {message}")
                except Exception as e:
                    logger.debug(f"解析响应以检查function call时出错: {e}")
            
            is_40310_error = False
            if target_response.status == 200 or target_response.status == 403:
                try:
                    json_body = orjson.loads(response_content)
                    if isinstance(json_body, dict) and json_body.get("status") == 40310:
                        is_40310_error = True
                        logger.warning(
                            f"Attempt {attempt + 1}: Detected status 40310. Message: {json_body.get('message', 'N/A')}"
                        )
                except orjson.JSONDecodeError:
                    pass 
            
            if is_40310_error:
                if self.current_request_frontend_key:
                    logger.info(f"Attempting to get a new API key for frontend key: {self.current_request_frontend_key}")
                    new_key_token_str = await get_new_api_key(self.current_request_frontend_key, self.client)
                    
                    if new_key_token_str:
                        await target_response.release()
                        if attempt < self.MAX_RETRIES:
                            logger.info(f"New key obtained. Retrying request (next attempt: {attempt + 2}).")
                            await asyncio.sleep(random.uniform(0.3, 0.7)) # 减少重试间隔
                            continue
                        else:
                            logger.warning(f"New key obtained, but max retries ({self.MAX_RETRIES + 1}) reached. Failing with last error.")
                    else:
                        logger.warning("Failed to obtain a new API key. Will not retry based on this error.")
                else:
                    logger.warning("40310 error detected, but no current_request_frontend_key was set. Cannot refresh key.")

            logger.debug(f"Attempt {attempt + 1}: Forwarding this response (status: {target_response.status}). No further retries for this error type or max attempts reached.")
            
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

# 在ChatForwarder类中添加获取模型列表的方法
async def fetch_models():
    """从sophnet获取模型列表并转换为OpenAI格式
    
    Returns:
        dict: OpenAI格式的模型列表
    """
    global models_cache, models_last_updated
    
    # 检查缓存是否有效
    current_time = time.time()
    if models_cache and current_time - models_last_updated < MODELS_CACHE_TTL:
        return models_cache
    
    try:
        # 创建一个连接器和会话
        connector = aiohttp.TCPConnector(limit=5)
        async with aiohttp.ClientSession(connector=connector) as session:
            # 直接获取匿名token
            token = await get_anonymous_token(session)
            if not token:
                logger.warning("无法获取匿名token，模型列表请求可能会失败")
                # 如果仍有缓存，返回缓存
                if models_cache:
                    return models_cache
                return {"object": "list", "data": []}
            
            # 准备请求头
            headers = {
                "User-Agent": f"Mozilla/5.0 ({platform.system()} NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{random.randint(90, 110)}.0.{random.randint(4000, 5000)}.{random.randint(10, 200)} Safari/537.36",
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
                "Referer": "https://sophnet.com/",
            }
            
            # 设置匿名token头
            headers["Authorization"] = f"Bearer {token}"
            
            logger.debug(f"获取模型列表使用的匿名token: {headers['Authorization']}...")
            
            # 发送请求获取模型列表
            async with session.get(SOPHNET_MODELS_API, headers=headers) as response:
                if response.status != 200:
                    logger.error(f"获取模型列表失败: {response.status}")
                    if models_cache:  # 如果有缓存，返回缓存
                        return models_cache
                    return {"object": "list", "data": []}
                
                sophnet_data = await response.json()
                
                if sophnet_data.get("status") != 0 or "result" not in sophnet_data:
                    logger.error(f"sophnet API返回错误: {sophnet_data.get('message', '未知错误')}")
                    if models_cache:
                        return models_cache
                    return {"object": "list", "data": []}
                
                # 转换为OpenAI格式
                current_timestamp = int(datetime.now().timestamp() * 1000)
                openai_models = {
                    "object": "list",
                    "data": []
                }
                
                for model in sophnet_data["result"]:
                    model_id = model.get("modelFamily") or model.get("displayName")
                    if not model_id:
                        continue
                        
                    # 创建OpenAI格式的模型数据
                    openai_model = {
                        "id": model_id,
                        "object": "model",
                        "created": current_timestamp,
                        "owned_by": "sophnet",
                        "permission": [
                            {
                                "id": f"modelperm-{model.get('id', random.randint(1, 100))}",
                                "object": "model_permission",
                                "created": current_timestamp,
                                "allow_create_engine": False,
                                "allow_sampling": True,
                                "allow_logprobs": False,
                                "allow_search_indices": False,
                                "allow_view": True,
                                "allow_fine_tuning": False,
                                "organization": "*",
                                "group": None,
                                "is_blocking": False
                            }
                        ],
                        "root": model_id,
                        "parent": None
                    }
                    openai_models["data"].append(openai_model)
                
                # 更新缓存和时间戳
                models_cache = openai_models
                models_last_updated = current_time
                
                return openai_models
    except Exception as e:
        logger.error(f"获取模型列表时出错: {e}")
        if models_cache:
            return models_cache
        return {"object": "list", "data": []}

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

def main():
    # 使用命令行参数直接启动uvicorn
    import sys
    sys.argv = ["uvicorn", "openai_chat_proxy:app", "--host", "0.0.0.0", "--port", "8000",
               f"--workers={WORKERS}", f"--log-level={LOG_LEVEL.lower()}", 
               "--loop=uvloop", "--http=httptools", "--limit-concurrency=1000",
               "--backlog=2048", f"--timeout-keep-alive={5}"]
    uvicorn.main()

if __name__ == "__main__":
    main()