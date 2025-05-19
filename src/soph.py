import asyncio
from datetime import datetime
import random
import time
from config import CURRENT_KEY_MAPPING, MODELS_CACHE_TTL, SOPHNET_MODELS_API, logger
import aiohttp
from utils import fake_useragent


# 添加模型缓存和最后更新时间
models_cache = None
models_last_updated = 0


# 添加从sophnet获取匿名token的功能
async def get_anonymous_token(client_session: aiohttp.ClientSession) -> str | None:
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
        "User-Agent": fake_useragent(),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Referer": "https://sophnet.com/",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "Pragma": "no-cache",
        "Cache-Control": "no-cache",
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
async def get_new_api_key(
    frontend_key: str, client_session: aiohttp.ClientSession | None = None
) -> str | None:
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
                "User-Agent": fake_useragent(),
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
                    logger.error(
                        f"sophnet API返回错误: {sophnet_data.get('message', '未知错误')}"
                    )
                    if models_cache:
                        return models_cache
                    return {"object": "list", "data": []}

                # 转换为OpenAI格式
                current_timestamp = int(datetime.now().timestamp() * 1000)
                openai_models = {"object": "list", "data": []}

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
                                "is_blocking": False,
                            }
                        ],
                        "root": model_id,
                        "parent": None,
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
