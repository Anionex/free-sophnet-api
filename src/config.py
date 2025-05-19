import yaml
import os
from pathlib import Path
from loguru import logger

# 设置默认配置
DEFAULT_CONFIG = {
    "timeout": 30,
    "base_url": "https://api.openai.com",
    "route_prefix": "/",
    "proxy": None,
    "proxy_url_path": "/v1/chat/completions",  # 新增: 自定义转发目标路径
    "field_aliases": {  # 新增: 字段别名映射
        # "原始字段名": "目标字段名"
        # 例如: "model": "model_id"
    },
    "api_keys": {},
    "ip_whitelist": [],
    "ip_blacklist": [],
    "connection_limit": 1000,  # 新增: HTTP连接池连接数上限
    "connection_limit_per_host": 100,  # 新增: 每个主机的连接数上限
    "keepalive_timeout": 60,  # 新增: 连接保持活跃时间(秒)
    "workers": None,  # 新增: 工作进程数，默认为CPU核心数
    "log_level": "INFO",  # 新增: 日志级别
}

# 加载配置文件
config_file_path = Path("config.yml")
if config_file_path.exists():
    with open(config_file_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    # 合并默认配置和文件配置
    for key, value in DEFAULT_CONFIG.items():
        if key not in config:
            config[key] = value
else:
    config = DEFAULT_CONFIG

# 实际配置参数 todo pydantic
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
DEFAULT_MAX_TOKENS = config.get("default_max_tokens", 4096)
DEFAULT_MODEL = config.get("default_model", "DeepSeek-V3-Fast")
# 新增配置参数
CONNECTION_LIMIT = config["connection_limit"]
CONNECTION_LIMIT_PER_HOST = config["connection_limit_per_host"]
KEEPALIVE_TIMEOUT = config["keepalive_timeout"]
FORCE_CLOSE = config["force_close"]
WORKERS = config["workers"] or os.cpu_count()
LOG_LEVEL = config["log_level"]

logger.info(f"FORCE_CLOSE: {FORCE_CLOSE}")
logger.info(f"KEEPALIVE_TIMEOUT: {KEEPALIVE_TIMEOUT}")

# 设置日志级别
logger.remove()
logger.add(
    "log/openai_chat_proxy_detailed.log",
    level=LOG_LEVEL,
    rotation="10 MB",
    compression="zip",
    retention="1 week",
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

# 键值映射记录，存储前端key到当前使用的实际key的映射
CURRENT_KEY_MAPPING = {}
