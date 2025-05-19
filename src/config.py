from functools import cache
import sys
import pprint
from loguru import logger
from pathlib import Path
from typing import Literal
import yaml
import os
from pydantic import BaseModel
from const import LOG_PATH


# def _build_logger():
#     logger = logging.Logger("ForwardSoph")
#     thandler = TimedRotatingFileHandler(
#         filename=Path(LOG_PATH) / "soph.log", when="d", backupCount=10, encoding="utf-8"
#     )
#     formatter = logging.Formatter(
#         fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
#         datefmt="%Y-%m-%d %H:%M:%S",
#     )
#     thandler.setFormatter(formatter)
#     logger.addHandler(thandler)
#     logger.setLevel(cfg_obj.log_level)
def _build_logger():
    # 移除默认的 loguru handler, 通常是输出到 stderr
    logger.remove()
    # 添加新的 handler
    # filename 参数可以直接使用 Path 对象
    logger.add(
        Path(LOG_PATH) / "soph.log",
        rotation="1 day",  # 每天轮换, 相当于 when="d"
        retention="10 days",  # 保留 10 天的日志文件, 相当于 backupCount=10
        encoding="utf-8",
        level=cfg_obj.log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}",
    )
    logger.add(sys.stdout, level=cfg_obj.log_level)


# 实际配置参数 todo pydantic
class Config(BaseModel):
    timeout: float = 10
    """
    连接 (connect) 超时 (秒)
    """

    api_host: str = "https://sophnet.com"
    """
    SophNet api host
    """

    proxy: str | None = None
    """
    转发请求代理
    """

    forward_chat_path: str = (
        "/api/open-apis/projects/Ar79PWUQUAhjJOja2orHs/chat/completions"
    )
    """
    转发目标的路径, 比如 openai `https://api.openai.com/v1/chat/completions` 则为 `/v1/chat/completions`
    """

    field_aliases: dict[str, str] = {}
    """
    请求字段映射: 接收请求字段名->转发请求字段名

    Examples:
        - {"model": "model_id"}: 将请求接收到的字段 {"model": "xxx"} 转换为 {"model_id": "xxx"}, 然后向上游发送请求
    """

    api_keys: dict[str, str] = {}
    """
    API 密钥映射: 接收请求 API KEY -> 转发请求 API KEY (上游 API KEY)
    """

    ip_whitelist: list[str] = []
    """
    接收请求 IP 白名单
    """

    ip_black_list: list[str] = []
    """
    接收请求 IP 黑名单

    Note:
        黑名单比白名单优先, 当 IP 同时处于黑名单和白名单, 禁止访问.
    """

    default_temperature: float = 0.6
    """
    LLM 请求默认温度
    """

    default_max_tokens: int = 10**8
    """
    LLM 默认最大 tokens 数
    """

    default_model: str = "DeepSeek-V3-Fast"
    """
    默认模型名称
    """

    connection_limit: int = 1000
    """
    HTTP 连接池连接数上限
    """

    connection_limit_per_host: int = 100
    """
    每个主机最大连接数限制
    """

    keepalive_timeout: int = 60
    """
    连接保持活跃时间 (秒)
    """

    models_cache_ttl: int = 300
    """
    缓存模型列表的时间 (秒)
    """

    force_close: bool = False
    """
    是否强制关闭连接, 当代理是轮询IP时, 启用以减少 429 too many requests 错误
    """

    workers: int | None = os.cpu_count()
    """
    新增: 工作进程数，默认为CPU核心数
    """

    log_level: Literal["INFO", "DEBUG", "WARNING", "ERROR"] = "INFO"
    """
    日志等级, 可选: "INFO", "DEBUG", "WARNING", "ERROR"
    """

    @property
    def validate_ip(self) -> bool:
        """
        是否使用 ip 过滤
        """
        return bool(self.ip_whitelist or self.ip_black_list)


_build_logger()

logger.info("Loading config...")
with open("config.yml", "r", encoding="utf-8") as f:
    try:
        cfg_obj = Config.model_validate(yaml.safe_load(f), strict=True)
    except IOError:
        cfg_obj = Config()
    # YAMLError and pydantic errors will panic
logger.info("Config loaded:\n%s", pprint.pformat(cfg_obj))
