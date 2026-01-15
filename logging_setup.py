import logging
import logging.handlers
import os
import json
from typing import Iterable, Optional
from pathlib import Path
from datetime import datetime


class RedactFilter(logging.Filter):
    """
    过滤日志中的敏感字段，简单字符串替换。
    """

    def __init__(self, secrets: Iterable[str]):
        super().__init__()
        # 过滤掉空值
        self.secrets = [s for s in secrets if s]

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        for secret in self.secrets:
            if secret and secret in msg:
                msg = msg.replace(secret, "***REDACTED***")
        record.msg = msg
        record.args = ()
        return True


class JSONFormatter(logging.Formatter):
    """
    JSON格式日志输出，便于日志分析和机器解析。
    """

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # 添加自定义字段（如果存在）
        if hasattr(record, "patient_id"):
            log_data["patient_id"] = record.patient_id
        if hasattr(record, "tool_name"):
            log_data["tool_name"] = record.tool_name
        if hasattr(record, "duration"):
            log_data["duration"] = record.duration

        return json.dumps(log_data, ensure_ascii=False)


_configured = False


def setup_logging(
    log_dir: Optional[str] = None,
    enable_file_rotation: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    enable_json_logs: bool = False,
):
    """
    统一日志配置，支持控制台、文件轮转和JSON格式输出。

    Args:
        log_dir: 日志目录路径，如果为None则使用环境变量LOG_DIR或默认"logs"
        enable_file_rotation: 是否启用日志轮转
        max_bytes: 单个日志文件最大大小（字节）
        backup_count: 保留的备份文件数量
        enable_json_logs: 是否启用JSON格式日志
    """
    global _configured
    if _configured:
        return

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_dir = log_dir or os.getenv("LOG_DIR", "logs")

    # 创建日志目录
    Path(log_dir).mkdir(exist_ok=True)

    # 标准格式化器
    standard_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # JSON格式化器
    json_formatter = JSONFormatter() if enable_json_logs else None

    handlers = []

    # 1. 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(standard_formatter)
    console_handler.setLevel(log_level)
    handlers.append(console_handler)

    # 2. 主日志文件（带轮转）
    main_log_file = os.path.join(log_dir, "hema_agent.log")
    if enable_file_rotation:
        main_file_handler = logging.handlers.RotatingFileHandler(
            main_log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
    else:
        main_file_handler = logging.FileHandler(
            main_log_file, encoding="utf-8"
        )
    main_file_handler.setFormatter(standard_formatter)
    main_file_handler.setLevel(log_level)
    handlers.append(main_file_handler)

    # 3. 错误日志文件（只记录ERROR及以上）
    error_log_file = os.path.join(log_dir, "errors.log")
    if enable_file_rotation:
        error_file_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
    else:
        error_file_handler = logging.FileHandler(
            error_log_file, encoding="utf-8"
        )
    error_file_handler.setFormatter(standard_formatter)
    error_file_handler.setLevel(logging.ERROR)
    handlers.append(error_file_handler)

    # 4. JSON日志文件（可选）
    if enable_json_logs:
        json_log_file = os.path.join(log_dir, "hema_agent.json.log")
        if enable_file_rotation:
            json_file_handler = logging.handlers.RotatingFileHandler(
                json_log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
        else:
            json_file_handler = logging.FileHandler(
                json_log_file, encoding="utf-8"
            )
        json_file_handler.setFormatter(json_formatter)
        json_file_handler.setLevel(log_level)
        handlers.append(json_file_handler)

    # 敏感信息过滤器
    secrets = [
        os.getenv("ONCOKB_TOKEN"),
        os.getenv("TAVILY_API_KEY"),
        os.getenv("PUBMED_API_KEY"),
        os.getenv("LOCAL_API_KEY"),
        os.getenv("OPENAI_API_KEY"),
        os.getenv("RERANKER_API_KEY"),
    ]
    redact_filter = RedactFilter(secrets)

    # 配置根日志记录器
    root = logging.getLogger()
    root.setLevel(log_level)

    # 移除现有的处理器（避免重复）
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    # 添加新处理器
    for handler in handlers:
        handler.addFilter(redact_filter)
        root.addHandler(handler)

    _configured = True

    # 降噪第三方日志
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb.telemetry").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # 记录日志系统初始化
    logger = logging.getLogger(__name__)
    logger.info(
        f"日志系统已初始化 | 级别={log_level} | 目录={log_dir} | "
        f"轮转={'启用' if enable_file_rotation else '禁用'} | "
        f"JSON={'启用' if enable_json_logs else '禁用'}"
    )


def get_module_logger(module_name: str) -> logging.Logger:
    """
    获取特定模块的日志记录器。

    Args:
        module_name: 模块名称

    Returns:
        配置好的Logger实例
    """
    return logging.getLogger(module_name)

