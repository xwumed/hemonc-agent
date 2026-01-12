import logging
import os
from typing import Iterable


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


_configured = False


def setup_logging():
    """统一日志配置，支持控制台与可选文件输出。"""
    global _configured
    if _configured:
        return

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_file = os.getenv("LOG_FILE")

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handlers = []
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    secrets = [
        os.getenv("ONCOKB_TOKEN"),
        os.getenv("TAVILY_API_KEY"),
        os.getenv("PUBMED_API_KEY"),
        os.getenv("LOCAL_API_KEY"),
        os.getenv("OPENAI_API_KEY"),
    ]
    redact_filter = RedactFilter(secrets)

    root = logging.getLogger()
    root.setLevel(log_level)
    for h in handlers:
        h.addFilter(redact_filter)
        root.addHandler(h)

    _configured = True

    # 降噪第三方日志
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb.telemetry").setLevel(logging.WARNING)

