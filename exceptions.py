"""
自定义异常类和错误处理装饰器

该模块提供了项目中使用的所有自定义异常类，以及统一的错误处理机制。
"""

import functools
import logging
from typing import Any, Callable, Optional, TypeVar, ParamSpec
import asyncio

logger = logging.getLogger(__name__)

# 类型变量
P = ParamSpec("P")
T = TypeVar("T")


# ==================== 异常类层次结构 ==================== #


class HemaAgentError(Exception):
    """
    所有自定义异常的基类。

    Attributes:
        message: 错误消息
        details: 额外的错误详情
    """

    def __init__(self, message: str, details: Optional[dict] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | 详情: {self.details}"
        return self.message


class ConfigurationError(HemaAgentError):
    """配置相关错误（如配置文件缺失、格式错误）"""

    pass


class RAGError(HemaAgentError):
    """RAG检索相关错误的基类"""

    pass


class DatabaseError(RAGError):
    """向量数据库相关错误（如连接失败、查询错误）"""

    pass


class EmbeddingError(RAGError):
    """Embedding生成相关错误"""

    pass


class RetrievalError(RAGError):
    """文档检索相关错误"""

    pass


class ToolExecutionError(HemaAgentError):
    """Agent工具执行错误的基类"""

    pass


class ToolTimeoutError(ToolExecutionError):
    """工具执行超时"""

    pass


class ToolValidationError(ToolExecutionError):
    """工具参数验证失败"""

    pass


class ExternalAPIError(HemaAgentError):
    """外部API调用错误的基类"""

    pass


class OpenAIAPIError(ExternalAPIError):
    """OpenAI API相关错误"""

    pass


class PubMedAPIError(ExternalAPIError):
    """PubMed API相关错误"""

    pass


class TavilyAPIError(ExternalAPIError):
    """Tavily搜索API相关错误"""

    pass


class CIViCAPIError(ExternalAPIError):
    """CIViC基因数据库API相关错误"""

    pass


class MemoryBankError(HemaAgentError):
    """患者记忆银行相关错误"""

    pass


class PatientDataError(HemaAgentError):
    """患者数据相关错误（如数据格式错误、缺失必要字段）"""

    pass


class AgentExecutionError(HemaAgentError):
    """Agent执行错误"""

    pass


# ==================== 错误处理装饰器 ==================== #


def with_error_handling(
    default_return: Any = None,
    log_level: int = logging.ERROR,
    reraise: bool = True,
    error_class: type[Exception] = HemaAgentError,
):
    """
    统一错误处理装饰器，用于同步函数。

    Args:
        default_return: 发生错误时的默认返回值
        log_level: 日志级别
        reraise: 是否重新抛出异常
        error_class: 要捕获并转换的异常类

    Example:
        @with_error_handling(default_return="", error_class=RAGError)
        def retrieve_docs(query: str) -> str:
            # 可能抛出异常的代码
            return result
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except error_class as e:
                logger.log(
                    log_level,
                    f"函数 {func.__name__} 执行失败: {e}",
                    exc_info=True,
                    extra={"function": func.__name__, "error_type": type(e).__name__},
                )
                if reraise:
                    raise
                return default_return
            except Exception as e:
                logger.log(
                    log_level,
                    f"函数 {func.__name__} 发生未预期错误: {type(e).__name__}: {e}",
                    exc_info=True,
                    extra={"function": func.__name__, "error_type": type(e).__name__},
                )
                if reraise:
                    # 将未知异常包装为自定义异常
                    raise error_class(
                        f"{func.__name__} 执行失败",
                        details={"original_error": str(e), "error_type": type(e).__name__},
                    ) from e
                return default_return

        return wrapper

    return decorator


def with_async_error_handling(
    default_return: Any = None,
    log_level: int = logging.ERROR,
    reraise: bool = True,
    error_class: type[Exception] = HemaAgentError,
):
    """
    统一错误处理装饰器，用于异步函数。

    Args:
        default_return: 发生错误时的默认返回值
        log_level: 日志级别
        reraise: 是否重新抛出异常
        error_class: 要捕获并转换的异常类

    Example:
        @with_async_error_handling(error_class=ToolExecutionError)
        async def call_external_api(query: str) -> dict:
            # 可能抛出异常的代码
            return result
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                return await func(*args, **kwargs)
            except error_class as e:
                logger.log(
                    log_level,
                    f"异步函数 {func.__name__} 执行失败: {e}",
                    exc_info=True,
                    extra={"function": func.__name__, "error_type": type(e).__name__},
                )
                if reraise:
                    raise
                return default_return
            except asyncio.TimeoutError as e:
                logger.log(
                    log_level,
                    f"异步函数 {func.__name__} 执行超时",
                    exc_info=True,
                    extra={"function": func.__name__},
                )
                if reraise:
                    raise ToolTimeoutError(
                        f"{func.__name__} 执行超时",
                        details={"original_error": str(e)},
                    ) from e
                return default_return
            except Exception as e:
                logger.log(
                    log_level,
                    f"异步函数 {func.__name__} 发生未预期错误: {type(e).__name__}: {e}",
                    exc_info=True,
                    extra={"function": func.__name__, "error_type": type(e).__name__},
                )
                if reraise:
                    raise error_class(
                        f"{func.__name__} 执行失败",
                        details={"original_error": str(e), "error_type": type(e).__name__},
                    ) from e
                return default_return

        return wrapper

    return decorator


def with_retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
):
    """
    重试装饰器，支持指数退避。

    Args:
        max_attempts: 最大尝试次数
        delay: 初始延迟时间（秒）
        backoff: 退避因子（每次重试延迟时间乘以此因子）
        exceptions: 需要重试的异常类型

    Example:
        @with_retry(max_attempts=3, delay=1.0, backoff=2.0, exceptions=(ConnectionError,))
        async def call_api():
            # 可能失败的API调用
            return response
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            current_delay = delay
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        logger.warning(
                            f"函数 {func.__name__} 第 {attempt}/{max_attempts} 次尝试失败: {e}. "
                            f"{current_delay:.1f}秒后重试...",
                            extra={
                                "function": func.__name__,
                                "attempt": attempt,
                                "max_attempts": max_attempts,
                            },
                        )
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"函数 {func.__name__} 在 {max_attempts} 次尝试后仍然失败",
                            exc_info=True,
                            extra={"function": func.__name__, "max_attempts": max_attempts},
                        )

            # 所有尝试都失败了，抛出最后一个异常
            raise last_exception

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            import time

            current_delay = delay
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        logger.warning(
                            f"函数 {func.__name__} 第 {attempt}/{max_attempts} 次尝试失败: {e}. "
                            f"{current_delay:.1f}秒后重试...",
                            extra={
                                "function": func.__name__,
                                "attempt": attempt,
                                "max_attempts": max_attempts,
                            },
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"函数 {func.__name__} 在 {max_attempts} 次尝试后仍然失败",
                            exc_info=True,
                            extra={"function": func.__name__, "max_attempts": max_attempts},
                        )

            raise last_exception

        # 根据函数类型返回相应的包装器
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# ==================== 错误上下文管理器 ==================== #


class ErrorContext:
    """
    错误上下文管理器，用于捕获和记录代码块中的错误。

    Example:
        with ErrorContext("处理患者数据", patient_id="12345"):
            # 可能抛出异常的代码
            process_patient_data(data)
    """

    def __init__(
        self,
        operation_name: str,
        log_level: int = logging.ERROR,
        reraise: bool = True,
        **context_data: Any,
    ):
        self.operation_name = operation_name
        self.log_level = log_level
        self.reraise = reraise
        self.context_data = context_data

    def __enter__(self):
        logger.debug(f"开始操作: {self.operation_name}", extra=self.context_data)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            logger.log(
                self.log_level,
                f"操作 '{self.operation_name}' 失败: {exc_type.__name__}: {exc_val}",
                exc_info=(exc_type, exc_val, exc_tb),
                extra=self.context_data,
            )
            return not self.reraise  # 返回True会抑制异常
        else:
            logger.debug(f"操作完成: {self.operation_name}", extra=self.context_data)
            return False


# ==================== 工具函数 ==================== #


def validate_required_fields(data: dict, required_fields: list[str], data_type: str = "数据") -> None:
    """
    验证字典中是否包含所有必需的字段。

    Args:
        data: 要验证的字典
        required_fields: 必需字段列表
        data_type: 数据类型描述（用于错误消息）

    Raises:
        PatientDataError: 如果缺少必需字段
    """
    missing_fields = [field for field in required_fields if field not in data or not data[field]]

    if missing_fields:
        raise PatientDataError(
            f"{data_type}缺少必需字段",
            details={"missing_fields": missing_fields, "provided_fields": list(data.keys())},
        )


def safe_api_call(func: Callable, *args, api_name: str = "API", **kwargs) -> Any:
    """
    安全地调用外部API，统一处理常见错误。

    Args:
        func: 要调用的API函数
        api_name: API名称（用于错误消息）
        *args, **kwargs: 传递给func的参数

    Returns:
        API调用结果

    Raises:
        ExternalAPIError: 如果API调用失败
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        raise ExternalAPIError(
            f"{api_name} 调用失败",
            details={
                "api_name": api_name,
                "error_type": type(e).__name__,
                "error_message": str(e),
            },
        ) from e
