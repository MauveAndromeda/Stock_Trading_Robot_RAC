"""
日志系统
"""
import sys
from pathlib import Path
from loguru import logger
from typing import Optional


def setup_logger(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    rotation: str = "1 day",
    retention: str = "30 days",
    format_type: str = "text"
) -> None:
    """
    配置日志系统

    Args:
        log_level: 日志级别
        log_file: 日志文件路径
        rotation: 日志轮转策略
        retention: 日志保留时间
        format_type: 格式类型 (text/json)
    """
    # 移除默认handler
    logger.remove()

    # 定义日志格式
    if format_type == "json":
        log_format = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}"
    else:
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

    # 控制台输出
    logger.add(
        sys.stdout,
        format=log_format,
        level=log_level,
        colorize=True if format_type == "text" else False
    )

    # 文件输出
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_file,
            format=log_format,
            level=log_level,
            rotation=rotation,
            retention=retention,
            encoding="utf-8",
            backtrace=True,
            diagnose=True
        )

    logger.info(f"Logger initialized with level: {log_level}")


def get_logger(name: str):
    """
    获取logger实例

    Args:
        name: logger名称

    Returns:
        logger实例
    """
    return logger.bind(name=name)


# 默认logger
default_logger = logger
