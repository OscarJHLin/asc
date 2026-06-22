"""Asc 集群系统自定义异常层次结构。

提供精细的异常分类，使错误类型能准确反映问题性质，
便于日志分析、监控告警和问题定位。

使用方式：
    from asc.core.exceptions import AscError, InferenceError

    try:
        ...
    except InferenceError as e:
        logger.error("推理失败: %s", e.message, extra=e.details)
"""

from __future__ import annotations


class AscError(Exception):
    """Asc 系统基础异常类。

    所有自定义异常的基类，提供统一的 message 和 details 属性。
    """

    def __init__(self, message: str, details: dict | None = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


# --- 网络通信 ---

class NetworkError(AscError):
    """网络通信相关异常。"""
    pass


class WebSocketError(NetworkError):
    """WebSocket 通信异常。"""
    pass


class ConnectionError(NetworkError):
    """连接建立或维护异常。"""
    pass


# --- 硬件 ---

class HardwareError(AscError):
    """硬件检测相关异常。"""
    pass


class GPUError(HardwareError):
    """GPU 检测或初始化异常。"""
    pass


# --- 推理引擎 ---

class InferenceError(AscError):
    """推理执行相关异常。"""
    pass


class EngineNotReadyError(InferenceError):
    """引擎未就绪异常。"""
    pass


# --- 配置 ---

class ConfigurationError(AscError):
    """配置相关异常。"""
    pass
