"""Asc 推理引擎抽象接口。

定义了推理引擎的统一契约，将"引擎构建"与"推理执行"明确分离：
- Builder 阶段（EngineBuilder）：查找可执行文件、加载模型、预热 -> 返回 Engine
- Engine 阶段（Engine）：submit(task) / step() 循环 -> 产出推理结果

设计背景：
    早期实现直接调用 llama-cli 命令行，每次推理都启动新进程，导致严重的
    冷启动延迟（数秒到数十秒）。改为 llama-server 常驻进程 + HTTP API 后，
    首次加载后推理延迟降至毫秒级，且天然支持流式 SSE 输出。

关键特性：
- 使用 llama-server 常驻进程 + HTTP API，而非每次启动 llama-cli
- 天然支持流式输出（SSE）
- 避免模型冷启动开销
- 支持分布式推理（通过 --rpc 和 --tensor-split 参数）

扩展指南：
    若需支持其他推理后端（如 vLLM、TensorRT-LLM），只需实现 EngineBuilder
    和 Engine 接口，无需修改上层调度代码。
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generator

# --- 值对象 ---


@dataclass(frozen=True)
class InferenceRequest:
    """推理请求。"""

    prompt: str
    max_tokens: int = 128
    temperature: float = 0.7


@dataclass(frozen=True)
class InferenceResult:
    """推理结果。"""

    text: str
    tokens_generated: int
    tokens_per_second: float
    error: str | None = None


@dataclass(frozen=True)
class LoadProgress:
    """模型加载进度。"""

    current: int
    total: int
    message: str

    @property
    def fraction(self) -> float:
        if self.total == 0:
            return 0.0
        return self.current / self.total


class EngineStatus(enum.Enum):
    """引擎状态。"""

    IDLE = "idle"
    LOADING = "loading"
    READY = "ready"
    RUNNING = "running"
    ERROR = "error"
    SHUTDOWN = "shutdown"


# --- Builder 抽象 ---


class EngineBuilder(ABC):
    """推理引擎构建器。

    将"构建阶段"和"运行阶段"分离：
    - 构建阶段：查找可执行文件、启动进程、加载模型、预热
    - 运行阶段：由 build() 返回的 Engine 负责
    """

    @abstractmethod
    def load(self) -> Generator[LoadProgress, None, None]:
        """加载模型，yield 加载进度。"""
        ...

    @abstractmethod
    def build(self) -> Engine:
        """构建完成，返回可用的 Engine 实例。"""
        ...


# --- Engine 抽象 ---


class Engine(ABC):
    """推理引擎运行时接口。

    核心循环：submit(task) -> step() -> (task_id, chunk)
    """

    @abstractmethod
    def submit(self, request: InferenceRequest) -> str:
        """提交推理请求，返回任务 ID。"""
        ...

    @abstractmethod
    def step(self) -> list[tuple[str, str]]:
        """推进推理，返回 [(task_id, token_chunk), ...]。"""
        ...

    @abstractmethod
    def close(self) -> None:
        """关闭引擎，释放资源。"""
        ...

    @abstractmethod
    def status(self) -> EngineStatus:
        """返回当前引擎状态。"""
        ...
