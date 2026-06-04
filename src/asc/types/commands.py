"""Asc 命令类型定义。

Command 表达"意图"，可被 Master 拒绝。
与 Event（表达"已发生的事实"）严格分离。

设计原则：
- Command 由 API 或 Worker 发出，表达希望执行的操作
- 只有 Master 处理 Command，处理成功后产生对应的 Event
- Command 可以被拒绝（如找不到可用实例、资源不足等）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union

from asc.types.common import InstanceId, NodeId, TaskId

# --- 命令类型标签 ---

CommandType = Literal[
    "create_instance",
    "delete_instance",
    "start_inference",
    "cancel_task",
    "shutdown_runner",
]


def command_type(cmd: Command) -> CommandType:
    """返回命令的类型标签字符串。"""
    return _COMMAND_TYPE_MAP[type(cmd)]


# --- 模型实例命令 ---


@dataclass(frozen=True)
class CreateInstance:
    """创建模型实例。Master 决定放置到哪些节点。"""

    model_id: str
    sharding: str  # "tensor" | "pipeline"


@dataclass(frozen=True)
class DeleteInstance:
    """删除模型实例。"""

    instance_id: InstanceId


# --- 推理任务命令 ---


@dataclass(frozen=True)
class StartInference:
    """启动推理任务。"""

    instance_id: InstanceId
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.7


@dataclass(frozen=True)
class CancelTask:
    """取消推理任务。"""

    task_id: TaskId


# --- Runner 管理命令 ---


@dataclass(frozen=True)
class ShutdownRunner:
    """关闭指定节点的 Runner。"""

    node_id: NodeId


# --- 命令联合类型 ---

Command = Union[
    CreateInstance,
    DeleteInstance,
    StartInference,
    CancelTask,
    ShutdownRunner,
]


# --- 类型标签映射 ---

_COMMAND_TYPE_MAP: dict[type[Command], CommandType] = {
    CreateInstance: "create_instance",
    DeleteInstance: "delete_instance",
    StartInference: "start_inference",
    CancelTask: "cancel_task",
    ShutdownRunner: "shutdown_runner",
}
