"""Asc 事件类型定义。

事件是状态变更的唯一载体。所有事件都是不可变的（frozen dataclass）。
事件表达"已发生的事实"，与 Command（表达"意图"）严格分离。

设计原则：
- 事件一旦产生，状态就一定变更了
- 事件是不可否认的
- 给定相同的事件序列，任何节点都能到达相同状态（确定性）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union

from asc.types.common import InstanceId, NodeId, TaskId

# --- 事件类型标签 ---

EventType = Literal[
    "node_joined",
    "node_left",
    "instance_created",
    "instance_deleted",
    "task_created",
    "task_completed",
    "task_failed",
    "task_cancelled",
    "runner_status_updated",
]


def event_type(event: Event) -> EventType:
    """返回事件的类型标签字符串。"""
    return _EVENT_TYPE_MAP[type(event)]


# --- 集群拓扑事件 ---


@dataclass(frozen=True)
class NodeJoined:
    """节点加入集群。"""

    node_id: NodeId
    ip: str
    port: int


@dataclass(frozen=True)
class NodeLeft:
    """节点离开集群（主动或超时）。"""

    node_id: NodeId


# --- 模型实例事件 ---


@dataclass(frozen=True)
class InstanceCreated:
    """模型实例创建成功。"""

    instance_id: InstanceId
    model_id: str
    node_ids: list[NodeId]
    sharding: str  # "tensor" | "pipeline"


@dataclass(frozen=True)
class InstanceDeleted:
    """模型实例删除。"""

    instance_id: InstanceId


# --- 推理任务事件 ---


@dataclass(frozen=True)
class TaskCreated:
    """推理任务创建。"""

    task_id: TaskId
    instance_id: InstanceId
    prompt: str


@dataclass(frozen=True)
class TaskCompleted:
    """推理任务完成。"""

    task_id: TaskId
    output: str


@dataclass(frozen=True)
class TaskFailed:
    """推理任务失败。"""

    task_id: TaskId
    error: str


@dataclass(frozen=True)
class TaskCancelled:
    """推理任务取消。"""

    task_id: TaskId


# --- Runner 状态事件 ---


@dataclass(frozen=True)
class RunnerStatusUpdated:
    """Worker Runner 状态变更。"""

    node_id: NodeId
    status: str  # "idle" | "loading" | "ready" | "running" | "error"


# --- 事件联合类型 ---

Event = Union[
    NodeJoined,
    NodeLeft,
    InstanceCreated,
    InstanceDeleted,
    TaskCreated,
    TaskCompleted,
    TaskFailed,
    TaskCancelled,
    RunnerStatusUpdated,
]


# --- 类型标签映射 ---

_EVENT_TYPE_MAP: dict[type[Event], EventType] = {
    NodeJoined: "node_joined",
    NodeLeft: "node_left",
    InstanceCreated: "instance_created",
    InstanceDeleted: "instance_deleted",
    TaskCreated: "task_created",
    TaskCompleted: "task_completed",
    TaskFailed: "task_failed",
    TaskCancelled: "task_cancelled",
    RunnerStatusUpdated: "runner_status_updated",
}


# --- IndexedEvent ---


@dataclass(frozen=True)
class IndexedEvent:
    """带全局递增索引的事件。

    Master 为每个事件分配递增索引，保证全局有序。
    Worker 通过索引判断是否遗漏事件。
    """

    event: Event
    index: int
