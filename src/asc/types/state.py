"""Asc 集群状态定义与事件应用。

ClusterState 是不可变的全局状态对象，包含集群所有信息。
apply() 是纯函数，接收 (State, IndexedEvent) 返回新 State。

设计原则：
- 状态不可变：所有变更通过 dataclasses.replace() 产生新对象
- 确定性：相同事件序列产生相同状态
- 事件是状态变更的唯一载体
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field, replace
from typing import Any

from asc.types.common import InstanceId, NodeId, TaskId
from asc.types.events import (
    Event,
    IndexedEvent,
    InstanceCreated,
    InstanceDeleted,
    NodeJoined,
    NodeLeft,
    RunnerStatusUpdated,
    TaskCancelled,
    TaskCompleted,
    TaskCreated,
    TaskFailed,
)

# --- 值对象 ---


@dataclass(frozen=True)
class NodeInfo:
    """节点信息快照。"""

    node_id: NodeId
    ip: str
    port: int
    runner_status: str = "unknown"


class InstanceState(enum.Enum):
    """模型实例状态。"""

    CREATING = "creating"
    RUNNING = "running"
    DEGRADED = "degraded"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass(frozen=True)
class InstanceInfo:
    """模型实例信息。"""

    instance_id: InstanceId
    model_id: str
    node_ids: list[NodeId]
    sharding: str
    state: InstanceState = InstanceState.CREATING
    rpc_endpoints: list[str] = field(default_factory=list)


class TaskStatus(enum.Enum):
    """推理任务状态。"""

    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class TaskInfo:
    """推理任务信息。"""

    task_id: TaskId
    instance_id: InstanceId
    prompt: str
    status: TaskStatus = TaskStatus.PENDING
    output: str = ""
    error: str = ""


# --- 集群状态 ---


@dataclass(frozen=True)
class ClusterState:
    """集群全局不可变状态。

    所有变更必须通过 apply(state, event) 产生新状态。
    """

    nodes: dict[NodeId, NodeInfo] = field(default_factory=dict)
    instances: dict[InstanceId, InstanceInfo] = field(default_factory=dict)
    tasks: dict[TaskId, TaskInfo] = field(default_factory=dict)
    event_index: int = 0


def empty_state() -> ClusterState:
    """创建初始空状态。"""
    return ClusterState()


# --- 事件应用 ---


def apply(state: ClusterState, indexed_event: IndexedEvent) -> ClusterState:
    """将事件应用到状态，返回新状态。

    纯函数：不修改原 state，不产生副作用。
    确定性：相同输入永远产生相同输出。
    """
    ie = indexed_event
    event = ie.event

    # 更新 event_index
    new_state = replace(state, event_index=ie.index)

    handler = _HANDLERS.get(type(event))
    if handler is None:
        return new_state  # 未知事件类型，忽略

    return handler(new_state, event)


# --- 事件处理器 ---


def _apply_node_joined(state: ClusterState, event: NodeJoined) -> ClusterState:
    new_nodes = {
        **state.nodes,
        event.node_id: NodeInfo(
            node_id=event.node_id,
            ip=event.ip,
            port=event.port,
        ),
    }
    return replace(state, nodes=new_nodes)


def _apply_node_left(state: ClusterState, event: NodeLeft) -> ClusterState:
    new_nodes = {k: v for k, v in state.nodes.items() if k != event.node_id}
    return replace(state, nodes=new_nodes)


def _apply_instance_created(state: ClusterState, event: InstanceCreated) -> ClusterState:
    new_instances = {
        **state.instances,
        event.instance_id: InstanceInfo(
            instance_id=event.instance_id,
            model_id=event.model_id,
            node_ids=list(event.node_ids),
            sharding=event.sharding,
            rpc_endpoints=list(event.rpc_endpoints),
        ),
    }
    return replace(state, instances=new_instances)


def _apply_instance_deleted(state: ClusterState, event: InstanceDeleted) -> ClusterState:
    new_instances = {k: v for k, v in state.instances.items() if k != event.instance_id}
    return replace(state, instances=new_instances)


def _apply_task_created(state: ClusterState, event: TaskCreated) -> ClusterState:
    new_task = TaskInfo(
        task_id=event.task_id,
        instance_id=event.instance_id,
        prompt=event.prompt,
    )
    new_tasks = {**state.tasks, event.task_id: new_task}
    return replace(state, tasks=new_tasks)


def _apply_task_completed(state: ClusterState, event: TaskCompleted) -> ClusterState:
    if event.task_id not in state.tasks:
        return state
    old_task = state.tasks[event.task_id]
    new_task = replace(old_task, status=TaskStatus.COMPLETED, output=event.output)
    new_tasks = {**state.tasks, event.task_id: new_task}
    return replace(state, tasks=new_tasks)


def _apply_task_failed(state: ClusterState, event: TaskFailed) -> ClusterState:
    if event.task_id not in state.tasks:
        return state
    old_task = state.tasks[event.task_id]
    new_task = replace(old_task, status=TaskStatus.FAILED, error=event.error)
    new_tasks = {**state.tasks, event.task_id: new_task}
    return replace(state, tasks=new_tasks)


def _apply_task_cancelled(state: ClusterState, event: TaskCancelled) -> ClusterState:
    if event.task_id not in state.tasks:
        return state
    old_task = state.tasks[event.task_id]
    new_task = replace(old_task, status=TaskStatus.CANCELLED)
    new_tasks = {**state.tasks, event.task_id: new_task}
    return replace(state, tasks=new_tasks)


def _apply_runner_status_updated(state: ClusterState, event: RunnerStatusUpdated) -> ClusterState:
    if event.node_id not in state.nodes:
        return state
    old_node = state.nodes[event.node_id]
    new_node = replace(old_node, runner_status=event.status)
    new_nodes = {**state.nodes, event.node_id: new_node}
    return replace(state, nodes=new_nodes)


# --- 处理器注册表 ---

_HANDLERS: dict[type[Event], Any] = {
    NodeJoined: _apply_node_joined,
    NodeLeft: _apply_node_left,
    InstanceCreated: _apply_instance_created,
    InstanceDeleted: _apply_instance_deleted,
    TaskCreated: _apply_task_created,
    TaskCompleted: _apply_task_completed,
    TaskFailed: _apply_task_failed,
    TaskCancelled: _apply_task_cancelled,
    RunnerStatusUpdated: _apply_runner_status_updated,
}
