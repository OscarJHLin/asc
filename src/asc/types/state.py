"""Asc 集群状态定义与事件应用。

ClusterState 是整个系统的核心数据结构，表示集群在某一时刻的完整快照。
采用"事件溯源（Event Sourcing）"架构：状态不是直接修改的，而是通过
应用一系列事件逐步演化得到的。

核心设计：
- 不可变性：所有变更通过 dataclasses.replace() 产生新对象，旧状态始终可用
- 纯函数：apply(state, event) -> new_state 无副作用，便于测试和重放
- 确定性：相同事件序列在任何节点上都能产生相同状态，这是分布式一致性的基础
- 事件是状态变更的唯一载体，禁止直接修改状态字段

性能优化：
- 使用 immutables.Map 替代 dict，基于 HAMT（Hash Array Mapped Trie）
- 状态更新从 O(n) 降至 O(1) 均摊，共享未变更节点
- 大规模集群（>1000节点）下显著减少内存和 CPU 开销
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field, replace
from typing import Any

from immutables import Map

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
    RUNNING = "running"
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
    使用 immutables.Map 替代 dict，状态更新 O(1) 均摊。
    """

    nodes: Map[NodeId, NodeInfo] = field(default_factory=Map)
    instances: Map[InstanceId, InstanceInfo] = field(default_factory=Map)
    tasks: Map[TaskId, TaskInfo] = field(default_factory=Map)
    event_index: int = 0


def empty_state() -> ClusterState:
    """创建初始空状态。"""
    return ClusterState()


# --- 事件应用 ---


def apply(state: ClusterState, indexed_event: IndexedEvent) -> ClusterState:
    """将事件应用到状态，返回新状态。

    这是整个系统状态变更的唯一合法入口。所有状态修改必须经过此函数，
    确保不可变性和可审计性。

    Args:
        state: 当前状态（不会被修改）
        indexed_event: 带全局索引的事件

    Returns:
        应用事件后的新状态

    设计约束：
        - 纯函数：不修改输入 state，不产生副作用
        - 确定性：相同输入永远产生相同输出
        - 幂等性：同一事件应用多次结果相同（基于 event_index）
        - 容错性：未知事件类型被静默忽略，防止新事件导致旧节点崩溃

    扩展指南：
        若需添加新事件类型，需在 _HANDLERS 字典中注册对应的处理器函数。
        处理器签名必须为 (ClusterState, EventSubtype) -> ClusterState。
    """
    ie = indexed_event
    event = ie.event

    # 更新 event_index：即使事件被忽略，索引也应前进，
    # 确保所有节点对"已处理到哪个位置"达成一致
    new_state = replace(state, event_index=ie.index)

    handler = _HANDLERS.get(type(event))
    if handler is None:
        return new_state  # 未知事件类型，忽略（向前兼容）

    return handler(new_state, event)


# --- 事件处理器 ---
# 使用 immutables.Map.set() / delete() 替代 dict 拷贝，O(1) 均摊


def _apply_node_joined(state: ClusterState, event: NodeJoined) -> ClusterState:
    new_nodes = state.nodes.set(
        event.node_id,
        NodeInfo(
            node_id=event.node_id,
            ip=event.ip,
            port=event.port,
        ),
    )
    return replace(state, nodes=new_nodes)


def _apply_node_left(state: ClusterState, event: NodeLeft) -> ClusterState:
    if event.node_id not in state.nodes:
        return state
    new_nodes = state.nodes.delete(event.node_id)
    return replace(state, nodes=new_nodes)


def _apply_instance_created(state: ClusterState, event: InstanceCreated) -> ClusterState:
    new_instances = state.instances.set(
        event.instance_id,
        InstanceInfo(
            instance_id=event.instance_id,
            model_id=event.model_id,
            node_ids=list(event.node_ids),
            sharding=event.sharding,
            rpc_endpoints=list(event.rpc_endpoints),
        ),
    )
    return replace(state, instances=new_instances)


def _apply_instance_deleted(state: ClusterState, event: InstanceDeleted) -> ClusterState:
    if event.instance_id not in state.instances:
        return state
    new_instances = state.instances.delete(event.instance_id)
    return replace(state, instances=new_instances)


def _apply_task_created(state: ClusterState, event: TaskCreated) -> ClusterState:
    new_task = TaskInfo(
        task_id=event.task_id,
        instance_id=event.instance_id,
        prompt=event.prompt,
    )
    new_tasks = state.tasks.set(event.task_id, new_task)
    return replace(state, tasks=new_tasks)


def _apply_task_completed(state: ClusterState, event: TaskCompleted) -> ClusterState:
    if event.task_id not in state.tasks:
        return state
    old_task = state.tasks[event.task_id]
    new_task = replace(old_task, status=TaskStatus.COMPLETED, output=event.output)
    new_tasks = state.tasks.set(event.task_id, new_task)
    return replace(state, tasks=new_tasks)


def _apply_task_failed(state: ClusterState, event: TaskFailed) -> ClusterState:
    if event.task_id not in state.tasks:
        return state
    old_task = state.tasks[event.task_id]
    new_task = replace(old_task, status=TaskStatus.FAILED, error=event.error)
    new_tasks = state.tasks.set(event.task_id, new_task)
    return replace(state, tasks=new_tasks)


def _apply_task_cancelled(state: ClusterState, event: TaskCancelled) -> ClusterState:
    if event.task_id not in state.tasks:
        return state
    old_task = state.tasks[event.task_id]
    new_task = replace(old_task, status=TaskStatus.CANCELLED)
    new_tasks = state.tasks.set(event.task_id, new_task)
    return replace(state, tasks=new_tasks)


def _apply_runner_status_updated(state: ClusterState, event: RunnerStatusUpdated) -> ClusterState:
    if event.node_id not in state.nodes:
        return state
    old_node = state.nodes[event.node_id]
    new_node = replace(old_node, runner_status=event.status)
    new_nodes = state.nodes.set(event.node_id, new_node)
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
