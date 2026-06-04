"""Asc 核心基础类型。

使用 NewType 实现类型安全的 ID，使用 frozen dataclass 实现不可变值对象。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import NewType

# --- 强类型 ID ---

NodeId = NewType("NodeId", str)
InstanceId = NewType("InstanceId", str)
TaskId = NewType("TaskId", str)
EventId = NewType("EventId", str)


# --- ID 生成器 ---


def generate_node_id() -> NodeId:
    """生成唯一的节点 ID。"""
    return NodeId(f"node-{uuid.uuid4().hex[:12]}")


def generate_instance_id() -> InstanceId:
    """生成唯一的实例 ID。"""
    return InstanceId(f"inst-{uuid.uuid4().hex[:12]}")


def generate_task_id() -> TaskId:
    """生成唯一的任务 ID。"""
    return TaskId(f"task-{uuid.uuid4().hex[:12]}")


def generate_event_id() -> EventId:
    """生成唯一的事件 ID。"""
    return EventId(f"evt-{uuid.uuid4().hex[:16]}")


# --- 不可变值对象 ---


@dataclass(frozen=True)
class SessionId:
    """标识一次 Master 选举周期。

    当 Master 发生变更时，SessionId 会更新，所有节点据此判断
    当前是否处于同一会话。
    """

    master_node_id: NodeId
    election_clock: int
