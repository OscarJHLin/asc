"""Asc 事件日志。

事件日志是事件溯源的关键组件：
- 持久化 IndexedEvent 到内存或磁盘
- 支持从任意索引开始重放
- 新节点可通过重放历史事件重建完整状态

设计原则：
- 事件日志只追加（append-only），不修改历史
- 每个事件有全局递增索引
- 磁盘格式：每行一个 JSON 对象（JSONL）
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import fields
from pathlib import Path
from typing import Generator

from asc.types.common import InstanceId, NodeId, TaskId
from asc.types.events import Event, IndexedEvent, event_type

# --- 事件序列化 ---


def _serialize_event(event: Event) -> dict:
    """将事件序列化为字典。"""
    result: dict = {"event_type": event_type(event)}

    # NewType 类型名到标记的映射
    typed_fields = _get_typed_field_names(event)

    result["data"] = {}
    for f in fields(event):
        val = getattr(event, f.name)
        type_tag = typed_fields.get(f.name)
        result["data"][f.name] = _serialize_value(val, type_tag)

    return result


def _get_typed_field_names(event: Event) -> dict[str, str]:
    """从字段的类型注解中提取 NewType 标记。"""
    tags: dict[str, str] = {}
    for f in fields(event):
        type_name = _get_type_name(f.type)
        if type_name in ("NodeId", "InstanceId", "TaskId", "EventId"):
            tags[f.name] = type_name
    return tags


def _get_type_name(type_hint: object) -> str:
    """从类型注解中提取类型名称。"""
    if isinstance(type_hint, str):
        return type_hint
    return getattr(type_hint, "__name__", "")


def _serialize_value(val: object, type_tag: str | None = None) -> object:
    """序列化单个值。"""
    if type_tag and type_tag in ("NodeId", "InstanceId", "TaskId", "EventId"):
        return {"__type__": type_tag, "value": val}
    if isinstance(val, list):
        return [_serialize_value(v) for v in val]
    return val


def _deserialize_value(val: object) -> object:
    """反序列化单个值。"""
    if isinstance(val, dict) and "__type__" in val:
        type_tag = val["__type__"]
        value = val["value"]
        if type_tag == "NodeId":
            return NodeId(value)
        elif type_tag == "InstanceId":
            return InstanceId(value)
        elif type_tag == "TaskId":
            return TaskId(value)
    elif isinstance(val, list):
        return [_deserialize_value(v) for v in val]
    return val


def _deserialize_event(data: dict) -> Event:
    """从字典反序列化事件。"""
    from asc.types.events import (
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

    type_map = {
        "node_joined": NodeJoined,
        "node_left": NodeLeft,
        "instance_created": InstanceCreated,
        "instance_deleted": InstanceDeleted,
        "task_created": TaskCreated,
        "task_completed": TaskCompleted,
        "task_failed": TaskFailed,
        "task_cancelled": TaskCancelled,
        "runner_status_updated": RunnerStatusUpdated,
    }

    event_cls = type_map.get(data["event_type"])
    if event_cls is None:
        raise ValueError(f"未知事件类型: {data['event_type']}")

    kwargs = {}
    for k, v in data["data"].items():
        kwargs[k] = _deserialize_value(v)

    return event_cls(**kwargs)


def serialize_indexed_event(ie: IndexedEvent) -> str:
    """将 IndexedEvent 序列化为 JSON 字符串。"""
    return json.dumps(
        {
            "index": ie.index,
            "event": _serialize_event(ie.event),
        },
        ensure_ascii=False,
    )


def deserialize_indexed_event(s: str) -> IndexedEvent:
    """从 JSON 字符串反序列化 IndexedEvent。"""
    data = json.loads(s)
    return IndexedEvent(
        index=data["index"],
        event=_deserialize_event(data["event"]),
    )


# --- 事件日志抽象 ---


class EventLog(ABC):
    """事件日志抽象接口。"""

    @abstractmethod
    def append(self, event: IndexedEvent) -> None:
        """追加事件。"""
        ...

    @abstractmethod
    def read_from(self, after_index: int) -> Generator[IndexedEvent, None, None]:
        """从指定索引之后开始读取事件。"""
        ...

    @abstractmethod
    def __len__(self) -> int: ...

    @property
    @abstractmethod
    def last_index(self) -> int: ...


class MemoryEventLog(EventLog):
    """内存事件日志。用于测试和快速场景。"""

    def __init__(self) -> None:
        self._events: list[IndexedEvent] = []

    def append(self, event: IndexedEvent) -> None:
        self._events.append(event)

    def read_from(self, after_index: int) -> Generator[IndexedEvent, None, None]:
        for ie in self._events:
            if ie.index > after_index:
                yield ie

    def __len__(self) -> int:
        return len(self._events)

    @property
    def last_index(self) -> int:
        if not self._events:
            return 0
        return self._events[-1].index


class DiskEventLog(EventLog):
    """磁盘事件日志。

    格式：JSONL（每行一个 IndexedEvent 的 JSON 表示）。
    支持持久化和跨实例读取。
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._count = 0
        self._last_index = 0
        # 读取已有文件
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self._count += 1
                        ie = deserialize_indexed_event(line)
                        self._last_index = max(self._last_index, ie.index)

    def append(self, event: IndexedEvent) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(serialize_indexed_event(event) + "\n")
        self._count += 1
        self._last_index = max(self._last_index, event.index)

    def read_from(self, after_index: int) -> Generator[IndexedEvent, None, None]:
        if not self._path.exists():
            return
        with open(self._path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ie = deserialize_indexed_event(line)
                if ie.index > after_index:
                    yield ie

    def __len__(self) -> int:
        return self._count

    @property
    def last_index(self) -> int:
        return self._last_index
