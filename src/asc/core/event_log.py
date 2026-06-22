"""Asc 事件日志。

事件日志是事件溯源的关键组件：
- 持久化 IndexedEvent 到内存或磁盘
- 支持从任意索引开始重放
- 新节点可通过重放历史事件重建完整状态

性能优化：
- 支持定期自动快照，状态重建从 O(n) 降至 O(1)
- 快照 + 增量事件模式，新节点快速同步
- 快照使用 zlib 压缩减少磁盘占用
"""

from __future__ import annotations

import contextlib
import json
import zlib
from abc import ABC, abstractmethod
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Generator

from asc.types.common import InstanceId, NodeId, TaskId
from asc.types.events import Event, IndexedEvent, event_type
from asc.types.state import ClusterState, apply, empty_state

# --- 事件序列化 ---


def _serialize_event(event: Event) -> dict[str, Any]:
    """将事件序列化为字典。"""
    result: dict[str, Any] = {"event_type": event_type(event)}

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
        elif type_tag == "EventId":
            from asc.types.common import EventId
            return EventId(value)
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


# --- 快照序列化（使用 JSON 替代 pickle，防止反序列化安全风险） ---


def _serialize_state(state: ClusterState) -> dict:
    """将 ClusterState 序列化为 JSON 安全字典。"""
    from immutables import Map

    def _convert_map(m: Map) -> dict:
        """将 immutables.Map 转为普通 dict，key 转 str，value 转为 dict。"""
        result = {}
        for k, v in m.items():
            key = str(k)
            if isinstance(v, dict):
                result[key] = v
            else:
                # dataclass -> dict
                result[key] = asdict(v)
        return result

    return {
        "event_index": state.event_index,
        "nodes": _convert_map(state.nodes),
        "instances": _convert_map(state.instances),
        "tasks": _convert_map(state.tasks),
    }


def _deserialize_state(data: dict) -> ClusterState:
    """从 JSON 安全字典反序列化 ClusterState。"""
    from immutables import Map

    from asc.types.state import InstanceInfo, NodeInfo, TaskInfo

    def _parse_nodes(raw: dict) -> Map[NodeId, NodeInfo]:
        items = {}
        for k, v in raw.items():
            node_id = NodeId(k) if not k.startswith("NodeId(") else NodeId(k.split("'")[1])
            items[node_id] = NodeInfo(**v) if isinstance(v, dict) else v
        return Map(items)

    def _parse_instances(raw: dict) -> Map[InstanceId, InstanceInfo]:
        items = {}
        for k, v in raw.items():
            inst_id = InstanceId(k)
            items[inst_id] = InstanceInfo(**v) if isinstance(v, dict) else v
        return Map(items)

    def _parse_tasks(raw: dict) -> Map[TaskId, TaskInfo]:
        items = {}
        for k, v in raw.items():
            task_id = TaskId(k)
            items[task_id] = TaskInfo(**v) if isinstance(v, dict) else v
        return Map(items)

    return ClusterState(
        event_index=data.get("event_index", 0),
        nodes=_parse_nodes(data.get("nodes", {})),
        instances=_parse_instances(data.get("instances", {})),
        tasks=_parse_tasks(data.get("tasks", {})),
    )


def _serialize_snapshot(state: ClusterState, event_index: int) -> bytes:
    """将快照序列化为压缩的 JSON 字节。"""
    snapshot = {
        "event_index": event_index,
        "state": _serialize_state(state),
    }
    return zlib.compress(json.dumps(snapshot, ensure_ascii=False).encode("utf-8"))


def _deserialize_snapshot(data: bytes) -> dict:
    """反序列化快照数据。"""
    raw = json.loads(zlib.decompress(data).decode("utf-8"))
    return {
        "event_index": raw.get("event_index", 0),
        "state": _deserialize_state(raw["state"]),
    }


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

    def __init__(self, max_events: int = 0) -> None:
        self._events: list[IndexedEvent] = []
        self._max_events = max_events

    def append(self, event: IndexedEvent) -> None:
        self._events.append(event)
        if self._max_events > 0 and len(self._events) > self._max_events:
            self._events = self._events[-self._max_events :]

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
    支持缓冲写入以减少 I/O 开销。
    """

    def __init__(
        self,
        path: Path,
        buffer_size: int = 100,
        flush_interval: float = 5.0,
    ) -> None:
        self._path = path
        self._count = 0
        self._last_index = 0
        self._buffer: list[str] = []
        self._buffer_size = buffer_size
        self._flush_interval = flush_interval
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
        line = serialize_indexed_event(event)
        self._count += 1
        self._last_index = max(self._last_index, event.index)

        if self._buffer_size <= 1:
            # 无缓冲，立即写入（向后兼容）
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        else:
            self._buffer.append(line)
            if len(self._buffer) >= self._buffer_size:
                self.flush()

    def flush(self) -> None:
        """将缓冲区中的事件写入磁盘。"""
        if not self._buffer:
            return
        with open(self._path, "a", encoding="utf-8") as f:
            for line in self._buffer:
                f.write(line + "\n")
        self._buffer.clear()

    def close(self) -> None:
        """刷新缓冲区并清理资源。"""
        self.flush()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.flush()

    def read_from(self, after_index: int) -> Generator[IndexedEvent, None, None]:
        self.flush()
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


class SnapshotEventLog(EventLog):
    """支持快照的事件日志。

    定期创建状态快照，加速状态重建：
    - 新节点加入时，只需传输快照 + 增量事件
    - 状态重建从 O(n) 降至 O(1)（有快照时）
    - 快照使用 zlib 压缩减少磁盘占用
    """

    def __init__(
        self,
        path: Path,
        snapshot_interval: int = 1000,
        buffer_size: int = 100,
    ) -> None:
        self._path = path
        self._snapshot_path = path.with_suffix(".snapshot")
        self._snapshot_interval = snapshot_interval
        self._events_since_snapshot = 0
        self._inner = DiskEventLog(path, buffer_size=buffer_size)
        self._snapshot_event_index: int = 0
        # 尝试加载已有快照
        if self._snapshot_path.exists():
            try:
                with open(self._snapshot_path, "rb") as f:
                    compressed = f.read()
                data = _deserialize_snapshot(compressed)
                self._snapshot_event_index = data.get("event_index", 0)
            except Exception:
                self._snapshot_event_index = 0

    def append(self, event: IndexedEvent) -> None:
        """追加事件，达到阈值时自动创建快照。"""
        self._inner.append(event)
        self._events_since_snapshot += 1

        if self._events_since_snapshot >= self._snapshot_interval:
            self._create_snapshot()

    def _create_snapshot(self) -> None:
        """创建状态快照（使用 JSON 序列化，安全可审计）。"""
        # 从当前日志重建状态
        state = empty_state()
        for ie in self._inner.read_from(0):
            state = apply(state, ie)

        # 保存快照（JSON + zlib 压缩，替代不安全的 pickle）
        compressed = _serialize_snapshot(state, state.event_index)
        self._snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._snapshot_path, "wb") as f:
            f.write(compressed)

        self._snapshot_event_index = state.event_index
        self._events_since_snapshot = 0

    def load_snapshot(self) -> ClusterState | None:
        """加载快照状态（使用 JSON 反序列化，安全）。"""
        if not self._snapshot_path.exists():
            return None
        try:
            with open(self._snapshot_path, "rb") as f:
                compressed = f.read()
            data = _deserialize_snapshot(compressed)
            return data["state"]
        except Exception:
            return None

    def read_from(self, after_index: int) -> Generator[IndexedEvent, None, None]:
        """从指定索引之后读取事件。"""
        yield from self._inner.read_from(after_index)

    def rebuild_state(self) -> ClusterState:
        """从快照 + 增量事件重建状态（快速路径）。"""
        snapshot = self.load_snapshot()
        if snapshot is not None:
            state = snapshot
            # 仅重放快照之后的事件
            for ie in self._inner.read_from(state.event_index):
                state = apply(state, ie)
            return state

        # 无快照，从头重放
        state = empty_state()
        for ie in self._inner.read_from(0):
            state = apply(state, ie)
        return state

    def flush(self) -> None:
        """刷新缓冲区。"""
        self._inner.flush()

    def close(self) -> None:
        """刷新并清理资源。"""
        self._inner.close()

    def __len__(self) -> int:
        return len(self._inner)

    @property
    def last_index(self) -> int:
        return self._inner.last_index
