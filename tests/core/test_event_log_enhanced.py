"""事件日志增强测试。

覆盖审计中发现的关键缺口：
- _deserialize_value 对 EventId 类型的处理
- MemoryEventLog.max_events 事件驱逐
- DiskEventLog 缓冲写入和刷新
- SnapshotEventLog 快照创建、加载、rebuild_state、损坏快照
- DiskEventLog.read_from 文件不存在
- 未知事件类型反序列化抛出 ValueError
"""

from pathlib import Path

import pytest

from asc.core.event_log import (
    DiskEventLog,
    MemoryEventLog,
    SnapshotEventLog,
    _deserialize_event,
    _deserialize_value,
    _serialize_value,
)
from asc.types.common import EventId, InstanceId, NodeId, TaskId
from asc.types.events import IndexedEvent, NodeJoined, NodeLeft


class TestDeserializeValue:
    """_deserialize_value 函数测试。"""

    def test_event_id_type(self):
        """EventId 类型应正确反序列化。"""
        val = {"__type__": "EventId", "value": "evt-abc123"}
        result = _deserialize_value(val)
        assert result == EventId("evt-abc123")

    def test_node_id_type(self):
        val = {"__type__": "NodeId", "value": "n1"}
        result = _deserialize_value(val)
        assert result == NodeId("n1")

    def test_instance_id_type(self):
        val = {"__type__": "InstanceId", "value": "i1"}
        result = _deserialize_value(val)
        assert result == InstanceId("i1")

    def test_task_id_type(self):
        val = {"__type__": "TaskId", "value": "t1"}
        result = _deserialize_value(val)
        assert result == TaskId("t1")

    def test_unknown_type_tag_returns_raw(self):
        """未知 __type__ 标记应原样返回字典。"""
        val = {"__type__": "UnknownType", "value": "x"}
        result = _deserialize_value(val)
        assert result == val

    def test_plain_value(self):
        """普通值应原样返回。"""
        assert _deserialize_value("hello") == "hello"
        assert _deserialize_value(42) == 42

    def test_list_value(self):
        """列表中的元素应递归反序列化。"""
        val = [{"__type__": "NodeId", "value": "n1"}, "plain"]
        result = _deserialize_value(val)
        assert result == [NodeId("n1"), "plain"]

    def test_nested_dict_without_type(self):
        """不含 __type__ 的字典应原样返回。"""
        val = {"key": "value"}
        result = _deserialize_value(val)
        assert result == {"key": "value"}


class TestSerializeValue:
    """_serialize_value 函数测试。"""

    def test_event_id_type(self):
        result = _serialize_value("evt-1", "EventId")
        assert result == {"__type__": "EventId", "value": "evt-1"}

    def test_list_serialization(self):
        result = _serialize_value([1, 2, 3])
        assert result == [1, 2, 3]


class TestMemoryEventLogMaxEvents:
    """MemoryEventLog 事件驱逐测试。"""

    def test_max_events_eviction(self):
        """超过 max_events 时应驱逐最旧的事件。"""
        log = MemoryEventLog(max_events=3)
        for i in range(5):
            event = NodeJoined(node_id=NodeId(f"n{i}"), ip=f"10.0.0.{i}", port=52415)
            log.append(IndexedEvent(event=event, index=i + 1))

        assert len(log) == 3
        events = list(log.read_from(0))
        indices = [e.index for e in events]
        assert indices == [3, 4, 5]

    def test_max_events_zero_means_unlimited(self):
        """max_events=0 表示无限制。"""
        log = MemoryEventLog(max_events=0)
        for i in range(10):
            event = NodeJoined(node_id=NodeId(f"n{i}"), ip=f"10.0.0.{i}", port=52415)
            log.append(IndexedEvent(event=event, index=i + 1))

        assert len(log) == 10

    def test_max_events_exact_fit(self):
        """事件数恰好等于 max_events 时不驱逐。"""
        log = MemoryEventLog(max_events=3)
        for i in range(3):
            event = NodeJoined(node_id=NodeId(f"n{i}"), ip=f"10.0.0.{i}", port=52415)
            log.append(IndexedEvent(event=event, index=i + 1))

        assert len(log) == 3
        events = list(log.read_from(0))
        assert events[0].index == 1

    def test_last_index_after_eviction(self):
        """驱逐后 last_index 应为最新事件的索引。"""
        log = MemoryEventLog(max_events=2)
        for i in range(4):
            event = NodeJoined(node_id=NodeId(f"n{i}"), ip=f"10.0.0.{i}", port=52415)
            log.append(IndexedEvent(event=event, index=i + 1))

        assert log.last_index == 4


class TestDiskEventLogBuffer:
    """DiskEventLog 缓冲写入和刷新测试。"""

    def test_buffer_write_and_flush(self, tmp_path: Path):
        """事件先写入缓冲区，flush 后才持久化到磁盘。"""
        log_path = tmp_path / "events.log"
        log = DiskEventLog(log_path, buffer_size=10)

        event = NodeJoined(node_id=NodeId("n1"), ip="10.0.0.1", port=52415)
        log.append(IndexedEvent(event=event, index=1))

        # 缓冲区未满，文件可能为空
        assert len(log) == 1

        # flush 后文件应有内容
        log.flush()
        assert log_path.exists()
        lines = log_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1

    def test_buffer_auto_flush_on_full(self, tmp_path: Path):
        """缓冲区满时自动 flush。"""
        log_path = tmp_path / "events.log"
        log = DiskEventLog(log_path, buffer_size=2)

        for i in range(3):
            event = NodeJoined(node_id=NodeId(f"n{i}"), ip=f"10.0.0.{i}", port=52415)
            log.append(IndexedEvent(event=event, index=i + 1))

        # 3 个事件，buffer_size=2，前两个触发自动 flush
        log.flush()
        lines = [line for line in log_path.read_text(encoding="utf-8").strip().split("\n") if line]
        assert len(lines) == 3

    def test_no_buffer_immediate_write(self, tmp_path: Path):
        """buffer_size=1 时应立即写入，无缓冲。"""
        log_path = tmp_path / "events.log"
        log = DiskEventLog(log_path, buffer_size=1)

        event = NodeJoined(node_id=NodeId("n1"), ip="10.0.0.1", port=52415)
        log.append(IndexedEvent(event=event, index=1))

        # 立即写入，无需 flush
        lines = [line for line in log_path.read_text(encoding="utf-8").strip().split("\n") if line]
        assert len(lines) == 1

    def test_close_flushes_buffer(self, tmp_path: Path):
        """close() 应刷新缓冲区。"""
        log_path = tmp_path / "events.log"
        log = DiskEventLog(log_path, buffer_size=100)

        event = NodeJoined(node_id=NodeId("n1"), ip="10.0.0.1", port=52415)
        log.append(IndexedEvent(event=event, index=1))

        log.close()
        lines = [line for line in log_path.read_text(encoding="utf-8").strip().split("\n") if line]
        assert len(lines) == 1

    def test_read_from_nonexistent_file(self, tmp_path: Path):
        """read_from 文件不存在时应返回空迭代器。"""
        log_path = tmp_path / "nonexistent.log"
        log = DiskEventLog(log_path, buffer_size=1)

        events = list(log.read_from(0))
        assert events == []

    def test_read_from_flushes_first(self, tmp_path: Path):
        """read_from 应先 flush 缓冲区。"""
        log_path = tmp_path / "events.log"
        log = DiskEventLog(log_path, buffer_size=100)

        event = NodeJoined(node_id=NodeId("n1"), ip="10.0.0.1", port=52415)
        log.append(IndexedEvent(event=event, index=1))

        # read_from 内部会 flush
        events = list(log.read_from(0))
        assert len(events) == 1


class TestSnapshotEventLog:
    """SnapshotEventLog 快照测试。"""

    def test_snapshot_creation(self, tmp_path: Path):
        """达到 snapshot_interval 时自动创建快照。"""
        log_path = tmp_path / "events.log"
        log = SnapshotEventLog(log_path, snapshot_interval=3, buffer_size=1)

        for i in range(3):
            event = NodeJoined(node_id=NodeId(f"n{i}"), ip=f"10.0.0.{i}", port=52415)
            log.append(IndexedEvent(event=event, index=i + 1))

        # 快照文件应已创建
        snapshot_path = log_path.with_suffix(".snapshot")
        assert snapshot_path.exists()

    def test_load_snapshot(self, tmp_path: Path):
        """load_snapshot 应返回正确的状态。"""
        log_path = tmp_path / "events.log"
        log = SnapshotEventLog(log_path, snapshot_interval=2, buffer_size=1)

        for i in range(2):
            event = NodeJoined(node_id=NodeId(f"n{i}"), ip=f"10.0.0.{i}", port=52415)
            log.append(IndexedEvent(event=event, index=i + 1))

        state = log.load_snapshot()
        assert state is not None
        assert len(state.nodes) == 2

    def test_load_snapshot_none_when_no_snapshot(self, tmp_path: Path):
        """无快照时 load_snapshot 返回 None。"""
        log_path = tmp_path / "events.log"
        log = SnapshotEventLog(log_path, snapshot_interval=100, buffer_size=1)

        event = NodeJoined(node_id=NodeId("n1"), ip="10.0.0.1", port=52415)
        log.append(IndexedEvent(event=event, index=1))

        assert log.load_snapshot() is None

    def test_rebuild_state_with_snapshot(self, tmp_path: Path):
        """rebuild_state 使用快照 + 增量事件重建状态。"""
        log_path = tmp_path / "events.log"
        log = SnapshotEventLog(log_path, snapshot_interval=2, buffer_size=1)

        # 添加 3 个事件：前 2 个触发快照，第 3 个是增量
        log.append(IndexedEvent(
            event=NodeJoined(node_id=NodeId("n1"), ip="10.0.0.1", port=52415), index=1
        ))
        log.append(IndexedEvent(
            event=NodeJoined(node_id=NodeId("n2"), ip="10.0.0.2", port=52415), index=2
        ))
        log.append(IndexedEvent(
            event=NodeLeft(node_id=NodeId("n1")), index=3
        ))

        state = log.rebuild_state()
        assert len(state.nodes) == 1
        assert NodeId("n2") in state.nodes
        assert NodeId("n1") not in state.nodes

    def test_rebuild_state_without_snapshot(self, tmp_path: Path):
        """无快照时 rebuild_state 从头重放所有事件。"""
        log_path = tmp_path / "events.log"
        log = SnapshotEventLog(log_path, snapshot_interval=100, buffer_size=1)

        log.append(IndexedEvent(
            event=NodeJoined(node_id=NodeId("n1"), ip="10.0.0.1", port=52415), index=1
        ))
        log.append(IndexedEvent(
            event=NodeJoined(node_id=NodeId("n2"), ip="10.0.0.2", port=52415), index=2
        ))

        state = log.rebuild_state()
        assert len(state.nodes) == 2

    def test_corrupted_snapshot_fallback(self, tmp_path: Path):
        """损坏的快照应被忽略，从头重放。"""
        log_path = tmp_path / "events.log"
        snapshot_path = log_path.with_suffix(".snapshot")

        # 写入损坏的快照
        snapshot_path.write_bytes(b"corrupted data that is not valid zlib")

        log = SnapshotEventLog(log_path, snapshot_interval=100, buffer_size=1)

        log.append(IndexedEvent(
            event=NodeJoined(node_id=NodeId("n1"), ip="10.0.0.1", port=52415), index=1
        ))

        # 损坏快照应被忽略，load_snapshot 返回 None
        assert log.load_snapshot() is None

        # rebuild_state 应从头重放
        state = log.rebuild_state()
        assert len(state.nodes) == 1

    def test_flush_delegates_to_inner(self, tmp_path: Path):
        """flush 应委托给内部 DiskEventLog。"""
        log_path = tmp_path / "events.log"
        log = SnapshotEventLog(log_path, snapshot_interval=100, buffer_size=100)

        event = NodeJoined(node_id=NodeId("n1"), ip="10.0.0.1", port=52415)
        log.append(IndexedEvent(event=event, index=1))

        log.flush()
        events = list(log.read_from(0))
        assert len(events) == 1


class TestUnknownEventType:
    """未知事件类型反序列化测试。"""

    def test_unknown_event_type_raises_value_error(self):
        """未知事件类型应抛出 ValueError。"""
        data = {
            "event_type": "unknown_event_type",
            "data": {"field": "value"},
        }
        with pytest.raises(ValueError, match="未知事件类型"):
            _deserialize_event(data)
