"""测试事件日志：DiskEventLog 持久化 + 重放恢复。

事件日志是事件溯源的关键组件：
- 持久化事件到磁盘
- 支持从任意索引开始重放
- 新节点可通过重放历史事件重建完整状态
"""

import tempfile
from pathlib import Path

from asc.core.event_log import DiskEventLog, MemoryEventLog
from asc.types.common import InstanceId, NodeId
from asc.types.events import (
    IndexedEvent,
    InstanceCreated,
    NodeJoined,
    NodeLeft,
)
from asc.types.state import apply, empty_state


class TestMemoryEventLog:
    """内存事件日志（用于测试和快速场景）。"""

    def test_append_and_read(self):
        log = MemoryEventLog()
        event = NodeJoined(node_id=NodeId("n1"), ip="10.0.0.1", port=52415)
        ie = IndexedEvent(event=event, index=1)
        log.append(ie)

        events = list(log.read_from(0))
        assert len(events) == 1
        assert events[0].index == 1

    def test_read_from_offset(self):
        log = MemoryEventLog()
        for i in range(5):
            event = NodeJoined(node_id=NodeId(f"n{i}"), ip=f"10.0.0.{i}", port=52415)
            log.append(IndexedEvent(event=event, index=i + 1))

        events = list(log.read_from(3))
        assert len(events) == 2
        assert events[0].index == 4
        assert events[1].index == 5

    def test_read_from_zero(self):
        log = MemoryEventLog()
        event = NodeJoined(node_id=NodeId("n1"), ip="10.0.0.1", port=52415)
        log.append(IndexedEvent(event=event, index=1))

        events = list(log.read_from(0))
        assert len(events) == 1

    def test_read_from_beyond_end(self):
        log = MemoryEventLog()
        event = NodeJoined(node_id=NodeId("n1"), ip="10.0.0.1", port=52415)
        log.append(IndexedEvent(event=event, index=1))

        events = list(log.read_from(10))
        assert len(events) == 0

    def test_len(self):
        log = MemoryEventLog()
        assert len(log) == 0
        event = NodeJoined(node_id=NodeId("n1"), ip="10.0.0.1", port=52415)
        log.append(IndexedEvent(event=event, index=1))
        assert len(log) == 1

    def test_last_index(self):
        log = MemoryEventLog()
        assert log.last_index == 0
        event = NodeJoined(node_id=NodeId("n1"), ip="10.0.0.1", port=52415)
        log.append(IndexedEvent(event=event, index=5))
        assert log.last_index == 5


class TestDiskEventLog:
    """磁盘事件日志。"""

    def test_append_and_read(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log = DiskEventLog(Path(tmpdir) / "events.log")
            event = NodeJoined(node_id=NodeId("n1"), ip="10.0.0.1", port=52415)
            log.append(IndexedEvent(event=event, index=1))

            events = list(log.read_from(0))
            assert len(events) == 1
            assert events[0].index == 1

    def test_read_from_offset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log = DiskEventLog(Path(tmpdir) / "events.log")
            for i in range(5):
                event = NodeJoined(node_id=NodeId(f"n{i}"), ip=f"10.0.0.{i}", port=52415)
                log.append(IndexedEvent(event=event, index=i + 1))

            events = list(log.read_from(3))
            assert len(events) == 2

    def test_persistence_across_instances(self):
        """不同实例读取同一文件。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "events.log"

            log1 = DiskEventLog(path)
            event = NodeJoined(node_id=NodeId("n1"), ip="10.0.0.1", port=52415)
            log1.append(IndexedEvent(event=event, index=1))
            del log1  # 关闭

            log2 = DiskEventLog(path)
            events = list(log2.read_from(0))
            assert len(events) == 1
            assert events[0].index == 1

    def test_len(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log = DiskEventLog(Path(tmpdir) / "events.log")
            assert len(log) == 0
            event = NodeJoined(node_id=NodeId("n1"), ip="10.0.0.1", port=52415)
            log.append(IndexedEvent(event=event, index=1))
            assert len(log) == 1

    def test_last_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log = DiskEventLog(Path(tmpdir) / "events.log")
            assert log.last_index == 0
            event = NodeJoined(node_id=NodeId("n1"), ip="10.0.0.1", port=52415)
            log.append(IndexedEvent(event=event, index=7))
            assert log.last_index == 7

    def test_multiple_event_types(self):
        """不同类型的事件都能正确持久化和读取。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            log = DiskEventLog(Path(tmpdir) / "events.log")
            log.append(
                IndexedEvent(
                    event=NodeJoined(node_id=NodeId("n1"), ip="10.0.0.1", port=52415), index=1
                )
            )
            log.append(
                IndexedEvent(
                    event=InstanceCreated(
                        instance_id=InstanceId("i1"),
                        model_id="m",
                        node_ids=[NodeId("n1")],
                        sharding="tensor",
                    ),
                    index=2,
                )
            )
            log.append(IndexedEvent(event=NodeLeft(node_id=NodeId("n1")), index=3))

            events = list(log.read_from(0))
            assert len(events) == 3
            assert events[0].index == 1
            assert events[1].index == 2
            assert events[2].index == 3


class TestEventLogReplay:
    """通过重放事件日志重建状态。"""

    def test_replay_rebuilds_state(self):
        """重放事件日志应产生与原始状态相同的结果。"""
        log = MemoryEventLog()

        # 构建事件序列
        events = [
            IndexedEvent(
                event=NodeJoined(node_id=NodeId("n1"), ip="10.0.0.1", port=52415), index=1
            ),
            IndexedEvent(
                event=NodeJoined(node_id=NodeId("n2"), ip="10.0.0.2", port=52415), index=2
            ),
            IndexedEvent(
                event=InstanceCreated(
                    instance_id=InstanceId("i1"),
                    model_id="m",
                    node_ids=[NodeId("n1")],
                    sharding="tensor",
                ),
                index=3,
            ),
            IndexedEvent(event=NodeLeft(node_id=NodeId("n2")), index=4),
        ]
        for ie in events:
            log.append(ie)

        # 重放
        state = empty_state()
        for ie in log.read_from(0):
            state = apply(state, ie)

        assert len(state.nodes) == 1
        assert NodeId("n1") in state.nodes
        assert NodeId("n2") not in state.nodes
        assert len(state.instances) == 1
        assert state.event_index == 4

    def test_partial_replay(self):
        """从中间索引开始重放。"""
        log = MemoryEventLog()
        events = [
            IndexedEvent(
                event=NodeJoined(node_id=NodeId("n1"), ip="10.0.0.1", port=52415), index=1
            ),
            IndexedEvent(
                event=NodeJoined(node_id=NodeId("n2"), ip="10.0.0.2", port=52415), index=2
            ),
            IndexedEvent(event=NodeLeft(node_id=NodeId("n1")), index=3),
        ]
        for ie in events:
            log.append(ie)

        # 假设节点已处理到 index=1，只需重放 2 和 3
        state = empty_state()
        state = apply(
            state,
            IndexedEvent(
                event=NodeJoined(node_id=NodeId("n1"), ip="10.0.0.1", port=52415), index=1
            ),
        )

        for ie in log.read_from(1):
            state = apply(state, ie)

        assert len(state.nodes) == 1
        assert NodeId("n2") in state.nodes
