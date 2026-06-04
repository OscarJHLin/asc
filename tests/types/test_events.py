"""测试事件类型定义和 IndexedEvent。"""

from asc.types.common import InstanceId, NodeId
from asc.types.events import (
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
    event_type,
)


class TestEventType:
    """event_type 辅助函数返回事件的类型标签字符串。"""

    def test_node_joined(self):
        e = NodeJoined(node_id=NodeId("n1"), ip="192.168.1.10", port=52415)
        assert event_type(e) == "node_joined"

    def test_node_left(self):
        e = NodeLeft(node_id=NodeId("n1"))
        assert event_type(e) == "node_left"

    def test_instance_created(self):
        e = InstanceCreated(
            instance_id=InstanceId("i1"),
            model_id="llama-3.1-8b",
            node_ids=[NodeId("n1")],
            sharding="tensor",
        )
        assert event_type(e) == "instance_created"

    def test_instance_deleted(self):
        e = InstanceDeleted(instance_id=InstanceId("i1"))
        assert event_type(e) == "instance_deleted"

    def test_task_created(self):
        e = TaskCreated(task_id="t1", instance_id=InstanceId("i1"), prompt="hello")
        assert event_type(e) == "task_created"

    def test_task_completed(self):
        e = TaskCompleted(task_id="t1", output="world")
        assert event_type(e) == "task_completed"

    def test_task_failed(self):
        e = TaskFailed(task_id="t1", error="OOM")
        assert event_type(e) == "task_failed"

    def test_task_cancelled(self):
        e = TaskCancelled(task_id="t1")
        assert event_type(e) == "task_cancelled"

    def test_runner_status_updated(self):
        e = RunnerStatusUpdated(node_id=NodeId("n1"), status="ready")
        assert event_type(e) == "runner_status_updated"


class TestEventImmutability:
    """所有事件必须不可变（frozen dataclass）。"""

    def test_node_joined_frozen(self):
        e = NodeJoined(node_id=NodeId("n1"), ip="10.0.0.1", port=52415)
        try:
            e.ip = "10.0.0.2"  # type: ignore[misc]
            raise AssertionError("Should be immutable")
        except (AttributeError, TypeError):
            pass

    def test_instance_created_frozen(self):
        e = InstanceCreated(
            instance_id=InstanceId("i1"),
            model_id="m",
            node_ids=[NodeId("n1")],
            sharding="tensor",
        )
        try:
            e.model_id = "other"  # type: ignore[misc]
            raise AssertionError("Should be immutable")
        except (AttributeError, TypeError):
            pass


class TestEventEquality:
    """相同字段值的事件应相等。"""

    def test_node_joined_equality(self):
        a = NodeJoined(node_id=NodeId("n1"), ip="10.0.0.1", port=52415)
        b = NodeJoined(node_id=NodeId("n1"), ip="10.0.0.1", port=52415)
        assert a == b

    def test_node_joined_inequality(self):
        a = NodeJoined(node_id=NodeId("n1"), ip="10.0.0.1", port=52415)
        b = NodeJoined(node_id=NodeId("n1"), ip="10.0.0.2", port=52415)
        assert a != b


class TestIndexedEvent:
    """IndexedEvent 为事件附加全局递增索引。"""

    def test_create(self):
        e = NodeJoined(node_id=NodeId("n1"), ip="10.0.0.1", port=52415)
        ie = IndexedEvent(event=e, index=1)
        assert ie.index == 1
        assert ie.event == e

    def test_frozen(self):
        e = NodeJoined(node_id=NodeId("n1"), ip="10.0.0.1", port=52415)
        ie = IndexedEvent(event=e, index=1)
        try:
            ie.index = 2  # type: ignore[misc]
            raise AssertionError("Should be immutable")
        except (AttributeError, TypeError):
            pass

    def test_index_ordering(self):
        e = NodeJoined(node_id=NodeId("n1"), ip="10.0.0.1", port=52415)
        ie1 = IndexedEvent(event=e, index=1)
        ie2 = IndexedEvent(event=e, index=2)
        assert ie1.index < ie2.index


class TestEventUnion:
    """Event 联合类型应能正确判别子类型。"""

    def test_isinstance_node_joined(self):
        e = NodeJoined(node_id=NodeId("n1"), ip="10.0.0.1", port=52415)
        assert isinstance(e, NodeJoined)

    def test_isinstance_instance_created(self):
        e = InstanceCreated(
            instance_id=InstanceId("i1"),
            model_id="m",
            node_ids=[NodeId("n1")],
            sharding="tensor",
        )
        assert isinstance(e, InstanceCreated)

    def test_discrimination(self):
        """不同事件类型不应是同一 isinstance。"""
        e1 = NodeJoined(node_id=NodeId("n1"), ip="10.0.0.1", port=52415)
        e2 = NodeLeft(node_id=NodeId("n1"))
        assert not isinstance(e1, NodeLeft)
        assert not isinstance(e2, NodeJoined)
