"""测试 ClusterState 和 apply() 纯函数。

apply(state, indexed_event) -> new_state 是事件溯源的核心：
- 纯函数，无副作用
- 确定性：相同输入永远产生相同输出
- 不可变：原 state 不被修改，返回新 state
"""

from asc.types.common import InstanceId, NodeId, TaskId
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
)
from asc.types.state import (
    ClusterState,
    InstanceInfo,
    NodeInfo,
    TaskInfo,
    TaskStatus,
    apply,
    empty_state,
)


class TestEmptyState:
    """初始空状态。"""

    def test_empty_state_has_no_nodes(self):
        s = empty_state()
        assert len(s.nodes) == 0

    def test_empty_state_has_no_instances(self):
        s = empty_state()
        assert len(s.instances) == 0

    def test_empty_state_has_no_tasks(self):
        s = empty_state()
        assert len(s.tasks) == 0

    def test_empty_state_event_index_is_zero(self):
        s = empty_state()
        assert s.event_index == 0


class TestStateImmutability:
    """State 必须不可变。"""

    def test_state_is_frozen(self):
        s = empty_state()
        try:
            s.event_index = 99  # type: ignore[misc]
            assert False, "Should be immutable"
        except (AttributeError, TypeError):
            pass

    def test_apply_returns_new_state(self):
        s1 = empty_state()
        event = NodeJoined(node_id=NodeId("n1"), ip="10.0.0.1", port=52415)
        ie = IndexedEvent(event=event, index=1)
        s2 = apply(s1, ie)
        assert s1 is not s2
        assert len(s1.nodes) == 0  # 原 state 不变
        assert len(s2.nodes) == 1  # 新 state 已更新


class TestApplyNodeJoined:
    """NodeJoined 事件将节点添加到集群。"""

    def test_adds_node(self):
        s = empty_state()
        event = NodeJoined(node_id=NodeId("n1"), ip="10.0.0.1", port=52415)
        s2 = apply(s, IndexedEvent(event=event, index=1))
        assert NodeId("n1") in s2.nodes

    def test_node_info_stored(self):
        s = empty_state()
        event = NodeJoined(node_id=NodeId("n1"), ip="10.0.0.1", port=52415)
        s2 = apply(s, IndexedEvent(event=event, index=1))
        node = s2.nodes[NodeId("n1")]
        assert node.ip == "10.0.0.1"
        assert node.port == 52415

    def test_increments_event_index(self):
        s = empty_state()
        event = NodeJoined(node_id=NodeId("n1"), ip="10.0.0.1", port=52415)
        s2 = apply(s, IndexedEvent(event=event, index=1))
        assert s2.event_index == 1

    def test_multiple_nodes(self):
        s = empty_state()
        s = apply(s, IndexedEvent(event=NodeJoined(node_id=NodeId("n1"), ip="10.0.0.1", port=52415), index=1))
        s = apply(s, IndexedEvent(event=NodeJoined(node_id=NodeId("n2"), ip="10.0.0.2", port=52415), index=2))
        assert len(s.nodes) == 2

    def test_duplicate_node_joined_updates_info(self):
        """同一节点重复加入应更新信息而非报错。"""
        s = empty_state()
        s = apply(s, IndexedEvent(event=NodeJoined(node_id=NodeId("n1"), ip="10.0.0.1", port=52415), index=1))
        s = apply(s, IndexedEvent(event=NodeJoined(node_id=NodeId("n1"), ip="10.0.0.99", port=52416), index=2))
        assert len(s.nodes) == 1
        assert s.nodes[NodeId("n1")].ip == "10.0.0.99"
        assert s.nodes[NodeId("n1")].port == 52416


class TestApplyNodeLeft:
    """NodeLeft 事件将节点从集群移除。"""

    def test_removes_node(self):
        s = empty_state()
        s = apply(s, IndexedEvent(event=NodeJoined(node_id=NodeId("n1"), ip="10.0.0.1", port=52415), index=1))
        s = apply(s, IndexedEvent(event=NodeLeft(node_id=NodeId("n1")), index=2))
        assert NodeId("n1") not in s.nodes

    def test_remove_nonexistent_node_is_noop(self):
        """移除不存在的节点不报错，幂等。"""
        s = empty_state()
        s2 = apply(s, IndexedEvent(event=NodeLeft(node_id=NodeId("ghost")), index=1))
        assert len(s2.nodes) == 0


class TestApplyInstanceCreated:
    """InstanceCreated 事件创建模型实例。"""

    def test_creates_instance(self):
        s = empty_state()
        event = InstanceCreated(
            instance_id=InstanceId("i1"),
            model_id="llama-3.1-8b",
            node_ids=[NodeId("n1")],
            sharding="tensor",
        )
        s = apply(s, IndexedEvent(event=event, index=1))
        assert InstanceId("i1") in s.instances

    def test_instance_info_stored(self):
        s = empty_state()
        event = InstanceCreated(
            instance_id=InstanceId("i1"),
            model_id="llama-3.1-8b",
            node_ids=[NodeId("n1"), NodeId("n2")],
            sharding="tensor",
        )
        s = apply(s, IndexedEvent(event=event, index=1))
        inst = s.instances[InstanceId("i1")]
        assert inst.model_id == "llama-3.1-8b"
        assert inst.node_ids == [NodeId("n1"), NodeId("n2")]
        assert inst.sharding == "tensor"


class TestApplyInstanceDeleted:
    """InstanceDeleted 事件删除模型实例。"""

    def test_deletes_instance(self):
        s = empty_state()
        s = apply(s, IndexedEvent(event=InstanceCreated(
            instance_id=InstanceId("i1"), model_id="m", node_ids=[NodeId("n1")], sharding="tensor"
        ), index=1))
        s = apply(s, IndexedEvent(event=InstanceDeleted(instance_id=InstanceId("i1")), index=2))
        assert InstanceId("i1") not in s.instances

    def test_delete_nonexistent_is_noop(self):
        s = empty_state()
        s2 = apply(s, IndexedEvent(event=InstanceDeleted(instance_id=InstanceId("ghost")), index=1))
        assert len(s2.instances) == 0


class TestApplyTaskEvents:
    """Task 相关事件。"""

    def test_task_created(self):
        s = empty_state()
        s = apply(s, IndexedEvent(event=InstanceCreated(
            instance_id=InstanceId("i1"), model_id="m", node_ids=[NodeId("n1")], sharding="tensor"
        ), index=1))
        s = apply(s, IndexedEvent(event=TaskCreated(
            task_id=TaskId("t1"), instance_id=InstanceId("i1"), prompt="hello"
        ), index=2))
        assert TaskId("t1") in s.tasks
        task = s.tasks[TaskId("t1")]
        assert task.status == TaskStatus.PENDING
        assert task.prompt == "hello"

    def test_task_completed(self):
        s = empty_state()
        s = apply(s, IndexedEvent(event=InstanceCreated(
            instance_id=InstanceId("i1"), model_id="m", node_ids=[NodeId("n1")], sharding="tensor"
        ), index=1))
        s = apply(s, IndexedEvent(event=TaskCreated(
            task_id=TaskId("t1"), instance_id=InstanceId("i1"), prompt="hello"
        ), index=2))
        s = apply(s, IndexedEvent(event=TaskCompleted(task_id=TaskId("t1"), output="world"), index=3))
        assert s.tasks[TaskId("t1")].status == TaskStatus.COMPLETED
        assert s.tasks[TaskId("t1")].output == "world"

    def test_task_failed(self):
        s = empty_state()
        s = apply(s, IndexedEvent(event=InstanceCreated(
            instance_id=InstanceId("i1"), model_id="m", node_ids=[NodeId("n1")], sharding="tensor"
        ), index=1))
        s = apply(s, IndexedEvent(event=TaskCreated(
            task_id=TaskId("t1"), instance_id=InstanceId("i1"), prompt="hello"
        ), index=2))
        s = apply(s, IndexedEvent(event=TaskFailed(task_id=TaskId("t1"), error="OOM"), index=3))
        assert s.tasks[TaskId("t1")].status == TaskStatus.FAILED
        assert s.tasks[TaskId("t1")].error == "OOM"

    def test_task_cancelled(self):
        s = empty_state()
        s = apply(s, IndexedEvent(event=InstanceCreated(
            instance_id=InstanceId("i1"), model_id="m", node_ids=[NodeId("n1")], sharding="tensor"
        ), index=1))
        s = apply(s, IndexedEvent(event=TaskCreated(
            task_id=TaskId("t1"), instance_id=InstanceId("i1"), prompt="hello"
        ), index=2))
        s = apply(s, IndexedEvent(event=TaskCancelled(task_id=TaskId("t1")), index=3))
        assert s.tasks[TaskId("t1")].status == TaskStatus.CANCELLED


class TestApplyRunnerStatusUpdated:
    """Runner 状态变更事件。"""

    def test_updates_runner_status(self):
        s = empty_state()
        s = apply(s, IndexedEvent(event=NodeJoined(node_id=NodeId("n1"), ip="10.0.0.1", port=52415), index=1))
        s = apply(s, IndexedEvent(event=RunnerStatusUpdated(node_id=NodeId("n1"), status="ready"), index=2))
        assert s.nodes[NodeId("n1")].runner_status == "ready"

    def test_runner_status_on_nonexistent_node(self):
        """不存在的节点更新 runner 状态应被忽略。"""
        s = empty_state()
        s2 = apply(s, IndexedEvent(event=RunnerStatusUpdated(node_id=NodeId("ghost"), status="ready"), index=1))
        assert len(s2.nodes) == 0


class TestApplyDeterminism:
    """apply 的确定性：相同事件序列产生相同状态。"""

    def test_deterministic(self):
        events = [
            IndexedEvent(event=NodeJoined(node_id=NodeId("n1"), ip="10.0.0.1", port=52415), index=1),
            IndexedEvent(event=NodeJoined(node_id=NodeId("n2"), ip="10.0.0.2", port=52415), index=2),
            IndexedEvent(event=InstanceCreated(
                instance_id=InstanceId("i1"), model_id="m", node_ids=[NodeId("n1")], sharding="tensor"
            ), index=3),
        ]

        s_a = empty_state()
        for ie in events:
            s_a = apply(s_a, ie)

        s_b = empty_state()
        for ie in events:
            s_b = apply(s_b, ie)

        assert s_a == s_b

    def test_event_order_matters(self):
        """不同事件顺序产生不同状态。"""
        s = empty_state()
        s1 = apply(s, IndexedEvent(event=NodeJoined(node_id=NodeId("n1"), ip="10.0.0.1", port=52415), index=1))
        s1 = apply(s1, IndexedEvent(event=NodeLeft(node_id=NodeId("n1")), index=2))

        s2 = apply(s, IndexedEvent(event=NodeLeft(node_id=NodeId("n1")), index=1))
        s2 = apply(s2, IndexedEvent(event=NodeJoined(node_id=NodeId("n1"), ip="10.0.0.1", port=52415), index=2))

        # 先加后删 -> 空；先删后加 -> 有节点
        assert len(s1.nodes) == 0
        assert len(s2.nodes) == 1
