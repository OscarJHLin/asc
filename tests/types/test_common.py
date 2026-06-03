"""测试基础类型：NodeId, InstanceId, TaskId, SessionId 等强类型。"""

from asc.types.common import (
    EventId,
    InstanceId,
    NodeId,
    SessionId,
    TaskId,
    generate_event_id,
    generate_instance_id,
    generate_node_id,
    generate_task_id,
)


class TestNodeId:
    """NodeId 是节点的唯一标识，基于 NewType 实现类型安全。"""

    def test_generate_creates_unique_ids(self):
        id1 = generate_node_id()
        id2 = generate_node_id()
        assert id1 != id2

    def test_generate_id_is_string(self):
        nid = generate_node_id()
        assert isinstance(nid, str)

    def test_generate_id_not_empty(self):
        nid = generate_node_id()
        assert len(nid) > 0

    def test_node_id_from_string(self):
        nid = NodeId("my-node-001")
        assert nid == "my-node-001"

    def test_node_id_equality(self):
        a = NodeId("node-a")
        b = NodeId("node-a")
        assert a == b

    def test_node_id_inequality(self):
        a = NodeId("node-a")
        b = NodeId("node-b")
        assert a != b

    def test_node_id_hashable(self):
        nid = NodeId("node-1")
        s = {nid}
        assert NodeId("node-1") in s


class TestInstanceId:
    def test_generate_unique(self):
        id1 = generate_instance_id()
        id2 = generate_instance_id()
        assert id1 != id2

    def test_from_string(self):
        iid = InstanceId("inst-001")
        assert iid == "inst-001"

    def test_hashable(self):
        iid = InstanceId("inst-1")
        s = {iid}
        assert InstanceId("inst-1") in s


class TestTaskId:
    def test_generate_unique(self):
        id1 = generate_task_id()
        id2 = generate_task_id()
        assert id1 != id2

    def test_from_string(self):
        tid = TaskId("task-001")
        assert tid == "task-001"


class TestEventId:
    def test_generate_unique(self):
        id1 = generate_event_id()
        id2 = generate_event_id()
        assert id1 != id2

    def test_from_string(self):
        eid = EventId("evt-001")
        assert eid == "evt-001"


class TestSessionId:
    """SessionId 标识一次 Master 选举周期。"""

    def test_create(self):
        sid = SessionId(master_node_id=NodeId("master-1"), election_clock=3)
        assert sid.master_node_id == NodeId("master-1")
        assert sid.election_clock == 3

    def test_equality(self):
        a = SessionId(master_node_id=NodeId("m"), election_clock=1)
        b = SessionId(master_node_id=NodeId("m"), election_clock=1)
        assert a == b

    def test_inequality_different_clock(self):
        a = SessionId(master_node_id=NodeId("m"), election_clock=1)
        b = SessionId(master_node_id=NodeId("m"), election_clock=2)
        assert a != b

    def test_immutable(self):
        sid = SessionId(master_node_id=NodeId("m"), election_clock=1)
        try:
            sid.election_clock = 2  # type: ignore[misc]
            assert False, "Should be immutable"
        except (AttributeError, TypeError):
            pass

    def test_hashable(self):
        sid = SessionId(master_node_id=NodeId("m"), election_clock=1)
        s = {sid}
        assert SessionId(master_node_id=NodeId("m"), election_clock=1) in s
