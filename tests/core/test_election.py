"""测试 Master 选举：Bully 算法。

Bully 算法：节点 ID 最大的成为 Master。
- 节点启动时发起选举
- 收到更高 ID 节点的选举消息时退让
- 选举超时后成为 Master
"""

from asc.core.election import (
    BullyElection,
    ElectionMessage,
    ElectionMessageType,
    ElectionState,
)


class TestElectionMessage:
    """选举消息。"""

    def test_create_election_message(self):
        msg = ElectionMessage(
            type=ElectionMessageType.ELECTION,
            sender_id="node-3",
            election_clock=1,
        )
        assert msg.type == ElectionMessageType.ELECTION
        assert msg.sender_id == "node-3"
        assert msg.election_clock == 1

    def test_coordinator_message(self):
        msg = ElectionMessage(
            type=ElectionMessageType.COORDINATOR,
            sender_id="node-5",
            election_clock=2,
        )
        assert msg.type == ElectionMessageType.COORDINATOR

    def test_alive_message(self):
        msg = ElectionMessage(
            type=ElectionMessageType.ALIVE,
            sender_id="node-4",
            election_clock=1,
        )
        assert msg.type == ElectionMessageType.ALIVE


class TestBullyElectionInitialState:
    """初始状态。"""

    def test_initial_state_is_idle(self):
        election = BullyElection(node_id="node-1", all_node_ids=["node-1", "node-2"])
        assert election.state == ElectionState.IDLE

    def test_initial_master_is_none(self):
        election = BullyElection(node_id="node-1", all_node_ids=["node-1", "node-2"])
        assert election.master_id is None


class TestBullyElectionStart:
    """发起选举。"""

    def test_start_election_transitions_to_electing(self):
        election = BullyElection(node_id="node-1", all_node_ids=["node-1", "node-2"])
        election.start_election()
        assert election.state == ElectionState.ELECTING

    def test_start_election_increments_clock(self):
        election = BullyElection(node_id="node-1", all_node_ids=["node-1", "node-2"])
        election.start_election()
        assert election.election_clock == 1
        election.start_election()
        assert election.election_clock == 2


class TestBullyElectionHighestNode:
    """最高 ID 节点自动成为 Master。"""

    def test_highest_node_becomes_master(self):
        """ID 最大的节点发起选举后，无更高 ID 节点响应，自动成为 Master。"""
        election = BullyElection(
            node_id="node-3",
            all_node_ids=["node-1", "node-2", "node-3"],
        )
        election.start_election()
        # node-3 是最高 ID，没有更高节点需要等待
        assert election.should_become_master()
        election.become_master()
        assert election.state == ElectionState.MASTER
        assert election.master_id == "node-3"

    def test_lower_node_waits_for_higher(self):
        """低 ID 节点发起选举后，需等待高 ID 节点响应。"""
        election = BullyElection(
            node_id="node-1",
            all_node_ids=["node-1", "node-2", "node-3"],
        )
        election.start_election()
        assert not election.should_become_master()
        assert election.state == ElectionState.ELECTING


class TestBullyElectionAliveResponse:
    """收到 ALIVE 消息时退让。"""

    def test_receive_alive_transitions_to_idle(self):
        election = BullyElection(
            node_id="node-1",
            all_node_ids=["node-1", "node-2"],
        )
        election.start_election()
        assert election.state == ElectionState.ELECTING

        # 收到更高 ID 节点的 ALIVE
        msg = ElectionMessage(
            type=ElectionMessageType.ALIVE,
            sender_id="node-2",
            election_clock=election.election_clock,
        )
        election.handle_message(msg)
        assert election.state == ElectionState.IDLE  # 退让


class TestBullyElectionCoordinator:
    """收到 COORDINATOR 消息时确认新 Master。"""

    def test_receive_coordinator_sets_master(self):
        election = BullyElection(
            node_id="node-1",
            all_node_ids=["node-1", "node-2", "node-3"],
        )
        msg = ElectionMessage(
            type=ElectionMessageType.COORDINATOR,
            sender_id="node-3",
            election_clock=1,
        )
        election.handle_message(msg)
        assert election.master_id == "node-3"
        assert election.state == ElectionState.WORKER

    def test_receive_coordinator_as_master(self):
        """已是 Master 的节点收到更高 ID 的 COORDINATOR，退让。"""
        election = BullyElection(
            node_id="node-2",
            all_node_ids=["node-1", "node-2", "node-3"],
        )
        election.start_election()
        # 假设 node-2 暂时成为 master
        election.become_master()
        assert election.state == ElectionState.MASTER

        # 收到 node-3 的 COORDINATOR
        msg = ElectionMessage(
            type=ElectionMessageType.COORDINATOR,
            sender_id="node-3",
            election_clock=election.election_clock + 1,
        )
        election.handle_message(msg)
        assert election.master_id == "node-3"
        assert election.state == ElectionState.WORKER


class TestBullyElectionIsMaster:
    """is_master 便捷方法。"""

    def test_is_master_true(self):
        election = BullyElection(node_id="node-1", all_node_ids=["node-1"])
        election.start_election()
        election.become_master()
        assert election.is_master

    def test_is_master_false(self):
        election = BullyElection(node_id="node-1", all_node_ids=["node-1", "node-2"])
        assert not election.is_master


class TestBullyElectionHigherNodes:
    """获取比自己 ID 高的节点列表。"""

    def test_higher_nodes(self):
        election = BullyElection(
            node_id="node-2",
            all_node_ids=["node-1", "node-2", "node-3", "node-4"],
        )
        higher = election.higher_node_ids
        assert higher == ["node-3", "node-4"]

    def test_no_higher_nodes(self):
        election = BullyElection(
            node_id="node-3",
            all_node_ids=["node-1", "node-2", "node-3"],
        )
        assert election.higher_node_ids == []
