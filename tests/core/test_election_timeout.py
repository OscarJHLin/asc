"""测试选举超时机制。"""

import time
from unittest.mock import patch

from asc.core.election import BullyElection, ElectionState


class TestCheckTimeoutNotElecting:
    """check_timeout 在非 ELECTING 状态下返回 False。"""

    def test_idle_state_returns_false(self):
        election = BullyElection(node_id="node-1", all_node_ids=["node-1"])
        assert election.check_timeout() is False

    def test_master_state_returns_false(self):
        election = BullyElection(node_id="node-1", all_node_ids=["node-1"])
        election.start_election()
        election.become_master()
        assert election.check_timeout() is False

    def test_worker_state_returns_false(self):
        election = BullyElection(node_id="node-1", all_node_ids=["node-1", "node-2"])
        election.start_election()
        election._state = ElectionState.WORKER
        assert election.check_timeout() is False


class TestCheckTimeoutAfterTimeout:
    """check_timeout 在超时后返回 True。"""

    def test_returns_true_after_timeout(self):
        election = BullyElection(
            node_id="node-1",
            all_node_ids=["node-1", "node-2"],
            election_timeout_sec=1.0,
        )
        election.start_election()

        # 模拟时间已经过了超时时间
        with patch("asc.core.election.time.time", return_value=time.time() + 2.0):
            assert election.check_timeout() is True

    def test_returns_false_before_timeout(self):
        election = BullyElection(
            node_id="node-1",
            all_node_ids=["node-1", "node-2"],
            election_timeout_sec=10.0,
        )
        election.start_election()
        assert election.check_timeout() is False

    def test_returns_false_when_start_time_not_set(self):
        election = BullyElection(
            node_id="node-1",
            all_node_ids=["node-1", "node-2"],
        )
        # 直接设置 ELECTING 状态但不调用 start_election
        election._state = ElectionState.ELECTING
        # _election_start_time 仍为 0.0
        assert election.check_timeout() is False


class TestHandleTimeoutNoHigherNodes:
    """handle_timeout 在没有更高 ID 节点时使节点成为 Master。"""

    def test_becomes_master_when_no_higher_nodes(self):
        election = BullyElection(
            node_id="node-3",
            all_node_ids=["node-1", "node-2", "node-3"],
            election_timeout_sec=1.0,
        )
        election.start_election()

        with patch("asc.core.election.time.time", return_value=time.time() + 2.0):
            election.handle_timeout()

        assert election.state == ElectionState.MASTER
        assert election.master_id == "node-3"


class TestHandleTimeoutHigherNodesExist:
    """handle_timeout 在有更高 ID 节点但未响应时重新发起选举。"""

    def test_restarts_election_when_higher_nodes_exist(self):
        election = BullyElection(
            node_id="node-1",
            all_node_ids=["node-1", "node-2", "node-3"],
            election_timeout_sec=1.0,
        )
        election.start_election()
        clock_before = election.election_clock

        with patch("asc.core.election.time.time", return_value=time.time() + 2.0):
            election.handle_timeout()

        # 选举时钟递增，说明重新发起了选举
        assert election.election_clock == clock_before + 1
        # 仍处于 ELECTING 状态
        assert election.state == ElectionState.ELECTING


class TestHandleTimeoutNotTimedOut:
    """handle_timeout 在未超时时不做任何操作。"""

    def test_no_action_when_not_timed_out(self):
        election = BullyElection(
            node_id="node-1",
            all_node_ids=["node-1", "node-2"],
            election_timeout_sec=10.0,
        )
        election.start_election()
        state_before = election.state
        election.handle_timeout()
        assert election.state == state_before


class TestElectionTimeoutConfigurable:
    """election_timeout_sec 可配置。"""

    def test_default_timeout(self):
        election = BullyElection(node_id="node-1", all_node_ids=["node-1"])
        assert election.election_timeout_sec == 5.0

    def test_custom_timeout(self):
        election = BullyElection(
            node_id="node-1",
            all_node_ids=["node-1"],
            election_timeout_sec=3.0,
        )
        assert election.election_timeout_sec == 3.0

    def test_custom_timeout_affects_check(self):
        election_short = BullyElection(
            node_id="node-1",
            all_node_ids=["node-1", "node-2"],
            election_timeout_sec=1.0,
        )
        election_long = BullyElection(
            node_id="node-1",
            all_node_ids=["node-1", "node-2"],
            election_timeout_sec=100.0,
        )

        election_short.start_election()
        election_long.start_election()

        # 2 秒后，短超时的应该超时，长超时的不应该
        with patch("asc.core.election.time.time", return_value=time.time() + 2.0):
            assert election_short.check_timeout() is True
            assert election_long.check_timeout() is False
