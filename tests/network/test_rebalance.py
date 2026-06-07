"""测试 ASC Cluster Link Protocol - 动态重平衡协议。"""

from asc.network.rebalance import (
    RebalanceAckMessage,
    RebalanceAction,
    RebalanceCompleteMessage,
    RebalancePlan,
    RebalanceRequestMessage,
    RebalanceType,
)


class TestRebalanceType:
    """重平衡类型。"""

    def test_values(self):
        assert RebalanceType.TENSOR_RESPLIT.value == "tensor_resplit"
        assert RebalanceType.PIPELINE_RESHIFT.value == "pipeline_reshift"
        assert RebalanceType.TASK_MIGRATION.value == "task_migration"


class TestRebalanceAction:
    """重平衡动作。"""

    def test_create(self):
        action = RebalanceAction(
            action_type=RebalanceType.TASK_MIGRATION,
            source_node_id="w1",
            target_node_id="w2",
            task_ids=["t1", "t2"],
            params={"reason": "overloaded"},
        )
        assert action.source_node_id == "w1"
        assert action.target_node_id == "w2"
        assert len(action.task_ids) == 2

    def test_to_dict_from_dict(self):
        action = RebalanceAction(
            action_type=RebalanceType.TENSOR_RESPLIT,
            source_node_id="w1",
            target_node_id="w2",
            task_ids=[],
            params={"tensor_split": [0.6, 0.4]},
        )
        d = action.to_dict()
        restored = RebalanceAction.from_dict(d)
        assert restored.action_type == RebalanceType.TENSOR_RESPLIT
        assert restored.params["tensor_split"] == [0.6, 0.4]


class TestRebalancePlan:
    """重平衡计划。"""

    def test_create(self):
        actions = [
            RebalanceAction(
                action_type=RebalanceType.TASK_MIGRATION,
                source_node_id="w1",
                target_node_id="w2",
                task_ids=["t1"],
                params={},
            ),
        ]
        plan = RebalancePlan(
            plan_id="plan-001",
            actions=actions,
            reason="load_imbalance",
        )
        assert plan.plan_id == "plan-001"
        assert len(plan.actions) == 1

    def test_to_dict_from_dict(self):
        actions = [
            RebalanceAction(
                action_type=RebalanceType.TASK_MIGRATION,
                source_node_id="w1",
                target_node_id="w2",
                task_ids=["t1"],
                params={},
            ),
        ]
        plan = RebalancePlan(
            plan_id="plan-001",
            actions=actions,
            reason="load_imbalance",
        )
        d = plan.to_dict()
        restored = RebalancePlan.from_dict(d)
        assert restored.plan_id == "plan-001"
        assert len(restored.actions) == 1


class TestRebalanceRequestMessage:
    """重平衡请求消息。"""

    def test_create(self):
        plan = RebalancePlan(
            plan_id="plan-001",
            actions=[],
            reason="test",
        )
        msg = RebalanceRequestMessage(
            plan=plan,
            initiator_id="master",
        )
        assert msg.initiator_id == "master"
        assert msg.plan.plan_id == "plan-001"

    def test_to_dict_from_dict(self):
        plan = RebalancePlan(
            plan_id="plan-001",
            actions=[
                RebalanceAction(
                    action_type=RebalanceType.PIPELINE_RESHIFT,
                    source_node_id="w1",
                    target_node_id="w2",
                    task_ids=[],
                    params={"layers": [0, 15]},
                ),
            ],
            reason="compute_imbalance",
        )
        msg = RebalanceRequestMessage(plan=plan, initiator_id="master")
        d = msg.to_dict()
        restored = RebalanceRequestMessage.from_dict(d)
        assert restored.plan.actions[0].action_type == RebalanceType.PIPELINE_RESHIFT


class TestRebalanceAckMessage:
    """重平衡确认消息。"""

    def test_create_accept(self):
        msg = RebalanceAckMessage(
            plan_id="plan-001",
            node_id="w1",
            accepted=True,
        )
        assert msg.accepted is True

    def test_create_reject(self):
        msg = RebalanceAckMessage(
            plan_id="plan-001",
            node_id="w2",
            accepted=False,
            reason="insufficient_resources",
        )
        assert msg.accepted is False
        assert msg.reason == "insufficient_resources"

    def test_to_dict_from_dict(self):
        msg = RebalanceAckMessage(
            plan_id="plan-001",
            node_id="w1",
            accepted=True,
        )
        d = msg.to_dict()
        restored = RebalanceAckMessage.from_dict(d)
        assert restored.accepted is True


class TestRebalanceCompleteMessage:
    """重平衡完成消息。"""

    def test_create_success(self):
        msg = RebalanceCompleteMessage(
            plan_id="plan-001",
            node_id="w1",
            success=True,
        )
        assert msg.success is True

    def test_create_failure(self):
        msg = RebalanceCompleteMessage(
            plan_id="plan-001",
            node_id="w1",
            success=False,
            error="migration_failed",
        )
        assert msg.success is False
        assert msg.error == "migration_failed"

    def test_to_dict_from_dict(self):
        msg = RebalanceCompleteMessage(
            plan_id="plan-001",
            node_id="w1",
            success=True,
        )
        d = msg.to_dict()
        restored = RebalanceCompleteMessage.from_dict(d)
        assert restored.success is True
