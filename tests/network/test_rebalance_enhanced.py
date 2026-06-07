"""增强测试 ASC Cluster Link Protocol - 动态重平衡协议。

覆盖边界条件、冻结验证、向后兼容性及完整生命周期往返。
"""

from __future__ import annotations

import pytest

from asc.network.rebalance import (
    RebalanceAckMessage,
    RebalanceAction,
    RebalanceCompleteMessage,
    RebalancePlan,
    RebalanceRequestMessage,
    RebalanceType,
)

# ---------------------------------------------------------------------------
# RebalanceAction 边界条件
# ---------------------------------------------------------------------------


class TestRebalanceActionEdgeCases:
    """RebalanceAction 边界条件测试。"""

    def test_empty_task_ids(self):
        """空 task_ids 列表应正常创建。"""
        action = RebalanceAction(
            action_type=RebalanceType.TASK_MIGRATION,
            source_node_id="w1",
            target_node_id="w2",
            task_ids=[],
            params={},
        )
        assert action.task_ids == []

    def test_many_task_ids(self):
        """包含大量 task_id 的列表应正常创建。"""
        ids = [f"task-{i}" for i in range(1000)]
        action = RebalanceAction(
            action_type=RebalanceType.TASK_MIGRATION,
            source_node_id="w1",
            target_node_id="w2",
            task_ids=ids,
            params={},
        )
        assert len(action.task_ids) == 1000
        assert action.task_ids[0] == "task-0"
        assert action.task_ids[-1] == "task-999"

    def test_empty_params_dict(self):
        """空 params 字典应正常创建。"""
        action = RebalanceAction(
            action_type=RebalanceType.TENSOR_RESPLIT,
            source_node_id="w1",
            target_node_id="w2",
            task_ids=["t1"],
            params={},
        )
        assert action.params == {}

    def test_complex_nested_params(self):
        """复杂嵌套 params 字典应正常序列化/反序列化。"""
        params = {
            "tensor_split": [0.6, 0.4],
            "config": {
                "layer_ranges": {"start": 0, "end": 15},
                "options": {"overlap": False, "priority": 1},
            },
            "tags": ["gpu", "high-mem"],
        }
        action = RebalanceAction(
            action_type=RebalanceType.PIPELINE_RESHIFT,
            source_node_id="w1",
            target_node_id="w2",
            task_ids=["t1"],
            params=params,
        )
        d = action.to_dict()
        restored = RebalanceAction.from_dict(d)
        assert restored.params["config"]["layer_ranges"]["end"] == 15
        assert restored.params["tags"] == ["gpu", "high-mem"]

    def test_self_rebalance_source_equals_target(self):
        """source_node_id == target_node_id（自重平衡）应正常创建。"""
        action = RebalanceAction(
            action_type=RebalanceType.TASK_MIGRATION,
            source_node_id="w1",
            target_node_id="w1",
            task_ids=["t1"],
            params={"reason": "internal_optimization"},
        )
        assert action.source_node_id == action.target_node_id


# ---------------------------------------------------------------------------
# RebalancePlan 边界条件
# ---------------------------------------------------------------------------


class TestRebalancePlanEdgeCases:
    """RebalancePlan 边界条件测试。"""

    def test_empty_actions_list(self):
        """空 actions 列表应正常创建。"""
        plan = RebalancePlan(
            plan_id="plan-empty",
            actions=[],
            reason="no_action_needed",
        )
        assert plan.actions == []

    def test_multiple_actions_with_different_types(self):
        """包含不同 RebalanceType 的多个 action 应正确保留类型。"""
        actions = [
            RebalanceAction(
                action_type=RebalanceType.TENSOR_RESPLIT,
                source_node_id="w1",
                target_node_id="w2",
                task_ids=["t1"],
                params={"split": [0.5, 0.5]},
            ),
            RebalanceAction(
                action_type=RebalanceType.PIPELINE_RESHIFT,
                source_node_id="w2",
                target_node_id="w3",
                task_ids=["t2"],
                params={"layers": [0, 10]},
            ),
            RebalanceAction(
                action_type=RebalanceType.TASK_MIGRATION,
                source_node_id="w3",
                target_node_id="w1",
                task_ids=["t3", "t4"],
                params={"priority": "high"},
            ),
        ]
        plan = RebalancePlan(
            plan_id="plan-mixed",
            actions=actions,
            reason="comprehensive_rebalance",
        )
        assert len(plan.actions) == 3
        assert plan.actions[0].action_type == RebalanceType.TENSOR_RESPLIT
        assert plan.actions[1].action_type == RebalanceType.PIPELINE_RESHIFT
        assert plan.actions[2].action_type == RebalanceType.TASK_MIGRATION

    def test_long_reason_string(self):
        """很长的 reason 字符串应正常保存。"""
        long_reason = "负载不均衡导致性能下降" * 500
        plan = RebalancePlan(
            plan_id="plan-long",
            actions=[],
            reason=long_reason,
        )
        assert len(plan.reason) == len(long_reason)

    def test_to_dict_from_dict_roundtrip_with_mixed_actions(self):
        """多种类型 action 的 plan 序列化/反序列化往返应一致。"""
        actions = [
            RebalanceAction(
                action_type=RebalanceType.TENSOR_RESPLIT,
                source_node_id="w1",
                target_node_id="w2",
                task_ids=["t1"],
                params={"split": [0.6, 0.4]},
            ),
            RebalanceAction(
                action_type=RebalanceType.PIPELINE_RESHIFT,
                source_node_id="w2",
                target_node_id="w3",
                task_ids=["t2"],
                params={"layers": [0, 15]},
            ),
            RebalanceAction(
                action_type=RebalanceType.TASK_MIGRATION,
                source_node_id="w3",
                target_node_id="w1",
                task_ids=["t3"],
                params={},
            ),
        ]
        plan = RebalancePlan(
            plan_id="plan-rt",
            actions=actions,
            reason="roundtrip_test",
        )
        d = plan.to_dict()
        restored = RebalancePlan.from_dict(d)
        assert restored.plan_id == "plan-rt"
        assert len(restored.actions) == 3
        assert restored.actions[0].action_type == RebalanceType.TENSOR_RESPLIT
        assert restored.actions[1].action_type == RebalanceType.PIPELINE_RESHIFT
        assert restored.actions[2].action_type == RebalanceType.TASK_MIGRATION


# ---------------------------------------------------------------------------
# RebalanceAckMessage 边界条件
# ---------------------------------------------------------------------------


class TestRebalanceAckMessageEdgeCases:
    """RebalanceAckMessage 边界条件测试。"""

    def test_accepted_true_default_empty_reason(self):
        """accepted=True 时 reason 默认为空字符串。"""
        msg = RebalanceAckMessage(
            plan_id="plan-001",
            node_id="w1",
            accepted=True,
        )
        assert msg.reason == ""

    def test_accepted_false_with_detailed_reason(self):
        """accepted=False 时附带详细拒绝原因。"""
        detailed = "GPU显存不足，无法承接额外的tensor分片，当前已使用95%显存"
        msg = RebalanceAckMessage(
            plan_id="plan-001",
            node_id="w2",
            accepted=False,
            reason=detailed,
        )
        assert msg.accepted is False
        assert msg.reason == detailed


# ---------------------------------------------------------------------------
# RebalanceCompleteMessage 边界条件
# ---------------------------------------------------------------------------


class TestRebalanceCompleteMessageEdgeCases:
    """RebalanceCompleteMessage 边界条件测试。"""

    def test_success_true_default_empty_error(self):
        """success=True 时 error 默认为空字符串。"""
        msg = RebalanceCompleteMessage(
            plan_id="plan-001",
            node_id="w1",
            success=True,
        )
        assert msg.error == ""

    def test_success_false_with_long_error(self):
        """success=False 时附带长错误信息。"""
        long_error = "迁移失败：源节点w1在传输第3个分片时发生网络中断，" * 100
        msg = RebalanceCompleteMessage(
            plan_id="plan-001",
            node_id="w1",
            success=False,
            error=long_error,
        )
        assert msg.success is False
        assert len(msg.error) == len(long_error)


# ---------------------------------------------------------------------------
# 完整重平衡生命周期往返
# ---------------------------------------------------------------------------


class TestRebalanceLifecycleRoundtrip:
    """完整重平衡生命周期往返测试。"""

    def test_full_lifecycle(self):
        """创建包含所有3种RebalanceType的计划 -> 请求 -> 确认 -> 完成。"""
        # 1. 创建包含3种类型的 action
        actions = [
            RebalanceAction(
                action_type=RebalanceType.TENSOR_RESPLIT,
                source_node_id="w1",
                target_node_id="w2",
                task_ids=["t1"],
                params={"split": [0.6, 0.4]},
            ),
            RebalanceAction(
                action_type=RebalanceType.PIPELINE_RESHIFT,
                source_node_id="w2",
                target_node_id="w3",
                task_ids=["t2"],
                params={"layers": [0, 15]},
            ),
            RebalanceAction(
                action_type=RebalanceType.TASK_MIGRATION,
                source_node_id="w3",
                target_node_id="w1",
                task_ids=["t3", "t4"],
                params={},
            ),
        ]
        plan = RebalancePlan(
            plan_id="plan-lifecycle",
            actions=actions,
            reason="load_imbalance",
        )

        # 2. Master 发送请求
        request = RebalanceRequestMessage(plan=plan, initiator_id="master")
        req_dict = request.to_dict()
        restored_req = RebalanceRequestMessage.from_dict(req_dict)
        assert restored_req.initiator_id == "master"
        assert len(restored_req.plan.actions) == 3

        # 3. Worker 确认
        ack = RebalanceAckMessage(
            plan_id=plan.plan_id,
            node_id="w1",
            accepted=True,
        )
        ack_dict = ack.to_dict()
        restored_ack = RebalanceAckMessage.from_dict(ack_dict)
        assert restored_ack.accepted is True
        assert restored_ack.reason == ""

        # 4. Worker 拒绝
        reject = RebalanceAckMessage(
            plan_id=plan.plan_id,
            node_id="w2",
            accepted=False,
            reason="资源不足",
        )
        reject_dict = reject.to_dict()
        restored_reject = RebalanceAckMessage.from_dict(reject_dict)
        assert restored_reject.accepted is False
        assert restored_reject.reason == "资源不足"

        # 5. Worker 完成通知
        complete = RebalanceCompleteMessage(
            plan_id=plan.plan_id,
            node_id="w1",
            success=True,
        )
        complete_dict = complete.to_dict()
        restored_complete = RebalanceCompleteMessage.from_dict(complete_dict)
        assert restored_complete.success is True
        assert restored_complete.error == ""

        # 6. Worker 失败通知
        fail = RebalanceCompleteMessage(
            plan_id=plan.plan_id,
            node_id="w3",
            success=False,
            error="迁移中断",
        )
        fail_dict = fail.to_dict()
        restored_fail = RebalanceCompleteMessage.from_dict(fail_dict)
        assert restored_fail.success is False
        assert restored_fail.error == "迁移中断"


# ---------------------------------------------------------------------------
# Frozen 验证
# ---------------------------------------------------------------------------


class TestFrozenVerification:
    """所有 frozen dataclass 不可变验证。"""

    def test_rebalance_action_frozen(self):
        """RebalanceAction 不可修改。"""
        action = RebalanceAction(
            action_type=RebalanceType.TASK_MIGRATION,
            source_node_id="w1",
            target_node_id="w2",
            task_ids=["t1"],
            params={},
        )
        with pytest.raises(AttributeError):
            action.source_node_id = "w3"  # type: ignore[misc]

    def test_rebalance_plan_frozen(self):
        """RebalancePlan 不可修改。"""
        plan = RebalancePlan(
            plan_id="plan-001",
            actions=[],
            reason="test",
        )
        with pytest.raises(AttributeError):
            plan.plan_id = "plan-002"  # type: ignore[misc]

    def test_rebalance_ack_message_frozen(self):
        """RebalanceAckMessage 不可修改。"""
        msg = RebalanceAckMessage(
            plan_id="plan-001",
            node_id="w1",
            accepted=True,
        )
        with pytest.raises(AttributeError):
            msg.accepted = False  # type: ignore[misc]

    def test_rebalance_complete_message_frozen(self):
        """RebalanceCompleteMessage 不可修改。"""
        msg = RebalanceCompleteMessage(
            plan_id="plan-001",
            node_id="w1",
            success=True,
        )
        with pytest.raises(AttributeError):
            msg.success = False  # type: ignore[misc]

    def test_rebalance_request_message_frozen(self):
        """RebalanceRequestMessage 不可修改。"""
        plan = RebalancePlan(
            plan_id="plan-001",
            actions=[],
            reason="test",
        )
        msg = RebalanceRequestMessage(plan=plan, initiator_id="master")
        with pytest.raises(AttributeError):
            msg.initiator_id = "other"  # type: ignore[misc]

    def test_rebalance_action_frozen_task_ids_mutation(self):
        """RebalanceAction 的 task_ids 列表本身可变，但属性不可重新赋值。"""
        action = RebalanceAction(
            action_type=RebalanceType.TASK_MIGRATION,
            source_node_id="w1",
            target_node_id="w2",
            task_ids=["t1"],
            params={},
        )
        with pytest.raises(AttributeError):
            action.task_ids = ["t2"]  # type: ignore[misc]


# ---------------------------------------------------------------------------
# from_dict 向后兼容性
# ---------------------------------------------------------------------------


class TestFromDictBackwardCompatibility:
    """from_dict 缺失可选字段时的向后兼容性测试。"""

    def test_ack_missing_reason_defaults_empty(self):
        """RebalanceAckMessage.from_dict 缺少 reason 字段时默认空字符串。"""
        d = {
            "plan_id": "plan-001",
            "node_id": "w1",
            "accepted": True,
        }
        msg = RebalanceAckMessage.from_dict(d)
        assert msg.reason == ""

    def test_complete_missing_error_defaults_empty(self):
        """RebalanceCompleteMessage.from_dict 缺少 error 字段时默认空字符串。"""
        d = {
            "plan_id": "plan-001",
            "node_id": "w1",
            "success": True,
        }
        msg = RebalanceCompleteMessage.from_dict(d)
        assert msg.error == ""

    def test_ack_with_reason_present(self):
        """RebalanceAckMessage.from_dict 包含 reason 字段时正常读取。"""
        d = {
            "plan_id": "plan-001",
            "node_id": "w1",
            "accepted": False,
            "reason": "资源不足",
        }
        msg = RebalanceAckMessage.from_dict(d)
        assert msg.reason == "资源不足"

    def test_complete_with_error_present(self):
        """RebalanceCompleteMessage.from_dict 包含 error 字段时正常读取。"""
        d = {
            "plan_id": "plan-001",
            "node_id": "w1",
            "success": False,
            "error": "迁移失败",
        }
        msg = RebalanceCompleteMessage.from_dict(d)
        assert msg.error == "迁移失败"
