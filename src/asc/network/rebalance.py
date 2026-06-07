"""ASC Cluster Link Protocol - 动态重平衡协议。

定义运行时算力重分配机制：
1. Master -> Workers: RebalanceRequestMessage (发起重平衡)
2. Workers -> Master: RebalanceAckMessage (确认/拒绝)
3. Workers -> Master: RebalanceCompleteMessage (完成通知)

重平衡类型：
- TENSOR_RESPLIT: 重新计算 tensor-split 比例
- PIPELINE_RESHIFT: 重新分配 pipeline 层范围
- TASK_MIGRATION: 将任务从高负载节点迁移到低负载节点
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class RebalanceType(enum.Enum):
    """重平衡类型。"""

    TENSOR_RESPLIT = "tensor_resplit"
    PIPELINE_RESHIFT = "pipeline_reshift"
    TASK_MIGRATION = "task_migration"


@dataclass(frozen=True)
class RebalanceAction:
    """单个重平衡动作。"""

    action_type: RebalanceType
    source_node_id: str
    target_node_id: str
    task_ids: list[str]
    params: dict

    def to_dict(self) -> dict:
        return {
            "action_type": self.action_type.value,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "task_ids": self.task_ids,
            "params": self.params,
        }

    @classmethod
    def from_dict(cls, d: dict) -> RebalanceAction:
        return cls(
            action_type=RebalanceType(d["action_type"]),
            source_node_id=d["source_node_id"],
            target_node_id=d["target_node_id"],
            task_ids=d["task_ids"],
            params=d["params"],
        )


@dataclass(frozen=True)
class RebalancePlan:
    """重平衡计划。"""

    plan_id: str
    actions: list[RebalanceAction]
    reason: str

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "actions": [a.to_dict() for a in self.actions],
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, d: dict) -> RebalancePlan:
        return cls(
            plan_id=d["plan_id"],
            actions=[RebalanceAction.from_dict(a) for a in d["actions"]],
            reason=d["reason"],
        )


@dataclass(frozen=True)
class RebalanceRequestMessage:
    """重平衡请求消息: Master -> Workers。"""

    plan: RebalancePlan
    initiator_id: str

    def to_dict(self) -> dict:
        return {
            "plan": self.plan.to_dict(),
            "initiator_id": self.initiator_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> RebalanceRequestMessage:
        return cls(
            plan=RebalancePlan.from_dict(d["plan"]),
            initiator_id=d["initiator_id"],
        )


@dataclass(frozen=True)
class RebalanceAckMessage:
    """重平衡确认消息: Worker -> Master。"""

    plan_id: str
    node_id: str
    accepted: bool
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "node_id": self.node_id,
            "accepted": self.accepted,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, d: dict) -> RebalanceAckMessage:
        return cls(
            plan_id=d["plan_id"],
            node_id=d["node_id"],
            accepted=d["accepted"],
            reason=d.get("reason", ""),
        )


@dataclass(frozen=True)
class RebalanceCompleteMessage:
    """重平衡完成消息: Worker -> Master。"""

    plan_id: str
    node_id: str
    success: bool
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "node_id": self.node_id,
            "success": self.success,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: dict) -> RebalanceCompleteMessage:
        return cls(
            plan_id=d["plan_id"],
            node_id=d["node_id"],
            success=d["success"],
            error=d.get("error", ""),
        )
