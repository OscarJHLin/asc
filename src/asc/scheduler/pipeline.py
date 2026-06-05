"""Asc Pipeline Parallel 分片。

将模型按层切分到不同节点，每个节点负责一部分层的计算，
形成流水线。支持按 VRAM 或算力比例分配层数。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineStage:
    """Pipeline 阶段：一个节点负责的一段层。"""

    node_id: str
    start_layer: int
    end_layer: int
    num_layers: int


@dataclass(frozen=True)
class PipelinePlan:
    """Pipeline 分片计划。"""

    stages: list[PipelineStage]
    total_layers: int

    @property
    def is_valid(self) -> bool:
        """检查计划是否有效：阶段连续且覆盖所有层。"""
        if not self.stages:
            return False
        if self.stages[0].start_layer != 0:
            return False
        if self.stages[-1].end_layer != self.total_layers - 1:
            return False
        for i in range(1, len(self.stages)):
            if self.stages[i].start_layer != self.stages[i - 1].end_layer + 1:
                return False
        return True


class PipelinePlanner:
    """Pipeline 分片规划器。

    支持按 VRAM 或算力比例分配层数，确保：
    - 所有阶段层数之和 = 总层数
    - 阶段连续覆盖 0 ~ total_layers-1
    - 权重为 0 的节点被排除
    """

    def plan_by_vram(
        self,
        total_layers: int,
        node_vram_mb: dict[str, int],
    ) -> PipelinePlan:
        """按 VRAM 比例分配层数。"""
        return self._plan(total_layers, node_vram_mb)

    def plan_by_compute(
        self,
        total_layers: int,
        node_compute_scores: dict[str, float],
    ) -> PipelinePlan:
        """按算力评分比例分配层数。"""
        return self._plan(total_layers, node_compute_scores)

    def _plan(
        self,
        total_layers: int,
        node_weights: dict[str, float],
    ) -> PipelinePlan:
        """计算 Pipeline 分片方案。

        Args:
            total_layers: 模型总层数
            node_weights: 各节点权重 {node_id: weight}

        Returns:
            PipelinePlan
        """
        active = {nid: w for nid, w in node_weights.items() if w > 0}
        if not active:
            return PipelinePlan(stages=[], total_layers=total_layers)

        total_weight = sum(active.values())
        stages: list[PipelineStage] = []
        remaining = total_layers
        sorted_nodes = sorted(active.keys())

        for i, node_id in enumerate(sorted_nodes):
            is_last = i == len(sorted_nodes) - 1
            if is_last:
                num = remaining
            else:
                num = max(1, round(total_layers * active[node_id] / total_weight))
                remaining -= num

            start = 0 if not stages else stages[-1].end_layer + 1
            end = start + num - 1
            stages.append(
                PipelineStage(
                    node_id=node_id,
                    start_layer=start,
                    end_layer=end,
                    num_layers=num,
                )
            )

        return PipelinePlan(stages=stages, total_layers=total_layers)
