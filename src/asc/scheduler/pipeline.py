"""Asc Pipeline Parallel 分片规划器。

Pipeline Parallelism（流水线并行）将模型按层纵向切分，每个节点负责
一部分层的计算。前一节点的输出作为后一节点的输入，形成流水线。

与 Tensor Parallelism（张量并行）的区别：
- Pipeline：按层切分，通信量小，但存在气泡（bubble）效率损失
- Tensor：按张量切分，通信量大，但无气泡

本模块支持按 VRAM 或算力比例分配层数，确保：
- 所有阶段层数之和 = 总层数
- 阶段连续覆盖 0 ~ total_layers-1，无遗漏无重叠
- 权重为 0 的节点被排除（无资源或不可用）

使用场景：
    当单节点 VRAM 不足以容纳完整模型时，将模型层分布到多个节点，
    每个节点只保留部分层的权重，通过流水线方式完成前向传播。
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
        node_weights: dict[str, int | float],
    ) -> PipelinePlan:
        """计算 Pipeline 分片方案。

        分配算法：
        1. 过滤权重 <= 0 的节点（无资源或不可用）
        2. 按权重比例计算每个节点应分配的层数
        3. 最后一个节点分配所有剩余层数，确保总和严格等于 total_layers
        4. 每个节点至少分配 1 层（只要还有剩余层数）

        Args:
            total_layers: 模型总层数（如 32、40、80 等）
            node_weights: 各节点权重 {node_id: weight}，权重可为 VRAM 大小或算力评分

        Returns:
            PipelinePlan，可通过 is_valid 属性验证连续性

        边界情况：
            - 无活跃节点：返回空 stages，is_valid == False
            - 总层数 < 节点数：最后一个节点可能分配到负数，需调用方预处理

        示例：
            total_layers=32, node_weights={"A": 16, "B": 16}
            -> A: 0-15 (16层), B: 16-31 (16层)
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
                # 最后一个节点分配所有剩余层数，确保总和严格等于 total_layers
                # 这是避免 round() 累积误差导致层数不匹配的关键
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
