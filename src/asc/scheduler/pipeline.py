"""Asc Pipeline Parallel 分片。

将模型按层切分到不同节点，每个节点负责一部分层的计算，
形成流水线。按各节点 VRAM 比例分配层数。
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

    按各节点 VRAM 比例分配层数，确保：
    - 所有阶段层数之和 = 总层数
    - 阶段连续覆盖 0 ~ total_layers-1
    - VRAM 为 0 的节点被排除
    """

    def plan(
        self,
        total_layers: int,
        node_vram_mb: dict[str, int],
    ) -> PipelinePlan:
        """计算 Pipeline 分片方案。

        Args:
            total_layers: 模型总层数
            node_vram_mb: 各节点空闲 VRAM {node_id: vram_mb}

        Returns:
            PipelinePlan
        """
        # 过滤 VRAM > 0 的节点
        active = {nid: vram for nid, vram in node_vram_mb.items() if vram > 0}
        if not active:
            return PipelinePlan(stages=[], total_layers=total_layers)

        total_vram = sum(active.values())
        stages: list[PipelineStage] = []
        remaining = total_layers
        sorted_nodes = list(active.keys())

        for i, node_id in enumerate(sorted_nodes):
            is_last = i == len(sorted_nodes) - 1
            if is_last:
                # 最后一个节点分配剩余所有层，避免舍入误差
                num = remaining
            else:
                # 按 VRAM 比例分配，至少 1 层
                num = max(1, round(total_layers * active[node_id] / total_vram))
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
