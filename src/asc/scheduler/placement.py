"""Asc 智能放置算法。

负责选择最优节点组合来放置模型实例。
基于 VRAM、GPU 类型、网络拓扑等维度决策。
优先选择更少的节点（减少通信开销）。
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from itertools import combinations

from asc.scheduler.topology import ClusterTopology


class PlacementStrategy(enum.Enum):
    """放置策略。"""

    TENSOR = "tensor"
    PIPELINE = "pipeline"


@dataclass(frozen=True)
class PlacementResult:
    """放置结果。"""

    success: bool
    selected_nodes: list[str]
    strategy: PlacementStrategy
    total_vram_free_mb: int
    reason: str = ""


class PlacementEngine:
    """智能放置引擎。

    算法：
    1. 从 1 个节点开始尝试，逐步增加
    2. 对每个节点数量，尝试所有组合
    3. 选择满足 VRAM 需求的最少节点组合
    4. 多个组合满足时，选择总 VRAM 最小的（资源利用率最高）
    """

    def place(
        self,
        model_vram_required_mb: int,
        topology: ClusterTopology,
        strategy: PlacementStrategy = PlacementStrategy.TENSOR,
    ) -> PlacementResult:
        """计算最优放置方案。"""
        nodes = list(topology.nodes.values())
        if not nodes:
            return PlacementResult(
                success=False,
                selected_nodes=[],
                strategy=strategy,
                total_vram_free_mb=0,
                reason="No nodes available",
            )

        # 按节点数量从小到大尝试
        for num_nodes in range(1, len(nodes) + 1):
            candidates = []
            for combo in combinations(nodes, num_nodes):
                total_vram = sum(n.resources.total_vram_free_mb for n in combo)
                if total_vram >= model_vram_required_mb:
                    # 算力评分越高越优先（降序），相同算力时选 VRAM 更小的
                    total_compute = sum(n.resources.compute_score for n in combo)
                    candidates.append((combo, total_vram, total_compute))

            if candidates:
                # 优先算力评分高，其次总 VRAM 小（资源利用率最高）
                candidates.sort(key=lambda x: (-x[2], x[1]))
                best_combo, best_vram, _ = candidates[0]
                return PlacementResult(
                    success=True,
                    selected_nodes=[n.node_id for n in best_combo],
                    strategy=strategy,
                    total_vram_free_mb=best_vram,
                )

        return PlacementResult(
            success=False,
            selected_nodes=[],
            strategy=strategy,
            total_vram_free_mb=sum(n.resources.total_vram_free_mb for n in nodes),
            reason="Insufficient VRAM across all nodes",
        )
