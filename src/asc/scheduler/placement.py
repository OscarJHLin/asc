"""Asc 智能放置算法。

负责选择最优节点组合来放置模型实例。
基于 VRAM、GPU 类型、网络拓扑等维度决策。
优先选择更少的节点（减少通信开销）。
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

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

    算法（贪心 + VRAM 降序排序，O(n log n)）：
    1. 过滤 VRAM 为 0 的不可用节点
    2. 按 VRAM 降序排序（同 VRAM 时按算力密度降序）
       - VRAM 大的节点优先选择，减少所需节点数
    3. 贪心累积：从 VRAM 最大的节点开始，逐个加入直到满足 VRAM 需求
    4. 局部优化：尝试移除多余节点，进一步减少节点数

    相比穷举 combinations 的 O(2^n)，贪心算法在 100+ 节点时仍可毫秒级完成。
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

        # 过滤 VRAM 为 0 的节点
        available = [n for n in nodes if n.resources.total_vram_free_mb > 0]
        if not available:
            return PlacementResult(
                success=False,
                selected_nodes=[],
                strategy=strategy,
                total_vram_free_mb=0,
                reason="No nodes with free VRAM",
            )

        # 按 VRAM 降序排序：优先选择 VRAM 大的节点以减少节点数
        # 同 VRAM 时按算力密度降序，确保同等容量下获得最大算力
        sorted_nodes = sorted(
            available,
            key=lambda n: (
                n.resources.total_vram_free_mb,
                n.resources.compute_score / max(n.resources.total_vram_free_mb, 1),
            ),
            reverse=True,
        )

        # 贪心累积：从 VRAM 最大的节点开始
        selected: list = []
        total_vram = 0

        for node in sorted_nodes:
            selected.append(node)
            total_vram += node.resources.total_vram_free_mb
            if total_vram >= model_vram_required_mb:
                break

        if total_vram < model_vram_required_mb:
            return PlacementResult(
                success=False,
                selected_nodes=[],
                strategy=strategy,
                total_vram_free_mb=sum(n.resources.total_vram_free_mb for n in nodes),
                reason="Insufficient VRAM across all nodes",
            )

        # 局部优化1：尝试用更少节点满足需求
        selected = self._try_reduce_nodes(selected, model_vram_required_mb)

        # 局部优化2：在同等节点数下，尝试替换为 VRAM 更小的节点
        selected = self._try_compact_nodes(selected, model_vram_required_mb, available)

        total_vram = sum(n.resources.total_vram_free_mb for n in selected)
        return PlacementResult(
            success=True,
            selected_nodes=[n.node_id for n in selected],
            strategy=strategy,
            total_vram_free_mb=total_vram,
        )

    def _try_reduce_nodes(
        self,
        selected: list,
        model_vram_required_mb: int,
    ) -> list:
        """尝试减少节点数：从VRAM贡献最小的节点开始移除。"""
        improved = list(selected)
        changed = True
        while changed and len(improved) > 1:
            changed = False
            # 找到移除后仍满足需求的最小VRAM节点
            for i in range(len(improved)):
                test = improved[:i] + improved[i + 1 :]
                test_vram = sum(n.resources.total_vram_free_mb for n in test)
                if test_vram >= model_vram_required_mb:
                    improved = test
                    changed = True
                    break
        return improved

    def _try_compact_nodes(
        self,
        selected: list,
        model_vram_required_mb: int,
        all_available: list,
    ) -> list:
        """在同等节点数下，尝试将已选节点替换为 VRAM 更小的节点。

        目的：当多个节点都能单独满足需求时，选择 VRAM 更小的节点，
        避免浪费大容量节点的资源。
        """
        selected_set = {n.node_id for n in selected}
        candidates = [n for n in all_available if n.node_id not in selected_set]
        # 按 VRAM 升序排列候选节点，优先尝试 VRAM 小的
        candidates.sort(key=lambda n: n.resources.total_vram_free_mb)

        improved = list(selected)
        changed = True
        while changed:
            changed = False
            # 按 VRAM 降序排列已选节点，优先替换 VRAM 大的
            for i in range(len(improved)):
                for cand in candidates:
                    if cand.node_id in {n.node_id for n in improved}:
                        continue
                    # 替换 improved[i] 为 cand
                    test = list(improved)
                    test[i] = cand
                    test_vram = sum(n.resources.total_vram_free_mb for n in test)
                    if test_vram >= model_vram_required_mb and (
                        cand.resources.total_vram_free_mb
                        < improved[i].resources.total_vram_free_mb
                    ):
                        improved = test
                        changed = True
                        break
                if changed:
                    break
        return improved
