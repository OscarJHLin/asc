"""动态负载均衡器。

根据集群状态动态调整请求分发策略，支持：
- 轮询（Round Robin）
- 最少连接（Least Connections）
- 算力加权（Compute Score Weighted）
- 局部性优先（模型已在节点上）
"""

from __future__ import annotations

from dataclasses import dataclass, field

from asc.types.state import ClusterState, InstanceState


@dataclass(frozen=True)
class NodeSelection:
    """节点选择结果。"""

    node_id: str
    weight: float
    reason: str


@dataclass
class LoadBalancer:
    """动态负载均衡器。

    维护各节点的活跃请求计数，支持多种调度策略。
    """

    strategy: str = "round_robin"
    _counter: int = field(default=0, repr=False)
    _active_requests: dict[str, int] = field(default_factory=dict, repr=False)
    _node_scores: dict[str, float] = field(default_factory=dict, repr=False)

    def register_node(self, node_id: str, compute_score: float = 1.0) -> None:
        """注册节点到负载均衡器。"""
        if node_id not in self._active_requests:
            self._active_requests[node_id] = 0
        self._node_scores[node_id] = compute_score

    def unregister_node(self, node_id: str) -> None:
        """从负载均衡器注销节点。"""
        self._active_requests.pop(node_id, None)
        self._node_scores.pop(node_id, None)

    def start_request(self, node_id: str) -> None:
        """记录请求开始。"""
        self._active_requests[node_id] = self._active_requests.get(node_id, 0) + 1

    def finish_request(self, node_id: str) -> None:
        """记录请求结束。"""
        if node_id in self._active_requests and self._active_requests[node_id] > 0:
            self._active_requests[node_id] -= 1

    def select(
        self,
        candidates: list[str],
        model_id: str | None = None,
        state: ClusterState | None = None,
    ) -> NodeSelection | None:
        """从候选节点中选择一个。

        Args:
            candidates: 候选节点 ID 列表
            model_id: 可选，请求模型 ID（用于局部性优先）
            state: 可选，集群状态（用于局部性判断）

        Returns:
            NodeSelection 或 None（无可用节点）
        """
        if not candidates:
            return None

        # 过滤掉未注册的节点
        available = [n for n in candidates if n in self._active_requests]
        if not available:
            return None

        if self.strategy == "round_robin":
            return self._round_robin(available)
        elif self.strategy == "least_connections":
            return self._least_connections(available)
        elif self.strategy == "weighted":
            return self._weighted(available)
        elif self.strategy == "locality":
            return self._locality(available, model_id, state)
        else:
            return self._round_robin(available)

    def _round_robin(self, candidates: list[str]) -> NodeSelection:
        idx = self._counter % len(candidates)
        self._counter += 1
        return NodeSelection(
            node_id=candidates[idx],
            weight=1.0,
            reason="round_robin",
        )

    def _least_connections(self, candidates: list[str]) -> NodeSelection:
        best = min(candidates, key=lambda n: self._active_requests.get(n, 0))
        return NodeSelection(
            node_id=best,
            weight=1.0,
            reason="least_connections",
        )

    def _weighted(self, candidates: list[str]) -> NodeSelection:
        scores = {n: self._node_scores.get(n, 1.0) for n in candidates}
        total = sum(scores.values())
        if total == 0:
            return self._round_robin(candidates)

        # 加权随机选择
        import random

        r = random.uniform(0, total)
        cumulative = 0.0
        for node_id in candidates:
            cumulative += scores[node_id]
            if r <= cumulative:
                return NodeSelection(
                    node_id=node_id,
                    weight=scores[node_id] / total,
                    reason="weighted",
                )

        return NodeSelection(
            node_id=candidates[-1],
            weight=scores[candidates[-1]] / total,
            reason="weighted",
        )

    def _locality(
        self,
        candidates: list[str],
        model_id: str | None,
        state: ClusterState | None,
    ) -> NodeSelection:
        """局部性优先：模型已在节点上的优先。"""
        if state is not None and model_id is not None:
            for inst in state.instances.values():
                if inst.model_id == model_id and inst.state == InstanceState.RUNNING:
                    for nid in inst.node_ids:
                        nid_str = str(nid)
                        if nid_str in candidates:
                            return NodeSelection(
                                node_id=nid_str,
                                weight=1.0,
                                reason="locality",
                            )

        #  fallback 到最少连接
        return self._least_connections(candidates)

    def rebalance_plan(
        self,
        threshold_ratio: float = 2.0,
    ) -> list[tuple[str, str]]:
        """生成负载重平衡计划。

        当最高负载节点与最低负载节点的比例超过阈值时，
        建议将部分请求从最高负载节点迁移到最低负载节点。

        Returns:
            [(from_node, to_node), ...]
        """
        if len(self._active_requests) < 2:
            return []

        loads = {
            nid: self._active_requests[nid] / max(self._node_scores.get(nid, 1.0), 0.001)
            for nid in self._active_requests
        }
        if not loads:
            return []

        max_node = max(loads, key=loads.get)
        min_node = min(loads, key=loads.get)

        if loads[max_node] / max(loads[min_node], 0.001) < threshold_ratio:
            return []

        return [(max_node, min_node)]
