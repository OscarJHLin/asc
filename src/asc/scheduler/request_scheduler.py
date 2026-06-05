"""请求调度器。

负责将推理请求调度到最优节点，支持多种策略：
- 轮询（Round Robin）
- 最少连接（Least Connections）
- 算力加权（Compute Score Weighted）
- 局部性优先（模型已在节点上）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from asc.scheduler.load_balancer import LoadBalancer
from asc.types.state import ClusterState


@dataclass(frozen=True)
class InferenceRequest:
    """推理请求。"""

    request_id: str
    model_id: str
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.7


@dataclass(frozen=True)
class ScheduleResult:
    """调度结果。"""

    request: InferenceRequest
    selected_node: str
    strategy: str
    estimated_latency_ms: float


class RequestScheduler:
    """请求调度器。

    基于 LoadBalancer 实现请求到节点的映射。
    """

    def __init__(
        self,
        load_balancer: LoadBalancer | None = None,
        latency_estimator: Callable[[str], float] | None = None,
    ) -> None:
        self._lb = load_balancer or LoadBalancer(strategy="round_robin")
        self._latency_fn = latency_estimator or (lambda _n: 0.0)

    def schedule(
        self,
        request: InferenceRequest,
        candidates: list[str],
        state: ClusterState | None = None,
    ) -> ScheduleResult | None:
        """调度请求到最优节点。

        Args:
            request: 推理请求
            candidates: 候选节点列表
            state: 集群状态

        Returns:
            ScheduleResult 或 None
        """
        selection = self._lb.select(
            candidates=candidates,
            model_id=request.model_id,
            state=state,
        )
        if selection is None:
            return None

        self._lb.start_request(selection.node_id)

        return ScheduleResult(
            request=request,
            selected_node=selection.node_id,
            strategy=selection.reason,
            estimated_latency_ms=self._latency_fn(selection.node_id),
        )

    def complete(self, node_id: str) -> None:
        """标记请求完成。"""
        self._lb.finish_request(node_id)

    def set_strategy(self, strategy: str) -> None:
        """切换调度策略。"""
        self._lb.strategy = strategy
