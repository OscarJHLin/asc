"""动态负载均衡器。

根据集群状态动态调整请求分发策略，支持四种调度算法：
- 轮询（Round Robin）：简单均匀分发，适合同构节点
- 最少连接（Least Connections）：优先选择当前活跃请求少的节点
- 算力加权（Compute Score Weighted）：按节点算力评分加权随机选择
- 局部性优先（Locality）：模型已加载的节点优先，减少加载开销

性能优化：
- 多维负载评分：综合 active_requests、VRAM 使用率、网络延迟、队列长度
- 请求级 TTL：防止 Worker 崩溃导致活跃请求计数泄漏
- 全局重平衡：支持多对多迁移计划

设计原则：
- 无状态核心：select() 为纯函数（除 _counter 外），便于测试
- 可观测性：NodeSelection 包含选择原因，便于调试和监控
- 可扩展性：新增策略只需添加 _xxx 方法并在 select() 中注册

线程安全：
    本类包含可变状态（_active_requests、_node_scores），MasterNode 在
    asyncio 事件循环中单线程调用，无需额外锁。若未来改为多线程，
    需对可变状态加锁保护。
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field

from asc.types.state import ClusterState, InstanceState

# 默认请求 TTL（秒）
DEFAULT_REQUEST_TTL = 300.0


@dataclass(frozen=True)
class NodeSelection:
    """节点选择结果。"""

    node_id: str
    weight: float
    reason: str


@dataclass
class NodeLoadMetrics:
    """节点负载多维指标。"""

    active_requests: int = 0
    vram_usage_mb: int = 0
    vram_total_mb: int = 1
    queue_length: int = 0
    avg_latency_ms: float = 0.0
    network_latency_ms: float = 0.0


@dataclass
class _TrackedRequest:
    """跟踪中的请求。"""

    node_id: str
    request_id: str
    start_time: float
    ttl_seconds: float


@dataclass
class LoadBalancer:
    """动态负载均衡器。

    维护各节点的活跃请求计数，支持多种调度策略。
    支持多维负载评分和请求级 TTL 自动清理。
    """

    strategy: str = "round_robin"
    _counter: itertools.count = field(default_factory=itertools.count, repr=False)
    _active_requests: dict[str, int] = field(default_factory=dict, repr=False)
    _node_scores: dict[str, float] = field(default_factory=dict, repr=False)
    _node_metrics: dict[str, NodeLoadMetrics] = field(default_factory=dict, repr=False)
    _tracked_requests: dict[str, _TrackedRequest] = field(default_factory=dict, repr=False)

    def register_node(self, node_id: str, compute_score: float = 1.0) -> None:
        """注册节点到负载均衡器。"""
        if node_id not in self._active_requests:
            self._active_requests[node_id] = 0
        self._node_scores[node_id] = compute_score

    def unregister_node(self, node_id: str) -> None:
        """从负载均衡器注销节点。"""
        self._active_requests.pop(node_id, None)
        self._node_scores.pop(node_id, None)
        self._node_metrics.pop(node_id, None)

    def start_request(self, node_id: str, request_id: str = "") -> None:
        """记录请求开始。"""
        self._active_requests[node_id] = self._active_requests.get(node_id, 0) + 1
        if request_id:
            self._tracked_requests[request_id] = _TrackedRequest(
                node_id=node_id,
                request_id=request_id,
                start_time=time.time(),
                ttl_seconds=DEFAULT_REQUEST_TTL,
            )

    def finish_request(self, node_id: str, request_id: str = "") -> None:
        """记录请求结束。"""
        if node_id in self._active_requests and self._active_requests[node_id] > 0:
            self._active_requests[node_id] -= 1
        if request_id:
            self._tracked_requests.pop(request_id, None)

    def update_metrics(self, node_id: str, metrics: NodeLoadMetrics) -> None:
        """更新节点负载指标。"""
        self._node_metrics[node_id] = metrics

    def cleanup_expired_requests(self) -> int:
        """清理超时请求，返回清理数量。"""
        now = time.time()
        expired = [
            req_id
            for req_id, req in self._tracked_requests.items()
            if now - req.start_time > req.ttl_seconds
        ]
        for req_id in expired:
            req = self._tracked_requests.pop(req_id)
            if req.node_id in self._active_requests and self._active_requests[req.node_id] > 0:
                self._active_requests[req.node_id] -= 1
        return len(expired)

    def _compute_load_score(self, node_id: str) -> float:
        """计算多维负载评分（越低越好）。"""
        m = self._node_metrics.get(node_id)
        compute_score = self._node_scores.get(node_id, 1.0)

        # 基础负载：活跃请求 / 算力
        request_load = self._active_requests.get(node_id, 0) / max(compute_score, 0.001)

        if m is None:
            return request_load

        # 多维负载评分
        vram_load = m.vram_usage_mb / max(m.vram_total_mb, 1)
        queue_load = m.queue_length / max(compute_score * 10, 1)
        latency_penalty = m.network_latency_ms / 100.0

        return (
            request_load * 0.4
            + vram_load * 0.3
            + queue_load * 0.2
            + latency_penalty * 0.1
        )

    def select(
        self,
        candidates: list[str],
        model_id: str | None = None,
        state: ClusterState | None = None,
    ) -> NodeSelection | None:
        """从候选节点中选择一个执行任务的节点。

        选择流程：
        1. 过滤未在负载均衡器注册的节点（确保有健康信息）
        2. 根据 strategy 字段分发到对应算法
        3. 返回包含选择原因的结果，便于日志和监控

        Args:
            candidates: 候选节点 ID 列表（通常来自 ClusterState.nodes）
            model_id: 可选，请求模型 ID（用于局部性优先策略）
            state: 可选，集群状态（用于局部性判断和实例查询）

        Returns:
            NodeSelection 包含选中节点 ID、权重和原因；
            若无可用节点则返回 None（调用者应处理此情况，如排队或报错）

        性能注意：
            除 weighted 策略使用 random.uniform 外，其余策略均为 O(n) 或 O(1)。
            候选节点数量通常 < 100，无需担心性能瓶颈。
        """
        if not candidates:
            return None

        # 过滤掉未注册的节点：未注册意味着未上报容量或已离线
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
        idx = next(self._counter) % len(candidates)
        return NodeSelection(
            node_id=candidates[idx],
            weight=1.0,
            reason="round_robin",
        )

    def _least_connections(self, candidates: list[str]) -> NodeSelection:
        best = min(candidates, key=lambda n: self._compute_load_score(n))
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
    ) -> list[tuple[str, str, int]]:
        """生成负载重平衡计划。

        当节点间负载差异超过阈值时，生成多对多迁移计划。
        使用贪心算法：每次将最高负载节点的任务迁移到最低负载节点。

        Returns:
            [(from_node, to_node, num_tasks), ...]
        """
        if len(self._active_requests) < 2:
            return []

        loads = {
            nid: self._active_requests[nid] / max(self._node_scores.get(nid, 1.0), 0.001)
            for nid in self._active_requests
        }
        if not loads:
            return []

        max_load = max(loads.values())
        min_load = min(loads.values())

        if max_load / max(min_load, 0.001) < threshold_ratio:
            return []

        # 生成多对多迁移计划
        plans: list[tuple[str, str, int]] = []
        # 按负载降序排列
        sorted_nodes = sorted(loads.keys(), key=lambda n: loads[n], reverse=True)

        # 贪心：高负载节点向低负载节点迁移
        high_idx = 0
        low_idx = len(sorted_nodes) - 1

        while high_idx < low_idx:
            high_node = sorted_nodes[high_idx]
            low_node = sorted_nodes[low_idx]
            high_load = loads[high_node]
            low_load = loads[low_node]

            if high_load / max(low_load, 0.001) < threshold_ratio:
                break

            # 计算需要迁移的任务数（使两者接近均值）
            avg_load = (high_load + low_load) / 2
            num_migrate = max(1, int(high_load - avg_load))

            if num_migrate > 0:
                plans.append((high_node, low_node, num_migrate))
                loads[high_node] -= num_migrate
                loads[low_node] += num_migrate

            high_idx += 1
            low_idx -= 1

        return plans
