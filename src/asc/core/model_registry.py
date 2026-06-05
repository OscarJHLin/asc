"""分布式模型注册表。

实现集群级模型管理：
- 注册/注销节点上的模型
- 查找模型在哪些节点可用
- 选择最佳下载源（最近/最快）
- 存储配额管理
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class ModelLocation:
    """模型在节点上的位置信息。"""

    node_id: str
    path: str
    size_mb: int
    network_latency_ms: float = 0.0


@dataclass
class NodeStorage:
    """节点存储信息。"""

    node_id: str
    total_mb: int
    used_mb: int = 0
    models: dict[str, ModelLocation] = field(default_factory=dict)

    @property
    def free_mb(self) -> int:
        return self.total_mb - self.used_mb

    def can_store(self, size_mb: int) -> bool:
        return self.free_mb >= size_mb


class ModelRegistry:
    """集群级模型注册表。

    维护每个节点上的模型分布，支持智能选源和存储配额管理。
    """

    def __init__(
        self,
        latency_fn: Callable[[str, str], float] | None = None,
    ) -> None:
        self._nodes: dict[str, NodeStorage] = {}
        self._latency_fn = latency_fn or (lambda _a, _b: 0.0)

    def register_node(self, node_id: str, total_mb: int) -> None:
        """注册节点存储。"""
        if node_id not in self._nodes:
            self._nodes[node_id] = NodeStorage(node_id=node_id, total_mb=total_mb)

    def unregister_node(self, node_id: str) -> None:
        """注销节点。"""
        self._nodes.pop(node_id, None)

    def register(
        self,
        node_id: str,
        model_id: str,
        path: str,
        size_mb: int,
    ) -> None:
        """注册节点上的模型。"""
        if node_id not in self._nodes:
            raise ValueError(f"节点 {node_id} 未注册")

        node = self._nodes[node_id]
        latency = self._latency_fn(node_id, model_id)
        location = ModelLocation(
            node_id=node_id,
            path=path,
            size_mb=size_mb,
            network_latency_ms=latency,
        )
        node.models[model_id] = location

    def unregister(self, node_id: str, model_id: str) -> None:
        """注销节点上的模型。"""
        node = self._nodes.get(node_id)
        if node is not None:
            node.models.pop(model_id, None)

    def find_model(self, model_id: str) -> list[ModelLocation]:
        """查找模型在哪些节点上可用。"""
        locations: list[ModelLocation] = []
        for node in self._nodes.values():
            loc = node.models.get(model_id)
            if loc is not None:
                locations.append(loc)
        return locations

    def get_best_source(
        self,
        model_id: str,
        requester: str,
    ) -> ModelLocation | None:
        """选择最佳下载源。

        策略：优先选择延迟最低的节点，其次选择本节点（如果可用）。
        """
        locations = self.find_model(model_id)
        if not locations:
            return None

        # 更新延迟信息
        updated: list[ModelLocation] = []
        for loc in locations:
            latency = self._latency_fn(requester, loc.node_id)
            updated.append(
                ModelLocation(
                    node_id=loc.node_id,
                    path=loc.path,
                    size_mb=loc.size_mb,
                    network_latency_ms=latency,
                )
            )

        # 按延迟排序，优先本地节点
        def sort_key(loc: ModelLocation) -> tuple[int, float]:
            is_local = 0 if loc.node_id == requester else 1
            return (is_local, loc.network_latency_ms)

        updated.sort(key=sort_key)
        return updated[0]

    def list_node_models(self, node_id: str) -> list[ModelLocation]:
        """列出节点上的所有模型。"""
        node = self._nodes.get(node_id)
        if node is None:
            return []
        return list(node.models.values())

    def get_storage(self, node_id: str) -> NodeStorage | None:
        """获取节点存储信息。"""
        return self._nodes.get(node_id)

    def update_storage_usage(self, node_id: str, used_mb: int) -> None:
        """更新节点已用存储。"""
        node = self._nodes.get(node_id)
        if node is not None:
            node.used_mb = used_mb

    def all_models(self) -> dict[str, list[ModelLocation]]:
        """返回所有模型及其分布。"""
        result: dict[str, list[ModelLocation]] = {}
        for node in self._nodes.values():
            for model_id, loc in node.models.items():
                result.setdefault(model_id, []).append(loc)
        return result
