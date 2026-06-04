"""Asc 集群拓扑构建。"""

from __future__ import annotations

from dataclasses import dataclass, field

from asc.worker.agent import NodeResources


@dataclass(frozen=True)
class NodeInfo:
    """拓扑中的节点信息。"""

    node_id: str
    is_local: bool
    address: str | None
    resources: NodeResources


@dataclass(frozen=True)
class ClusterTopology:
    """集群拓扑。"""

    master_id: str
    nodes: dict[str, NodeInfo] = field(default_factory=dict)

    @property
    def total_vram_free_mb(self) -> int:
        return sum(n.resources.total_vram_free_mb for n in self.nodes.values())

    @property
    def worker_nodes(self) -> list[NodeInfo]:
        return [n for n in self.nodes.values() if not n.is_local]


def build_topology(
    master_id: str,
    resources: dict[str, NodeResources],
    addresses: dict[str, str],
) -> ClusterTopology:
    """从资源信息构建集群拓扑。"""
    nodes: dict[str, NodeInfo] = {}
    for node_id, res in resources.items():
        nodes[node_id] = NodeInfo(
            node_id=node_id,
            is_local=(node_id == master_id),
            address=addresses.get(node_id),
            resources=res,
        )

    return ClusterTopology(master_id=master_id, nodes=nodes)
