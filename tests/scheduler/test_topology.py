"""测试集群拓扑构建模块。"""

from asc.scheduler.topology import ClusterTopology, NodeInfo, build_topology
from asc.worker.agent import NodeResources


class TestNodeInfo:
    """NodeInfo 数据类测试。"""

    def test_fields(self):
        res = NodeResources(
            cpu_count=8,
            cpu_percent=10.0,
            memory_total_mb=32000,
            memory_free_mb=16000,
        )
        node = NodeInfo(
            node_id="n1",
            is_local=True,
            address="127.0.0.1:8080",
            resources=res,
        )
        assert node.node_id == "n1"
        assert node.is_local is True
        assert node.address == "127.0.0.1:8080"
        assert node.resources == res


class TestClusterTopology:
    """ClusterTopology 测试。"""

    def _make_resources(self, vram_free_mb: int = 0) -> NodeResources:
        return NodeResources(
            cpu_count=4,
            cpu_percent=5.0,
            memory_total_mb=16000,
            memory_free_mb=8000,
            gpus=[],
            compute_score=1.0,
        )

    def test_empty_topology(self):
        topo = ClusterTopology(master_id="m1")
        assert topo.master_id == "m1"
        assert topo.nodes == {}
        assert topo.total_vram_free_mb == 0
        assert topo.worker_nodes == []

    def test_total_vram_free_mb(self):
        res1 = NodeResources(
            cpu_count=4,
            cpu_percent=5.0,
            memory_total_mb=16000,
            memory_free_mb=8000,
            gpus=[],
            compute_score=1.0,
        )
        res2 = NodeResources(
            cpu_count=4,
            cpu_percent=5.0,
            memory_total_mb=16000,
            memory_free_mb=8000,
            gpus=[],
            compute_score=1.0,
        )
        topo = ClusterTopology(
            master_id="m1",
            nodes={
                "m1": NodeInfo(
                    node_id="m1",
                    is_local=True,
                    address="127.0.0.1",
                    resources=res1,
                ),
                "w1": NodeInfo(
                    node_id="w1",
                    is_local=False,
                    address="10.0.0.2",
                    resources=res2,
                ),
            },
        )
        assert topo.total_vram_free_mb == 0

    def test_worker_nodes(self):
        res = self._make_resources()
        topo = ClusterTopology(
            master_id="m1",
            nodes={
                "m1": NodeInfo(
                    node_id="m1", is_local=True, address=None, resources=res
                ),
                "w1": NodeInfo(
                    node_id="w1", is_local=False, address="10.0.0.2", resources=res
                ),
                "w2": NodeInfo(
                    node_id="w2", is_local=False, address="10.0.0.3", resources=res
                ),
            },
        )
        workers = topo.worker_nodes
        assert len(workers) == 2
        assert {w.node_id for w in workers} == {"w1", "w2"}

    def test_worker_nodes_empty(self):
        topo = ClusterTopology(
            master_id="m1",
            nodes={
                "m1": NodeInfo(
                    node_id="m1", is_local=True, address=None, resources=self._make_resources()
                ),
            },
        )
        assert topo.worker_nodes == []


class TestBuildTopology:
    """build_topology 函数测试。"""

    def _make_resources(self) -> NodeResources:
        return NodeResources(
            cpu_count=4,
            cpu_percent=5.0,
            memory_total_mb=16000,
            memory_free_mb=8000,
            gpus=[],
            compute_score=1.0,
        )

    def test_basic_build(self):
        resources = {
            "m1": self._make_resources(),
            "w1": self._make_resources(),
        }
        addresses = {
            "m1": "127.0.0.1",
            "w1": "10.0.0.2",
        }
        topo = build_topology("m1", resources, addresses)

        assert topo.master_id == "m1"
        assert set(topo.nodes.keys()) == {"m1", "w1"}
        assert topo.nodes["m1"].is_local is True
        assert topo.nodes["w1"].is_local is False
        assert topo.nodes["m1"].address == "127.0.0.1"
        assert topo.nodes["w1"].address == "10.0.0.2"

    def test_missing_address(self):
        resources = {"m1": self._make_resources()}
        addresses = {}
        topo = build_topology("m1", resources, addresses)
        assert topo.nodes["m1"].address is None

    def test_empty_resources(self):
        topo = build_topology("m1", {}, {})
        assert topo.nodes == {}
        assert topo.total_vram_free_mb == 0

    def test_local_flag_only_master(self):
        resources = {
            "m1": self._make_resources(),
            "w1": self._make_resources(),
            "w2": self._make_resources(),
        }
        addresses = {
            "m1": "127.0.0.1",
            "w1": "10.0.0.2",
            "w2": "10.0.0.3",
        }
        topo = build_topology("m1", resources, addresses)
        assert topo.nodes["m1"].is_local is True
        assert topo.nodes["w1"].is_local is False
        assert topo.nodes["w2"].is_local is False
