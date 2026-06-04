"""测试智能放置算法。

Placement 负责：
- 选择最优节点组合来放置模型实例
- 基于 VRAM、GPU 类型、网络拓扑等维度决策
- 支持 Tensor Parallel 和 Pipeline Parallel 策略
"""

from asc.scheduler.placement import PlacementEngine, PlacementResult, PlacementStrategy
from asc.scheduler.topology import ClusterTopology, build_topology
from asc.worker.agent import GPUInfo, NodeResources


def _make_resources(vram_free_mb: int, cpu_count: int = 8) -> NodeResources:
    return NodeResources(
        cpu_count=cpu_count,
        cpu_percent=25.0,
        memory_total_mb=32768,
        memory_free_mb=24000,
        gpus=[
            GPUInfo(
                index=0, name="GPU", vram_total_mb=vram_free_mb + 1000, vram_free_mb=vram_free_mb
            )
        ],
    )


class TestPlacementEngine:
    """放置引擎。"""

    def test_single_node_sufficient_vram(self):
        """单节点 VRAM 足够时，放置在本地。"""
        topo = build_topology(
            "master",
            {
                "master": _make_resources(16000),
            },
            {},
        )

        engine = PlacementEngine()
        result = engine.place(
            model_vram_required_mb=8000,
            topology=topo,
            strategy=PlacementStrategy.TENSOR,
        )
        assert result.success
        assert len(result.selected_nodes) == 1
        assert "master" in result.selected_nodes

    def test_single_node_insufficient_vram(self):
        """单节点 VRAM 不足时，放置失败。"""
        topo = build_topology(
            "master",
            {
                "master": _make_resources(4000),
            },
            {},
        )

        engine = PlacementEngine()
        result = engine.place(
            model_vram_required_mb=8000,
            topology=topo,
            strategy=PlacementStrategy.TENSOR,
        )
        assert not result.success

    def test_distributed_two_nodes(self):
        """两节点组合 VRAM 足够。"""
        topo = build_topology(
            "master",
            {
                "master": _make_resources(4000),
                "worker-1": _make_resources(6000),
            },
            {"worker-1": "10.0.0.2:52415"},
        )

        engine = PlacementEngine()
        result = engine.place(
            model_vram_required_mb=8000,
            topology=topo,
            strategy=PlacementStrategy.TENSOR,
        )
        assert result.success
        assert len(result.selected_nodes) == 2

    def test_distributed_three_nodes(self):
        """三节点组合。"""
        topo = build_topology(
            "master",
            {
                "master": _make_resources(3000),
                "worker-1": _make_resources(3000),
                "worker-2": _make_resources(3000),
            },
            {"worker-1": "10.0.0.2:52415", "worker-2": "10.0.0.3:52415"},
        )

        engine = PlacementEngine()
        result = engine.place(
            model_vram_required_mb=8000,
            topology=topo,
            strategy=PlacementStrategy.TENSOR,
        )
        assert result.success
        assert len(result.selected_nodes) == 3

    def test_prefer_fewer_nodes(self):
        """优先选择更少的节点（通信开销更小）。"""
        topo = build_topology(
            "master",
            {
                "master": _make_resources(8000),
                "worker-1": _make_resources(8000),
                "worker-2": _make_resources(3000),
            },
            {"worker-1": "10.0.0.2:52415", "worker-2": "10.0.0.3:52415"},
        )

        engine = PlacementEngine()
        result = engine.place(
            model_vram_required_mb=8000,
            topology=topo,
            strategy=PlacementStrategy.TENSOR,
        )
        assert result.success
        # 单节点即可满足
        assert len(result.selected_nodes) == 1

    def test_no_nodes_at_all(self):
        """无节点可用。"""
        topo = ClusterTopology(master_id="master", nodes={})
        engine = PlacementEngine()
        result = engine.place(
            model_vram_required_mb=8000,
            topology=topo,
            strategy=PlacementStrategy.TENSOR,
        )
        assert not result.success


class TestPlacementResult:
    """放置结果。"""

    def test_success_result(self):
        result = PlacementResult(
            success=True,
            selected_nodes=["master", "worker-1"],
            strategy=PlacementStrategy.TENSOR,
            total_vram_free_mb=20000,
        )
        assert result.success
        assert len(result.selected_nodes) == 2

    def test_failure_result(self):
        result = PlacementResult(
            success=False,
            selected_nodes=[],
            strategy=PlacementStrategy.TENSOR,
            total_vram_free_mb=0,
            reason="Insufficient VRAM",
        )
        assert not result.success
        assert result.reason == "Insufficient VRAM"
