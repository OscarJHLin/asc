"""测试分布式推理调度器。

分布式推理流程：
1. Master 查询 Worker 资源
2. 计算 tensor-split 方案
3. 启动 Worker rpc-server
4. Master llama-server 连接 Worker rpc-server
5. 推理
6. 清理
"""

from asc.scheduler.splitter import TensorSplitCalculator
from asc.scheduler.topology import build_topology
from asc.worker.agent import NodeResources
from asc.worker.gpu_info import GPUInfo


class TestTensorSplitCalculator:
    """张量分割计算。"""

    def test_single_node(self):
        """单节点无需分割。"""
        result = TensorSplitCalculator.calculate(
            local_vram_free_mb=8000,
            workers_vram_free_mb={},
        )
        assert result.splits == [1.0]
        assert result.rpc_endpoints == []
        assert result.is_distributed is False

    def test_two_nodes(self):
        """两节点按 VRAM 比例分割。"""
        result = TensorSplitCalculator.calculate(
            local_vram_free_mb=8000,
            workers_vram_free_mb={"worker-1": 4000},
        )
        assert len(result.splits) == 2
        assert result.is_distributed is True
        # 8000:4000 = 2:1
        assert result.splits[0] > result.splits[1]

    def test_three_nodes(self):
        """三节点分割。"""
        result = TensorSplitCalculator.calculate(
            local_vram_free_mb=8000,
            workers_vram_free_mb={"w1": 6000, "w2": 2000},
        )
        assert len(result.splits) == 3
        # 8000:6000:2000
        assert result.splits[0] > result.splits[1] > result.splits[2]

    def test_zero_vram_worker_excluded(self):
        """VRAM 为 0 的 Worker 被排除。"""
        result = TensorSplitCalculator.calculate(
            local_vram_free_mb=8000,
            workers_vram_free_mb={"w1": 0, "w2": 4000},
        )
        assert len(result.splits) == 2  # local + w2
        assert result.is_distributed is True

    def test_format_for_llama(self):
        """格式化为 llama-cli --tensor-split 参数。"""
        result = TensorSplitCalculator.calculate(
            local_vram_free_mb=8000,
            workers_vram_free_mb={"w1": 4000},
        )
        formatted = result.format_tensor_split()
        # 应该是逗号分隔的浮点数
        parts = formatted.split(",")
        assert len(parts) == 2

    def test_rpc_endpoints(self):
        """RPC 端点列表。"""
        result = TensorSplitCalculator.calculate(
            local_vram_free_mb=8000,
            workers_vram_free_mb={
                "w1": 4000,
                "w2": 2000,
            },
            worker_addresses={"w1": "10.0.0.2:50052", "w2": "10.0.0.3:50052"},
        )
        assert len(result.rpc_endpoints) == 2
        assert "10.0.0.2:50052" in result.rpc_endpoints

    def test_local_weight_boost(self):
        """本地权重加成（本地通信更快）。"""
        result_normal = TensorSplitCalculator.calculate(
            local_vram_free_mb=8000,
            workers_vram_free_mb={"w1": 8000},
        )
        result_boosted = TensorSplitCalculator.calculate(
            local_vram_free_mb=8000,
            workers_vram_free_mb={"w1": 8000},
            local_weight=1.5,
        )
        # 加成后本地占比应更大
        assert result_boosted.splits[0] > result_normal.splits[0]


class TestClusterTopology:
    """集群拓扑。"""

    def test_build_from_resources(self):
        """从资源信息构建拓扑。"""
        resources = {
            "master": NodeResources(
                cpu_count=8,
                cpu_percent=25.0,
                memory_total_mb=32768,
                memory_free_mb=24000,
                gpus=[GPUInfo(index=0, name="RTX 4090", vram_total_mb=24564, vram_free_mb=20000)],
            ),
            "worker-1": NodeResources(
                cpu_count=4,
                cpu_percent=10.0,
                memory_total_mb=16384,
                memory_free_mb=12000,
                gpus=[GPUInfo(index=0, name="RTX 3060", vram_total_mb=12288, vram_free_mb=10000)],
            ),
        }
        addresses = {"worker-1": "10.0.0.2:52415"}

        topo = build_topology("master", resources, addresses)
        assert len(topo.nodes) == 2
        assert topo.total_vram_free_mb == 30000
        assert topo.nodes["master"].is_local is True
        assert topo.nodes["worker-1"].is_local is False

    def test_total_vram_no_gpus(self):
        """无 GPU 节点的 VRAM 为 0。"""
        resources = {
            "master": NodeResources(
                cpu_count=8,
                cpu_percent=25.0,
                memory_total_mb=32768,
                memory_free_mb=24000,
                gpus=[],
            ),
        }
        topo = build_topology("master", resources, {})
        assert topo.total_vram_free_mb == 0
