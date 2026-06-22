"""测试 DistributedOrchestrator 分布式推理编排器。"""

from unittest.mock import MagicMock, patch

import pytest

from asc.master.orchestrator import DistributedOrchestrator, OrchestratorResult
from asc.scheduler.placement import PlacementEngine, PlacementResult, PlacementStrategy
from asc.scheduler.splitter import TensorSplitResult
from asc.types import InstanceId, NodeId
from asc.types.state import NodeInfo as StateNodeInfo
from asc.worker.agent import GPUInfo, NodeResources


def _make_resources(vram_free_mb: int, compute_score: float = 0.0) -> NodeResources:
    return NodeResources(
        cpu_count=8,
        cpu_percent=25.0,
        memory_total_mb=32768,
        memory_free_mb=24000,
        gpus=[
            GPUInfo(
                index=0,
                name="GPU",
                vram_total_mb=vram_free_mb + 1000,
                vram_free_mb=vram_free_mb,
            )
        ],
        compute_score=compute_score,
    )


class TestOrchestratorInit:
    """初始化。"""

    def test_default_init(self):
        orch = DistributedOrchestrator()
        assert orch._placement is not None
        assert orch._splitter is not None

    def test_custom_init(self):
        placement = PlacementEngine()
        splitter = MagicMock()
        orch = DistributedOrchestrator(placement_engine=placement, split_calculator=splitter)
        assert orch._placement is placement
        assert orch._splitter is splitter


class TestOrchestratorResult:
    """编排结果值对象。"""

    def test_success(self):
        result = OrchestratorResult(
            success=True,
            instance_id=InstanceId("i1"),
            node_ids=[NodeId("n1")],
            rpc_endpoints=["10.0.0.1:50052"],
            tensor_split=[1.0],
        )
        assert result.success is True
        assert result.instance_id == InstanceId("i1")
        assert result.node_ids == [NodeId("n1")]
        assert result.rpc_endpoints == ["10.0.0.1:50052"]
        assert result.tensor_split == [1.0]

    def test_failure(self):
        result = OrchestratorResult(success=False, error="放置失败")
        assert result.success is False
        assert result.error == "放置失败"
        assert result.node_ids == []
        assert result.rpc_endpoints == []
        assert result.tensor_split == []


class TestCreateInstanceLocal:
    """单节点本地推理场景。"""

    @pytest.mark.asyncio
    @patch.object(PlacementEngine, "place")
    @patch.object(DistributedOrchestrator, "_start_llama_server")
    async def test_single_node_local(self, mock_start, mock_place):
        mock_place.return_value = PlacementResult(
            success=True,
            selected_nodes=["master"],
            strategy=PlacementStrategy.TENSOR,
            total_vram_free_mb=10000,
        )

        orch = DistributedOrchestrator()
        nodes = {
            NodeId("master"): StateNodeInfo(node_id=NodeId("master"), ip="127.0.0.1", port=52415),
        }
        resources = {NodeId("master"): _make_resources(10000)}

        result = await orch.create_instance(
            instance_id=InstanceId("i1"),
            model_id="llama-3.1-8b",
            model_path="/models/llama.gguf",
            model_vram_required_mb=8000,
            nodes=nodes,
            node_resources=resources,
        )

        assert result.success is True
        assert result.node_ids == [NodeId("master")]
        assert result.rpc_endpoints == []
        assert result.tensor_split == [1.0]
        mock_start.assert_called_once()


class TestCreateInstanceDistributed:
    """多节点分布式推理场景。"""

    @pytest.mark.asyncio
    @patch.object(PlacementEngine, "place")
    @patch.object(DistributedOrchestrator, "_start_llama_server")
    @patch.object(DistributedOrchestrator, "_request_rpc_start")
    async def test_multi_node_success(self, mock_rpc, mock_start, mock_place):
        mock_place.return_value = PlacementResult(
            success=True,
            selected_nodes=["master", "worker-1"],
            strategy=PlacementStrategy.TENSOR,
            total_vram_free_mb=20000,
        )
        mock_rpc.side_effect = ["127.0.0.1:50052", "10.0.0.2:50052"]

        orch = DistributedOrchestrator()
        nodes = {
            NodeId("master"): StateNodeInfo(
                node_id=NodeId("master"), ip="127.0.0.1", port=52415
            ),
            NodeId("worker-1"): StateNodeInfo(
                node_id=NodeId("worker-1"), ip="10.0.0.2", port=52415
            ),
        }
        resources = {
            NodeId("master"): _make_resources(10000),
            NodeId("worker-1"): _make_resources(10000),
        }

        result = await orch.create_instance(
            instance_id=InstanceId("i1"),
            model_id="llama-3.1-8b",
            model_path="/models/llama.gguf",
            model_vram_required_mb=15000,
            nodes=nodes,
            node_resources=resources,
        )

        assert result.success is True
        assert len(result.node_ids) == 2
        # master 是本地节点，不启动 RPC Server，只有 worker-1 启动
        assert len(result.rpc_endpoints) == 1
        assert result.rpc_endpoints == ["10.0.0.2:50052"]
        assert len(result.tensor_split) == 2
        mock_start.assert_called_once()

    @pytest.mark.asyncio
    @patch.object(PlacementEngine, "place")
    async def test_placement_failure(self, mock_place):
        mock_place.return_value = PlacementResult(
            success=False,
            selected_nodes=[],
            strategy=PlacementStrategy.TENSOR,
            total_vram_free_mb=0,
            reason="Insufficient VRAM",
        )

        orch = DistributedOrchestrator()
        result = await orch.create_instance(
            instance_id=InstanceId("i1"),
            model_id="llama-3.1-8b",
            model_path="/models/llama.gguf",
            model_vram_required_mb=99999,
            nodes={},
            node_resources={},
        )

        assert result.success is False
        assert "Insufficient VRAM" in result.error


class TestDeleteInstance:
    """删除实例。"""

    @pytest.mark.asyncio
    @patch.object(DistributedOrchestrator, "_stop_worker_rpc_server")
    async def test_delete_stops_rpc_servers(self, mock_stop):
        orch = DistributedOrchestrator()
        nodes = {
            NodeId("worker-1"): StateNodeInfo(
                node_id=NodeId("worker-1"), ip="10.0.0.2", port=52415
            ),
        }

        await orch.delete_instance(
            instance_id=InstanceId("i1"),
            node_ids=[NodeId("worker-1")],
            nodes=nodes,
        )

        mock_stop.assert_called_once_with(NodeId("worker-1"), nodes[NodeId("worker-1")])


class TestTensorSplitCalculation:
    """张量分割计算。"""

    def test_calculate_tensor_split(self):
        orch = DistributedOrchestrator()
        nodes = {
            NodeId("master"): StateNodeInfo(
                node_id=NodeId("master"), ip="127.0.0.1", port=52415
            ),
            NodeId("worker-1"): StateNodeInfo(
                node_id=NodeId("worker-1"), ip="10.0.0.2", port=52415
            ),
        }
        resources = {
            NodeId("master"): _make_resources(10000),
            NodeId("worker-1"): _make_resources(10000),
        }

        result = orch._calculate_tensor_split(
            [NodeId("master"), NodeId("worker-1")],
            resources,
            nodes,
        )

        assert isinstance(result, TensorSplitResult)
        assert result.is_distributed is True
        assert len(result.splits) == 2
        assert abs(sum(result.splits) - 1.0) < 0.01
