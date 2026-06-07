"""分布式编排器增强测试。

覆盖审计中发现的关键缺口：
- _handle_rpc_failures 处理部分 RPC 失败
- _retry_with_fewer_nodes 最大深度限制
- _request_rpc_start 各种 httpx 异常
- _stop_worker_rpc_server 各种 httpx 异常
- _create_local_instance 失败
- _calculate_tensor_split 无本地节点
- _build_topology 空节点
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
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


def _make_nodes_and_resources() -> tuple[dict, dict]:
    """创建测试用的节点和资源。"""
    nodes = {
        NodeId("master"): StateNodeInfo(node_id=NodeId("master"), ip="127.0.0.1", port=52415),
        NodeId("worker-1"): StateNodeInfo(node_id=NodeId("worker-1"), ip="10.0.0.2", port=52415),
        NodeId("worker-2"): StateNodeInfo(node_id=NodeId("worker-2"), ip="10.0.0.3", port=52415),
    }
    resources = {
        NodeId("master"): _make_resources(10000),
        NodeId("worker-1"): _make_resources(10000),
        NodeId("worker-2"): _make_resources(10000),
    }
    return nodes, resources


class TestHandleRpcFailures:
    """_handle_rpc_failures 处理部分 RPC 失败。"""

    @pytest.mark.asyncio
    async def test_all_rpc_succeed(self):
        """所有 RPC 成功时返回 None。"""
        orch = DistributedOrchestrator()
        nodes, resources = _make_nodes_and_resources()

        result = await orch._handle_rpc_failures(
            worker_node_ids=[NodeId("worker-1"), NodeId("worker-2")],
            selected_node_ids=[NodeId("master"), NodeId("worker-1"), NodeId("worker-2")],
            rpc_endpoints=["10.0.0.2:50052", "10.0.0.3:50052"],
            nodes=nodes,
            instance_id=InstanceId("i1"),
            model_id="m",
            model_path="/m.gguf",
            model_vram_required_mb=10000,
            node_resources=resources,
            strategy=PlacementStrategy.TENSOR,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_partial_rpc_failure_triggers_retry(self):
        """部分 RPC 失败应触发重试。"""
        orch = DistributedOrchestrator()
        nodes, resources = _make_nodes_and_resources()

        with patch.object(PlacementEngine, "place") as mock_place, \
             patch.object(DistributedOrchestrator, "_start_llama_server"):
            mock_place.return_value = PlacementResult(
                success=True,
                selected_nodes=["master"],
                strategy=PlacementStrategy.TENSOR,
                total_vram_free_mb=10000,
            )

            result = await orch._handle_rpc_failures(
                worker_node_ids=[NodeId("worker-1"), NodeId("worker-2")],
                selected_node_ids=[NodeId("master"), NodeId("worker-1"), NodeId("worker-2")],
                rpc_endpoints=["10.0.0.2:50052"],  # worker-2 失败
                nodes=nodes,
                instance_id=InstanceId("i1"),
                model_id="m",
                model_path="/m.gguf",
                model_vram_required_mb=10000,
                node_resources=resources,
                strategy=PlacementStrategy.TENSOR,
            )

            # 应返回重试结果（成功或失败）
            assert result is not None
            assert isinstance(result, OrchestratorResult)

    @pytest.mark.asyncio
    async def test_no_worker_nodes(self):
        """无 Worker 节点时返回 None。"""
        orch = DistributedOrchestrator()
        nodes, resources = _make_nodes_and_resources()

        result = await orch._handle_rpc_failures(
            worker_node_ids=[],
            selected_node_ids=[NodeId("master")],
            rpc_endpoints=[],
            nodes=nodes,
            instance_id=InstanceId("i1"),
            model_id="m",
            model_path="/m.gguf",
            model_vram_required_mb=10000,
            node_resources=resources,
            strategy=PlacementStrategy.TENSOR,
        )

        assert result is None


class TestRetryWithFewerNodes:
    """_retry_with_fewer_nodes 最大深度限制测试。"""

    @pytest.mark.asyncio
    async def test_max_depth_limit(self):
        """重试深度超过 3 时应返回失败。"""
        orch = DistributedOrchestrator()
        nodes, resources = _make_nodes_and_resources()

        result = await orch._retry_with_fewer_nodes(
            instance_id=InstanceId("i1"),
            model_id="m",
            model_path="/m.gguf",
            model_vram_required_mb=10000,
            nodes=nodes,
            node_resources=resources,
            successful_nodes=[NodeId("master"), NodeId("worker-1")],
            strategy=PlacementStrategy.TENSOR,
            _retry_depth=3,
        )

        assert result.success is False
        assert "重试次数超限" in result.error

    @pytest.mark.asyncio
    async def test_no_successful_nodes(self):
        """所有节点失败时应返回失败。"""
        orch = DistributedOrchestrator()
        nodes, resources = _make_nodes_and_resources()

        result = await orch._retry_with_fewer_nodes(
            instance_id=InstanceId("i1"),
            model_id="m",
            model_path="/m.gguf",
            model_vram_required_mb=10000,
            nodes=nodes,
            node_resources=resources,
            successful_nodes=[],
            strategy=PlacementStrategy.TENSOR,
            _retry_depth=0,
        )

        assert result.success is False
        assert "所有节点 RPC 启动失败" in result.error


class TestRequestRpcStart:
    """_request_rpc_start 各种 httpx 异常测试。"""

    @pytest.mark.asyncio
    async def test_connect_error(self):
        """连接失败应返回 None。"""
        orch = DistributedOrchestrator()
        node_info = StateNodeInfo(node_id=NodeId("w1"), ip="10.0.0.2", port=52415)

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
            mock_client_cls.return_value = mock_client

            result = await orch._request_rpc_start(node_info, timeout=5.0)

        assert result is None

    @pytest.mark.asyncio
    async def test_timeout_error(self):
        """连接超时应返回 None。"""
        orch = DistributedOrchestrator()
        node_info = StateNodeInfo(node_id=NodeId("w1"), ip="10.0.0.2", port=52415)

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
            mock_client_cls.return_value = mock_client

            result = await orch._request_rpc_start(node_info, timeout=5.0)

        assert result is None

    @pytest.mark.asyncio
    async def test_http_status_error(self):
        """HTTP 错误状态应返回 None。"""
        orch = DistributedOrchestrator()
        node_info = StateNodeInfo(node_id=NodeId("w1"), ip="10.0.0.2", port=52415)

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error", request=MagicMock(), response=mock_resp
        )

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            result = await orch._request_rpc_start(node_info, timeout=5.0)

        assert result is None

    @pytest.mark.asyncio
    async def test_missing_endpoint_in_response(self):
        """响应中缺少 endpoint 字段应返回 None。"""
        orch = DistributedOrchestrator()
        node_info = StateNodeInfo(node_id=NodeId("w1"), ip="10.0.0.2", port=52415)

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"status": "ok"}  # 无 endpoint

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            result = await orch._request_rpc_start(node_info, timeout=5.0)

        assert result is None

    @pytest.mark.asyncio
    async def test_successful_rpc_start(self):
        """成功启动 RPC 应返回 endpoint。"""
        orch = DistributedOrchestrator()
        node_info = StateNodeInfo(node_id=NodeId("w1"), ip="10.0.0.2", port=52415)

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"endpoint": "10.0.0.2:50052"}

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            result = await orch._request_rpc_start(node_info, timeout=5.0)

        assert result == "10.0.0.2:50052"


class TestStopWorkerRpcServer:
    """_stop_worker_rpc_server 各种 httpx 异常测试。"""

    @patch("httpx.post")
    def test_connect_error(self, mock_post):
        """连接失败不应抛出异常。"""
        mock_post.side_effect = httpx.ConnectError("Connection refused")
        orch = DistributedOrchestrator()
        node_info = StateNodeInfo(node_id=NodeId("w1"), ip="10.0.0.2", port=52415)

        # 不应抛出异常
        orch._stop_worker_rpc_server(NodeId("w1"), node_info)

    @patch("httpx.post")
    def test_timeout_error(self, mock_post):
        """超时不应抛出异常。"""
        mock_post.side_effect = httpx.TimeoutException("Timeout")
        orch = DistributedOrchestrator()
        node_info = StateNodeInfo(node_id=NodeId("w1"), ip="10.0.0.2", port=52415)

        orch._stop_worker_rpc_server(NodeId("w1"), node_info)

    @patch("httpx.post")
    def test_http_status_error(self, mock_post):
        """HTTP 错误状态不应抛出异常。"""
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error", request=MagicMock(), response=mock_resp
        )
        mock_post.return_value = mock_resp

        orch = DistributedOrchestrator()
        node_info = StateNodeInfo(node_id=NodeId("w1"), ip="10.0.0.2", port=52415)

        orch._stop_worker_rpc_server(NodeId("w1"), node_info)

    @patch("httpx.post")
    def test_successful_stop(self, mock_post):
        """成功停止应正常完成。"""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        orch = DistributedOrchestrator()
        node_info = StateNodeInfo(node_id=NodeId("w1"), ip="10.0.0.2", port=52415)

        orch._stop_worker_rpc_server(NodeId("w1"), node_info)
        mock_post.assert_called_once()


class TestCreateLocalInstance:
    """_create_local_instance 测试。"""

    @patch.object(DistributedOrchestrator, "_start_llama_server")
    async def test_success(self, mock_start):
        """成功创建本地实例。"""
        mock_start.return_value = MagicMock()  # 返回 mock engine
        orch = DistributedOrchestrator()
        result = await orch._create_local_instance(
            instance_id=InstanceId("i1"),
            model_id="m",
            model_path="/m.gguf",
        )
        assert result.success is True
        assert result.instance_id == InstanceId("i1")
        assert result.tensor_split == [1.0]

    @patch.object(DistributedOrchestrator, "_start_llama_server")
    async def test_failure(self, mock_start):
        """启动 llama-server 失败应返回错误。"""
        mock_start.side_effect = RuntimeError("Process crashed")
        orch = DistributedOrchestrator()
        result = await orch._create_local_instance(
            instance_id=InstanceId("i1"),
            model_id="m",
            model_path="/m.gguf",
        )
        assert result.success is False
        assert "启动本地 llama-server 失败" in result.error


class TestCalculateTensorSplit:
    """_calculate_tensor_split 测试。"""

    def test_no_local_node(self):
        """无本地节点时，应取第一个节点作为 master。"""
        orch = DistributedOrchestrator()
        nodes = {
            NodeId("worker-1"): StateNodeInfo(
                node_id=NodeId("worker-1"), ip="10.0.0.2", port=52415
            ),
            NodeId("worker-2"): StateNodeInfo(
                node_id=NodeId("worker-2"), ip="10.0.0.3", port=52415
            ),
        }
        resources = {
            NodeId("worker-1"): _make_resources(8000),
            NodeId("worker-2"): _make_resources(6000),
        }

        result = orch._calculate_tensor_split(
            [NodeId("worker-1"), NodeId("worker-2")],
            resources,
            nodes,
        )

        assert isinstance(result, TensorSplitResult)
        assert result.is_distributed is True
        assert len(result.splits) == 2

    def test_with_local_node(self):
        """有本地节点时正常计算。"""
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
            NodeId("worker-1"): _make_resources(8000),
        }

        result = orch._calculate_tensor_split(
            [NodeId("master"), NodeId("worker-1")],
            resources,
            nodes,
        )

        assert isinstance(result, TensorSplitResult)
        assert result.is_distributed is True
        assert len(result.splits) == 2


class TestBuildTopology:
    """_build_topology 测试。"""

    def test_empty_nodes(self):
        """空节点列表应正常返回拓扑。"""
        orch = DistributedOrchestrator()
        topology = orch._build_topology({}, {})
        assert topology.master_id == "master"
        assert len(topology.nodes) == 0

    def test_with_nodes(self):
        """有节点时应正确构建拓扑。"""
        orch = DistributedOrchestrator()
        nodes = {
            NodeId("master"): StateNodeInfo(
                node_id=NodeId("master"), ip="127.0.0.1", port=52415
            ),
        }
        resources = {
            NodeId("master"): _make_resources(10000),
        }

        topology = orch._build_topology(nodes, resources)
        assert topology.master_id == "master"
        assert len(topology.nodes) == 1
