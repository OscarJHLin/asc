"""分布式编排器增强测试。

覆盖审计中发现的关键缺口：
- _handle_rpc_failures 处理部分 RPC 失败
- _retry_with_fewer_nodes 最大深度限制
- _request_rpc_start Binary Frame 协议
- _stop_worker_rpc_server Binary Frame 协议
- _create_local_instance 失败
- _calculate_tensor_split 无本地节点
- _build_topology 空节点
"""

from unittest.mock import MagicMock, patch

import pytest

from asc.master.orchestrator import DistributedOrchestrator, OrchestratorResult
from asc.network.protocol import Channel, Envelope, Message, MessageType
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
    """_request_rpc_start Binary Frame 协议测试。"""

    @pytest.mark.asyncio
    async def test_no_tcp_connection(self):
        """无 TCP 连接时应返回 None。"""
        orch = DistributedOrchestrator()
        node_info = StateNodeInfo(node_id=NodeId("w1"), ip="10.0.0.2", port=52415)

        result = await orch._request_rpc_start(node_info, timeout=1.0)
        assert result is None

    @pytest.mark.asyncio
    async def test_send_failure(self):
        """发送失败应返回 None。"""
        mock_tcp = MagicMock()

        async def mock_send(conn_id, envelope):
            return False

        mock_tcp.send = mock_send

        orch = DistributedOrchestrator(
            tcp_server=mock_tcp,
            conn_node_map={"conn-1": "w1"},
        )
        node_info = StateNodeInfo(node_id=NodeId("w1"), ip="10.0.0.2", port=52415)

        result = await orch._request_rpc_start(node_info, timeout=1.0)
        assert result is None

    @pytest.mark.asyncio
    async def test_timeout(self):
        """等待 ACK 超时应返回 None。"""
        mock_tcp = MagicMock()

        async def mock_send(conn_id, envelope):
            # 不回复 ACK，模拟超时
            return True

        mock_tcp.send = mock_send

        orch = DistributedOrchestrator(
            tcp_server=mock_tcp,
            conn_node_map={"conn-1": "w1"},
        )
        node_info = StateNodeInfo(node_id=NodeId("w1"), ip="10.0.0.2", port=52415)

        result = await orch._request_rpc_start(node_info, timeout=0.1)
        assert result is None

    @pytest.mark.asyncio
    async def test_error_response(self):
        """Worker 返回错误状态应返回 None。"""
        mock_tcp = MagicMock()

        async def mock_send(conn_id, envelope):
            request_id = envelope.message.payload.get("request_id", "")
            ack = Envelope(
                channel=Channel.COMMANDS,
                message=Message(
                    type=MessageType.RPC_START_ACK,
                    sender_id="w1",
                    payload={"request_id": request_id, "status": "error", "error": "not found"},
                ),
            )
            await orch.handle_rpc_start_ack(ack)
            return True

        mock_tcp.send = mock_send

        orch = DistributedOrchestrator(
            tcp_server=mock_tcp,
            conn_node_map={"conn-1": "w1"},
        )
        node_info = StateNodeInfo(node_id=NodeId("w1"), ip="10.0.0.2", port=52415)

        result = await orch._request_rpc_start(node_info, timeout=5.0)
        assert result is None

    @pytest.mark.asyncio
    async def test_successful_rpc_start(self):
        """成功启动 RPC 应返回 endpoint。"""
        mock_tcp = MagicMock()

        async def mock_send(conn_id, envelope):
            request_id = envelope.message.payload.get("request_id", "")
            ack = Envelope(
                channel=Channel.COMMANDS,
                message=Message(
                    type=MessageType.RPC_START_ACK,
                    sender_id="w1",
                    payload={"request_id": request_id, "status": "ok", "endpoint": "10.0.0.2:50052", "port": 50052},
                ),
            )
            await orch.handle_rpc_start_ack(ack)
            return True

        mock_tcp.send = mock_send

        orch = DistributedOrchestrator(
            tcp_server=mock_tcp,
            conn_node_map={"conn-1": "w1"},
        )
        node_info = StateNodeInfo(node_id=NodeId("w1"), ip="10.0.0.2", port=52415)

        result = await orch._request_rpc_start(node_info, timeout=5.0)
        assert result == "10.0.0.2:50052"


class TestStopWorkerRpcServer:
    """_stop_worker_rpc_server Binary Frame 协议测试。"""

    @pytest.mark.asyncio
    async def test_no_tcp_connection(self):
        """无 TCP 连接时不应抛出异常。"""
        orch = DistributedOrchestrator()
        node_info = StateNodeInfo(node_id=NodeId("w1"), ip="10.0.0.2", port=52415)

        await orch._stop_worker_rpc_server(NodeId("w1"), node_info)

    @pytest.mark.asyncio
    async def test_send_failure(self):
        """发送失败不应抛出异常。"""
        mock_tcp = MagicMock()

        async def mock_send(conn_id, envelope):
            return False

        mock_tcp.send = mock_send

        orch = DistributedOrchestrator(
            tcp_server=mock_tcp,
            conn_node_map={"conn-1": "w1"},
        )
        node_info = StateNodeInfo(node_id=NodeId("w1"), ip="10.0.0.2", port=52415)

        await orch._stop_worker_rpc_server(NodeId("w1"), node_info)

    @pytest.mark.asyncio
    async def test_timeout(self):
        """等待 ACK 超时不应抛出异常。"""
        mock_tcp = MagicMock()

        async def mock_send(conn_id, envelope):
            # 不回复 ACK，模拟超时
            return True

        mock_tcp.send = mock_send

        orch = DistributedOrchestrator(
            tcp_server=mock_tcp,
            conn_node_map={"conn-1": "w1"},
        )
        node_info = StateNodeInfo(node_id=NodeId("w1"), ip="10.0.0.2", port=52415)

        await orch._stop_worker_rpc_server(NodeId("w1"), node_info)

    @pytest.mark.asyncio
    async def test_successful_stop(self):
        """成功停止应正常完成。"""
        mock_tcp = MagicMock()

        async def mock_send(conn_id, envelope):
            request_id = envelope.message.payload.get("request_id", "")
            ack = Envelope(
                channel=Channel.COMMANDS,
                message=Message(
                    type=MessageType.RPC_STOP_ACK,
                    sender_id="w1",
                    payload={"request_id": request_id, "status": "ok"},
                ),
            )
            await orch.handle_rpc_stop_ack(ack)
            return True

        mock_tcp.send = mock_send

        orch = DistributedOrchestrator(
            tcp_server=mock_tcp,
            conn_node_map={"conn-1": "w1"},
        )
        node_info = StateNodeInfo(node_id=NodeId("w1"), ip="10.0.0.2", port=52415)

        await orch._stop_worker_rpc_server(NodeId("w1"), node_info)


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


class TestPipelineStrategy:
    """Pipeline 并行策略测试。"""

    def _make_orchestrator_with_mock_placement(self, selected_nodes):
        """创建 mock 了放置决策的编排器。"""
        orch = DistributedOrchestrator()
        placement = PlacementResult(
            success=True,
            selected_nodes=selected_nodes,
            strategy=PlacementStrategy.PIPELINE,
            total_vram_free_mb=20000,
        )
        orch._placement = MagicMock()
        orch._placement.place = MagicMock(return_value=placement)
        return orch

    @pytest.mark.asyncio
    async def test_pipeline_strategy_uses_pipeline_planner(self):
        """Pipeline 策略应使用 PipelinePlanner 进行层分配。"""
        orch = self._make_orchestrator_with_mock_placement(["n1", "n2"])

        nodes = {
            NodeId("n1"): StateNodeInfo(node_id=NodeId("n1"), ip="10.0.0.1", port=52415),
            NodeId("n2"): StateNodeInfo(node_id=NodeId("n2"), ip="10.0.0.2", port=52415),
        }
        resources = {
            NodeId("n1"): _make_resources(8000),
            NodeId("n2"): _make_resources(8000),
        }

        mock_engine = MagicMock()
        mock_engine.close = MagicMock()

        with patch.object(orch, "_start_llama_server", return_value=mock_engine):
            result = await orch.create_instance(
                instance_id=InstanceId("inst-1"),
                model_id="llama-7b",
                model_path="/models/llama.gguf",
                model_vram_required_mb=16000,
                nodes=nodes,
                node_resources=resources,
                strategy=PlacementStrategy.PIPELINE,
            )

        assert result.success is True
        assert result.pipeline_plan is not None
        assert result.pipeline_plan.is_valid is True
        assert len(result.pipeline_plan.stages) == 2
        assert result.tensor_split == []  # Pipeline 不使用 tensor-split
        assert result.rpc_endpoints == []  # Pipeline 不使用 RPC

    @pytest.mark.asyncio
    async def test_pipeline_stages_cover_all_layers(self):
        """Pipeline 阶段应覆盖所有层。"""
        orch = self._make_orchestrator_with_mock_placement(["n1", "n2"])

        nodes = {
            NodeId("n1"): StateNodeInfo(node_id=NodeId("n1"), ip="10.0.0.1", port=52415),
            NodeId("n2"): StateNodeInfo(node_id=NodeId("n2"), ip="10.0.0.2", port=52415),
        }
        resources = {
            NodeId("n1"): _make_resources(8000),
            NodeId("n2"): _make_resources(8000),
        }

        mock_engine = MagicMock()
        mock_engine.close = MagicMock()

        with patch.object(orch, "_start_llama_server", return_value=mock_engine):
            result = await orch.create_instance(
                instance_id=InstanceId("inst-1"),
                model_id="llama-7b",
                model_path="/models/llama.gguf",
                model_vram_required_mb=16000,
                nodes=nodes,
                node_resources=resources,
                strategy=PlacementStrategy.PIPELINE,
            )

        plan = result.pipeline_plan
        assert plan.stages[0].start_layer == 0
        assert plan.stages[-1].end_layer == 31  # total_layers=32
        total_layers = sum(s.num_layers for s in plan.stages)
        assert total_layers == 32

    @pytest.mark.asyncio
    async def test_pipeline_engine_failure_cleans_up(self):
        """Pipeline 阶段启动失败时应清理已启动的引擎。"""
        orch = self._make_orchestrator_with_mock_placement(["n1", "n2"])

        nodes = {
            NodeId("n1"): StateNodeInfo(node_id=NodeId("n1"), ip="10.0.0.1", port=52415),
            NodeId("n2"): StateNodeInfo(node_id=NodeId("n2"), ip="10.0.0.2", port=52415),
        }
        resources = {
            NodeId("n1"): _make_resources(8000),
            NodeId("n2"): _make_resources(8000),
        }

        first_engine = MagicMock()
        first_engine.close = MagicMock()
        call_count = 0

        async def mock_start(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return first_engine
            raise RuntimeError("启动失败")

        with patch.object(orch, "_start_llama_server", side_effect=mock_start):
            result = await orch.create_instance(
                instance_id=InstanceId("inst-1"),
                model_id="llama-7b",
                model_path="/models/llama.gguf",
                model_vram_required_mb=16000,
                nodes=nodes,
                node_resources=resources,
                strategy=PlacementStrategy.PIPELINE,
            )

        assert result.success is False
        assert "启动失败" in result.error
        first_engine.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_pipeline_engines_stored_with_stage_keys(self):
        """Pipeline 引擎应使用 instance_id::node_id 格式的 key 存储。"""
        orch = self._make_orchestrator_with_mock_placement(["n1", "n2"])

        nodes = {
            NodeId("n1"): StateNodeInfo(node_id=NodeId("n1"), ip="10.0.0.1", port=52415),
            NodeId("n2"): StateNodeInfo(node_id=NodeId("n2"), ip="10.0.0.2", port=52415),
        }
        resources = {
            NodeId("n1"): _make_resources(8000),
            NodeId("n2"): _make_resources(8000),
        }

        mock_engine = MagicMock()
        mock_engine.close = MagicMock()

        with patch.object(orch, "_start_llama_server", return_value=mock_engine):
            await orch.create_instance(
                instance_id=InstanceId("inst-1"),
                model_id="llama-7b",
                model_path="/models/llama.gguf",
                model_vram_required_mb=16000,
                nodes=nodes,
                node_resources=resources,
                strategy=PlacementStrategy.PIPELINE,
            )

        # 应有 2 个引擎（n1 和 n2 各一个）
        assert len(orch._engines) == 2
        keys = list(orch._engines.keys())
        assert all("inst-1::" in str(k) for k in keys)
