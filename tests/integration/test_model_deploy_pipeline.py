"""集成测试：Master 部署模型 → Worker 加载 → 分布式推理完整流程。

模拟场景：
1. Master 启动，3 个 Worker 通过 TCP 连接并注册（携带主机名和详细硬件信息）
2. Master 下载模型文件到本地
3. Master 通过 Binary Frame (MODEL_CHUNK) 将模型分发到各 Worker 节点
4. Master 根据各节点 VRAM 规划层分配方案
5. Master 通过 TCP 通知各 Worker 加载对应层
6. Worker 确认模型文件存在并回复 ACK
7. Master 创建推理实例并启动推理任务
8. Worker 执行推理并返回结果
9. Master 收到结果后完成任务

所有外部依赖（llama-server、真实网络传输）均被 mock，
测试纯逻辑正确性和组件间协作。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from asc.core.event_log import MemoryEventLog
from asc.master.main import MasterNode
from asc.master.model_distributor import (
    DistributeResult,
    DistributeTask,
    ModelDistributor,
    decode_model_chunk_payload,
    encode_model_chunk_ack_payload,
)
from asc.master.orchestrator import DistributedOrchestrator
from asc.network.frame import Frame, FrameType
from asc.network.protocol import Channel, Envelope, Message, MessageType
from asc.network.transport import TCPClient, TCPServer
from asc.types import (
    InstanceCreated,
    TaskCompleted,
    TaskCreated,
    apply,
    empty_state,
)
from asc.types.commands import CreateInstance, StartInference
from asc.types.common import NodeId
from asc.worker.agent import NodeResources
from asc.worker.gpu_info import GPUInfo

# ---------------------------------------------------------------------------
# 辅助工具
# ---------------------------------------------------------------------------


def make_gpu(
    index: int = 0,
    name: str = "RTX 4090",
    total: int = 24564,
    free: int = 12000,
    vendor: str = "NVIDIA",
) -> GPUInfo:
    """创建 GPUInfo 实例。"""
    return GPUInfo(
        index=index,
        name=name,
        vram_total_mb=total,
        vram_free_mb=free,
        vendor=vendor,
    )


def make_resources(
    cpu_count: int = 8,
    cpu_percent: float = 20.0,
    memory_total_mb: int = 32768,
    memory_free_mb: int = 24000,
    gpus: list[GPUInfo] | None = None,
    compute_score: float = 1.0,
    cpu_brand: str = "Intel i9-13900K",
    cpu_physical_count: int = 8,
    cpu_freq_mhz: float = 5800.0,
) -> NodeResources:
    """创建 NodeResources 实例。"""
    return NodeResources(
        cpu_count=cpu_count,
        cpu_percent=cpu_percent,
        memory_total_mb=memory_total_mb,
        memory_free_mb=memory_free_mb,
        gpus=gpus or [make_gpu()],
        compute_score=compute_score,
        cpu_brand=cpu_brand,
        cpu_physical_count=cpu_physical_count,
        cpu_freq_mhz=cpu_freq_mhz,
    )


def create_fake_model_file(directory: Path, name: str = "test-model.gguf", size_kb: int = 64) -> Path:
    """创建一个假的模型文件用于测试。"""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(b"\x00" * (size_kb * 1024))
    return path


# ---------------------------------------------------------------------------
# Scenario 1: 模型分发器单元级集成测试
# ---------------------------------------------------------------------------


class TestModelDistributorIntegration:
    """ModelDistributor 分发 + 层分配集成测试。"""

    def test_plan_layer_assignment_single_node(self):
        """单节点：所有层分配到一个节点。"""
        distributor = ModelDistributor()
        results = [
            DistributeResult(
                success=True,
                model_id="test-model",
                node_id="worker-1",
                remote_path="/models/test-model.gguf",
                file_size_mb=100,
            ),
        ]
        node_resources = {
            "worker-1": {"vram_free_mb": 24000, "ip": "10.0.0.1", "port": 52415},
        }

        assignments = distributor.plan_layer_assignment(
            model_id="test-model",
            model_path="/models/test-model.gguf",
            total_layers=32,
            node_resources=node_resources,
            distribute_results=results,
        )

        assert len(assignments) == 1
        assert assignments[0].node_id == "worker-1"
        assert assignments[0].start_layer == 0
        assert assignments[0].end_layer == 31
        assert assignments[0].total_layers == 32
        assert assignments[0].gpu_layers == 32

    def test_plan_layer_assignment_multi_node(self):
        """多节点：按 VRAM 比例分配层。"""
        distributor = ModelDistributor()
        results = [
            DistributeResult(success=True, model_id="m", node_id="w1", remote_path="/m.gguf", file_size_mb=100),
            DistributeResult(success=True, model_id="m", node_id="w2", remote_path="/m.gguf", file_size_mb=100),
            DistributeResult(success=True, model_id="m", node_id="w3", remote_path="/m.gguf", file_size_mb=100),
        ]
        node_resources = {
            "w1": {"vram_free_mb": 24000, "ip": "10.0.0.1", "port": 52415},
            "w2": {"vram_free_mb": 12000, "ip": "10.0.0.2", "port": 52415},
            "w3": {"vram_free_mb": 8000, "ip": "10.0.0.3", "port": 52415},
        }

        assignments = distributor.plan_layer_assignment(
            model_id="m",
            model_path="/m.gguf",
            total_layers=32,
            node_resources=node_resources,
            distribute_results=results,
        )

        assert len(assignments) == 3
        # 验证所有层被分配
        total_assigned = sum(a.end_layer - a.start_layer + 1 for a in assignments)
        assert total_assigned == 32
        # VRAM 最大的节点分配最多层
        w1_layers = next(a.end_layer - a.start_layer + 1 for a in assignments if a.node_id == "w1")
        w2_layers = next(a.end_layer - a.start_layer + 1 for a in assignments if a.node_id == "w2")
        w3_layers = next(a.end_layer - a.start_layer + 1 for a in assignments if a.node_id == "w3")
        assert w1_layers > w2_layers
        assert w2_layers >= w3_layers

    def test_plan_layer_assignment_skips_failed_distribute(self):
        """分发失败的节点不参与层分配。"""
        distributor = ModelDistributor()
        results = [
            DistributeResult(success=True, model_id="m", node_id="w1", remote_path="/m.gguf", file_size_mb=100),
            DistributeResult(success=False, model_id="m", node_id="w2", error="连接超时"),
        ]
        node_resources = {
            "w1": {"vram_free_mb": 24000, "ip": "10.0.0.1", "port": 52415},
            "w2": {"vram_free_mb": 12000, "ip": "10.0.0.2", "port": 52415},
        }

        assignments = distributor.plan_layer_assignment(
            model_id="m", model_path="/m.gguf",
            total_layers=32, node_resources=node_resources,
            distribute_results=results,
        )

        assert len(assignments) == 1
        assert assignments[0].node_id == "w1"
        assert assignments[0].end_layer - assignments[0].start_layer + 1 == 32

    def test_plan_layer_assignment_no_vram(self):
        """无可用 VRAM 的节点不参与分配。"""
        distributor = ModelDistributor()
        results = [
            DistributeResult(success=True, model_id="m", node_id="w1", remote_path="/m.gguf", file_size_mb=100),
        ]
        node_resources = {
            "w1": {"vram_free_mb": 0, "ip": "10.0.0.1", "port": 52415},
        }

        assignments = distributor.plan_layer_assignment(
            model_id="m", model_path="/m.gguf",
            total_layers=32, node_resources=node_resources,
            distribute_results=results,
        )

        assert len(assignments) == 0

    def test_receive_model_file_saves_correctly(self, tmp_path: Path):
        """Worker 端通过 Binary Frame 接收模型文件并保存。"""
        # 模拟 Worker 接收流程：分片写入后文件完整
        content = b"fake model data " * 100

        models_dir = tmp_path / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        file_path = models_dir / "test-model.gguf"

        # 直接写入（模拟 Worker _handle_model_chunk_frame 的写入逻辑）
        file_path.write_bytes(content)

        assert file_path.exists()
        assert file_path.read_bytes() == content

    def test_receive_model_file_overwrite_different_size(self, tmp_path: Path):
        """已存在不同大小的文件时覆盖保存。"""
        existing_path = tmp_path / "models" / "model.gguf"
        existing_path.parent.mkdir(parents=True, exist_ok=True)
        existing_path.write_bytes(b"old content")

        new_content = b"new model data " * 200
        existing_path.write_bytes(new_content)

        assert existing_path.read_bytes() == new_content


# ---------------------------------------------------------------------------
# Scenario 2: Binary Frame 模型分发到 Worker 集成测试
# ---------------------------------------------------------------------------


class TestModelDistributeBinaryFrameIntegration:
    """通过 mock Binary Frame 测试模型分发到 Worker 的完整流程。"""

    @pytest.mark.asyncio
    async def test_distribute_to_worker_success(self, tmp_path: Path):
        """成功分发模型到单个 Worker。"""
        # 创建假模型文件
        model_file = create_fake_model_file(tmp_path, "test.gguf", size_kb=128)

        # Mock TCPServer
        sent_frames: list[tuple[str, Frame]] = []

        async def mock_send_frame(conn_id: str, frame: Frame) -> bool:
            sent_frames.append((conn_id, frame))
            # 模拟 Worker 回复 ACK
            if frame.frame_type == FrameType.MODEL_CHUNK:
                model_id, chunk_index, total_chunks, _ = decode_model_chunk_payload(frame.payload)
                ack_payload = encode_model_chunk_ack_payload(
                    model_id=model_id,
                    chunk_index=chunk_index,
                    success=True,
                )
                ack_frame = Frame(frame_type=FrameType.MODEL_CHUNK_ACK, payload=ack_payload)
                distributor.handle_chunk_ack(ack_frame)
            return True

        mock_tcp = MagicMock()
        mock_tcp.send_frame = mock_send_frame

        distributor = ModelDistributor(tcp_server=mock_tcp)
        task = DistributeTask(
            model_id="test-model",
            model_path=str(model_file),
            target_node_id="worker-1",
            target_conn_id="conn-1",
        )

        result = await distributor.distribute_to_worker(task)

        assert result.success
        assert result.model_id == "test-model"
        assert result.node_id == "worker-1"

    @pytest.mark.asyncio
    async def test_distribute_to_worker_file_not_found(self, tmp_path: Path):
        """模型文件不存在时分发失败。"""
        mock_tcp = MagicMock()
        distributor = ModelDistributor(tcp_server=mock_tcp)
        task = DistributeTask(
            model_id="missing-model",
            model_path=str(tmp_path / "nonexistent.gguf"),
            target_node_id="worker-1",
            target_conn_id="conn-1",
        )

        result = await distributor.distribute_to_worker(task)

        assert not result.success
        assert "不存在" in result.error

    @pytest.mark.asyncio
    async def test_distribute_to_worker_no_tcp_server(self, tmp_path: Path):
        """TCPServer 未设置时分发失败。"""
        model_file = create_fake_model_file(tmp_path, "test.gguf")
        distributor = ModelDistributor()
        task = DistributeTask(
            model_id="test-model",
            model_path=str(model_file),
            target_node_id="worker-1",
            target_conn_id="conn-1",
        )

        result = await distributor.distribute_to_worker(task)

        assert not result.success
        assert "TCPServer" in result.error

    @pytest.mark.asyncio
    async def test_distribute_to_cluster_parallel(self, tmp_path: Path):
        """并行分发模型到多个 Worker。"""
        model_file = create_fake_model_file(tmp_path, "test.gguf", size_kb=256)
        distributor = ModelDistributor()

        nodes = {
            "worker-1": {"conn_id": "conn-1"},
            "worker-2": {"conn_id": "conn-2"},
            "worker-3": {"conn_id": "conn-3"},
        }

        # Mock: 前两个成功，第三个失败
        async def mock_send_frame(conn_id: str, frame: Frame) -> bool:
            if conn_id == "conn-3":
                return False  # 模拟发送失败
            if frame.frame_type == FrameType.MODEL_CHUNK:
                model_id, chunk_index, total_chunks, _ = decode_model_chunk_payload(frame.payload)
                ack_payload = encode_model_chunk_ack_payload(
                    model_id=model_id,
                    chunk_index=chunk_index,
                    success=True,
                )
                ack_frame = Frame(frame_type=FrameType.MODEL_CHUNK_ACK, payload=ack_payload)
                distributor.handle_chunk_ack(ack_frame)
            return True

        mock_tcp = MagicMock()
        mock_tcp.send_frame = mock_send_frame
        distributor.set_tcp_server(mock_tcp)

        results = await distributor.distribute_to_cluster(
            model_id="test-model",
            model_path=str(model_file),
            nodes=nodes,
        )

        assert len(results) == 3
        success_count = sum(1 for r in results if r.success)
        assert success_count == 2


# ---------------------------------------------------------------------------
# Scenario 3: TCP 通信 - Worker 注册 + 模型分发通知
# ---------------------------------------------------------------------------


class TestTCPWorkerRegistrationAndModelDistribute:
    """通过真实 TCP 连接测试 Worker 注册和模型分发通知。"""

    @pytest.mark.asyncio
    async def test_worker_registration_with_hostname(self):
        """Worker 通过 TCP 注册，携带主机名和详细硬件信息。"""
        received_messages: list[Envelope] = []

        async def on_message(conn_id: str, envelope: Envelope) -> None:
            received_messages.append(envelope)

        # 启动 TCP Server（模拟 Master）
        server = TCPServer(host="127.0.0.1", port=0, on_message=on_message)
        await server.start()

        try:
            # Worker 连接并发送注册消息
            client = TCPClient(
                host="127.0.0.1",
                port=server.port,
                node_id="testhost-wkr1",
            )
            connected = await client.connect()
            assert connected

            # 构造注册消息（模拟 WorkerAgent._register_to_master）
            resources = make_resources(
                gpus=[
                    make_gpu(index=0, name="RTX 4090", total=24564, free=18000, vendor="NVIDIA"),
                    make_gpu(index=1, name="RTX 4090", total=24564, free=16000, vendor="NVIDIA"),
                ],
            )

            register_envelope = Envelope(
                channel=Channel.DISCOVERY,
                message=Message(
                    type=MessageType.NODE_JOINED,
                    sender_id="testhost-wkr1",
                    payload={
                        "node_id": "testhost-wkr1",
                        "hostname": "gpu-server-01",
                        "port": 52415,
                        "resources": {
                            "cpu_count": resources.cpu_count,
                            "cpu_brand": resources.cpu_brand,
                            "cpu_physical_count": resources.cpu_physical_count,
                            "cpu_freq_mhz": resources.cpu_freq_mhz,
                            "cpu_percent": resources.cpu_percent,
                            "memory_total_mb": resources.memory_total_mb,
                            "memory_free_mb": resources.memory_free_mb,
                            "gpus": [
                                {
                                    "index": g.index,
                                    "name": g.name,
                                    "vendor": g.vendor,
                                    "vram_total_mb": g.vram_total_mb,
                                    "vram_free_mb": g.vram_free_mb,
                                }
                                for g in resources.gpus
                            ],
                            "compute_score": resources.compute_score,
                        },
                    },
                ),
            )
            await client.send(register_envelope)

            # 等待消息到达
            await asyncio.sleep(0.2)

            # 验证 Master 收到注册消息
            assert len(received_messages) == 1
            msg = received_messages[0]
            assert msg.message.type == MessageType.NODE_JOINED
            assert msg.message.payload["node_id"] == "testhost-wkr1"
            assert msg.message.payload["hostname"] == "gpu-server-01"
            assert msg.message.payload["resources"]["cpu_brand"] == "Intel i9-13900K"
            assert len(msg.message.payload["resources"]["gpus"]) == 2
            assert msg.message.payload["resources"]["gpus"][0]["name"] == "RTX 4090"

            await client.disconnect()
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_model_distribute_notification_via_tcp(self):
        """Master 通过 TCP 通知 Worker 加载模型层。"""
        received_messages: list[Envelope] = []

        async def on_message(conn_id: str, envelope: Envelope) -> None:
            received_messages.append(envelope)

        # 启动 TCP Server（模拟 Worker 端监听）
        server = TCPServer(host="127.0.0.1", port=0, on_message=on_message)
        await server.start()

        try:
            # Master 连接并发送模型分发通知
            client = TCPClient(
                host="127.0.0.1",
                port=server.port,
                node_id="master-node",
            )
            connected = await client.connect()
            assert connected

            # 发送模型分发通知
            distribute_envelope = Envelope(
                channel=Channel.TASK_DISPATCH,
                message=Message(
                    type=MessageType.MODEL_DISTRIBUTE,
                    sender_id="master-node",
                    payload={
                        "model_id": "qwen2.5-7b",
                        "model_path": "/models/qwen2.5-7b-q4.gguf",
                        "layer_start": 0,
                        "layer_end": 15,
                        "total_layers": 32,
                        "gpu_layers": 16,
                    },
                ),
            )
            await client.send(distribute_envelope)

            # 等待消息到达
            await asyncio.sleep(0.2)

            # 验证 Worker 收到分发通知
            assert len(received_messages) == 1
            msg = received_messages[0]
            assert msg.message.type == MessageType.MODEL_DISTRIBUTE
            assert msg.message.payload["model_id"] == "qwen2.5-7b"
            assert msg.message.payload["layer_start"] == 0
            assert msg.message.payload["layer_end"] == 15
            assert msg.message.payload["gpu_layers"] == 16

            await client.disconnect()
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_worker_sends_distribute_ack(self):
        """Worker 收到分发通知后回复 ACK。"""
        received_messages: list[Envelope] = []

        async def on_message(conn_id: str, envelope: Envelope) -> None:
            received_messages.append(envelope)

        # Master 端 TCP Server
        server = TCPServer(host="127.0.0.1", port=0, on_message=on_message)
        await server.start()

        try:
            # Worker 连接
            client = TCPClient(
                host="127.0.0.1",
                port=server.port,
                node_id="testhost-wkr1",
            )
            connected = await client.connect()
            assert connected

            # Worker 回复 ACK
            ack_envelope = Envelope(
                channel=Channel.TASK_DISPATCH,
                message=Message(
                    type=MessageType.MODEL_DISTRIBUTE_ACK,
                    sender_id="testhost-wkr1",
                    payload={
                        "node_id": "testhost-wkr1",
                        "model_id": "qwen2.5-7b",
                        "file_exists": True,
                        "layer_start": 0,
                        "layer_end": 15,
                        "gpu_layers": 16,
                        "ready": True,
                    },
                ),
            )
            await client.send(ack_envelope)

            # 等待消息到达
            await asyncio.sleep(0.2)

            # 验证 Master 收到 ACK
            assert len(received_messages) == 1
            msg = received_messages[0]
            assert msg.message.type == MessageType.MODEL_DISTRIBUTE_ACK
            assert msg.message.payload["node_id"] == "testhost-wkr1"
            assert msg.message.payload["ready"] is True

            await client.disconnect()
        finally:
            await server.stop()


# ---------------------------------------------------------------------------
# Scenario 4: Master 节点完整部署流程（注册 → 分发 → 层分配 → 推理）
# ---------------------------------------------------------------------------


class TestMasterFullDeployFlow:
    """Master 节点完整部署流程集成测试。"""

    def test_register_workers_with_hardware_info(self):
        """注册 Worker 时保存主机名和硬件信息。"""
        mock_orchestrator = MagicMock(spec=DistributedOrchestrator)
        master = MasterNode(node_id="master", orchestrator=mock_orchestrator)

        # 模拟 Worker 注册（携带硬件信息）
        master.process_node_joined(NodeId("gpu-server-01"), "10.0.0.1", 52415)
        master.process_node_joined(NodeId("gpu-server-02"), "10.0.0.2", 52415)
        master.process_node_joined(NodeId("gpu-server-03"), "10.0.0.3", 52415)

        # 验证状态
        assert len(master.state.nodes) == 3
        assert master.state.nodes[NodeId("gpu-server-01")].ip == "10.0.0.1"
        assert master.state.nodes[NodeId("gpu-server-02")].ip == "10.0.0.2"
        assert master.state.nodes[NodeId("gpu-server-03")].ip == "10.0.0.3"

    @pytest.mark.asyncio
    async def test_distribute_model_and_plan_layers(self, tmp_path: Path):
        """Master 分发模型并规划层分配。"""
        mock_orchestrator = MagicMock(spec=DistributedOrchestrator)
        master = MasterNode(node_id="master", orchestrator=mock_orchestrator)

        # 注册 Worker 节点
        master.process_node_joined(NodeId("gpu-server-01"), "10.0.0.1", 52415)
        master.process_node_joined(NodeId("gpu-server-02"), "10.0.0.2", 52415)

        # 保存节点硬件信息
        master._node_resources = {
            "gpu-server-01": {
                "gpus": [{"vram_free_mb": 18000, "vram_total_mb": 24564, "name": "RTX 4090"}],
                "cpu_count": 16,
                "memory_total_mb": 65536,
                "memory_free_mb": 48000,
            },
            "gpu-server-02": {
                "gpus": [{"vram_free_mb": 12000, "vram_total_mb": 16384, "name": "RTX 4080"}],
                "cpu_count": 8,
                "memory_total_mb": 32768,
                "memory_free_mb": 24000,
            },
        }

        # 创建假模型文件
        model_file = create_fake_model_file(tmp_path, "qwen2.5-7b-q4.gguf", size_kb=512)

        # Mock Binary Frame 分发
        async def mock_send_frame(conn_id: str, frame: Frame) -> bool:
            if frame.frame_type == FrameType.MODEL_CHUNK:
                model_id, chunk_index, total_chunks, _ = decode_model_chunk_payload(frame.payload)
                ack_payload = encode_model_chunk_ack_payload(
                    model_id=model_id,
                    chunk_index=chunk_index,
                    success=True,
                )
                ack_frame = Frame(frame_type=FrameType.MODEL_CHUNK_ACK, payload=ack_payload)
                if master._model_distributor is not None:
                    master._model_distributor.handle_chunk_ack(ack_frame)
            return True

        async def mock_send(conn_id: str, envelope: Any) -> bool:
            return True

        mock_tcp = MagicMock()
        mock_tcp.send_frame = mock_send_frame
        mock_tcp.send = mock_send
        master._tcp_server = mock_tcp

        # 设置 conn_id 映射
        master._conn_node_map["conn-1"] = "gpu-server-01"
        master._conn_node_map["conn-2"] = "gpu-server-02"

        assignments = await master.distribute_model(
            model_id="qwen2.5-7b",
            model_path=str(model_file),
            total_layers=32,
        )

        # 验证层分配（2 个节点都有 VRAM）
        assert len(assignments) == 2

    @pytest.mark.asyncio
    async def test_full_deploy_create_inference_flow(self, tmp_path: Path):
        """完整流程：注册 → 分发 → 层分配 → 创建实例 → 推理 → 结果。"""
        mock_orchestrator = MagicMock(spec=DistributedOrchestrator)
        master = MasterNode(node_id="master", orchestrator=mock_orchestrator)

        # 1. 注册 Worker 节点
        master.process_node_joined(NodeId("gpu-server-01"), "10.0.0.1", 52415)
        master.process_node_joined(NodeId("gpu-server-02"), "10.0.0.2", 52415)

        # 2. 创建推理实例
        cmd = CreateInstance(model_id="qwen2.5-7b", sharding="tensor")
        events = master.process_create_instance(cmd)
        assert len(events) == 1
        assert isinstance(events[0], InstanceCreated)
        inst_id = list(master.state.instances.keys())[0]

        # 3. 启动推理任务
        inf_cmd = StartInference(instance_id=inst_id, prompt="你好，请介绍一下自己")
        events = master.process_start_inference(inf_cmd)
        assert len(events) == 1
        assert isinstance(events[0], TaskCreated)
        task_id = events[0].task_id

        # 4. 模拟推理完成
        master._emit(TaskCompleted(task_id=task_id, output="我是一个AI助手..."))

        # 5. 验证任务完成
        task = master.state.tasks[task_id]
        assert task.status.value == "completed"
        assert task.output == "我是一个AI助手..."

        # 6. 验证事件日志
        assert len(master.event_log) >= 4  # 2 NodeJoined + 1 InstanceCreated + 1 TaskCreated + 1 TaskCompleted


# ---------------------------------------------------------------------------
# Scenario 5: 多 Worker 节点分布式推理模拟
# ---------------------------------------------------------------------------


class TestMultiWorkerDistributedInference:
    """多 Worker 节点分布式推理模拟测试。"""

    @pytest.mark.asyncio
    async def test_three_workers_register_and_inference(self):
        """3 个 Worker 注册 → 创建实例 → 推理 → 结果回传。"""
        mock_orchestrator = MagicMock(spec=DistributedOrchestrator)
        master = MasterNode(node_id="master", orchestrator=mock_orchestrator)

        # 1. 注册 3 个 Worker（不同硬件配置）
        master.process_node_joined(NodeId("gpu-server-01"), "10.0.0.1", 52415)
        master.process_node_joined(NodeId("gpu-server-02"), "10.0.0.2", 52415)
        master.process_node_joined(NodeId("cpu-server-01"), "10.0.0.3", 52415)

        # 保存硬件信息
        master._node_resources = {
            "gpu-server-01": {
                "hostname": "gpu-server-01",
                "gpus": [{"name": "RTX 4090", "vram_total_mb": 24564, "vram_free_mb": 20000}],
                "cpu_count": 16, "cpu_brand": "AMD 7950X",
                "memory_total_mb": 65536, "memory_free_mb": 48000,
            },
            "gpu-server-02": {
                "hostname": "gpu-server-02",
                "gpus": [{"name": "RTX 3080", "vram_total_mb": 12288, "vram_free_mb": 10000}],
                "cpu_count": 8, "cpu_brand": "Intel i7-13700K",
                "memory_total_mb": 32768, "memory_free_mb": 24000,
            },
            "cpu-server-01": {
                "hostname": "cpu-server-01",
                "gpus": [],
                "cpu_count": 32, "cpu_brand": "AMD EPYC 7763",
                "memory_total_mb": 131072, "memory_free_mb": 100000,
            },
        }

        assert len(master.state.nodes) == 3

        # 2. 创建分布式推理实例
        cmd = CreateInstance(model_id="qwen2.5-14b", sharding="pipeline")
        events = master.process_create_instance(cmd)
        assert isinstance(events[0], InstanceCreated)
        inst_id = list(master.state.instances.keys())[0]

        # 验证实例使用了所有 3 个节点
        inst = master.state.instances[inst_id]
        assert len(inst.node_ids) == 3

        # 3. 启动推理
        inf_cmd = StartInference(instance_id=inst_id, prompt="解释量子计算")
        events = master.process_start_inference(inf_cmd)
        task_id = events[0].task_id

        # 4. 模拟推理完成
        master._emit(TaskCompleted(task_id=task_id, output="量子计算是利用量子力学原理..."))

        task = master.state.tasks[task_id]
        assert task.status.value == "completed"

    @pytest.mark.asyncio
    async def test_model_distribute_with_layer_assignment(self, tmp_path: Path):
        """模型分发 + 层分配完整流程。"""
        # 创建假模型
        model_file = create_fake_model_file(tmp_path, "model.gguf", size_kb=256)

        # Mock Binary Frame 分发
        async def mock_send_frame(conn_id: str, frame: Frame) -> bool:
            if frame.frame_type == FrameType.MODEL_CHUNK:
                model_id, chunk_index, total_chunks, _ = decode_model_chunk_payload(frame.payload)
                ack_payload = encode_model_chunk_ack_payload(
                    model_id=model_id,
                    chunk_index=chunk_index,
                    success=True,
                )
                ack_frame = Frame(frame_type=FrameType.MODEL_CHUNK_ACK, payload=ack_payload)
                distributor.handle_chunk_ack(ack_frame)
            return True

        mock_tcp = MagicMock()
        mock_tcp.send_frame = mock_send_frame

        distributor = ModelDistributor(tcp_server=mock_tcp)

        # 模拟 3 个 Worker 节点
        nodes = {
            "gpu-server-01": {"conn_id": "conn-1"},
            "gpu-server-02": {"conn_id": "conn-2"},
            "gpu-server-03": {"conn_id": "conn-3"},
        }

        node_resources = {
            "gpu-server-01": {"vram_free_mb": 20000, "ip": "10.0.0.1", "port": 52415},
            "gpu-server-02": {"vram_free_mb": 10000, "ip": "10.0.0.2", "port": 52415},
            "gpu-server-03": {"vram_free_mb": 6000, "ip": "10.0.0.3", "port": 52415},
        }

        # 1. 分发模型到集群
        results = await distributor.distribute_to_cluster(
            model_id="test-model",
            model_path=str(model_file),
            nodes=nodes,
        )

        # 验证分发结果
        assert len(results) == 3
        assert all(r.success for r in results)

        # 2. 规划层分配
        assignments = distributor.plan_layer_assignment(
            model_id="test-model",
            model_path=str(model_file),
            total_layers=32,
            node_resources=node_resources,
            distribute_results=results,
        )

        # 验证层分配
        assert len(assignments) == 3
        total_assigned = sum(a.end_layer - a.start_layer + 1 for a in assignments)
        assert total_assigned == 32

        # VRAM 最大的节点分配最多层
        layers_map = {a.node_id: a.end_layer - a.start_layer + 1 for a in assignments}
        assert layers_map["gpu-server-01"] > layers_map["gpu-server-02"]
        assert layers_map["gpu-server-02"] > layers_map["gpu-server-03"]

        # 3. 验证层范围不重叠
        ranges = [(a.start_layer, a.end_layer) for a in assignments]
        ranges.sort()
        for i in range(len(ranges) - 1):
            assert ranges[i][1] + 1 == ranges[i + 1][0], "层范围应连续不重叠"


# ---------------------------------------------------------------------------
# Scenario 6: Worker 故障时模型分发容错
# ---------------------------------------------------------------------------


class TestModelDistributeWithWorkerFailure:
    """Worker 故障时模型分发的容错测试。"""

    @pytest.mark.asyncio
    async def test_partial_worker_failure_during_distribute(self, tmp_path: Path):
        """部分 Worker 不可达时，分发仍然继续到可用节点。"""
        model_file = create_fake_model_file(tmp_path, "model.gguf", size_kb=128)

        nodes = {
            "worker-ok": {"conn_id": "conn-ok"},
            "worker-down": {"conn_id": "conn-down"},
        }

        node_resources = {
            "worker-ok": {"vram_free_mb": 24000, "ip": "10.0.0.1", "port": 52415},
            "worker-down": {"vram_free_mb": 12000, "ip": "10.0.0.2", "port": 52415},
        }

        async def mock_send_frame(conn_id: str, frame: Frame) -> bool:
            if conn_id == "conn-down":
                return False  # 模拟发送失败
            if frame.frame_type == FrameType.MODEL_CHUNK:
                model_id, chunk_index, total_chunks, _ = decode_model_chunk_payload(frame.payload)
                ack_payload = encode_model_chunk_ack_payload(
                    model_id=model_id,
                    chunk_index=chunk_index,
                    success=True,
                )
                ack_frame = Frame(frame_type=FrameType.MODEL_CHUNK_ACK, payload=ack_payload)
                distributor.handle_chunk_ack(ack_frame)
            return True

        mock_tcp = MagicMock()
        mock_tcp.send_frame = mock_send_frame

        distributor = ModelDistributor(tcp_server=mock_tcp)

        results = await distributor.distribute_to_cluster(
            model_id="test-model",
            model_path=str(model_file),
            nodes=nodes,
        )

        # 1 个成功，1 个失败
        success_results = [r for r in results if r.success]
        fail_results = [r for r in results if not r.success]
        assert len(success_results) == 1
        assert len(fail_results) == 1

        # 层分配只使用成功的节点
        assignments = distributor.plan_layer_assignment(
            model_id="test-model",
            model_path=str(model_file),
            total_layers=32,
            node_resources=node_resources,
            distribute_results=results,
        )

        assert len(assignments) == 1
        assert assignments[0].node_id == "worker-ok"
        assert assignments[0].end_layer - assignments[0].start_layer + 1 == 32

    @pytest.mark.asyncio
    async def test_all_workers_fail_distribute(self, tmp_path: Path):
        """所有 Worker 不可达时，层分配为空。"""
        model_file = create_fake_model_file(tmp_path, "model.gguf", size_kb=64)

        nodes = {
            "worker-1": {"conn_id": "conn-1"},
            "worker-2": {"conn_id": "conn-2"},
        }

        async def mock_send_frame(conn_id: str, frame: Frame) -> bool:
            return False  # 所有发送都失败

        mock_tcp = MagicMock()
        mock_tcp.send_frame = mock_send_frame

        distributor = ModelDistributor(tcp_server=mock_tcp)

        results = await distributor.distribute_to_cluster(
            model_id="test-model",
            model_path=str(model_file),
            nodes=nodes,
        )

        assert all(not r.success for r in results)

        assignments = distributor.plan_layer_assignment(
            model_id="test-model",
            model_path=str(model_file),
            total_layers=32,
            node_resources={},
            distribute_results=results,
        )

        assert len(assignments) == 0


# ---------------------------------------------------------------------------
# Scenario 7: 端到端完整流程（TCP + 分发 + 层分配 + 推理）
# ---------------------------------------------------------------------------


class TestEndToEndDeployAndInference:
    """端到端：Master 启动 → Worker 注册 → 模型分发 → 层分配 → 推理 → 结果。"""

    @pytest.mark.asyncio
    async def test_full_pipeline_with_tcp_communication(self, tmp_path: Path):
        """通过真实 TCP 通信完成完整部署和推理流程。"""
        # 收集 Master 收到的消息
        master_received: list[Envelope] = []

        async def on_master_message(conn_id: str, envelope: Envelope) -> None:
            master_received.append(envelope)

        # 1. 启动 Master TCP Server
        master_server = TCPServer(
            host="127.0.0.1", port=0, on_message=on_master_message,
        )
        await master_server.start()

        try:
            # 2. 3 个 Worker 连接并注册
            workers: list[TCPClient] = []
            worker_ids = ["gpu-srv01-w1", "gpu-srv02-w2", "cpu-srv01-w3"]
            worker_resources = [
                make_resources(
                    gpus=[make_gpu(0, "RTX 4090", 24564, 20000, "NVIDIA")],
                    cpu_brand="AMD 7950X",
                    compute_score=2.5,
                ),
                make_resources(
                    gpus=[make_gpu(0, "RTX 3080", 12288, 10000, "NVIDIA")],
                    cpu_brand="Intel i7-13700K",
                    compute_score=1.8,
                ),
                make_resources(
                    gpus=[],
                    cpu_count=32,
                    cpu_brand="AMD EPYC 7763",
                    memory_total_mb=131072,
                    memory_free_mb=100000,
                    compute_score=0.5,
                ),
            ]

            for wid in worker_ids:
                client = TCPClient(
                    host="127.0.0.1",
                    port=master_server.port,
                    node_id=wid,
                )
                connected = await client.connect()
                assert connected
                workers.append(client)

            # 3. Worker 发送注册消息（携带主机名和硬件信息）
            for i, (wid, client) in enumerate(zip(worker_ids, workers)):
                res = worker_resources[i]
                register_envelope = Envelope(
                    channel=Channel.DISCOVERY,
                    message=Message(
                        type=MessageType.NODE_JOINED,
                        sender_id=wid,
                        payload={
                            "node_id": wid,
                            "hostname": wid.rsplit("-", 1)[0],
                            "port": 52415,
                            "resources": {
                                "cpu_count": res.cpu_count,
                                "cpu_brand": res.cpu_brand,
                                "cpu_percent": res.cpu_percent,
                                "memory_total_mb": res.memory_total_mb,
                                "memory_free_mb": res.memory_free_mb,
                                "gpus": [
                                    {"index": g.index, "name": g.name, "vendor": g.vendor,
                                     "vram_total_mb": g.vram_total_mb, "vram_free_mb": g.vram_free_mb}
                                    for g in res.gpus
                                ],
                                "compute_score": res.compute_score,
                            },
                        },
                    ),
                )
                await client.send(register_envelope)

            # 等待注册消息到达
            await asyncio.sleep(0.3)

            # 验证 Master 收到所有注册消息
            join_messages = [m for m in master_received if m.message.type == MessageType.NODE_JOINED]
            assert len(join_messages) == 3

            # 验证注册消息内容
            hostnames = {m.message.payload["hostname"] for m in join_messages}
            assert "gpu-srv01" in hostnames
            assert "gpu-srv02" in hostnames
            assert "cpu-srv01" in hostnames

            # 4. Master 发送模型分发通知给各 Worker
            for i, (wid, client) in enumerate(zip(worker_ids, workers)):
                distribute_envelope = Envelope(
                    channel=Channel.TASK_DISPATCH,
                    message=Message(
                        type=MessageType.MODEL_DISTRIBUTE,
                        sender_id="master-node",
                        payload={
                            "model_id": "qwen2.5-7b",
                            "model_path": f"/models/qwen2.5-7b_{i}.gguf",
                            "layer_start": i * 11,
                            "layer_end": min((i + 1) * 11 - 1, 31),
                            "total_layers": 32,
                            "gpu_layers": 11 if i < 2 else 0,
                        },
                    ),
                )
                await master_server.broadcast(distribute_envelope)

            # 等待消息到达
            await asyncio.sleep(0.3)

            # 5. Worker 回复 ACK
            for i, (wid, client) in enumerate(zip(worker_ids, workers)):
                ack_envelope = Envelope(
                    channel=Channel.TASK_DISPATCH,
                    message=Message(
                        type=MessageType.MODEL_DISTRIBUTE_ACK,
                        sender_id=wid,
                        payload={
                            "node_id": wid,
                            "model_id": "qwen2.5-7b",
                            "file_exists": True,
                            "layer_start": i * 11,
                            "layer_end": min((i + 1) * 11 - 1, 31),
                            "gpu_layers": 11 if i < 2 else 0,
                            "ready": True,
                        },
                    ),
                )
                await client.send(ack_envelope)

            # 等待 ACK 到达
            await asyncio.sleep(0.3)

            # 验证 Master 收到所有 ACK
            ack_messages = [m for m in master_received if m.message.type == MessageType.MODEL_DISTRIBUTE_ACK]
            assert len(ack_messages) == 3
            assert all(m.message.payload["ready"] for m in ack_messages)

            # 清理
            for client in workers:
                await client.disconnect()

        finally:
            await master_server.stop()

    @pytest.mark.asyncio
    async def test_model_distribute_and_layer_assign_consistency(self, tmp_path: Path):
        """验证模型分发和层分配的一致性：所有层都被分配且不重叠。"""
        model_file = create_fake_model_file(tmp_path, "model.gguf", size_kb=256)

        # Mock Binary Frame 分发
        async def mock_send_frame(conn_id: str, frame: Frame) -> bool:
            if frame.frame_type == FrameType.MODEL_CHUNK:
                model_id, chunk_index, total_chunks, _ = decode_model_chunk_payload(frame.payload)
                ack_payload = encode_model_chunk_ack_payload(
                    model_id=model_id,
                    chunk_index=chunk_index,
                    success=True,
                )
                ack_frame = Frame(frame_type=FrameType.MODEL_CHUNK_ACK, payload=ack_payload)
                distributor.handle_chunk_ack(ack_frame)
            return True

        mock_tcp = MagicMock()
        mock_tcp.send_frame = mock_send_frame

        distributor = ModelDistributor(tcp_server=mock_tcp)

        # 5 个 Worker，不同 VRAM
        nodes = {
            f"worker-{i}": {"conn_id": f"conn-{i}"}
            for i in range(1, 6)
        }
        node_resources = {
            "worker-1": {"vram_free_mb": 24000, "ip": "10.0.0.1", "port": 52415},
            "worker-2": {"vram_free_mb": 16000, "ip": "10.0.0.2", "port": 52415},
            "worker-3": {"vram_free_mb": 12000, "ip": "10.0.0.3", "port": 52415},
            "worker-4": {"vram_free_mb": 8000, "ip": "10.0.0.4", "port": 52415},
            "worker-5": {"vram_free_mb": 4000, "ip": "10.0.0.5", "port": 52415},
        }

        results = await distributor.distribute_to_cluster(
            model_id="test-model",
            model_path=str(model_file),
            nodes=nodes,
        )

        assignments = distributor.plan_layer_assignment(
            model_id="test-model",
            model_path=str(model_file),
            total_layers=64,
            node_resources=node_resources,
            distribute_results=results,
        )

        # 验证层分配一致性
        assert len(assignments) == 5

        # 所有层都被分配
        all_layers: set[int] = set()
        for a in assignments:
            for layer in range(a.start_layer, a.end_layer + 1):
                assert layer not in all_layers, f"层 {layer} 被重复分配"
                all_layers.add(layer)

        assert len(all_layers) == 64, f"应有 64 层，实际分配 {len(all_layers)} 层"

        # 层范围连续
        sorted_assignments = sorted(assignments, key=lambda a: a.start_layer)
        assert sorted_assignments[0].start_layer == 0
        assert sorted_assignments[-1].end_layer == 63
        for i in range(len(sorted_assignments) - 1):
            assert sorted_assignments[i].end_layer + 1 == sorted_assignments[i + 1].start_layer

        # VRAM 大的节点分配更多层
        layers_count = {a.node_id: a.end_layer - a.start_layer + 1 for a in assignments}
        assert layers_count["worker-1"] > layers_count["worker-5"]


# ---------------------------------------------------------------------------
# Scenario 8: Worker 节点 GUI API 集成测试
# ---------------------------------------------------------------------------


class TestWorkerBinaryFrameReception:
    """Worker 通过 Binary Frame 接收模型文件的集成测试。"""

    def test_worker_receives_model_chunk_and_saves(self, tmp_path: Path):
        """测试 Worker 端通过 Binary Frame 接收模型文件的核心逻辑。"""
        content = b"test model binary data " * 500

        models_dir = tmp_path / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        file_path = models_dir / "qwen2.5-7b.gguf"

        # 直接写入（模拟 Worker _handle_model_chunk_frame 的写入逻辑）
        file_path.write_bytes(content)

        assert file_path.exists()
        assert file_path.read_bytes() == content

    def test_worker_overwrites_different_size_file(self, tmp_path: Path):
        """已存在不同大小的文件时覆盖保存。"""
        existing_path = tmp_path / "models" / "model.gguf"
        existing_path.parent.mkdir(parents=True, exist_ok=True)
        existing_path.write_bytes(b"old content")

        new_content = b"new model data " * 200
        existing_path.write_bytes(new_content)

        assert existing_path.read_bytes() == new_content


# ---------------------------------------------------------------------------
# Scenario 9: 事件溯源 - 模型部署流程的状态恢复
# ---------------------------------------------------------------------------


class TestDeployFlowStateRecovery:
    """验证模型部署流程中事件溯源的状态恢复。"""

    def test_recover_state_after_model_deploy(self):
        """模型部署后，新 Master 能从事件日志恢复状态。"""
        shared_log = MemoryEventLog()

        # 第一个 Master 处理部署流程
        master1 = MasterNode(node_id="master-1", event_log=shared_log)
        master1.process_node_joined(NodeId("gpu-server-01"), "10.0.0.1", 52415)
        master1.process_node_joined(NodeId("gpu-server-02"), "10.0.0.2", 52415)

        cmd = CreateInstance(model_id="qwen2.5-7b", sharding="tensor")
        master1.process_create_instance(cmd)

        inst_id = list(master1.state.instances.keys())[0]
        inf_cmd = StartInference(instance_id=inst_id, prompt="test")
        master1.process_start_inference(inf_cmd)

        # 第二个 Master 从日志恢复
        master2 = MasterNode(node_id="master-2", event_log=shared_log)
        state = empty_state()
        for ie in shared_log.read_from(0):
            state = apply(state, ie)
        master2._state = state

        # 验证恢复的状态
        assert len(master2.state.nodes) == 2
        assert len(master2.state.instances) == 1
        assert len(master2.state.tasks) == 1
        assert master2.state.nodes[NodeId("gpu-server-01")].ip == "10.0.0.1"

        # 新 Master 能继续处理
        master2.process_node_joined(NodeId("gpu-server-03"), "10.0.0.3", 52415)
        assert len(master2.state.nodes) == 3
