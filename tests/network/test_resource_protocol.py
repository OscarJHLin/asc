"""测试 RESOURCE_QUERY/RESPONSE Binary Frame 协议。

覆盖：
- RESOURCE_QUERY/RESOURCE_RESPONSE 消息创建与解析
- ENVELOPE 帧编码/解码（资源查询消息使用 ENVELOPE 帧类型）
- Worker Agent 处理 RESOURCE_QUERY 命令
- Orchestrator 通过 Binary Frame 查询 Worker 资源
- 端到端资源查询流程
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from asc.network.frame import FrameType, decode_frame, encode_frame
from asc.network.protocol import (
    Channel,
    Envelope,
    Message,
    MessageType,
    decode_envelope,
    encode_envelope,
)
from asc.worker.agent import WorkerAgent

# ------------------------------------------------------------------
# 测试：MessageType 枚举
# ------------------------------------------------------------------


class TestResourceMessageType:
    """资源查询消息类型枚举。"""

    def test_resource_query_value(self):
        assert MessageType.RESOURCE_QUERY.value == "resource_query"

    def test_resource_response_value(self):
        assert MessageType.RESOURCE_RESPONSE.value == "resource_response"

    def test_resource_types_are_distinct(self):
        assert MessageType.RESOURCE_QUERY != MessageType.RESOURCE_RESPONSE
        assert MessageType.RESOURCE_QUERY.value != MessageType.RESOURCE_RESPONSE.value


# ------------------------------------------------------------------
# 测试：资源查询消息创建与序列化
# ------------------------------------------------------------------


class TestResourceMessageCreation:
    """资源查询消息创建与序列化。"""

    def test_resource_query_message(self):
        msg = Message(
            type=MessageType.RESOURCE_QUERY,
            sender_id="master",
            payload={"request_id": "req-1", "node_id": "worker-1"},
        )
        assert msg.type == MessageType.RESOURCE_QUERY
        assert msg.sender_id == "master"
        assert msg.payload["request_id"] == "req-1"

    def test_resource_response_message(self):
        msg = Message(
            type=MessageType.RESOURCE_RESPONSE,
            sender_id="worker-1",
            payload={
                "node_id": "worker-1",
                "cpu_count": 8,
                "memory_total_mb": 32768,
                "memory_free_mb": 24000,
                "total_vram_free_mb": 8192,
                "gpus": [
                    {"index": 0, "name": "RTX 4090", "vram_total_mb": 24576, "vram_free_mb": 8192}
                ],
            },
        )
        assert msg.type == MessageType.RESOURCE_RESPONSE
        assert msg.payload["cpu_count"] == 8
        assert len(msg.payload["gpus"]) == 1

    def test_resource_query_envelope_uses_commands_channel(self):
        """资源查询消息应使用 COMMANDS 通道。"""
        env = Envelope(
            channel=Channel.COMMANDS,
            message=Message(
                type=MessageType.RESOURCE_QUERY,
                sender_id="master",
                payload={"request_id": "req-1"},
            ),
        )
        assert env.channel == Channel.COMMANDS

    def test_resource_response_envelope_uses_commands_channel(self):
        env = Envelope(
            channel=Channel.COMMANDS,
            message=Message(
                type=MessageType.RESOURCE_RESPONSE,
                sender_id="worker-1",
                payload={"node_id": "worker-1"},
            ),
        )
        assert env.channel == Channel.COMMANDS


class TestResourceMessageSerialization:
    """资源查询消息序列化/反序列化。"""

    def test_resource_query_roundtrip_msgpack(self):
        env = Envelope(
            channel=Channel.COMMANDS,
            message=Message(
                type=MessageType.RESOURCE_QUERY,
                sender_id="master",
                payload={"request_id": "req-1", "node_id": "worker-1"},
            ),
            target="worker-1",
        )
        data = encode_envelope(env)
        decoded = decode_envelope(data)
        assert decoded.channel == Channel.COMMANDS
        assert decoded.message.type == MessageType.RESOURCE_QUERY
        assert decoded.message.payload["request_id"] == "req-1"
        assert decoded.target == "worker-1"

    def test_resource_response_roundtrip_msgpack(self):
        gpu_data = [
            {
                "index": 0,
                "name": "RTX 4090",
                "vendor": "nvidia",
                "vram_total_mb": 24576,
                "vram_free_mb": 8192,
                "compute_capability": "8.9",
            }
        ]
        env = Envelope(
            channel=Channel.COMMANDS,
            message=Message(
                type=MessageType.RESOURCE_RESPONSE,
                sender_id="worker-1",
                payload={
                    "node_id": "worker-1",
                    "cpu_count": 16,
                    "cpu_brand": "AMD Ryzen 9",
                    "cpu_physical_count": 8,
                    "cpu_freq_mhz": 4800.0,
                    "cpu_percent": 25.5,
                    "memory_total_mb": 65536,
                    "memory_free_mb": 48000,
                    "compute_score": 85.0,
                    "disk_free_mb": 500000,
                    "network_mbps": 10000.0,
                    "total_vram_free_mb": 8192,
                    "gpus": gpu_data,
                },
            ),
        )
        data = encode_envelope(env)
        decoded = decode_envelope(data)
        assert decoded.message.type == MessageType.RESOURCE_RESPONSE
        assert decoded.message.payload["cpu_count"] == 16
        assert decoded.message.payload["total_vram_free_mb"] == 8192
        assert len(decoded.message.payload["gpus"]) == 1
        assert decoded.message.payload["gpus"][0]["name"] == "RTX 4090"


# ------------------------------------------------------------------
# 测试：资源查询消息作为 ENVELOPE 帧传输
# ------------------------------------------------------------------


class TestResourceFrameTransmission:
    """资源查询消息通过 ENVELOPE 帧传输。"""

    def test_resource_query_as_envelope_frame(self):
        """RESOURCE_QUERY 使用 ENVELOPE 帧类型（0x00）。"""
        from asc.network.frame import encode_envelope_frame

        env = Envelope(
            channel=Channel.COMMANDS,
            message=Message(
                type=MessageType.RESOURCE_QUERY,
                sender_id="master",
                payload={"request_id": "req-1"},
            ),
        )
        frame = encode_envelope_frame(env)
        assert frame.frame_type == FrameType.ENVELOPE

    def test_resource_query_frame_roundtrip(self):
        from asc.network.frame import decode_envelope_frame, encode_envelope_frame

        env = Envelope(
            channel=Channel.COMMANDS,
            message=Message(
                type=MessageType.RESOURCE_QUERY,
                sender_id="master",
                payload={"request_id": "req-1", "node_id": "worker-1"},
            ),
            target="worker-1",
        )
        frame = encode_envelope_frame(env)
        data = encode_frame(frame)
        decoded_frame = decode_frame(data)

        assert decoded_frame.frame_type == FrameType.ENVELOPE
        decoded_env = decode_envelope_frame(decoded_frame)
        assert decoded_env.message.type == MessageType.RESOURCE_QUERY
        assert decoded_env.message.payload["request_id"] == "req-1"

    def test_resource_response_frame_roundtrip(self):
        from asc.network.frame import decode_envelope_frame, encode_envelope_frame

        env = Envelope(
            channel=Channel.COMMANDS,
            message=Message(
                type=MessageType.RESOURCE_RESPONSE,
                sender_id="worker-1",
                payload={
                    "node_id": "worker-1",
                    "cpu_count": 8,
                    "memory_total_mb": 32768,
                    "total_vram_free_mb": 8192,
                },
            ),
        )
        frame = encode_envelope_frame(env)
        data = encode_frame(frame)
        decoded_frame = decode_frame(data)

        decoded_env = decode_envelope_frame(decoded_frame)
        assert decoded_env.message.type == MessageType.RESOURCE_RESPONSE
        assert decoded_env.message.payload["cpu_count"] == 8
        assert decoded_env.message.payload["total_vram_free_mb"] == 8192


# ------------------------------------------------------------------
# 测试：Worker Agent 处理 RESOURCE_QUERY
# ------------------------------------------------------------------


class TestWorkerAgentResourceHandler:
    """Worker Agent 处理 RESOURCE_QUERY 命令。"""

    @pytest.mark.asyncio
    async def test_handle_resource_query_sends_response(self):
        """收到 RESOURCE_QUERY 后应回复 RESOURCE_RESPONSE。"""
        agent = WorkerAgent(node_id="worker-1")

        sent_envelopes: list[Envelope] = []
        mock_client = MagicMock()

        async def mock_send(envelope: Envelope) -> bool:
            sent_envelopes.append(envelope)
            return True

        mock_client.send = mock_send
        agent._client = mock_client

        # Mock hardware detection
        agent._hardware_detector = MagicMock()
        agent._hardware_detector.detect_cpu = MagicMock(return_value=MagicMock(
            physical_count=8, freq_mhz=4800.0, brand="AMD Ryzen 9"
        ))
        agent._hardware_detector.detect_gpus = MagicMock(return_value=[])
        agent._hardware_detector.detect_disk = MagicMock(return_value=MagicMock(free_mb=500000))
        agent._hardware_detector.detect_network = MagicMock(return_value=MagicMock(estimated_mbps=10000.0))

        # Mock benchmark
        agent._benchmark_score = MagicMock()
        agent._benchmark_score.run_benchmark = MagicMock(return_value=MagicMock(relative_score=85.0))

        envelope = Envelope(
            channel=Channel.COMMANDS,
            message=Message(
                type=MessageType.RESOURCE_QUERY,
                sender_id="master",
                payload={"request_id": "req-1", "node_id": "worker-1"},
            ),
        )

        await agent._handle_resource_query(envelope)

        assert len(sent_envelopes) == 1
        response = sent_envelopes[0]
        assert response.message.type == MessageType.RESOURCE_RESPONSE
        assert response.message.payload["node_id"] == "worker-1"
        assert "cpu_count" in response.message.payload
        assert "memory_total_mb" in response.message.payload
        assert "total_vram_free_mb" in response.message.payload
        assert "gpus" in response.message.payload

    @pytest.mark.asyncio
    async def test_handle_resource_query_no_client(self):
        """没有 TCP 连接时不应崩溃。"""
        agent = WorkerAgent(node_id="worker-1")
        agent._client = None

        envelope = Envelope(
            channel=Channel.COMMANDS,
            message=Message(
                type=MessageType.RESOURCE_QUERY,
                sender_id="master",
                payload={},
            ),
        )

        # 不应抛出异常
        await agent._handle_resource_query(envelope)

    @pytest.mark.asyncio
    async def test_on_message_dispatches_resource_query(self):
        """_on_message 应正确分派 RESOURCE_QUERY 消息。"""
        agent = WorkerAgent(node_id="worker-1")

        sent_envelopes: list[Envelope] = []
        mock_client = MagicMock()

        async def mock_send(envelope: Envelope) -> bool:
            sent_envelopes.append(envelope)
            return True

        mock_client.send = mock_send
        agent._client = mock_client

        # Mock hardware
        agent._hardware_detector = MagicMock()
        agent._hardware_detector.detect_cpu = MagicMock(return_value=MagicMock(
            physical_count=4, freq_mhz=3000.0, brand="Intel"
        ))
        agent._hardware_detector.detect_gpus = MagicMock(return_value=[])
        agent._hardware_detector.detect_disk = MagicMock(return_value=MagicMock(free_mb=100000))
        agent._hardware_detector.detect_network = MagicMock(return_value=MagicMock(estimated_mbps=1000.0))
        agent._benchmark_score = MagicMock()
        agent._benchmark_score.run_benchmark = MagicMock(return_value=MagicMock(relative_score=50.0))

        envelope = Envelope(
            channel=Channel.COMMANDS,
            message=Message(
                type=MessageType.RESOURCE_QUERY,
                sender_id="master",
                payload={},
            ),
        )

        await agent._on_message(envelope)
        assert len(sent_envelopes) == 1
        assert sent_envelopes[0].message.type == MessageType.RESOURCE_RESPONSE


# ------------------------------------------------------------------
# 测试：Orchestrator 通过 Binary Frame 查询资源
# ------------------------------------------------------------------


class TestOrchestratorResourceQuery:
    """Orchestrator 通过 Binary Frame 协议查询 Worker 资源。"""

    @pytest.mark.asyncio
    async def test_query_resources_sends_envelope(self):
        """query_resources 应通过 TCPServer 发送 RESOURCE_QUERY ENVELOPE 帧。"""
        from asc.master.orchestrator import DistributedOrchestrator
        from asc.types import NodeId

        sent_envelopes: list[tuple[str, Envelope]] = []

        mock_tcp = MagicMock()

        async def mock_send(conn_id: str, envelope: Envelope) -> bool:
            sent_envelopes.append((conn_id, envelope))
            request_id = envelope.message.payload.get("request_id", "")
            response = Envelope(
                channel=Channel.COMMANDS,
                message=Message(
                    type=MessageType.RESOURCE_RESPONSE,
                    sender_id="worker-1",
                    payload={
                        "request_id": request_id,
                        "node_id": "worker-1",
                        "cpu_count": 8,
                        "memory_total_mb": 32768,
                        "total_vram_free_mb": 8192,
                    },
                ),
            )
            await orch.handle_resource_response(response)
            return True

        mock_tcp.send = mock_send

        orch = DistributedOrchestrator(
            tcp_server=mock_tcp,
            conn_node_map={"conn-1": "worker-1"},
        )

        result = await orch.query_resources(NodeId("worker-1"), timeout=5.0)

        assert result is not None
        assert result["cpu_count"] == 8
        assert result["total_vram_free_mb"] == 8192
        assert len(sent_envelopes) == 1
        conn_id, env = sent_envelopes[0]
        assert conn_id == "conn-1"
        assert env.message.type == MessageType.RESOURCE_QUERY
        assert env.channel == Channel.COMMANDS

    @pytest.mark.asyncio
    async def test_query_resources_no_connection(self):
        """找不到节点连接时应返回 None。"""
        from asc.master.orchestrator import DistributedOrchestrator
        from asc.types import NodeId

        orch = DistributedOrchestrator(tcp_server=MagicMock())

        result = await orch.query_resources(NodeId("worker-1"), timeout=1.0)
        assert result is None

    @pytest.mark.asyncio
    async def test_query_resources_timeout(self):
        """等待响应超时时应返回 None。"""
        from asc.master.orchestrator import DistributedOrchestrator
        from asc.types import NodeId

        mock_tcp = MagicMock()

        async def mock_send(conn_id: str, envelope: Envelope) -> bool:
            # 不回复，模拟超时
            return True

        mock_tcp.send = mock_send

        orch = DistributedOrchestrator(
            tcp_server=mock_tcp,
            conn_node_map={"conn-1": "worker-1"},
        )

        result = await orch.query_resources(NodeId("worker-1"), timeout=0.1)
        assert result is None


# ------------------------------------------------------------------
# 测试：端到端资源查询流程（Mock TCP）
# ------------------------------------------------------------------


class TestResourceQueryE2E:
    """端到端资源查询流程测试。"""

    @pytest.mark.asyncio
    async def test_full_resource_query_flow(self):
        """完整的资源查询流程：Master 发送 RESOURCE_QUERY -> Worker 处理 -> 回复 RESPONSE。"""
        # Worker 侧
        agent = WorkerAgent(node_id="worker-1")
        worker_sent: list[Envelope] = []

        mock_client = MagicMock()

        async def mock_send(envelope: Envelope) -> bool:
            worker_sent.append(envelope)
            return True

        mock_client.send = mock_send
        agent._client = mock_client

        # Mock hardware
        agent._hardware_detector = MagicMock()
        agent._hardware_detector.detect_cpu = MagicMock(return_value=MagicMock(
            physical_count=8, freq_mhz=4800.0, brand="AMD Ryzen 9"
        ))
        agent._hardware_detector.detect_gpus = MagicMock(return_value=[])
        agent._hardware_detector.detect_disk = MagicMock(return_value=MagicMock(free_mb=500000))
        agent._hardware_detector.detect_network = MagicMock(return_value=MagicMock(estimated_mbps=10000.0))
        agent._benchmark_score = MagicMock()
        agent._benchmark_score.run_benchmark = MagicMock(return_value=MagicMock(relative_score=85.0))

        # Master 侧
        from asc.master.orchestrator import DistributedOrchestrator
        from asc.types import NodeId

        master_sent: list[tuple[str, Envelope]] = []

        mock_tcp = MagicMock()

        async def mock_tcp_send(conn_id: str, envelope: Envelope) -> bool:
            master_sent.append((conn_id, envelope))
            # 模拟：Master 发送的消息被 Worker 接收处理
            await agent._on_message(envelope)
            # Worker 的回复被 Orchestrator 接收
            if worker_sent:
                response = worker_sent[-1]
                if response.message.type == MessageType.RESOURCE_RESPONSE:
                    request_id = envelope.message.payload.get("request_id", "")
                    response_with_id = Envelope(
                        channel=response.channel,
                        message=Message(
                            type=response.message.type,
                            sender_id=response.message.sender_id,
                            payload={**response.message.payload, "request_id": request_id},
                        ),
                    )
                    await orch.handle_resource_response(response_with_id)
            return True

        mock_tcp.send = mock_tcp_send

        orch = DistributedOrchestrator(
            tcp_server=mock_tcp,
            conn_node_map={"conn-1": "worker-1"},
        )

        result = await orch.query_resources(NodeId("worker-1"), timeout=5.0)

        # 验证端到端流程
        assert result is not None
        assert result["node_id"] == "worker-1"
        assert "cpu_count" in result
        assert "memory_total_mb" in result
        assert len(master_sent) == 1
        assert master_sent[0][1].message.type == MessageType.RESOURCE_QUERY
        assert len(worker_sent) == 1
        assert worker_sent[0].message.type == MessageType.RESOURCE_RESPONSE

    @pytest.mark.asyncio
    async def test_no_http_used_in_resource_query(self):
        """验证资源查询不再使用 httpx。"""
        from asc.master.orchestrator import DistributedOrchestrator
        from asc.types import NodeId

        mock_tcp = MagicMock()

        async def mock_send(conn_id: str, envelope: Envelope) -> bool:
            request_id = envelope.message.payload.get("request_id", "")
            response = Envelope(
                channel=Channel.COMMANDS,
                message=Message(
                    type=MessageType.RESOURCE_RESPONSE,
                    sender_id="worker-1",
                    payload={
                        "request_id": request_id,
                        "node_id": "worker-1",
                        "cpu_count": 8,
                        "total_vram_free_mb": 8192,
                    },
                ),
            )
            await orch.handle_resource_response(response)
            return True

        mock_tcp.send = mock_send

        orch = DistributedOrchestrator(
            tcp_server=mock_tcp,
            conn_node_map={"conn-1": "worker-1"},
        )

        with patch("httpx.AsyncClient", side_effect=AssertionError("httpx should not be used")):
            result = await orch.query_resources(NodeId("worker-1"), timeout=5.0)
            assert result is not None
            assert result["cpu_count"] == 8


# ------------------------------------------------------------------
# 测试：CONFIG_UPDATE 消息处理
# ------------------------------------------------------------------


class TestConfigUpdateHandler:
    """CONFIG_UPDATE 消息处理（确保通过 Binary Frame 工作）。"""

    @pytest.mark.asyncio
    async def test_handle_config_update(self):
        """收到 CONFIG_UPDATE 后应更新集群配置。"""
        agent = WorkerAgent(node_id="worker-1")

        # Mock hardware for get_resources
        agent._hardware_detector = MagicMock()
        agent._hardware_detector.detect_cpu = MagicMock(return_value=MagicMock(
            physical_count=8, freq_mhz=4800.0, brand="AMD"
        ))
        agent._hardware_detector.detect_gpus = MagicMock(return_value=[
            MagicMock(index=0, name="GPU", vendor="nvidia", vram_total_mb=24576, vram_free_mb=8192, compute_capability="8.9")
        ])
        agent._hardware_detector.detect_disk = MagicMock(return_value=MagicMock(free_mb=500000))
        agent._hardware_detector.detect_network = MagicMock(return_value=MagicMock(estimated_mbps=10000.0))
        agent._benchmark_score = MagicMock()
        agent._benchmark_score.run_benchmark = MagicMock(return_value=MagicMock(relative_score=85.0))

        envelope = Envelope(
            channel=Channel.COMMANDS,
            message=Message(
                type=MessageType.CONFIG_UPDATE,
                sender_id="master",
                payload={
                    "vram_limit_percent": 80,
                    "memory_limit_percent": 90,
                    "offload_ratio": 0.3,
                },
            ),
        )

        await agent._handle_config_update(envelope)

        # 验证配置已更新
        policy = agent._cluster_config.get_node_policy("worker-1")
        assert policy.vram_limit_mb is not None
        assert policy.memory_limit_mb is not None
        assert abs(policy.memory_offload_ratio - 0.3) < 0.01

    @pytest.mark.asyncio
    async def test_on_message_dispatches_config_update(self):
        """_on_message 应正确分派 CONFIG_UPDATE 消息。"""
        agent = WorkerAgent(node_id="worker-1")

        # Mock hardware
        agent._hardware_detector = MagicMock()
        agent._hardware_detector.detect_cpu = MagicMock(return_value=MagicMock(
            physical_count=4, freq_mhz=3000.0, brand="Intel"
        ))
        agent._hardware_detector.detect_gpus = MagicMock(return_value=[])
        agent._hardware_detector.detect_disk = MagicMock(return_value=MagicMock(free_mb=100000))
        agent._hardware_detector.detect_network = MagicMock(return_value=MagicMock(estimated_mbps=1000.0))
        agent._benchmark_score = MagicMock()
        agent._benchmark_score.run_benchmark = MagicMock(return_value=MagicMock(relative_score=50.0))

        envelope = Envelope(
            channel=Channel.COMMANDS,
            message=Message(
                type=MessageType.CONFIG_UPDATE,
                sender_id="master",
                payload={"vram_limit_percent": 50},
            ),
        )

        # 不应抛出异常
        await agent._on_message(envelope)

        policy = agent._cluster_config.get_node_policy("worker-1")
        assert policy.vram_limit_mb is not None
