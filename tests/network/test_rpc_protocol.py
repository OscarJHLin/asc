"""测试 RPC_START/STOP Binary Frame 协议。

覆盖：
- RPC_START/RPC_STOP/RPC_START_ACK/RPC_STOP_ACK 消息创建与解析
- ENVELOPE 帧编码/解码（RPC 控制消息使用 ENVELOPE 帧类型）
- Worker Agent 处理 RPC_START/RPC_STOP 命令
- Orchestrator 通过 Binary Frame 发送 RPC 命令并等待 ACK
- 端到端 RPC 控制流程
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


class TestRpcMessageType:
    """RPC 控制消息类型枚举。"""

    def test_rpc_start_value(self):
        assert MessageType.RPC_START.value == "rpc_start"

    def test_rpc_stop_value(self):
        assert MessageType.RPC_STOP.value == "rpc_stop"

    def test_rpc_start_ack_value(self):
        assert MessageType.RPC_START_ACK.value == "rpc_start_ack"

    def test_rpc_stop_ack_value(self):
        assert MessageType.RPC_STOP_ACK.value == "rpc_stop_ack"

    def test_all_rpc_types_are_distinct(self):
        rpc_types = [
            MessageType.RPC_START,
            MessageType.RPC_STOP,
            MessageType.RPC_START_ACK,
            MessageType.RPC_STOP_ACK,
        ]
        values = [t.value for t in rpc_types]
        assert len(values) == len(set(values))


# ------------------------------------------------------------------
# 测试：RPC 消息创建与序列化
# ------------------------------------------------------------------


class TestRpcMessageCreation:
    """RPC 控制消息创建与序列化。"""

    def test_rpc_start_message(self):
        msg = Message(
            type=MessageType.RPC_START,
            sender_id="master",
            payload={"request_id": "req-1", "node_id": "worker-1"},
        )
        assert msg.type == MessageType.RPC_START
        assert msg.sender_id == "master"
        assert msg.payload["request_id"] == "req-1"

    def test_rpc_start_ack_message(self):
        msg = Message(
            type=MessageType.RPC_START_ACK,
            sender_id="worker-1",
            payload={"status": "ok", "endpoint": "10.0.0.2:50052", "port": 50052},
        )
        assert msg.type == MessageType.RPC_START_ACK
        assert msg.payload["status"] == "ok"
        assert msg.payload["endpoint"] == "10.0.0.2:50052"

    def test_rpc_stop_message(self):
        msg = Message(
            type=MessageType.RPC_STOP,
            sender_id="master",
            payload={"request_id": "req-2", "node_id": "worker-1"},
        )
        assert msg.type == MessageType.RPC_STOP

    def test_rpc_stop_ack_message(self):
        msg = Message(
            type=MessageType.RPC_STOP_ACK,
            sender_id="worker-1",
            payload={"status": "ok"},
        )
        assert msg.type == MessageType.RPC_STOP_ACK
        assert msg.payload["status"] == "ok"

    def test_rpc_start_envelope_uses_commands_channel(self):
        """RPC 控制消息应使用 COMMANDS 通道。"""
        env = Envelope(
            channel=Channel.COMMANDS,
            message=Message(
                type=MessageType.RPC_START,
                sender_id="master",
                payload={"request_id": "req-1"},
            ),
        )
        assert env.channel == Channel.COMMANDS

    def test_rpc_start_ack_envelope_uses_commands_channel(self):
        env = Envelope(
            channel=Channel.COMMANDS,
            message=Message(
                type=MessageType.RPC_START_ACK,
                sender_id="worker-1",
                payload={"status": "ok", "endpoint": "0.0.0.0:50052"},
            ),
        )
        assert env.channel == Channel.COMMANDS


class TestRpcMessageSerialization:
    """RPC 消息序列化/反序列化。"""

    def test_rpc_start_roundtrip_msgpack(self):
        """MessagePack 序列化往返。"""
        env = Envelope(
            channel=Channel.COMMANDS,
            message=Message(
                type=MessageType.RPC_START,
                sender_id="master",
                payload={"request_id": "req-1", "node_id": "worker-1"},
            ),
            target="worker-1",
        )
        data = encode_envelope(env)
        decoded = decode_envelope(data)
        assert decoded.channel == Channel.COMMANDS
        assert decoded.message.type == MessageType.RPC_START
        assert decoded.message.sender_id == "master"
        assert decoded.message.payload["request_id"] == "req-1"
        assert decoded.target == "worker-1"

    def test_rpc_start_ack_roundtrip_msgpack(self):
        env = Envelope(
            channel=Channel.COMMANDS,
            message=Message(
                type=MessageType.RPC_START_ACK,
                sender_id="worker-1",
                payload={"status": "ok", "endpoint": "10.0.0.2:50052", "port": 50052},
            ),
        )
        data = encode_envelope(env)
        decoded = decode_envelope(data)
        assert decoded.message.type == MessageType.RPC_START_ACK
        assert decoded.message.payload["endpoint"] == "10.0.0.2:50052"

    def test_rpc_stop_roundtrip_msgpack(self):
        env = Envelope(
            channel=Channel.COMMANDS,
            message=Message(
                type=MessageType.RPC_STOP,
                sender_id="master",
                payload={"request_id": "req-2", "node_id": "worker-1"},
            ),
        )
        data = encode_envelope(env)
        decoded = decode_envelope(data)
        assert decoded.message.type == MessageType.RPC_STOP

    def test_rpc_stop_ack_roundtrip_msgpack(self):
        env = Envelope(
            channel=Channel.COMMANDS,
            message=Message(
                type=MessageType.RPC_STOP_ACK,
                sender_id="worker-1",
                payload={"status": "ok"},
            ),
        )
        data = encode_envelope(env)
        decoded = decode_envelope(data)
        assert decoded.message.type == MessageType.RPC_STOP_ACK
        assert decoded.message.payload["status"] == "ok"


# ------------------------------------------------------------------
# 测试：RPC 消息作为 ENVELOPE 帧传输
# ------------------------------------------------------------------


class TestRpcFrameTransmission:
    """RPC 控制消息通过 ENVELOPE 帧传输。"""

    def test_rpc_start_as_envelope_frame(self):
        """RPC_START 使用 ENVELOPE 帧类型（0x00），不是专用帧类型。"""
        from asc.network.frame import encode_envelope_frame

        env = Envelope(
            channel=Channel.COMMANDS,
            message=Message(
                type=MessageType.RPC_START,
                sender_id="master",
                payload={"request_id": "req-1"},
            ),
        )
        frame = encode_envelope_frame(env)
        assert frame.frame_type == FrameType.ENVELOPE

    def test_rpc_start_frame_roundtrip(self):
        """RPC_START 消息通过 Frame 编解码完整往返。"""
        from asc.network.frame import decode_envelope_frame, encode_envelope_frame

        env = Envelope(
            channel=Channel.COMMANDS,
            message=Message(
                type=MessageType.RPC_START,
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
        assert decoded_env.message.type == MessageType.RPC_START
        assert decoded_env.message.payload["request_id"] == "req-1"
        assert decoded_env.target == "worker-1"

    def test_rpc_start_ack_frame_roundtrip(self):
        from asc.network.frame import decode_envelope_frame, encode_envelope_frame

        env = Envelope(
            channel=Channel.COMMANDS,
            message=Message(
                type=MessageType.RPC_START_ACK,
                sender_id="worker-1",
                payload={"status": "ok", "endpoint": "10.0.0.2:50052", "port": 50052},
            ),
        )
        frame = encode_envelope_frame(env)
        data = encode_frame(frame)
        decoded_frame = decode_frame(data)

        decoded_env = decode_envelope_frame(decoded_frame)
        assert decoded_env.message.type == MessageType.RPC_START_ACK
        assert decoded_env.message.payload["endpoint"] == "10.0.0.2:50052"

    def test_rpc_stop_frame_roundtrip(self):
        from asc.network.frame import decode_envelope_frame, encode_envelope_frame

        env = Envelope(
            channel=Channel.COMMANDS,
            message=Message(
                type=MessageType.RPC_STOP,
                sender_id="master",
                payload={"request_id": "req-2", "node_id": "worker-1"},
            ),
        )
        frame = encode_envelope_frame(env)
        data = encode_frame(frame)
        decoded_frame = decode_frame(data)

        decoded_env = decode_envelope_frame(decoded_frame)
        assert decoded_env.message.type == MessageType.RPC_STOP

    def test_rpc_stop_ack_frame_roundtrip(self):
        from asc.network.frame import decode_envelope_frame, encode_envelope_frame

        env = Envelope(
            channel=Channel.COMMANDS,
            message=Message(
                type=MessageType.RPC_STOP_ACK,
                sender_id="worker-1",
                payload={"status": "ok"},
            ),
        )
        frame = encode_envelope_frame(env)
        data = encode_frame(frame)
        decoded_frame = decode_frame(data)

        decoded_env = decode_envelope_frame(decoded_frame)
        assert decoded_env.message.type == MessageType.RPC_STOP_ACK


# ------------------------------------------------------------------
# 测试：Worker Agent 处理 RPC_START/RPC_STOP
# ------------------------------------------------------------------


class TestWorkerAgentRpcHandlers:
    """Worker Agent 处理 RPC 控制命令。"""

    @pytest.mark.asyncio
    async def test_handle_rpc_start_sends_ack(self):
        """收到 RPC_START 后应启动 RPC Server 并回复 RPC_START_ACK。"""
        agent = WorkerAgent(node_id="worker-1")

        # Mock client
        sent_envelopes: list[Envelope] = []
        mock_client = MagicMock()

        async def mock_send(envelope: Envelope) -> bool:
            sent_envelopes.append(envelope)
            return True

        mock_client.send = mock_send
        agent._client = mock_client

        # Mock RPC server
        agent._rpc_server = MagicMock()
        agent._rpc_server.start = MagicMock(return_value=50052)
        agent._rpc_server.endpoint = "0.0.0.0:50052"

        # 构造 RPC_START 消息
        envelope = Envelope(
            channel=Channel.COMMANDS,
            message=Message(
                type=MessageType.RPC_START,
                sender_id="master",
                payload={"request_id": "req-1", "node_id": "worker-1"},
            ),
        )

        await agent._handle_rpc_start(envelope)

        # 验证发送了 RPC_START_ACK
        assert len(sent_envelopes) == 1
        ack = sent_envelopes[0]
        assert ack.message.type == MessageType.RPC_START_ACK
        assert ack.message.payload["status"] == "ok"
        assert ack.message.payload["endpoint"] == "0.0.0.0:50052"
        assert ack.message.payload["port"] == 50052

    @pytest.mark.asyncio
    async def test_handle_rpc_start_error_sends_error_ack(self):
        """RPC 启动失败时应回复包含错误信息的 RPC_START_ACK。"""
        agent = WorkerAgent(node_id="worker-1")

        sent_envelopes: list[Envelope] = []
        mock_client = MagicMock()

        async def mock_send(envelope: Envelope) -> bool:
            sent_envelopes.append(envelope)
            return True

        mock_client.send = mock_send
        agent._client = mock_client

        # Mock RPC server 启动失败
        agent._rpc_server = MagicMock()
        agent._rpc_server.start = MagicMock(side_effect=FileNotFoundError("未找到 rpc-server"))

        envelope = Envelope(
            channel=Channel.COMMANDS,
            message=Message(
                type=MessageType.RPC_START,
                sender_id="master",
                payload={"request_id": "req-1"},
            ),
        )

        await agent._handle_rpc_start(envelope)

        assert len(sent_envelopes) == 1
        ack = sent_envelopes[0]
        assert ack.message.type == MessageType.RPC_START_ACK
        assert ack.message.payload["status"] == "error"
        assert "未找到 rpc-server" in ack.message.payload["error"]

    @pytest.mark.asyncio
    async def test_handle_rpc_stop_sends_ack(self):
        """收到 RPC_STOP 后应停止 RPC Server 并回复 RPC_STOP_ACK。"""
        agent = WorkerAgent(node_id="worker-1")

        sent_envelopes: list[Envelope] = []
        mock_client = MagicMock()

        async def mock_send(envelope: Envelope) -> bool:
            sent_envelopes.append(envelope)
            return True

        mock_client.send = mock_send
        agent._client = mock_client

        # Mock RPC server
        agent._rpc_server = MagicMock()
        agent._rpc_server.stop = MagicMock()

        envelope = Envelope(
            channel=Channel.COMMANDS,
            message=Message(
                type=MessageType.RPC_STOP,
                sender_id="master",
                payload={"request_id": "req-2", "node_id": "worker-1"},
            ),
        )

        await agent._handle_rpc_stop(envelope)

        assert len(sent_envelopes) == 1
        ack = sent_envelopes[0]
        assert ack.message.type == MessageType.RPC_STOP_ACK
        assert ack.message.payload["status"] == "ok"
        agent._rpc_server.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_rpc_start_no_client(self):
        """没有 TCP 连接时不应崩溃。"""
        agent = WorkerAgent(node_id="worker-1")
        agent._client = None

        agent._rpc_server = MagicMock()
        agent._rpc_server.start = MagicMock(return_value=50052)
        agent._rpc_server.endpoint = "0.0.0.0:50052"

        envelope = Envelope(
            channel=Channel.COMMANDS,
            message=Message(
                type=MessageType.RPC_START,
                sender_id="master",
                payload={},
            ),
        )

        # 不应抛出异常
        await agent._handle_rpc_start(envelope)

    @pytest.mark.asyncio
    async def test_on_message_dispatches_rpc_start(self):
        """_on_message 应正确分派 RPC_START 消息。"""
        agent = WorkerAgent(node_id="worker-1")

        sent_envelopes: list[Envelope] = []
        mock_client = MagicMock()

        async def mock_send(envelope: Envelope) -> bool:
            sent_envelopes.append(envelope)
            return True

        mock_client.send = mock_send
        agent._client = mock_client
        agent._rpc_server = MagicMock()
        agent._rpc_server.start = MagicMock(return_value=50052)
        agent._rpc_server.endpoint = "0.0.0.0:50052"

        envelope = Envelope(
            channel=Channel.COMMANDS,
            message=Message(
                type=MessageType.RPC_START,
                sender_id="master",
                payload={},
            ),
        )

        await agent._on_message(envelope)
        assert len(sent_envelopes) == 1
        assert sent_envelopes[0].message.type == MessageType.RPC_START_ACK

    @pytest.mark.asyncio
    async def test_on_message_dispatches_rpc_stop(self):
        """_on_message 应正确分派 RPC_STOP 消息。"""
        agent = WorkerAgent(node_id="worker-1")

        sent_envelopes: list[Envelope] = []
        mock_client = MagicMock()

        async def mock_send(envelope: Envelope) -> bool:
            sent_envelopes.append(envelope)
            return True

        mock_client.send = mock_send
        agent._client = mock_client
        agent._rpc_server = MagicMock()
        agent._rpc_server.stop = MagicMock()

        envelope = Envelope(
            channel=Channel.COMMANDS,
            message=Message(
                type=MessageType.RPC_STOP,
                sender_id="master",
                payload={},
            ),
        )

        await agent._on_message(envelope)
        assert len(sent_envelopes) == 1
        assert sent_envelopes[0].message.type == MessageType.RPC_STOP_ACK


# ------------------------------------------------------------------
# 测试：Orchestrator 通过 Binary Frame 发送 RPC 命令
# ------------------------------------------------------------------


class TestOrchestratorRpcBinaryFrame:
    """Orchestrator 通过 Binary Frame 协议发送 RPC 控制命令。"""

    @pytest.mark.asyncio
    async def test_request_rpc_start_sends_envelope(self):
        """_request_rpc_start 应通过 TCPServer 发送 ENVELOPE 帧。"""
        from asc.master.orchestrator import DistributedOrchestrator
        from asc.types import NodeId
        from asc.types.state import NodeInfo

        sent_envelopes: list[tuple[str, Envelope]] = []

        mock_tcp = MagicMock()

        async def mock_send(conn_id: str, envelope: Envelope) -> bool:
            sent_envelopes.append((conn_id, envelope))
            # 模拟 Worker 立即回复 ACK
            request_id = envelope.message.payload.get("request_id", "")
            ack_envelope = Envelope(
                channel=Channel.COMMANDS,
                message=Message(
                    type=MessageType.RPC_START_ACK,
                    sender_id="worker-1",
                    payload={
                        "request_id": request_id,
                        "status": "ok",
                        "endpoint": "10.0.0.2:50052",
                        "port": 50052,
                    },
                ),
            )
            await orch.handle_rpc_start_ack(ack_envelope)
            return True

        mock_tcp.send = mock_send

        orch = DistributedOrchestrator(
            tcp_server=mock_tcp,
            conn_node_map={"conn-1": "worker-1"},
        )

        node_info = NodeInfo(
            node_id=NodeId("worker-1"),
            ip="10.0.0.2",
            port=52415,
            runner_status="online",
        )

        endpoint = await orch._request_rpc_start(node_info, timeout=5.0)

        assert endpoint == "10.0.0.2:50052"
        assert len(sent_envelopes) == 1
        conn_id, env = sent_envelopes[0]
        assert conn_id == "conn-1"
        assert env.message.type == MessageType.RPC_START
        assert env.channel == Channel.COMMANDS

    @pytest.mark.asyncio
    async def test_request_rpc_start_no_connection(self):
        """找不到节点连接时应返回 None。"""
        from asc.master.orchestrator import DistributedOrchestrator
        from asc.types import NodeId
        from asc.types.state import NodeInfo

        orch = DistributedOrchestrator(tcp_server=MagicMock())

        node_info = NodeInfo(
            node_id=NodeId("worker-1"),
            ip="10.0.0.2",
            port=52415,
            runner_status="online",
        )

        result = await orch._request_rpc_start(node_info, timeout=1.0)
        assert result is None

    @pytest.mark.asyncio
    async def test_request_rpc_start_error_response(self):
        """Worker 返回错误时应返回 None。"""
        from asc.master.orchestrator import DistributedOrchestrator
        from asc.types import NodeId
        from asc.types.state import NodeInfo

        mock_tcp = MagicMock()

        async def mock_send(conn_id: str, envelope: Envelope) -> bool:
            request_id = envelope.message.payload.get("request_id", "")
            ack_envelope = Envelope(
                channel=Channel.COMMANDS,
                message=Message(
                    type=MessageType.RPC_START_ACK,
                    sender_id="worker-1",
                    payload={
                        "request_id": request_id,
                        "status": "error",
                        "error": "rpc-server not found",
                    },
                ),
            )
            await orch.handle_rpc_start_ack(ack_envelope)
            return True

        mock_tcp.send = mock_send

        orch = DistributedOrchestrator(
            tcp_server=mock_tcp,
            conn_node_map={"conn-1": "worker-1"},
        )

        node_info = NodeInfo(
            node_id=NodeId("worker-1"),
            ip="10.0.0.2",
            port=52415,
            runner_status="online",
        )

        result = await orch._request_rpc_start(node_info, timeout=5.0)
        assert result is None

    @pytest.mark.asyncio
    async def test_stop_worker_rpc_server_sends_envelope(self):
        """_stop_worker_rpc_server 应通过 TCPServer 发送 RPC_STOP ENVELOPE 帧。"""
        from asc.master.orchestrator import DistributedOrchestrator
        from asc.types import NodeId
        from asc.types.state import NodeInfo

        sent_envelopes: list[tuple[str, Envelope]] = []

        mock_tcp = MagicMock()

        async def mock_send(conn_id: str, envelope: Envelope) -> bool:
            sent_envelopes.append((conn_id, envelope))
            request_id = envelope.message.payload.get("request_id", "")
            ack_envelope = Envelope(
                channel=Channel.COMMANDS,
                message=Message(
                    type=MessageType.RPC_STOP_ACK,
                    sender_id="worker-1",
                    payload={"request_id": request_id, "status": "ok"},
                ),
            )
            await orch.handle_rpc_stop_ack(ack_envelope)
            return True

        mock_tcp.send = mock_send

        orch = DistributedOrchestrator(
            tcp_server=mock_tcp,
            conn_node_map={"conn-1": "worker-1"},
        )

        node_info = NodeInfo(
            node_id=NodeId("worker-1"),
            ip="10.0.0.2",
            port=52415,
            runner_status="online",
        )

        await orch._stop_worker_rpc_server(NodeId("worker-1"), node_info)

        assert len(sent_envelopes) == 1
        conn_id, env = sent_envelopes[0]
        assert conn_id == "conn-1"
        assert env.message.type == MessageType.RPC_STOP
        assert env.channel == Channel.COMMANDS

    @pytest.mark.asyncio
    async def test_stop_worker_rpc_server_no_connection(self):
        """找不到节点连接时不应崩溃。"""
        from asc.master.orchestrator import DistributedOrchestrator
        from asc.types import NodeId
        from asc.types.state import NodeInfo

        orch = DistributedOrchestrator(tcp_server=MagicMock())

        node_info = NodeInfo(
            node_id=NodeId("worker-1"),
            ip="10.0.0.2",
            port=52415,
            runner_status="online",
        )

        # 不应抛出异常
        await orch._stop_worker_rpc_server(NodeId("worker-1"), node_info)


# ------------------------------------------------------------------
# 测试：端到端 RPC 控制流程（Mock TCP）
# ------------------------------------------------------------------


class TestRpcControlE2E:
    """端到端 RPC 控制流程测试。"""

    @pytest.mark.asyncio
    async def test_full_rpc_start_flow(self):
        """完整的 RPC 启动流程：Master 发送 RPC_START -> Worker 处理 -> 回复 ACK。"""
        # Worker 侧
        agent = WorkerAgent(node_id="worker-1")
        worker_sent: list[Envelope] = []

        mock_client = MagicMock()

        async def mock_send(envelope: Envelope) -> bool:
            worker_sent.append(envelope)
            return True

        mock_client.send = mock_send
        agent._client = mock_client
        agent._rpc_server = MagicMock()
        agent._rpc_server.start = MagicMock(return_value=50052)
        agent._rpc_server.endpoint = "0.0.0.0:50052"

        # Master 侧
        from asc.master.orchestrator import DistributedOrchestrator
        from asc.types import NodeId
        from asc.types.state import NodeInfo

        master_sent: list[tuple[str, Envelope]] = []

        mock_tcp = MagicMock()

        async def mock_tcp_send(conn_id: str, envelope: Envelope) -> bool:
            master_sent.append((conn_id, envelope))
            # 模拟：Master 发送的消息被 Worker 接收处理
            await agent._on_message(envelope)
            # Worker 的回复被 Orchestrator 接收
            if worker_sent:
                ack = worker_sent[-1]
                if ack.message.type == MessageType.RPC_START_ACK:
                    # 将 request_id 传递到 ACK
                    request_id = envelope.message.payload.get("request_id", "")
                    ack_with_id = Envelope(
                        channel=ack.channel,
                        message=Message(
                            type=ack.message.type,
                            sender_id=ack.message.sender_id,
                            payload={**ack.message.payload, "request_id": request_id},
                        ),
                    )
                    await orch.handle_rpc_start_ack(ack_with_id)
            return True

        mock_tcp.send = mock_tcp_send

        orch = DistributedOrchestrator(
            tcp_server=mock_tcp,
            conn_node_map={"conn-1": "worker-1"},
        )

        node_info = NodeInfo(
            node_id=NodeId("worker-1"),
            ip="10.0.0.2",
            port=52415,
            runner_status="online",
        )

        endpoint = await orch._request_rpc_start(node_info, timeout=5.0)

        # 验证端到端流程
        assert endpoint == "0.0.0.0:50052"
        assert len(master_sent) == 1
        assert master_sent[0][1].message.type == MessageType.RPC_START
        assert len(worker_sent) == 1
        assert worker_sent[0].message.type == MessageType.RPC_START_ACK

    @pytest.mark.asyncio
    async def test_no_http_used_in_rpc_control(self):
        """验证 RPC 控制不再使用 httpx。"""
        from asc.master.orchestrator import DistributedOrchestrator
        from asc.types import NodeId
        from asc.types.state import NodeInfo

        mock_tcp = MagicMock()

        async def mock_send(conn_id: str, envelope: Envelope) -> bool:
            request_id = envelope.message.payload.get("request_id", "")
            ack = Envelope(
                channel=Channel.COMMANDS,
                message=Message(
                    type=MessageType.RPC_START_ACK,
                    sender_id="worker-1",
                    payload={"request_id": request_id, "status": "ok", "endpoint": "10.0.0.2:50052", "port": 50052},
                ),
            )
            await orch.handle_rpc_start_ack(ack)
            return True

        mock_tcp.send = mock_send

        orch = DistributedOrchestrator(
            tcp_server=mock_tcp,
            conn_node_map={"conn-1": "worker-1"},
        )

        node_info = NodeInfo(
            node_id=NodeId("worker-1"),
            ip="10.0.0.2",
            port=52415,
            runner_status="online",
        )

        with patch("httpx.AsyncClient", side_effect=AssertionError("httpx should not be used")):
            endpoint = await orch._request_rpc_start(node_info, timeout=5.0)
            assert endpoint == "10.0.0.2:50052"
