"""TCP 传输层增强测试。

覆盖服务器/客户端边界情况、连接管理、顺序消息、
定向发送、混合消息类型及连接清理等场景。
"""

import asyncio

import pytest

from asc.network.frame import (
    Frame,
    FrameType,
)
from asc.network.protocol import Channel, Envelope, Message, MessageType
from asc.network.transport import TCPClient, TCPServer

# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _make_envelope(sender: str, msg_type: MessageType, payload: dict | None = None) -> Envelope:
    """构造测试用 Envelope。"""
    return Envelope(
        channel=Channel.EVENTS,
        message=Message(type=msg_type, sender_id=sender, payload=payload or {}),
    )


def _make_frame(frame_type: FrameType, payload: bytes = b"test") -> Frame:
    """构造测试用 Frame。"""
    return Frame(frame_type=frame_type, payload=payload)


# ---------------------------------------------------------------------------
# 1. 服务器边界情况
# ---------------------------------------------------------------------------


class TestServerEdgeCases:
    """服务器边界情况测试。"""

    @pytest.mark.asyncio
    async def test_port_property_before_start_returns_configured_port(self):
        """启动前 port 属性返回配置的端口号。"""
        server = TCPServer(host="127.0.0.1", port=9999)
        assert server.port == 9999
        # 不启动，直接清理
        del server

    @pytest.mark.asyncio
    async def test_port_property_after_start_returns_bound_port(self):
        """启动后 port 属性返回实际绑定的端口号。"""
        server = TCPServer(host="127.0.0.1", port=0)
        await server.start()
        assert server.port > 0
        assert server.port != 0
        await server.stop()

    @pytest.mark.asyncio
    async def test_is_running_false_before_start(self):
        """启动前 is_running 为 False。"""
        server = TCPServer(host="127.0.0.1", port=0)
        assert server.is_running is False

    @pytest.mark.asyncio
    async def test_is_running_true_after_start(self):
        """启动后 is_running 为 True。"""
        server = TCPServer(host="127.0.0.1", port=0)
        await server.start()
        assert server.is_running is True
        await server.stop()

    @pytest.mark.asyncio
    async def test_is_running_false_after_stop(self):
        """停止后 is_running 为 False。"""
        server = TCPServer(host="127.0.0.1", port=0)
        await server.start()
        await server.stop()
        assert server.is_running is False

    @pytest.mark.asyncio
    async def test_connections_empty_before_any_connections(self):
        """无连接时 connections 字典为空。"""
        server = TCPServer(host="127.0.0.1", port=0)
        await server.start()
        assert len(server.connections) == 0
        await server.stop()

    @pytest.mark.asyncio
    async def test_send_to_nonexistent_connection_returns_false(self):
        """向不存在的连接发送 Envelope 返回 False。"""
        server = TCPServer(host="127.0.0.1", port=0)
        await server.start()
        env = _make_envelope("n1", MessageType.HEARTBEAT)
        result = await server.send("nonexistent_conn_id", env)
        assert result is False
        await server.stop()

    @pytest.mark.asyncio
    async def test_send_frame_to_nonexistent_connection_returns_false(self):
        """向不存在的连接发送 Frame 返回 False。"""
        server = TCPServer(host="127.0.0.1", port=0)
        await server.start()
        frame = _make_frame(FrameType.TASK_DISPATCH)
        result = await server.send_frame("nonexistent_conn_id", frame)
        assert result is False
        await server.stop()


# ---------------------------------------------------------------------------
# 2. 客户端边界情况
# ---------------------------------------------------------------------------


class TestClientEdgeCases:
    """客户端边界情况测试。"""

    @pytest.mark.asyncio
    async def test_is_connected_false_before_connect(self):
        """连接前 is_connected 为 False。"""
        client = TCPClient(host="127.0.0.1", port=0, node_id="n1")
        assert client.is_connected is False

    @pytest.mark.asyncio
    async def test_send_when_not_connected_returns_false(self):
        """未连接时发送 Envelope 返回 False。"""
        client = TCPClient(host="127.0.0.1", port=0, node_id="n1")
        env = _make_envelope("n1", MessageType.HEARTBEAT)
        result = await client.send(env)
        assert result is False

    @pytest.mark.asyncio
    async def test_send_frame_when_not_connected_returns_false(self):
        """未连接时发送 Frame 返回 False。"""
        client = TCPClient(host="127.0.0.1", port=0, node_id="n1")
        frame = _make_frame(FrameType.TASK_DISPATCH)
        result = await client.send_frame(frame)
        assert result is False

    @pytest.mark.asyncio
    async def test_connect_to_nonexistent_server_returns_false(self):
        """连接到不存在的服务器返回 False。"""
        client = TCPClient(host="127.0.0.1", port=59999, node_id="n1")
        result = await client.connect()
        assert result is False
        assert client.is_connected is False

    @pytest.mark.asyncio
    async def test_disconnect_when_already_disconnected_no_error(self):
        """已断开连接时再次 disconnect 不抛出异常。"""
        client = TCPClient(host="127.0.0.1", port=0, node_id="n1")
        # 从未连接，直接 disconnect
        await client.disconnect()
        assert client.is_connected is False


# ---------------------------------------------------------------------------
# 3. 连接管理
# ---------------------------------------------------------------------------


class TestConnectionHandling:
    """连接管理测试。"""

    @pytest.mark.asyncio
    async def test_server_handles_client_disconnect_gracefully(self):
        """服务器优雅处理客户端断开连接。"""
        server = TCPServer(host="127.0.0.1", port=0)
        await server.start()
        port = server.port

        client = TCPClient(host="127.0.0.1", port=port, node_id="n1")
        await client.connect()
        await asyncio.sleep(0.1)
        assert len(server.connections) == 1

        # 客户端主动断开
        await client.disconnect()
        await asyncio.sleep(0.2)
        assert len(server.connections) == 0

        await server.stop()

    @pytest.mark.asyncio
    async def test_multiple_clients_connect_and_disconnect(self):
        """多个客户端连接和断开。"""
        server = TCPServer(host="127.0.0.1", port=0)
        await server.start()
        port = server.port

        client1 = TCPClient(host="127.0.0.1", port=port, node_id="n1")
        client2 = TCPClient(host="127.0.0.1", port=port, node_id="n2")
        client3 = TCPClient(host="127.0.0.1", port=port, node_id="n3")

        await client1.connect()
        await client2.connect()
        await client3.connect()
        await asyncio.sleep(0.1)
        assert len(server.connections) == 3

        await client2.disconnect()
        await asyncio.sleep(0.2)
        assert len(server.connections) == 2

        await client1.disconnect()
        await client3.disconnect()
        await asyncio.sleep(0.2)
        assert len(server.connections) == 0

        await server.stop()

    @pytest.mark.asyncio
    async def test_server_tracks_connections_count_matches(self):
        """服务器跟踪连接数与实际客户端数匹配。"""
        server = TCPServer(host="127.0.0.1", port=0)
        await server.start()
        port = server.port

        clients = []
        for i in range(5):
            c = TCPClient(host="127.0.0.1", port=port, node_id=f"n{i}")
            await c.connect()
            clients.append(c)

        await asyncio.sleep(0.2)
        assert len(server.connections) == 5

        # 断开前 3 个
        for c in clients[:3]:
            await c.disconnect()
        await asyncio.sleep(0.2)
        assert len(server.connections) == 2

        # 断开剩余
        for c in clients[3:]:
            await c.disconnect()
        await asyncio.sleep(0.2)
        assert len(server.connections) == 0

        await server.stop()


# ---------------------------------------------------------------------------
# 4. 顺序消息
# ---------------------------------------------------------------------------


class TestSequentialMessages:
    """顺序消息测试。"""

    @pytest.mark.asyncio
    async def test_client_sends_multiple_envelopes_in_sequence(self):
        """客户端依次发送多条 Envelope，服务器按序接收。"""
        received: list[Envelope] = []

        async def on_message(conn_id: str, envelope: Envelope) -> None:
            received.append(envelope)

        server = TCPServer(host="127.0.0.1", port=0, on_message=on_message)
        await server.start()
        port = server.port

        client = TCPClient(host="127.0.0.1", port=port, node_id="n1")
        await client.connect()

        msg_types = [MessageType.NODE_JOINED, MessageType.HEARTBEAT, MessageType.TASK_CREATED]
        for mt in msg_types:
            env = _make_envelope("n1", mt, payload={"seq": mt.value})
            await client.send(env)

        await asyncio.sleep(0.5)
        assert len(received) == 3
        for i, mt in enumerate(msg_types):
            assert received[i].message.type == mt

        await client.disconnect()
        await server.stop()

    @pytest.mark.asyncio
    async def test_client_sends_multiple_frames_in_sequence(self):
        """客户端依次发送多条 Frame，服务器按序接收。"""
        received: list[Frame] = []

        async def on_frame(conn_id: str, frame: Frame) -> None:
            received.append(frame)

        server = TCPServer(host="127.0.0.1", port=0, on_frame=on_frame)
        await server.start()
        port = server.port

        client = TCPClient(host="127.0.0.1", port=port, node_id="n1")
        await client.connect()

        frame_types = [
            FrameType.TASK_DISPATCH, FrameType.CAPACITY_REPORT,
            FrameType.REBALANCE_REQUEST,
        ]
        payloads = [b"seq0", b"seq1", b"seq2"]
        for ft, pl in zip(frame_types, payloads, strict=False):
            frame = Frame(frame_type=ft, payload=pl)
            await client.send_frame(frame)

        await asyncio.sleep(0.5)
        assert len(received) == 3
        for i, (ft, pl) in enumerate(zip(frame_types, payloads, strict=False)):
            assert received[i].frame_type == ft
            assert received[i].payload == pl

        await client.disconnect()
        await server.stop()

    @pytest.mark.asyncio
    async def test_server_sends_multiple_envelopes_to_client_in_sequence(self):
        """服务器依次向客户端发送多条 Envelope，客户端按序接收。"""
        received: list[Envelope] = []

        server = TCPServer(host="127.0.0.1", port=0)
        await server.start()
        port = server.port

        async def on_message(env: Envelope) -> None:
            received.append(env)

        client = TCPClient(
            host="127.0.0.1",
            port=port,
            node_id="n1",
            on_message=on_message,
        )
        await client.connect()
        await asyncio.sleep(0.1)

        # 获取连接 ID
        conn_ids = list(server.connections.keys())
        assert len(conn_ids) == 1
        conn_id = conn_ids[0]

        msg_types = [MessageType.HEARTBEAT, MessageType.DISCOVER, MessageType.NODE_LEFT]
        for mt in msg_types:
            env = _make_envelope("server", mt, payload={"seq": mt.value})
            await server.send(conn_id, env)

        await asyncio.sleep(0.5)
        assert len(received) == 3
        for i, mt in enumerate(msg_types):
            assert received[i].message.type == mt

        await client.disconnect()
        await server.stop()

    @pytest.mark.asyncio
    async def test_server_sends_multiple_frames_to_client_in_sequence(self):
        """服务器依次向客户端发送多条 Frame，客户端按序接收。"""
        received: list[Frame] = []

        server = TCPServer(host="127.0.0.1", port=0)
        await server.start()
        port = server.port

        async def on_frame(frame: Frame) -> None:
            received.append(frame)

        client = TCPClient(
            host="127.0.0.1",
            port=port,
            node_id="n1",
            on_frame=on_frame,
        )
        await client.connect()
        await asyncio.sleep(0.1)

        conn_ids = list(server.connections.keys())
        assert len(conn_ids) == 1
        conn_id = conn_ids[0]

        frame_types = [FrameType.HEARTBEAT, FrameType.ACK, FrameType.MODEL_CHUNK]
        payloads = [b"p0", b"p1", b"p2"]
        for ft, pl in zip(frame_types, payloads, strict=False):
            frame = Frame(frame_type=ft, payload=pl)
            await server.send_frame(conn_id, frame)

        await asyncio.sleep(0.5)
        assert len(received) == 3
        for i, (ft, pl) in enumerate(zip(frame_types, payloads, strict=False)):
            assert received[i].frame_type == ft
            assert received[i].payload == pl

        await client.disconnect()
        await server.stop()


# ---------------------------------------------------------------------------
# 5. 定向发送
# ---------------------------------------------------------------------------


class TestDirectedSend:
    """定向发送测试。"""

    @pytest.mark.asyncio
    async def test_server_sends_envelope_to_specific_client(self):
        """服务器向指定客户端发送 Envelope，仅目标客户端收到。"""
        received: dict[str, list[Envelope]] = {"c1": [], "c2": []}

        server = TCPServer(host="127.0.0.1", port=0)
        await server.start()
        port = server.port

        async def on_msg_c1(env: Envelope) -> None:
            received["c1"].append(env)

        async def on_msg_c2(env: Envelope) -> None:
            received["c2"].append(env)

        client1 = TCPClient(
            host="127.0.0.1", port=port, node_id="c1", on_message=on_msg_c1,
        )
        client2 = TCPClient(
            host="127.0.0.1", port=port, node_id="c2", on_message=on_msg_c2,
        )
        await client1.connect()
        await client2.connect()
        await asyncio.sleep(0.1)

        conn_ids = list(server.connections.keys())
        assert len(conn_ids) == 2
        target_conn_id = conn_ids[0]

        env = _make_envelope("server", MessageType.HEARTBEAT, payload={"target": "only_you"})
        await server.send(target_conn_id, env)

        await asyncio.sleep(0.3)
        # 只有一个客户端收到
        total = len(received["c1"]) + len(received["c2"])
        assert total == 1

        await client1.disconnect()
        await client2.disconnect()
        await server.stop()

    @pytest.mark.asyncio
    async def test_server_sends_frame_to_specific_client(self):
        """服务器向指定客户端发送 Frame，仅目标客户端收到。"""
        received: dict[str, list[Frame]] = {"c1": [], "c2": []}

        server = TCPServer(host="127.0.0.1", port=0)
        await server.start()
        port = server.port

        async def on_frame_c1(frame: Frame) -> None:
            received["c1"].append(frame)

        async def on_frame_c2(frame: Frame) -> None:
            received["c2"].append(frame)

        client1 = TCPClient(
            host="127.0.0.1", port=port, node_id="c1", on_frame=on_frame_c1,
        )
        client2 = TCPClient(
            host="127.0.0.1", port=port, node_id="c2", on_frame=on_frame_c2,
        )
        await client1.connect()
        await client2.connect()
        await asyncio.sleep(0.1)

        conn_ids = list(server.connections.keys())
        assert len(conn_ids) == 2
        target_conn_id = conn_ids[1]

        frame = _make_frame(FrameType.TASK_DISPATCH, payload=b"directed_frame")
        await server.send_frame(target_conn_id, frame)

        await asyncio.sleep(0.3)
        total = len(received["c1"]) + len(received["c2"])
        assert total == 1

        await client1.disconnect()
        await client2.disconnect()
        await server.stop()

    @pytest.mark.asyncio
    async def test_directed_send_not_received_by_other_client(self):
        """定向发送时，非目标客户端确认未收到消息。"""
        received_c1: list[Envelope] = []
        received_c2: list[Envelope] = []

        server = TCPServer(host="127.0.0.1", port=0)
        await server.start()
        port = server.port

        async def on_msg_c1(env: Envelope) -> None:
            received_c1.append(env)

        async def on_msg_c2(env: Envelope) -> None:
            received_c2.append(env)

        client1 = TCPClient(
            host="127.0.0.1", port=port, node_id="c1", on_message=on_msg_c1,
        )
        client2 = TCPClient(
            host="127.0.0.1", port=port, node_id="c2", on_message=on_msg_c2,
        )
        await client1.connect()
        await client2.connect()
        await asyncio.sleep(0.1)

        conn_ids = list(server.connections.keys())
        # 向 client2 的连接定向发送
        target_conn_id = conn_ids[1]

        env = _make_envelope("server", MessageType.DISCOVER, payload={"for": "c2"})
        await server.send(target_conn_id, env)

        await asyncio.sleep(0.3)
        # 目标客户端收到 1 条，非目标客户端收到 0 条
        assert len(received_c2) + len(received_c1) == 1
        if len(received_c2) == 1:
            assert received_c1 == []
        else:
            assert received_c1 == []

        await client1.disconnect()
        await client2.disconnect()
        await server.stop()


# ---------------------------------------------------------------------------
# 6. 混合消息类型
# ---------------------------------------------------------------------------


class TestMixedMessageTypes:
    """混合消息类型测试。"""

    @pytest.mark.asyncio
    async def test_interleaved_envelope_and_frame_from_client(self):
        """客户端交替发送 Envelope 和 Frame，服务器两种回调均正确触发。"""
        received_env: list[Envelope] = []
        received_frame: list[Frame] = []

        async def on_message(conn_id: str, envelope: Envelope) -> None:
            received_env.append(envelope)

        async def on_frame(conn_id: str, frame: Frame) -> None:
            received_frame.append(frame)

        server = TCPServer(
            host="127.0.0.1", port=0,
            on_message=on_message, on_frame=on_frame,
        )
        await server.start()
        port = server.port

        client = TCPClient(host="127.0.0.1", port=port, node_id="n1")
        await client.connect()

        # 交替发送
        env1 = _make_envelope("n1", MessageType.HEARTBEAT, payload={"i": 0})
        await client.send(env1)

        frame1 = _make_frame(FrameType.TASK_DISPATCH, payload=b"interleaved_1")
        await client.send_frame(frame1)

        env2 = _make_envelope("n1", MessageType.NODE_JOINED, payload={"i": 2})
        await client.send(env2)

        frame2 = _make_frame(FrameType.CAPACITY_REPORT, payload=b"interleaved_2")
        await client.send_frame(frame2)

        await asyncio.sleep(0.5)
        assert len(received_env) == 2
        assert received_env[0].message.type == MessageType.HEARTBEAT
        assert received_env[1].message.type == MessageType.NODE_JOINED

        assert len(received_frame) == 4  # on_frame 回调对 ENVELOPE 帧也会触发
        assert received_frame[0].frame_type == FrameType.ENVELOPE
        assert received_frame[1].frame_type == FrameType.TASK_DISPATCH
        assert received_frame[2].frame_type == FrameType.ENVELOPE
        assert received_frame[3].frame_type == FrameType.CAPACITY_REPORT

        await client.disconnect()
        await server.stop()

    @pytest.mark.asyncio
    async def test_interleaved_envelope_and_frame_from_server(self):
        """服务器交替发送 Envelope 和 Frame，客户端两种回调均正确触发。"""
        received_env: list[Envelope] = []
        received_frame: list[Frame] = []

        server = TCPServer(host="127.0.0.1", port=0)
        await server.start()
        port = server.port

        async def on_message(env: Envelope) -> None:
            received_env.append(env)

        async def on_frame(frame: Frame) -> None:
            received_frame.append(frame)

        client = TCPClient(
            host="127.0.0.1", port=port, node_id="n1",
            on_message=on_message, on_frame=on_frame,
        )
        await client.connect()
        await asyncio.sleep(0.1)

        conn_ids = list(server.connections.keys())
        conn_id = conn_ids[0]

        # 交替发送
        env1 = _make_envelope("server", MessageType.HEARTBEAT, payload={"i": 0})
        await server.send(conn_id, env1)

        frame1 = _make_frame(FrameType.MODEL_CHUNK, payload=b"chunk_0")
        await server.send_frame(conn_id, frame1)

        env2 = _make_envelope("server", MessageType.DISCOVER, payload={"i": 2})
        await server.send(conn_id, env2)

        frame2 = _make_frame(FrameType.ACK, payload=b"ack_0")
        await server.send_frame(conn_id, frame2)

        await asyncio.sleep(0.5)
        assert len(received_env) == 2
        assert received_env[0].message.type == MessageType.HEARTBEAT
        assert received_env[1].message.type == MessageType.DISCOVER

        assert len(received_frame) == 4  # on_frame 对 ENVELOPE 帧也触发
        assert received_frame[0].frame_type == FrameType.ENVELOPE
        assert received_frame[1].frame_type == FrameType.MODEL_CHUNK
        assert received_frame[2].frame_type == FrameType.ENVELOPE
        assert received_frame[3].frame_type == FrameType.ACK

        await client.disconnect()
        await server.stop()


# ---------------------------------------------------------------------------
# 7. 连接清理
# ---------------------------------------------------------------------------


class TestConnectionCleanup:
    """连接清理测试。"""

    @pytest.mark.asyncio
    async def test_server_stop_closes_all_client_connections(self):
        """服务器停止后所有客户端连接被关闭。"""
        server = TCPServer(host="127.0.0.1", port=0)
        await server.start()
        port = server.port

        clients = []
        for i in range(3):
            c = TCPClient(host="127.0.0.1", port=port, node_id=f"n{i}")
            await c.connect()
            clients.append(c)

        await asyncio.sleep(0.1)
        assert len(server.connections) == 3

        # 停止服务器
        await server.stop()
        await asyncio.sleep(0.2)

        # 所有客户端应检测到连接断开
        for c in clients:
            assert c.is_connected is False

    @pytest.mark.asyncio
    async def test_client_disconnect_cleans_up_properly(self):
        """客户端断开连接后状态正确清理。"""
        server = TCPServer(host="127.0.0.1", port=0)
        await server.start()
        port = server.port

        client = TCPClient(host="127.0.0.1", port=port, node_id="n1")
        await client.connect()
        assert client.is_connected is True

        await client.disconnect()
        assert client.is_connected is False
        # 服务器端连接也应被清理
        await asyncio.sleep(0.2)
        assert len(server.connections) == 0

        await server.stop()

    @pytest.mark.asyncio
    async def test_server_connections_cleared_after_stop(self):
        """服务器停止后 connections 字典被清空。"""
        server = TCPServer(host="127.0.0.1", port=0)
        await server.start()
        port = server.port

        client = TCPClient(host="127.0.0.1", port=port, node_id="n1")
        await client.connect()
        await asyncio.sleep(0.1)
        assert len(server.connections) == 1

        await server.stop()
        assert len(server.connections) == 0

    @pytest.mark.asyncio
    async def test_multiple_disconnects_no_error(self):
        """多次 disconnect 不抛出异常。"""
        server = TCPServer(host="127.0.0.1", port=0)
        await server.start()
        port = server.port

        client = TCPClient(host="127.0.0.1", port=port, node_id="n1")
        await client.connect()
        await client.disconnect()
        # 再次 disconnect 不应抛出异常
        await client.disconnect()
        assert client.is_connected is False

        await server.stop()
