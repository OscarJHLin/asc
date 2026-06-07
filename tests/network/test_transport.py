"""测试 TCP 传输层。

使用 asyncio 实现异步 TCP Server/Client，
统一使用 Binary Frame 作为线协议。
"""

import asyncio

import pytest

from asc.network.frame import Frame, FrameType, encode_frame
from asc.network.protocol import (
    Channel,
    Envelope,
    Message,
    MessageType,
)
from asc.network.transport import TCPClient, TCPServer


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


class TestTCPServer:
    """TCP 服务器测试。"""

    @pytest.mark.asyncio
    async def test_server_starts_and_stops(self):
        server = TCPServer(host="127.0.0.1", port=0)
        await server.start()
        assert server.is_running
        assert server.port > 0
        await server.stop()
        assert not server.is_running

    @pytest.mark.asyncio
    async def test_server_accepts_connection(self):
        server = TCPServer(host="127.0.0.1", port=0)
        await server.start()
        port = server.port

        _reader, writer = await asyncio.open_connection("127.0.0.1", port)
        await asyncio.sleep(0.1)
        assert len(server.connections) == 1

        writer.close()
        await writer.wait_closed()
        await server.stop()

    @pytest.mark.asyncio
    async def test_server_receives_envelope_message(self):
        """服务器接收 Envelope 消息 (通过 Binary Frame)。"""
        received = []

        async def on_message(conn_id: str, envelope: Envelope) -> None:
            received.append(envelope)

        server = TCPServer(host="127.0.0.1", port=0, on_message=on_message)
        await server.start()
        port = server.port

        _reader, writer = await asyncio.open_connection("127.0.0.1", port)
        # 构造 ENVELOPE 帧并发送
        msg = Message(type=MessageType.HEARTBEAT, sender_id="n1", payload={"status": "alive"})
        env = Envelope(channel=Channel.HEARTBEATS, message=msg)

        from asc.network.frame import encode_envelope_frame, write_frame_to_bytes
        frame = encode_envelope_frame(env)
        data = write_frame_to_bytes(frame)
        writer.write(data)
        await writer.drain()

        await asyncio.sleep(0.2)
        assert len(received) == 1
        assert received[0].message.type == MessageType.HEARTBEAT

        writer.close()
        await writer.wait_closed()
        await server.stop()

    @pytest.mark.asyncio
    async def test_server_receives_raw_frame(self):
        """服务器接收原始 Frame (非 ENVELOPE 类型)。"""
        frames_received = []

        async def on_frame(conn_id: str, frame: Frame) -> None:
            frames_received.append(frame)

        server = TCPServer(host="127.0.0.1", port=0, on_frame=on_frame)
        await server.start()
        port = server.port

        _reader, writer = await asyncio.open_connection("127.0.0.1", port)
        # 发送 TASK_DISPATCH 帧
        frame = Frame(frame_type=FrameType.TASK_DISPATCH, payload=b"task_data")
        data = encode_frame(frame)
        writer.write(data)
        await writer.drain()

        await asyncio.sleep(0.2)
        assert len(frames_received) == 1
        assert frames_received[0].frame_type == FrameType.TASK_DISPATCH
        assert frames_received[0].payload == b"task_data"

        writer.close()
        await writer.wait_closed()
        await server.stop()


class TestTCPClient:
    """TCP 客户端测试。"""

    @pytest.mark.asyncio
    async def test_client_connect(self):
        server = TCPServer(host="127.0.0.1", port=0)
        await server.start()
        port = server.port

        client = TCPClient(host="127.0.0.1", port=port, node_id="n1")
        connected = await client.connect()
        assert connected
        assert client.is_connected

        await client.disconnect()
        await server.stop()

    @pytest.mark.asyncio
    async def test_client_send_envelope(self):
        """客户端发送 Envelope 消息。"""
        received = []

        async def on_message(conn_id: str, envelope: Envelope) -> None:
            received.append(envelope)

        server = TCPServer(host="127.0.0.1", port=0, on_message=on_message)
        await server.start()
        port = server.port

        client = TCPClient(host="127.0.0.1", port=port, node_id="n1")
        await client.connect()

        msg = Message(type=MessageType.NODE_JOINED, sender_id="n1", payload={"ip": "10.0.0.1"})
        env = Envelope(channel=Channel.EVENTS, message=msg)
        await client.send(env)

        await asyncio.sleep(0.2)
        assert len(received) == 1
        assert received[0].message.type == MessageType.NODE_JOINED

        await client.disconnect()
        await server.stop()

    @pytest.mark.asyncio
    async def test_client_send_raw_frame(self):
        """客户端发送原始 Frame。"""
        frames_received = []

        async def on_frame(conn_id: str, frame: Frame) -> None:
            frames_received.append(frame)

        server = TCPServer(host="127.0.0.1", port=0, on_frame=on_frame)
        await server.start()
        port = server.port

        client = TCPClient(host="127.0.0.1", port=port, node_id="n1")
        await client.connect()

        frame = Frame(frame_type=FrameType.CAPACITY_REPORT, payload=b"capacity_data")
        await client.send_frame(frame)

        await asyncio.sleep(0.2)
        assert len(frames_received) == 1
        assert frames_received[0].frame_type == FrameType.CAPACITY_REPORT

        await client.disconnect()
        await server.stop()

    @pytest.mark.asyncio
    async def test_client_receive_envelope(self):
        """服务器向客户端推送 Envelope 消息。"""
        client_received = []

        server = TCPServer(host="127.0.0.1", port=0)
        await server.start()
        port = server.port

        client = TCPClient(
            host="127.0.0.1",
            port=port,
            node_id="n1",
            on_message=lambda env: client_received.append(env),
        )
        await client.connect()
        await asyncio.sleep(0.1)

        msg = Message(type=MessageType.HEARTBEAT, sender_id="server", payload={})
        env = Envelope(channel=Channel.HEARTBEATS, message=msg)
        await server.broadcast(env)

        await asyncio.sleep(0.2)
        assert len(client_received) == 1
        assert client_received[0].message.type == MessageType.HEARTBEAT

        await client.disconnect()
        await server.stop()

    @pytest.mark.asyncio
    async def test_client_receive_raw_frame(self):
        """服务器向客户端推送原始 Frame。"""
        client_frames = []

        server = TCPServer(host="127.0.0.1", port=0)
        await server.start()
        port = server.port

        client = TCPClient(
            host="127.0.0.1",
            port=port,
            node_id="n1",
            on_frame=lambda frame: client_frames.append(frame),
        )
        await client.connect()
        await asyncio.sleep(0.1)

        frame = Frame(frame_type=FrameType.REBALANCE_REQUEST, payload=b"rebalance_data")
        await server.broadcast_frame(frame)

        await asyncio.sleep(0.2)
        assert len(client_frames) == 1
        assert client_frames[0].frame_type == FrameType.REBALANCE_REQUEST

        await client.disconnect()
        await server.stop()


class TestTCPServerBroadcast:
    """服务器广播消息。"""

    @pytest.mark.asyncio
    async def test_broadcast_envelope_to_multiple_clients(self):
        received = {1: [], 2: []}

        server = TCPServer(host="127.0.0.1", port=0)
        await server.start()
        port = server.port

        client1 = TCPClient(
            host="127.0.0.1",
            port=port,
            node_id="n1",
            on_message=lambda env: received[1].append(env),
        )
        client2 = TCPClient(
            host="127.0.0.1",
            port=port,
            node_id="n2",
            on_message=lambda env: received[2].append(env),
        )

        await client1.connect()
        await client2.connect()
        await asyncio.sleep(0.1)

        msg = Message(type=MessageType.HEARTBEAT, sender_id="server", payload={})
        env = Envelope(channel=Channel.HEARTBEATS, message=msg)
        await server.broadcast(env)

        await asyncio.sleep(0.2)
        assert len(received[1]) == 1
        assert len(received[2]) == 1

        await client1.disconnect()
        await client2.disconnect()
        await server.stop()

    @pytest.mark.asyncio
    async def test_broadcast_frame_to_multiple_clients(self):
        """广播原始 Frame 到多个客户端。"""
        frames = {1: [], 2: []}

        server = TCPServer(host="127.0.0.1", port=0)
        await server.start()
        port = server.port

        client1 = TCPClient(
            host="127.0.0.1",
            port=port,
            node_id="n1",
            on_frame=lambda f: frames[1].append(f),
        )
        client2 = TCPClient(
            host="127.0.0.1",
            port=port,
            node_id="n2",
            on_frame=lambda f: frames[2].append(f),
        )

        await client1.connect()
        await client2.connect()
        await asyncio.sleep(0.1)

        frame = Frame(frame_type=FrameType.MODEL_CHUNK, payload=b"chunk_data")
        await server.broadcast_frame(frame)

        await asyncio.sleep(0.2)
        assert len(frames[1]) == 1
        assert len(frames[2]) == 1
        assert frames[1][0].frame_type == FrameType.MODEL_CHUNK

        await client1.disconnect()
        await client2.disconnect()
        await server.stop()
