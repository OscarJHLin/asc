"""测试 TCP 传输层。

使用 asyncio 实现异步 TCP Server/Client，
支持：连接管理、消息发送/接收、自动重连。
"""

import asyncio

import pytest

from asc.network.protocol import (
    Channel,
    Envelope,
    Message,
    MessageType,
    encode_envelope,
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
        server = TCPServer(host="127.0.0.1", port=0)  # port=0 让 OS 分配
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

        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        # 等待服务器接受连接
        await asyncio.sleep(0.1)
        assert len(server.connections) == 1

        writer.close()
        await writer.wait_closed()
        await server.stop()

    @pytest.mark.asyncio
    async def test_server_receives_message(self):
        received = []

        async def on_message(conn_id: str, envelope: Envelope) -> None:
            received.append(envelope)

        server = TCPServer(host="127.0.0.1", port=0, on_message=on_message)
        await server.start()
        port = server.port

        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        msg = Message(type=MessageType.HEARTBEAT, sender_id="n1", payload={"status": "alive"})
        env = Envelope(channel=Channel.HEARTBEATS, message=msg)
        data = encode_envelope(env)
        # 协议：4字节长度前缀 + 数据
        header = len(data).to_bytes(4, "big")
        writer.write(header + data)
        await writer.drain()

        await asyncio.sleep(0.2)
        assert len(received) == 1
        assert received[0].message.type == MessageType.HEARTBEAT

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
    async def test_client_send_message(self):
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
    async def test_client_receive_message(self):
        """服务器向客户端推送消息。"""
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

        # 服务器向所有连接发送消息
        msg = Message(type=MessageType.HEARTBEAT, sender_id="server", payload={})
        env = Envelope(channel=Channel.HEARTBEATS, message=msg)
        await server.broadcast(env)

        await asyncio.sleep(0.2)
        assert len(client_received) == 1
        assert client_received[0].message.type == MessageType.HEARTBEAT

        await client.disconnect()
        await server.stop()


class TestTCPServerBroadcast:
    """服务器广播消息。"""

    @pytest.mark.asyncio
    async def test_broadcast_to_multiple_clients(self):
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
