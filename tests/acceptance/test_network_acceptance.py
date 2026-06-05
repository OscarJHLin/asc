"""ASC 验收测试 - 网络模块

测试范围：network/protocol.py, network/router.py, network/transport.py, network/discovery.py
测试维度：功能测试、边界条件测试、异常场景测试
"""

from __future__ import annotations

import asyncio
import json

import pytest

from asc.network.discovery import DiscoveryMessage, NodeDiscovery
from asc.network.protocol import (
    Channel,
    Envelope,
    Message,
    MessageType,
    decode_envelope,
    encode_envelope,
)
from asc.network.router import MessageRouter

# ======================================================================
# 1. network/protocol.py 测试
# ======================================================================


class TestMessage:
    """Message 消息测试。"""

    def test_message_creation(self):
        msg = Message(
            type=MessageType.HEARTBEAT,
            sender_id="node-1",
            payload={"status": "ok"},
        )
        assert msg.type == MessageType.HEARTBEAT
        assert msg.sender_id == "node-1"
        assert msg.payload["status"] == "ok"

    def test_message_auto_timestamp(self):
        """消息自动设置时间戳。"""
        msg = Message(
            type=MessageType.HEARTBEAT,
            sender_id="node-1",
            payload={},
        )
        assert msg.timestamp > 0

    def test_message_custom_timestamp(self):
        """自定义时间戳。"""
        msg = Message(
            type=MessageType.HEARTBEAT,
            sender_id="node-1",
            payload={},
            timestamp=12345.0,
        )
        assert msg.timestamp == 12345.0

    def test_message_frozen(self):
        """消息不可修改。"""
        msg = Message(
            type=MessageType.HEARTBEAT,
            sender_id="node-1",
            payload={},
        )
        with pytest.raises(AttributeError):
            msg.sender_id = "node-2"

    def test_message_empty_payload(self):
        """空 payload。"""
        msg = Message(
            type=MessageType.HEARTBEAT,
            sender_id="node-1",
            payload={},
        )
        assert msg.payload == {}


class TestEnvelope:
    """Envelope 信封测试。"""

    def test_envelope_creation(self):
        msg = Message(
            type=MessageType.HEARTBEAT,
            sender_id="node-1",
            payload={},
        )
        env = Envelope(channel=Channel.HEARTBEATS, message=msg)
        assert env.channel == Channel.HEARTBEATS
        assert env.target is None

    def test_envelope_with_target(self):
        msg = Message(
            type=MessageType.HEARTBEAT,
            sender_id="node-1",
            payload={},
        )
        env = Envelope(channel=Channel.HEARTBEATS, message=msg, target="node-2")
        assert env.target == "node-2"

    def test_envelope_frozen(self):
        msg = Message(
            type=MessageType.HEARTBEAT,
            sender_id="node-1",
            payload={},
        )
        env = Envelope(channel=Channel.HEARTBEATS, message=msg)
        with pytest.raises(AttributeError):
            env.channel = Channel.EVENTS


class TestEncodeDecodeEnvelope:
    """信封序列化/反序列化测试。"""

    def test_roundtrip(self):
        msg = Message(
            type=MessageType.NODE_JOINED,
            sender_id="node-1",
            payload={"ip": "192.168.1.1", "port": 52415},
        )
        env = Envelope(channel=Channel.EVENTS, message=msg)
        data = encode_envelope(env)
        env2 = decode_envelope(data)
        assert env2.channel == Channel.EVENTS
        assert env2.message.type == MessageType.NODE_JOINED
        assert env2.message.sender_id == "node-1"
        assert env2.message.payload["ip"] == "192.168.1.1"

    def test_roundtrip_with_target(self):
        msg = Message(
            type=MessageType.CREATE_INSTANCE,
            sender_id="node-1",
            payload={"model": "llama-7b"},
        )
        env = Envelope(channel=Channel.COMMANDS, message=msg, target="node-2")
        data = encode_envelope(env)
        env2 = decode_envelope(data)
        assert env2.target == "node-2"

    def test_roundtrip_preserves_timestamp(self):
        msg = Message(
            type=MessageType.HEARTBEAT,
            sender_id="node-1",
            payload={},
            timestamp=12345.0,
        )
        env = Envelope(channel=Channel.HEARTBEATS, message=msg)
        data = encode_envelope(env)
        env2 = decode_envelope(data)
        assert env2.message.timestamp == 12345.0

    def test_decode_invalid_json(self):
        """无效 JSON 应抛出异常。"""
        with pytest.raises(json.JSONDecodeError):
            decode_envelope(b"not json")

    def test_decode_invalid_channel(self):
        """无效通道应抛出 ValueError。"""
        data = json.dumps({
            "channel": "invalid_channel",
            "message": {
                "type": "heartbeat",
                "sender_id": "n1",
                "payload": {},
                "timestamp": 1.0,
            },
        }).encode("utf-8")
        with pytest.raises(ValueError):
            decode_envelope(data)

    def test_decode_invalid_message_type(self):
        """无效消息类型应抛出 ValueError。"""
        data = json.dumps({
            "channel": "events",
            "message": {
                "type": "invalid_type",
                "sender_id": "n1",
                "payload": {},
                "timestamp": 1.0,
            },
        }).encode("utf-8")
        with pytest.raises(ValueError):
            decode_envelope(data)


class TestChannelEnum:
    """Channel 枚举测试。"""

    def test_all_channels(self):
        assert Channel.EVENTS.value == "events"
        assert Channel.COMMANDS.value == "commands"
        assert Channel.HEARTBEATS.value == "heartbeats"
        assert Channel.ELECTION.value == "election"
        assert Channel.DISCOVERY.value == "discovery"


class TestMessageTypeEnum:
    """MessageType 枚举测试。"""

    def test_event_types(self):
        assert MessageType.NODE_JOINED.value == "node_joined"
        assert MessageType.NODE_LEFT.value == "node_left"
        assert MessageType.INSTANCE_CREATED.value == "instance_created"
        assert MessageType.INSTANCE_DELETED.value == "instance_deleted"
        assert MessageType.TASK_CREATED.value == "task_created"
        assert MessageType.TASK_COMPLETED.value == "task_completed"
        assert MessageType.TASK_FAILED.value == "task_failed"
        assert MessageType.TASK_CANCELLED.value == "task_cancelled"
        assert MessageType.RUNNER_STATUS.value == "runner_status"

    def test_command_types(self):
        assert MessageType.CREATE_INSTANCE.value == "create_instance"
        assert MessageType.DELETE_INSTANCE.value == "delete_instance"
        assert MessageType.START_INFERENCE.value == "start_inference"
        assert MessageType.CANCEL_TASK.value == "cancel_task"
        assert MessageType.SHUTDOWN_RUNNER.value == "shutdown_runner"

    def test_control_types(self):
        assert MessageType.HEARTBEAT.value == "heartbeat"
        assert MessageType.DISCOVER.value == "discover"
        assert MessageType.DISCOVER_RESPONSE.value == "discover_response"
        assert MessageType.ELECTION.value == "election"


class TestProtocolBoundary:
    """协议边界条件测试。"""

    def test_large_payload(self):
        """大 payload 序列化/反序列化。"""
        large_data = {"key": "x" * 10000}
        msg = Message(
            type=MessageType.HEARTBEAT,
            sender_id="node-1",
            payload=large_data,
        )
        env = Envelope(channel=Channel.HEARTBEATS, message=msg)
        data = encode_envelope(env)
        env2 = decode_envelope(data)
        assert len(env2.message.payload["key"]) == 10000

    def test_unicode_payload(self):
        """Unicode payload。"""
        msg = Message(
            type=MessageType.HEARTBEAT,
            sender_id="node-1",
            payload={"text": "你好世界🎉"},
        )
        env = Envelope(channel=Channel.HEARTBEATS, message=msg)
        data = encode_envelope(env)
        env2 = decode_envelope(data)
        assert env2.message.payload["text"] == "你好世界🎉"

    def test_nested_payload(self):
        """嵌套 payload。"""
        msg = Message(
            type=MessageType.HEARTBEAT,
            sender_id="node-1",
            payload={"nested": {"deep": {"value": 42}}},
        )
        env = Envelope(channel=Channel.HEARTBEATS, message=msg)
        data = encode_envelope(env)
        env2 = decode_envelope(data)
        assert env2.message.payload["nested"]["deep"]["value"] == 42


# ======================================================================
# 2. network/router.py 测试
# ======================================================================


class TestMessageRouterSubscribe:
    """MessageRouter 订阅测试。"""

    def test_subscribe_and_publish(self):
        router = MessageRouter(node_id="node-1")
        received = []
        router.subscribe(Channel.EVENTS, lambda env: received.append(env))

        msg = Message(type=MessageType.NODE_JOINED, sender_id="node-2", payload={})
        env = Envelope(channel=Channel.EVENTS, message=msg)
        router.publish_local(env)
        assert len(received) == 1

    def test_multiple_subscribers(self):
        router = MessageRouter(node_id="node-1")
        received_a = []
        received_b = []
        router.subscribe(Channel.EVENTS, lambda env: received_a.append(env))
        router.subscribe(Channel.EVENTS, lambda env: received_b.append(env))

        msg = Message(type=MessageType.NODE_JOINED, sender_id="node-2", payload={})
        env = Envelope(channel=Channel.EVENTS, message=msg)
        router.publish_local(env)
        assert len(received_a) == 1
        assert len(received_b) == 1

    def test_channel_isolation(self):
        """不同通道的消息互不干扰。"""
        router = MessageRouter(node_id="node-1")
        events_received = []
        commands_received = []
        router.subscribe(Channel.EVENTS, lambda env: events_received.append(env))
        router.subscribe(Channel.COMMANDS, lambda env: commands_received.append(env))

        msg = Message(type=MessageType.NODE_JOINED, sender_id="node-2", payload={})
        env = Envelope(channel=Channel.EVENTS, message=msg)
        router.publish_local(env)
        assert len(events_received) == 1
        assert len(commands_received) == 0

    def test_no_subscribers(self):
        """无订阅者发布不报错。"""
        router = MessageRouter(node_id="node-1")
        msg = Message(type=MessageType.HEARTBEAT, sender_id="node-1", payload={})
        env = Envelope(channel=Channel.HEARTBEATS, message=msg)
        router.publish_local(env)  # 不应抛异常


class TestMessageRouterUnsubscribe:
    """MessageRouter 取消订阅测试。"""

    def test_unsubscribe(self):
        router = MessageRouter(node_id="node-1")
        received = []
        def handler(env):
            received.append(env)
        router.subscribe(Channel.EVENTS, handler)
        router.unsubscribe(Channel.EVENTS, handler)

        msg = Message(type=MessageType.NODE_JOINED, sender_id="node-2", payload={})
        env = Envelope(channel=Channel.EVENTS, message=msg)
        router.publish_local(env)
        assert len(received) == 0

    def test_unsubscribe_nonexistent_handler(self):
        """取消不存在的处理器不报错。"""
        router = MessageRouter(node_id="node-1")
        router.unsubscribe(Channel.EVENTS, lambda env: None)

    def test_unsubscribe_nonexistent_channel(self):
        """取消不存在的通道不报错。"""
        router = MessageRouter(node_id="node-1")
        router.unsubscribe(Channel.DISCOVERY, lambda env: None)


class TestMessageRouterLoopDetection:
    """MessageRouter 回环检测测试。"""

    def test_handle_incoming_ignores_self(self):
        """忽略自己发出的消息。"""
        router = MessageRouter(node_id="node-1")
        received = []
        router.subscribe(Channel.EVENTS, lambda env: received.append(env))

        msg = Message(type=MessageType.NODE_JOINED, sender_id="node-1", payload={})
        env = Envelope(channel=Channel.EVENTS, message=msg)
        router.handle_incoming(env)
        assert len(received) == 0

    def test_handle_incoming_accepts_others(self):
        """接受其他节点发出的消息。"""
        router = MessageRouter(node_id="node-1")
        received = []
        router.subscribe(Channel.EVENTS, lambda env: received.append(env))

        msg = Message(type=MessageType.NODE_JOINED, sender_id="node-2", payload={})
        env = Envelope(channel=Channel.EVENTS, message=msg)
        router.handle_incoming(env)
        assert len(received) == 1


class TestMessageRouterRemote:
    """MessageRouter 远程发送测试。"""

    @pytest.mark.asyncio
    async def test_publish_remote_no_sender(self):
        """没有远程发送器时返回 False。"""
        router = MessageRouter(node_id="node-1")
        msg = Message(type=MessageType.HEARTBEAT, sender_id="node-1", payload={})
        env = Envelope(channel=Channel.HEARTBEATS, message=msg)
        result = await router.publish_remote(env, "node-2")
        assert result is False

    @pytest.mark.asyncio
    async def test_publish_remote_with_sender(self):
        """有远程发送器时调用。"""
        router = MessageRouter(node_id="node-1")

        async def mock_sender(env, target):
            return True

        router.set_remote_sender(mock_sender)
        msg = Message(type=MessageType.HEARTBEAT, sender_id="node-1", payload={})
        env = Envelope(channel=Channel.HEARTBEATS, message=msg)
        result = await router.publish_remote(env, "node-2")
        assert result is True


# ======================================================================
# 3. network/transport.py 测试
# ======================================================================


class TestTCPServerClient:
    """TCP 服务器/客户端集成测试。"""

    @pytest.mark.asyncio
    async def test_server_start_stop(self):
        from asc.network.transport import TCPServer
        server = TCPServer(host="127.0.0.1", port=0)
        await server.start()
        assert server.is_running is True
        assert server.port > 0
        await server.stop()
        assert server.is_running is False

    @pytest.mark.asyncio
    async def test_client_connect_disconnect(self):
        from asc.network.transport import TCPClient, TCPServer
        server = TCPServer(host="127.0.0.1", port=0)
        await server.start()
        port = server.port

        client = TCPClient(host="127.0.0.1", port=port, node_id="node-1")
        result = await client.connect()
        assert result is True
        assert client.is_connected is True

        await client.disconnect()
        assert client.is_connected is False

        await server.stop()

    @pytest.mark.asyncio
    async def test_send_receive_message(self):
        from asc.network.transport import TCPClient, TCPServer
        received_messages = []

        async def on_message(conn_id, envelope):
            received_messages.append(envelope)

        server = TCPServer(host="127.0.0.1", port=0, on_message=on_message)
        await server.start()
        port = server.port

        client = TCPClient(host="127.0.0.1", port=port, node_id="node-1")
        await client.connect()

        msg = Message(type=MessageType.HEARTBEAT, sender_id="node-1", payload={"status": "ok"})
        env = Envelope(channel=Channel.HEARTBEATS, message=msg)
        result = await client.send(env)
        assert result is True

        # 等待消息传递
        await asyncio.sleep(0.2)
        assert len(received_messages) == 1
        assert received_messages[0].message.type == MessageType.HEARTBEAT

        await client.disconnect()
        await server.stop()

    @pytest.mark.asyncio
    async def test_client_connect_to_nonexistent(self):
        """连接不存在的服务器应返回 False。"""
        from asc.network.transport import TCPClient
        client = TCPClient(host="127.0.0.1", port=59999, node_id="node-1")
        result = await client.connect()
        assert result is False

    @pytest.mark.asyncio
    async def test_client_send_when_disconnected(self):
        """未连接时发送消息应返回 False。"""
        from asc.network.transport import TCPClient
        client = TCPClient(host="127.0.0.1", port=59999, node_id="node-1")
        msg = Message(type=MessageType.HEARTBEAT, sender_id="node-1", payload={})
        env = Envelope(channel=Channel.HEARTBEATS, message=msg)
        result = await client.send(env)
        assert result is False

    @pytest.mark.asyncio
    async def test_server_broadcast(self):
        """服务器广播消息。"""
        from asc.network.transport import TCPClient, TCPServer
        client_received = []

        async def on_client_message(envelope):
            client_received.append(envelope)

        server = TCPServer(host="127.0.0.1", port=0)
        await server.start()
        port = server.port

        client = TCPClient(
            host="127.0.0.1", port=port, node_id="node-1", on_message=on_client_message
        )
        await client.connect()
        await asyncio.sleep(0.1)

        msg = Message(type=MessageType.HEARTBEAT, sender_id="server", payload={})
        env = Envelope(channel=Channel.HEARTBEATS, message=msg)
        await server.broadcast(env)
        await asyncio.sleep(0.2)

        assert len(client_received) == 1

        await client.disconnect()
        await server.stop()

    @pytest.mark.asyncio
    async def test_server_send_to_nonexistent_connection(self):
        """向不存在的连接发送消息应返回 False。"""
        from asc.network.transport import TCPServer
        server = TCPServer(host="127.0.0.1", port=0)
        await server.start()

        msg = Message(type=MessageType.HEARTBEAT, sender_id="server", payload={})
        env = Envelope(channel=Channel.HEARTBEATS, message=msg)
        result = await server.send("nonexistent-conn", env)
        assert result is False

        await server.stop()

    @pytest.mark.asyncio
    async def test_multiple_clients(self):
        """多个客户端连接。"""
        from asc.network.transport import TCPClient, TCPServer
        server = TCPServer(host="127.0.0.1", port=0)
        await server.start()
        port = server.port

        clients = []
        for i in range(3):
            client = TCPClient(host="127.0.0.1", port=port, node_id=f"node-{i}")
            await client.connect()
            clients.append(client)

        await asyncio.sleep(0.1)
        assert len(server.connections) == 3

        for client in clients:
            await client.disconnect()
        await server.stop()


# ======================================================================
# 4. network/discovery.py 测试
# ======================================================================


class TestDiscoveryMessage:
    """DiscoveryMessage 测试。"""

    def test_creation(self):
        msg = DiscoveryMessage(node_id="n1", ip="192.168.1.1", port=52415)
        assert msg.node_id == "n1"
        assert msg.magic == "ASC_DISCOVER"

    def test_to_json(self):
        msg = DiscoveryMessage(node_id="n1", ip="192.168.1.1", port=52415)
        j = msg.to_json()
        data = json.loads(j)
        assert data["magic"] == "ASC_DISCOVER"
        assert data["node_id"] == "n1"

    def test_from_json_valid(self):
        raw = json.dumps({
            "magic": "ASC_DISCOVER",
            "node_id": "n1",
            "ip": "192.168.1.1",
            "port": 52415,
        })
        msg = DiscoveryMessage.from_json(raw)
        assert msg is not None
        assert msg.node_id == "n1"

    def test_from_json_invalid_magic(self):
        raw = json.dumps({
            "magic": "WRONG_MAGIC",
            "node_id": "n1",
            "ip": "192.168.1.1",
            "port": 52415,
        })
        msg = DiscoveryMessage.from_json(raw)
        assert msg is None

    def test_from_json_invalid_json(self):
        msg = DiscoveryMessage.from_json("not json")
        assert msg is None

    def test_from_json_missing_fields(self):
        raw = json.dumps({"magic": "ASC_DISCOVER"})
        with pytest.raises(KeyError):
            DiscoveryMessage.from_json(raw)

    def test_roundtrip(self):
        msg = DiscoveryMessage(node_id="n1", ip="192.168.1.1", port=52415)
        j = msg.to_json()
        msg2 = DiscoveryMessage.from_json(j)
        assert msg2 is not None
        assert msg2.node_id == msg.node_id
        assert msg2.ip == msg.ip
        assert msg2.port == msg.port


class TestNodeDiscovery:
    """NodeDiscovery 测试。"""

    def test_creation(self):
        nd = NodeDiscovery(node_id="n1", port=52415)
        assert nd.node_id == "n1"
        assert nd.port == 52415

    def test_broadcast_address(self):
        nd = NodeDiscovery(node_id="n1", port=52415)
        host, port, family = nd.broadcast_address
        assert host == "<broadcast>"
        assert port == 52416  # port + 1
        assert family == 2  # AF_INET

    def test_parse_message_valid(self):
        nd = NodeDiscovery(node_id="n1", port=52415)
        raw = json.dumps({
            "magic": "ASC_DISCOVER",
            "node_id": "n2",
            "ip": "192.168.1.2",
            "port": 52416,
        }).encode("utf-8")
        msg = nd.parse_message(raw)
        assert msg is not None
        assert msg.node_id == "n2"

    def test_parse_message_self_ignored(self):
        """忽略自己发出的广播。"""
        nd = NodeDiscovery(node_id="n1", port=52415)
        raw = json.dumps({
            "magic": "ASC_DISCOVER",
            "node_id": "n1",
            "ip": "192.168.1.1",
            "port": 52415,
        }).encode("utf-8")
        msg = nd.parse_message(raw)
        assert msg is None

    def test_parse_message_invalid_magic(self):
        nd = NodeDiscovery(node_id="n1", port=52415)
        raw = json.dumps({
            "magic": "WRONG",
            "node_id": "n2",
            "ip": "1.1.1.1",
            "port": 80,
        }).encode("utf-8")
        msg = nd.parse_message(raw)
        assert msg is None

    def test_parse_message_invalid_utf8(self):
        nd = NodeDiscovery(node_id="n1", port=52415)
        msg = nd.parse_message(b"\xff\xfe")
        assert msg is None

    def test_scan_addresses(self):
        nd = NodeDiscovery(node_id="n1", port=52415)
        addrs = nd.scan_addresses("192.168.1")
        assert len(addrs) == 254
        assert addrs[0] == "192.168.1.1"
        assert addrs[-1] == "192.168.1.254"
