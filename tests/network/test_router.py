"""测试消息路由器 MessageRouter。

MessageRouter 是节点间消息分发的核心：
- 按通道（Channel）订阅/发布消息
- 支持本地和远程消息路由
- 与 TCP 传输层集成
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from asc.network.protocol import Channel, Envelope, Message, MessageType
from asc.network.router import MessageRouter


class TestMessageRouterSubscribe:
    """订阅机制。"""

    def test_subscribe_to_channel(self):
        router = MessageRouter(node_id="n1")
        handler = MagicMock()
        router.subscribe(Channel.EVENTS, handler)
        assert Channel.EVENTS in router._subscribers
        assert handler in router._subscribers[Channel.EVENTS]

    def test_subscribe_multiple_handlers(self):
        router = MessageRouter(node_id="n1")
        h1, h2 = MagicMock(), MagicMock()
        router.subscribe(Channel.EVENTS, h1)
        router.subscribe(Channel.EVENTS, h2)
        assert len(router._subscribers[Channel.EVENTS]) == 2

    def test_subscribe_multiple_channels(self):
        router = MessageRouter(node_id="n1")
        h1, h2 = MagicMock(), MagicMock()
        router.subscribe(Channel.EVENTS, h1)
        router.subscribe(Channel.HEARTBEATS, h2)
        assert len(router._subscribers) == 2

    def test_unsubscribe(self):
        router = MessageRouter(node_id="n1")
        handler = MagicMock()
        router.subscribe(Channel.EVENTS, handler)
        router.unsubscribe(Channel.EVENTS, handler)
        assert handler not in router._subscribers[Channel.EVENTS]


class TestMessageRouterPublish:
    """发布消息。"""

    def test_publish_calls_subscribers(self):
        router = MessageRouter(node_id="n1")
        handler = MagicMock()
        router.subscribe(Channel.EVENTS, handler)

        msg = Message(type=MessageType.NODE_JOINED, sender_id="n2", payload={})
        env = Envelope(channel=Channel.EVENTS, message=msg)
        router.publish_local(env)

        handler.assert_called_once_with(env)

    def test_publish_to_wrong_channel_does_not_call(self):
        router = MessageRouter(node_id="n1")
        handler = MagicMock()
        router.subscribe(Channel.EVENTS, handler)

        msg = Message(type=MessageType.HEARTBEAT, sender_id="n2", payload={})
        env = Envelope(channel=Channel.HEARTBEATS, message=msg)
        router.publish_local(env)

        handler.assert_not_called()

    def test_publish_calls_all_subscribers_on_channel(self):
        router = MessageRouter(node_id="n1")
        h1, h2 = MagicMock(), MagicMock()
        router.subscribe(Channel.EVENTS, h1)
        router.subscribe(Channel.EVENTS, h2)

        msg = Message(type=MessageType.NODE_JOINED, sender_id="n2", payload={})
        env = Envelope(channel=Channel.EVENTS, message=msg)
        router.publish_local(env)

        h1.assert_called_once_with(env)
        h2.assert_called_once_with(env)

    def test_publish_no_subscribers_is_noop(self):
        router = MessageRouter(node_id="n1")
        msg = Message(type=MessageType.HEARTBEAT, sender_id="n2", payload={})
        env = Envelope(channel=Channel.HEARTBEATS, message=msg)
        # 不应抛异常
        router.publish_local(env)


class TestMessageRouterRemotePublish:
    """远程消息发布（通过 TCP 发送到其他节点）。"""

    @pytest.mark.asyncio
    async def test_publish_remote_sends_to_transport(self):
        router = MessageRouter(node_id="n1")
        mock_send = AsyncMock(return_value=True)
        router.set_remote_sender(mock_send)

        msg = Message(type=MessageType.NODE_JOINED, sender_id="n1", payload={})
        env = Envelope(channel=Channel.EVENTS, message=msg)
        await router.publish_remote(env, target="n2")

        mock_send.assert_called_once()
        sent_env = mock_send.call_args[0][0]
        assert sent_env.message.type == MessageType.NODE_JOINED

    @pytest.mark.asyncio
    async def test_publish_remote_no_transport_is_noop(self):
        router = MessageRouter(node_id="n1")
        msg = Message(type=MessageType.HEARTBEAT, sender_id="n1", payload={})
        env = Envelope(channel=Channel.HEARTBEATS, message=msg)
        # 不应抛异常
        await router.publish_remote(env, target="n2")


class TestMessageRouterIncoming:
    """处理收到的远程消息。"""

    def test_handle_incoming_dispatches_to_channel(self):
        router = MessageRouter(node_id="n1")
        handler = MagicMock()
        router.subscribe(Channel.EVENTS, handler)

        msg = Message(type=MessageType.NODE_JOINED, sender_id="n2", payload={"ip": "10.0.0.1"})
        env = Envelope(channel=Channel.EVENTS, message=msg)
        router.handle_incoming(env)

        handler.assert_called_once_with(env)

    def test_handle_incoming_ignores_own_messages(self):
        """忽略自己发出的消息。"""
        router = MessageRouter(node_id="n1")
        handler = MagicMock()
        router.subscribe(Channel.EVENTS, handler)

        msg = Message(type=MessageType.NODE_JOINED, sender_id="n1", payload={})
        env = Envelope(channel=Channel.EVENTS, message=msg)
        router.handle_incoming(env)

        handler.assert_not_called()
