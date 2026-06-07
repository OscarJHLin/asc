"""增强测试消息路由器 MessageRouter。

覆盖原有测试未涉及的边界情况：
- 订阅边界（重复订阅、取消未订阅的处理器、空通道取消）
- 异步处理器支持
- 多通道交叉隔离
- 远程发布边界（返回 False、抛异常、替换 sender）
- 收到远程消息边界（不同 sender_id、同 node_id 回环过滤、多处理器分发）
- Envelope 集成（各种 Channel/MessageType 组合的完整往返）
- Node ID 边界（空字符串、Unicode）
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from asc.network.protocol import Channel, Envelope, Message, MessageType
from asc.network.router import MessageRouter

# ---------------------------------------------------------------------------
# 辅助：快速构造信封
# ---------------------------------------------------------------------------

def _make_env(
    channel: Channel = Channel.EVENTS,
    msg_type: MessageType = MessageType.NODE_JOINED,
    sender_id: str = "n2",
    payload: dict | None = None,
    target: str | None = None,
) -> Envelope:
    """构造一个 Envelope 实例，减少样板代码。"""
    return Envelope(
        channel=channel,
        message=Message(type=msg_type, sender_id=sender_id, payload=payload or {}),
        target=target,
    )


# ===================================================================
# 1. 订阅边界
# ===================================================================


class TestSubscribeEdgeCases:
    """订阅机制边界情况。"""

    def test_subscribe_same_handler_twice(self):
        """同一处理器重复订阅同一通道，应被添加两次（不做去重）。"""
        router = MessageRouter(node_id="n1")
        handler = MagicMock()
        router.subscribe(Channel.EVENTS, handler)
        router.subscribe(Channel.EVENTS, handler)

        # router.subscribe 不做去重，直接 append
        assert router._subscribers[Channel.EVENTS].count(handler) == 2

    @pytest.mark.asyncio
    async def test_subscribe_same_handler_twice_receives_twice(self):
        """重复订阅的处理器在 publish_local 时被调用两次。"""
        router = MessageRouter(node_id="n1")
        handler = MagicMock()
        router.subscribe(Channel.EVENTS, handler)
        router.subscribe(Channel.EVENTS, handler)

        env = _make_env()
        await router.publish_local(env)

        assert handler.call_count == 2

    def test_unsubscribe_handler_never_subscribed(self):
        """取消订阅一个从未订阅的处理器，不应抛异常。"""
        router = MessageRouter(node_id="n1")
        handler = MagicMock()
        # 不应抛异常
        router.unsubscribe(Channel.EVENTS, handler)

    def test_unsubscribe_from_channel_with_no_subscribers(self):
        """取消订阅一个没有任何订阅者的通道，不应抛异常。"""
        router = MessageRouter(node_id="n1")
        handler = MagicMock()
        # 不应抛异常
        router.unsubscribe(Channel.HEARTBEATS, handler)

    def test_unsubscribe_removes_only_first_occurrence(self):
        """如果同一处理器被订阅两次，unsubscribe 只移除第一次出现。"""
        router = MessageRouter(node_id="n1")
        handler = MagicMock()
        router.subscribe(Channel.EVENTS, handler)
        router.subscribe(Channel.EVENTS, handler)

        router.unsubscribe(Channel.EVENTS, handler)

        # 还剩一次
        assert router._subscribers[Channel.EVENTS].count(handler) == 1


# ===================================================================
# 2. 异步处理器支持
# ===================================================================


class TestAsyncHandler:
    """异步处理器（协程函数）支持。"""

    def test_subscribe_async_handler(self):
        """订阅异步处理器到通道，应正常添加。"""
        router = MessageRouter(node_id="n1")
        async_handler = AsyncMock()
        router.subscribe(Channel.EVENTS, async_handler)

        assert async_handler in router._subscribers[Channel.EVENTS]

    @pytest.mark.asyncio
    async def test_publish_local_with_async_handler_fire_and_forget(self):
        """publish_local 调用异步处理器时，检测到协程并用 create_task 调度。"""
        router = MessageRouter(node_id="n1")
        async_handler = AsyncMock()
        router.subscribe(Channel.EVENTS, async_handler)

        env = _make_env()
        await router.publish_local(env)

        # AsyncMock 被调用（返回协程），publish_local 检测到协程并 create_task
        async_handler.assert_called_once_with(env)

    @pytest.mark.asyncio
    async def test_publish_local_mixed_sync_and_async_handlers(self):
        """同一通道同时有同步和异步处理器，publish_local 都能调用。"""
        router = MessageRouter(node_id="n1")
        sync_handler = MagicMock()
        async_handler = AsyncMock()
        router.subscribe(Channel.EVENTS, sync_handler)
        router.subscribe(Channel.EVENTS, async_handler)

        env = _make_env()
        await router.publish_local(env)

        sync_handler.assert_called_once_with(env)
        async_handler.assert_called_once_with(env)


# ===================================================================
# 3. 多通道交叉隔离
# ===================================================================


class TestMultipleChannels:
    """多通道场景：隔离与交叉订阅。"""

    def test_same_handler_subscribed_to_multiple_channels(self):
        """同一处理器订阅多个通道，应出现在各通道的订阅列表中。"""
        router = MessageRouter(node_id="n1")
        handler = MagicMock()
        router.subscribe(Channel.EVENTS, handler)
        router.subscribe(Channel.HEARTBEATS, handler)

        assert handler in router._subscribers[Channel.EVENTS]
        assert handler in router._subscribers[Channel.HEARTBEATS]

    @pytest.mark.asyncio
    async def test_message_on_one_channel_does_not_trigger_handler_on_another(self):
        """发布到通道 A 的消息不会触发通道 B 的处理器。"""
        router = MessageRouter(node_id="n1")
        events_handler = MagicMock()
        heartbeat_handler = MagicMock()
        router.subscribe(Channel.EVENTS, events_handler)
        router.subscribe(Channel.HEARTBEATS, heartbeat_handler)

        env = _make_env(channel=Channel.EVENTS)
        await router.publish_local(env)

        events_handler.assert_called_once_with(env)
        heartbeat_handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_different_handlers_on_different_channels_receive_correct_messages(self):
        """不同通道的处理器各自收到自己通道的消息。"""
        router = MessageRouter(node_id="n1")
        events_handler = MagicMock()
        heartbeat_handler = MagicMock()
        command_handler = MagicMock()
        router.subscribe(Channel.EVENTS, events_handler)
        router.subscribe(Channel.HEARTBEATS, heartbeat_handler)
        router.subscribe(Channel.COMMANDS, command_handler)

        env_events = _make_env(channel=Channel.EVENTS, msg_type=MessageType.NODE_JOINED)
        env_heartbeat = _make_env(channel=Channel.HEARTBEATS, msg_type=MessageType.HEARTBEAT)
        env_command = _make_env(channel=Channel.COMMANDS, msg_type=MessageType.CREATE_INSTANCE)

        await router.publish_local(env_events)
        await router.publish_local(env_heartbeat)
        await router.publish_local(env_command)

        events_handler.assert_called_once_with(env_events)
        heartbeat_handler.assert_called_once_with(env_heartbeat)
        command_handler.assert_called_once_with(env_command)

    @pytest.mark.asyncio
    async def test_shared_handler_receives_from_all_subscribed_channels(self):
        """同一处理器订阅多个通道时，各通道消息都能触发它。"""
        router = MessageRouter(node_id="n1")
        handler = MagicMock()
        router.subscribe(Channel.EVENTS, handler)
        router.subscribe(Channel.HEARTBEATS, handler)

        env1 = _make_env(channel=Channel.EVENTS)
        env2 = _make_env(channel=Channel.HEARTBEATS)

        await router.publish_local(env1)
        await router.publish_local(env2)

        assert handler.call_count == 2
        handler.assert_any_call(env1)
        handler.assert_any_call(env2)


# ===================================================================
# 4. 远程发布边界
# ===================================================================


class TestRemotePublishEdgeCases:
    """远程消息发布边界情况。"""

    @pytest.mark.asyncio
    async def test_publish_remote_returns_false_when_sender_returns_false(self):
        """remote_sender 返回 False 时，publish_remote 应返回 False。"""
        router = MessageRouter(node_id="n1")
        mock_send = AsyncMock(return_value=False)
        router.set_remote_sender(mock_send)

        env = _make_env()
        result = await router.publish_remote(env, target="n2")

        assert result is False
        mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_publish_remote_propagates_exception_from_sender(self):
        """remote_sender 抛出异常时，publish_remote 应向上传播。"""
        router = MessageRouter(node_id="n1")
        mock_send = AsyncMock(side_effect=ConnectionError("连接断开"))
        router.set_remote_sender(mock_send)

        env = _make_env()
        with pytest.raises(ConnectionError, match="连接断开"):
            await router.publish_remote(env, target="n2")

    @pytest.mark.asyncio
    async def test_set_remote_sender_replaces_previous(self):
        """set_remote_sender 替换之前的发送器。"""
        router = MessageRouter(node_id="n1")
        old_sender = AsyncMock(return_value=True)
        new_sender = AsyncMock(return_value=True)

        router.set_remote_sender(old_sender)
        router.set_remote_sender(new_sender)

        env = _make_env()
        await router.publish_remote(env, target="n2")

        old_sender.assert_not_called()
        new_sender.assert_called_once()

    @pytest.mark.asyncio
    async def test_publish_remote_passes_envelope_and_target(self):
        """publish_remote 正确传递 envelope 和 target 参数。"""
        router = MessageRouter(node_id="n1")
        mock_send = AsyncMock(return_value=True)
        router.set_remote_sender(mock_send)

        env = _make_env()
        await router.publish_remote(env, target="node-xyz")

        mock_send.assert_called_once_with(env, "node-xyz")

    @pytest.mark.asyncio
    async def test_publish_remote_without_sender_returns_false(self):
        """未设置 remote_sender 时，publish_remote 返回 False。"""
        router = MessageRouter(node_id="n1")
        env = _make_env()
        result = await router.publish_remote(env, target="n2")
        assert result is False


# ===================================================================
# 5. 收到远程消息边界
# ===================================================================


class TestHandleIncomingEdgeCases:
    """处理收到的远程消息边界情况。"""

    @pytest.mark.asyncio
    async def test_handle_incoming_from_different_sender_id_dispatches(self):
        """来自不同 sender_id 的消息应被正常分发。"""
        router = MessageRouter(node_id="n1")
        handler = MagicMock()
        router.subscribe(Channel.EVENTS, handler)

        env = _make_env(sender_id="n99")
        await router.handle_incoming(env)

        handler.assert_called_once_with(env)

    @pytest.mark.asyncio
    async def test_handle_incoming_from_same_node_id_is_ignored(self):
        """来自与 router 相同 node_id 的消息应被忽略（回环防护）。"""
        router = MessageRouter(node_id="n1")
        handler = MagicMock()
        router.subscribe(Channel.EVENTS, handler)

        env = _make_env(sender_id="n1")
        await router.handle_incoming(env)

        handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_incoming_multiple_handlers_all_receive(self):
        """同一通道上的多个处理器都应收到远程消息。"""
        router = MessageRouter(node_id="n1")
        h1 = MagicMock()
        h2 = MagicMock()
        h3 = MagicMock()
        router.subscribe(Channel.EVENTS, h1)
        router.subscribe(Channel.EVENTS, h2)
        router.subscribe(Channel.EVENTS, h3)

        env = _make_env(sender_id="n2")
        await router.handle_incoming(env)

        h1.assert_called_once_with(env)
        h2.assert_called_once_with(env)
        h3.assert_called_once_with(env)

    @pytest.mark.asyncio
    async def test_handle_incoming_no_subscribers_is_noop(self):
        """收到消息但通道无订阅者时，不应抛异常。"""
        router = MessageRouter(node_id="n1")
        env = _make_env(channel=Channel.ELECTION)
        # 不应抛异常
        await router.handle_incoming(env)

    @pytest.mark.asyncio
    async def test_handle_incoming_only_filters_matching_node_id(self):
        """只有 sender_id 完全匹配 node_id 时才过滤，部分匹配不过滤。"""
        router = MessageRouter(node_id="n1")
        handler = MagicMock()
        router.subscribe(Channel.EVENTS, handler)

        # sender_id 包含 node_id 但不完全相同
        env = _make_env(sender_id="n10")
        await router.handle_incoming(env)

        handler.assert_called_once_with(env)


# ===================================================================
# 6. Envelope 集成
# ===================================================================


class TestEnvelopeIntegration:
    """Envelope 与路由器的完整集成测试。"""

    @pytest.mark.asyncio
    async def test_roundtrip_events_channel(self):
        """EVENTS 通道的完整往返：构造 → 本地发布 → 处理器收到。"""
        router = MessageRouter(node_id="n1")
        handler = MagicMock()
        router.subscribe(Channel.EVENTS, handler)

        env = Envelope(
            channel=Channel.EVENTS,
            message=Message(
                type=MessageType.NODE_LEFT,
                sender_id="n2",
                payload={"reason": "shutdown"},
            ),
        )
        await router.publish_local(env)

        handler.assert_called_once_with(env)
        received_env = handler.call_args[0][0]
        assert received_env.channel == Channel.EVENTS
        assert received_env.message.type == MessageType.NODE_LEFT
        assert received_env.message.payload == {"reason": "shutdown"}

    @pytest.mark.asyncio
    async def test_roundtrip_commands_channel(self):
        """COMMANDS 通道的完整往返。"""
        router = MessageRouter(node_id="n1")
        handler = MagicMock()
        router.subscribe(Channel.COMMANDS, handler)

        env = Envelope(
            channel=Channel.COMMANDS,
            message=Message(
                type=MessageType.CREATE_INSTANCE,
                sender_id="n3",
                payload={"model": "gpt-4"},
            ),
        )
        await router.publish_local(env)

        handler.assert_called_once_with(env)
        received_env = handler.call_args[0][0]
        assert received_env.channel == Channel.COMMANDS
        assert received_env.message.type == MessageType.CREATE_INSTANCE

    @pytest.mark.asyncio
    async def test_roundtrip_heartbeat_channel(self):
        """HEARTBEATS 通道的完整往返。"""
        router = MessageRouter(node_id="n1")
        handler = MagicMock()
        router.subscribe(Channel.HEARTBEATS, handler)

        env = Envelope(
            channel=Channel.HEARTBEATS,
            message=Message(
                type=MessageType.HEARTBEAT,
                sender_id="n2",
                payload={"load": 0.5},
            ),
        )
        await router.publish_local(env)

        handler.assert_called_once()
        assert handler.call_args[0][0].channel == Channel.HEARTBEATS

    @pytest.mark.asyncio
    async def test_roundtrip_election_channel(self):
        """ELECTION 通道的完整往返。"""
        router = MessageRouter(node_id="n1")
        handler = MagicMock()
        router.subscribe(Channel.ELECTION, handler)

        env = Envelope(
            channel=Channel.ELECTION,
            message=Message(
                type=MessageType.ELECTION,
                sender_id="n5",
                payload={"candidate": "n5"},
            ),
        )
        await router.publish_local(env)

        handler.assert_called_once()
        assert handler.call_args[0][0].message.type == MessageType.ELECTION

    @pytest.mark.asyncio
    async def test_roundtrip_task_dispatch_channel(self):
        """TASK_DISPATCH 通道的完整往返。"""
        router = MessageRouter(node_id="n1")
        handler = MagicMock()
        router.subscribe(Channel.TASK_DISPATCH, handler)

        env = Envelope(
            channel=Channel.TASK_DISPATCH,
            message=Message(
                type=MessageType.TASK_DISPATCH,
                sender_id="coordinator",
                payload={"task_id": "t-001"},
            ),
        )
        await router.publish_local(env)

        handler.assert_called_once()
        assert handler.call_args[0][0].channel == Channel.TASK_DISPATCH

    @pytest.mark.asyncio
    async def test_roundtrip_capacity_channel(self):
        """CAPACITY 通道的完整往返。"""
        router = MessageRouter(node_id="n1")
        handler = MagicMock()
        router.subscribe(Channel.CAPACITY, handler)

        env = Envelope(
            channel=Channel.CAPACITY,
            message=Message(
                type=MessageType.CAPACITY_REPORT,
                sender_id="n2",
                payload={"gpu_free": 3},
            ),
        )
        await router.publish_local(env)

        handler.assert_called_once()
        assert handler.call_args[0][0].message.type == MessageType.CAPACITY_REPORT

    @pytest.mark.asyncio
    async def test_roundtrip_rebalance_channel(self):
        """REBALANCE 通道的完整往返。"""
        router = MessageRouter(node_id="n1")
        handler = MagicMock()
        router.subscribe(Channel.REBALANCE, handler)

        env = Envelope(
            channel=Channel.REBALANCE,
            message=Message(
                type=MessageType.REBALANCE_REQUEST,
                sender_id="leader",
                payload={"plan": ["n1", "n2"]},
            ),
        )
        await router.publish_local(env)

        handler.assert_called_once()
        assert handler.call_args[0][0].message.type == MessageType.REBALANCE_REQUEST

    @pytest.mark.asyncio
    async def test_handle_incoming_dispatches_correct_channel(self):
        """handle_incoming 将消息分发到正确通道的订阅者。"""
        router = MessageRouter(node_id="n1")
        events_handler = MagicMock()
        command_handler = MagicMock()
        router.subscribe(Channel.EVENTS, events_handler)
        router.subscribe(Channel.COMMANDS, command_handler)

        env = Envelope(
            channel=Channel.COMMANDS,
            message=Message(
                type=MessageType.SHUTDOWN_RUNNER,
                sender_id="n3",
                payload={},
            ),
        )
        await router.handle_incoming(env)

        command_handler.assert_called_once_with(env)
        events_handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_envelope_with_target_field(self):
        """带 target 字段的 Envelope 也能正常路由。"""
        router = MessageRouter(node_id="n1")
        handler = MagicMock()
        router.subscribe(Channel.EVENTS, handler)

        env = Envelope(
            channel=Channel.EVENTS,
            message=Message(
                type=MessageType.NODE_JOINED,
                sender_id="n2",
                payload={},
            ),
            target="n3",
        )
        await router.publish_local(env)

        handler.assert_called_once_with(env)
        assert handler.call_args[0][0].target == "n3"


# ===================================================================
# 7. Node ID 边界
# ===================================================================


class TestNodeIdHandling:
    """Node ID 边界情况。"""

    @pytest.mark.asyncio
    async def test_router_with_empty_string_node_id(self):
        """空字符串 node_id 的路由器能正常工作。"""
        router = MessageRouter(node_id="")
        handler = MagicMock()
        router.subscribe(Channel.EVENTS, handler)

        # sender_id 为空时应被过滤（回环防护）
        env = _make_env(sender_id="")
        await router.handle_incoming(env)
        handler.assert_not_called()

        # sender_id 非空时应正常分发
        env2 = _make_env(sender_id="n2")
        await router.handle_incoming(env2)
        handler.assert_called_once_with(env2)

    @pytest.mark.asyncio
    async def test_router_with_unicode_node_id(self):
        """Unicode node_id 的路由器能正常工作。"""
        router = MessageRouter(node_id="节点-🚀")
        handler = MagicMock()
        router.subscribe(Channel.EVENTS, handler)

        # 来自自身的消息应被过滤
        env = _make_env(sender_id="节点-🚀")
        await router.handle_incoming(env)
        handler.assert_not_called()

        # 来自其他节点的消息应正常分发
        env2 = _make_env(sender_id="节点-🔧")
        await router.handle_incoming(env2)
        handler.assert_called_once_with(env2)

    @pytest.mark.asyncio
    async def test_router_with_unicode_node_id_publish_local(self):
        """Unicode node_id 的路由器本地发布正常。"""
        router = MessageRouter(node_id="节点-🚀")
        handler = MagicMock()
        router.subscribe(Channel.HEARTBEATS, handler)

        env = _make_env(channel=Channel.HEARTBEATS, sender_id="节点-🚀")
        await router.publish_local(env)

        handler.assert_called_once_with(env)

    @pytest.mark.asyncio
    async def test_router_with_unicode_node_id_remote_publish(self):
        """Unicode node_id 的路由器远程发布正常。"""
        router = MessageRouter(node_id="节点-🚀")
        mock_send = AsyncMock(return_value=True)
        router.set_remote_sender(mock_send)

        env = _make_env(sender_id="节点-🚀")
        result = await router.publish_remote(env, target="节点-🔧")

        assert result is True
        mock_send.assert_called_once_with(env, "节点-🔧")
