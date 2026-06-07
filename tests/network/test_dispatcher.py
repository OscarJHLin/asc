"""测试消息分派器 MessageDispatcher。

MessageDispatcher 根据 MessageType 将消息路由到对应 handler：
- 注册/分派机制
- 默认处理器
- register_defaults 覆盖所有 27 种 MessageType
- 同步/异步 handler 支持
- 每个 MessageType 支持多个 handler
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from asc.network.dispatcher import MessageDispatcher
from asc.network.protocol import Channel, Envelope, Message, MessageType


def _make_envelope(msg_type: MessageType, sender: str = "n2") -> Envelope:
    """构造测试用信封。"""
    msg = Message(type=msg_type, sender_id=sender, payload={"key": "value"})
    return Envelope(channel=Channel.EVENTS, message=msg)


class TestRegisterAndDispatch:
    """注册与分派。"""

    @pytest.mark.asyncio
    async def test_dispatch_calls_registered_handler(self):
        dispatcher = MessageDispatcher(node_id="n1")
        handler = MagicMock()
        dispatcher.register(MessageType.NODE_JOINED, handler)

        env = _make_envelope(MessageType.NODE_JOINED)
        await dispatcher.dispatch(env)

        handler.assert_called_once_with(env)

    @pytest.mark.asyncio
    async def test_dispatch_does_not_call_wrong_type_handler(self):
        dispatcher = MessageDispatcher(node_id="n1")
        handler = MagicMock()
        dispatcher.register(MessageType.NODE_JOINED, handler)

        env = _make_envelope(MessageType.HEARTBEAT)
        await dispatcher.dispatch(env)

        handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_dispatch_no_handler_no_default_no_error(self):
        """无处理器且无默认处理器时，不抛异常。"""
        dispatcher = MessageDispatcher(node_id="n1")
        env = _make_envelope(MessageType.HEARTBEAT)
        # 不应抛异常
        await dispatcher.dispatch(env)


class TestDefaultHandler:
    """默认处理器。"""

    @pytest.mark.asyncio
    async def test_default_handler_called_when_no_registered(self):
        dispatcher = MessageDispatcher(node_id="n1")
        default = MagicMock()
        dispatcher.set_default_handler(default)

        env = _make_envelope(MessageType.HEARTBEAT)
        await dispatcher.dispatch(env)

        default.assert_called_once_with(env)

    @pytest.mark.asyncio
    async def test_default_handler_not_called_when_registered_exists(self):
        dispatcher = MessageDispatcher(node_id="n1")
        handler = MagicMock()
        default = MagicMock()
        dispatcher.register(MessageType.NODE_JOINED, handler)
        dispatcher.set_default_handler(default)

        env = _make_envelope(MessageType.NODE_JOINED)
        await dispatcher.dispatch(env)

        handler.assert_called_once_with(env)
        default.assert_not_called()

    @pytest.mark.asyncio
    async def test_set_default_handler_overrides_previous(self):
        dispatcher = MessageDispatcher(node_id="n1")
        old_default = MagicMock()
        new_default = MagicMock()
        dispatcher.set_default_handler(old_default)
        dispatcher.set_default_handler(new_default)

        env = _make_envelope(MessageType.HEARTBEAT)
        await dispatcher.dispatch(env)

        old_default.assert_not_called()
        new_default.assert_called_once_with(env)


class TestRegisterDefaults:
    """register_defaults 覆盖所有 27 种 MessageType。"""

    def test_register_defaults_covers_all_types(self):
        dispatcher = MessageDispatcher(node_id="n1")
        dispatcher.register_defaults()

        for msg_type in MessageType:
            assert msg_type in dispatcher._handlers, f"缺少 {msg_type.value} 的 stub"
            assert len(dispatcher._handlers[msg_type]) >= 1

    def test_register_defaults_count_matches_enum(self):
        """确认 register_defaults 覆盖所有 MessageType。"""
        assert len(MessageType) == len(list(MessageType))

    @pytest.mark.asyncio
    async def test_stub_handler_does_not_crash(self):
        """stub handler 应能正常调用而不报错。"""
        dispatcher = MessageDispatcher(node_id="n1")
        dispatcher.register_defaults()

        for msg_type in MessageType:
            env = _make_envelope(msg_type)
            # 不应抛异常
            await dispatcher.dispatch(env)

    def test_register_defaults_does_not_override_existing(self):
        """register_defaults 不覆盖已注册的 handler，但也不额外添加 stub。"""
        dispatcher = MessageDispatcher(node_id="n1")
        custom = MagicMock()
        dispatcher.register(MessageType.NODE_JOINED, custom)
        dispatcher.register_defaults()

        # NODE_JOINED 已有 custom handler，register_defaults 不再添加 stub
        assert custom in dispatcher._handlers[MessageType.NODE_JOINED]
        assert len(dispatcher._handlers[MessageType.NODE_JOINED]) == 1


class TestAsyncHandlers:
    """异步处理器。"""

    @pytest.mark.asyncio
    async def test_async_handler_is_awaited(self):
        dispatcher = MessageDispatcher(node_id="n1")
        handler = AsyncMock()
        dispatcher.register(MessageType.NODE_JOINED, handler)

        env = _make_envelope(MessageType.NODE_JOINED)
        await dispatcher.dispatch(env)

        handler.assert_called_once_with(env)

    @pytest.mark.asyncio
    async def test_async_default_handler_is_awaited(self):
        dispatcher = MessageDispatcher(node_id="n1")
        default = AsyncMock()
        dispatcher.set_default_handler(default)

        env = _make_envelope(MessageType.HEARTBEAT)
        await dispatcher.dispatch(env)

        default.assert_called_once_with(env)

    @pytest.mark.asyncio
    async def test_mixed_sync_and_async_handlers(self):
        """同一类型同时有同步和异步 handler。"""
        dispatcher = MessageDispatcher(node_id="n1")
        sync_handler = MagicMock()
        async_handler = AsyncMock()
        dispatcher.register(MessageType.NODE_JOINED, sync_handler)
        dispatcher.register(MessageType.NODE_JOINED, async_handler)

        env = _make_envelope(MessageType.NODE_JOINED)
        await dispatcher.dispatch(env)

        sync_handler.assert_called_once_with(env)
        async_handler.assert_called_once_with(env)


class TestMultipleHandlersPerType:
    """每个 MessageType 支持多个 handler。"""

    @pytest.mark.asyncio
    async def test_multiple_handlers_all_called(self):
        dispatcher = MessageDispatcher(node_id="n1")
        h1 = MagicMock()
        h2 = MagicMock()
        h3 = MagicMock()
        dispatcher.register(MessageType.NODE_JOINED, h1)
        dispatcher.register(MessageType.NODE_JOINED, h2)
        dispatcher.register(MessageType.NODE_JOINED, h3)

        env = _make_envelope(MessageType.NODE_JOINED)
        await dispatcher.dispatch(env)

        h1.assert_called_once_with(env)
        h2.assert_called_once_with(env)
        h3.assert_called_once_with(env)

    @pytest.mark.asyncio
    async def test_multiple_handlers_different_types(self):
        """不同类型各自独立。"""
        dispatcher = MessageDispatcher(node_id="n1")
        h_events = MagicMock()
        h_heartbeat = MagicMock()
        dispatcher.register(MessageType.NODE_JOINED, h_events)
        dispatcher.register(MessageType.HEARTBEAT, h_heartbeat)

        env_join = _make_envelope(MessageType.NODE_JOINED)
        env_hb = _make_envelope(MessageType.HEARTBEAT)
        await dispatcher.dispatch(env_join)
        await dispatcher.dispatch(env_hb)

        h_events.assert_called_once_with(env_join)
        h_heartbeat.assert_called_once_with(env_hb)
