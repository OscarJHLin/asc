"""Asc 消息分派器。

根据 MessageType 将消息路由到对应 handler，
与 MessageRouter（按 Channel 路由）互补：
- MessageRouter：通道级别订阅/发布
- MessageDispatcher：消息类型级别分派

确保所有 27 种 MessageType 至少有日志 stub，
避免消息被静默丢弃。
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Callable

from asc.network.protocol import Envelope, MessageType

logger = logging.getLogger(__name__)

# 处理器类型：同步或异步
Handler = Callable[[Envelope], None]


class MessageDispatcher:
    """消息分派器 — 根据 MessageType 将消息路由到对应 handler。"""

    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        self._handlers: dict[MessageType, list[Callable]] = defaultdict(list)
        self._default_handler: Callable | None = None

    def register(self, msg_type: MessageType, handler: Callable) -> None:
        """注册消息处理器。"""
        self._handlers[msg_type].append(handler)

    def set_default_handler(self, handler: Callable) -> None:
        """设置默认处理器（未注册类型的消息）。"""
        self._default_handler = handler

    async def dispatch(self, envelope: Envelope) -> None:
        """分派消息到对应处理器。"""
        msg_type = envelope.message.type
        handlers = self._handlers.get(msg_type, [])

        if not handlers:
            if self._default_handler is not None:
                await self._invoke(self._default_handler, envelope)
            else:
                logger.warning(
                    "[%s] 未注册处理器，消息被丢弃: type=%s sender=%s",
                    self.node_id,
                    msg_type.value,
                    envelope.message.sender_id,
                )
            return

        for handler in handlers:
            await self._invoke(handler, envelope)

    @staticmethod
    async def _invoke(handler: Callable, envelope: Envelope) -> None:
        """调用处理器，自动处理同步/异步。"""
        result = handler(envelope)
        if asyncio.iscoroutine(result):
            await result

    def register_defaults(self) -> None:
        """注册默认处理器（stub + logging）。

        为所有 27 种 MessageType 注册日志 stub，
        确保所有消息至少被记录。
        """

        def _make_stub(msg_type: MessageType) -> Handler:
            def _stub(envelope: Envelope) -> None:
                logger.info(
                    "[%s] %s: sender=%s payload=%s",
                    self.node_id,
                    msg_type.value,
                    envelope.message.sender_id,
                    envelope.message.payload,
                )

            return _stub

        for msg_type in MessageType:
            if not self._handlers.get(msg_type):
                self._handlers[msg_type].append(_make_stub(msg_type))
