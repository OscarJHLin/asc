"""Asc 消息路由器。

按通道（Channel）订阅/发布消息，是节点间消息分发的核心。
支持：
- 本地发布：消息分发给本地订阅者
- 远程发布：消息通过 TCP 发送到其他节点
- 收到远程消息：分发给本地订阅者（忽略自己发出的消息）
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Callable, Coroutine

from asc.network.protocol import Channel, Envelope

# 处理器类型：同步或异步
Handler = Callable[[Envelope], None | Coroutine[None, None, None]]
RemoteSender = Callable[[Envelope, str], Coroutine[None, None, bool]]


class MessageRouter:
    """消息路由器。

    按通道分发消息给订阅者，支持本地和远程路由。
    """

    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        self._subscribers: dict[Channel, list[Handler]] = defaultdict(list)
        self._remote_sender: RemoteSender | None = None

    def subscribe(self, channel: Channel, handler: Handler) -> None:
        """订阅指定通道的消息。"""
        self._subscribers[channel].append(handler)

    def unsubscribe(self, channel: Channel, handler: Handler) -> None:
        """取消订阅。"""
        handlers = self._subscribers.get(channel, [])
        if handler in handlers:
            handlers.remove(handler)

    def set_remote_sender(self, sender: RemoteSender) -> None:
        """设置远程消息发送器（TCP 传输层）。"""
        self._remote_sender = sender

    async def publish_local(self, envelope: Envelope) -> None:
        """本地发布消息，分发给订阅者。

        如果 handler 返回协程，使用 asyncio.create_task 调度执行，
        并添加异常回调避免静默丢弃错误。
        """
        handlers = self._subscribers.get(envelope.channel, [])
        for handler in handlers:
            result = handler(envelope)
            if asyncio.iscoroutine(result):
                task = asyncio.create_task(result)
                task.add_done_callback(self._on_handler_done)

    @staticmethod
    def _on_handler_done(task: asyncio.Task) -> None:
        """处理 handler 任务完成，记录异常。"""
        try:
            task.result()
        except Exception as exc:
            import logging
            logging.getLogger(__name__).exception("Message handler failed: %s", exc)

    async def publish_remote(self, envelope: Envelope, target: str) -> bool:
        """远程发布消息，通过 TCP 发送到目标节点。"""
        if self._remote_sender is None:
            return False
        return await self._remote_sender(envelope, target)

    async def handle_incoming(self, envelope: Envelope) -> None:
        """处理收到的远程消息。

        忽略自己发出的消息（避免回环），分发给本地订阅者。
        """
        if envelope.message.sender_id == self.node_id:
            return
        await self.publish_local(envelope)
