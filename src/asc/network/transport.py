"""Asc TCP 传输层。

基于 asyncio 实现的高性能异步 TCP Server/Client，用于节点间通信。
统一使用 Binary Frame 作为线协议，上层 API 保持 Envelope 级别不变，
底层自动完成 Envelope <-> Frame 的转换。

设计原则：
- 异步非阻塞：所有 IO 操作使用 await，适合事件驱动架构
- 统一协议：Binary Frame 解决 TCP 粘包问题，所有消息共享同一套编解码
- 双模式支持：同时支持 Envelope（高层消息）和原始 Frame（低层控制）
- 透明桥接：上层代码无需感知 Frame 存在，继续操作 Envelope 即可

关键实现细节：
- 使用 asyncio.start_server() 和 asyncio.open_connection() 建立连接
- 通过 read_frame_from_stream() 逐帧读取，自动处理粘包和半包
- 广播时遍历所有连接，逐个发送，失败连接静默忽略
- 连接断开时自动清理 _connections 字典，防止内存泄漏

线程安全：
    所有可变状态（_connections、_running）仅在 asyncio 事件循环中访问，
    无需额外锁。若未来改为多线程，需对 _connections 加锁。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Callable, Coroutine

from asc.network.frame import (
    Frame,
    FrameType,
    decode_envelope_frame,
    encode_envelope_frame,
    read_frame_from_stream,
    write_frame_to_bytes,
)
from asc.network.protocol import Envelope

logger = logging.getLogger(__name__)

# 回调类型
OnMessageCallback = Callable[[str, Envelope], Coroutine[None, None, None]]
OnFrameCallback = Callable[[str, Frame], Coroutine[None, None, None]]
ClientOnMessageCallback = Callable[[Envelope], Coroutine[None, None, None]]
ClientOnFrameCallback = Callable[[Frame], Coroutine[None, None, None]]


class TCPServer:
    """异步 TCP 服务器。

    接受多个客户端连接，接收消息并分发到回调。
    支持向特定连接或所有连接发送消息。
    底层使用 Binary Frame 协议，上层 API 仍为 Envelope。
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 0,
        on_message: OnMessageCallback | None = None,
        on_frame: OnFrameCallback | None = None,
    ) -> None:
        self.host = host
        self._port = port
        self._on_message = on_message
        self._on_frame = on_frame
        self._server: asyncio.Server | None = None
        self._connections: dict[str, asyncio.StreamWriter] = {}
        self._running = False

    @property
    def port(self) -> int:
        if self._server is not None:
            return self._server.sockets[0].getsockname()[1]
        return self._port

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def connections(self) -> dict[str, asyncio.StreamWriter]:
        return dict(self._connections)

    async def start(self) -> None:
        """启动 TCP 服务器。"""
        self._server = await asyncio.start_server(
            self._handle_connection,
            self.host,
            self._port,
        )
        self._running = True

    async def stop(self) -> None:
        """停止 TCP 服务器。"""
        self._running = False
        for _conn_id, writer in list(self._connections.items()):
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
        self._connections.clear()

        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def send(self, conn_id: str, envelope: Envelope) -> bool:
        """向指定连接发送 Envelope 消息。"""
        writer = self._connections.get(conn_id)
        if writer is None:
            return False
        return await self._send_envelope_to_writer(writer, envelope)

    async def send_frame(self, conn_id: str, frame: Frame) -> bool:
        """向指定连接发送原始 Frame。"""
        writer = self._connections.get(conn_id)
        if writer is None:
            return False
        return await self._send_frame_to_writer(writer, frame)

    async def broadcast(self, envelope: Envelope) -> None:
        """向所有连接并发广播 Envelope 消息。"""
        frame = encode_envelope_frame(envelope)
        data = write_frame_to_bytes(frame)

        async def _send_to_writer(writer: asyncio.StreamWriter) -> None:
            try:
                writer.write(data)
                await writer.drain()
            except Exception:
                pass

        if not self._connections:
            return
        await asyncio.gather(
            *[_send_to_writer(w) for w in list(self._connections.values())],
            return_exceptions=True,
        )

    async def broadcast_frame(self, frame: Frame) -> None:
        """向所有连接并发广播原始 Frame。"""
        data = write_frame_to_bytes(frame)

        async def _send_to_writer(writer: asyncio.StreamWriter) -> None:
            try:
                writer.write(data)
                await writer.drain()
            except Exception:
                pass

        if not self._connections:
            return
        await asyncio.gather(
            *[_send_to_writer(w) for w in list(self._connections.values())],
            return_exceptions=True,
        )

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """处理新连接。"""
        conn_id = f"{writer.get_extra_info('peername')}"
        self._connections[conn_id] = writer

        try:
            while self._running:
                frame = await read_frame_from_stream(reader)
                if frame is None:
                    break

                # 原始帧回调
                if self._on_frame is not None:
                    await self._on_frame(conn_id, frame)

                # Envelope 回调 (仅 ENVELOPE 帧)
                if self._on_message is not None and frame.frame_type == FrameType.ENVELOPE:
                    try:
                        envelope = decode_envelope_frame(frame)
                        await self._on_message(conn_id, envelope)
                    except Exception:
                        logger.warning("处理 Envelope 帧异常: conn_id=%s", conn_id, exc_info=True)
        except (ConnectionResetError, asyncio.IncompleteReadError):
            pass
        finally:
            self._connections.pop(conn_id, None)
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    async def _send_envelope_to_writer(
        self, writer: asyncio.StreamWriter, envelope: Envelope
    ) -> bool:
        """向指定 writer 发送 Envelope 消息。"""
        try:
            frame = encode_envelope_frame(envelope)
            data = write_frame_to_bytes(frame)
            writer.write(data)
            await writer.drain()
            return True
        except Exception:
            return False

    async def _send_frame_to_writer(
        self, writer: asyncio.StreamWriter, frame: Frame
    ) -> bool:
        """向指定 writer 发送原始 Frame。"""
        try:
            data = write_frame_to_bytes(frame)
            writer.write(data)
            await writer.drain()
            return True
        except Exception:
            return False


class TCPClient:
    """异步 TCP 客户端。

    连接到远程 TCP 服务器，支持发送和接收消息。
    底层使用 Binary Frame 协议，上层 API 仍为 Envelope。
    """

    def __init__(
        self,
        host: str,
        port: int,
        node_id: str,
        on_message: ClientOnMessageCallback | None = None,
        on_frame: ClientOnFrameCallback | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.node_id = node_id
        self._on_message = on_message
        self._on_frame = on_frame
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._connected = False
        self._recv_task: asyncio.Task | None = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> bool:
        """连接到服务器。"""
        try:
            self._reader, self._writer = await asyncio.open_connection(self.host, self.port)
            self._connected = True
            self._recv_task = asyncio.create_task(self._recv_loop())
            return True
        except Exception:
            return False

    async def disconnect(self) -> None:
        """断开连接。"""
        self._connected = False
        if self._recv_task is not None:
            self._recv_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._recv_task
            self._recv_task = None

        if self._writer is not None:
            self._writer.close()
            with contextlib.suppress(Exception):
                await self._writer.wait_closed()
            self._writer = None
            self._reader = None

    async def send(self, envelope: Envelope) -> bool:
        """发送 Envelope 消息到服务器。"""
        if not self._connected or self._writer is None:
            return False
        try:
            frame = encode_envelope_frame(envelope)
            data = write_frame_to_bytes(frame)
            self._writer.write(data)
            await self._writer.drain()
            return True
        except Exception:
            self._connected = False
            return False

    async def send_frame(self, frame: Frame) -> bool:
        """发送原始 Frame 到服务器。"""
        if not self._connected or self._writer is None:
            return False
        try:
            data = write_frame_to_bytes(frame)
            self._writer.write(data)
            await self._writer.drain()
            return True
        except Exception:
            self._connected = False
            return False

    async def _recv_loop(self) -> None:
        """接收消息循环。"""
        assert self._reader is not None
        try:
            while self._connected:
                frame = await read_frame_from_stream(self._reader)
                if frame is None:
                    break

                # 原始帧回调
                if self._on_frame is not None:
                    result = self._on_frame(frame)
                    if asyncio.iscoroutine(result):
                        await result

                # Envelope 回调 (仅 ENVELOPE 帧)
                if self._on_message is not None and frame.frame_type == FrameType.ENVELOPE:
                    try:
                        envelope = decode_envelope_frame(frame)
                        result = self._on_message(envelope)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception:
                        logger.warning("客户端处理 Envelope 帧异常", exc_info=True)
        except (asyncio.IncompleteReadError, ConnectionResetError, asyncio.CancelledError):
            pass
        finally:
            self._connected = False
