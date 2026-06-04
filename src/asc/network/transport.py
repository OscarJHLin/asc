"""Asc TCP 传输层。

基于 asyncio 实现异步 TCP Server/Client，用于节点间通信。
协议格式：4 字节大端长度前缀 + JSON 数据。

设计原则：
- 异步非阻塞，适合事件驱动架构
- 长度前缀协议，解决 TCP 粘包问题
- 支持广播和定向发送
"""

from __future__ import annotations

import asyncio
import contextlib
import struct
from typing import Callable, Coroutine

from asc.network.protocol import Envelope, decode_envelope, encode_envelope

# 回调类型
OnMessageCallback = Callable[[str, Envelope], Coroutine[None, None, None]]
ClientOnMessageCallback = Callable[[Envelope], Coroutine[None, None, None]]


class TCPServer:
    """异步 TCP 服务器。

    接受多个客户端连接，接收消息并分发到回调。
    支持向特定连接或所有连接发送消息。
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 0,
        on_message: OnMessageCallback | None = None,
    ) -> None:
        self.host = host
        self._port = port
        self._on_message = on_message
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
        # 关闭所有客户端连接
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
        """向指定连接发送消息。"""
        writer = self._connections.get(conn_id)
        if writer is None:
            return False
        return await self._send_to_writer(writer, envelope)

    async def broadcast(self, envelope: Envelope) -> None:
        """向所有连接广播消息。"""
        data = encode_envelope(envelope)
        header = struct.pack("!I", len(data))
        for writer in list(self._connections.values()):
            try:
                writer.write(header + data)
                await writer.drain()
            except Exception:
                pass

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """处理新连接。"""
        conn_id = f"{writer.get_extra_info('peername')}"
        self._connections[conn_id] = writer

        try:
            while self._running:
                envelope = await self._read_envelope(reader)
                if envelope is None:
                    break
                if self._on_message is not None:
                    await self._on_message(conn_id, envelope)
        except (ConnectionResetError, asyncio.IncompleteReadError):
            pass
        finally:
            self._connections.pop(conn_id, None)
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    async def _read_envelope(self, reader: asyncio.StreamReader) -> Envelope | None:
        """从流中读取一个信封。"""
        try:
            header = await reader.readexactly(4)
        except asyncio.IncompleteReadError:
            return None

        length = struct.unpack("!I", header)[0]
        if length > 10 * 1024 * 1024:  # 10MB 上限
            return None

        try:
            data = await reader.readexactly(length)
        except asyncio.IncompleteReadError:
            return None

        return decode_envelope(data)

    async def _send_to_writer(self, writer: asyncio.StreamWriter, envelope: Envelope) -> bool:
        """向指定 writer 发送消息。"""
        try:
            data = encode_envelope(envelope)
            header = struct.pack("!I", len(data))
            writer.write(header + data)
            await writer.drain()
            return True
        except Exception:
            return False


class TCPClient:
    """异步 TCP 客户端。

    连接到远程 TCP 服务器，支持发送和接收消息。
    """

    def __init__(
        self,
        host: str,
        port: int,
        node_id: str,
        on_message: ClientOnMessageCallback | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.node_id = node_id
        self._on_message = on_message
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
            # 启动接收循环
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
        """发送消息到服务器。"""
        if not self._connected or self._writer is None:
            return False
        try:
            data = encode_envelope(envelope)
            header = struct.pack("!I", len(data))
            self._writer.write(header + data)
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
                header = await self._reader.readexactly(4)
                length = struct.unpack("!I", header)[0]
                if length > 10 * 1024 * 1024:
                    break
                data = await self._reader.readexactly(length)
                envelope = decode_envelope(data)
                if self._on_message is not None:
                    result = self._on_message(envelope)
                    if asyncio.iscoroutine(result):
                        await result
        except (asyncio.IncompleteReadError, ConnectionResetError, asyncio.CancelledError):
            pass
        finally:
            self._connected = False
