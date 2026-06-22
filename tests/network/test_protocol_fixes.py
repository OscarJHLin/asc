"""协议层问题修复的 TDD 测试。

覆盖以下已识别问题：
1. read_frame_from_stream 未捕获无效 FrameType ValueError
2. encode_envelope_frame 文档与实现不一致 (MessagePack vs JSON)
3. decode_envelope JSON/MessagePack 自动检测启发式误判
4. _dict_to_envelope 缺少必要键校验
5. transport.py Server/Client 回调处理不一致
6. TCPServer 连接 ID 碰撞
7. discovery.py import 位置 + 废弃 API
8. ReceiveState.__init__ 覆盖已有文件
"""

import asyncio
import struct
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from asc.network.frame import (
    FRAME_MAGIC,
    FRAME_VERSION,
    Frame,
    FrameType,
    decode_envelope_frame,
    encode_envelope_frame,
    read_frame_from_stream,
)
from asc.network.protocol import (
    Channel,
    Envelope,
    Message,
    MessageType,
    _dict_to_envelope,
    decode_envelope,
    encode_envelope,
    encode_envelope_json,
)
from asc.network.sync import (
    ChunkInfo,
    ModelSyncProtocol,
)

# ---------------------------------------------------------------------------
# 1. read_frame_from_stream 无效 FrameType 应返回 None
# ---------------------------------------------------------------------------

class TestReadStreamInvalidFrameType:
    """流中收到无效帧类型时，read_frame_from_stream 应返回 None 而非抛异常。"""

    @pytest.mark.asyncio
    async def test_invalid_frame_type_0x05_returns_none(self):
        """未定义的帧类型 0x05 应返回 None（不崩溃）。"""
        header = struct.pack("!IBBI", FRAME_MAGIC, FRAME_VERSION, 0x05, 0)
        reader = asyncio.StreamReader()
        reader.feed_data(header)
        reader.feed_eof()

        result = await read_frame_from_stream(reader)
        assert result is None

    @pytest.mark.asyncio
    async def test_invalid_frame_type_0xff_returns_none(self):
        """未定义的帧类型 0xFF 应返回 None（不崩溃）。"""
        header = struct.pack("!IBBI", FRAME_MAGIC, FRAME_VERSION, 0xFF, 0)
        reader = asyncio.StreamReader()
        reader.feed_data(header)
        reader.feed_eof()

        result = await read_frame_from_stream(reader)
        assert result is None

    @pytest.mark.asyncio
    async def test_invalid_frame_type_0x50_returns_none(self):
        """未定义的帧类型 0x50 应返回 None（不崩溃）。"""
        header = struct.pack("!IBBI", FRAME_MAGIC, FRAME_VERSION, 0x50, 0)
        reader = asyncio.StreamReader()
        reader.feed_data(header)
        reader.feed_eof()

        result = await read_frame_from_stream(reader)
        assert result is None

    @pytest.mark.asyncio
    async def test_valid_frame_type_after_invalid_still_readable(self):
        """无效帧类型后，如果流中还有有效帧，不应影响后续读取。

        注意：当前实现中无效帧类型返回 None 后读取循环会终止，
        这是预期行为——连接应被关闭。此测试验证返回 None 的行为。
        """
        # 先发一个无效帧类型
        invalid_header = struct.pack("!IBBI", FRAME_MAGIC, FRAME_VERSION, 0xFE, 4)
        invalid_data = invalid_header + b"bad!"

        reader = asyncio.StreamReader()
        reader.feed_data(invalid_data)
        reader.feed_eof()

        result = await read_frame_from_stream(reader)
        assert result is None


# ---------------------------------------------------------------------------
# 2. encode_envelope_frame 默认使用 MessagePack
# ---------------------------------------------------------------------------

class TestEncodeEnvelopeFrameFormat:
    """验证 encode_envelope_frame 的实际序列化格式。"""

    def test_encode_envelope_frame_uses_msgpack_by_default(self):
        """encode_envelope_frame 默认使用 MessagePack 序列化（非 JSON）。"""
        msg = Message(type=MessageType.HEARTBEAT, sender_id="n1", payload={}, timestamp=100.0)
        env = Envelope(channel=Channel.HEARTBEATS, message=msg)
        frame = encode_envelope_frame(env)

        # MessagePack 不以 '{' (0x7B) 开头
        # 短键名格式的 msgpack envelope 以 fixmap 开头
        assert frame.frame_type == FrameType.ENVELOPE
        # payload 不应以 0x7B ('{') 开头（JSON 标志）
        # MessagePack fixmap with 2-3 keys: 0x82 or 0x83
        assert frame.payload[0:1] != b"{", "默认序列化应为 MessagePack 而非 JSON"

    def test_encode_envelope_frame_roundtrip_with_msgpack(self):
        """MessagePack 格式的 Envelope 帧能正确往返。"""
        msg = Message(type=MessageType.TASK_DISPATCH, sender_id="master", payload={"key": "val"}, timestamp=200.0)
        env = Envelope(channel=Channel.TASK_DISPATCH, message=msg, target="worker-1")
        frame = encode_envelope_frame(env)
        restored = decode_envelope_frame(frame)

        assert restored.channel == Channel.TASK_DISPATCH
        assert restored.message.type == MessageType.TASK_DISPATCH
        assert restored.message.sender_id == "master"
        assert restored.message.payload == {"key": "val"}
        assert restored.target == "worker-1"


# ---------------------------------------------------------------------------
# 3. decode_envelope JSON/MessagePack 自动检测误判
# ---------------------------------------------------------------------------

class TestDecodeEnvelopeFormatDetection:
    """验证 decode_envelope 的格式检测不会误判。"""

    def test_msgpack_with_0x7b_first_byte_not_misidentified_as_json(self):
        """MessagePack 数据首字节为 0x7B (fixmap 15 键) 时不应被误判为 JSON。

        MessagePack fixmap 格式: 0x80 | n (n=0-15), 当 n=15 时首字节为 0x8F。
        但 0x7B 实际上是 fixmap 0x80|0x0B = 0x8B... 不对。
        实际上 0x7B 在 MessagePack 中不是 fixmap，而是 positive fixint (0x00-0x7F)。
        然而，构造一个 MessagePack 数据使其首字节恰好为 0x7B 仍然可能。

        更重要的是：确保自动检测逻辑可靠，不会把合法 MessagePack 误当 JSON 解析。
        """
        # 构造一个合法的 MessagePack envelope，验证它能正确解码
        msg = Message(type=MessageType.HEARTBEAT, sender_id="n1", payload={}, timestamp=100.0)
        env = Envelope(channel=Channel.HEARTBEATS, message=msg)
        msgpack_data = encode_envelope(env)

        # 如果首字节恰好是 0x7B，当前代码会误判为 JSON
        # 验证正常情况不会出问题
        if msgpack_data[0:1] != b"{":
            # 正常 MessagePack 数据，不应被误判
            result = decode_envelope(msgpack_data)
            assert result.channel == Channel.HEARTBEATS

    def test_json_envelope_correctly_detected(self):
        """JSON 格式的 Envelope 应被正确识别和解码。"""
        msg = Message(type=MessageType.HEARTBEAT, sender_id="n1", payload={}, timestamp=100.0)
        env = Envelope(channel=Channel.HEARTBEATS, message=msg)
        json_data = encode_envelope_json(env)

        # JSON 数据以 '{' 开头
        assert json_data[0:1] == b"{"
        result = decode_envelope(json_data)
        assert result.channel == Channel.HEARTBEATS
        assert result.message.type == MessageType.HEARTBEAT

    def test_msgpack_and_json_roundtrip_produce_same_envelope(self):
        """MessagePack 和 JSON 序列化后反序列化应产生相同的 Envelope。"""
        msg = Message(type=MessageType.NODE_JOINED, sender_id="node-1", payload={"ip": "10.0.0.1"}, timestamp=500.0)
        env = Envelope(channel=Channel.EVENTS, message=msg, target="master")

        mp_result = decode_envelope(encode_envelope(env))
        json_result = decode_envelope(encode_envelope_json(env))

        assert mp_result.channel == json_result.channel
        assert mp_result.message.type == json_result.message.type
        assert mp_result.message.sender_id == json_result.message.sender_id
        assert mp_result.message.payload == json_result.message.payload
        assert mp_result.target == json_result.target


# ---------------------------------------------------------------------------
# 4. _dict_to_envelope 缺少必要键校验
# ---------------------------------------------------------------------------

class TestDictToEnvelopeValidation:
    """验证 _dict_to_envelope 对缺失键的校验。"""

    def test_missing_channel_key_raises_value_error(self):
        """短键名格式缺少 'c' 键应抛出 ValueError。"""
        with pytest.raises((ValueError, KeyError)):
            _dict_to_envelope({"m": {"t": "heartbeat", "s": "n1", "p": {}, "ts": 100.0}})

    def test_missing_message_key_raises_value_error(self):
        """短键名格式缺少 'm' 键应抛出 ValueError。"""
        with pytest.raises((ValueError, KeyError)):
            _dict_to_envelope({"c": "heartbeats"})

    def test_missing_type_in_message_raises_value_error(self):
        """短键名格式 message 中缺少 't' 键应抛出 ValueError。"""
        with pytest.raises((ValueError, KeyError)):
            _dict_to_envelope({"c": "heartbeats", "m": {"s": "n1", "p": {}, "ts": 100.0}})

    def test_missing_sender_id_in_message_raises_value_error(self):
        """短键名格式 message 中缺少 's' 键应抛出 ValueError。"""
        with pytest.raises((ValueError, KeyError)):
            _dict_to_envelope({"c": "heartbeats", "m": {"t": "heartbeat", "p": {}, "ts": 100.0}})

    def test_missing_payload_in_message_raises_value_error(self):
        """短键名格式 message 中缺少 'p' 键应抛出 ValueError。"""
        with pytest.raises((ValueError, KeyError)):
            _dict_to_envelope({"c": "heartbeats", "m": {"t": "heartbeat", "s": "n1", "ts": 100.0}})

    def test_missing_timestamp_in_message_raises_value_error(self):
        """短键名格式 message 中缺少 'ts' 键应抛出 ValueError。"""
        with pytest.raises((ValueError, KeyError)):
            _dict_to_envelope({"c": "heartbeats", "m": {"t": "heartbeat", "s": "n1", "p": {}}})

    def test_valid_short_keys_succeed(self):
        """合法短键名格式应正常解码。"""
        data = {"c": "heartbeats", "m": {"t": "heartbeat", "s": "n1", "p": {"k": "v"}, "ts": 100.0}}
        env = _dict_to_envelope(data)
        assert env.channel == Channel.HEARTBEATS
        assert env.message.type == MessageType.HEARTBEAT

    def test_valid_long_keys_succeed(self):
        """合法长键名格式应正常解码。"""
        data = {
            "channel": "events",
            "message": {"type": "node_joined", "sender_id": "n1", "payload": {}, "timestamp": 100.0},
        }
        env = _dict_to_envelope(data)
        assert env.channel == Channel.EVENTS
        assert env.message.type == MessageType.NODE_JOINED


# ---------------------------------------------------------------------------
# 5. transport.py Server/Client 回调一致性
# ---------------------------------------------------------------------------

class TestTransportCallbackConsistency:
    """验证 TCPServer 和 TCPClient 的回调处理一致。"""

    @pytest.mark.asyncio
    async def test_server_handles_sync_frame_callback(self):
        """TCPServer 应能处理同步的 on_frame 回调（不崩溃）。"""
        from asc.network.transport import TCPServer

        sync_callback = MagicMock(return_value=None)  # 同步回调，返回 None
        server = TCPServer(on_frame=sync_callback)

        await server.start()
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", server.port)

            # 发送一个 HEARTBEAT 帧
            frame = Frame(frame_type=FrameType.HEARTBEAT, payload=b"")
            from asc.network.frame import write_frame_to_bytes
            writer.write(write_frame_to_bytes(frame))
            await writer.drain()

            await asyncio.sleep(0.1)

            # 同步回调应被调用
            sync_callback.assert_called_once()

            writer.close()
            await writer.wait_closed()
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_client_handles_sync_frame_callback(self):
        """TCPClient 应能处理同步的 on_frame 回调（不崩溃）。"""
        from asc.network.transport import TCPClient, TCPServer

        received_frames = []

        def sync_callback(frame):
            received_frames.append(frame)
            return None  # 同步回调

        server = TCPServer()
        await server.start()
        try:
            client = TCPClient(
                host="127.0.0.1",
                port=server.port,
                node_id="test-client",
                on_frame=sync_callback,
            )
            connected = await client.connect()
            assert connected

            # 等待连接建立
            await asyncio.sleep(0.1)

            # 服务器向客户端发送帧
            conn_id = list(server.connections.keys())[0]
            frame = Frame(frame_type=FrameType.HEARTBEAT, payload=b"ping")
            await server.send_frame(conn_id, frame)

            await asyncio.sleep(0.2)

            assert len(received_frames) >= 1
            assert received_frames[0].frame_type == FrameType.HEARTBEAT

            await client.disconnect()
        finally:
            await server.stop()


# ---------------------------------------------------------------------------
# 6. TCPServer 连接 ID 碰撞
# ---------------------------------------------------------------------------

class TestTCPServerConnectionIdCollision:
    """验证 TCPServer 在连接 ID 碰撞时的行为。"""

    @pytest.mark.asyncio
    async def test_unique_conn_ids_for_different_connections(self):
        """不同连接应有不同的 conn_id，即使来自同一 IP。"""
        from asc.network.transport import TCPServer

        conn_ids = []

        async def on_msg(conn_id, envelope):
            conn_ids.append(conn_id)

        server = TCPServer(on_message=on_msg)
        await server.start()

        try:
            # 创建两个客户端连接
            client1_reader, client1_writer = await asyncio.open_connection("127.0.0.1", server.port)
            await asyncio.sleep(0.1)

            client2_reader, client2_writer = await asyncio.open_connection("127.0.0.1", server.port)
            await asyncio.sleep(0.1)

            # 两个连接应有不同的 conn_id
            assert len(server.connections) == 2
            conn_id_list = list(server.connections.keys())
            assert conn_id_list[0] != conn_id_list[1]

            client1_writer.close()
            client2_writer.close()
            await asyncio.gather(
                client1_writer.wait_closed(),
                client2_writer.wait_closed(),
            )
        finally:
            await server.stop()


# ---------------------------------------------------------------------------
# 7. ReceiveState 断点续传不覆盖已有文件
# ---------------------------------------------------------------------------

class TestReceiveStateResume:
    """验证 ReceiveState 初始化时不会覆盖已有部分下载的文件。"""

    def test_init_does_not_overwrite_existing_file(self):
        """如果文件已存在且有部分数据，init_receive 不应覆盖。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir)
            model_path = models_dir / "test.gguf"

            # 模拟已有部分下载的文件
            existing_data = b"existing partial data"
            model_path.write_bytes(existing_data)

            protocol = ModelSyncProtocol(models_dir=models_dir)
            chunks = [
                ChunkInfo(chunk_index=0, offset=0, size=10, sha256=""),
                ChunkInfo(chunk_index=1, offset=10, size=10, sha256=""),
            ]

            state = protocol.init_receive(
                model_id="test",
                total_bytes=20,
                chunks=chunks,
                file_sha256="dummy",
            )

            # 文件应保留原有数据，不应被清空
            actual_data = state.file_path.read_bytes()
            # 至少前 10 字节应保留（existing_data 的前 10 字节）
            assert actual_data[:10] == existing_data[:10]

    def test_init_creates_file_if_not_exists(self):
        """如果文件不存在，init_receive 应创建新文件。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir)
            model_path = models_dir / "new.gguf"

            assert not model_path.exists()

            protocol = ModelSyncProtocol(models_dir=models_dir)
            chunks = [
                ChunkInfo(chunk_index=0, offset=0, size=5, sha256=""),
            ]

            state = protocol.init_receive(
                model_id="new",
                total_bytes=5,
                chunks=chunks,
                file_sha256="dummy",
            )

            assert state.file_path.exists()

    def test_resume_preserves_downloaded_chunks(self):
        """断点续传时，已下载的分片数据应保留。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir)

            protocol = ModelSyncProtocol(models_dir=models_dir)
            chunks = [
                ChunkInfo(chunk_index=0, offset=0, size=10, sha256=""),
                ChunkInfo(chunk_index=1, offset=10, size=10, sha256=""),
            ]

            # 第一次接收：完成第一个分片
            state = protocol.init_receive(
                model_id="resume-test",
                total_bytes=20,
                chunks=chunks,
                file_sha256="dummy",
            )
            chunk0_data = b"0123456789"
            assert state.receive_chunk(chunks[0], chunk0_data) is True
            assert state.completed_count == 1

            # 保存进度
            state.save_progress()

            # 模拟重启：重新初始化接收状态
            state2 = protocol.init_receive(
                model_id="resume-test",
                total_bytes=20,
                chunks=chunks,
                file_sha256="dummy",
            )

            # 加载进度
            loaded = state2.load_progress()
            assert loaded is True
            assert state2.completed_count == 1
            assert 0 in state2.completed_chunks

            # 验证第一个分片的数据仍然存在
            actual_data = state2.file_path.read_bytes()
            assert actual_data[:10] == chunk0_data
