"""ASC 二进制帧协议增强测试。

覆盖帧头格式验证、解码错误场景、二进制 payload 边界、
流式读取边界、Envelope 桥接全面性、write_frame_to_bytes 一致性。
"""

import asyncio
import struct

import pytest

from asc.network.frame import (
    FRAME_HEADER_SIZE,
    FRAME_MAGIC,
    FRAME_VERSION,
    MAX_FRAME_SIZE,
    Frame,
    FrameType,
    decode_envelope_frame,
    decode_frame,
    encode_envelope_frame,
    encode_frame,
    read_frame_from_stream,
    write_frame_to_bytes,
)
from asc.network.protocol import (
    Channel,
    Envelope,
    Message,
    MessageType,
)

# ---------------------------------------------------------------------------
# 1. 帧头格式验证
# ---------------------------------------------------------------------------

class TestFrameHeaderFormat:
    """帧头字节布局精确验证。"""

    def test_header_exact_byte_layout(self):
        """验证帧头为 magic(4B big-endian) + version(1B) + type(1B) + length(4B big-endian)。"""
        frame = Frame(frame_type=FrameType.HEARTBEAT, payload=b"abc")
        data = encode_frame(frame)

        # 总长度 = 帧头 + payload
        assert len(data) == FRAME_HEADER_SIZE + 3

        # 前 4 字节: magic, big-endian
        magic_bytes = data[:4]
        assert struct.unpack("!I", magic_bytes)[0] == FRAME_MAGIC

        # 第 5 字节: version
        assert data[4] == FRAME_VERSION

        # 第 6 字节: type
        assert data[5] == FrameType.HEARTBEAT.value

        # 第 7-10 字节: length, big-endian
        length_bytes = data[6:10]
        assert struct.unpack("!I", length_bytes)[0] == 3

    def test_each_frame_type_produces_correct_byte(self):
        """验证每个 FrameType 在帧头中产生正确的字节值。"""
        for ftype in FrameType:
            frame = Frame(frame_type=ftype, payload=b"")
            data = encode_frame(frame)
            assert data[5] == ftype.value, f"{ftype} 应产生字节 0x{ftype.value:02X}"

    def test_payload_length_field_matches_actual_size(self):
        """验证帧头中 length 字段与实际 payload 大小一致。"""
        for size in (0, 1, 255, 256, 1000, 65535, 65536):
            payload = b"\x00" * size
            frame = Frame(frame_type=FrameType.MODEL_CHUNK, payload=payload)
            data = encode_frame(frame)
            _, _, _, length = struct.unpack("!IBBI", data[:FRAME_HEADER_SIZE])
            assert length == size
            assert len(data) == FRAME_HEADER_SIZE + size

    def test_magic_bytes_are_asc_null(self):
        """验证 magic 字节为 'ASC\\0'。"""
        frame = Frame(frame_type=FrameType.ENVELOPE, payload=b"")
        data = encode_frame(frame)
        assert data[:4] == b"ASC\x00"


# ---------------------------------------------------------------------------
# 2. 解码错误场景
# ---------------------------------------------------------------------------

class TestDecodeErrors:
    """解码错误场景。"""

    def test_data_too_short_1_byte(self):
        """1 字节数据应抛出 ValueError。"""
        with pytest.raises(ValueError, match="数据过短"):
            decode_frame(b"\x00")

    def test_data_too_short_9_bytes(self):
        """9 字节数据（差 1 字节到帧头大小）应抛出 ValueError。"""
        with pytest.raises(ValueError, match="数据过短"):
            decode_frame(b"\x00" * 9)

    def test_valid_magic_wrong_version_0(self):
        """有效 magic 但版本为 0 应抛出 ValueError。"""
        data = struct.pack("!IBBI", FRAME_MAGIC, 0, 0x01, 0)
        with pytest.raises(ValueError, match="不支持的版本"):
            decode_frame(data)

    def test_valid_magic_wrong_version_255(self):
        """有效 magic 但版本为 255 应抛出 ValueError。"""
        data = struct.pack("!IBBI", FRAME_MAGIC, 255, 0x01, 0)
        with pytest.raises(ValueError, match="不支持的版本"):
            decode_frame(data)

    def test_payload_truncated(self):
        """帧头声明 100 字节但实际只有 50 字节 payload 应抛出 ValueError。"""
        header = struct.pack("!IBBI", FRAME_MAGIC, FRAME_VERSION, 0x01, 100)
        data = header + b"\x00" * 50
        with pytest.raises(ValueError, match="数据长度不匹配"):
            decode_frame(data)

    def test_payload_longer_than_declared(self):
        """帧头声明 10 字节但实际有 20 字节 payload 应抛出 ValueError。"""
        header = struct.pack("!IBBI", FRAME_MAGIC, FRAME_VERSION, 0x01, 10)
        data = header + b"\x00" * 20
        with pytest.raises(ValueError, match="数据长度不匹配"):
            decode_frame(data)

    def test_invalid_frame_type_0xff(self):
        """无效帧类型 0xFF 应抛出 ValueError。"""
        header = struct.pack("!IBBI", FRAME_MAGIC, FRAME_VERSION, 0xFF, 0)
        with pytest.raises(ValueError):
            decode_frame(header)

    def test_invalid_frame_type_0x05(self):
        """控制范围(0x01-0x0F)内未定义的帧类型 0x05 应抛出 ValueError。"""
        header = struct.pack("!IBBI", FRAME_MAGIC, FRAME_VERSION, 0x05, 0)
        with pytest.raises(ValueError):
            decode_frame(header)

    def test_invalid_magic(self):
        """无效 magic 应抛出 ValueError。"""
        data = struct.pack("!IBBI", 0xDEADBEEF, FRAME_VERSION, 0x01, 0)
        with pytest.raises(ValueError, match="无效 magic"):
            decode_frame(data)


# ---------------------------------------------------------------------------
# 3. 二进制 payload 边界情况
# ---------------------------------------------------------------------------

class TestBinaryPayloadEdgeCases:
    """二进制 payload 边界情况。"""

    def test_all_zero_bytes_payload(self):
        """全零 payload 编解码。"""
        payload = b"\x00" * 256
        frame = Frame(frame_type=FrameType.MODEL_CHUNK, payload=payload)
        decoded = decode_frame(encode_frame(frame))
        assert decoded.payload == payload

    def test_all_0xff_bytes_payload(self):
        """全 0xFF payload 编解码。"""
        payload = b"\xFF" * 256
        frame = Frame(frame_type=FrameType.MODEL_CHUNK, payload=payload)
        decoded = decode_frame(encode_frame(frame))
        assert decoded.payload == payload

    def test_payload_with_every_byte_value(self):
        """payload 包含每个字节值 0x00-0xFF。"""
        payload = bytes(range(256))
        frame = Frame(frame_type=FrameType.MODEL_CHUNK, payload=payload)
        decoded = decode_frame(encode_frame(frame))
        assert decoded.payload == payload

    def test_empty_payload(self):
        """空 payload (0 字节) 编解码。"""
        frame = Frame(frame_type=FrameType.HEARTBEAT, payload=b"")
        data = encode_frame(frame)
        assert len(data) == FRAME_HEADER_SIZE
        decoded = decode_frame(data)
        assert decoded.payload == b""

    def test_payload_at_max_frame_size_boundary(self):
        """payload 恰好为 MAX_FRAME_SIZE (10MB) 编解码。"""
        payload = b"\xAB" * MAX_FRAME_SIZE
        frame = Frame(frame_type=FrameType.MODEL_CHUNK, payload=payload)
        data = encode_frame(frame)
        assert len(data) == FRAME_HEADER_SIZE + MAX_FRAME_SIZE
        decoded = decode_frame(data)
        assert len(decoded.payload) == MAX_FRAME_SIZE
        assert decoded.payload == payload


# ---------------------------------------------------------------------------
# 4. 流式读取边界情况
# ---------------------------------------------------------------------------

class TestReadStreamEdgeCases:
    """流式读取边界情况。"""

    @pytest.mark.asyncio
    async def test_frame_split_across_feed_data(self):
        """帧被拆分到多次 feed_data 调用（模拟 TCP 分片）。"""
        frame = Frame(frame_type=FrameType.TASK_DISPATCH, payload=b"hello world")
        data = write_frame_to_bytes(frame)

        reader = asyncio.StreamReader()
        # 将数据拆成 3 段喂入
        chunk1 = data[:4]
        chunk2 = data[4:8]
        chunk3 = data[8:]
        reader.feed_data(chunk1)
        reader.feed_data(chunk2)
        reader.feed_data(chunk3)
        reader.feed_eof()

        result = await read_frame_from_stream(reader)
        assert result is not None
        assert result.frame_type == FrameType.TASK_DISPATCH
        assert result.payload == b"hello world"

    @pytest.mark.asyncio
    async def test_multiple_frames_concatenated_in_single_feed(self):
        """多个帧在单次 feed_data 中拼接。"""
        frame1 = Frame(frame_type=FrameType.HEARTBEAT, payload=b"")
        frame2 = Frame(frame_type=FrameType.TASK_RESULT, payload=b"result")
        frame3 = Frame(frame_type=FrameType.ACK, payload=b"ok")

        combined = (
            write_frame_to_bytes(frame1)
            + write_frame_to_bytes(frame2)
            + write_frame_to_bytes(frame3)
        )

        reader = asyncio.StreamReader()
        reader.feed_data(combined)
        reader.feed_eof()

        r1 = await read_frame_from_stream(reader)
        r2 = await read_frame_from_stream(reader)
        r3 = await read_frame_from_stream(reader)

        assert r1 is not None and r1.frame_type == FrameType.HEARTBEAT
        assert r2 is not None and r2.frame_type == FrameType.TASK_RESULT
        assert r3 is not None and r3.frame_type == FrameType.ACK
        assert r2.payload == b"result"
        assert r3.payload == b"ok"

    @pytest.mark.asyncio
    async def test_stream_invalid_version_returns_none(self):
        """流中版本号无效返回 None。"""
        data = struct.pack("!IBBI", FRAME_MAGIC, 99, 0x01, 0)
        reader = asyncio.StreamReader()
        reader.feed_data(data)
        reader.feed_eof()

        result = await read_frame_from_stream(reader)
        assert result is None

    @pytest.mark.asyncio
    async def test_stream_oversized_frame_returns_none(self):
        """流中超大帧 (> MAX_FRAME_SIZE) 返回 None。"""
        data = struct.pack("!IBBI", FRAME_MAGIC, FRAME_VERSION, 0x01, MAX_FRAME_SIZE + 1)
        reader = asyncio.StreamReader()
        reader.feed_data(data)
        reader.feed_eof()

        result = await read_frame_from_stream(reader)
        assert result is None

    @pytest.mark.asyncio
    async def test_read_after_eof_returns_none(self):
        """EOF 后再读取返回 None。"""
        reader = asyncio.StreamReader()
        reader.feed_eof()

        result = await read_frame_from_stream(reader)
        assert result is None

    @pytest.mark.asyncio
    async def test_stream_payload_truncated_returns_none(self):
        """流中 payload 被截断（不完整）返回 None。"""
        header = struct.pack("!IBBI", FRAME_MAGIC, FRAME_VERSION, 0x01, 100)
        # 只提供部分 payload
        reader = asyncio.StreamReader()
        reader.feed_data(header + b"\x00" * 50)
        reader.feed_eof()

        result = await read_frame_from_stream(reader)
        assert result is None


# ---------------------------------------------------------------------------
# 5. Envelope 桥接全面性
# ---------------------------------------------------------------------------

class TestEnvelopeBridgeComprehensive:
    """Envelope 桥接全面测试。"""

    def test_roundtrip_every_channel(self):
        """每个 Channel 类型都能正确往返。"""
        for channel in Channel:
            msg = Message(type=MessageType.HEARTBEAT, sender_id="n1", payload={"ch": channel.value})
            env = Envelope(channel=channel, message=msg)
            frame = encode_envelope_frame(env)
            restored = decode_envelope_frame(frame)
            assert restored.channel == channel

    def test_roundtrip_every_message_type(self):
        """每个 MessageType 都能正确往返。"""
        for mtype in MessageType:
            msg = Message(type=mtype, sender_id="n1", payload={"t": mtype.value})
            env = Envelope(channel=Channel.EVENTS, message=msg)
            frame = encode_envelope_frame(env)
            restored = decode_envelope_frame(frame)
            assert restored.message.type == mtype

    def test_complex_nested_payload(self):
        """复杂嵌套 payload 往返。"""
        payload = {
            "layers": [
                {"name": "layer0", "weights": [0.1, 0.2, 0.3]},
                {"name": "layer1", "weights": [0.4, 0.5, 0.6]},
            ],
            "metadata": {
                "version": 2,
                "tags": ["v2", "prod"],
                "config": {"lr": 0.001, "epochs": 10},
            },
        }
        msg = Message(type=MessageType.TASK_DISPATCH, sender_id="master", payload=payload)
        env = Envelope(channel=Channel.TASK_DISPATCH, message=msg)
        frame = encode_envelope_frame(env)
        restored = decode_envelope_frame(frame)
        assert restored.message.payload == payload

    def test_unicode_payload_through_frame_bridge(self):
        """Unicode payload 通过帧桥接往返。"""
        payload = {
            "name": "模型-中文测试",
            "description": "こんにちは世界 🌍",
            "emoji": "🚀🎉",
        }
        msg = Message(type=MessageType.CREATE_INSTANCE, sender_id="n1", payload=payload)
        env = Envelope(channel=Channel.COMMANDS, message=msg)
        frame = encode_envelope_frame(env)
        restored = decode_envelope_frame(frame)
        assert restored.message.payload == payload

    def test_target_none_vs_set(self):
        """target=None 与 target="node-1" 的往返。"""
        msg = Message(type=MessageType.HEARTBEAT, sender_id="n1", payload={})

        # target=None
        env_no_target = Envelope(channel=Channel.HEARTBEATS, message=msg)
        frame_no = encode_envelope_frame(env_no_target)
        restored_no = decode_envelope_frame(frame_no)
        assert restored_no.target is None

        # target="node-1"
        env_with_target = Envelope(channel=Channel.HEARTBEATS, message=msg, target="node-1")
        frame_with = encode_envelope_frame(env_with_target)
        restored_with = decode_envelope_frame(frame_with)
        assert restored_with.target == "node-1"

    def test_envelope_via_wire_roundtrip(self):
        """Envelope 经完整帧编码→字节流→解码→Envelope 往返。"""
        msg = Message(
            type=MessageType.TASK_RESULT,
            sender_id="runner-1",
            payload={"status": "done", "score": 0.99},
        )
        env = Envelope(channel=Channel.TASK_DISPATCH, message=msg, target="master")

        wire = write_frame_to_bytes(encode_envelope_frame(env))
        decoded_frame = decode_frame(wire)
        restored = decode_envelope_frame(decoded_frame)

        assert restored.channel == Channel.TASK_DISPATCH
        assert restored.message.type == MessageType.TASK_RESULT
        assert restored.message.sender_id == "runner-1"
        assert restored.message.payload["score"] == 0.99
        assert restored.target == "master"


# ---------------------------------------------------------------------------
# 6. write_frame_to_bytes 一致性
# ---------------------------------------------------------------------------

class TestWriteFrameToBytesConsistency:
    """write_frame_to_bytes 与 encode_frame 一致性。"""

    def test_empty_payload(self):
        """空 payload 输出一致。"""
        frame = Frame(frame_type=FrameType.HEARTBEAT, payload=b"")
        assert write_frame_to_bytes(frame) == encode_frame(frame)

    def test_various_frame_types(self):
        """各种帧类型输出一致。"""
        for ftype in FrameType:
            frame = Frame(frame_type=ftype, payload=b"\x01\x02\x03")
            assert write_frame_to_bytes(frame) == encode_frame(frame)

    def test_large_payload(self):
        """大 payload 输出一致。"""
        payload = b"\xAB" * (1024 * 1024)
        frame = Frame(frame_type=FrameType.MODEL_CHUNK, payload=payload)
        assert write_frame_to_bytes(frame) == encode_frame(frame)

    def test_binary_payload(self):
        """二进制 payload 输出一致。"""
        payload = bytes(range(256))
        frame = Frame(frame_type=FrameType.MODEL_CHUNK, payload=payload)
        assert write_frame_to_bytes(frame) == encode_frame(frame)

    def test_envelope_frame(self):
        """Envelope 帧输出一致。"""
        msg = Message(type=MessageType.HEARTBEAT, sender_id="n1", payload={})
        env = Envelope(channel=Channel.HEARTBEATS, message=msg)
        frame = encode_envelope_frame(env)
        assert write_frame_to_bytes(frame) == encode_frame(frame)
