"""测试 ASC Cluster Link Protocol - 统一二进制帧格式。"""

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


class TestFrameType:
    """帧类型枚举。"""

    def test_envelope_frame_type(self):
        """Envelope 帧类型。"""
        assert FrameType.ENVELOPE.value == 0x00

    def test_control_frame_types(self):
        """控制帧类型。"""
        assert FrameType.HEARTBEAT.value == 0x01
        assert FrameType.ACK.value == 0x02
        assert FrameType.NACK.value == 0x03

    def test_task_frame_types(self):
        """任务帧类型。"""
        assert FrameType.TASK_DISPATCH.value == 0x10
        assert FrameType.TASK_ACCEPT.value == 0x11
        assert FrameType.TASK_PROGRESS.value == 0x12
        assert FrameType.TASK_RESULT.value == 0x13
        assert FrameType.TASK_CANCEL.value == 0x14

    def test_capacity_frame_types(self):
        """容量帧类型。"""
        assert FrameType.CAPACITY_REPORT.value == 0x20
        assert FrameType.CAPACITY_QUERY.value == 0x21
        assert FrameType.CAPACITY_RESPONSE.value == 0x22

    def test_rebalance_frame_types(self):
        """重平衡帧类型。"""
        assert FrameType.REBALANCE_REQUEST.value == 0x30
        assert FrameType.REBALANCE_ACK.value == 0x31
        assert FrameType.REBALANCE_COMPLETE.value == 0x32

    def test_data_frame_types(self):
        """数据帧类型。"""
        assert FrameType.MODEL_CHUNK.value == 0x40
        assert FrameType.MODEL_CHUNK_ACK.value == 0x41


class TestFrameConstants:
    """帧常量。"""

    def test_magic(self):
        assert FRAME_MAGIC == 0x41534300  # "ASC\0"

    def test_version(self):
        assert FRAME_VERSION == 1

    def test_header_size(self):
        assert FRAME_HEADER_SIZE == 10

    def test_max_frame_size(self):
        assert MAX_FRAME_SIZE == 10 * 1024 * 1024


class TestFrame:
    """帧数据类。"""

    def test_create(self):
        frame = Frame(frame_type=FrameType.HEARTBEAT, payload=b"")
        assert frame.frame_type == FrameType.HEARTBEAT
        assert frame.payload == b""

    def test_create_with_payload(self):
        frame = Frame(frame_type=FrameType.TASK_DISPATCH, payload=b"hello")
        assert frame.payload == b"hello"

    def test_frozen(self):
        frame = Frame(frame_type=FrameType.HEARTBEAT, payload=b"")
        try:
            frame.frame_type = FrameType.ACK  # type: ignore[misc]
            raise AssertionError("Should be immutable")
        except (AttributeError, TypeError):
            pass


class TestEncodeDecodeFrame:
    """帧编解码。"""

    def test_encode_decode_heartbeat(self):
        frame = Frame(frame_type=FrameType.HEARTBEAT, payload=b"")
        data = encode_frame(frame)
        decoded = decode_frame(data)
        assert decoded.frame_type == FrameType.HEARTBEAT
        assert decoded.payload == b""

    def test_encode_decode_with_payload(self):
        payload = b"test data 12345"
        frame = Frame(frame_type=FrameType.TASK_DISPATCH, payload=payload)
        data = encode_frame(frame)
        decoded = decode_frame(data)
        assert decoded.frame_type == FrameType.TASK_DISPATCH
        assert decoded.payload == payload

    def test_encode_header_format(self):
        frame = Frame(frame_type=FrameType.HEARTBEAT, payload=b"")
        data = encode_frame(frame)
        magic, version, ftype, length = struct.unpack("!IBBI", data[:FRAME_HEADER_SIZE])
        assert magic == FRAME_MAGIC
        assert version == FRAME_VERSION
        assert ftype == FrameType.HEARTBEAT.value
        assert length == 0

    def test_encode_with_payload_header(self):
        payload = b"abc"
        frame = Frame(frame_type=FrameType.TASK_RESULT, payload=payload)
        data = encode_frame(frame)
        magic, version, ftype, length = struct.unpack("!IBBI", data[:FRAME_HEADER_SIZE])
        assert length == 3
        assert data[FRAME_HEADER_SIZE:] == payload

    def test_decode_invalid_magic(self):
        data = struct.pack("!IBBI", 0xDEADBEEF, 1, 0x01, 0)
        try:
            decode_frame(data)
            raise AssertionError("Should raise ValueError")
        except ValueError:
            pass

    def test_decode_truncated_data(self):
        data = struct.pack("!IBBI", FRAME_MAGIC, 1, 0x01, 100)
        try:
            decode_frame(data)
            raise AssertionError("Should raise ValueError")
        except ValueError:
            pass

    def test_decode_unsupported_version(self):
        data = struct.pack("!IBBI", FRAME_MAGIC, 99, 0x01, 0)
        try:
            decode_frame(data)
            raise AssertionError("Should raise ValueError")
        except ValueError:
            pass

    def test_roundtrip_all_frame_types(self):
        for ftype in FrameType:
            frame = Frame(frame_type=ftype, payload=b"\x01\x02\x03")
            data = encode_frame(frame)
            decoded = decode_frame(data)
            assert decoded.frame_type == ftype
            assert decoded.payload == b"\x01\x02\x03"

    def test_large_payload(self):
        payload = b"\x00" * (1024 * 1024)
        frame = Frame(frame_type=FrameType.MODEL_CHUNK, payload=payload)
        data = encode_frame(frame)
        decoded = decode_frame(data)
        assert decoded.payload == payload
        assert len(data) == FRAME_HEADER_SIZE + len(payload)


class TestEnvelopeBridge:
    """Envelope 桥接函数测试。"""

    def test_encode_envelope_frame(self):
        msg = Message(type=MessageType.HEARTBEAT, sender_id="n1", payload={"status": "alive"})
        env = Envelope(channel=Channel.HEARTBEATS, message=msg)
        frame = encode_envelope_frame(env)

        assert frame.frame_type == FrameType.ENVELOPE
        assert len(frame.payload) > 0

    def test_decode_envelope_frame(self):
        msg = Message(type=MessageType.NODE_JOINED, sender_id="n2", payload={"ip": "10.0.0.1"})
        env = Envelope(channel=Channel.EVENTS, message=msg, target="n3")
        frame = encode_envelope_frame(env)
        restored = decode_envelope_frame(frame)

        assert restored.channel == Channel.EVENTS
        assert restored.message.type == MessageType.NODE_JOINED
        assert restored.message.sender_id == "n2"
        assert restored.message.payload == {"ip": "10.0.0.1"}
        assert restored.target == "n3"

    def test_decode_envelope_frame_wrong_type(self):
        """非 ENVELOPE 帧类型不能解码为 Envelope。"""
        frame = Frame(frame_type=FrameType.HEARTBEAT, payload=b"not_json")
        try:
            decode_envelope_frame(frame)
            raise AssertionError("Should raise ValueError")
        except ValueError as e:
            assert "ENVELOPE" in str(e)

    def test_envelope_frame_roundtrip_via_wire(self):
        """Envelope 通过 Frame 编码后在线上传输，再解码回来。"""
        msg = Message(
            type=MessageType.CREATE_INSTANCE,
            sender_id="master",
            payload={"model": "qwen"},
        )
        env = Envelope(channel=Channel.COMMANDS, message=msg)

        # 编码为 Frame -> 编码为字节流
        frame = encode_envelope_frame(env)
        wire_data = write_frame_to_bytes(frame)

        # 从字节流解码 Frame -> 解码 Envelope
        decoded_frame = decode_frame(wire_data)
        restored_env = decode_envelope_frame(decoded_frame)

        assert restored_env.channel == Channel.COMMANDS
        assert restored_env.message.type == MessageType.CREATE_INSTANCE
        assert restored_env.message.payload["model"] == "qwen"

    def test_envelope_frame_preserves_timestamp(self):
        """Envelope 帧保留时间戳。"""
        msg = Message(type=MessageType.HEARTBEAT, sender_id="n1", payload={})
        original_ts = msg.timestamp
        env = Envelope(channel=Channel.HEARTBEATS, message=msg)

        frame = encode_envelope_frame(env)
        restored = decode_envelope_frame(frame)

        assert abs(restored.message.timestamp - original_ts) < 0.001


class TestReadStream:
    """流式读取测试。"""

    @pytest.mark.asyncio
    async def test_read_frame_from_stream(self):
        """从 asyncio 流中读取帧。"""
        frame = Frame(frame_type=FrameType.TASK_DISPATCH, payload=b"hello")
        data = write_frame_to_bytes(frame)

        reader = asyncio.StreamReader()
        reader.feed_data(data)
        reader.feed_eof()

        result = await read_frame_from_stream(reader)
        assert result is not None
        assert result.frame_type == FrameType.TASK_DISPATCH
        assert result.payload == b"hello"

    @pytest.mark.asyncio
    async def test_read_frame_from_stream_empty(self):
        """空流返回 None。"""
        reader = asyncio.StreamReader()
        reader.feed_eof()

        result = await read_frame_from_stream(reader)
        assert result is None

    @pytest.mark.asyncio
    async def test_read_multiple_frames(self):
        """连续读取多个帧。"""
        frame1 = Frame(frame_type=FrameType.HEARTBEAT, payload=b"")
        frame2 = Frame(frame_type=FrameType.TASK_RESULT, payload=b"result")

        reader = asyncio.StreamReader()
        reader.feed_data(write_frame_to_bytes(frame1))
        reader.feed_data(write_frame_to_bytes(frame2))
        reader.feed_eof()

        result1 = await read_frame_from_stream(reader)
        result2 = await read_frame_from_stream(reader)

        assert result1 is not None
        assert result1.frame_type == FrameType.HEARTBEAT
        assert result2 is not None
        assert result2.frame_type == FrameType.TASK_RESULT

    @pytest.mark.asyncio
    async def test_read_frame_invalid_magic(self):
        """无效 magic 返回 None。"""
        data = struct.pack("!IBBI", 0xDEADBEEF, 1, 0x01, 0)
        reader = asyncio.StreamReader()
        reader.feed_data(data)
        reader.feed_eof()

        result = await read_frame_from_stream(reader)
        assert result is None

    @pytest.mark.asyncio
    async def test_read_frame_oversized(self):
        """超大帧返回 None。"""
        data = struct.pack("!IBBI", FRAME_MAGIC, 1, 0x01, MAX_FRAME_SIZE + 1)
        reader = asyncio.StreamReader()
        reader.feed_data(data)
        reader.feed_eof()

        result = await read_frame_from_stream(reader)
        assert result is None


class TestWriteFrameToBytes:
    """write_frame_to_bytes 测试。"""

    def test_equals_encode_frame(self):
        """write_frame_to_bytes 等价于 encode_frame。"""
        frame = Frame(frame_type=FrameType.HEARTBEAT, payload=b"test")
        assert write_frame_to_bytes(frame) == encode_frame(frame)
