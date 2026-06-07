"""ASC Cluster Link Protocol - 统一二进制帧格式。

所有节点间通信的唯一线协议，解决 TCP 流式传输中的粘包/拆包问题：
- 帧头: magic(4B) + version(1B) + type(1B) + length(4B) = 10 字节
- 帧体: payload (变长，最大 10MB)

设计背景：
    早期版本维护两套协议（JSON Envelope + Binary Frame），导致传输层代码
    重复和路由逻辑分裂。统一为 Binary Frame 后，所有消息共享同一套编解码、
    同一套流式读取逻辑，大幅降低维护复杂度。

帧类型按功能分区:
- 0x00:       Envelope 帧 (控制面 JSON 消息，兼容旧协议)
- 0x01-0x0F:  控制帧 (心跳/ACK/NACK)
- 0x10-0x1F:  任务帧 (分派/接受/进度/结果/取消)
- 0x20-0x2F:  容量帧 (上报/查询/响应)
- 0x30-0x3F:  重平衡帧 (请求/确认/完成)
- 0x40-0x4F:  数据帧 (模型分片)

统一方案：
- Binary Frame 是唯一线协议，所有节点间消息必须经此格式传输
- 控制面消息 (Envelope) 作为 ENVELOPE 帧类型的 JSON payload
- 数据面消息使用各自专属帧类型，payload 可为 JSON 或原始二进制
- 传输层统一使用 read_frame_from_stream() 从 asyncio StreamReader 逐帧读取

线程安全：
    Frame 为不可变 frozen dataclass，encode/decode 为纯函数，无状态共享，
    可在多协程环境下安全使用。
"""

from __future__ import annotations

import asyncio
import enum
import struct
from dataclasses import dataclass

from asc.network.protocol import Envelope, decode_envelope, encode_envelope

# 帧常量
FRAME_MAGIC: int = 0x41534300  # "ASC\0"
FRAME_VERSION: int = 1
FRAME_HEADER_SIZE: int = 10  # 4 + 1 + 1 + 4
MAX_FRAME_SIZE: int = 10 * 1024 * 1024  # 10MB 上限


class FrameType(enum.Enum):
    """帧类型。"""

    # 统一 Envelope 帧 (0x00) - 控制面 JSON 消息
    ENVELOPE = 0x00

    # 控制 (0x01-0x0F)
    HEARTBEAT = 0x01
    ACK = 0x02
    NACK = 0x03

    # 任务 (0x10-0x1F)
    TASK_DISPATCH = 0x10
    TASK_ACCEPT = 0x11
    TASK_PROGRESS = 0x12
    TASK_RESULT = 0x13
    TASK_CANCEL = 0x14

    # 容量 (0x20-0x2F)
    CAPACITY_REPORT = 0x20
    CAPACITY_QUERY = 0x21
    CAPACITY_RESPONSE = 0x22

    # 重平衡 (0x30-0x3F)
    REBALANCE_REQUEST = 0x30
    REBALANCE_ACK = 0x31
    REBALANCE_COMPLETE = 0x32

    # 数据 (0x40-0x4F)
    MODEL_CHUNK = 0x40
    MODEL_CHUNK_ACK = 0x41


@dataclass(frozen=True)
class Frame:
    """二进制帧。"""

    frame_type: FrameType
    payload: bytes


def encode_frame(frame: Frame) -> bytes:
    """将帧编码为二进制数据。

    格式: [magic:4B][version:1B][type:1B][length:4B][payload:变长]
    使用大端序（网络字节序）打包，确保跨平台一致性。

    Args:
        frame: 待编码的 Frame 对象

    Returns:
        完整的二进制帧数据（含 10 字节头部）

    性能注意：
        本函数为纯计算，无 IO 操作，可直接在主协程调用。
    """
    header = struct.pack(
        "!IBBI",
        FRAME_MAGIC,
        FRAME_VERSION,
        frame.frame_type.value,
        len(frame.payload),
    )
    return header + frame.payload


def decode_frame(data: bytes) -> Frame:
    """从二进制数据解码帧。

    严格的完整性校验：magic、version、payload 长度必须全部匹配。
    任何不匹配都会抛出 ValueError，防止解析错误数据导致状态混乱。

    Args:
        data: 接收到的原始字节数据

    Returns:
        解码后的 Frame 对象

    Raises:
        ValueError: magic 不匹配、版本不支持、数据截断或长度不匹配

    安全注意：
        若 length 字段声称的 payload 长度大于实际数据，说明数据尚未收齐，
        调用者应继续从 TCP 流读取，而非直接丢弃。
    """
    if len(data) < FRAME_HEADER_SIZE:
        raise ValueError(f"数据过短: {len(data)} < {FRAME_HEADER_SIZE}")

    magic, version, ftype, length = struct.unpack("!IBBI", data[:FRAME_HEADER_SIZE])

    if magic != FRAME_MAGIC:
        raise ValueError(f"无效 magic: 0x{magic:08X}, 期望 0x{FRAME_MAGIC:08X}")

    if version != FRAME_VERSION:
        raise ValueError(f"不支持的版本: {version}, 期望 {FRAME_VERSION}")

    payload = data[FRAME_HEADER_SIZE:]
    if len(payload) != length:
        raise ValueError(
            f"数据长度不匹配: 期望 {length}, 实际 {len(payload)}"
        )

    return Frame(
        frame_type=FrameType(ftype),
        payload=payload,
    )


# --- Envelope 桥接函数 ---
# 将 Envelope (控制面 JSON) 编码为 ENVELOPE 帧，实现统一线协议

def encode_envelope_frame(envelope: Envelope) -> Frame:
    """将 Envelope 编码为 ENVELOPE 帧。

    payload 为 Envelope 的 JSON 字节。
    """
    json_bytes = encode_envelope(envelope)
    return Frame(frame_type=FrameType.ENVELOPE, payload=json_bytes)


def decode_envelope_frame(frame: Frame) -> Envelope:
    """从 ENVELOPE 帧解码回 Envelope。

    Raises:
        ValueError: 帧类型不是 ENVELOPE
    """
    if frame.frame_type != FrameType.ENVELOPE:
        raise ValueError(
            f"帧类型不是 ENVELOPE: {frame.frame_type}"
        )
    return decode_envelope(frame.payload)


# --- 流式读取辅助 ---
# 用于 transport.py 从 asyncio StreamReader 逐帧读取

async def read_frame_from_stream(reader: asyncio.StreamReader) -> Frame | None:
    """从 asyncio 流中读取一个完整帧。

    这是整个传输层唯一的消息读取入口。所有 TCP 连接（Server 和 Client）
    必须使用此函数读取消息，确保统一的粘包处理、magic 校验和长度限制。

    Args:
        reader: asyncio StreamReader，通常来自 open_connection 或 start_server

    Returns:
        成功解码的 Frame；若连接关闭或数据异常则返回 None

    异常处理策略：
        - IncompleteReadError：对端关闭连接，返回 None，上层应清理连接
        - magic/version 不匹配：返回 None，防止非法数据破坏状态机
        - length > MAX_FRAME_SIZE：返回 None，防止内存耗尽攻击（DoS）

    性能注意：
        本函数包含两次 await readexactly()，会挂起协程直到数据到达，
        不会阻塞事件循环中的其他任务。
    """
    try:
        header = await reader.readexactly(FRAME_HEADER_SIZE)
    except asyncio.IncompleteReadError:
        return None

    _magic, _version, ftype, length = struct.unpack("!IBBI", header)

    if _magic != FRAME_MAGIC:
        return None
    if _version != FRAME_VERSION:
        return None
    if length > MAX_FRAME_SIZE:
        return None

    try:
        payload = await reader.readexactly(length)
    except asyncio.IncompleteReadError:
        return None

    return Frame(
        frame_type=FrameType(ftype),
        payload=payload,
    )


def write_frame_to_bytes(frame: Frame) -> bytes:
    """将帧编码为可直接写入流的字节。"""
    return encode_frame(frame)
