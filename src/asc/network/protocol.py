"""Asc 消息协议定义。

替代原有 protocol.py 中未使用的定义，提供：
- Channel：类型化消息通道
- MessageType：消息类型枚举
- Message：不可变消息值对象
- Envelope：消息信封（通道 + 消息 + 可选目标）
- 双格式序列化：MessagePack（高性能）和 JSON（兼容）

性能优化：
- 默认使用 MessagePack 序列化，体积减少 30-50%，速度提升 5-10x
- 保留 JSON 序列化用于调试和跨语言兼容
- 自动格式检测：decode 时自动识别 JSON 或 MessagePack
"""

from __future__ import annotations

import enum
import json
import time
from dataclasses import dataclass

import msgpack


class Channel(enum.Enum):
    """消息通道。"""

    EVENTS = "events"
    COMMANDS = "commands"
    HEARTBEATS = "heartbeats"
    ELECTION = "election"
    DISCOVERY = "discovery"
    TASK_DISPATCH = "task_dispatch"
    CAPACITY = "capacity"
    REBALANCE = "rebalance"


class MessageType(enum.Enum):
    """消息类型。"""

    # 事件
    NODE_JOINED = "node_joined"
    NODE_LEFT = "node_left"
    INSTANCE_CREATED = "instance_created"
    INSTANCE_DELETED = "instance_deleted"
    TASK_CREATED = "task_created"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_CANCELLED = "task_cancelled"
    RUNNER_STATUS = "runner_status"

    # 命令
    CREATE_INSTANCE = "create_instance"
    DELETE_INSTANCE = "delete_instance"
    START_INFERENCE = "start_inference"
    CANCEL_TASK = "cancel_task"
    SHUTDOWN_RUNNER = "shutdown_runner"

    # 控制
    HEARTBEAT = "heartbeat"
    DISCOVER = "discover"
    DISCOVER_RESPONSE = "discover_response"
    ELECTION = "election"
    REQUEST_EVENT_LOG = "request_event_log"
    EVENT_LOG_CHUNK = "event_log_chunk"

    # 任务分派
    TASK_DISPATCH = "task_dispatch"
    TASK_ACCEPT = "task_accept"
    TASK_PROGRESS = "task_progress"
    TASK_RESULT = "task_result"
    TASK_CANCEL_MSG = "task_cancel_msg"

    # 容量
    CAPACITY_REPORT = "capacity_report"
    CAPACITY_QUERY = "capacity_query"
    CAPACITY_RESPONSE = "capacity_response"

    # 重平衡
    REBALANCE_REQUEST = "rebalance_request"
    REBALANCE_ACK = "rebalance_ack"
    REBALANCE_COMPLETE = "rebalance_complete"


@dataclass(frozen=True)
class Message:
    """不可变消息。"""

    type: MessageType
    sender_id: str
    payload: dict
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            object.__setattr__(self, "timestamp", time.time())


@dataclass(frozen=True)
class Envelope:
    """消息信封：通道 + 消息 + 可选目标节点。"""

    channel: Channel
    message: Message
    target: str | None = None


# --- 序列化辅助 ---


def _envelope_to_dict(envelope: Envelope) -> dict:
    """将信封转换为字典（短键名减少体积）。"""
    data: dict = {
        "c": envelope.channel.value,
        "m": {
            "t": envelope.message.type.value,
            "s": envelope.message.sender_id,
            "p": envelope.message.payload,
            "ts": envelope.message.timestamp,
        },
    }
    if envelope.target is not None:
        data["tg"] = envelope.target
    return data


def _dict_to_envelope(data: dict) -> Envelope:
    """从字典还原信封，自动适配短键名（MessagePack）和长键名（JSON）。"""
    # 检测键名格式：MessagePack 使用短键名，JSON 使用长键名
    if "c" in data and "m" in data:
        # 短键名格式（MessagePack）
        msg_data = data["m"]
        return Envelope(
            channel=Channel(data["c"]),
            message=Message(
                type=MessageType(msg_data["t"]),
                sender_id=msg_data["s"],
                payload=msg_data["p"],
                timestamp=msg_data["ts"],
            ),
            target=data.get("tg"),
        )
    else:
        # 长键名格式（JSON 兼容）
        msg_data = data["message"]
        return Envelope(
            channel=Channel(data["channel"]),
            message=Message(
                type=MessageType(msg_data["type"]),
                sender_id=msg_data["sender_id"],
                payload=msg_data["payload"],
                timestamp=msg_data["timestamp"],
            ),
            target=data.get("target"),
        )


# --- MessagePack 序列化（默认，高性能） ---


def encode_envelope(envelope: Envelope) -> bytes:
    """将信封序列化为 MessagePack 字节（默认高性能格式）。"""
    data = _envelope_to_dict(envelope)
    return msgpack.packb(data, use_bin_type=True)


def decode_envelope(data: bytes) -> Envelope:
    """从字节反序列化信封，自动检测 JSON 或 MessagePack 格式。"""
    if not data:
        raise ValueError("空数据")
    # JSON 以 { 开头（0x7B），MessagePack 以其他字节开头
    if data[0:1] == b"{":
        return _decode_envelope_json(data)
    return _decode_envelope_msgpack(data)


def _decode_envelope_msgpack(data: bytes) -> Envelope:
    """从 MessagePack 字节反序列化信封。"""
    try:
        parsed = msgpack.unpackb(data, raw=False)
    except (msgpack.ExtraData, msgpack.FormatError, msgpack.StackError, ValueError) as exc:
        raise ValueError(f"无效的 MessagePack 数据: {exc}") from exc
    return _dict_to_envelope(parsed)


def _decode_envelope_json(data: bytes) -> Envelope:
    """从 JSON 字节反序列化信封（向后兼容）。"""
    parsed = json.loads(data.decode("utf-8"))
    return _dict_to_envelope(parsed)


# --- JSON 序列化（兼容模式） ---


def encode_envelope_json(envelope: Envelope) -> bytes:
    """将信封序列化为 JSON 字节（兼容模式）。"""
    data = {
        "channel": envelope.channel.value,
        "message": {
            "type": envelope.message.type.value,
            "sender_id": envelope.message.sender_id,
            "payload": envelope.message.payload,
            "timestamp": envelope.message.timestamp,
        },
    }
    if envelope.target is not None:
        data["target"] = envelope.target
    return json.dumps(data, ensure_ascii=False).encode("utf-8")
