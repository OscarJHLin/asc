"""Asc 消息协议定义。

替代原有 protocol.py 中未使用的定义，提供：
- Channel：类型化消息通道
- MessageType：消息类型枚举
- Message：不可变消息值对象
- Envelope：消息信封（通道 + 消息 + 可选目标）
- JSON 序列化/反序列化

设计原则：
- JSON 序列化，跨语言兼容
- 消息带时间戳和发送者信息
- 通道隔离不同类型的消息
"""

from __future__ import annotations

import enum
import json
import time
from dataclasses import dataclass


class Channel(enum.Enum):
    """消息通道。"""

    EVENTS = "events"
    COMMANDS = "commands"
    HEARTBEATS = "heartbeats"
    ELECTION = "election"
    DISCOVERY = "discovery"


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


def encode_envelope(envelope: Envelope) -> bytes:
    """将信封序列化为 JSON 字节。"""
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


def decode_envelope(data: bytes) -> Envelope:
    """从 JSON 字节反序列化信封。"""
    parsed = json.loads(data.decode("utf-8"))
    channel = Channel(parsed["channel"])
    msg_data = parsed["message"]
    message = Message(
        type=MessageType(msg_data["type"]),
        sender_id=msg_data["sender_id"],
        payload=msg_data["payload"],
        timestamp=msg_data["timestamp"],
    )
    target = parsed.get("target")
    return Envelope(channel=channel, message=message, target=target)
