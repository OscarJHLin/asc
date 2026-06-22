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
from typing import Any

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
    SYSTEM = "system"  # 系统内部消息（认证、握手机制）


class MessageType(enum.Enum):
    """消息类型。

    已使用的类型标注了使用位置；预留类型标注了预期用途。
    """

    # 事件（预留：事件溯源恢复时使用）
    NODE_JOINED = "node_joined"           # Worker -> Master: 节点加入
    NODE_LEFT = "node_left"               # Worker -> Master: 节点离开
    INSTANCE_CREATED = "instance_created"  # 预留：实例创建事件
    INSTANCE_DELETED = "instance_deleted"  # 预留：实例删除事件
    TASK_CREATED = "task_created"          # 预留：任务创建事件
    TASK_COMPLETED = "task_completed"      # 预留：任务完成事件
    TASK_FAILED = "task_failed"            # 预留：任务失败事件
    TASK_CANCELLED = "task_cancelled"      # 预留：任务取消事件
    RUNNER_STATUS = "runner_status"        # 预留：Runner 状态变更事件

    # 命令（预留：API -> Master 命令通道）
    CREATE_INSTANCE = "create_instance"    # 预留：创建实例命令
    DELETE_INSTANCE = "delete_instance"    # 预留：删除实例命令
    START_INFERENCE = "start_inference"    # 预留：启动推理命令
    CANCEL_TASK = "cancel_task"            # Master -> Worker: 取消任务
    SHUTDOWN_RUNNER = "shutdown_runner"    # 预留：关闭 Runner 命令

    # 控制
    AUTH = "auth"                                  # Client -> Server: 认证握手
    ACK = "ack"                                    # Server -> Client: 确认响应
    HEARTBEAT = "heartbeat"                        # Worker -> Master: 心跳
    DISCOVER = "discover"                          # 预留：节点发现广播
    DISCOVER_RESPONSE = "discover_response"        # 预留：节点发现响应
    ELECTION = "election"                          # 选举：发起选举
    ELECTION_ALIVE = "election_alive"              # 选举：存活宣告
    ELECTION_COORDINATOR = "election_coordinator"  # 选举：Master 宣告
    REQUEST_EVENT_LOG = "request_event_log"        # 预留：请求事件日志
    EVENT_LOG_CHUNK = "event_log_chunk"            # 预留：事件日志分片

    # 任务分派
    TASK_DISPATCH = "task_dispatch"    # Master -> Worker: 分派任务
    TASK_ACCEPT = "task_accept"        # Worker -> Master: 接受任务
    TASK_PROGRESS = "task_progress"    # 预留：任务进度上报
    TASK_RESULT = "task_result"        # Worker -> Master: 任务结果

    # 容量
    CAPACITY_REPORT = "capacity_report"    # Worker -> Master: 容量上报
    CAPACITY_QUERY = "capacity_query"      # 预留：容量查询
    CAPACITY_RESPONSE = "capacity_response"  # 预留：容量查询响应

    # 重平衡（预留：运行时算力重分配）
    REBALANCE_REQUEST = "rebalance_request"    # 预留：重平衡请求
    REBALANCE_ACK = "rebalance_ack"            # 预留：重平衡确认
    REBALANCE_COMPLETE = "rebalance_complete"  # 预留：重平衡完成

    # 配置广播
    CONFIG_UPDATE = "config_update"  # Master -> Worker: 配置更新

    # RPC 控制
    RPC_START = "rpc_start"          # Master -> Worker: 启动 RPC Server
    RPC_STOP = "rpc_stop"            # Master -> Worker: 停止 RPC Server
    RPC_START_ACK = "rpc_start_ack"  # Worker -> Master: RPC 启动确认
    RPC_STOP_ACK = "rpc_stop_ack"    # Worker -> Master: RPC 停止确认

    # 资源查询
    RESOURCE_QUERY = "resource_query"      # Master -> Worker: 查询资源
    RESOURCE_RESPONSE = "resource_response"  # Worker -> Master: 资源响应

    # 模型分发
    MODEL_DISTRIBUTE = "model_distribute"          # Master -> Worker: 模型分发通知
    MODEL_CHUNK = "model_chunk"                    # 预留：模型分片消息类型
    MODEL_DISTRIBUTE_ACK = "model_distribute_ack"  # Worker -> Master: 模型分发确认
    LOAD_MODEL = "load_model"                      # Master -> Worker: 加载模型到内存
    LOAD_MODEL_ACK = "load_model_ack"              # Worker -> Master: 模型加载结果
    UNLOAD_MODEL = "unload_model"                  # Master -> Worker: 卸载模型
    UNLOAD_MODEL_ACK = "unload_model_ack"          # Worker -> Master: 模型卸载结果
    SERVER_SETTINGS = "server_settings"              # Master -> Worker: 服务器设置同步


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


def _envelope_to_dict(envelope: Envelope) -> dict[str, Any]:
    """将信封转换为字典（短键名减少体积）。"""
    data: dict[str, Any] = {
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
        if "t" not in msg_data:
            raise ValueError("消息缺少 't' (type) 字段")
        if "s" not in msg_data:
            raise ValueError("消息缺少 's' (sender_id) 字段")
        if "p" not in msg_data:
            raise ValueError("消息缺少 'p' (payload) 字段")
        if "ts" not in msg_data:
            raise ValueError("消息缺少 'ts' (timestamp) 字段")
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
        if "channel" not in data:
            raise ValueError("信封缺少 'channel' 字段")
        if "message" not in data:
            raise ValueError("信封缺少 'message' 字段")
        msg_data = data["message"]
        if "type" not in msg_data:
            raise ValueError("消息缺少 'type' 字段")
        if "sender_id" not in msg_data:
            raise ValueError("消息缺少 'sender_id' 字段")
        if "payload" not in msg_data:
            raise ValueError("消息缺少 'payload' 字段")
        if "timestamp" not in msg_data:
            raise ValueError("消息缺少 'timestamp' 字段")
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
    """从字节反序列化信封，自动检测 JSON 或 MessagePack 格式。

    检测策略：
    - JSON 对象以 '{' 开头 (0x7B)，且第二个非空白字符应为 '"' 或 '}'
    - MessagePack 数据以其他字节开头
    - 为避免误判，同时检查前两个字节：JSON 对象一定是 '{"' 或 '{}'
    """
    if not data:
        raise ValueError("空数据")
    # JSON 对象必须以 '{"' 或 '{}' 开头（忽略 BOM）
    # 仅检查 0x7B 不够：MessagePack positive fixint 123 的首字节也是 0x7B
    if data[0:1] == b"{" and len(data) > 1 and data[1:2] in (b'"', b"}"):
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
