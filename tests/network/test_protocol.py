"""测试消息协议：TypedChannel、Message 序列化/反序列化。

消息协议是节点间通信的基础，替代原有 protocol.py 中未使用的定义。
设计原则：
- 每种消息有明确的 channel（类型化通道）
- JSON 序列化，跨语言兼容
- 消息带时间戳和发送者信息
"""

import json
import time

from asc.network.protocol import (
    Channel,
    Envelope,
    Message,
    MessageType,
    decode_envelope,
    encode_envelope,
)


class TestMessageType:
    """消息类型枚举。"""

    def test_event_message_types(self):
        assert MessageType.NODE_JOINED.value == "node_joined"
        assert MessageType.NODE_LEFT.value == "node_left"
        assert MessageType.INSTANCE_CREATED.value == "instance_created"
        assert MessageType.INSTANCE_DELETED.value == "instance_deleted"
        assert MessageType.TASK_CREATED.value == "task_created"
        assert MessageType.TASK_COMPLETED.value == "task_completed"
        assert MessageType.TASK_FAILED.value == "task_failed"
        assert MessageType.TASK_CANCELLED.value == "task_cancelled"
        assert MessageType.RUNNER_STATUS.value == "runner_status"

    def test_command_message_types(self):
        assert MessageType.CREATE_INSTANCE.value == "create_instance"
        assert MessageType.DELETE_INSTANCE.value == "delete_instance"
        assert MessageType.START_INFERENCE.value == "start_inference"
        assert MessageType.CANCEL_TASK.value == "cancel_task"
        assert MessageType.SHUTDOWN_RUNNER.value == "shutdown_runner"

    def test_control_message_types(self):
        assert MessageType.HEARTBEAT.value == "heartbeat"
        assert MessageType.DISCOVER.value == "discover"
        assert MessageType.DISCOVER_RESPONSE.value == "discover_response"
        assert MessageType.ELECTION.value == "election"
        assert MessageType.REQUEST_EVENT_LOG.value == "request_event_log"
        assert MessageType.EVENT_LOG_CHUNK.value == "event_log_chunk"


class TestMessage:
    """消息值对象。"""

    def test_create_message(self):
        msg = Message(
            type=MessageType.HEARTBEAT,
            sender_id="node-1",
            payload={"status": "alive"},
        )
        assert msg.type == MessageType.HEARTBEAT
        assert msg.sender_id == "node-1"
        assert msg.payload == {"status": "alive"}
        assert msg.timestamp > 0

    def test_message_auto_timestamp(self):
        before = time.time()
        msg = Message(type=MessageType.HEARTBEAT, sender_id="n1", payload={})
        after = time.time()
        assert before <= msg.timestamp <= after

    def test_message_frozen(self):
        msg = Message(type=MessageType.HEARTBEAT, sender_id="n1", payload={})
        try:
            msg.sender_id = "n2"  # type: ignore[misc]
            raise AssertionError("Should be immutable")
        except (AttributeError, TypeError):
            pass


class TestEnvelope:
    """信封：消息 + 通道 + 可选目标。"""

    def test_create_envelope(self):
        msg = Message(type=MessageType.HEARTBEAT, sender_id="n1", payload={})
        env = Envelope(channel=Channel.HEARTBEATS, message=msg)
        assert env.channel == Channel.HEARTBEATS
        assert env.message == msg
        assert env.target is None

    def test_envelope_with_target(self):
        msg = Message(type=MessageType.HEARTBEAT, sender_id="n1", payload={})
        env = Envelope(channel=Channel.HEARTBEATS, message=msg, target="n2")
        assert env.target == "n2"

    def test_envelope_frozen(self):
        msg = Message(type=MessageType.HEARTBEAT, sender_id="n1", payload={})
        env = Envelope(channel=Channel.HEARTBEATS, message=msg)
        try:
            env.channel = Channel.EVENTS  # type: ignore[misc]
            raise AssertionError("Should be immutable")
        except (AttributeError, TypeError):
            pass


class TestChannel:
    """通道定义。"""

    def test_channel_names(self):
        assert Channel.EVENTS.value == "events"
        assert Channel.COMMANDS.value == "commands"
        assert Channel.HEARTBEATS.value == "heartbeats"
        assert Channel.ELECTION.value == "election"
        assert Channel.DISCOVERY.value == "discovery"


class TestSerialization:
    """序列化/反序列化。"""

    def test_encode_decode_roundtrip(self):
        msg = Message(
            type=MessageType.NODE_JOINED,
            sender_id="node-1",
            payload={"ip": "10.0.0.1", "port": 52415},
        )
        env = Envelope(channel=Channel.EVENTS, message=msg)
        data = encode_envelope(env)
        assert isinstance(data, bytes)

        env2 = decode_envelope(data)
        assert env2.channel == env.channel
        assert env2.message.type == env.message.type
        assert env2.message.sender_id == env.message.sender_id
        assert env2.message.payload == env.message.payload
        assert env2.target == env.target

    def test_encode_is_json(self):
        msg = Message(type=MessageType.HEARTBEAT, sender_id="n1", payload={})
        env = Envelope(channel=Channel.HEARTBEATS, message=msg)
        data = encode_envelope(env)
        parsed = json.loads(data.decode("utf-8"))
        assert parsed["channel"] == "heartbeats"
        assert parsed["message"]["type"] == "heartbeat"

    def test_decode_invalid_data_raises(self):
        try:
            decode_envelope(b"not json at all")
            raise AssertionError("Should raise ValueError")
        except (ValueError, json.JSONDecodeError):
            pass

    def test_encode_with_target(self):
        msg = Message(type=MessageType.START_INFERENCE, sender_id="n1", payload={"prompt": "hi"})
        env = Envelope(channel=Channel.COMMANDS, message=msg, target="n2")
        data = encode_envelope(env)
        env2 = decode_envelope(data)
        assert env2.target == "n2"

    def test_roundtrip_preserves_timestamp(self):
        msg = Message(type=MessageType.HEARTBEAT, sender_id="n1", payload={})
        env = Envelope(channel=Channel.HEARTBEATS, message=msg)
        data = encode_envelope(env)
        env2 = decode_envelope(data)
        assert env2.message.timestamp == msg.timestamp
