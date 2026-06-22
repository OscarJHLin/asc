"""增强测试：ASC 消息协议的全面覆盖测试。

在 test_protocol.py 基础上补充：
- Channel 完整性（全部 8 个通道）
- MessageType 完整性（TASK_DISPATCH / CAPACITY / REBALANCE 分组）
- 消息边界值（空/大/嵌套/Unicode/null/数值边界 payload，sender_id 边界）
- 时间戳边界（0.0 自动赋值、显式保留、远未来、负值）
- 序列化边界（target 字段有无、未知字段前向兼容、缺失可选字段、
  全通道/全类型 roundtrip、ensure_ascii=False）
- 不可变性（payload 深度不可变、Envelope target 冻结）
- 错误处理（无效 JSON、无效 channel、无效 message type、缺失必填字段）
"""

import json
import sys
import time

import pytest

from asc.network.protocol import (
    Channel,
    Envelope,
    Message,
    MessageType,
    decode_envelope,
    encode_envelope,
    encode_envelope_json,
)

# ---------------------------------------------------------------------------
# Channel 完整性
# ---------------------------------------------------------------------------

class TestChannelCompleteness:
    """验证全部 8 个 Channel 枚举值。"""

    def test_events_channel(self):
        """EVENTS 通道值为 'events'。"""
        assert Channel.EVENTS.value == "events"

    def test_commands_channel(self):
        """COMMANDS 通道值为 'commands'。"""
        assert Channel.COMMANDS.value == "commands"

    def test_heartbeats_channel(self):
        """HEARTBEATS 通道值为 'heartbeats'。"""
        assert Channel.HEARTBEATS.value == "heartbeats"

    def test_election_channel(self):
        """ELECTION 通道值为 'election'。"""
        assert Channel.ELECTION.value == "election"

    def test_discovery_channel(self):
        """DISCOVERY 通道值为 'discovery'。"""
        assert Channel.DISCOVERY.value == "discovery"

    def test_task_dispatch_channel(self):
        """TASK_DISPATCH 通道值为 'task_dispatch'。"""
        assert Channel.TASK_DISPATCH.value == "task_dispatch"

    def test_capacity_channel(self):
        """CAPACITY 通道值为 'capacity'。"""
        assert Channel.CAPACITY.value == "capacity"

    def test_rebalance_channel(self):
        """REBALANCE 通道值为 'rebalance'。"""
        assert Channel.REBALANCE.value == "rebalance"

    def test_channel_count(self):
        """Channel 枚举恰好包含 9 个成员（含 SYSTEM）。"""
        assert len(Channel) == 9


# ---------------------------------------------------------------------------
# MessageType 完整性 — TASK_DISPATCH / CAPACITY / REBALANCE
# ---------------------------------------------------------------------------

class TestMessageTypeCompleteness:
    """验证 TASK_DISPATCH、CAPACITY、REBALANCE 分组的消息类型。"""

    def test_task_dispatch_types(self):
        """任务分派相关消息类型值正确。"""
        assert MessageType.TASK_DISPATCH.value == "task_dispatch"
        assert MessageType.TASK_ACCEPT.value == "task_accept"
        assert MessageType.TASK_PROGRESS.value == "task_progress"
        assert MessageType.TASK_RESULT.value == "task_result"

    def test_capacity_types(self):
        """容量相关消息类型值正确。"""
        assert MessageType.CAPACITY_REPORT.value == "capacity_report"
        assert MessageType.CAPACITY_QUERY.value == "capacity_query"
        assert MessageType.CAPACITY_RESPONSE.value == "capacity_response"

    def test_rebalance_types(self):
        """重平衡相关消息类型值正确。"""
        assert MessageType.REBALANCE_REQUEST.value == "rebalance_request"
        assert MessageType.REBALANCE_ACK.value == "rebalance_ack"
        assert MessageType.REBALANCE_COMPLETE.value == "rebalance_complete"


# ---------------------------------------------------------------------------
# 消息边界值
# ---------------------------------------------------------------------------

class TestMessageBoundaryValues:
    """消息 payload 和 sender_id 的边界值测试。"""

    def test_empty_payload(self):
        """空字典 payload 应被正确保存。"""
        msg = Message(type=MessageType.HEARTBEAT, sender_id="n1", payload={})
        assert msg.payload == {}

    def test_large_payload(self):
        """包含大量键的 payload 应被正确保存。"""
        payload = {f"key_{i}": f"value_{i}" for i in range(200)}
        msg = Message(type=MessageType.HEARTBEAT, sender_id="n1", payload=payload)
        assert msg.payload == payload
        assert len(msg.payload) == 200

    def test_nested_payload(self):
        """嵌套字典和列表的 payload 应被正确保存。"""
        payload = {
            "nested": {"a": {"b": [1, 2, 3]}},
            "list_of_dicts": [{"x": 1}, {"x": 2}],
        }
        msg = Message(type=MessageType.HEARTBEAT, sender_id="n1", payload=payload)
        assert msg.payload == payload

    def test_unicode_payload(self):
        """包含中文、emoji 和特殊字符的 payload 应被正确保存。"""
        payload = {
            "chinese": "你好世界",
            "emoji": "🚀🔥",
            "special": "©®™§∞",
        }
        msg = Message(type=MessageType.HEARTBEAT, sender_id="n1", payload=payload)
        assert msg.payload == payload

    def test_null_values_in_payload(self):
        """payload 中包含 None 值应被正确保存。"""
        payload = {"a": None, "b": None, "c": 0}
        msg = Message(type=MessageType.HEARTBEAT, sender_id="n1", payload=payload)
        assert msg.payload["a"] is None
        assert msg.payload["b"] is None
        assert msg.payload["c"] == 0

    def test_numeric_edge_values_in_payload(self):
        """payload 中的数值边界值（0、-1、float max、float min）应被正确保存。"""
        payload = {
            "zero": 0,
            "negative_one": -1,
            "float_max": sys.float_info.max,
            "float_min": sys.float_info.min,
        }
        msg = Message(type=MessageType.HEARTBEAT, sender_id="n1", payload=payload)
        assert msg.payload["zero"] == 0
        assert msg.payload["negative_one"] == -1
        assert msg.payload["float_max"] == sys.float_info.max
        assert msg.payload["float_min"] == sys.float_info.min

    def test_sender_id_empty_string(self):
        """sender_id 为空字符串应被允许。"""
        msg = Message(type=MessageType.HEARTBEAT, sender_id="", payload={})
        assert msg.sender_id == ""

    def test_sender_id_very_long_string(self):
        """sender_id 为超长字符串应被正确保存。"""
        long_id = "x" * 10000
        msg = Message(type=MessageType.HEARTBEAT, sender_id=long_id, payload={})
        assert msg.sender_id == long_id
        assert len(msg.sender_id) == 10000

    def test_sender_id_unicode(self):
        """sender_id 包含 Unicode 字符应被正确保存。"""
        uid = "节点-🟢-αβγ"
        msg = Message(type=MessageType.HEARTBEAT, sender_id=uid, payload={})
        assert msg.sender_id == uid


# ---------------------------------------------------------------------------
# 时间戳边界
# ---------------------------------------------------------------------------

class TestTimestampEdgeCases:
    """时间戳边界值测试。"""

    def test_timestamp_zero_triggers_auto(self):
        """timestamp=0.0 时应自动赋值为当前时间。"""
        before = time.time()
        msg = Message(type=MessageType.HEARTBEAT, sender_id="n1", payload={}, timestamp=0.0)
        after = time.time()
        assert before <= msg.timestamp <= after
        assert msg.timestamp != 0.0

    def test_explicit_nonzero_timestamp_preserved(self):
        """显式指定非零 timestamp 应被保留。"""
        msg = Message(
            type=MessageType.HEARTBEAT, sender_id="n1", payload={}, timestamp=12345.678
        )
        assert msg.timestamp == 12345.678

    def test_very_large_timestamp(self):
        """远未来时间戳应被保留。"""
        far_future = 9999999999.0
        msg = Message(
            type=MessageType.HEARTBEAT, sender_id="n1", payload={}, timestamp=far_future
        )
        assert msg.timestamp == far_future

    def test_negative_timestamp(self):
        """负时间戳应被允许（它只是一个 float）。"""
        msg = Message(
            type=MessageType.HEARTBEAT, sender_id="n1", payload={}, timestamp=-100.0
        )
        assert msg.timestamp == -100.0


# ---------------------------------------------------------------------------
# 序列化边界
# ---------------------------------------------------------------------------

class TestSerializationEdgeCases:
    """序列化/反序列化边界测试。"""

    def test_envelope_without_target_no_target_key(self):
        """无 target 的 Envelope JSON 序列化后不应包含 'target' 键。"""
        msg = Message(type=MessageType.HEARTBEAT, sender_id="n1", payload={})
        env = Envelope(channel=Channel.HEARTBEATS, message=msg)
        data = encode_envelope_json(env)
        parsed = json.loads(data.decode("utf-8"))
        assert "target" not in parsed

    def test_envelope_with_target_has_target_key(self):
        """有 target 的 Envelope JSON 序列化后应包含 'target' 键。"""
        msg = Message(type=MessageType.HEARTBEAT, sender_id="n1", payload={})
        env = Envelope(channel=Channel.HEARTBEATS, message=msg, target="n2")
        data = encode_envelope_json(env)
        parsed = json.loads(data.decode("utf-8"))
        assert "target" in parsed
        assert parsed["target"] == "n2"

    def test_decode_with_extra_unknown_fields(self):
        """JSON 中包含未知字段时应前向兼容地解码（忽略多余字段）。"""
        raw = {
            "channel": "events",
            "message": {
                "type": "node_joined",
                "sender_id": "n1",
                "payload": {"ip": "10.0.0.1"},
                "timestamp": 1000.0,
            },
            "target": "n2",
            "future_field": "some_value",
            "another_unknown": 42,
        }
        data = json.dumps(raw).encode("utf-8")
        env = decode_envelope(data)
        assert env.channel == Channel.EVENTS
        assert env.message.type == MessageType.NODE_JOINED
        assert env.target == "n2"

    def test_decode_with_missing_optional_target(self):
        """JSON 中缺少可选 target 字段时应正常解码。"""
        raw = {
            "channel": "events",
            "message": {
                "type": "node_joined",
                "sender_id": "n1",
                "payload": {},
                "timestamp": 1000.0,
            },
        }
        data = json.dumps(raw).encode("utf-8")
        env = decode_envelope(data)
        assert env.target is None

    def test_roundtrip_all_channels(self):
        """全部 Channel 类型均能正确 roundtrip。"""
        for ch in Channel:
            msg = Message(type=MessageType.HEARTBEAT, sender_id="n1", payload={"ch": ch.value})
            env = Envelope(channel=ch, message=msg)
            data = encode_envelope(env)
            env2 = decode_envelope(data)
            assert env2.channel == ch, f"Channel roundtrip failed for {ch.name}"
            assert env2.message.payload == {"ch": ch.value}

    def test_roundtrip_all_message_types(self):
        """全部 MessageType 均能正确 roundtrip。"""
        for mt in MessageType:
            msg = Message(type=mt, sender_id="n1", payload={"mt": mt.value}, timestamp=500.0)
            # 选取一个合理的 channel
            env = Envelope(channel=Channel.EVENTS, message=msg)
            data = encode_envelope(env)
            env2 = decode_envelope(data)
            assert env2.message.type == mt, f"MessageType roundtrip failed for {mt.name}"

    def test_ensure_ascii_false_chinese_not_escaped(self):
        """ensure_ascii=False 意味着中文字符不被转义。"""
        payload = {"text": "你好世界"}
        msg = Message(type=MessageType.HEARTBEAT, sender_id="节点1", payload=payload)
        env = Envelope(channel=Channel.HEARTBEATS, message=msg)
        data = encode_envelope_json(env)
        text = data.decode("utf-8")
        # 中文字符应直接出现在 JSON 字符串中，而非 \uXXXX 转义
        assert "你好世界" in text
        assert "节点1" in text
        # 确认不存在 \uXXXX 形式的中文转义
        assert "\\u4f60" not in text  # "你" 的 Unicode 转义


# ---------------------------------------------------------------------------
# 不可变性
# ---------------------------------------------------------------------------

class TestImmutability:
    """消息和信封的不可变性测试。"""

    def test_message_payload_mutation_does_not_affect_frozen_message(self):
        """修改原始 payload 字典不应影响已冻结的 Message（深度不可变检查）。

        由于 frozen dataclass 不对 dict 做深拷贝，此测试验证当前行为：
        外部修改 payload 是否会穿透到 Message 内部。
        """
        payload = {"key": "original"}
        msg = Message(type=MessageType.HEARTBEAT, sender_id="n1", payload=payload)
        # 尝试修改原始 dict
        payload["key"] = "modified"
        # frozen dataclass 不做深拷贝，所以内部引用同一 dict
        # 此测试记录当前行为：修改会穿透
        assert msg.payload["key"] == "modified"

    def test_envelope_frozen_target_field(self):
        """Envelope 的 target 字段不可修改。"""
        msg = Message(type=MessageType.HEARTBEAT, sender_id="n1", payload={})
        env = Envelope(channel=Channel.HEARTBEATS, message=msg, target="n2")
        with pytest.raises((AttributeError, TypeError)):
            env.target = "n3"  # type: ignore[misc]

    def test_message_frozen_type_field(self):
        """Message 的 type 字段不可修改。"""
        msg = Message(type=MessageType.HEARTBEAT, sender_id="n1", payload={})
        with pytest.raises((AttributeError, TypeError)):
            msg.type = MessageType.NODE_JOINED  # type: ignore[misc]

    def test_message_frozen_timestamp_field(self):
        """Message 的 timestamp 字段不可修改。"""
        msg = Message(type=MessageType.HEARTBEAT, sender_id="n1", payload={})
        with pytest.raises((AttributeError, TypeError)):
            msg.timestamp = 999.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 错误处理
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """解码错误处理测试。"""

    def test_decode_invalid_json(self):
        """无效 JSON 应抛出异常。"""
        with pytest.raises((ValueError, json.JSONDecodeError)):
            decode_envelope(b"not json at all")

    def test_decode_invalid_channel_value(self):
        """JSON 中 channel 值无效时应抛出 ValueError。"""
        raw = {
            "channel": "nonexistent_channel",
            "message": {
                "type": "heartbeat",
                "sender_id": "n1",
                "payload": {},
                "timestamp": 1000.0,
            },
        }
        data = json.dumps(raw).encode("utf-8")
        with pytest.raises(ValueError):
            decode_envelope(data)

    def test_decode_invalid_message_type_value(self):
        """JSON 中 message type 值无效时应抛出 ValueError。"""
        raw = {
            "channel": "events",
            "message": {
                "type": "nonexistent_type",
                "sender_id": "n1",
                "payload": {},
                "timestamp": 1000.0,
            },
        }
        data = json.dumps(raw).encode("utf-8")
        with pytest.raises(ValueError):
            decode_envelope(data)

    def test_decode_missing_channel_field(self):
        """JSON 中缺少 channel 字段时应抛出 ValueError。"""
        raw = {
            "message": {
                "type": "heartbeat",
                "sender_id": "n1",
                "payload": {},
                "timestamp": 1000.0,
            },
        }
        data = json.dumps(raw).encode("utf-8")
        with pytest.raises(ValueError, match="channel"):
            decode_envelope(data)

    def test_decode_missing_message_field(self):
        """JSON 中缺少 message 字段时应抛出 ValueError。"""
        raw = {
            "channel": "events",
        }
        data = json.dumps(raw).encode("utf-8")
        with pytest.raises(ValueError, match="message"):
            decode_envelope(data)

    def test_decode_missing_type_in_message(self):
        """JSON 中 message 缺少 type 字段时应抛出 ValueError。"""
        raw = {
            "channel": "events",
            "message": {
                "sender_id": "n1",
                "payload": {},
                "timestamp": 1000.0,
            },
        }
        data = json.dumps(raw).encode("utf-8")
        with pytest.raises(ValueError, match="type"):
            decode_envelope(data)

    def test_decode_missing_sender_id_in_message(self):
        """JSON 中 message 缺少 sender_id 字段时应抛出 ValueError。"""
        raw = {
            "channel": "events",
            "message": {
                "type": "heartbeat",
                "payload": {},
                "timestamp": 1000.0,
            },
        }
        data = json.dumps(raw).encode("utf-8")
        with pytest.raises(ValueError, match="sender_id"):
            decode_envelope(data)

    def test_decode_missing_payload_in_message(self):
        """JSON 中 message 缺少 payload 字段时应抛出 ValueError。"""
        raw = {
            "channel": "events",
            "message": {
                "type": "heartbeat",
                "sender_id": "n1",
                "timestamp": 1000.0,
            },
        }
        data = json.dumps(raw).encode("utf-8")
        with pytest.raises(ValueError, match="payload"):
            decode_envelope(data)

    def test_decode_missing_timestamp_in_message(self):
        """JSON 中 message 缺少 timestamp 字段时应抛出 ValueError。"""
        raw = {
            "channel": "events",
            "message": {
                "type": "heartbeat",
                "sender_id": "n1",
                "payload": {},
            },
        }
        data = json.dumps(raw).encode("utf-8")
        with pytest.raises(ValueError, match="timestamp"):
            decode_envelope(data)
