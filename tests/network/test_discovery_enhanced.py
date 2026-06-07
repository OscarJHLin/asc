"""增强测试节点发现协议。

覆盖 DiscoveryMessage 序列化/反序列化边界、
NodeDiscovery 解析与扫描细节、端口边界值。
"""

from __future__ import annotations

import json

import pytest

from asc.network.discovery import _MAGIC, DiscoveryMessage, NodeDiscovery

# ---------------------------------------------------------------------------
# DiscoveryMessage 边界条件
# ---------------------------------------------------------------------------


class TestDiscoveryMessageEdgeCases:
    """DiscoveryMessage 边界条件测试。"""

    def test_to_json_from_json_roundtrip_preserves_all_fields(self):
        """to_json/from_json 往返应精确保留所有字段。"""
        msg = DiscoveryMessage(
            node_id="node-42",
            ip="10.0.0.42",
            port=52415,
        )
        raw = msg.to_json()
        restored = DiscoveryMessage.from_json(raw)
        assert restored is not None
        assert restored.node_id == "node-42"
        assert restored.ip == "10.0.0.42"
        assert restored.port == 52415
        assert restored.magic == _MAGIC

    def test_from_json_with_malformed_json(self):
        """from_json 传入非 JSON 字符串应返回 None。"""
        assert DiscoveryMessage.from_json("this is not json at all") is None

    def test_from_json_missing_node_id(self):
        """from_json 缺少 node_id 字段应抛出 KeyError。"""
        raw = json.dumps({"magic": _MAGIC, "ip": "10.0.0.1", "port": 80})
        with pytest.raises(KeyError):
            DiscoveryMessage.from_json(raw)

    def test_from_json_missing_ip(self):
        """from_json 缺少 ip 字段应抛出 KeyError。"""
        raw = json.dumps({"magic": _MAGIC, "node_id": "n1", "port": 80})
        with pytest.raises(KeyError):
            DiscoveryMessage.from_json(raw)

    def test_from_json_missing_port(self):
        """from_json 缺少 port 字段应抛出 KeyError。"""
        raw = json.dumps({"magic": _MAGIC, "node_id": "n1", "ip": "10.0.0.1"})
        with pytest.raises(KeyError):
            DiscoveryMessage.from_json(raw)

    def test_from_json_with_extra_unknown_fields(self):
        """from_json 包含额外未知字段应正常解析（前向兼容）。"""
        raw = json.dumps({
            "magic": _MAGIC,
            "node_id": "n1",
            "ip": "10.0.0.1",
            "port": 8080,
            "version": "2.0",
            "capabilities": ["gpu", "tpu"],
        })
        msg = DiscoveryMessage.from_json(raw)
        assert msg is not None
        assert msg.node_id == "n1"
        assert msg.port == 8080

    def test_from_json_with_none_input(self):
        """from_json 传入 None 应返回 None。"""
        assert DiscoveryMessage.from_json(None) is None  # type: ignore[arg-type]

    def test_magic_default_value(self):
        """magic 字段默认值应为 _MAGIC。"""
        msg = DiscoveryMessage(node_id="n1", ip="10.0.0.1", port=80)
        assert msg.magic == "ASC_DISCOVER"


# ---------------------------------------------------------------------------
# NodeDiscovery 边界条件
# ---------------------------------------------------------------------------


class TestNodeDiscoveryEdgeCases:
    """NodeDiscovery 边界条件测试。"""

    def test_parse_message_with_non_utf8_bytes(self):
        """parse_message 传入非 UTF-8 字节应返回 None。"""
        disc = NodeDiscovery(node_id="n1", port=52415)
        # 0xFF 不是合法 UTF-8
        result = disc.parse_message(b"\xff\xfe\xfd")
        assert result is None

    def test_parse_message_with_wrong_magic(self):
        """parse_message 传入 magic 不匹配的合法 JSON 应返回 None。"""
        disc = NodeDiscovery(node_id="n1", port=52415)
        raw = json.dumps({
            "magic": "WRONG_MAGIC",
            "node_id": "n2",
            "ip": "10.0.0.2",
            "port": 52415,
        }).encode()
        result = disc.parse_message(raw)
        assert result is None

    def test_parse_message_from_different_node(self):
        """parse_message 接收来自不同节点的发现消息应正常解析。"""
        disc = NodeDiscovery(node_id="n1", port=52415)
        raw = json.dumps({
            "magic": _MAGIC,
            "node_id": "n2",
            "ip": "10.0.0.2",
            "port": 52416,
        }).encode()
        msg = disc.parse_message(raw)
        assert msg is not None
        assert msg.node_id == "n2"
        assert msg.ip == "10.0.0.2"
        assert msg.port == 52416

    def test_broadcast_address_port_plus_one(self):
        """broadcast_address 的端口应为节点端口 +1。"""
        disc = NodeDiscovery(node_id="n1", port=52415)
        host, port, family = disc.broadcast_address
        assert host == "<broadcast>"
        assert port == 52416
        assert family == 2  # AF_INET

    def test_scan_addresses_generates_254_ips(self):
        """scan_addresses 应生成 254 个 IP（1-254，不含 0 和 255）。"""
        disc = NodeDiscovery(node_id="n1", port=52415)
        ips = disc.scan_addresses("192.168.1")
        assert len(ips) == 254
        assert ips[0] == "192.168.1.1"
        assert ips[-1] == "192.168.1.254"
        assert "192.168.1.0" not in ips
        assert "192.168.1.255" not in ips

    def test_scan_addresses_with_different_subnet(self):
        """scan_addresses 使用不同子网前缀应正确生成 IP。"""
        disc = NodeDiscovery(node_id="n1", port=52415)
        ips = disc.scan_addresses("10.0.0")
        assert len(ips) == 254
        assert ips[0] == "10.0.0.1"
        assert ips[-1] == "10.0.0.254"


# ---------------------------------------------------------------------------
# 端口边界值
# ---------------------------------------------------------------------------


class TestPortBoundaryValues:
    """端口边界值测试。"""

    def test_port_zero_ephemeral(self):
        """port=0（临时端口）应正常创建 DiscoveryMessage。"""
        msg = DiscoveryMessage(node_id="n1", ip="10.0.0.1", port=0)
        assert msg.port == 0
        raw = msg.to_json()
        restored = DiscoveryMessage.from_json(raw)
        assert restored is not None
        assert restored.port == 0

    def test_port_max_65535(self):
        """port=65535（最大端口）应正常创建 DiscoveryMessage。"""
        msg = DiscoveryMessage(node_id="n1", ip="10.0.0.1", port=65535)
        assert msg.port == 65535
        raw = msg.to_json()
        restored = DiscoveryMessage.from_json(raw)
        assert restored is not None
        assert restored.port == 65535

    def test_discovery_port_overflow_65535_plus_1(self):
        """端口65535 +1 = 65536 的发现端口边界情况。"""
        disc = NodeDiscovery(node_id="n1", port=65535)
        host, port, family = disc.broadcast_address
        # 65535 + 1 = 65536，超出有效端口范围但代码不做限制
        assert port == 65536

    def test_node_discovery_port_zero_broadcast(self):
        """port=0 时发现端口应为1。"""
        disc = NodeDiscovery(node_id="n1", port=0)
        host, port, family = disc.broadcast_address
        assert port == 1
