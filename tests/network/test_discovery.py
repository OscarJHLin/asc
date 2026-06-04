"""测试节点发现：UDP 广播 + 并发扫描。"""

from asc.network.discovery import DiscoveryMessage, NodeDiscovery


class TestDiscoveryMessage:
    """发现消息。"""

    def test_create_broadcast(self):
        msg = DiscoveryMessage(
            node_id="node-1",
            ip="10.0.0.1",
            port=52415,
            magic="ASC_DISCOVER",
        )
        assert msg.node_id == "node-1"
        assert msg.magic == "ASC_DISCOVER"

    def test_to_json(self):
        msg = DiscoveryMessage(
            node_id="node-1",
            ip="10.0.0.1",
            port=52415,
            magic="ASC_DISCOVER",
        )
        import json

        data = json.loads(msg.to_json())
        assert data["magic"] == "ASC_DISCOVER"
        assert data["node_id"] == "node-1"

    def test_from_json(self):
        import json

        raw = json.dumps(
            {
                "magic": "ASC_DISCOVER",
                "node_id": "node-1",
                "ip": "10.0.0.1",
                "port": 52415,
            }
        )
        msg = DiscoveryMessage.from_json(raw)
        assert msg.node_id == "node-1"
        assert msg.port == 52415

    def test_from_json_invalid_magic(self):
        import json

        raw = json.dumps({"magic": "INVALID", "node_id": "n1", "ip": "1.1.1.1", "port": 80})
        msg = DiscoveryMessage.from_json(raw)
        assert msg is None


class TestNodeDiscovery:
    """节点发现。"""

    def test_create(self):
        disc = NodeDiscovery(node_id="node-1", port=52415)
        assert disc.node_id == "node-1"

    def test_get_broadcast_address(self):
        disc = NodeDiscovery(node_id="n1", port=52415)
        addr = disc.broadcast_address
        assert addr[1] == 52416  # discovery port

    def test_parse_discovery_message(self):
        disc = NodeDiscovery(node_id="n1", port=52415)
        import json

        raw = json.dumps(
            {
                "magic": "ASC_DISCOVER",
                "node_id": "node-2",
                "ip": "10.0.0.2",
                "port": 52415,
            }
        ).encode()
        msg = disc.parse_message(raw)
        assert msg is not None
        assert msg.node_id == "node-2"

    def test_parse_invalid_message(self):
        disc = NodeDiscovery(node_id="n1", port=52415)
        msg = disc.parse_message(b"not json")
        assert msg is None

    def test_parse_own_message_ignored(self):
        disc = NodeDiscovery(node_id="n1", port=52415)
        import json

        raw = json.dumps(
            {
                "magic": "ASC_DISCOVER",
                "node_id": "n1",  # 自己
                "ip": "10.0.0.1",
                "port": 52415,
            }
        ).encode()
        msg = disc.parse_message(raw)
        assert msg is None  # 忽略自己的广播

    def test_concurrent_scan_addresses(self):
        """并发扫描应生成正确的 IP 列表。"""
        disc = NodeDiscovery(node_id="n1", port=52415)
        ips = disc.scan_addresses("192.168.1")
        assert len(ips) == 254
        assert "192.168.1.1" in ips
        assert "192.168.1.254" in ips
