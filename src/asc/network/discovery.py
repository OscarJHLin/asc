"""节点发现：UDP 广播 + 并发扫描。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

_MAGIC = "ASC_DISCOVER"
_DISCOVERY_PORT_OFFSET = 1  # 发现端口 = 节点端口 + 1


@dataclass(frozen=True)
class DiscoveryMessage:
    """节点发现广播消息。"""

    node_id: str
    ip: str
    port: int
    magic: str = _MAGIC

    def to_json(self) -> str:
        """序列化为 JSON 字符串。"""
        return json.dumps(
            {
                "magic": self.magic,
                "node_id": self.node_id,
                "ip": self.ip,
                "port": self.port,
            }
        )

    @classmethod
    def from_json(cls, raw: str) -> Optional[DiscoveryMessage]:
        """从 JSON 字符串反序列化，magic 不匹配返回 None。"""
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
        if data.get("magic") != _MAGIC:
            return None
        return cls(
            node_id=data["node_id"],
            ip=data["ip"],
            port=data["port"],
            magic=data["magic"],
        )


class NodeDiscovery:
    """节点发现服务：广播自身、解析发现消息、并发扫描子网。"""

    def __init__(self, node_id: str, port: int) -> None:
        self.node_id = node_id
        self.port = port

    @property
    def broadcast_address(self) -> tuple[str, int, int]:
        """返回 UDP 广播地址 (host, port, family)。

        发现端口 = 节点端口 + 1，使用 IPv4 广播地址。
        """
        return ("<broadcast>", self.port + _DISCOVERY_PORT_OFFSET, 2)

    def parse_message(self, raw: bytes) -> Optional[DiscoveryMessage]:
        """解析收到的 UDP 数据，忽略自身消息和无效消息。"""
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return None
        msg = DiscoveryMessage.from_json(text)
        if msg is None:
            return None
        # 忽略自己发出的广播
        if msg.node_id == self.node_id:
            return None
        return msg

    def scan_addresses(self, subnet: str) -> list[str]:
        """生成子网内所有主机 IP 列表 (1..254)。"""
        return [f"{subnet}.{i}" for i in range(1, 255)]
