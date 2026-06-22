"""节点发现：UDP 广播 + 并发扫描。

提供集群节点的自动发现机制，使新加入的节点无需手动配置 Master 地址即可被发现。

发现机制：
1. UDP 广播：节点启动时向子网广播 DiscoveryMessage，携带自身 ID、IP、端口
2. 消息验证：通过 magic 字段过滤非法消息，防止非 Asc 节点干扰
3. 子网扫描：扫描指定子网的所有 IP（1-254），检测潜在节点

设计原则：
- 零配置：新节点启动后自动广播，无需手动指定 IP
- 安全过滤：magic 字段作为简单认证，防止跨集群干扰
- 自排除：parse_message() 自动忽略自己发出的广播

使用场景：
    局域网内快速组建集群，适合家庭、办公室或同 VPC 的云服务器。
    跨网段或公网部署时，建议结合手动指定 Master IP（如 asc master --host）。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import socket
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

_MAGIC = "ASC_DISCOVER"
_DISCOVERY_PORT_OFFSET = 1  # 发现端口 = 节点端口 + 1


@dataclass(frozen=True)
class DiscoveryMessage:
    """节点发现广播消息。"""

    node_id: str
    ip: str
    port: int
    role: str = "worker"  # master 或 worker
    magic: str = _MAGIC

    def to_json(self) -> str:
        """序列化为 JSON 字符串。"""
        return json.dumps(
            {
                "magic": self.magic,
                "node_id": self.node_id,
                "ip": self.ip,
                "port": self.port,
                "role": self.role,
            }
        )

    @classmethod
    def from_json(cls, raw: str) -> Optional[DiscoveryMessage]:
        """从 JSON 字符串反序列化，magic 不匹配返回 None。"""
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(data, dict) or data.get("magic") != _MAGIC:
            return None
        return cls(
            node_id=data["node_id"],
            ip=data["ip"],
            port=data["port"],
            role=data.get("role", "worker"),
            magic=data["magic"],
        )


class NodeDiscovery:
    """节点发现服务：广播自身、解析发现消息、并发扫描子网。"""

    def __init__(self, node_id: str, port: int, role: str = "worker") -> None:
        self.node_id = node_id
        self.port = port
        self.role = role
        self._udp_socket: Optional[socket.socket] = None
        self._running = False
        self._discovered_masters: list[dict[str, Any]] = []

    @property
    def broadcast_address(self) -> tuple[str, int, int]:
        """返回 UDP 广播地址 (host, port, family)。

        发现端口 = 节点端口 + 1，使用 IPv4 广播地址。
        """
        return ("<broadcast>", self.port + _DISCOVERY_PORT_OFFSET, 2)

    @property
    def discovery_port(self) -> int:
        """返回发现端口。"""
        return self.port + _DISCOVERY_PORT_OFFSET

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

    async def start_discovery_server(self) -> None:
        """启动 UDP 发现服务器，监听广播消息。"""
        self._running = True

        # 创建 UDP socket
        self._udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self._udp_socket.setblocking(False)

        try:
            self._udp_socket.bind(("0.0.0.0", self.discovery_port))
            logger.info("Discovery server listening on UDP port %s", self.discovery_port)
        except OSError as e:
            logger.warning("Failed to bind discovery port %s: %s", self.discovery_port, e)
            return

        # 使用 asyncio 监听 UDP
        loop = asyncio.get_running_loop()

        while self._running:
            try:
                data, addr = await asyncio.wait_for(
                    loop.sock_recvfrom(self._udp_socket, 1024),
                    timeout=1.0,
                )
                msg = self.parse_message(data)
                if msg:
                    logger.debug("Discovered node: %s at %s:%s (role=%s)",
                               msg.node_id, msg.ip, msg.port, msg.role)
                    if msg.role == "master":
                        # 记录发现的 Master
                        master_info = {
                            "node_id": msg.node_id,
                            "ip": msg.ip,
                            "port": msg.port,
                            "discovered_at": asyncio.get_running_loop().time(),
                        }
                        # 去重
                        existing = [m for m in self._discovered_masters
                                  if m["ip"] == msg.ip and m["port"] == msg.port]
                        if not existing:
                            self._discovered_masters.append(master_info)
                            logger.info("Discovered master: %s at %s:%s",
                                      msg.node_id, msg.ip, msg.port)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.debug("Discovery receive error: %s", e)

    async def broadcast_presence(self, local_ip: str) -> None:
        """定期广播自身存在。"""
        if self._udp_socket is None:
            self._udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            self._udp_socket.setblocking(False)

        msg = DiscoveryMessage(
            node_id=self.node_id,
            ip=local_ip,
            port=self.port,
            role=self.role,
        )
        data = msg.to_json().encode("utf-8")

        loop = asyncio.get_running_loop()

        while self._running:
            try:
                await loop.sock_sendto(
                    self._udp_socket,
                    data,
                    ("<broadcast>", self.discovery_port),
                )
                logger.debug("Broadcasted discovery message")
            except Exception as e:
                logger.debug("Broadcast error: %s", e)

            # 每 5 秒广播一次
            await asyncio.sleep(5.0)

    def get_discovered_masters(self) -> list[dict[str, Any]]:
        """获取已发现的 Master 节点列表。"""
        # 清理过期的发现（超过 60 秒）
        try:
            current_time = asyncio.get_running_loop().time()
        except RuntimeError:
            current_time = 0
        self._discovered_masters = [
            m for m in self._discovered_masters
            if current_time - m.get("discovered_at", 0) < 60
        ]
        return self._discovered_masters

    async def discover_master(self, local_ip: str, timeout: float = 10.0) -> Optional[str]:
        """发现 Master 节点，返回 Master IP 地址。

        Args:
            local_ip: 本机 IP 地址
            timeout: 超时时间（秒）

        Returns:
            Master 节点 IP 地址，未找到返回 None
        """
        # 先广播自身
        broadcast_task = asyncio.create_task(self.broadcast_presence(local_ip))

        # 启动临时发现服务器
        discover_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        discover_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        discover_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        discover_socket.setblocking(False)

        try:
            discover_socket.bind(("0.0.0.0", self.discovery_port))
        except OSError:
            pass  # 端口可能被占用

        loop = asyncio.get_running_loop()
        start_time = loop.time()

        try:
            while loop.time() - start_time < timeout:
                try:
                    data, addr = await asyncio.wait_for(
                        loop.sock_recvfrom(discover_socket, 1024),
                        timeout=1.0,
                    )
                    msg = self.parse_message(data)
                    if msg and msg.role == "master":
                        logger.info("Found master: %s at %s:%s",
                                  msg.node_id, msg.ip, msg.port)
                        return msg.ip
                except asyncio.TimeoutError:
                    continue
                except Exception:
                    continue
        finally:
            broadcast_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await broadcast_task
            discover_socket.close()

        return None

    def stop(self) -> None:
        """停止发现服务。"""
        self._running = False
        if self._udp_socket:
            try:
                self._udp_socket.close()
            except Exception:
                pass
            self._udp_socket = None
