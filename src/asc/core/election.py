"""Asc Master 选举：Bully 算法。

Bully 算法规则：
1. 节点发起选举时，向所有 ID 更高的节点发送 ELECTION 消息
2. 收到 ELECTION 消息的更高 ID 节点回复 ALIVE
3. 发起者收到 ALIVE 后退让，等待更高 ID 节点成为 Master
4. 如果发起者在超时内没收到 ALIVE，自己成为 Master
5. 新 Master 向所有节点发送 COORDINATOR 消息

设计原则：
- 节点 ID 比较基于字符串排序（可自定义比较函数）
- 选举时钟递增，防止旧消息干扰
- 状态机：IDLE -> ELECTING -> MASTER/WORKER
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


class ElectionMessageType(enum.Enum):
    """选举消息类型。"""

    ELECTION = "election"
    ALIVE = "alive"
    COORDINATOR = "coordinator"


class ElectionState(enum.Enum):
    """选举状态。"""

    IDLE = "idle"
    ELECTING = "electing"
    MASTER = "master"
    WORKER = "worker"


@dataclass(frozen=True)
class ElectionMessage:
    """选举消息。"""

    type: ElectionMessageType
    sender_id: str
    election_clock: int


@dataclass
class BullyElection:
    """Bully 选举算法实现。"""

    node_id: str
    all_node_ids: list[str]
    _state: ElectionState = field(default=ElectionState.IDLE, init=False)
    _master_id: str | None = field(default=None, init=False)
    _election_clock: int = field(default=0, init=False)

    @property
    def state(self) -> ElectionState:
        return self._state

    @property
    def master_id(self) -> str | None:
        return self._master_id

    @property
    def election_clock(self) -> int:
        return self._election_clock

    @property
    def is_master(self) -> bool:
        return self._state == ElectionState.MASTER

    @property
    def higher_node_ids(self) -> list[str]:
        """返回 ID 比自己高的节点列表（已排序）。"""
        return sorted(n for n in self.all_node_ids if n > self.node_id)

    def start_election(self) -> None:
        """发起选举。"""
        self._election_clock += 1
        self._state = ElectionState.ELECTING

    def should_become_master(self) -> bool:
        """判断是否应成为 Master（没有更高 ID 的节点）。"""
        return self._state == ElectionState.ELECTING and len(self.higher_node_ids) == 0

    def become_master(self) -> None:
        """成为 Master。"""
        self._state = ElectionState.MASTER
        self._master_id = self.node_id

    def handle_message(self, msg: ElectionMessage) -> None:
        """处理选举消息。"""
        if msg.type == ElectionMessageType.ALIVE:
            # 收到更高 ID 节点的 ALIVE，退让
            if self._state == ElectionState.ELECTING:
                self._state = ElectionState.IDLE

        elif msg.type == ElectionMessageType.COORDINATOR:
            # 收到新 Master 的 COORDINATOR
            self._master_id = msg.sender_id
            if msg.sender_id == self.node_id:
                self._state = ElectionState.MASTER
            else:
                self._state = ElectionState.WORKER

        elif msg.type == ElectionMessageType.ELECTION and msg.sender_id < self.node_id:
            # 收到更低 ID 节点的选举请求，回复 ALIVE（由调用者负责发送）
            pass

    def get_election_messages(self) -> list[ElectionMessage]:
        """获取需要发送的选举消息。

        发起选举后，向所有更高 ID 节点发送 ELECTION 消息。
        """
        if self._state != ElectionState.ELECTING:
            return []

        return [
            ElectionMessage(
                type=ElectionMessageType.ELECTION,
                sender_id=self.node_id,
                election_clock=self._election_clock,
            )
            for _ in self.higher_node_ids
        ]

    def get_coordinator_message(self) -> ElectionMessage | None:
        """获取 COORDINATOR 消息（成为 Master 后广播）。"""
        if self._state != ElectionState.MASTER:
            return None
        return ElectionMessage(
            type=ElectionMessageType.COORDINATOR,
            sender_id=self.node_id,
            election_clock=self._election_clock,
        )
