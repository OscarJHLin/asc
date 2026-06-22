"""Asc Master 选举：Bully 算法实现。

Bully 算法是一种经典的分布式 leader 选举算法，核心思想是"ID 最大的存活节点成为 Master"。

完整规则：
1. 节点发起选举时，向所有 ID 更高的节点发送 ELECTION 消息
2. 收到 ELECTION 消息的更高 ID 节点回复 ALIVE
3. 发起者收到 ALIVE 后退让，等待更高 ID 节点成为 Master
4. 如果发起者在超时内没收到 ALIVE，自己成为 Master
5. 新 Master 向所有节点发送 COORDINATOR 消息宣告主权

设计原则与关键机制：
- 节点 ID 比较基于字符串排序（可自定义比较函数）
- 选举时钟（election_clock）单调递增，防止旧消息干扰新选举
- 状态机：IDLE -> ELECTING -> MASTER/WORKER
- 超时机制：默认 5 秒，防止无限等待

协议集成：
    ElectionMessage 可通过 election_to_envelope() 转换为 Envelope，
    经由 Binary Frame 协议的 Channel.ELECTION 通道传输。
    接收方通过 envelope_to_election() 还原为 ElectionMessage。

线程安全：
    本模块为纯同步实现，无 asyncio 依赖。MasterNode 在事件循环中调用时
    需通过 asyncio.Lock 保护状态变更，或确保单线程访问。
"""

from __future__ import annotations

import enum
import time
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
    """Bully 选举算法实现。

    维护本地选举状态，不直接发送网络消息。调用方通过 get_election_messages()
    和 get_coordinator_message() 获取待发送的消息，自行决定传输方式。

    这种设计将"算法逻辑"与"网络传输"解耦，便于单元测试和适配不同传输层。

    Attributes:
        node_id: 本节点唯一标识
        all_node_ids: 集群中所有已知节点 ID 列表（含本节点）
        election_timeout_sec: 选举超时时间（秒），默认 5 秒
    """

    node_id: str
    all_node_ids: list[str]
    _state: ElectionState = field(default=ElectionState.IDLE, init=False)
    _master_id: str | None = field(default=None, init=False)
    _election_clock: int = field(default=0, init=False)
    _election_start_time: float = field(default=0.0, init=False)
    election_timeout_sec: float = 5.0

    def update_node_ids(self, node_ids: list[str]) -> None:
        """动态更新集群节点 ID 列表。"""
        self.all_node_ids = node_ids

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
        self._election_start_time = time.time()

    def should_become_master(self) -> bool:
        """判断是否应成为 Master（没有更高 ID 的节点）。"""
        return self._state == ElectionState.ELECTING and len(self.higher_node_ids) == 0

    def become_master(self) -> None:
        """成为 Master。"""
        self._state = ElectionState.MASTER
        self._master_id = self.node_id

    def check_timeout(self) -> bool:
        """检查选举是否超时。

        Returns:
            True 如果超时且仍处于 ELECTING 状态
        """
        if self._state != ElectionState.ELECTING:
            return False
        if self._election_start_time <= 0:
            return False
        return (time.time() - self._election_start_time) > self.election_timeout_sec

    def handle_timeout(self) -> None:
        """处理选举超时 — 如果没有更高 ID 节点响应，自己成为 Master。"""
        if self.check_timeout():
            if len(self.higher_node_ids) == 0 or self.should_become_master():
                self.become_master()
            else:
                # 有更高 ID 节点但未响应，重新发起选举
                self.start_election()

    def handle_message(self, msg: ElectionMessage) -> None:
        """处理选举消息，更新本地状态。

        注意：本方法只修改本地状态，不发送网络消息。
        对于需要回复的场景（如收到更低 ID 的 ELECTION），
        调用者应检查返回值或状态后自行发送响应。

        Args:
            msg: 收到的选举消息
        """
        if msg.type == ElectionMessageType.ALIVE:
            # 收到更高 ID 节点的 ALIVE，说明有更高优先级的节点存活，退让
            if self._state == ElectionState.ELECTING:
                self._state = ElectionState.IDLE

        elif msg.type == ElectionMessageType.COORDINATOR:
            # 忽略过时的 COORDINATOR 消息（基于选举时钟）
            # 这是防止网络延迟导致旧 COORDINATOR 覆盖新选举结果的关键机制
            if msg.election_clock < self._election_clock:
                return
            self._election_clock = msg.election_clock
            self._master_id = msg.sender_id
            if msg.sender_id == self.node_id:
                self._state = ElectionState.MASTER
            else:
                self._state = ElectionState.WORKER

        elif msg.type == ElectionMessageType.ELECTION and msg.sender_id < self.node_id:
            # 收到更低 ID 节点的选举请求，本节点应回复 ALIVE（由调用者负责发送）
            # 这里仅做状态确认，不实际发送消息
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


# ---------------------------------------------------------------------------
# 协议桥接：ElectionMessage <-> Envelope
# ---------------------------------------------------------------------------

# ElectionMessageType 到 MessageType 的映射
_ELECTION_TYPE_MAP = {
    ElectionMessageType.ELECTION: "election",
    ElectionMessageType.ALIVE: "election_alive",
    ElectionMessageType.COORDINATOR: "election_coordinator",
}

# MessageType 值到 ElectionMessageType 的反向映射
_ELECTION_TYPE_REVERSE = {v: k for k, v in _ELECTION_TYPE_MAP.items()}


def election_to_envelope(msg: ElectionMessage, target: str | None = None):
    """将 ElectionMessage 转换为 Envelope，用于通过 Binary Frame 传输。

    Args:
        msg: 选举消息
        target: 可选目标节点 ID

    Returns:
        Envelope 对象，可通过 TCP 传输
    """
    from asc.network.protocol import Channel, Envelope, Message, MessageType

    msg_type_value = _ELECTION_TYPE_MAP[msg.type]
    msg_type = MessageType(msg_type_value)

    return Envelope(
        channel=Channel.ELECTION,
        message=Message(
            type=msg_type,
            sender_id=msg.sender_id,
            payload={
                "election_clock": msg.election_clock,
            },
        ),
        target=target,
    )


def envelope_to_election(envelope) -> ElectionMessage | None:
    """从 Envelope 还原 ElectionMessage。

    Args:
        envelope: 收到的 Envelope

    Returns:
        ElectionMessage 或 None（如果不是选举消息）
    """
    msg_type_value = envelope.message.type.value
    if msg_type_value not in _ELECTION_TYPE_REVERSE:
        return None

    return ElectionMessage(
        type=_ELECTION_TYPE_REVERSE[msg_type_value],
        sender_id=envelope.message.sender_id,
        election_clock=envelope.message.payload.get("election_clock", 0),
    )
