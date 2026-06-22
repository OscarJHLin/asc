"""故障检测与自动转移模块。

本模块实现高可用集群的核心机制：及时发现节点故障并触发转移，
同时支持优雅关闭以避免误报。

核心组件：
- FailureDetector：基于心跳超时和主动探测检测节点健康状态
- FailoverManager：封装 FailureDetector，提供故障回调和恢复检测
- GracefulShutdown：支持优雅关闭，确保进行中的请求完成后再停止

健康状态机：
    HEALTHY -> SUSPECT（心跳超时，但未达最大错过次数）
    SUSPECT -> FAILED（错过次数达到阈值，或主动探测失败）
    SUSPECT -> HEALTHY（收到心跳，恢复正常）
    FAILED -> HEALTHY（恢复后重新注册）

配置参数（FailoverConfig）：
    - heartbeat_timeout_sec: 心跳超时时间（默认 30 秒）
    - suspect_threshold: 标记为 SUSPECT 的错过次数阈值（默认 2 次）
    - max_missed_heartbeats: 标记为 FAILED 的最大错过次数（默认 3 次）
    - recovery_interval_sec: 恢复检查间隔（默认 60 秒）

使用模式：
    detector = FailureDetector(config=FailoverConfig())
    detector.register("node-1")
    detector.heartbeat("node-1")  # 收到心跳时调用
    health = detector.check("node-1")  # 定期检查
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable


class NodeHealth(str, Enum):
    """节点健康状态。"""

    HEALTHY = "healthy"
    SUSPECT = "suspect"
    FAILED = "failed"


@dataclass
class NodeHeartbeat:
    """节点心跳记录。"""

    node_id: str
    last_seen: float
    missed_count: int = 0


@dataclass
class FailoverConfig:
    """故障转移配置。"""

    heartbeat_timeout_sec: float = 30.0
    suspect_threshold: int = 2
    max_missed_heartbeats: int = 3
    recovery_interval_sec: float = 60.0


class FailureDetector:
    """故障检测器。

    基于心跳超时和可选的主动探测检测节点故障。
    采用"渐进式怀疑"策略：不是一次超时立即判定失败，而是积累错过次数，
    避免网络抖动导致的误判。

    探测函数（probe_fn）：
        若提供，会在心跳超时时调用，进行二次确认。
        例如：尝试 TCP 连接或发送 ping 消息。
        若探测成功，节点状态仍为 HEALTHY；若失败，增加错过计数。
    """

    def __init__(
        self,
        config: FailoverConfig | None = None,
        probe_fn: Callable[[str], bool] | None = None,
    ) -> None:
        self._config = config or FailoverConfig()
        self._probe = probe_fn or (lambda _n: True)
        self._heartbeats: dict[str, NodeHeartbeat] = {}

    def register(self, node_id: str) -> None:
        """注册节点到故障检测器。

        新注册节点初始状态为 HEALTHY（刚加入集群）。
        """
        if node_id not in self._heartbeats:
            self._heartbeats[node_id] = NodeHeartbeat(
                node_id=node_id,
                last_seen=time.time(),
            )

    def unregister(self, node_id: str) -> None:
        """从故障检测器注销节点。

        通常在节点主动离开或确认故障后调用，清理内存。
        """
        self._heartbeats.pop(node_id, None)

    def heartbeat(self, node_id: str) -> None:
        """收到节点心跳，重置超时计数。

        应在每次收到节点心跳消息时调用（如 MasterNode._handle_heartbeat）。
        重置 last_seen 和 missed_count，使节点恢复 HEALTHY 状态。
        """
        if node_id in self._heartbeats:
            self._heartbeats[node_id].last_seen = time.time()
            self._heartbeats[node_id].missed_count = 0

    def check(self, node_id: str) -> NodeHealth:
        """检查节点健康状态。

        检查逻辑：
        1. 若节点未注册，直接返回 FAILED（不应检查未知节点）
        2. 计算距离上次心跳的时间 elapsed
        3. 若 elapsed > heartbeat_timeout_sec，增加 missed_count
        4. 根据 missed_count 判断是 SUSPECT 还是 FAILED
        5. 若心跳未超时，执行主动探测作为二次确认

        Args:
            node_id: 要检查的节点 ID

        Returns:
            NodeHealth 枚举值：HEALTHY、SUSPECT 或 FAILED
        """
        hb = self._heartbeats.get(node_id)
        if hb is None:
            return NodeHealth.FAILED

        elapsed = time.time() - hb.last_seen

        if elapsed > self._config.heartbeat_timeout_sec:
            hb.missed_count += 1
            if hb.missed_count >= self._config.max_missed_heartbeats:
                return NodeHealth.FAILED
            if hb.missed_count >= self._config.suspect_threshold:
                return NodeHealth.SUSPECT

            # 心跳已超时，使用主动探测作为二次确认
            if not self._probe(node_id):
                # 探测也失败，加速故障判定
                if hb.missed_count + 1 >= self._config.max_missed_heartbeats:
                    return NodeHealth.FAILED
                return NodeHealth.SUSPECT

            # 探测成功但心跳已超时，仍标记为 SUSPECT
            return NodeHealth.SUSPECT

        return NodeHealth.HEALTHY

    def check_all(self) -> dict[str, NodeHealth]:
        """检查所有已注册节点的健康状态。"""
        return {nid: self.check(nid) for nid in self._heartbeats}

    def failed_nodes(self) -> list[str]:
        """返回当前已判定为 FAILED 的节点列表。"""
        return [
            nid for nid in self._heartbeats
            if self.check(nid) == NodeHealth.FAILED
        ]


class FailoverManager:
    """故障转移管理器。

    处理节点故障时的自动恢复流程。
    """

    def __init__(
        self,
        detector: FailureDetector | None = None,
        on_node_failed: Callable[[str], None] | None = None,
        on_node_recovered: Callable[[str], None] | None = None,
    ) -> None:
        self._detector = detector or FailureDetector()
        self._on_failed = on_node_failed
        self._on_recovered = on_node_recovered
        self._failed_nodes: set[str] = set()

    def register(self, node_id: str) -> None:
        """注册节点。"""
        self._detector.register(node_id)

    def heartbeat(self, node_id: str) -> None:
        """处理心跳。"""
        self._detector.heartbeat(node_id)

        # 检查是否从失败中恢复
        if node_id in self._failed_nodes:
            health = self._detector.check(node_id)
            if health == NodeHealth.HEALTHY:
                self._failed_nodes.discard(node_id)
                if self._on_recovered is not None:
                    self._on_recovered(node_id)

    def scan(self) -> list[str]:
        """扫描并处理故障节点。

        Returns:
            本次扫描新发现的故障节点列表
        """
        newly_failed: list[str] = []
        for node_id, health in self._detector.check_all().items():
            if health == NodeHealth.FAILED and node_id not in self._failed_nodes:
                self._failed_nodes.add(node_id)
                newly_failed.append(node_id)
                if self._on_failed is not None:
                    self._on_failed(node_id)
        return newly_failed

    def is_failed(self, node_id: str) -> bool:
        """检查节点是否已标记为失败。"""
        return node_id in self._failed_nodes

    def unregister(self, node_id: str) -> None:
        """注销节点，清理所有相关状态。"""
        self._failed_nodes.discard(node_id)
        self._detector.unregister(node_id)


class GracefulShutdown:
    """优雅关闭管理器。"""

    def __init__(
        self,
        timeout_sec: float = 30.0,
        on_shutdown: Callable[[], None] | None = None,
    ) -> None:
        self._timeout = timeout_sec
        self._on_shutdown = on_shutdown
        self._shutting_down = False
        self._active_requests = 0
        self._lock = asyncio.Lock()
        self._zero_event = asyncio.Event()
        self._zero_event.set()  # Initially no active requests

    @property
    def is_shutting_down(self) -> bool:
        return self._shutting_down

    def start_shutdown(self) -> None:
        """开始关闭流程。"""
        self._shutting_down = True

    async def start_request(self) -> bool:
        """尝试开始请求，关闭中拒绝新请求。"""
        async with self._lock:
            if self._shutting_down:
                return False
            self._active_requests += 1
            self._zero_event.clear()
            return True

    async def finish_request(self) -> None:
        """完成请求。"""
        async with self._lock:
            self._active_requests = max(0, self._active_requests - 1)
            if self._active_requests == 0:
                self._zero_event.set()

    async def wait_for_completion(self) -> bool:
        """等待所有请求完成。

        Returns:
            True 如果所有请求已完成，False 如果超时
        """
        try:
            await asyncio.wait_for(self._zero_event.wait(), timeout=self._timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def shutdown(self) -> None:
        """执行完整关闭流程。"""
        self.start_shutdown()
        await self.wait_for_completion()
        if self._on_shutdown is not None:
            self._on_shutdown()
