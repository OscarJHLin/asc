"""故障检测与自动转移模块。

提供：
- 节点故障检测（心跳超时、连接探测）
- 自动故障转移
- 优雅关闭支持
"""

from __future__ import annotations

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

    基于心跳超时和连接探测检测节点故障。
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
        """注册节点到故障检测器。"""
        if node_id not in self._heartbeats:
            self._heartbeats[node_id] = NodeHeartbeat(
                node_id=node_id,
                last_seen=time.time(),
            )

    def unregister(self, node_id: str) -> None:
        """从故障检测器注销节点。"""
        self._heartbeats.pop(node_id, None)

    def heartbeat(self, node_id: str) -> None:
        """收到节点心跳。"""
        if node_id in self._heartbeats:
            self._heartbeats[node_id].last_seen = time.time()
            self._heartbeats[node_id].missed_count = 0

    def check(self, node_id: str) -> NodeHealth:
        """检查节点健康状态。"""
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

        # 主动探测
        if not self._probe(node_id):
            hb.missed_count += 1
            if hb.missed_count >= self._config.max_missed_heartbeats:
                return NodeHealth.FAILED
            return NodeHealth.SUSPECT

        return NodeHealth.HEALTHY

    def check_all(self) -> dict[str, NodeHealth]:
        """检查所有节点。"""
        return {nid: self.check(nid) for nid in self._heartbeats}

    def failed_nodes(self) -> list[str]:
        """返回已失败的节点列表。"""
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

    @property
    def is_shutting_down(self) -> bool:
        return self._shutting_down

    def start_shutdown(self) -> None:
        """开始关闭流程。"""
        self._shutting_down = True

    def start_request(self) -> bool:
        """尝试开始请求，关闭中拒绝新请求。"""
        if self._shutting_down:
            return False
        self._active_requests += 1
        return True

    def finish_request(self) -> None:
        """完成请求。"""
        self._active_requests = max(0, self._active_requests - 1)

    def wait_for_completion(self) -> bool:
        """等待所有请求完成。

        Returns:
            True 如果所有请求已完成，False 如果超时
        """
        start = time.time()
        while self._active_requests > 0:
            if time.time() - start > self._timeout:
                return False
            time.sleep(0.1)
        return True

    def shutdown(self) -> None:
        """执行完整关闭流程。"""
        self.start_shutdown()
        self.wait_for_completion()
        if self._on_shutdown is not None:
            self._on_shutdown()
