"""测试故障检测与自动转移模块。

覆盖：
- FailureDetector：注册、心跳、健康检查、超时降级、主动探测、注销
- FailoverManager：注册、心跳恢复、扫描故障、回调触发、is_failed
- GracefulShutdown：请求控制、关闭流程、等待完成
"""

import asyncio
import time

from asc.core.failover import (
    FailoverConfig,
    FailoverManager,
    FailureDetector,
    GracefulShutdown,
    NodeHealth,
)


class TestNodeHealth:
    """NodeHealth 枚举值。"""

    def test_health_values(self):
        assert NodeHealth.HEALTHY == "healthy"
        assert NodeHealth.SUSPECT == "suspect"
        assert NodeHealth.FAILED == "failed"


class TestFailoverConfig:
    """FailoverConfig 默认值。"""

    def test_default_values(self):
        cfg = FailoverConfig()
        assert cfg.heartbeat_timeout_sec == 30.0
        assert cfg.suspect_threshold == 2
        assert cfg.max_missed_heartbeats == 3
        assert cfg.recovery_interval_sec == 60.0


class TestFailureDetectorRegisterAndHeartbeat:
    """注册与心跳保持节点健康。"""

    def test_register_then_check_is_healthy(self):
        detector = FailureDetector()
        detector.register("node-a")
        assert detector.check("node-a") == NodeHealth.HEALTHY

    def test_heartbeat_resets_missed_count(self):
        detector = FailureDetector(
            config=FailoverConfig(
                heartbeat_timeout_sec=0.01, suspect_threshold=1, max_missed_heartbeats=2
            )
        )
        detector.register("node-b")
        time.sleep(0.02)
        # 第一次检查会进入 SUSPECT
        assert detector.check("node-b") == NodeHealth.SUSPECT
        # 发送心跳恢复
        detector.heartbeat("node-b")
        assert detector.check("node-b") == NodeHealth.HEALTHY

    def test_check_unregistered_node_returns_failed(self):
        detector = FailureDetector()
        assert detector.check("unknown") == NodeHealth.FAILED


class TestFailureDetectorMissedHeartbeats:
    """心跳超时导致 SUSPECT 再到 FAILED。"""

    def test_missed_heartbeats_suspect_then_failed(self):
        cfg = FailoverConfig(
            heartbeat_timeout_sec=0.01, suspect_threshold=1, max_missed_heartbeats=2
        )
        detector = FailureDetector(config=cfg)
        detector.register("node-c")
        time.sleep(0.02)
        assert detector.check("node-c") == NodeHealth.SUSPECT
        time.sleep(0.02)
        assert detector.check("node-c") == NodeHealth.FAILED

    def test_exactly_at_max_missed_returns_failed(self):
        cfg = FailoverConfig(
            heartbeat_timeout_sec=0.005, suspect_threshold=1, max_missed_heartbeats=1
        )
        detector = FailureDetector(config=cfg)
        detector.register("node-d")
        time.sleep(0.01)
        assert detector.check("node-d") == NodeHealth.FAILED


class TestFailureDetectorProbeFn:
    """自定义 probe_fn 在心跳超时后作为二次确认。"""

    def test_probe_false_after_timeout_leads_to_suspect(self):
        """心跳超时 + probe 返回 False 导致 SUSPECT。"""
        cfg = FailoverConfig(
            heartbeat_timeout_sec=0.0, suspect_threshold=1, max_missed_heartbeats=3
        )
        detector = FailureDetector(config=cfg, probe_fn=lambda _n: False)
        detector.register("node-e")
        # 心跳已超时（timeout=0），probe 返回 False，应标记为 SUSPECT
        assert detector.check("node-e") == NodeHealth.SUSPECT

    def test_probe_false_after_timeout_leads_to_failed(self):
        """心跳超时 + probe 返回 False 多次导致 FAILED。"""
        cfg = FailoverConfig(
            heartbeat_timeout_sec=0.0, suspect_threshold=1, max_missed_heartbeats=2
        )
        detector = FailureDetector(config=cfg, probe_fn=lambda _n: False)
        detector.register("node-f")
        assert detector.check("node-f") == NodeHealth.SUSPECT
        assert detector.check("node-f") == NodeHealth.FAILED

    def test_probe_not_called_when_heartbeat_ok(self):
        """心跳正常时，即使 probe 返回 False 也应保持 HEALTHY。"""
        cfg = FailoverConfig(
            heartbeat_timeout_sec=100.0, suspect_threshold=1, max_missed_heartbeats=2
        )
        detector = FailureDetector(config=cfg, probe_fn=lambda _n: False)
        detector.register("node-g")
        # 心跳未超时，probe 不应影响健康状态
        assert detector.check("node-g") == NodeHealth.HEALTHY

    def test_probe_true_keeps_healthy(self):
        cfg = FailoverConfig(
            heartbeat_timeout_sec=10.0, suspect_threshold=1, max_missed_heartbeats=2
        )
        detector = FailureDetector(config=cfg, probe_fn=lambda _n: True)
        detector.register("node-g2")
        assert detector.check("node-g2") == NodeHealth.HEALTHY


class TestFailureDetectorUnregister:
    """注销节点。"""

    def test_unregister_removes_node(self):
        detector = FailureDetector()
        detector.register("node-h")
        assert detector.check("node-h") == NodeHealth.HEALTHY
        detector.unregister("node-h")
        assert detector.check("node-h") == NodeHealth.FAILED

    def test_unregister_unknown_is_noop(self):
        detector = FailureDetector()
        detector.unregister("node-x")
        assert detector.check("node-x") == NodeHealth.FAILED


class TestFailureDetectorCheckAllAndFailedNodes:
    """批量检查与失败列表。"""

    def test_check_all_returns_all_statuses(self):
        cfg = FailoverConfig(
            heartbeat_timeout_sec=0.01, suspect_threshold=1, max_missed_heartbeats=3
        )
        detector = FailureDetector(config=cfg)
        detector.register("node-i")
        detector.register("node-j")
        time.sleep(0.02)
        # 第一次 check_all 会让每个节点 missed_count +1，状态为 SUSPECT
        assert detector.check_all() == {
            "node-i": NodeHealth.SUSPECT,
            "node-j": NodeHealth.SUSPECT,
        }

    def test_failed_nodes_returns_only_failed(self):
        cfg = FailoverConfig(
            heartbeat_timeout_sec=0.005, suspect_threshold=1, max_missed_heartbeats=2
        )
        detector = FailureDetector(config=cfg)
        detector.register("node-k")
        detector.register("node-l")
        time.sleep(0.01)
        # 先对 node-k 连续检查两次使其进入 FAILED，node-l 只检查一次保持 SUSPECT
        detector.check("node-k")
        detector.check("node-k")
        assert detector.failed_nodes() == ["node-k"]


class TestFailoverManagerRegister:
    """FailoverManager 注册。"""

    def test_register_then_not_failed(self):
        manager = FailoverManager()
        manager.register("node-m")
        assert not manager.is_failed("node-m")


class TestFailoverManagerHeartbeatRecovery:
    """心跳恢复失败状态。"""

    def test_heartbeat_recovers_failed_node(self):
        cfg = FailoverConfig(
            heartbeat_timeout_sec=0.005, suspect_threshold=1, max_missed_heartbeats=1
        )
        detector = FailureDetector(config=cfg)
        manager = FailoverManager(detector=detector)
        manager.register("node-n")
        time.sleep(0.01)
        manager.scan()
        assert manager.is_failed("node-n")

        # 心跳恢复
        manager.heartbeat("node-n")
        assert not manager.is_failed("node-n")

    def test_heartbeat_recovery_invokes_callback(self):
        recovered = []
        cfg = FailoverConfig(
            heartbeat_timeout_sec=0.005, suspect_threshold=1, max_missed_heartbeats=1
        )
        detector = FailureDetector(config=cfg)
        manager = FailoverManager(
            detector=detector,
            on_node_failed=lambda _n: None,
            on_node_recovered=lambda nid: recovered.append(nid),
        )
        manager.register("node-o")
        time.sleep(0.01)
        manager.scan()
        manager.heartbeat("node-o")
        assert recovered == ["node-o"]


class TestFailoverManagerScan:
    """扫描检测故障并触发回调。"""

    def test_scan_detects_newly_failed(self):
        cfg = FailoverConfig(
            heartbeat_timeout_sec=0.005, suspect_threshold=1, max_missed_heartbeats=1
        )
        detector = FailureDetector(config=cfg)
        manager = FailoverManager(detector=detector)
        manager.register("node-p")
        time.sleep(0.01)
        newly_failed = manager.scan()
        assert newly_failed == ["node-p"]
        assert manager.is_failed("node-p")

    def test_scan_does_not_duplicate_failed(self):
        cfg = FailoverConfig(
            heartbeat_timeout_sec=0.005, suspect_threshold=1, max_missed_heartbeats=1
        )
        detector = FailureDetector(config=cfg)
        manager = FailoverManager(detector=detector)
        manager.register("node-q")
        time.sleep(0.01)
        assert manager.scan() == ["node-q"]
        assert manager.scan() == []

    def test_scan_invokes_on_node_failed_callback(self):
        failed = []
        cfg = FailoverConfig(
            heartbeat_timeout_sec=0.005, suspect_threshold=1, max_missed_heartbeats=1
        )
        detector = FailureDetector(config=cfg)
        manager = FailoverManager(
            detector=detector,
            on_node_failed=lambda nid: failed.append(nid),
        )
        manager.register("node-r")
        time.sleep(0.01)
        manager.scan()
        assert failed == ["node-r"]


class TestFailoverManagerIsFailed:
    """is_failed 边界情况。"""

    def test_is_failed_for_unregistered_node(self):
        manager = FailoverManager()
        assert not manager.is_failed("node-s")


class TestGracefulShutdownRequestLifecycle:
    """请求生命周期控制。"""

    async def test_start_request_allowed_normally(self):
        gs = GracefulShutdown()
        assert await gs.start_request() is True
        assert gs._active_requests == 1

    async def test_start_request_rejected_after_shutdown(self):
        gs = GracefulShutdown()
        gs.start_shutdown()
        assert await gs.start_request() is False

    async def test_finish_request_decrements_count(self):
        gs = GracefulShutdown()
        await gs.start_request()
        await gs.start_request()
        await gs.finish_request()
        assert gs._active_requests == 1
        await gs.finish_request()
        assert gs._active_requests == 0

    async def test_finish_request_does_not_go_negative(self):
        gs = GracefulShutdown()
        await gs.finish_request()
        assert gs._active_requests == 0


class TestGracefulShutdownWaitForCompletion:
    """等待所有请求完成。"""

    async def test_wait_returns_true_when_no_requests(self):
        gs = GracefulShutdown()
        assert await gs.wait_for_completion() is True

    async def test_wait_returns_true_after_requests_finish(self):
        gs = GracefulShutdown(timeout_sec=1.0)
        await gs.start_request()
        await gs.start_request()

        async def finish_later():
            await asyncio.sleep(0.05)
            await gs.finish_request()
            await gs.finish_request()

        asyncio.create_task(finish_later())
        assert await gs.wait_for_completion() is True

    async def test_wait_returns_false_on_timeout(self):
        gs = GracefulShutdown(timeout_sec=0.01)
        await gs.start_request()
        assert await gs.wait_for_completion() is False
        await gs.finish_request()


class TestGracefulShutdownShutdown:
    """完整关闭流程。"""

    async def test_shutdown_invokes_callback(self):
        called = []
        gs = GracefulShutdown(on_shutdown=lambda: called.append(1))
        await gs.shutdown()
        assert called == [1]
        assert gs.is_shutting_down

    async def test_shutdown_without_callback(self):
        gs = GracefulShutdown()
        await gs.shutdown()
        assert gs.is_shutting_down
