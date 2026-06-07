"""测试 Master 事件循环。

验证 MasterNode 的 run/stop、消息处理、心跳检测、故障检测等。
使用 mock 替代 TCPServer，不实际启动网络。
"""

from __future__ import annotations

import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

from asc.core.failover import FailoverConfig, FailoverManager, FailureDetector
from asc.master.main import MasterNode
from asc.network.protocol import Channel, Envelope, Message, MessageType
from asc.scheduler.load_balancer import LoadBalancer
from asc.types import NodeId


def _make_envelope(
    msg_type: MessageType,
    sender_id: str = "worker-1",
    payload: dict | None = None,
    channel: Channel = Channel.HEARTBEATS,
) -> Envelope:
    """构造测试用 Envelope。"""
    return Envelope(
        channel=channel,
        message=Message(
            type=msg_type,
            sender_id=sender_id,
            payload=payload or {},
        ),
    )


def _make_mock_tcp_server() -> AsyncMock:
    """构造 mock TCPServer。"""
    server = AsyncMock()
    server.start = AsyncMock()
    server.stop = AsyncMock()
    server.broadcast = AsyncMock()
    server.send = AsyncMock()
    server.port = 52414
    server.is_running = False
    return server


async def _run_then_stop(master: MasterNode) -> None:
    """启动 master 后立即停止，用于测试 run() 启动行为。"""
    run_task = asyncio.create_task(master.run())
    await asyncio.sleep(0.05)
    await master.stop()
    run_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await run_task


class TestRunStartsTCPServer:
    """run() 启动 TCP Server。"""

    async def test_run_creates_tcp_server_if_none(self):
        """未注入 TCPServer 时，run() 应自动创建并启动。"""
        mock_server = _make_mock_tcp_server()

        with patch("asc.master.main.TCPServer", return_value=mock_server):
            master = MasterNode(node_id="master")
            await _run_then_stop(master)

        mock_server.start.assert_awaited_once()

    async def test_run_uses_injected_tcp_server(self):
        """注入的 TCPServer 应被直接使用。"""
        mock_server = _make_mock_tcp_server()
        master = MasterNode(node_id="master", tcp_server=mock_server)
        await _run_then_stop(master)

        mock_server.start.assert_awaited_once()

    async def test_run_starts_background_tasks(self):
        """run() 应启动健康检查和调度两个后台任务。"""
        mock_server = _make_mock_tcp_server()
        master = MasterNode(node_id="master", tcp_server=mock_server)

        run_task = asyncio.create_task(master.run())
        await asyncio.sleep(0.05)

        assert len(master._tasks) == 2
        assert master._running is True

        await master.stop()
        run_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await run_task


class TestHandleNodeJoined:
    """_handle_node_joined 更新集群状态。"""

    async def test_node_joined_updates_state(self):
        """节点注册后，集群状态应包含新节点。"""
        master = MasterNode(node_id="master")

        envelope = _make_envelope(
            msg_type=MessageType.NODE_JOINED,
            sender_id="conn-1",
            payload={"node_id": "worker-1", "ip": "10.0.0.2", "port": 52415},
            channel=Channel.DISCOVERY,
        )
        await master._handle_node_joined(envelope)

        assert NodeId("worker-1") in master.state.nodes
        node_info = master.state.nodes[NodeId("worker-1")]
        assert node_info.ip == "10.0.0.2"
        assert node_info.port == 52415

    async def test_node_joined_registers_in_failure_detector(self):
        """节点注册后，故障检测器应包含该节点。"""
        detector = FailureDetector()
        master = MasterNode(node_id="master", failure_detector=detector)

        envelope = _make_envelope(
            msg_type=MessageType.NODE_JOINED,
            sender_id="conn-1",
            payload={"node_id": "worker-1", "ip": "10.0.0.2", "port": 52415},
            channel=Channel.DISCOVERY,
        )
        await master._handle_node_joined(envelope)

        assert "worker-1" in detector._heartbeats

    async def test_node_joined_registers_in_load_balancer(self):
        """节点注册后，负载均衡器应包含该节点。"""
        lb = LoadBalancer()
        master = MasterNode(node_id="master", load_balancer=lb)

        envelope = _make_envelope(
            msg_type=MessageType.NODE_JOINED,
            sender_id="conn-1",
            payload={"node_id": "worker-1", "ip": "10.0.0.2", "port": 52415},
            channel=Channel.DISCOVERY,
        )
        await master._handle_node_joined(envelope)

        assert "worker-1" in lb._active_requests

    async def test_node_joined_updates_conn_map(self):
        """节点注册后，连接映射应更新。"""
        master = MasterNode(node_id="master")

        envelope = _make_envelope(
            msg_type=MessageType.NODE_JOINED,
            sender_id="conn-1",
            payload={"node_id": "worker-1", "ip": "10.0.0.2", "port": 52415},
            channel=Channel.DISCOVERY,
        )
        await master._handle_node_joined(envelope)

        assert master._conn_node_map["conn-1"] == "worker-1"


class TestHandleHeartbeat:
    """_handle_heartbeat 更新故障检测器。"""

    async def test_heartbeat_updates_failure_detector(self):
        """心跳应更新故障检测器的 last_seen。"""
        detector = FailureDetector()
        master = MasterNode(node_id="master", failure_detector=detector)

        # 先注册节点
        detector.register("worker-1")
        old_last_seen = detector._heartbeats["worker-1"].last_seen

        # 等一点时间让时间戳不同
        await asyncio.sleep(0.01)

        envelope = _make_envelope(
            msg_type=MessageType.HEARTBEAT,
            payload={"node_id": "worker-1"},
        )
        await master._handle_heartbeat(envelope)

        new_last_seen = detector._heartbeats["worker-1"].last_seen
        assert new_last_seen > old_last_seen

    async def test_heartbeat_updates_failover_manager(self):
        """心跳应更新故障转移管理器。"""
        detector = FailureDetector()
        failover = FailoverManager(detector=detector)
        master = MasterNode(
            node_id="master",
            failure_detector=detector,
            failover_manager=failover,
        )

        # 注册节点
        failover.register("worker-1")

        envelope = _make_envelope(
            msg_type=MessageType.HEARTBEAT,
            payload={"node_id": "worker-1"},
        )
        await master._handle_heartbeat(envelope)

        # 心跳后 missed_count 应为 0
        assert detector._heartbeats["worker-1"].missed_count == 0

    async def test_heartbeat_ignores_empty_node_id(self):
        """空 node_id 的心跳应被忽略。"""
        detector = FailureDetector()
        master = MasterNode(node_id="master", failure_detector=detector)

        envelope = _make_envelope(
            msg_type=MessageType.HEARTBEAT,
            payload={"node_id": ""},
        )
        await master._handle_heartbeat(envelope)

        # 不应有任何节点注册
        assert len(detector._heartbeats) == 0


class TestHealthCheckLoop:
    """_health_check_loop 检测故障节点。"""

    async def test_health_check_detects_failed_nodes(self):
        """健康检查应检测到超时节点并触发 NodeLeft。"""
        config = FailoverConfig(
            heartbeat_timeout_sec=0.01,
            suspect_threshold=1,
            max_missed_heartbeats=1,
        )
        detector = FailureDetector(config=config, probe_fn=lambda _: False)
        failover = FailoverManager(
            detector=detector,
            on_node_failed=MagicMock(),
        )
        master = MasterNode(
            node_id="master",
            failure_detector=detector,
            failover_manager=failover,
        )

        # 注册节点并加入集群状态
        detector.register("worker-1")
        failover.register("worker-1")
        master.process_node_joined(NodeId("worker-1"), "10.0.0.2", 52415)

        # 等待心跳超时
        await asyncio.sleep(0.05)

        # 手动触发一次健康检查（模拟 _health_check_loop 的一次迭代）
        newly_failed = failover.scan()
        assert "worker-1" in newly_failed

        # 处理故障节点
        for node_id in newly_failed:
            master.process_node_left(NodeId(node_id))

        assert NodeId("worker-1") not in master.state.nodes

    async def test_health_check_loop_runs_periodically(self):
        """健康检查循环应周期运行。"""
        config = FailoverConfig(
            heartbeat_timeout_sec=0.01,
            suspect_threshold=1,
            max_missed_heartbeats=1,
        )
        detector = FailureDetector(config=config, probe_fn=lambda _: False)
        failover = FailoverManager(
            detector=detector,
            on_node_failed=MagicMock(),
        )
        mock_server = _make_mock_tcp_server()
        master = MasterNode(
            node_id="master",
            failure_detector=detector,
            failover_manager=failover,
            tcp_server=mock_server,
        )

        # 注册节点
        detector.register("worker-1")
        failover.register("worker-1")
        master.process_node_joined(NodeId("worker-1"), "10.0.0.2", 52415)

        # 启动 run（健康检查间隔 10s，但我们会很快停止）
        # 用 mock 缩短 sleep 时间
        with patch("asc.master.main.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            sleep_count = 0

            async def fake_sleep(seconds):
                nonlocal sleep_count
                sleep_count += 1
                if sleep_count >= 2:
                    master._running = False
                    return
                await asyncio.sleep(0.01)

            mock_sleep.side_effect = fake_sleep

            run_task = asyncio.create_task(master.run())
            await asyncio.sleep(0.1)

        # 清理
        with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
            await asyncio.wait_for(run_task, timeout=1.0)
        await master.stop()


class TestGracefulStop:
    """优雅停止。"""

    async def test_stop_cancels_background_tasks(self):
        """stop() 应取消所有后台任务。"""
        mock_server = _make_mock_tcp_server()
        master = MasterNode(node_id="master", tcp_server=mock_server)

        run_task = asyncio.create_task(master.run())
        await asyncio.sleep(0.05)

        assert len(master._tasks) == 2
        assert master._running is True

        await master.stop()

        assert master._running is False
        assert len(master._tasks) == 0

        run_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await run_task

    async def test_stop_stops_tcp_server(self):
        """stop() 应停止 TCP Server。"""
        mock_server = _make_mock_tcp_server()
        master = MasterNode(node_id="master", tcp_server=mock_server)

        run_task = asyncio.create_task(master.run())
        await asyncio.sleep(0.05)
        await master.stop()

        mock_server.stop.assert_awaited_once()

        run_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await run_task

    async def test_stop_without_run(self):
        """未运行时 stop() 不应报错。"""
        master = MasterNode(node_id="master")
        await master.stop()  # 不应抛异常


class TestOnMessage:
    """_on_message 消息分派。"""

    async def test_on_message_dispatches_node_joined(self):
        """NODE_JOINED 消息应被分派到 _handle_node_joined。"""
        master = MasterNode(node_id="master")

        envelope = _make_envelope(
            msg_type=MessageType.NODE_JOINED,
            sender_id="conn-1",
            payload={"node_id": "worker-1", "ip": "10.0.0.2", "port": 52415},
            channel=Channel.DISCOVERY,
        )
        await master._on_message("conn-1", envelope)

        assert NodeId("worker-1") in master.state.nodes

    async def test_on_message_dispatches_heartbeat(self):
        """HEARTBEAT 消息应被分派到 _handle_heartbeat。"""
        detector = FailureDetector()
        master = MasterNode(node_id="master", failure_detector=detector)
        detector.register("worker-1")

        envelope = _make_envelope(
            msg_type=MessageType.HEARTBEAT,
            payload={"node_id": "worker-1"},
        )
        await master._on_message("conn-1", envelope)

        # 心跳后 missed_count 应为 0
        assert detector._heartbeats["worker-1"].missed_count == 0

    async def test_on_message_ignores_unknown_type(self):
        """未知消息类型应被忽略。"""
        master = MasterNode(node_id="master")

        envelope = _make_envelope(
            msg_type=MessageType.DISCOVER,
            payload={},
        )
        await master._on_message("conn-1", envelope)

        # 不应有任何状态变化
        assert len(master.state.nodes) == 0


class TestHandleTaskResult:
    """_handle_task_result 处理任务结果。"""

    async def test_task_completed(self):
        """任务完成应产生 TaskCompleted 事件。"""
        master = MasterNode(node_id="master")
        # 先创建节点和实例、任务
        master.process_node_joined(NodeId("n1"), "10.0.0.1", 52415)
        from asc.types.commands import CreateInstance, StartInference

        master.process_create_instance(CreateInstance(model_id="m", sharding="tensor"))
        inst_id = list(master.state.instances.keys())[0]
        master.process_start_inference(StartInference(instance_id=inst_id, prompt="hi"))
        task_id = list(master.state.tasks.keys())[0]

        envelope = _make_envelope(
            msg_type=MessageType.TASK_RESULT,
            payload={
                "task_id": str(task_id),
                "status": "completed",
                "output": "Hello!",
            },
            channel=Channel.TASK_DISPATCH,
        )
        await master._handle_task_result(envelope)

        from asc.types.state import TaskStatus

        assert master.state.tasks[task_id].status == TaskStatus.COMPLETED
        assert master.state.tasks[task_id].output == "Hello!"

    async def test_task_failed(self):
        """任务失败应产生 TaskFailed 事件。"""
        master = MasterNode(node_id="master")
        master.process_node_joined(NodeId("n1"), "10.0.0.1", 52415)
        from asc.types.commands import CreateInstance, StartInference

        master.process_create_instance(CreateInstance(model_id="m", sharding="tensor"))
        inst_id = list(master.state.instances.keys())[0]
        master.process_start_inference(StartInference(instance_id=inst_id, prompt="hi"))
        task_id = list(master.state.tasks.keys())[0]

        envelope = _make_envelope(
            msg_type=MessageType.TASK_RESULT,
            payload={
                "task_id": str(task_id),
                "status": "failed",
                "error": "OOM",
            },
            channel=Channel.TASK_DISPATCH,
        )
        await master._handle_task_result(envelope)

        from asc.types.state import TaskStatus

        assert master.state.tasks[task_id].status == TaskStatus.FAILED
        assert master.state.tasks[task_id].error == "OOM"


class TestHandleCapacityReport:
    """_handle_capacity_report 处理容量上报。"""

    async def test_capacity_report_updates_load_balancer(self):
        """容量上报应更新负载均衡器的 compute_score。"""
        lb = LoadBalancer()
        master = MasterNode(node_id="master", load_balancer=lb)

        envelope = _make_envelope(
            msg_type=MessageType.CAPACITY_REPORT,
            payload={"node_id": "worker-1", "compute_score": 3.5},
            channel=Channel.CAPACITY,
        )
        await master._handle_capacity_report(envelope)

        assert lb._node_scores["worker-1"] == 3.5
