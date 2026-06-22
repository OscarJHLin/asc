"""测试 Worker Agent 事件循环。

覆盖：
- register_to_master 发送正确消息
- heartbeat_loop 周期发送心跳
- capacity_report_loop 周期上报容量
- handle_task_dispatch 处理任务分派
- handle_cancel_task 处理任务取消
- graceful stop 优雅停止
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from asc.network.protocol import Channel, Envelope, Message, MessageType
from asc.worker.agent import WorkerAgent


def _make_agent() -> WorkerAgent:
    """创建用于测试的 WorkerAgent，跳过 BenchmarkScore 初始化。"""
    with patch("asc.worker.agent.BenchmarkScore"), patch("asc.worker.agent.HardwareDetector"):
        return WorkerAgent(node_id="test-node-1", port=52415)


def _make_dispatch_envelope(task_id: str = "task-001") -> Envelope:
    """构造 TASK_DISPATCH Envelope。"""
    return Envelope(
        channel=Channel.TASK_DISPATCH,
        message=Message(
            type=MessageType.TASK_DISPATCH,
            sender_id="master-1",
            payload={
                "task_id": task_id,
                "instance_id": "inst-1",
                "prompt": "hello",
            },
        ),
    )


def _make_cancel_envelope(task_id: str = "task-001") -> Envelope:
    """构造 CANCEL_TASK Envelope。"""
    return Envelope(
        channel=Channel.COMMANDS,
        message=Message(
            type=MessageType.CANCEL_TASK,
            sender_id="master-1",
            payload={"task_id": task_id, "reason": "timeout"},
        ),
    )


class TestRegisterToMaster:
    """测试 _register_to_master 发送正确注册消息。"""

    async def test_sends_node_joined_on_discovery_channel(self):
        agent = _make_agent()
        mock_client = AsyncMock()
        mock_client.is_connected = True
        mock_client.send = AsyncMock(return_value=True)
        agent._client = mock_client

        with patch.object(agent, "get_resources") as mock_res:
            mock_res.return_value = MagicMock(
                cpu_count=8,
                memory_total_mb=32768,
                memory_free_mb=24000,
                gpus=[],
                compute_score=1.5,
            )
            await agent._register_to_master()

        mock_client.send.assert_awaited_once()
        envelope: Envelope = mock_client.send.call_args[0][0]

        assert envelope.channel == Channel.DISCOVERY
        assert envelope.message.type == MessageType.NODE_JOINED
        assert envelope.message.sender_id == "test-node-1"
        assert envelope.message.payload["node_id"] == "test-node-1"
        assert envelope.message.payload["port"] == 52415
        assert "resources" in envelope.message.payload
        assert envelope.message.payload["resources"]["cpu_count"] == 8
        assert envelope.message.payload["resources"]["compute_score"] == 1.5

    async def test_no_send_if_client_none(self):
        agent = _make_agent()
        agent._client = None
        # 不应抛异常
        await agent._register_to_master()


class TestHeartbeatLoop:
    """测试 _heartbeat_loop 周期发送心跳。"""

    async def test_sends_heartbeat_periodically(self):
        agent = _make_agent()
        mock_client = AsyncMock()
        mock_client.is_connected = True
        mock_client.send = AsyncMock(return_value=True)
        agent._client = mock_client
        agent._running = True

        send_count = 0

        async def fake_send(envelope):
            nonlocal send_count
            send_count += 1
            if send_count >= 3:
                agent._running = False

        mock_client.send.side_effect = fake_send

        # 使用极短的间隔加速测试
        with patch("asc.worker.agent.HEARTBEAT_INTERVAL", 0.01):
            await agent._heartbeat_loop()

        assert send_count >= 3

        # 验证发送的消息格式
        for call in mock_client.send.call_args_list:
            envelope = call[0][0]
            assert envelope.channel == Channel.HEARTBEATS
            assert envelope.message.type == MessageType.HEARTBEAT
            assert envelope.message.sender_id == "test-node-1"
            assert envelope.message.payload["node_id"] == "test-node-1"

    async def test_stops_on_cancel(self):
        agent = _make_agent()
        mock_client = AsyncMock()
        mock_client.is_connected = True
        mock_client.send = AsyncMock(return_value=True)
        agent._client = mock_client
        agent._running = True

        async def run_loop():
            with patch("asc.worker.agent.HEARTBEAT_INTERVAL", 0.01):
                await agent._heartbeat_loop()

        task = asyncio.create_task(run_loop())
        await asyncio.sleep(0.05)
        task.cancel()
        with contextlib_suppress_cancel():
            await task

        # 不应抛异常，正常退出


class TestCapacityReportLoop:
    """测试 _capacity_report_loop 周期上报容量。"""

    async def test_sends_capacity_report_periodically(self):
        agent = _make_agent()
        mock_client = AsyncMock()
        mock_client.is_connected = True
        mock_client.send = AsyncMock(return_value=True)
        agent._client = mock_client
        agent._running = True

        send_count = 0

        async def fake_send(envelope):
            nonlocal send_count
            send_count += 1
            if send_count >= 2:
                agent._running = False

        mock_client.send.side_effect = fake_send

        with patch.object(agent, "get_resources") as mock_res:
            mock_res.return_value = MagicMock(
                cpu_count=8,
                cpu_percent=25.0,
                memory_total_mb=32768,
                memory_free_mb=24000,
                total_vram_free_mb=0,
                compute_score=1.5,
            )
            with patch("asc.worker.agent.CAPACITY_REPORT_INTERVAL", 0.01):
                await agent._capacity_report_loop()

        assert send_count >= 2

        for call in mock_client.send.call_args_list:
            envelope = call[0][0]
            assert envelope.channel == Channel.CAPACITY
            assert envelope.message.type == MessageType.CAPACITY_REPORT
            assert envelope.message.sender_id == "test-node-1"
            assert envelope.message.payload["node_id"] == "test-node-1"
            assert envelope.message.payload["cpu_count"] == 8
            assert envelope.message.payload["compute_score"] == 1.5


class TestHandleTaskDispatch:
    """测试 _handle_task_dispatch 处理任务分派。"""

    async def test_sends_task_accept(self):
        agent = _make_agent()
        mock_client = AsyncMock()
        mock_client.is_connected = True
        mock_client.send = AsyncMock(return_value=True)
        agent._client = mock_client

        envelope = _make_dispatch_envelope(task_id="task-001")
        await agent._handle_task_dispatch(envelope)

        # 应发送 TASK_ACCEPT
        accept_call = mock_client.send.call_args_list[0]
        accept_env: Envelope = accept_call[0][0]
        assert accept_env.channel == Channel.TASK_DISPATCH
        assert accept_env.message.type == MessageType.TASK_ACCEPT
        assert accept_env.message.payload["task_id"] == "task-001"
        assert accept_env.message.payload["node_id"] == "test-node-1"
        assert accept_env.message.payload["accepted"] is True

    async def test_creates_inference_task(self):
        agent = _make_agent()
        mock_client = AsyncMock()
        mock_client.is_connected = True
        mock_client.send = AsyncMock(return_value=True)
        agent._client = mock_client

        envelope = _make_dispatch_envelope(task_id="task-002")
        await agent._handle_task_dispatch(envelope)

        # 应在 _tasks 中创建任务
        assert "task-002" in agent._tasks

        # 等待推理任务完成
        task = agent._tasks["task-002"]
        await asyncio.sleep(0.1)
        if not task.done():
            task.cancel()
            with contextlib_suppress_cancel():
                await task


class TestHandleCancelTask:
    """测试 _handle_cancel_task 处理任务取消。"""

    async def test_cancels_running_task(self):
        agent = _make_agent()
        mock_client = AsyncMock()
        mock_client.is_connected = True
        mock_client.send = AsyncMock(return_value=True)
        agent._client = mock_client

        # 先分派一个任务
        dispatch_env = _make_dispatch_envelope(task_id="task-cancel-1")
        await agent._handle_task_dispatch(dispatch_env)
        assert "task-cancel-1" in agent._tasks

        # 取消任务
        cancel_env = _make_cancel_envelope(task_id="task-cancel-1")
        await agent._handle_cancel_task(cancel_env)

        # 任务应被移除
        assert "task-cancel-1" not in agent._tasks

    async def test_cancel_nonexistent_task(self):
        agent = _make_agent()
        # 取消不存在的任务，不应抛异常
        cancel_env = _make_cancel_envelope(task_id="nonexistent")
        await agent._handle_cancel_task(cancel_env)


class TestGracefulStop:
    """测试优雅停止。"""

    async def test_stop_sets_running_false(self):
        agent = _make_agent()
        agent._running = True
        await agent.stop()
        assert agent._running is False

    async def test_cleanup_cancels_tasks_and_disconnects(self):
        agent = _make_agent()
        mock_client = AsyncMock()
        mock_client.is_connected = True
        mock_client.disconnect = AsyncMock()
        agent._client = mock_client

        # 创建真实的 asyncio.Task（会立即被取消）
        async def dummy_loop():
            await asyncio.sleep(100)

        agent._heartbeat_task = asyncio.create_task(dummy_loop())
        agent._capacity_task = asyncio.create_task(dummy_loop())

        await agent._cleanup()

        assert agent._heartbeat_task is None
        assert agent._capacity_task is None
        mock_client.disconnect.assert_awaited_once()
        assert agent._client is None

    async def test_run_connects_and_registers(self):
        agent = _make_agent()

        with patch("asc.worker.agent.TCPClient") as mock_tcp_client:
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock(return_value=True)
            mock_client.send = AsyncMock(return_value=True)
            mock_client.is_connected = True
            mock_client.disconnect = AsyncMock()
            mock_tcp_client.return_value = mock_client

            with patch.object(agent, "get_resources") as mock_res:
                mock_res.return_value = MagicMock(
                    cpu_count=8,
                    cpu_percent=25.0,
                    memory_total_mb=32768,
                    memory_free_mb=24000,
                    gpus=[],
                    compute_score=1.5,
                    total_vram_free_mb=0,
                )

                # 让 run() 在注册后很快停止
                async def stop_soon():
                    await asyncio.sleep(0.05)
                    await agent.stop()

                stop_task = asyncio.create_task(stop_soon())

                with (
                    patch("asc.worker.agent.HEARTBEAT_INTERVAL", 0.01),
                    patch("asc.worker.agent.CAPACITY_REPORT_INTERVAL", 0.01),
                ):
                    await agent.run("127.0.0.1", 9999)

                await stop_task

            # 验证 TCPClient 被正确创建
            mock_tcp_client.assert_called_once_with(
                host="127.0.0.1",
                port=9999,
                node_id="test-node-1",
                on_message=agent._on_message,
                on_frame=agent._on_frame,
                auth_token=None,
            )
            mock_client.connect.assert_awaited_once()

            # 验证注册消息被发送
            first_send = mock_client.send.call_args_list[0]
            reg_env: Envelope = first_send[0][0]
            assert reg_env.channel == Channel.DISCOVERY
            assert reg_env.message.type == MessageType.NODE_JOINED

    async def test_run_handles_connection_failure(self):
        agent = _make_agent()

        with patch("asc.worker.agent.TCPClient") as mock_tcp_client:
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock(return_value=False)
            mock_tcp_client.return_value = mock_client

            await agent.run("127.0.0.1", 9999)

            assert agent._running is False


def contextlib_suppress_cancel():
    """辅助：抑制 asyncio.CancelledError。"""
    import contextlib

    return contextlib.suppress(asyncio.CancelledError)
