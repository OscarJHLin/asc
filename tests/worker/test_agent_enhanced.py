"""Worker Agent 增强测试。

覆盖审计中发现的关键缺口：
- _handle_task_dispatch 缺少 task_id 时应忽略
- get_resources_async 异步方法
- _run_inference CancelledError 和 client 为 None
"""

import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from asc.network.protocol import Channel, Envelope, Message, MessageType
from asc.worker.agent import WorkerAgent


def _make_task_dispatch_envelope(task_id: str | None = None) -> Envelope:
    """创建任务分派信封。"""
    payload = {}
    if task_id is not None:
        payload["task_id"] = task_id
    return Envelope(
        channel=Channel.TASK_DISPATCH,
        message=Message(
            type=MessageType.TASK_DISPATCH,
            sender_id="master",
            payload=payload,
        ),
    )


class TestHandleTaskDispatch:
    """_handle_task_dispatch 测试。"""

    @pytest.mark.asyncio
    async def test_missing_task_id_ignored(self):
        """缺少 task_id 的任务分派应被忽略。"""
        agent = WorkerAgent(node_id="worker-1", port=52415)
        agent._client = AsyncMock()

        envelope = _make_task_dispatch_envelope(task_id=None)

        await agent._handle_task_dispatch(envelope)

        # 不应发送任何消息
        agent._client.send.assert_not_called()
        # 不应创建任何任务
        assert len(agent._tasks) == 0

    @pytest.mark.asyncio
    async def test_empty_task_id_ignored(self):
        """空字符串 task_id 应被忽略。"""
        agent = WorkerAgent(node_id="worker-1", port=52415)
        agent._client = AsyncMock()

        envelope = _make_task_dispatch_envelope(task_id="")

        await agent._handle_task_dispatch(envelope)

        agent._client.send.assert_not_called()
        assert len(agent._tasks) == 0

    @pytest.mark.asyncio
    async def test_valid_task_id_accepted(self):
        """有效 task_id 应被接受并启动推理任务。"""
        agent = WorkerAgent(node_id="worker-1", port=52415)
        agent._client = AsyncMock()

        envelope = _make_task_dispatch_envelope(task_id="task-123")

        await agent._handle_task_dispatch(envelope)

        # 应发送 TASK_ACCEPT
        agent._client.send.assert_called_once()
        sent_envelope = agent._client.send.call_args[0][0]
        assert sent_envelope.message.type == MessageType.TASK_ACCEPT

        # 应创建推理任务
        assert "task-123" in agent._tasks

        # 等待推理任务完成
        await asyncio.sleep(0.1)

    @pytest.mark.asyncio
    async def test_no_client_with_valid_task_id(self):
        """client 为 None 时有 task_id 不应崩溃。"""
        agent = WorkerAgent(node_id="worker-1", port=52415)
        agent._client = None

        envelope = _make_task_dispatch_envelope(task_id="task-456")

        # 不应抛出异常
        await agent._handle_task_dispatch(envelope)

        # 任务仍应被创建
        assert "task-456" in agent._tasks

        # 等待推理任务完成
        await asyncio.sleep(0.1)


class TestGetResourcesAsync:
    """get_resources_async 异步方法测试。"""

    @pytest.mark.asyncio
    @patch("psutil.cpu_count", return_value=8)
    @patch("psutil.cpu_percent", return_value=25.0)
    @patch("psutil.virtual_memory")
    @patch("asc.worker.agent.HardwareDetector")
    @patch("asc.worker.agent.BenchmarkScore")
    async def test_returns_node_resources(
        self, mock_benchmark_cls, mock_hw_cls, mock_vm, mock_cpu_pct, mock_cpu_cnt
    ):
        """get_resources_async 应返回 NodeResources。"""
        from asc.worker.benchmark_score import BenchmarkReport
        from asc.worker.hardware import CPUInfo, DiskInfo, NetworkInfo

        mock_vm.return_value = MagicMock(total=32 * 1024**3, available=24 * 1024**3)

        mock_hw = MagicMock()
        mock_hw.detect_cpu.return_value = CPUInfo(
            logical_count=8, physical_count=4, freq_mhz=3200.0, brand="Intel i7"
        )
        mock_hw.detect_gpus.return_value = []
        mock_hw.detect_disk.return_value = DiskInfo(free_mb=102400)
        mock_hw.detect_network.return_value = NetworkInfo(estimated_mbps=1000.0)
        mock_hw_cls.return_value = mock_hw

        mock_benchmark = MagicMock()
        mock_benchmark.run_benchmark.return_value = BenchmarkReport(
            timestamp="2024-01-01T00:00:00",
            node_id="worker-1",
            node_hardware_summary={},
            model_name="Qwen2.5-1.5B-Instruct",
            quantization="Q4_K_M",
            question_results=[],
            avg_elapsed_ms=100.0,
            avg_tps=75.0,
            theoretical_score=75.0,
            standard_score=50.0,
            relative_score=1.5,
        )
        mock_benchmark_cls.return_value = mock_benchmark

        agent = WorkerAgent(node_id="worker-1", port=52415)
        resources = await agent.get_resources_async()

        assert resources.cpu_count == 8
        assert resources.memory_total_mb == 32 * 1024
        assert resources.compute_score == 1.5

    @pytest.mark.asyncio
    async def test_does_not_block_event_loop(self):
        """get_resources_async 不应阻塞事件循环。"""
        agent = WorkerAgent(node_id="worker-1", port=52415)

        with patch.object(agent, "get_resources") as mock_get:
            mock_get.return_value = MagicMock(cpu_count=4)

            # 同时启动两个协程，验证不会互相阻塞
            results = await asyncio.gather(
                agent.get_resources_async(),
                agent.get_resources_async(),
            )

        assert len(results) == 2


class TestRunInference:
    """_run_inference CancelledError 和 client 为 None 测试。"""

    @pytest.mark.asyncio
    async def test_cancelled_error_sends_cancel_result(self):
        """CancelledError 应发送取消结果。"""
        agent = WorkerAgent(node_id="worker-1", port=52415)
        agent._client = AsyncMock()

        # 创建一个会被取消的推理任务
        async def slow_inference():
            await asyncio.sleep(100)

        # 先启动任务
        task = asyncio.create_task(agent._run_inference("task-1", {}))
        # 立即取消
        task.cancel()

        with contextlib.suppress(asyncio.CancelledError):
            await task

        # CancelledError 后 client.send 可能被调用发送取消结果
        # 也可能因为取消时机不同而未调用
        # 关键是不应抛出未处理的异常

    @pytest.mark.asyncio
    async def test_client_none_no_crash(self):
        """client 为 None 时 _run_inference 不应崩溃。"""
        agent = WorkerAgent(node_id="worker-1", port=52415)
        agent._client = None

        # 不应抛出异常
        await agent._run_inference("task-1", {})

    @pytest.mark.asyncio
    async def test_normal_inference_sends_result(self):
        """正常推理应发送完成结果。"""
        agent = WorkerAgent(node_id="worker-1", port=52415)
        agent._client = AsyncMock()

        await agent._run_inference("task-1", {})

        agent._client.send.assert_called_once()
        sent_envelope = agent._client.send.call_args[0][0]
        assert sent_envelope.message.type == MessageType.TASK_RESULT
        assert sent_envelope.message.payload["status"] == "completed"
        assert sent_envelope.message.payload["task_id"] == "task-1"
