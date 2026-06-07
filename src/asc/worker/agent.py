"""Asc Worker Agent。

Worker Agent 运行在 Worker 节点上，是集群的数据平面。负责执行 Master 分派的任务，
并持续上报自身状态，使 Master 能够做出合理的调度决策。

核心职责：
- 资源查询：检测 CPU/GPU/内存/磁盘/网络等硬件信息
- 基准测试：运行轻量级 benchmark 计算算力评分（compute_score）
- RPC Server 生命周期管理：启动/停止 llama.cpp RPC Server
- 事件循环：连接 Master、周期性心跳（5s）、容量上报（30s）、任务分派处理

事件循环流程：
    1. 通过 TCPClient 连接到 Master
    2. 发送 NODE_JOINED 注册消息（携带资源信息）
    3. 启动心跳循环：每 5 秒发送 HEARTBEAT
    4. 启动容量上报循环：每 30 秒发送 CAPACITY_REPORT
    5. 监听任务分派：收到 TASK_DISPATCH 后执行推理并返回结果

容错机制：
- 连接断开时自动停止事件循环，等待外部重启
- 任务取消支持：收到 CANCEL_TASK 时取消对应的 asyncio.Task
- 优雅关闭：stop() 方法设置标志位，循环自然退出后清理资源

扩展性：
    _run_inference() 当前为占位实现，需对接实际的 Engine 或 Runner。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field
from typing import Any

import psutil

from asc.network.protocol import Channel, Envelope, Message, MessageType
from asc.network.transport import TCPClient
from asc.worker.benchmark_score import BenchmarkScore
from asc.worker.gpu_info import GPUInfo
from asc.worker.hardware import HardwareDetector
from asc.worker.rpc_server import RpcServer

logger = logging.getLogger(__name__)

# 定时间隔
HEARTBEAT_INTERVAL = 5.0
CAPACITY_REPORT_INTERVAL = 30.0


@dataclass(frozen=True)
class NodeResources:
    """节点资源信息。"""

    cpu_count: int
    cpu_percent: float
    memory_total_mb: int
    memory_free_mb: int
    gpus: list[GPUInfo] = field(default_factory=list)
    compute_score: float = 0.0
    cpu_physical_count: int = 0
    cpu_freq_mhz: float = 0.0
    cpu_brand: str = ""
    disk_free_mb: int = 0
    network_mbps: float = 0.0

    @property
    def total_vram_free_mb(self) -> int:
        return sum(g.vram_free_mb for g in self.gpus)


class WorkerAgent:
    """Worker 节点 Agent。

    提供资源查询、健康检查、RPC 管理等功能。
    通过 HTTP API 对外暴露。
    支持事件循环：连接 Master、心跳、容量上报、任务分派。
    """

    def __init__(
        self,
        node_id: str,
        port: int = 52415,
        benchmark_score: BenchmarkScore | None = None,
        hardware_detector: HardwareDetector | None = None,
        rpc_server: RpcServer | None = None,
    ) -> None:
        self.node_id = node_id
        self.port = port
        self._rpc_server = rpc_server or RpcServer()
        self._benchmark_score = benchmark_score or BenchmarkScore(node_id=node_id)
        self._hardware_detector = hardware_detector or HardwareDetector()

        # 事件循环状态
        self._client: TCPClient | None = None
        self._running = False
        self._heartbeat_task: asyncio.Task | None = None
        self._capacity_task: asyncio.Task | None = None
        self._tasks: dict[str, asyncio.Task] = {}

    # ------------------------------------------------------------------
    # 事件循环
    # ------------------------------------------------------------------

    async def run(self, master_host: str, master_port: int) -> None:
        """Worker 主事件循环。

        1. 连接 Master
        2. 发送 REGISTER
        3. 启动心跳/容量上报定时任务
        4. 监听任务分派
        """
        self._running = True

        self._client = TCPClient(
            host=master_host,
            port=master_port,
            node_id=self.node_id,
            on_message=self._on_message,
        )

        connected = await self._client.connect()
        if not connected:
            logger.error("无法连接到 Master %s:%s", master_host, master_port)
            self._running = False
            return

        logger.info("已连接到 Master %s:%s", master_host, master_port)

        await self._register_to_master()

        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._capacity_task = asyncio.create_task(self._capacity_report_loop())

        try:
            while self._running:
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            pass
        finally:
            await self._cleanup()

    async def stop(self) -> None:
        """优雅停止。"""
        self._running = False

    async def _cleanup(self) -> None:
        """清理所有异步任务和连接。"""
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._heartbeat_task
            self._heartbeat_task = None

        if self._capacity_task is not None:
            self._capacity_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._capacity_task
            self._capacity_task = None

        for _task_id, task in list(self._tasks.items()):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks.clear()

        if self._client is not None:
            await self._client.disconnect()
            self._client = None

    async def _on_message(self, envelope: Envelope) -> None:
        """处理从 Master 收到的消息。"""
        msg_type = envelope.message.type
        if msg_type == MessageType.TASK_DISPATCH:
            await self._handle_task_dispatch(envelope)
        elif msg_type == MessageType.CANCEL_TASK:
            await self._handle_cancel_task(envelope)

    async def _register_to_master(self) -> None:
        """向 Master 注册。"""
        if self._client is None:
            return

        resources = self.get_resources()
        envelope = Envelope(
            channel=Channel.DISCOVERY,
            message=Message(
                type=MessageType.NODE_JOINED,
                sender_id=self.node_id,
                payload={
                    "node_id": self.node_id,
                    "port": self.port,
                    "resources": {
                        "cpu_count": resources.cpu_count,
                        "memory_total_mb": resources.memory_total_mb,
                        "memory_free_mb": resources.memory_free_mb,
                        "gpus": [
                            {
                                "index": g.index,
                                "name": g.name,
                                "vram_total_mb": g.vram_total_mb,
                                "vram_free_mb": g.vram_free_mb,
                            }
                            for g in resources.gpus
                        ],
                        "compute_score": resources.compute_score,
                    },
                },
            ),
        )
        await self._client.send(envelope)
        logger.info("已向 Master 发送注册消息")

    async def _heartbeat_loop(self) -> None:
        """周期发送心跳 (5s)。"""
        try:
            while self._running:
                if self._client is not None and self._client.is_connected:
                    envelope = Envelope(
                        channel=Channel.HEARTBEATS,
                        message=Message(
                            type=MessageType.HEARTBEAT,
                            sender_id=self.node_id,
                            payload={"node_id": self.node_id},
                        ),
                    )
                    await self._client.send(envelope)
                await asyncio.sleep(HEARTBEAT_INTERVAL)
        except asyncio.CancelledError:
            pass

    async def _capacity_report_loop(self) -> None:
        """周期上报容量 (30s)。"""
        try:
            while self._running:
                if self._client is not None and self._client.is_connected:
                    resources = self.get_resources()
                    envelope = Envelope(
                        channel=Channel.CAPACITY,
                        message=Message(
                            type=MessageType.CAPACITY_REPORT,
                            sender_id=self.node_id,
                            payload={
                                "node_id": self.node_id,
                                "cpu_count": resources.cpu_count,
                                "cpu_percent": resources.cpu_percent,
                                "memory_total_mb": resources.memory_total_mb,
                                "memory_free_mb": resources.memory_free_mb,
                                "total_vram_free_mb": resources.total_vram_free_mb,
                                "compute_score": resources.compute_score,
                            },
                        ),
                    )
                    await self._client.send(envelope)
                await asyncio.sleep(CAPACITY_REPORT_INTERVAL)
        except asyncio.CancelledError:
            pass

    async def _handle_task_dispatch(self, envelope: Envelope) -> None:
        """处理任务分派。"""
        payload = envelope.message.payload
        task_id = payload.get("task_id")

        if not task_id:
            logger.warning("收到缺少 task_id 的任务分派，已忽略")
            return

        # 回复 TASK_ACCEPT
        if self._client is not None:
            accept_envelope = Envelope(
                channel=Channel.TASK_DISPATCH,
                message=Message(
                    type=MessageType.TASK_ACCEPT,
                    sender_id=self.node_id,
                    payload={
                        "task_id": task_id,
                        "node_id": self.node_id,
                        "accepted": True,
                        "estimated_latency_ms": 0.0,
                    },
                ),
            )
            await self._client.send(accept_envelope)

        # 启动异步推理任务
        task = asyncio.create_task(self._run_inference(task_id, payload))
        self._tasks[task_id] = task
        task.add_done_callback(lambda t: self._tasks.pop(task_id, None))

    async def _handle_cancel_task(self, envelope: Envelope) -> None:
        """处理任务取消。"""
        payload = envelope.message.payload
        task_id = payload.get("task_id", "")
        task = self._tasks.pop(task_id, None)
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def _run_inference(self, task_id: str, payload: dict) -> None:
        """执行推理任务（占位实现）。"""
        try:
            # TODO: 对接 Runner 执行实际推理
            result_envelope = Envelope(
                channel=Channel.TASK_DISPATCH,
                message=Message(
                    type=MessageType.TASK_RESULT,
                    sender_id=self.node_id,
                    payload={
                        "task_id": task_id,
                        "node_id": self.node_id,
                        "status": "completed",
                        "output_text": "",
                        "tokens_generated": 0,
                        "tokens_per_second": 0.0,
                        "latency_ms": 0.0,
                    },
                ),
            )
            if self._client is not None:
                await self._client.send(result_envelope)
        except asyncio.CancelledError:
            if self._client is not None:
                cancel_result = Envelope(
                    channel=Channel.TASK_DISPATCH,
                    message=Message(
                        type=MessageType.TASK_RESULT,
                        sender_id=self.node_id,
                        payload={
                            "task_id": task_id,
                            "node_id": self.node_id,
                            "status": "cancelled",
                            "output_text": "",
                            "tokens_generated": 0,
                            "tokens_per_second": 0.0,
                            "latency_ms": 0.0,
                        },
                    ),
                )
                await self._client.send(cancel_result)

    # ------------------------------------------------------------------
    # 原有同步方法
    # ------------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        """健康检查。"""
        return {
            "status": "ok",
            "node_id": self.node_id,
        }

    def get_resources(self) -> NodeResources:
        """获取节点资源信息。

        注意：此方法包含同步阻塞调用（psutil.cpu_percent、benchmark），
        在异步上下文中应通过 asyncio.to_thread() 调用。
        """
        cpu_count = psutil.cpu_count(logical=True) or 0
        cpu_percent = psutil.cpu_percent(interval=0)
        mem = psutil.virtual_memory()

        cpu_info = self._hardware_detector.detect_cpu()
        gpus = self._hardware_detector.detect_gpus()
        disk = self._hardware_detector.detect_disk()
        network = self._hardware_detector.detect_network()

        try:
            report = self._benchmark_score.run_benchmark()
            compute_score = report.relative_score
        except Exception:
            logger.warning("基准测试运行失败，使用默认 compute_score=0.0", exc_info=True)
            compute_score = 0.0

        return NodeResources(
            cpu_count=cpu_count,
            cpu_percent=cpu_percent,
            memory_total_mb=int(mem.total // (1024 * 1024)),
            memory_free_mb=int(mem.available // (1024 * 1024)),
            gpus=gpus,
            compute_score=compute_score,
            cpu_physical_count=cpu_info.physical_count,
            cpu_freq_mhz=cpu_info.freq_mhz,
            cpu_brand=cpu_info.brand,
            disk_free_mb=disk.free_mb,
            network_mbps=network.estimated_mbps,
        )

    async def get_resources_async(self) -> NodeResources:
        """异步获取节点资源信息，不阻塞事件循环。"""
        return await asyncio.to_thread(self.get_resources)

    def start_rpc(self, port: int | None = None) -> dict[str, Any]:
        """启动 RPC Server。

        Args:
            port: 指定端口，None 则自动分配
        """
        try:
            assigned_port = self._rpc_server.start(port=port)
            return {"status": "ok", "port": assigned_port, "endpoint": self._rpc_server.endpoint}
        except (FileNotFoundError, RuntimeError) as e:
            return {"status": "error", "error": str(e)}

    def stop_rpc(self) -> dict[str, Any]:
        """停止 RPC Server。"""
        self._rpc_server.stop()
        return {"status": "ok"}

    def rpc_status(self) -> dict[str, Any]:
        """查询 RPC Server 状态。"""
        return self._rpc_server.status()




