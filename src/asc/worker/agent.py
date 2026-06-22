"""Asc Worker Agent。

Worker Agent 运行在 Worker 节点上，是集群的数据平面。负责执行 Master 分派的任务，
并持续上报自身状态，使 Master 能够做出合理的调度决策。

核心职责：
- 资源查询：检测 CPU/GPU/内存/磁盘/网络等硬件信息
- 基准测试：运行轻量级 benchmark 计算算力评分（compute_score）
- RPC Server 生命周期管理：启动/停止 llama.cpp RPC Server
- 推理执行：通过 LlamaServerEngine 执行 Master 分派的推理任务
- 事件循环：连接 Master、周期性心跳（5s）、容量上报（30s）、任务分派处理

事件循环流程：
    1. 通过 TCPClient 连接到 Master
    2. 发送 NODE_JOINED 注册消息（携带资源信息）
    3. 启动心跳循环：每 5 秒发送 HEARTBEAT
    4. 启动容量上报循环：每 30 秒发送 CAPACITY_REPORT
    5. 监听任务分派：收到 TASK_DISPATCH 后执行推理并返回结果

推理执行流程：
    1. 收到 TASK_DISPATCH（含 prompt、model_id 等）
    2. 查找或构建本地 LlamaServerEngine
    3. 通过 engine.submit_async() 执行推理
    4. 将推理结果通过 TASK_RESULT 返回 Master

容错机制：
- 连接断开时自动停止事件循环，等待外部重启
- 任务取消支持：收到 CANCEL_TASK 时取消对应的 asyncio.Task
- 优雅关闭：stop() 方法设置标志位，循环自然退出后清理资源
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import platform
import socket
from dataclasses import dataclass, field, replace
from typing import Any

import psutil

from asc.core.cluster_config import ClusterConfig, NodeResourcePolicy
from asc.network.frame import Frame, FrameType
from asc.network.protocol import Channel, Envelope, Message, MessageType
from asc.network.transport import TCPClient
from asc.worker.benchmark_score import BenchmarkScore
from asc.worker.gpu_info import GPUInfo
from asc.worker.hardware import HardwareDetector
from asc.worker.rpc_server import RpcServer

logger = logging.getLogger(__name__)

# 定时间隔
HEARTBEAT_INTERVAL = 5.0
CAPACITY_REPORT_INTERVAL = 10.0


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
    cpu_instruction_set_extensions: list[str] = field(default_factory=list)
    disk_free_mb: int = 0
    network_mbps: float = 0.0
    os_type: str = ""
    os_arch: str = ""

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
        cluster_config: ClusterConfig | None = None,
        models_dir: str | None = None,
        skip_benchmark: bool = False,
        auth_token: str | None = None,
    ) -> None:
        self.node_id = node_id
        self.port = port
        self._rpc_server = rpc_server or RpcServer()
        self._benchmark_score = benchmark_score or BenchmarkScore(node_id=node_id)
        self._hardware_detector = hardware_detector or HardwareDetector()
        self._cluster_config = cluster_config or ClusterConfig()
        self._models_dir = models_dir or "./models"
        self._skip_benchmark = skip_benchmark
        self._benchmark_failed = False
        self._auth_token = auth_token or os.getenv("ASC_AUTH_TOKEN")

        # 事件循环状态
        self._client: TCPClient | None = None
        self._running = False
        self._heartbeat_task: asyncio.Task | None = None
        self._capacity_task: asyncio.Task | None = None
        self._tasks: dict[str, asyncio.Task] = {}
        # 模型分片接收状态: model_id -> {"total_chunks": int, "chunks": dict[int, bytes]}
        self._model_receive_state: dict[str, dict[str, Any]] = {}
        # 推理引擎缓存: model_id -> LlamaServerEngine
        self._engines: dict[str, Any] = {}
        # 模型层分配信息: model_id -> {layer_start, layer_end, gpu_layers, ...}
        self._model_assignments: dict[str, dict[str, Any]] = {}
        # 模型最后访问时间: model_id -> timestamp
        self._model_last_access: dict[str, float] = {}
        # Auto-unload 定时任务
        self._auto_unload_task: asyncio.Task | None = None
        # 服务器设置（由 Master 下发）
        self._server_settings: dict[str, Any] = {
            "jit_loading": False,
            "auto_unload": False,
            "max_idle_ttl_minutes": 60,
            "only_keep_last_model": False,
        }

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
            on_frame=self._on_frame,
            auth_token=self._auth_token,
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
        self._auto_unload_task = asyncio.create_task(self._auto_unload_loop())

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

        if self._auto_unload_task is not None:
            self._auto_unload_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._auto_unload_task
            self._auto_unload_task = None

        for _task_id, task in list(self._tasks.items()):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks.clear()

        # 关闭所有推理引擎
        for model_id, engine in self._engines.items():
            try:
                engine.close()
            except Exception as e:
                logger.warning("关闭引擎 %s 失败: %s", model_id, e)
        self._engines.clear()

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
        elif msg_type == MessageType.MODEL_DISTRIBUTE:
            await self._handle_model_distribute(envelope)
        elif msg_type == MessageType.LOAD_MODEL:
            await self._handle_load_model(envelope)
        elif msg_type == MessageType.UNLOAD_MODEL:
            await self._handle_unload_model(envelope)
        elif msg_type == MessageType.SERVER_SETTINGS:
            await self._handle_server_settings(envelope)
        elif msg_type == MessageType.CONFIG_UPDATE:
            await self._handle_config_update(envelope)
        elif msg_type == MessageType.RPC_START:
            await self._handle_rpc_start(envelope)
        elif msg_type == MessageType.RPC_STOP:
            await self._handle_rpc_stop(envelope)
        elif msg_type == MessageType.RESOURCE_QUERY:
            await self._handle_resource_query(envelope)

    async def _on_frame(self, frame: Frame) -> None:
        """处理从 Master 收到的原始帧。"""
        if frame.frame_type == FrameType.MODEL_CHUNK:
            await self._handle_model_chunk_frame(frame)
        # 其他帧类型由 Envelope 回调处理

    async def _register_to_master(self) -> None:
        """向 Master 注册，携带主机名和详细硬件信息。"""
        if self._client is None:
            return

        logger.info("正在向 Master 注册，获取本地资源信息...")
        resources = await self.get_resources_async()
        hostname = socket.gethostname()
        envelope = Envelope(
            channel=Channel.DISCOVERY,
            message=Message(
                type=MessageType.NODE_JOINED,
                sender_id=self.node_id,
                payload={
                    "node_id": self.node_id,
                    "hostname": hostname,
                    "port": self.port,
                    "resources": {
                        "cpu_count": resources.cpu_count,
                        "cpu_brand": resources.cpu_brand,
                        "cpu_percent": resources.cpu_percent,
                        "cpu_physical_count": resources.cpu_physical_count,
                        "cpu_freq_mhz": resources.cpu_freq_mhz,
                        "cpu_instruction_set_extensions": resources.cpu_instruction_set_extensions,
                        "memory_total_mb": resources.memory_total_mb,
                        "memory_free_mb": resources.memory_free_mb,
                        "gpus": [
                            {
                                "index": g.index,
                                "name": g.name,
                                "vendor": g.vendor,
                                "vram_total_mb": g.vram_total_mb,
                                "vram_free_mb": g.vram_free_mb,
                                "compute_capability": g.compute_capability,
                                "detection_platform": g.detection_platform,
                            }
                            for g in resources.gpus
                        ],
                        "compute_score": resources.compute_score,
                        "disk_free_mb": resources.disk_free_mb,
                        "network_mbps": resources.network_mbps,
                        "os_type": resources.os_type,
                        "os_arch": resources.os_arch,
                    },
                },
            ),
        )
        sent = await self._client.send(envelope)
        if sent:
            logger.info("已向 Master 发送注册消息 (hostname=%s, compute_score=%.2f)", hostname, resources.compute_score)
        else:
            logger.error("向 Master 发送注册消息失败")

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
                    resources = await self.get_resources_async()
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

    async def _report_capacity_now(self) -> None:
        """立即上报一次资源容量（不等待心跳周期）。"""
        if self._client is None or not self._client.is_connected:
            return
        try:
            resources = await self.get_resources_async()
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
            logger.debug("主动上报资源容量完成")
        except Exception as e:
            logger.warning("主动上报资源容量失败: %s", e)

    async def _auto_unload_loop(self) -> None:
        """定时检查并卸载空闲模型。"""
        try:
            while self._running:
                await asyncio.sleep(60)  # 每分钟检查一次
                settings = getattr(self, "_server_settings", {})
                if not settings.get("auto_unload", False):
                    continue

                ttl_minutes = settings.get("max_idle_ttl_minutes", 60)
                only_last = settings.get("only_keep_last_model", False)
                now = asyncio.get_event_loop().time()

                # 获取按最后访问时间排序的模型列表
                loaded_models = []
                for mid, eng in list(self._engines.items()):
                    if eng.status().value in ("ready", "running"):
                        last_access = self._model_last_access.get(mid, 0)
                        loaded_models.append((mid, last_access))

                if not loaded_models:
                    continue

                # Only Keep Last: 卸载除最近使用的所有模型
                if only_last and len(loaded_models) > 1:
                    loaded_models.sort(key=lambda x: x[1], reverse=True)
                    for mid, _ in loaded_models[1:]:
                        logger.info("Auto-unload (only_keep_last): 卸载模型 %s", mid)
                        await self._unload_model(mid)

                # TTL 超时卸载
                ttl_seconds = ttl_minutes * 60
                for mid, last_access in loaded_models:
                    idle_time = now - last_access if last_access > 0 else 0
                    if last_access > 0 and idle_time > ttl_seconds:
                        logger.info(
                            "Auto-unload (idle %dmin > TTL %dmin): 卸载模型 %s",
                            int(idle_time // 60), ttl_minutes, mid,
                        )
                        await self._unload_model(mid)
        except asyncio.CancelledError:
            pass

    async def _unload_model(self, model_id: str) -> None:
        """卸载指定模型，释放资源。"""
        engine = self._engines.pop(model_id, None)
        if engine is not None:
            try:
                engine.close()
                logger.info("模型 %s 已卸载，资源已释放", model_id)
            except Exception as e:
                logger.warning("模型 %s 卸载时出错: %s", model_id, e)

        self._model_last_access.pop(model_id, None)

        # 通知 Master 资源变化
        await self._report_capacity_now()

    async def _handle_server_settings(self, envelope: Envelope) -> None:
        """处理 Master 下发的服务器设置。"""
        payload = envelope.message.payload
        self._server_settings = {
            "jit_loading": payload.get("jit_loading", False),
            "auto_unload": payload.get("auto_unload", False),
            "max_idle_ttl_minutes": payload.get("max_idle_ttl_minutes", 60),
            "only_keep_last_model": payload.get("only_keep_last_model", False),
        }
        logger.info("收到服务器设置更新: %s", self._server_settings)

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

    async def _handle_model_distribute(self, envelope: Envelope) -> None:
        """处理模型分发通知。

        Master 通知 Worker 模型文件已通过 Binary Frame 上传完成，
        Worker 确认模型文件存在并回复 ACK，同时保存层分配信息。
        """
        payload = envelope.message.payload
        model_id = payload.get("model_id", "")
        model_path = self._resolve_local_model_path(model_id, payload.get("model_path", ""))
        layer_start = payload.get("layer_start", 0)
        layer_end = payload.get("layer_end", 0)
        gpu_layers = payload.get("gpu_layers", -1)

        logger.info(
            "收到模型分发通知: %s, 路径: %s, 层: %d-%d, GPU层: %d",
            model_id, model_path, layer_start, layer_end, gpu_layers,
        )

        # 保存层分配信息（供推理时使用）
        self._model_assignments[model_id] = {
            "model_path": model_path,
            "layer_start": layer_start,
            "layer_end": layer_end,
            "gpu_layers": gpu_layers,
            "total_layers": payload.get("total_layers", 0),
        }

        # 确认模型文件存在
        from pathlib import Path
        path = Path(model_path)
        file_exists = path.exists()

        # 回复 ACK
        if self._client is not None:
            ack_envelope = Envelope(
                channel=Channel.TASK_DISPATCH,
                message=Message(
                    type=MessageType.MODEL_DISTRIBUTE_ACK,
                    sender_id=self.node_id,
                    payload={
                        "node_id": self.node_id,
                        "model_id": model_id,
                        "file_exists": file_exists,
                        "layer_start": layer_start,
                        "layer_end": layer_end,
                        "gpu_layers": gpu_layers,
                        "ready": file_exists,
                    },
                ),
            )
            await self._client.send(ack_envelope)

    def _resolve_local_model_path(self, model_id: str, model_path: str) -> str:
        """将 Master 发来的模型路径解析为 Worker 本地路径。

        Master 可能发送其本地绝对路径（如 /home/user/.asc/models/xxx.gguf），
        在跨平台集群中该路径在 Worker 上不存在。此方法按以下优先级查找：
        1. 原路径是否存在（同机部署场景）
        2. Worker 配置的 models_dir
        3. 系统默认的 ~/.asc/models/（与 Master 默认目录一致）
        """
        from pathlib import Path

        if model_path and Path(model_path).exists():
            return model_path

        search_dirs: list[Path] = []

        # 1. Worker 配置的 models_dir
        worker_models_dir = Path(self._models_dir).resolve()
        worker_models_dir.mkdir(parents=True, exist_ok=True)
        search_dirs.append(worker_models_dir)

        # 2. 系统默认 ~/.asc/models（与 Master 的 _get_models_dir 保持一致）
        default_models_dir = Path.home() / ".asc" / "models"
        default_models_dir.mkdir(parents=True, exist_ok=True)
        if default_models_dir.resolve() != worker_models_dir:
            search_dirs.append(default_models_dir)

        # 如果 model_path 包含文件名，优先用文件名精确匹配
        if model_path:
            file_name = Path(model_path).name
            for d in search_dirs:
                candidate = d / file_name
                if candidate.exists():
                    return str(candidate)

        # 按 model_id 模糊匹配 .gguf 文件
        for d in search_dirs:
            gguf_files = list(d.glob(f"{model_id}*.gguf"))
            if gguf_files:
                return str(gguf_files[0])

        # 都找不到则原样返回，让后续逻辑报出原始路径错误
        return model_path

    async def _handle_load_model(self, envelope: Envelope) -> None:
        """处理 LOAD_MODEL 消息：加载模型到内存。

        接收 Master 发来的完整模型参数，保存到 _model_assignments 并
        立即启动 llama-server 进程加载模型。
        """
        payload = envelope.message.payload
        model_id = payload.get("model_id", "")
        model_path = self._resolve_local_model_path(model_id, payload.get("model_path", ""))
        gpu_layers = payload.get("gpu_layers", -1)

        logger.info(
            "收到 LOAD_MODEL: %s, 路径: %s, GPU层: %d",
            model_id, model_path, gpu_layers,
        )

        # 资源预检：检查可用显存和内存
        from pathlib import Path
        model_file = Path(model_path)
        model_size_mb = 0
        if model_file.exists():
            model_size_mb = model_file.stat().st_size // (1024 * 1024)

        resources = await asyncio.to_thread(self.get_resources)
        free_vram_mb = sum(g.vram_free_mb for g in resources.gpus)
        free_memory_mb = resources.memory_free_mb

        # 应用资源限制策略
        policy = self._cluster_config.get_node_policy(self.node_id)
        if policy.vram_limit_mb is not None:
            effective_vram = policy.effective_vram_mb() or policy.vram_limit_mb
            free_vram_mb = min(free_vram_mb, effective_vram)
        if policy.memory_limit_mb is not None:
            effective_mem = policy.effective_memory_mb() or free_memory_mb
            free_memory_mb = min(free_memory_mb, effective_mem)

        # 计算已加载模型占用的资源
        loaded_vram_mb = 0
        loaded_memory_mb = 0
        for mid, eng in self._engines.items():
            if eng.status().value in ("ready", "running"):
                assignment = self._model_assignments.get(mid, {})
                est_size = assignment.get("_estimated_size_mb", 0)
                if est_size == 0:
                    # 粗略估算：模型文件大小
                    mp = assignment.get("model_path", "")
                    if mp and Path(mp).exists():
                        est_size = Path(mp).stat().st_size // (1024 * 1024)
                loaded_vram_mb += est_size
                loaded_memory_mb += est_size

        available_vram_mb = max(0, free_vram_mb - loaded_vram_mb)
        available_memory_mb = max(0, free_memory_mb - loaded_memory_mb)

        logger.info(
            "资源预检: 模型 %s (%dMB), 可用VRAM=%dMB, 可用RAM=%dMB, 已加载模型=%d个(占用VRAM~%dMB)",
            model_id, model_size_mb, available_vram_mb, available_memory_mb,
            len([e for e in self._engines.values() if e.status().value in ("ready", "running")]),
            loaded_vram_mb,
        )

        # 预检失败：显存或内存不足
        if model_size_mb > 0:
            if gpu_layers != 0 and available_vram_mb > 0 and model_size_mb > available_vram_mb:
                logger.warning(
                    "资源预检警告: 模型 %s (%dMB) 可能超出可用VRAM (%dMB), 仍尝试加载",
                    model_id, model_size_mb, available_vram_mb,
                )
            if available_memory_mb > 0 and model_size_mb > available_memory_mb:
                logger.warning(
                    "资源预检警告: 模型 %s (%dMB) 可能超出可用RAM (%dMB), 仍尝试加载",
                    model_id, model_size_mb, available_memory_mb,
                )

        # 资源估算与安全护栏检查
        from asc.worker.resource_estimator import estimate_model_resources
        estimate = estimate_model_resources(
            model_path=model_path,
            config=payload,
            available_vram_mb=available_vram_mb,
            available_ram_mb=available_memory_mb,
        )
        if not estimate.passes_guardrails:
            for warning in estimate.warnings:
                logger.warning("资源护栏: %s", warning)
            # 仍然允许加载（用户可能选择忽略警告），但记录警告

        # 保存所有参数到 _model_assignments
        self._model_assignments[model_id] = {
            "model_path": model_path,
            "gpu_layers": gpu_layers,
            "context_length": payload.get("context_length", 4096),
            "cpu_threads": payload.get("cpu_threads", 0),
            "batch_size": payload.get("batch_size", 512),
            "flash_attention": payload.get("flash_attention", True),
            "keep_in_memory": payload.get("keep_in_memory", True),
            "use_mmap": payload.get("use_mmap", True),
            "offload_kv_to_gpu": payload.get("offload_kv_to_gpu", True),
            "kv_quantization": payload.get("kv_quantization", ""),
            "rope_freq_base": payload.get("rope_freq_base", 0.0),
            "rope_freq_scale": payload.get("rope_freq_scale", 0.0),
            "seed": payload.get("seed", -1),
            "_estimated_size_mb": model_size_mb,
        }

        # 立即加载模型
        success = False
        error_msg = ""
        try:
            engine = await self._get_or_create_engine(model_id)
            if engine is not None:
                success = True
                logger.info("模型 %s 加载成功", model_id)
            else:
                error_msg = "引擎创建返回 None"
                logger.error("模型 %s 加载失败: %s", model_id, error_msg)
        except Exception as e:
            error_msg = str(e)
            logger.error("模型 %s 加载异常: %s", model_id, e)

        # 发送 LOAD_MODEL_ACK 通知 Master
        if self._client is not None and self._client.is_connected:
            ack_payload = {
                "node_id": self.node_id,
                "model_id": model_id,
                "success": success,
                "error": error_msg,
            }
            if success:
                # 附带加载后的资源快照
                post_res = self.get_resources()
                ack_payload["vram_free_mb"] = sum(g.vram_free_mb for g in post_res.gpus)
                ack_payload["memory_free_mb"] = post_res.memory_free_mb
                ack_payload["loaded_models"] = len(self._engines)

            ack_envelope = Envelope(
                channel=Channel.TASK_DISPATCH,
                message=Message(
                    type=MessageType.LOAD_MODEL_ACK,
                    sender_id=self.node_id,
                    payload=ack_payload,
                ),
            )
            try:
                await self._client.send(ack_envelope)
            except Exception as e:
                logger.warning("发送 LOAD_MODEL_ACK 失败: %s", e)

        # 主动上报资源变化（不等待心跳周期）
        if success:
            await self._report_capacity_now()

    async def _handle_unload_model(self, envelope: Envelope) -> None:
        """处理 Master 发来的卸载模型请求。"""
        payload = envelope.message.payload
        model_id = payload.get("model_id", "")
        logger.info("收到 UNLOAD_MODEL: %s", model_id)

        success = False
        error = ""

        if model_id in self._engines:
            try:
                engine = self._engines.pop(model_id)
                if hasattr(engine, "aclose"):
                    await engine.aclose()
                elif hasattr(engine, "close"):
                    engine.close()
                # 清理模型分配记录
                self._model_assignments.pop(model_id, None)
                success = True
                logger.info("模型 %s 已卸载", model_id)
            except Exception as e:
                error = str(e)
                logger.error("模型 %s 卸载失败: %s", model_id, e)
        else:
            error = f"Model {model_id} not loaded on this node"
            logger.warning("模型 %s 未在本节点加载，无法卸载", model_id)

        # 发送 UNLOAD_MODEL_ACK
        if self._client is not None and self._client.is_connected:
            resources = await asyncio.to_thread(self.get_resources)
            ack = Envelope(
                channel=Channel.TASK_DISPATCH,
                message=Message(
                    type=MessageType.UNLOAD_MODEL_ACK,
                    sender_id=self.node_id,
                    payload={
                        "node_id": self.node_id,
                        "model_id": model_id,
                        "success": success,
                        "error": error,
                        "vram_free_mb": resources.total_vram_free_mb,
                        "memory_free_mb": resources.memory_free_mb,
                        "loaded_models": len(self._engines),
                    },
                ),
            )
            try:
                await self._client.send(ack)
            except Exception as e:
                logger.warning("发送 UNLOAD_MODEL_ACK 失败: %s", e)

        if success:
            await self._report_capacity_now()

    async def _handle_config_update(self, envelope: Envelope) -> None:
        """处理配置更新。"""
        payload = envelope.message.payload
        logger.info("收到配置更新: %s", payload)


        vram_percent = payload.get("vram_limit_percent", 100)
        memory_percent = payload.get("memory_limit_percent", 100)
        offload_ratio = payload.get("offload_ratio", 0.0)

        resources = await asyncio.to_thread(self.get_resources)
        total_vram = sum(g.vram_total_mb for g in resources.gpus)
        total_memory = resources.memory_total_mb

        vram_limit = int(total_vram * vram_percent / 100) if vram_percent < 100 else None
        memory_limit = int(total_memory * memory_percent / 100) if memory_percent < 100 else None

        policy = NodeResourcePolicy(
            vram_limit_mb=vram_limit,
            memory_limit_mb=memory_limit,
            memory_offload_ratio=offload_ratio,
        )
        self._cluster_config.set_node_policy(self.node_id, policy)

    async def _handle_rpc_start(self, envelope: Envelope) -> None:
        """处理 RPC_START 命令：启动 RPC Server 并回复 RPC_START_ACK。"""
        payload = envelope.message.payload
        port = payload.get("port")
        request_id = payload.get("request_id")

        result = await asyncio.to_thread(self.start_rpc, port=port)

        if self._client is not None:
            ack_payload = {
                "status": result.get("status", "error"),
                "endpoint": result.get("endpoint"),
                "port": result.get("port"),
                "error": result.get("error"),
            }
            if request_id:
                ack_payload["request_id"] = request_id
            ack_envelope = Envelope(
                channel=Channel.COMMANDS,
                message=Message(
                    type=MessageType.RPC_START_ACK,
                    sender_id=self.node_id,
                    payload=ack_payload,
                ),
            )
            await self._client.send(ack_envelope)

    async def _handle_rpc_stop(self, envelope: Envelope) -> None:
        """处理 RPC_STOP 命令：停止 RPC Server 并回复 RPC_STOP_ACK。"""
        result = await asyncio.to_thread(self.stop_rpc)

        if self._client is not None:
            ack_envelope = Envelope(
                channel=Channel.COMMANDS,
                message=Message(
                    type=MessageType.RPC_STOP_ACK,
                    sender_id=self.node_id,
                    payload={
                        "status": result.get("status", "ok"),
                    },
                ),
            )
            await self._client.send(ack_envelope)

    async def _handle_resource_query(self, envelope: Envelope) -> None:
        """处理 RESOURCE_QUERY 命令：返回当前资源信息。"""
        resources = await self.get_resources_async()

        if self._client is not None:
            response_envelope = Envelope(
                channel=Channel.COMMANDS,
                message=Message(
                    type=MessageType.RESOURCE_RESPONSE,
                    sender_id=self.node_id,
                    payload={
                        "node_id": self.node_id,
                        "cpu_count": resources.cpu_count,
                        "cpu_brand": resources.cpu_brand,
                        "cpu_physical_count": resources.cpu_physical_count,
                        "cpu_freq_mhz": resources.cpu_freq_mhz,
                        "cpu_percent": resources.cpu_percent,
                        "memory_total_mb": resources.memory_total_mb,
                        "memory_free_mb": resources.memory_free_mb,
                        "compute_score": resources.compute_score,
                        "disk_free_mb": resources.disk_free_mb,
                        "network_mbps": resources.network_mbps,
                        "total_vram_free_mb": resources.total_vram_free_mb,
                        "gpus": [
                            {
                                "index": g.index,
                                "name": g.name,
                                "vendor": g.vendor,
                                "vram_total_mb": g.vram_total_mb,
                                "vram_free_mb": g.vram_free_mb,
                                "compute_capability": g.compute_capability,
                                "detection_platform": g.detection_platform,
                            }
                            for g in resources.gpus
                        ],
                    },
                ),
            )
            await self._client.send(response_envelope)

    async def _handle_model_chunk_frame(self, frame: Frame) -> None:
        """处理 MODEL_CHUNK 帧。

        解析分片负载，写入文件，并回复 MODEL_CHUNK_ACK。
        """
        from asc.master.model_distributor import (
            decode_model_chunk_payload,
            encode_model_chunk_ack_payload,
        )

        model_id, chunk_index, total_chunks, chunk_data = decode_model_chunk_payload(
            frame.payload
        )

        logger.debug(
            "收到模型分片: %s, chunk %d/%d, 数据大小: %d",
            model_id, chunk_index, total_chunks, len(chunk_data),
        )

        # 初始化或获取接收状态
        if model_id not in self._model_receive_state:
            self._model_receive_state[model_id] = {
                "total_chunks": total_chunks,
                "chunks": {},
            }
        state = self._model_receive_state[model_id]
        state["chunks"][chunk_index] = chunk_data

        # 写入分片到文件
        success = True
        try:
            from pathlib import Path
            models_dir = Path(self._models_dir).resolve()
            models_dir.mkdir(parents=True, exist_ok=True)

            # 验证 model_id 防止路径遍历
            safe_model_id = model_id.replace("\\", "/").split("/")[-1].replace("..", "")
            file_path = (models_dir / f"{safe_model_id}.gguf").resolve()

            # 确保最终路径在 models_dir 内
            if not str(file_path).startswith(str(models_dir)):
                logger.error("路径遍历攻击检测: model_id=%s -> %s", model_id, file_path)
                return

            # 预分配文件空间（首个分片时）
            if chunk_index == 0 and not file_path.exists():
                # 估算文件总大小（从 total_chunks 和当前 chunk 大小推算）
                file_path.touch()

            # 将分片写入正确偏移位置
            from asc.master.model_distributor import CHUNK_SIZE
            offset = chunk_index * CHUNK_SIZE
            with open(file_path, "r+b" if file_path.exists() else "wb") as f:
                f.seek(offset)
                f.write(chunk_data)

        except Exception as e:
            logger.error("写入模型分片失败: %s, chunk %d: %s", model_id, chunk_index, e)
            success = False

        # 检查是否所有分片都已接收
        if success and len(state["chunks"]) == state["total_chunks"]:
            logger.info("模型 %s 所有分片接收完成 (%d 个分片)", model_id, total_chunks)
            # 清理接收状态
            self._model_receive_state.pop(model_id, None)

        # 回复 MODEL_CHUNK_ACK
        if self._client is not None and self._client.is_connected:
            ack_payload = encode_model_chunk_ack_payload(
                model_id=model_id,
                chunk_index=chunk_index,
                success=success,
            )
            ack_frame = Frame(frame_type=FrameType.MODEL_CHUNK_ACK, payload=ack_payload)
            try:
                await self._client.send_frame(ack_frame)
            except Exception as e:
                logger.warning("发送 MODEL_CHUNK_ACK 失败: %s", e)
        else:
            logger.warning("无法发送 MODEL_CHUNK_ACK: 客户端未连接")

    async def _run_inference(self, task_id: str, payload: dict) -> None:
        """执行推理任务。

        通过本地 LlamaServerEngine 执行推理，支持模型自动加载和引擎缓存。

        流程：
        1. 从 payload 中提取 model_id 和 prompt
        2. 查找或构建对应的 LlamaServerEngine
        3. 执行推理并计算性能指标
        4. 将结果通过 TASK_RESULT 返回 Master
        """
        import time

        from asc.engine.base import InferenceRequest
        from asc.engine.llama_server import LlamaServerBuilder

        model_id = payload.get("model_id", "")
        prompt = payload.get("prompt", "")
        max_tokens = payload.get("max_tokens", 128)
        temperature = payload.get("temperature", 0.7)

        # 更新模型最后访问时间（用于 auto-unload）
        self._model_last_access[model_id] = asyncio.get_running_loop().time()

        start_time = time.monotonic()

        try:
            # 获取或构建推理引擎
            engine = await self._get_or_create_engine(model_id)
            if engine is None:
                raise RuntimeError(f"无法为模型 {model_id} 创建推理引擎")

            # 执行推理
            request = InferenceRequest(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            output_text = await engine.submit_async(request)

            elapsed = time.monotonic() - start_time
            # 估算 token 数（粗略：按字符数/4 估算，实际应从引擎获取）
            tokens_generated = max(1, len(output_text) // 4)
            tokens_per_second = tokens_generated / elapsed if elapsed > 0 else 0.0

            result_envelope = Envelope(
                channel=Channel.TASK_DISPATCH,
                message=Message(
                    type=MessageType.TASK_RESULT,
                    sender_id=self.node_id,
                    payload={
                        "task_id": task_id,
                        "node_id": self.node_id,
                        "status": "completed",
                        "output": output_text,
                        "output_text": output_text,
                        "tokens_generated": tokens_generated,
                        "tokens_per_second": round(tokens_per_second, 2),
                        "latency_ms": round(elapsed * 1000, 2),
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

        except Exception as e:
            logger.error("推理任务 %s 执行失败: %s", task_id, e)
            elapsed = time.monotonic() - start_time
            if self._client is not None:
                error_result = Envelope(
                    channel=Channel.TASK_DISPATCH,
                    message=Message(
                        type=MessageType.TASK_RESULT,
                        sender_id=self.node_id,
                        payload={
                            "task_id": task_id,
                            "node_id": self.node_id,
                            "status": "failed",
                            "error": str(e),
                            "output_text": "",
                            "tokens_generated": 0,
                            "tokens_per_second": 0.0,
                            "latency_ms": round(elapsed * 1000, 2),
                        },
                    ),
                )
                await self._client.send(error_result)

    async def _get_or_create_engine(self, model_id: str) -> Any:
        """获取或创建推理引擎。

        引擎缓存策略：同一 model_id 复用同一引擎实例，避免重复加载模型。
        引擎使用 llama-server 常驻进程，首次加载后推理延迟为毫秒级。

        Args:
            model_id: 模型 ID

        Returns:
            LlamaServerEngine 实例，或 None（模型未就绪）
        """
        # 缓存命中
        if model_id in self._engines:
            engine = self._engines[model_id]
            if engine.status().value in ("ready", "running"):
                return engine
            # 引擎异常，移除缓存并重建
            try:
                engine.close()
            except (OSError, RuntimeError) as e:
                logger.debug("关闭异常引擎 %s 时出错: %s", model_id, e)
            del self._engines[model_id]

        # 查找模型路径和参数
        assignment = self._model_assignments.get(model_id)
        if assignment is not None:
            raw_path = assignment.get("model_path", "")
            model_path = self._resolve_local_model_path(model_id, raw_path)
            # 如果解析出了新路径，更新 assignment 供后续使用
            if model_path != raw_path:
                assignment = dict(assignment)
                assignment["model_path"] = model_path
                self._model_assignments[model_id] = assignment
            gpu_layers = assignment.get("gpu_layers", -1)
        else:
            # 尝试从 models_dir 查找
            from pathlib import Path
            models_dir = Path(self._models_dir)
            # 查找 .gguf 文件
            gguf_files = list(models_dir.glob(f"{model_id}*.gguf"))
            if not gguf_files:
                logger.error("未找到模型文件: %s", model_id)
                return None
            model_path = str(gguf_files[0])
            gpu_layers = -1
            assignment = {}

        if not model_path:
            logger.error("模型 %s 路径为空", model_id)
            return None

        from pathlib import Path
        if not Path(model_path).exists():
            logger.error("模型文件不存在: %s", model_path)
            return None

        # 构建 LlamaServerEngine（传递所有参数）
        from asc.engine.llama_server import LlamaServerBuilder

        builder = LlamaServerBuilder(
            model_path=model_path,
            n_gpu_layers=gpu_layers,
            context_length=assignment.get("context_length", 4096),
            cpu_threads=assignment.get("cpu_threads", 0),
            batch_size=assignment.get("batch_size", 512),
            flash_attention=assignment.get("flash_attention", True),
            use_mmap=assignment.get("use_mmap", True),
            seed=assignment.get("seed", -1),
            rope_freq_base=assignment.get("rope_freq_base", 0.0),
            rope_freq_scale=assignment.get("rope_freq_scale", 0.0),
            offload_kv_to_gpu=assignment.get("offload_kv_to_gpu", True),
            kv_quantization=assignment.get("kv_quantization", ""),
        )

        try:
            # 异步加载模型
            async for _progress in builder.aload():
                logger.debug(
                    "模型 %s 加载进度: %s", model_id, _progress.message
                )

            engine = await builder.abuild()
            self._engines[model_id] = engine

            # 加载后资源监控日志
            post_resources = await asyncio.to_thread(self.get_resources)
            post_free_vram = sum(g.vram_free_mb for g in post_resources.gpus)
            post_free_mem = post_resources.memory_free_mb
            logger.info(
                "模型 %s 引擎已就绪 | 加载后: VRAM可用=%dMB, RAM可用=%dMB | 已加载模型=%d个",
                model_id, post_free_vram, post_free_mem, len(self._engines),
            )
            return engine

        except FileNotFoundError as e:
            logger.error("llama-server 可执行文件未找到: %s", e)
            return None
        except TimeoutError as e:
            logger.error("模型 %s 加载超时: %s", model_id, e)
            return None
        except Exception as e:
            logger.error("模型 %s 引擎构建失败: %s", model_id, e)
            return None

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
        """获取节点资源信息，应用管理员配置的资源限制。

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
            if self._skip_benchmark or self._benchmark_failed:
                if self._benchmark_failed:
                    logger.debug("基准测试已跳过（之前失败）")
                compute_score = 0.0
            else:
                report = self._benchmark_score.run_benchmark()
                compute_score = report.relative_score
        except FileNotFoundError as e:
            logger.warning("基准模型文件缺失，跳过基准测试: %s", e)
            self._benchmark_failed = True
            compute_score = 0.0
        except Exception as e:
            logger.warning("基准测试运行失败，使用默认 compute_score=0.0: %s", e)
            self._benchmark_failed = True
            compute_score = 0.0

        # 应用管理员配置的资源限制
        policy = self._cluster_config.get_node_policy(self.node_id)
        memory_total_mb = int(mem.total // (1024 * 1024))
        memory_free_mb = int(mem.available // (1024 * 1024))
        disk_free_mb = disk.free_mb

        # 内存限制
        if policy.memory_limit_mb is not None:
            memory_total_mb = min(memory_total_mb, policy.memory_limit_mb)
            memory_free_mb = min(memory_free_mb, int(policy.effective_memory_mb() or memory_free_mb))

        # 显存限制：对每个 GPU 的 vram_free 应用限制
        if policy.vram_limit_mb is not None:
            effective_vram = policy.effective_vram_mb() or policy.vram_limit_mb
            gpus = [
                replace(g, vram_free_mb=min(g.vram_free_mb, effective_vram))
                for g in gpus
            ]

        # 磁盘限制
        if policy.disk_limit_mb is not None:
            effective_disk = policy.effective_disk_mb() or disk_free_mb
            disk_free_mb = min(disk_free_mb, effective_disk)

        return NodeResources(
            cpu_count=cpu_count,
            cpu_percent=cpu_percent,
            memory_total_mb=memory_total_mb,
            memory_free_mb=memory_free_mb,
            gpus=gpus,
            compute_score=compute_score,
            cpu_physical_count=cpu_info.physical_count,
            cpu_freq_mhz=cpu_info.freq_mhz,
            cpu_brand=cpu_info.brand,
            cpu_instruction_set_extensions=cpu_info.instruction_set_extensions,
            disk_free_mb=disk_free_mb,
            network_mbps=network.estimated_mbps,
            os_type=platform.system(),
            os_arch=platform.machine(),
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




