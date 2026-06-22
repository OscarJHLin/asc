"""Asc Master 节点主循环。

Master 是集群的控制平面，负责维护全局状态、调度任务和管理节点生命周期。
采用"事件驱动状态机"架构：所有状态变更通过 Command -> Event -> State 流程完成。

核心职责：
- 处理 Command（来自 API 或 Worker），产生对应的 Event
- 维护 ClusterState（不可变全局状态）
- 调度待处理任务到可用 Worker 节点
- 写入事件日志（EventLog），支持状态重放和审计
- 运行事件循环：TCP 服务、心跳检测（10s）、任务调度（5s）

事件驱动流程：
    Command（意图） -> MasterNode 处理 -> Event（事实） -> apply() -> ClusterState
    例如：StartInference -> TaskCreated -> state.tasks[task_id] = TaskInfo(...)

容错机制：
- FailureDetector：基于心跳超时检测节点故障
- FailoverManager：故障时自动注销节点并清理任务
- 负载均衡器：动态选择最优 Worker 执行任务

扩展性：
    新增消息类型只需在 _MESSAGE_HANDLERS 字典中注册处理函数，
    无需修改事件循环核心逻辑。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os

from asc.api.server import create_app
from asc.core.election import BullyElection, ElectionMessage, ElectionMessageType, election_to_envelope, envelope_to_election
from asc.core.event_log import EventLog, MemoryEventLog, SnapshotEventLog
from asc.core.failover import FailoverManager, FailureDetector
from asc.master.model_distributor import ModelDistributor
from asc.master.orchestrator import DistributedOrchestrator
from asc.network.frame import Frame, FrameType
from asc.network.protocol import Channel, Envelope, Message, MessageType
from asc.network.transport import TCPServer
from asc.scheduler.load_balancer import LoadBalancer
from asc.types import (
    ClusterState,
    IndexedEvent,
    InstanceCreated,
    InstanceDeleted,
    NodeId,
    NodeJoined,
    NodeLeft,
    RunnerStatusUpdated,
    TaskCancelled,
    TaskCompleted,
    TaskCreated,
    TaskFailed,
    TaskId,
    apply,
    empty_state,
    generate_instance_id,
    generate_task_id,
)
from asc.types.commands import (
    CancelTask,
    CreateInstance,
    DeleteInstance,
    ShutdownRunner,
    StartInference,
)
from asc.types.events import Event

logger = logging.getLogger(__name__)


class MasterNode:
    """Master 节点。

    处理 Command -> 产生 Event -> 更新 State -> 写入 EventLog。
    """

    def __init__(
        self,
        node_id: str,
        event_log: EventLog | None = None,
        data_dir: str | None = None,
        orchestrator: DistributedOrchestrator | None = None,
        failure_detector: FailureDetector | None = None,
        failover_manager: FailoverManager | None = None,
        load_balancer: LoadBalancer | None = None,
        tcp_server: TCPServer | None = None,
        ssl_context: "ssl.SSLContext | None" = None,
        api_host: str = "0.0.0.0",
        api_port: int | None = None,
        auth_token: str | None = None,
    ) -> None:
        self.node_id = node_id
        self._state = empty_state()
        self._next_index = 1
        # 事件日志优先级：显式传入 > data_dir 磁盘持久化 > 内存日志
        if event_log is not None:
            self.event_log = event_log
        elif data_dir is not None:
            from pathlib import Path
            log_path = Path(data_dir) / "events.jsonl"
            self.event_log = SnapshotEventLog(log_path, snapshot_interval=100)
        else:
            self.event_log = MemoryEventLog()
        self._orchestrator = orchestrator if orchestrator is not None else DistributedOrchestrator()
        self._failure_detector = (
            failure_detector if failure_detector is not None else FailureDetector()
        )
        self._failover_manager = (
            failover_manager
            if failover_manager is not None
            else FailoverManager(
                detector=self._failure_detector,
                on_node_failed=self._on_node_failed,
            )
        )
        self._load_balancer = load_balancer if load_balancer is not None else LoadBalancer()
        self._tcp_server = tcp_server
        self._ssl_context = ssl_context
        self._api_host = api_host
        self._api_port = api_port
        self._auth_token = auth_token or os.getenv("ASC_AUTH_TOKEN")
        self._api_server_task: asyncio.Task | None = None
        self._tasks: list[asyncio.Task] = []
        self._running = False
        # conn_id -> node_id 映射
        self._conn_node_map: dict[str, str] = {}
        # task_id -> node_id 映射（记录任务被分配到了哪个节点）
        self._task_node_map: dict[str, str] = {}
        # FastAPI app 引用，用于状态更新同步
        self._app: Any | None = None
        # node_id -> 硬件资源信息（注册时携带的详细硬件数据）
        self._node_resources: dict[str, dict] = {}
        # 模型分发器（延迟初始化，需要 TCPServer）
        self._model_distributor: Any = None
        # 选举模块（多 Master 场景下决定谁是真正的 Master）
        self._election: BullyElection | None = None
        self._election_task: asyncio.Task | None = None

    @property
    def state(self) -> ClusterState:
        return self._state

    def _emit(self, event: Event) -> list[Event]:
        """发出事件，更新状态，写入日志。

        这是 Master 节点状态变更的核心方法，所有状态修改必须经过此函数。
        通过 apply() 纯函数确保不可变性，通过 event_log 确保可审计性。

        Args:
            event: 要发出的事件

        Returns:
            包含该事件的列表（便于链式调用和统一返回类型）

        线程安全：
            本方法修改可变状态（_state、_next_index），当前在 asyncio 事件循环
            中单线程调用。若未来改为多线程，需加锁保护。
        """
        indexed = IndexedEvent(event=event, index=self._next_index)
        self._state = apply(self._state, indexed)
        self._next_index += 1
        self.event_log.append(indexed)
        # 同步更新 FastAPI app 的状态引用（ClusterState 不可变，需同步新对象）
        if self._app is not None:
            self._app.state.cluster_state = self._state
            self._app.state.conn_node_map = self._conn_node_map
            self._app.state.node_resources = self._node_resources
            # 通知 WebSocket 客户端状态变更
            try:
                import asyncio
                loop = asyncio.get_running_loop()
                task = loop.create_task(self._notify_ws_clients())
                task.add_done_callback(self._on_ws_notify_done)
            except RuntimeError:
                pass
        return [event]

    async def _notify_ws_clients(self) -> None:
        """通知所有 WebSocket 客户端集群状态变更。"""
        if self._app is None:
            return
        ws_clients = getattr(self._app.state, "ws_clients", None)
        if not ws_clients:
            return

        from asc.api.server import _send_cluster_state
        disconnected = set()
        for ws in list(ws_clients):
            try:
                await _send_cluster_state(ws, self._app)
            except Exception:
                disconnected.add(ws)
        for ws in disconnected:
            ws_clients.discard(ws)

    @staticmethod
    def _on_ws_notify_done(task: "asyncio.Task[None]") -> None:
        """create_task 回调：记录未捕获异常，避免静默丢失。"""
        exc = task.exception()
        if exc is not None:
            logger.warning("WebSocket 通知任务异常: %s", exc)

    # ------------------------------------------------------------------
    # 事件循环
    # ------------------------------------------------------------------

    async def run(self, host: str = "0.0.0.0", port: int = 52414) -> None:
        """Master 主事件循环。

        1. 启动 TCP Server
        2. 启动 API Server（如果配置了 api_port）
        3. 启动心跳检测定时任务
        4. 启动调度器定时任务
        5. 启动选举超时检测（如果存在多 Master 场景）
        """
        self._running = True

        # 初始化 TCP Server（如果未注入）
        if self._tcp_server is None:
            self._tcp_server = TCPServer(
                host=host,
                port=port,
                on_message=self._on_message,
                on_frame=self._on_frame,
                ssl_context=self._ssl_context,
                auth_token=self._auth_token,
            )
        else:
            # 注入的 TCPServer 也需要设置 on_frame 回调
            if self._tcp_server._on_frame is None:
                self._tcp_server._on_frame = self._on_frame

        # 启动 TCP Server
        await self._tcp_server.start()
        logger.info("Master %s started on %s:%s", self.node_id, host, self._tcp_server.port)

        # 将 TCPServer 和连接映射传递给编排器
        self._orchestrator.set_tcp_server(self._tcp_server)

        # 提前初始化 ModelDistributor，确保 API 端点后台任务能共享同一实例
        if self._model_distributor is None:
            self._model_distributor = ModelDistributor(tcp_server=self._tcp_server)
        else:
            self._model_distributor.set_tcp_server(self._tcp_server)

        # 启动 API Server（与 Master 共享状态）
        if self._api_port is not None:
            self._api_server_task = asyncio.create_task(self._run_api_server())

        # 启动后台定时任务
        health_task = asyncio.create_task(self._health_check_loop())
        schedule_task = asyncio.create_task(self._schedule_loop())
        self._tasks = [health_task, schedule_task]

        # 启动选举超时检测（如果选举模块已初始化）
        if self._election is not None:
            self._election_task = asyncio.create_task(self._election_timeout_loop())
            self._tasks.append(self._election_task)

        # 等待所有任务（正常情况下不会结束）
        all_tasks = self._tasks[:]
        if self._api_server_task is not None:
            all_tasks.append(self._api_server_task)
        await asyncio.gather(*all_tasks)

    async def _run_api_server(self) -> None:
        """启动与 Master 集成的 API Server。"""
        import uvicorn

        from asc.core.cluster_config import ClusterConfig

        app = create_app(
            cluster_state=self._state,
            cluster_config=ClusterConfig(),
            tcp_server=self._tcp_server,
            conn_node_map=self._conn_node_map,
            model_distributor=self._model_distributor,
        )
        self._app = app
        logger.info(
            "API Server starting on %s:%s", self._api_host, self._api_port
        )
        config = uvicorn.Config(
            app,
            host=self._api_host,
            port=self._api_port,
            log_level="info",
        )
        server = uvicorn.Server(config)
        await server.serve()

    async def stop(self) -> None:
        """优雅停止。"""
        self._running = False

        # 取消后台任务
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks.clear()

        # 停止 TCP Server
        if self._tcp_server is not None:
            await self._tcp_server.stop()

        logger.info("Master %s stopped", self.node_id)

    async def _on_message(self, conn_id: str, envelope: Envelope) -> None:
        """处理收到的消息。"""
        msg_type = envelope.message.type

        # 优先处理选举消息（Channel.ELECTION）
        if envelope.channel == Channel.ELECTION:
            await self._handle_election_message(envelope)
            return

        handler = _MESSAGE_HANDLERS.get(msg_type)
        if handler is not None:
            # NODE_JOINED / NODE_LEFT 需要 conn_id 来维护连接映射
            if msg_type in (MessageType.NODE_JOINED, MessageType.NODE_LEFT):
                await handler(self, envelope, conn_id=conn_id)
            else:
                await handler(self, envelope)
        elif msg_type == MessageType.RPC_START_ACK:
            await self._orchestrator.handle_rpc_start_ack(envelope)
        elif msg_type == MessageType.RPC_STOP_ACK:
            await self._orchestrator.handle_rpc_stop_ack(envelope)
        elif msg_type == MessageType.RESOURCE_RESPONSE:
            await self._orchestrator.handle_resource_response(envelope)
        else:
            logger.debug("Unhandled message type: %s", msg_type)

    async def _on_frame(self, conn_id: str, frame: Frame) -> None:
        """处理收到的原始帧。

        处理 MODEL_CHUNK_ACK 等非 Envelope 帧。
        """
        if frame.frame_type == FrameType.MODEL_CHUNK_ACK:
            if self._model_distributor is not None:
                self._model_distributor.handle_chunk_ack(frame)

    async def _handle_node_joined(self, envelope: Envelope, conn_id: str) -> None:
        """处理节点注册，记录主机名和详细硬件信息。"""
        payload = envelope.message.payload
        node_id = payload.get("node_id", "")
        ip = payload.get("ip", "")
        port = payload.get("port", 0)
        hostname = payload.get("hostname", "")

        # 更新连接映射：conn_id (TCP UUID) -> node_id
        self._conn_node_map[conn_id] = node_id

        # 注册到集群状态
        self.process_node_joined(NodeId(node_id), ip, port)

        # 注册到故障检测器
        self._failure_detector.register(node_id)

        # 注册到故障转移管理器
        self._failover_manager.register(node_id)

        # 注册到负载均衡器
        self._load_balancer.register_node(node_id)

        # 保存节点硬件信息到状态（用于模型分发和层分配）
        resources = payload.get("resources", {})
        if resources:
            self._node_resources[node_id] = resources

        # 更新编排器的连接映射
        self._orchestrator.update_conn_map(self._conn_node_map)

        # 更新选举模块的节点列表
        if self._election is not None:
            all_ids = list(self._conn_node_map.values()) + [self.node_id]
            self._election.update_node_ids(all_ids)

        logger.info(
            "Node joined: %s (hostname=%s, %s:%s)",
            node_id, hostname, ip, port,
        )

    async def _handle_heartbeat(self, envelope: Envelope) -> None:
        """处理心跳。"""
        node_id = envelope.message.payload.get("node_id", "")
        if not node_id:
            return

        # 更新故障检测器
        self._failure_detector.heartbeat(node_id)

        # 更新故障转移管理器
        self._failover_manager.heartbeat(node_id)

        # 收到心跳说明节点在线，若 runner_status 不是 online 则更新
        nid = NodeId(node_id)
        if nid in self._state.nodes and self._state.nodes[nid].runner_status != "online":
            self._emit(RunnerStatusUpdated(node_id=nid, status="online"))

    async def _handle_capacity_report(self, envelope: Envelope) -> None:
        """处理容量上报。"""
        payload = envelope.message.payload
        node_id = payload.get("node_id", "")
        compute_score = payload.get("compute_score", 1.0)

        if node_id:
            self._load_balancer.register_node(node_id, compute_score)
            # 更新 node_resources 中的资源快照
            if node_id not in self._node_resources:
                self._node_resources[node_id] = {}
            res = self._node_resources[node_id]
            if payload.get("memory_free_mb") is not None:
                res["memory_free_mb"] = payload["memory_free_mb"]
            if payload.get("cpu_percent") is not None:
                res["cpu_percent"] = payload["cpu_percent"]
            if payload.get("cpu_count") is not None:
                res["cpu_count"] = payload["cpu_count"]
            if payload.get("compute_score") is not None:
                res["compute_score"] = payload["compute_score"]
            if payload.get("total_vram_free_mb") is not None and res.get("gpus"):
                total_vram_free = payload["total_vram_free_mb"]
                gpus = res["gpus"]
                if len(gpus) > 0:
                    total_vram = sum(g.get("vram_total_mb", 0) for g in gpus)
                    if total_vram > 0:
                        for g in gpus:
                            ratio = g.get("vram_total_mb", 0) / total_vram
                            g["vram_free_mb"] = int(total_vram_free * ratio)
            self._node_resources[node_id] = res
            # 推送 WebSocket 更新
            if self._app is not None:
                await self._notify_ws_clients()

    async def _handle_load_model_ack(self, envelope: Envelope) -> None:
        """处理 Worker 的模型加载结果通知。"""
        payload = envelope.message.payload
        node_id = payload.get("node_id", "")
        model_id = payload.get("model_id", "")
        success = payload.get("success", False)
        error = payload.get("error", "")

        if success:
            logger.info(
                "节点 %s 模型 %s 加载成功 (VRAM可用=%sMB, RAM可用=%sMB, 已加载=%s个)",
                node_id, model_id,
                payload.get("vram_free_mb", "?"),
                payload.get("memory_free_mb", "?"),
                payload.get("loaded_models", "?"),
            )
            # 更新 node_resources 中的资源快照
            if node_id not in self._node_resources:
                self._node_resources[node_id] = {}
            res = self._node_resources[node_id]
            if payload.get("memory_free_mb") is not None:
                res["memory_free_mb"] = payload.get("memory_free_mb", res.get("memory_free_mb", 0))
            # 更新 GPU 显存信息
            if payload.get("vram_free_mb") is not None and res.get("gpus"):
                total_vram_free = payload["vram_free_mb"]
                # 按比例分配到各 GPU
                gpus = res["gpus"]
                if len(gpus) > 0:
                    total_vram = sum(g.get("vram_total_mb", 0) for g in gpus)
                    if total_vram > 0:
                        for g in gpus:
                            ratio = g.get("vram_total_mb", 0) / total_vram
                            g["vram_free_mb"] = int(total_vram_free * ratio)
            # 追踪节点已加载的模型 ID 列表
            loaded = res.get("loaded_model_ids", [])
            if model_id not in loaded:
                loaded.append(model_id)
            res["loaded_model_ids"] = loaded
            self._node_resources[node_id] = res
        else:
            logger.error(
                "节点 %s 模型 %s 加载失败: %s",
                node_id, model_id, error,
            )

        # 通过 WebSocket 推送加载进度
        if self._app is not None:
            from asc.api.server import _notify_ws_clients
            try:
                await _notify_ws_clients(self._app, {
                    "type": "model_load_progress",
                    "data": {
                        "model_id": model_id,
                        "status": "completed" if success else "failed",
                        "node_id": node_id,
                        "node_status": "success" if success else "failed",
                        "error": error if not success else "",
                    },
                })
            except Exception:
                pass
            # 发送完整集群状态更新
            await self._notify_ws_clients()

    async def _handle_unload_model_ack(self, envelope: Envelope) -> None:
        """处理 Worker 的模型卸载结果通知。"""
        payload = envelope.message.payload
        node_id = payload.get("node_id", "")
        model_id = payload.get("model_id", "")
        success = payload.get("success", False)
        error = payload.get("error", "")

        if success:
            logger.info("节点 %s 模型 %s 卸载成功", node_id, model_id)
            # 更新 node_resources
            if node_id in self._node_resources:
                res = self._node_resources[node_id]
                loaded = res.get("loaded_model_ids", [])
                if model_id in loaded:
                    loaded.remove(model_id)
                res["loaded_model_ids"] = loaded
                if payload.get("memory_free_mb") is not None:
                    res["memory_free_mb"] = payload["memory_free_mb"]
                if payload.get("vram_free_mb") is not None and res.get("gpus"):
                    total_vram_free = payload["vram_free_mb"]
                    gpus = res["gpus"]
                    if len(gpus) > 0:
                        total_vram = sum(g.get("vram_total_mb", 0) for g in gpus)
                        if total_vram > 0:
                            for g in gpus:
                                ratio = g.get("vram_total_mb", 0) / total_vram
                                g["vram_free_mb"] = int(total_vram_free * ratio)
                self._node_resources[node_id] = res
        else:
            logger.error("节点 %s 模型 %s 卸载失败: %s", node_id, model_id, error)

        # 推送 WebSocket 更新
        if self._app is not None:
            from asc.api.server import _notify_ws_clients
            try:
                await _notify_ws_clients(self._app, {
                    "type": "model_unload_progress",
                    "data": {
                        "model_id": model_id,
                        "status": "completed" if success else "failed",
                        "node_id": node_id,
                        "error": error if not success else "",
                    },
                })
            except Exception:
                pass
            await self._notify_ws_clients()

    async def _handle_task_result(self, envelope: Envelope) -> None:
        """处理任务结果。"""
        payload = envelope.message.payload
        task_id = payload.get("task_id", "")
        status = payload.get("status", "")
        output = payload.get("output", "")
        error = payload.get("error", "")

        if not task_id:
            return

        if status == "completed":
            self._emit(TaskCompleted(task_id=TaskId(task_id), output=output))
        elif status == "failed":
            self._emit(TaskFailed(task_id=TaskId(task_id), error=error))

        # 如果 API 端点在等待此任务结果，通知它
        if self._app is not None and hasattr(self._app.state, "pending_results"):
            future = self._app.state.pending_results.pop(task_id, None)
            if future is not None and not future.done():
                future.set_result(payload)

    async def _handle_node_left(self, envelope: Envelope, conn_id: str) -> None:
        """处理节点离开消息。"""
        payload = envelope.message.payload
        node_id = payload.get("node_id", "")
        if not node_id:
            return

        # 清理该节点上的运行中任务（通过 _task_node_map 查找）
        failed_task_ids = []
        for task_id, assigned_node in list(self._task_node_map.items()):
            if assigned_node == node_id:
                failed_task_ids.append(task_id)
        for task_id in failed_task_ids:
            self._task_node_map.pop(task_id, None)
            # 如果 API 端点在等待此任务结果，通知它失败
            if self._app is not None and hasattr(self._app.state, "pending_results"):
                future = self._app.state.pending_results.pop(task_id, None)
                if future is not None and not future.done():
                    future.set_exception(RuntimeError(f"Node {node_id} left, task {task_id} failed"))
        if failed_task_ids:
            logger.info("Node %s left, %d tasks failed: %s", node_id, len(failed_task_ids), failed_task_ids)

        # 更新故障检测器
        self._failure_detector.unregister(node_id)

        # 更新故障转移管理器
        self._failover_manager.unregister(node_id)

        # 更新负载均衡器
        self._load_balancer.unregister_node(node_id)

        # 更新集群状态
        self.process_node_left(NodeId(node_id))

        # 清理连接映射（conn_id -> node_id）
        self._conn_node_map.pop(conn_id, None)

        # 更新选举模块的节点列表
        if self._election is not None:
            all_ids = list(self._conn_node_map.values()) + [self.node_id]
            self._election.update_node_ids(all_ids)

        logger.info("Node left: %s", node_id)

    async def _health_check_loop(self) -> None:
        """周期健康检查 (10s)。"""
        try:
            while self._running:
                await asyncio.sleep(10.0)
                if not self._running:
                    break

                # 扫描故障节点
                newly_failed = self._failover_manager.scan()
                for node_id in newly_failed:
                    logger.warning("Node failed: %s", node_id)
                    self.process_node_left(NodeId(node_id))

        except asyncio.CancelledError:
            pass

    async def _schedule_loop(self) -> None:
        """周期调度 (5s)。"""
        try:
            while self._running:
                await asyncio.sleep(5.0)
                if not self._running:
                    break

                # 调度待处理任务
                await self._dispatch_pending_tasks()

        except asyncio.CancelledError:
            pass

    async def _dispatch_pending_tasks(self) -> None:
        """分派待处理的任务到可用节点。"""
        from asc.types.state import TaskStatus

        pending_tasks = [
            t for t in self._state.tasks.values()
            if t.status == TaskStatus.PENDING
        ]
        if not pending_tasks:
            return

        available_nodes = list(self._state.nodes.keys())
        if not available_nodes:
            return

        for task in pending_tasks:
            selection = self._load_balancer.select(
                candidates=[str(nid) for nid in available_nodes],
                state=self._state,
            )
            if selection is None:
                continue

            # 通过 TCP 发送任务分派消息（定向发送到选中节点）
            if self._tcp_server is not None:
                # 查找目标节点的连接 ID
                target_conn_id = None
                for cid, nid in self._conn_node_map.items():
                    if nid == selection.node_id:
                        target_conn_id = cid
                        break

                if target_conn_id is None:
                    logger.warning("无法找到节点 %s 的连接，跳过任务分派", selection.node_id)
                    continue

                dispatch_envelope = Envelope(
                    channel=Channel.TASK_DISPATCH,
                    message=Message(
                        type=MessageType.TASK_DISPATCH,
                        sender_id=self.node_id,
                        payload={
                            "task_id": str(task.task_id),
                            "instance_id": str(task.instance_id),
                            "prompt": task.prompt,
                        },
                    ),
                    target=selection.node_id,
                )
                await self._tcp_server.send(target_conn_id, dispatch_envelope)
                self._load_balancer.start_request(selection.node_id)
                # 记录任务分配关系
                self._task_node_map[str(task.task_id)] = selection.node_id

    def _on_node_failed(self, node_id: str) -> None:
        """节点失败回调。"""
        logger.warning("Failover triggered for node: %s", node_id)
        self._load_balancer.unregister_node(node_id)

    # ------------------------------------------------------------------
    # 选举机制
    # ------------------------------------------------------------------

    def init_election(self, all_node_ids: list[str]) -> None:
        """初始化选举模块。

        在多 Master 场景下，通过 Bully 算法决定谁是真正的 Master。
        单 Master 部署时无需调用此方法。

        Args:
            all_node_ids: 集群中所有 Master 候选节点 ID 列表（含本节点）
        """
        self._election = BullyElection(
            node_id=self.node_id,
            all_node_ids=all_node_ids,
        )
        logger.info("选举模块已初始化，节点列表: %s", all_node_ids)

    async def start_election(self) -> None:
        """发起选举。

        向所有 ID 更高的节点发送 ELECTION 消息，
        如果没有更高 ID 节点或超时无响应，自己成为 Master。
        """
        if self._election is None:
            return

        self._election.start_election()
        logger.info("发起选举，当前时钟: %d", self._election.election_clock)

        # 发送 ELECTION 消息到所有更高 ID 节点
        election_messages = self._election.get_election_messages()
        for msg in election_messages:
            envelope = election_to_envelope(msg, target=msg.sender_id)
            if self._tcp_server is not None:
                # 广播到所有连接（选举消息需要到达所有 Master 候选）
                await self._tcp_server.broadcast(envelope)

        # 如果没有更高 ID 节点，直接成为 Master
        if self._election.should_become_master():
            self._election.become_master()
            await self._broadcast_coordinator()

    async def _handle_election_message(self, envelope: Envelope) -> None:
        """处理选举消息。"""
        if self._election is None:
            # 收到选举消息但未初始化选举模块，自动初始化
            sender_id = envelope.message.sender_id
            all_ids = list(self._conn_node_map.values()) + [self.node_id]
            self.init_election(all_ids)

        election_msg = envelope_to_election(envelope)
        if election_msg is None:
            return

        logger.debug(
            "收到选举消息: type=%s, sender=%s, clock=%d",
            election_msg.type.value, election_msg.sender_id, election_msg.election_clock,
        )

        self._election.handle_message(election_msg)

        # 收到更低 ID 节点的 ELECTION，回复 ALIVE（定向发送，非广播）
        if election_msg.type.value == "election" and election_msg.sender_id < self.node_id:
            alive_msg = ElectionMessage(
                type=ElectionMessageType.ALIVE,
                sender_id=self.node_id,
                election_clock=self._election.election_clock,
            )
            alive_envelope = election_to_envelope(alive_msg, target=election_msg.sender_id)
            if self._tcp_server is not None:
                # 查找目标节点的 conn_id，定向发送而非广播
                target_conn_id = None
                for cid, nid in self._conn_node_map.items():
                    if nid == election_msg.sender_id:
                        target_conn_id = cid
                        break
                if target_conn_id:
                    await self._tcp_server.send(target_conn_id, alive_envelope)
                else:
                    # 找不到目标连接，回退到广播
                    await self._tcp_server.broadcast(alive_envelope)

        # 收到 COORDINATOR，确认新 Master
        if election_msg.type.value == "coordinator":
            if self._election.is_master:
                logger.info("本节点成为集群 Master（选举确认）")
            else:
                logger.info("节点 %s 成为集群 Master", self._election.master_id)

    async def _election_timeout_loop(self) -> None:
        """选举超时检测循环。

        定期检查选举是否超时，超时则自己成为 Master。
        """
        try:
            while self._running:
                await asyncio.sleep(1.0)
                if not self._running or self._election is None:
                    break

                if self._election.check_timeout():
                    self._election.handle_timeout()
                    if self._election.is_master:
                        await self._broadcast_coordinator()

        except asyncio.CancelledError:
            pass

    async def _broadcast_coordinator(self) -> None:
        """广播 COORDINATOR 消息，宣告自己是 Master。"""
        if self._election is None:
            return

        coordinator_msg = self._election.get_coordinator_message()
        if coordinator_msg is None:
            return

        envelope = election_to_envelope(coordinator_msg)
        if self._tcp_server is not None:
            await self._tcp_server.broadcast(envelope)
        logger.info("已广播 COORDINATOR 消息，选举时钟: %d", coordinator_msg.election_clock)

    # ------------------------------------------------------------------
    # Command 处理（原有逻辑）
    # ------------------------------------------------------------------

    def process_node_joined(self, node_id: NodeId, ip: str, port: int) -> list[Event]:
        """处理节点加入。"""
        return self._emit(NodeJoined(node_id=node_id, ip=ip, port=port))

    def process_node_left(self, node_id: NodeId) -> list[Event]:
        """处理节点离开。"""
        return self._emit(NodeLeft(node_id=node_id))

    def process_create_instance(self, cmd: CreateInstance) -> list[Event]:
        """处理创建实例命令。"""
        node_ids = list(self._state.nodes.keys())
        if not node_ids:
            raise ValueError("无可用节点")

        inst_id = generate_instance_id()

        # TODO: 通过编排器计算最优节点分配，而非全节点部署
        return self._emit(
            InstanceCreated(
                instance_id=inst_id,
                model_id=cmd.model_id,
                node_ids=node_ids,
                sharding=cmd.sharding,
                rpc_endpoints=[],
            )
        )

    async def process_delete_instance(self, cmd: DeleteInstance) -> list[Event]:
        """处理删除实例命令。"""
        if cmd.instance_id not in self._state.instances:
            raise ValueError(f"实例 {cmd.instance_id} 不存在")

        # 调用编排器停止 llama-server 进程和 RPC Servers（异步）
        instance = self._state.instances[cmd.instance_id]
        await self._orchestrator.delete_instance(
            instance_id=cmd.instance_id,
            node_ids=instance.node_ids,
            nodes=self._state.nodes,
        )

        return self._emit(InstanceDeleted(instance_id=cmd.instance_id))

    def process_start_inference(self, cmd: StartInference) -> list[Event]:
        """处理推理命令。"""
        if cmd.instance_id not in self._state.instances:
            raise ValueError(f"实例 {cmd.instance_id} 不存在")

        task_id = generate_task_id()
        return self._emit(
            TaskCreated(
                task_id=task_id,
                instance_id=cmd.instance_id,
                prompt=cmd.prompt,
            )
        )

    def process_cancel_task(self, cmd: CancelTask) -> list[Event]:
        """处理取消任务命令。"""
        return self._emit(TaskCancelled(task_id=cmd.task_id))

    def process_shutdown_runner(self, cmd: ShutdownRunner) -> list[Event]:
        """处理关闭 Runner 命令。"""
        return self._emit(RunnerStatusUpdated(node_id=cmd.node_id, status="shutdown"))

    async def distribute_model(
        self,
        model_id: str,
        model_path: str,
        total_layers: int = 32,
    ) -> list[Any]:
        """将模型分发到集群 Worker 节点，并规划层分配。

        流程：
        1. 通过 ModelDistributor 将模型文件以 Binary Frame 分片分发到各 Worker
        2. 根据各节点 VRAM 比例规划层分配
        3. 通过 TCP 通知各 Worker 加载对应层

        Args:
            model_id: 模型 ID
            model_path: Master 上的模型文件路径
            total_layers: 模型总层数

        Returns:
            LayerAssignment 列表
        """
        # 初始化模型分发器（绑定 TCPServer）
        if self._model_distributor is None:
            self._model_distributor = ModelDistributor(tcp_server=self._tcp_server)
        else:
            self._model_distributor.set_tcp_server(self._tcp_server)

        # 构建目标节点列表（使用 conn_id 替代 IP:Port）
        target_nodes: dict[str, dict[str, Any]] = {}
        node_resources_map: dict[str, dict[str, Any]] = {}

        for node_id, node_info in self._state.nodes.items():
            # 查找该节点的 conn_id
            conn_id = None
            for cid, nid in self._conn_node_map.items():
                if nid == str(node_id):
                    conn_id = cid
                    break

            if conn_id is None:
                logger.warning("节点 %s 无活跃连接，跳过", node_id)
                continue

            ip = node_info.ip
            port = node_info.port
            target_nodes[str(node_id)] = {"conn_id": conn_id}
            # 获取硬件资源信息
            res = self._node_resources.get(str(node_id), {})
            vram_free = 0
            gpus = res.get("gpus", [])
            if gpus and isinstance(gpus[0], dict):
                vram_free = gpus[0].get("vram_free_mb", 0)
            node_resources_map[str(node_id)] = {
                "ip": ip,
                "port": port,
                "vram_free_mb": vram_free,
            }

        if not target_nodes:
            logger.warning("无可用 Worker 节点进行模型分发")
            return []

        # 1. 分发模型文件（通过 Binary Frame）
        distribute_results = await self._model_distributor.distribute_to_cluster(
            model_id=model_id,
            model_path=model_path,
            nodes=target_nodes,
        )

        # 2. 规划层分配
        assignments = self._model_distributor.plan_layer_assignment(
            model_id=model_id,
            model_path=model_path,
            total_layers=total_layers,
            node_resources=node_resources_map,
            distribute_results=distribute_results,
        )

        # 3. 通过 TCP 通知各 Worker 加载对应层
        for assignment in assignments:
            target_conn_id = None
            for cid, nid in self._conn_node_map.items():
                if nid == assignment.node_id:
                    target_conn_id = cid
                    break

            if target_conn_id is None:
                logger.warning("无法找到节点 %s 的连接，跳过层分配通知", assignment.node_id)
                continue

            notify_envelope = Envelope(
                channel=Channel.TASK_DISPATCH,
                message=Message(
                    type=MessageType.MODEL_DISTRIBUTE,
                    sender_id=self.node_id,
                    payload={
                        "model_id": model_id,
                        "model_path": assignment.model_path,
                        "layer_start": assignment.start_layer,
                        "layer_end": assignment.end_layer,
                        "total_layers": assignment.total_layers,
                        "gpu_layers": assignment.gpu_layers,
                    },
                ),
                target=assignment.node_id,
            )
            if self._tcp_server is not None:
                await self._tcp_server.send(target_conn_id, notify_envelope)

        logger.info(
            "模型 %s 分发和层分配完成: %d 个节点",
            model_id, len(assignments),
        )

        return assignments


# ------------------------------------------------------------------
# 消息分派表
# ------------------------------------------------------------------

_MESSAGE_HANDLERS: dict[MessageType, object] = {
    MessageType.NODE_JOINED: MasterNode._handle_node_joined,
    MessageType.NODE_LEFT: MasterNode._handle_node_left,
    MessageType.HEARTBEAT: MasterNode._handle_heartbeat,
    MessageType.CAPACITY_REPORT: MasterNode._handle_capacity_report,
    MessageType.TASK_RESULT: MasterNode._handle_task_result,
    MessageType.LOAD_MODEL_ACK: MasterNode._handle_load_model_ack,
    MessageType.UNLOAD_MODEL_ACK: MasterNode._handle_unload_model_ack,
}
