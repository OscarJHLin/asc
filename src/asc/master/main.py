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

from asc.core.event_log import EventLog, MemoryEventLog
from asc.core.failover import FailoverManager, FailureDetector
from asc.master.orchestrator import DistributedOrchestrator
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
        orchestrator: DistributedOrchestrator | None = None,
        failure_detector: FailureDetector | None = None,
        failover_manager: FailoverManager | None = None,
        load_balancer: LoadBalancer | None = None,
        tcp_server: TCPServer | None = None,
    ) -> None:
        self.node_id = node_id
        self._state = empty_state()
        self._next_index = 1
        self.event_log = event_log if event_log is not None else MemoryEventLog()
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
        self._tasks: list[asyncio.Task] = []
        self._running = False
        # conn_id -> node_id 映射
        self._conn_node_map: dict[str, str] = {}

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
        return [event]

    # ------------------------------------------------------------------
    # 事件循环
    # ------------------------------------------------------------------

    async def run(self, host: str = "0.0.0.0", port: int = 52414) -> None:
        """Master 主事件循环。

        1. 启动 TCP Server
        2. 设置消息分派
        3. 启动心跳检测定时任务
        4. 启动调度器定时任务
        """
        self._running = True

        # 初始化 TCP Server（如果未注入）
        if self._tcp_server is None:
            self._tcp_server = TCPServer(
                host=host,
                port=port,
                on_message=self._on_message,
            )

        # 启动 TCP Server
        await self._tcp_server.start()
        logger.info("Master %s started on %s:%s", self.node_id, host, self._tcp_server.port)

        # 启动后台定时任务
        health_task = asyncio.create_task(self._health_check_loop())
        schedule_task = asyncio.create_task(self._schedule_loop())
        self._tasks = [health_task, schedule_task]

        # 等待所有任务（正常情况下不会结束）
        await asyncio.gather(*self._tasks)

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
        handler = _MESSAGE_HANDLERS.get(msg_type)
        if handler is not None:
            await handler(self, envelope)
        else:
            logger.debug("Unhandled message type: %s", msg_type)

    async def _handle_node_joined(self, envelope: Envelope) -> None:
        """处理节点注册。"""
        payload = envelope.message.payload
        node_id = payload.get("node_id", "")
        ip = payload.get("ip", "")
        port = payload.get("port", 0)

        # 更新连接映射
        sender_id = envelope.message.sender_id
        self._conn_node_map[sender_id] = node_id

        # 注册到集群状态
        self.process_node_joined(NodeId(node_id), ip, port)

        # 注册到故障检测器
        self._failure_detector.register(node_id)

        # 注册到故障转移管理器
        self._failover_manager.register(node_id)

        # 注册到负载均衡器
        self._load_balancer.register_node(node_id)

        logger.info("Node joined: %s (%s:%s)", node_id, ip, port)

    async def _handle_heartbeat(self, envelope: Envelope) -> None:
        """处理心跳。"""
        node_id = envelope.message.payload.get("node_id", "")
        if not node_id:
            return

        # 更新故障检测器
        self._failure_detector.heartbeat(node_id)

        # 更新故障转移管理器
        self._failover_manager.heartbeat(node_id)

    async def _handle_capacity_report(self, envelope: Envelope) -> None:
        """处理容量上报。"""
        payload = envelope.message.payload
        node_id = payload.get("node_id", "")
        compute_score = payload.get("compute_score", 1.0)

        if node_id:
            self._load_balancer.register_node(node_id, compute_score)

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

    async def _handle_node_left(self, envelope: Envelope) -> None:
        """处理节点离开消息。"""
        payload = envelope.message.payload
        node_id = payload.get("node_id", "")
        if not node_id:
            return

        # 更新故障检测器
        self._failure_detector.unregister(node_id)

        # 更新故障转移管理器
        self._failover_manager.unregister(node_id)

        # 更新负载均衡器
        self._load_balancer.unregister_node(node_id)

        # 更新集群状态
        self.process_node_left(NodeId(node_id))

        # 清理连接映射
        conn_id = envelope.message.sender_id
        if conn_id in self._conn_node_map:
            del self._conn_node_map[conn_id]

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
                        target=selection.node_id,
                    ),
                )
                await self._tcp_server.send(target_conn_id, dispatch_envelope)
                self._load_balancer.start_request(selection.node_id)

    def _on_node_failed(self, node_id: str) -> None:
        """节点失败回调。"""
        logger.warning("Failover triggered for node: %s", node_id)
        self._load_balancer.unregister_node(node_id)

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

        # 简化：直接选择所有节点，实际应通过编排器计算
        # 预留：集成 DistributedOrchestrator
        return self._emit(
            InstanceCreated(
                instance_id=inst_id,
                model_id=cmd.model_id,
                node_ids=node_ids,
                sharding=cmd.sharding,
                rpc_endpoints=[],
            )
        )

    def process_delete_instance(self, cmd: DeleteInstance) -> list[Event]:
        """处理删除实例命令。"""
        if cmd.instance_id not in self._state.instances:
            raise ValueError(f"实例 {cmd.instance_id} 不存在")

        # 调用编排器停止 llama-server 进程和 RPC Servers
        instance = self._state.instances[cmd.instance_id]
        self._orchestrator.delete_instance(
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


# ------------------------------------------------------------------
# 消息分派表
# ------------------------------------------------------------------

_MESSAGE_HANDLERS: dict[MessageType, object] = {
    MessageType.NODE_JOINED: MasterNode._handle_node_joined,
    MessageType.NODE_LEFT: MasterNode._handle_node_left,
    MessageType.HEARTBEAT: MasterNode._handle_heartbeat,
    MessageType.CAPACITY_REPORT: MasterNode._handle_capacity_report,
    MessageType.TASK_RESULT: MasterNode._handle_task_result,
}
