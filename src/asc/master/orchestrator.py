"""分布式推理编排器。

负责 Master 侧的分布式推理编排：
- 接收 CreateInstance 命令
- 使用 PlacementEngine 选择最优节点组合
- Tensor 策略：使用 TensorSplitCalculator 计算 tensor-split 参数，启动带 --rpc 的 llama-server
- Pipeline 策略：使用 PipelinePlanner 分配层，每个阶段启动独立 llama-server
- 通过 Binary Frame 协议向 Workers 发送 RPC 启动/停止命令
- 等待所有 RPC Server 就绪
- 启动 llama-server（带 --rpc 和 --tensor-split 参数）
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from asc.engine.llama_server import LlamaServerBuilder, LlamaServerEngine
from asc.network.protocol import Channel, Envelope, Message, MessageType
from asc.network.transport import TCPServer
from asc.scheduler.pipeline import PipelinePlan, PipelinePlanner
from asc.scheduler.placement import PlacementEngine, PlacementStrategy
from asc.scheduler.splitter import TensorSplitCalculator
from asc.scheduler.topology import ClusterTopology, build_topology
from asc.types import InstanceId, NodeId
from asc.types.state import NodeInfo
from asc.worker.agent import NodeResources

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OrchestratorResult:
    """编排结果。"""

    success: bool
    instance_id: InstanceId | None = None
    node_ids: list[NodeId] = field(default_factory=list)
    rpc_endpoints: list[str] = field(default_factory=list)
    tensor_split: list[float] = field(default_factory=list)
    pipeline_plan: PipelinePlan | None = None
    error: str = ""


class DistributedOrchestrator:
    """分布式推理编排器。"""

    def __init__(
        self,
        placement_engine: PlacementEngine | None = None,
        split_calculator: TensorSplitCalculator | None = None,
        pipeline_planner: PipelinePlanner | None = None,
        tcp_server: TCPServer | None = None,
        conn_node_map: dict[str, str] | None = None,
    ) -> None:
        self._placement = placement_engine or PlacementEngine()
        self._splitter = split_calculator or TensorSplitCalculator()
        self._pipeline_planner = pipeline_planner or PipelinePlanner()
        self._engines: dict[InstanceId, LlamaServerEngine] = {}
        self._tcp_server = tcp_server
        self._conn_node_map = conn_node_map or {}
        # node_id -> conn_id 反向映射
        self._node_conn_map: dict[str, str] = {
            v: k for k, v in self._conn_node_map.items()
        }
        # 等待中的 RPC ACK 和 RESOURCE_RESPONSE: request_id -> asyncio.Future
        self._pending_acks: dict[str, asyncio.Future] = {}

    def set_tcp_server(self, tcp_server: TCPServer) -> None:
        """设置 TCPServer 实例。"""
        self._tcp_server = tcp_server

    def update_conn_map(self, conn_node_map: dict[str, str]) -> None:
        """更新连接映射。"""
        self._conn_node_map = conn_node_map
        self._node_conn_map = {v: k for k, v in conn_node_map.items()}

    def _get_conn_id(self, node_id: NodeId | str) -> str | None:
        """根据 node_id 查找对应的 conn_id。"""
        return self._node_conn_map.get(str(node_id))

    async def handle_rpc_start_ack(self, envelope: Envelope) -> None:
        """处理 RPC_START_ACK 响应。"""
        request_id = envelope.message.payload.get("request_id", "")
        future = self._pending_acks.pop(request_id, None)
        if future is not None and not future.done():
            future.set_result(envelope)

    async def handle_rpc_stop_ack(self, envelope: Envelope) -> None:
        """处理 RPC_STOP_ACK 响应。"""
        request_id = envelope.message.payload.get("request_id", "")
        future = self._pending_acks.pop(request_id, None)
        if future is not None and not future.done():
            future.set_result(envelope)

    async def handle_resource_response(self, envelope: Envelope) -> None:
        """处理 RESOURCE_RESPONSE 响应。"""
        request_id = envelope.message.payload.get("request_id", "")
        future = self._pending_acks.pop(request_id, None)
        if future is not None and not future.done():
            future.set_result(envelope)

    async def create_instance(
        self,
        instance_id: InstanceId,
        model_id: str,
        model_path: str,
        model_vram_required_mb: int,
        nodes: dict[NodeId, NodeInfo],
        node_resources: dict[NodeId, NodeResources],
        strategy: PlacementStrategy = PlacementStrategy.TENSOR,
        rpc_timeout: float = 30.0,
        _retry_depth: int = 0,
    ) -> OrchestratorResult:
        """创建分布式推理实例。

        Args:
            instance_id: 实例 ID
            model_id: 模型 ID
            model_path: 模型文件路径
            model_vram_required_mb: 模型所需 VRAM (MB)
            nodes: 节点信息 {node_id: NodeInfo}
            node_resources: 节点资源 {node_id: NodeResources}
            strategy: 放置策略
            rpc_timeout: RPC Server 启动超时（秒）

        Returns:
            OrchestratorResult
        """
        # 1. 构建拓扑 & 放置决策
        topology = self._build_topology(nodes, node_resources)
        placement = self._placement.place(
            model_vram_required_mb=model_vram_required_mb,
            topology=topology,
            strategy=strategy,
        )

        # 2. 校验放置结果
        selected_node_ids = self._validate_placement(placement)
        if selected_node_ids is None:
            return OrchestratorResult(
                success=False,
                error=placement.reason or "放置失败",
            )

        # 3. 单节点本地推理
        if len(selected_node_ids) == 1 and selected_node_ids[0] == NodeId("master"):
            return await self._create_local_instance(
                instance_id=instance_id,
                model_id=model_id,
                model_path=model_path,
            )

        # 4. Pipeline 策略：按层分片，每阶段独立 llama-server
        if strategy == PlacementStrategy.PIPELINE:
            return await self._create_pipeline_instance(
                instance_id=instance_id,
                model_id=model_id,
                model_path=model_path,
                selected_node_ids=selected_node_ids,
                node_resources=node_resources,
                nodes=nodes,
            )

        # 5. Tensor 策略：启动 Worker RPC Servers
        worker_node_ids = [
            nid for nid in selected_node_ids
            if nodes[nid].ip not in ("127.0.0.1", "localhost")
        ]
        rpc_endpoints = await self._start_worker_rpc_servers(
            selected_node_ids, nodes, timeout=rpc_timeout
        )

        # 5. 处理部分 RPC 失败
        rpc_result = await self._handle_rpc_failures(
            worker_node_ids=worker_node_ids,
            selected_node_ids=selected_node_ids,
            rpc_endpoints=rpc_endpoints,
            nodes=nodes,
            instance_id=instance_id,
            model_id=model_id,
            model_path=model_path,
            model_vram_required_mb=model_vram_required_mb,
            node_resources=node_resources,
            strategy=strategy,
        )
        if rpc_result is not None:
            return rpc_result

        # 6. 计算 tensor-split
        split_result = self._calculate_tensor_split(
            selected_node_ids, node_resources, nodes
        )

        # 7. 启动 llama-server（异步，不阻塞事件循环）
        try:
            engine = await self._start_llama_server(
                model_path=model_path,
                rpc_endpoints=split_result.rpc_endpoints,
                tensor_split=split_result.splits,
            )
            self._engines[instance_id] = engine
        except Exception as e:
            return OrchestratorResult(
                success=False,
                error=f"启动 llama-server 失败: {e}",
            )

        return OrchestratorResult(
            success=True,
            instance_id=instance_id,
            node_ids=selected_node_ids,
            rpc_endpoints=split_result.rpc_endpoints,
            tensor_split=split_result.splits,
        )

    async def delete_instance(
        self,
        instance_id: InstanceId,
        node_ids: list[NodeId],
        nodes: dict[NodeId, NodeInfo],
    ) -> None:
        """删除分布式推理实例，停止所有 RPC Server 和 llama-server 进程。"""
        # 停止 llama-server 引擎（释放 GPU/内存资源）
        engine = self._engines.pop(instance_id, None)
        if engine is not None:
            engine.close()
            logger.info("实例 %s 的 llama-server 已停止", instance_id)

        # 停止 Worker RPC Servers（异步，不阻塞事件循环）
        stop_tasks = []
        for node_id in node_ids:
            if node_id in nodes:
                stop_tasks.append(
                    self._stop_worker_rpc_server(node_id, nodes[node_id])
                )
        if stop_tasks:
            await asyncio.gather(*stop_tasks, return_exceptions=True)

    # ------------------------------------------------------------------
    # 私有方法
    # ------------------------------------------------------------------

    def _build_topology(
        self,
        nodes: dict[NodeId, NodeInfo],
        node_resources: dict[NodeId, NodeResources],
    ) -> ClusterTopology:
        """从节点信息构建拓扑。"""
        master_id = str(next(iter(nodes.keys()))) if nodes else "master"
        resources = {str(nid): node_resources[nid] for nid in nodes if nid in node_resources}
        addresses = {str(nid): f"{nodes[nid].ip}:{nodes[nid].port}" for nid in nodes}
        return build_topology(master_id, resources, addresses)

    def _validate_placement(self, placement: Any) -> list[NodeId] | None:
        """校验放置结果。

        Returns:
            选中节点 ID 列表，放置失败时返回 None。
        """
        if not placement.success:
            return None
        return [NodeId(n) for n in placement.selected_nodes]

    async def _handle_rpc_failures(
        self,
        worker_node_ids: list[NodeId],
        selected_node_ids: list[NodeId],
        rpc_endpoints: list[str],
        nodes: dict[NodeId, NodeInfo],
        instance_id: InstanceId,
        model_id: str,
        model_path: str,
        model_vram_required_mb: int,
        node_resources: dict[NodeId, NodeResources],
        strategy: PlacementStrategy,
    ) -> OrchestratorResult | None:
        """处理部分 Worker RPC 启动失败。

        当部分 Worker 启动失败时，尝试排除失败节点重新计算。

        Returns:
            需要提前返回的 OrchestratorResult（失败或重试结果），无需处理时返回 None。
        """
        if not worker_node_ids or len(rpc_endpoints) >= len(worker_node_ids):
            return None

        successful_workers = [
            nid for nid in worker_node_ids
            if any(nodes[nid].ip in ep for ep in rpc_endpoints)
        ]
        successful_nodes = [
            nid for nid in selected_node_ids
            if nid not in worker_node_ids or nid in successful_workers
        ]
        if len(successful_nodes) < len(selected_node_ids):
            return await self._retry_with_fewer_nodes(
                instance_id=instance_id,
                model_id=model_id,
                model_path=model_path,
                model_vram_required_mb=model_vram_required_mb,
                nodes=nodes,
                node_resources=node_resources,
                successful_nodes=successful_nodes,
                strategy=strategy,
            )
        return None

    async def _create_local_instance(
        self,
        instance_id: InstanceId,
        model_id: str,
        model_path: str,
    ) -> OrchestratorResult:
        """创建单节点本地实例。"""
        try:
            engine = await self._start_llama_server(
                model_path=model_path,
                rpc_endpoints=[],
                tensor_split=[1.0],
            )
            self._engines[instance_id] = engine
            return OrchestratorResult(
                success=True,
                instance_id=instance_id,
                node_ids=[NodeId("master")],
                rpc_endpoints=[],
                tensor_split=[1.0],
            )
        except Exception as e:
            return OrchestratorResult(
                success=False,
                error=f"启动本地 llama-server 失败: {e}",
            )

    async def _create_pipeline_instance(
        self,
        instance_id: InstanceId,
        model_id: str,
        model_path: str,
        selected_node_ids: list[NodeId],
        node_resources: dict[NodeId, NodeResources],
        nodes: dict[NodeId, NodeInfo],
    ) -> OrchestratorResult:
        """创建 Pipeline 并行实例。

        使用 PipelinePlanner 按节点 VRAM 比例分配模型层，
        每个阶段在对应节点上启动独立的 llama-server。

        注意：当前实现为规划 + 启动阶段。实际的层间数据路由
        （前一阶段输出传给后一阶段）需要在推理执行层实现。
        """
        # 1. 按 VRAM 比例规划 Pipeline 阶段
        node_vram = {
            str(nid): node_resources[nid].total_vram_free_mb
            for nid in selected_node_ids
            if nid in node_resources
        }
        # 默认 32 层，实际应从模型元数据获取
        pipeline_plan = self._pipeline_planner.plan_by_vram(
            total_layers=32,
            node_vram_mb=node_vram,
        )

        if not pipeline_plan.is_valid:
            return OrchestratorResult(
                success=False,
                error="Pipeline 分片规划失败：无法有效分配层",
                pipeline_plan=pipeline_plan,
            )

        # 2. 为每个阶段启动 llama-server
        stage_engines: dict[str, LlamaServerEngine] = {}
        for stage in pipeline_plan.stages:
            try:
                engine = await self._start_llama_server(
                    model_path=model_path,
                    rpc_endpoints=[],
                    tensor_split=[1.0],
                )
                stage_engines[stage.node_id] = engine
            except Exception as e:
                # 清理已启动的引擎
                for eng in stage_engines.values():
                    eng.close()
                return OrchestratorResult(
                    success=False,
                    error=f"Pipeline 阶段 {stage.node_id} 启动失败: {e}",
                    pipeline_plan=pipeline_plan,
                )

        # 3. 存储引擎（使用 instance_id 前缀区分多阶段）
        for node_id, engine in stage_engines.items():
            stage_key = InstanceId(f"{instance_id}::{node_id}")
            self._engines[stage_key] = engine

        return OrchestratorResult(
            success=True,
            instance_id=instance_id,
            node_ids=[NodeId(s.node_id) for s in pipeline_plan.stages],
            rpc_endpoints=[],
            tensor_split=[],
            pipeline_plan=pipeline_plan,
        )

    async def _start_worker_rpc_servers(
        self,
        node_ids: list[NodeId],
        nodes: dict[NodeId, NodeInfo],
        timeout: float = 30.0,
    ) -> list[str]:
        """向 Workers 并行发送启动 RPC Server 请求。

        返回成功启动的 RPC 端点列表。
        """
        tasks: list[asyncio.Task[str | None]] = []
        for node_id in node_ids:
            if node_id not in nodes:
                continue
            node_info = nodes[node_id]
            # 本地节点不需要启动 RPC Server
            if node_info.ip in ("127.0.0.1", "localhost"):
                continue
            tasks.append(asyncio.create_task(
                self._request_rpc_start(node_info, timeout)
            ))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        endpoints: list[str] = []
        for result in results:
            if isinstance(result, BaseException):
                logger.warning("RPC 启动请求异常: %s", result)
                continue
            if result is not None:
                endpoints.append(result)
        return endpoints

    async def _request_rpc_start(self, node_info: NodeInfo, timeout: float) -> str | None:
        """请求单个 Worker 启动 RPC Server。

        通过 Binary Frame 协议发送 RPC_START 命令，
        等待 Worker 回复 RPC_START_ACK，返回 RPC 端点地址。
        """
        conn_id = self._get_conn_id(node_info.node_id)
        if conn_id is None or self._tcp_server is None:
            logger.warning("无法找到节点 %s 的 TCP 连接", node_info.node_id)
            return None

        import uuid
        request_id = str(uuid.uuid4())

        envelope = Envelope(
            channel=Channel.COMMANDS,
            message=Message(
                type=MessageType.RPC_START,
                sender_id="master",
                payload={
                    "request_id": request_id,
                    "node_id": str(node_info.node_id),
                },
            ),
            target=str(node_info.node_id),
        )

        # 注册等待 Future
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Envelope] = loop.create_future()
        self._pending_acks[request_id] = future

        try:
            sent = await self._tcp_server.send(conn_id, envelope)
            if not sent:
                logger.warning("发送 RPC_START 到 %s 失败", node_info.ip)
                return None

            # 等待 ACK
            ack_envelope = await asyncio.wait_for(future, timeout=timeout)
            payload = ack_envelope.message.payload
            status = payload.get("status", "error")
            endpoint = payload.get("endpoint")

            if status == "ok" and endpoint:
                logger.info("Worker %s RPC 启动成功: %s", node_info.ip, endpoint)
                return endpoint
            error = payload.get("error", "未知错误")
            logger.warning("Worker %s RPC 启动失败: %s", node_info.ip, error)
            return None
        except asyncio.TimeoutError:
            logger.warning("等待 Worker %s RPC_START_ACK 超时", node_info.ip)
            return None
        except Exception:
            logger.exception("请求 Worker %s RPC 启动时发生未知错误", node_info.ip)
            return None
        finally:
            self._pending_acks.pop(request_id, None)

    async def _stop_worker_rpc_server(self, node_id: NodeId, node_info: NodeInfo) -> None:
        """请求 Worker 停止 RPC Server（通过 Binary Frame 协议）。"""
        conn_id = self._get_conn_id(node_id)
        if conn_id is None or self._tcp_server is None:
            logger.warning("无法找到节点 %s 的 TCP 连接，跳过 RPC 停止", node_id)
            return

        import uuid
        request_id = str(uuid.uuid4())

        envelope = Envelope(
            channel=Channel.COMMANDS,
            message=Message(
                type=MessageType.RPC_STOP,
                sender_id="master",
                payload={
                    "request_id": request_id,
                    "node_id": str(node_id),
                },
            ),
            target=str(node_id),
        )

        # 注册等待 Future
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Envelope] = loop.create_future()
        self._pending_acks[request_id] = future

        try:
            sent = await self._tcp_server.send(conn_id, envelope)
            if not sent:
                logger.warning("发送 RPC_STOP 到 %s 失败", node_info.ip)
                return

            # 等待 ACK（短超时，停止操作不应阻塞太久）
            await asyncio.wait_for(future, timeout=10.0)
            logger.info("Worker %s RPC 已停止", node_info.ip)
        except asyncio.TimeoutError:
            logger.warning("等待 Worker %s RPC_STOP_ACK 超时", node_info.ip)
        except Exception:
            logger.exception("停止 Worker %s RPC 时发生未知错误", node_info.ip)
        finally:
            self._pending_acks.pop(request_id, None)

    def _calculate_tensor_split(
        self,
        node_ids: list[NodeId],
        node_resources: dict[NodeId, NodeResources],
        nodes: dict[NodeId, NodeInfo],
    ) -> Any:
        """计算张量分割方案。"""
        # 找到 master 节点
        master_id = None
        master_vram = 0
        workers_vram = {}
        worker_addresses = {}

        for nid in node_ids:
            res = node_resources.get(nid)
            if res is None:
                continue
            vram = res.total_vram_free_mb
            if nodes[nid].ip in ("127.0.0.1", "localhost"):
                master_id = nid
                master_vram = vram
            else:
                workers_vram[str(nid)] = vram
                worker_addresses[str(nid)] = f"{nodes[nid].ip}:50052"

        if master_id is None:
            # 没有本地节点，取第一个作为 master
            master_id = node_ids[0]
            master_vram = node_resources[master_id].total_vram_free_mb
            workers_vram.pop(str(master_id), None)
            worker_addresses.pop(str(master_id), None)

        return self._splitter.calculate(
            local_vram_free_mb=master_vram,
            workers_vram_free_mb=workers_vram,
            worker_addresses=worker_addresses,
        )

    async def _start_llama_server(
        self,
        model_path: str,
        rpc_endpoints: list[str],
        tensor_split: list[float],
    ) -> LlamaServerEngine:
        """启动 llama-server（原生异步，不阻塞事件循环）。

        使用 asyncio.create_subprocess_exec 启动子进程，
        使用 asyncio.sleep + httpx.AsyncClient 进行健康检查，
        完全不阻塞事件循环，无需 asyncio.to_thread 包装。
        """
        builder = LlamaServerBuilder(
            model_path=model_path,
            rpc_servers=rpc_endpoints,
            tensor_split=tensor_split,
        )
        # 使用原生异步路径
        async for _ in builder.aload():
            pass
        return await builder.abuild()

    async def _retry_with_fewer_nodes(
        self,
        instance_id: InstanceId,
        model_id: str,
        model_path: str,
        model_vram_required_mb: int,
        nodes: dict[NodeId, NodeInfo],
        node_resources: dict[NodeId, NodeResources],
        successful_nodes: list[NodeId],
        strategy: PlacementStrategy,
        _retry_depth: int = 0,
    ) -> OrchestratorResult:
        """排除失败节点后重试。"""
        # 防止无限递归
        if _retry_depth >= 3:
            return OrchestratorResult(
                success=False,
                error="重试次数超限，所有节点 RPC 启动失败",
            )

        # 过滤掉失败节点
        filtered_nodes = {nid: info for nid, info in nodes.items() if nid in successful_nodes}
        filtered_resources = {
            nid: res for nid, res in node_resources.items()
            if nid in successful_nodes
        }

        if not filtered_nodes:
            return OrchestratorResult(
                success=False,
                error="所有节点 RPC 启动失败",
            )

        # 递归调用，但减少节点
        return await self.create_instance(
            instance_id=instance_id,
            model_id=model_id,
            model_path=model_path,
            model_vram_required_mb=model_vram_required_mb,
            nodes=filtered_nodes,
            node_resources=filtered_resources,
            strategy=strategy,
            _retry_depth=_retry_depth + 1,
        )

    async def query_resources(
        self,
        node_id: NodeId,
        timeout: float = 10.0,
    ) -> dict[str, Any] | None:
        """通过 Binary Frame 协议查询 Worker 节点资源。

        Args:
            node_id: 目标节点 ID
            timeout: 等待响应超时时间

        Returns:
            资源信息字典，失败返回 None
        """
        conn_id = self._get_conn_id(node_id)
        if conn_id is None or self._tcp_server is None:
            logger.warning("无法找到节点 %s 的 TCP 连接", node_id)
            return None

        import uuid
        request_id = str(uuid.uuid4())

        envelope = Envelope(
            channel=Channel.COMMANDS,
            message=Message(
                type=MessageType.RESOURCE_QUERY,
                sender_id="master",
                payload={
                    "request_id": request_id,
                    "node_id": str(node_id),
                },
            ),
            target=str(node_id),
        )

        loop = asyncio.get_running_loop()
        future: asyncio.Future[Envelope] = loop.create_future()
        self._pending_acks[request_id] = future

        try:
            sent = await self._tcp_server.send(conn_id, envelope)
            if not sent:
                logger.warning("发送 RESOURCE_QUERY 到 %s 失败", node_id)
                return None

            ack_envelope = await asyncio.wait_for(future, timeout=timeout)
            return ack_envelope.message.payload
        except asyncio.TimeoutError:
            logger.warning("等待 Worker %s RESOURCE_RESPONSE 超时", node_id)
            return None
        except Exception:
            logger.exception("查询 Worker %s 资源时发生未知错误", node_id)
            return None
        finally:
            self._pending_acks.pop(request_id, None)
