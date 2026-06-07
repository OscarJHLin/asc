"""分布式推理编排器。

负责 Master 侧的分布式推理编排：
- 接收 CreateInstance 命令
- 使用 PlacementEngine 选择最优节点组合
- 使用 TensorSplitCalculator 计算 tensor-split 参数
- 向 Workers 发送启动 RPC Server 请求
- 等待所有 RPC Server 就绪
- 启动 llama-server（带 --rpc 和 --tensor-split 参数）
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

from asc.engine.llama_server import LlamaServerBuilder, LlamaServerEngine
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
    error: str = ""


class DistributedOrchestrator:
    """分布式推理编排器。"""

    def __init__(
        self,
        placement_engine: PlacementEngine | None = None,
        split_calculator: TensorSplitCalculator | None = None,
    ) -> None:
        self._placement = placement_engine or PlacementEngine()
        self._splitter = split_calculator or TensorSplitCalculator()
        self._engines: dict[InstanceId, LlamaServerEngine] = {}

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

        # 4. 多节点分布式：启动 Worker RPC Servers
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

    def delete_instance(
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

        # 停止 Worker RPC Servers
        for node_id in node_ids:
            if node_id in nodes:
                self._stop_worker_rpc_server(node_id, nodes[node_id])

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

        通过 HTTP POST 调用 Worker 的 /rpc/start 端点，
        返回 Worker 汇报的 RPC 端点地址，失败时返回 None。
        """
        url = f"http://{node_info.ip}:{node_info.port}/rpc/start"
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(url, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            endpoint = data.get("endpoint")
            if endpoint:
                logger.info("Worker %s RPC 启动成功: %s", node_info.ip, endpoint)
                return endpoint
            logger.warning("Worker %s RPC 响应缺少 endpoint 字段", node_info.ip)
            return None
        except httpx.ConnectError:
            logger.warning("无法连接到 Worker %s:%s", node_info.ip, node_info.port)
            return None
        except httpx.TimeoutException:
            logger.warning("连接 Worker %s:%s 超时", node_info.ip, node_info.port)
            return None
        except httpx.HTTPStatusError as e:
            logger.warning("Worker %s 返回错误状态: %s", node_info.ip, e.response.status_code)
            return None
        except Exception:
            logger.exception("请求 Worker %s RPC 启动时发生未知错误", node_info.ip)
            return None

    def _stop_worker_rpc_server(self, node_id: NodeId, node_info: NodeInfo) -> None:
        """请求 Worker 停止 RPC Server。"""
        url = f"http://{node_info.ip}:{node_info.port}/rpc/stop"
        try:
            resp = httpx.post(url, timeout=10.0)
            resp.raise_for_status()
            logger.info("Worker %s RPC 已停止", node_info.ip)
        except httpx.ConnectError:
            logger.warning("停止 Worker %s:%s RPC 时无法连接", node_info.ip, node_info.port)
        except httpx.TimeoutException:
            logger.warning("停止 Worker %s:%s RPC 时超时", node_info.ip, node_info.port)
        except httpx.HTTPStatusError as e:
            logger.warning(
                "停止 Worker %s RPC 时返回错误状态: %s",
                node_info.ip, e.response.status_code,
            )
        except Exception:
            logger.exception("停止 Worker %s RPC 时发生未知错误", node_info.ip)

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
        """启动 llama-server（异步，不阻塞事件循环）。

        使用 asyncio.to_thread 将阻塞的启动流程放到线程池执行，
        避免阻塞事件循环数十秒。
        """
        builder = LlamaServerBuilder(
            model_path=model_path,
            rpc_servers=rpc_endpoints,
            tensor_split=tensor_split,
        )
        # 启动并等待就绪（在线程池中执行，避免阻塞事件循环）
        await asyncio.to_thread(lambda: list(builder.load()))
        return await asyncio.to_thread(builder.build)

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
