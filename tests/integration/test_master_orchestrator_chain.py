"""集成测试：Master → Orchestrator → Worker RPC 完整链路。

验证事件驱动状态机、分布式编排、故障检测与转移等核心流程的端到端协作。
所有外部依赖（llama-server、网络）均被 mock，测试纯逻辑正确性。
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from asc.core.event_log import MemoryEventLog
from asc.core.failover import FailoverConfig, FailoverManager, FailureDetector, NodeHealth
from asc.master.main import MasterNode
from asc.master.orchestrator import DistributedOrchestrator
from asc.scheduler.placement import PlacementStrategy
from asc.types import (
    InstanceCreated,
    NodeJoined,
    TaskCancelled,
    TaskCompleted,
    TaskCreated,
    TaskFailed,
    apply,
    empty_state,
)
from asc.types.commands import CancelTask, CreateInstance, DeleteInstance, StartInference
from asc.types.common import InstanceId, NodeId
from asc.types.events import IndexedEvent
from asc.types.state import NodeInfo, TaskStatus
from asc.worker.agent import NodeResources
from asc.worker.gpu_info import GPUInfo

# ---------------------------------------------------------------------------
# 辅助工具
# ---------------------------------------------------------------------------


def make_gpu(index: int = 0, name: str = "RTX 4090", total: int = 24564, free: int = 12000) -> GPUInfo:
    return GPUInfo(index=index, name=name, vram_total_mb=total, vram_free_mb=free)


def make_resources(
    cpu_count: int = 8,
    cpu_percent: float = 20.0,
    memory_total_mb: int = 32768,
    memory_free_mb: int = 24000,
    gpus: list[GPUInfo] | None = None,
    compute_score: float = 1.0,
) -> NodeResources:
    return NodeResources(
        cpu_count=cpu_count,
        cpu_percent=cpu_percent,
        memory_total_mb=memory_total_mb,
        memory_free_mb=memory_free_mb,
        gpus=gpus or [make_gpu()],
        compute_score=compute_score,
    )


# ---------------------------------------------------------------------------
# Scenario 1: 多节点注册 → 任务分派 → 推理 → 结果
# ---------------------------------------------------------------------------


class TestMultiNodeRegistrationToInference:
    """多节点注册 → 创建实例 → 推理 → 结果回传，验证状态转换正确。"""

    def test_full_lifecycle(self):
        """完整生命周期：注册 → 实例 → 推理 → 完成。"""
        # 1. 创建 MasterNode（mock orchestrator 避免真实 RPC）
        mock_orchestrator = MagicMock(spec=DistributedOrchestrator)
        master = MasterNode(node_id="master", orchestrator=mock_orchestrator)

        # 2. 注册 2 个 Worker 节点
        master.process_node_joined(NodeId("worker-1"), "10.0.0.1", 52415)
        master.process_node_joined(NodeId("worker-2"), "10.0.0.2", 52415)

        # 验证状态：2 个节点已注册
        assert len(master.state.nodes) == 2
        assert NodeId("worker-1") in master.state.nodes
        assert NodeId("worker-2") in master.state.nodes
        node1_info = master.state.nodes[NodeId("worker-1")]
        assert node1_info.ip == "10.0.0.1"
        assert node1_info.port == 52415

        # 3. 创建实例
        cmd = CreateInstance(model_id="llama-3.1-8b", sharding="tensor")
        events = master.process_create_instance(cmd)

        # 验证事件类型
        assert len(events) == 1
        assert isinstance(events[0], InstanceCreated)
        assert events[0].model_id == "llama-3.1-8b"
        assert events[0].sharding == "tensor"

        # 验证状态：1 个实例
        assert len(master.state.instances) == 1
        inst_id = list(master.state.instances.keys())[0]
        inst = master.state.instances[inst_id]
        assert inst.model_id == "llama-3.1-8b"
        assert inst.sharding == "tensor"
        assert len(inst.node_ids) == 2  # 两个节点

        # 4. 启动推理
        inf_cmd = StartInference(instance_id=inst_id, prompt="Hello, world!")
        events = master.process_start_inference(inf_cmd)

        # 验证事件类型
        assert len(events) == 1
        assert isinstance(events[0], TaskCreated)
        task_id = events[0].task_id

        # 验证状态：1 个 PENDING 任务
        assert len(master.state.tasks) == 1
        task = master.state.tasks[task_id]
        assert task.status == TaskStatus.PENDING
        assert task.instance_id == inst_id
        assert task.prompt == "Hello, world!"

        # 5. 模拟任务完成（直接 emit TaskCompleted）
        master._emit(TaskCompleted(task_id=task_id, output="Hi there!"))

        # 验证状态：任务已完成
        task = master.state.tasks[task_id]
        assert task.status == TaskStatus.COMPLETED
        assert task.output == "Hi there!"

        # 6. 验证事件日志完整性
        assert len(master.event_log) == 5  # 2 NodeJoined + 1 InstanceCreated + 1 TaskCreated + 1 TaskCompleted

    def test_task_failure_path(self):
        """推理失败路径：注册 → 实例 → 推理 → 失败。"""
        mock_orchestrator = MagicMock(spec=DistributedOrchestrator)
        master = MasterNode(node_id="master", orchestrator=mock_orchestrator)

        master.process_node_joined(NodeId("worker-1"), "10.0.0.1", 52415)
        cmd = CreateInstance(model_id="llama-3.1-8b", sharding="tensor")
        master.process_create_instance(cmd)

        inst_id = list(master.state.instances.keys())[0]
        inf_cmd = StartInference(instance_id=inst_id, prompt="test")
        events = master.process_start_inference(inf_cmd)
        task_id = events[0].task_id

        # 模拟任务失败
        master._emit(TaskFailed(task_id=task_id, error="OOM"))

        task = master.state.tasks[task_id]
        assert task.status == TaskStatus.FAILED
        assert task.error == "OOM"

    def test_cancel_task(self):
        """取消任务路径。"""
        mock_orchestrator = MagicMock(spec=DistributedOrchestrator)
        master = MasterNode(node_id="master", orchestrator=mock_orchestrator)

        master.process_node_joined(NodeId("worker-1"), "10.0.0.1", 52415)
        cmd = CreateInstance(model_id="llama-3.1-8b", sharding="tensor")
        master.process_create_instance(cmd)

        inst_id = list(master.state.instances.keys())[0]
        inf_cmd = StartInference(instance_id=inst_id, prompt="test")
        events = master.process_start_inference(inf_cmd)
        task_id = events[0].task_id

        # 取消任务
        cancel_cmd = CancelTask(task_id=task_id)
        events = master.process_cancel_task(cancel_cmd)
        assert isinstance(events[0], TaskCancelled)

        task = master.state.tasks[task_id]
        assert task.status == TaskStatus.CANCELLED


# ---------------------------------------------------------------------------
# Scenario 2: Master failover → 新 Master 选举 → 状态恢复
# ---------------------------------------------------------------------------


class TestMasterFailoverAndStateRecovery:
    """Master 故障转移：从事件日志重建状态。"""

    def test_state_recovery_from_event_log(self):
        """第一个 Master 处理事件后，第二个 Master 从 event log 恢复状态。"""
        # 1. 共享的 EventLog
        shared_log = MemoryEventLog()

        # 2. 第一个 Master 处理若干事件
        master1 = MasterNode(node_id="master-1", event_log=shared_log)
        master1.process_node_joined(NodeId("worker-1"), "10.0.0.1", 52415)
        master1.process_node_joined(NodeId("worker-2"), "10.0.0.2", 52415)
        cmd = CreateInstance(model_id="llama-3.1-8b", sharding="tensor")
        master1.process_create_instance(cmd)

        # 记录第一个 Master 的状态
        m1_node_count = len(master1.state.nodes)
        m1_instance_count = len(master1.state.instances)
        m1_event_index = master1.state.event_index

        # 3. 模拟 failover：第二个 Master 从 event log 重建状态
        master2 = MasterNode(node_id="master-2", event_log=shared_log)

        # 从事件日志重放状态
        state = empty_state()
        for ie in shared_log.read_from(0):
            state = apply(state, ie)
        master2._state = state
        master2._next_index = shared_log.last_index + 1

        # 4. 验证状态一致性
        assert len(master2.state.nodes) == m1_node_count
        assert len(master2.state.instances) == m1_instance_count
        assert master2.state.event_index == m1_event_index

        # 验证节点信息一致
        assert master2.state.nodes[NodeId("worker-1")].ip == "10.0.0.1"
        assert master2.state.nodes[NodeId("worker-2")].ip == "10.0.0.2"

        # 验证实例信息一致
        inst_id = list(master1.state.instances.keys())[0]
        assert inst_id in master2.state.instances
        assert master2.state.instances[inst_id].model_id == "llama-3.1-8b"

    def test_new_master_can_continue_processing(self):
        """恢复状态后，新 Master 能继续处理新事件。"""
        shared_log = MemoryEventLog()

        # 第一个 Master
        master1 = MasterNode(node_id="master-1", event_log=shared_log)
        master1.process_node_joined(NodeId("worker-1"), "10.0.0.1", 52415)

        # 第二个 Master 从日志恢复
        master2 = MasterNode(node_id="master-2", event_log=shared_log)
        state = empty_state()
        for ie in shared_log.read_from(0):
            state = apply(state, ie)
        master2._state = state
        master2._next_index = shared_log.last_index + 1

        # 新 Master 继续处理
        master2.process_node_joined(NodeId("worker-2"), "10.0.0.2", 52415)
        assert len(master2.state.nodes) == 2
        assert master2.state.event_index > master1.state.event_index


# ---------------------------------------------------------------------------
# Scenario 3: Worker failure → task handling
# ---------------------------------------------------------------------------


class TestWorkerFailureHandling:
    """Worker 故障 → 故障检测 → 状态清理。"""

    def test_failure_detector_detects_failed_node(self):
        """FailureDetector 能检测到心跳超时的节点。"""
        # 使用极短的超时配置加速测试
        config = FailoverConfig(
            heartbeat_timeout_sec=0.01,
            suspect_threshold=1,
            max_missed_heartbeats=2,
        )
        detector = FailureDetector(config=config)
        detector.register("worker-1")

        # 初始状态：HEALTHY
        assert detector.check("worker-1") == NodeHealth.HEALTHY

        # 等待心跳超时
        time.sleep(0.05)

        # 超时后：SUSPECT 或 FAILED
        health = detector.check("worker-1")
        assert health in (NodeHealth.SUSPECT, NodeHealth.FAILED)

    def test_failover_manager_triggers_callback(self):
        """FailoverManager 检测到故障时触发回调。"""
        failed_nodes: list[str] = []

        config = FailoverConfig(
            heartbeat_timeout_sec=0.01,
            suspect_threshold=1,
            max_missed_heartbeats=1,
        )
        detector = FailureDetector(config=config)
        manager = FailoverManager(
            detector=detector,
            on_node_failed=lambda nid: failed_nodes.append(nid),
        )
        manager.register("worker-1")

        # 等待心跳超时
        time.sleep(0.05)

        # 扫描故障节点
        newly_failed = manager.scan()

        assert "worker-1" in newly_failed
        assert "worker-1" in failed_nodes
        assert manager.is_failed("worker-1")

    def test_master_cleans_up_on_worker_failure(self):
        """Master 检测到 Worker 故障后清理状态。"""
        config = FailoverConfig(
            heartbeat_timeout_sec=0.01,
            suspect_threshold=1,
            max_missed_heartbeats=1,
        )
        detector = FailureDetector(config=config)
        failed_callback_nodes: list[str] = []

        def on_node_failed(node_id: str) -> None:
            failed_callback_nodes.append(node_id)

        failover_manager = FailoverManager(
            detector=detector,
            on_node_failed=on_node_failed,
        )

        master = MasterNode(
            node_id="master",
            failure_detector=detector,
            failover_manager=failover_manager,
        )

        # 注册 Worker
        master.process_node_joined(NodeId("worker-1"), "10.0.0.1", 52415)
        master._failure_detector.register("worker-1")
        master._failover_manager.register("worker-1")
        master._load_balancer.register_node("worker-1")

        assert len(master.state.nodes) == 1

        # 等待心跳超时
        time.sleep(0.05)

        # 扫描故障
        newly_failed = master._failover_manager.scan()
        assert "worker-1" in newly_failed

        # 模拟 health_check_loop 中的处理：注销故障节点
        for node_id in newly_failed:
            master.process_node_left(NodeId(node_id))

        # 验证状态已清理
        assert len(master.state.nodes) == 0
        assert NodeId("worker-1") not in master.state.nodes

    def test_heartbeat_resets_failure_state(self):
        """心跳能重置故障检测状态。"""
        config = FailoverConfig(
            heartbeat_timeout_sec=0.01,
            suspect_threshold=1,
            max_missed_heartbeats=3,
        )
        detector = FailureDetector(config=config)
        manager = FailoverManager(detector=detector)
        manager.register("worker-1")

        # 等待部分超时
        time.sleep(0.03)
        detector.check("worker-1")  # 触发 missed_count 增加

        # 发送心跳重置
        manager.heartbeat("worker-1")

        # 验证节点恢复
        health = detector.check("worker-1")
        assert health == NodeHealth.HEALTHY


# ---------------------------------------------------------------------------
# Scenario 4: Orchestrator create_instance with mocked RPC
# ---------------------------------------------------------------------------


class TestOrchestratorWithMockedRPC:
    """Orchestrator 通过 mock TCPServer 调用 Worker RPC（Binary Frame 协议）。"""

    @pytest.mark.asyncio
    async def test_create_instance_multi_node(self):
        """多节点分布式创建实例：mock RPC 启动和 llama-server。"""
        mock_tcp = MagicMock()

        async def mock_send(conn_id, envelope):
            from asc.network.protocol import Channel, Envelope, Message, MessageType
            request_id = envelope.message.payload.get("request_id", "")
            ack = Envelope(
                channel=Channel.COMMANDS,
                message=Message(
                    type=MessageType.RPC_START_ACK,
                    sender_id="worker",
                    payload={"request_id": request_id, "status": "ok", "endpoint": "0.0.0.0:50052", "port": 50052},
                ),
            )
            await orchestrator.handle_rpc_start_ack(ack)
            return True

        mock_tcp.send = mock_send

        conn_node_map = {"conn-w1": "worker-1", "conn-w2": "worker-2"}
        orchestrator = DistributedOrchestrator(
            tcp_server=mock_tcp,
            conn_node_map=conn_node_map,
        )

        nodes = {
            NodeId("master"): NodeInfo(node_id=NodeId("master"), ip="127.0.0.1", port=52415),
            NodeId("worker-1"): NodeInfo(node_id=NodeId("worker-1"), ip="10.0.0.1", port=52415),
            NodeId("worker-2"): NodeInfo(node_id=NodeId("worker-2"), ip="10.0.0.2", port=52415),
        }
        node_resources = {
            NodeId("master"): make_resources(gpus=[make_gpu(free=8000)]),
            NodeId("worker-1"): make_resources(gpus=[make_gpu(free=12000)]),
            NodeId("worker-2"): make_resources(gpus=[make_gpu(free=10000)]),
        }

        # Mock llama-server 启动
        mock_engine = MagicMock()
        mock_engine.close = MagicMock()

        with patch.object(orchestrator, "_start_llama_server", return_value=mock_engine):
            result = await orchestrator.create_instance(
                instance_id=InstanceId("inst-test-001"),
                model_id="llama-3.1-8b",
                model_path="/models/llama.gguf",
                model_vram_required_mb=20000,
                nodes=nodes,
                node_resources=node_resources,
                strategy=PlacementStrategy.TENSOR,
            )

        # 验证结果
        assert result.success
        assert result.instance_id == InstanceId("inst-test-001")
        assert len(result.node_ids) >= 2  # 至少 2 个节点
        assert len(result.tensor_split) >= 2  # 至少 2 个 split 比例
        assert result.instance_id in orchestrator._engines

    @pytest.mark.asyncio
    async def test_create_instance_insufficient_vram(self):
        """VRAM 不足时创建实例失败。"""
        orchestrator = DistributedOrchestrator()

        nodes = {
            NodeId("worker-1"): NodeInfo(node_id=NodeId("worker-1"), ip="10.0.0.1", port=52415),
        }
        node_resources = {
            NodeId("worker-1"): make_resources(gpus=[make_gpu(free=2000)]),
        }

        result = await orchestrator.create_instance(
            instance_id=InstanceId("inst-fail"),
            model_id="big-model",
            model_path="/models/big.gguf",
            model_vram_required_mb=50000,  # 远超可用 VRAM
            nodes=nodes,
            node_resources=node_resources,
        )

        assert not result.success
        assert result.error  # 有错误信息

    @pytest.mark.asyncio
    async def test_delete_instance_cleans_up(self):
        """删除实例时清理引擎和 RPC。"""
        mock_tcp = MagicMock()

        async def mock_send(conn_id, envelope):
            from asc.network.protocol import Channel, Envelope, Message, MessageType
            request_id = envelope.message.payload.get("request_id", "")
            ack = Envelope(
                channel=Channel.COMMANDS,
                message=Message(
                    type=MessageType.RPC_STOP_ACK,
                    sender_id="worker-1",
                    payload={"request_id": request_id, "status": "ok"},
                ),
            )
            await orchestrator.handle_rpc_stop_ack(ack)
            return True

        mock_tcp.send = mock_send

        orchestrator = DistributedOrchestrator(
            tcp_server=mock_tcp,
            conn_node_map={"conn-w1": "worker-1"},
        )

        # 预先放入一个 mock engine
        mock_engine = MagicMock()
        orchestrator._engines[InstanceId("inst-del")] = mock_engine

        nodes = {
            NodeId("worker-1"): NodeInfo(node_id=NodeId("worker-1"), ip="10.0.0.1", port=52415),
        }

        await orchestrator.delete_instance(
            instance_id=InstanceId("inst-del"),
            node_ids=[NodeId("worker-1")],
            nodes=nodes,
        )

        # 验证 engine 已清理
        assert InstanceId("inst-del") not in orchestrator._engines
        mock_engine.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_instance_rpc_failure_triggers_retry(self):
        """部分 RPC 启动失败时触发重试。"""
        mock_tcp = MagicMock()

        async def mock_send(conn_id, envelope):
            from asc.network.protocol import Channel, Envelope, Message, MessageType
            request_id = envelope.message.payload.get("request_id", "")
            node_id = envelope.message.payload.get("node_id", "")
            if node_id == "worker-2":
                # worker-2 失败：不回复 ACK，模拟超时
                return True
            # worker-1 成功
            ack = Envelope(
                channel=Channel.COMMANDS,
                message=Message(
                    type=MessageType.RPC_START_ACK,
                    sender_id=node_id,
                    payload={"request_id": request_id, "status": "ok", "endpoint": "10.0.0.1:50052", "port": 50052},
                ),
            )
            await orchestrator.handle_rpc_start_ack(ack)
            return True

        mock_tcp.send = mock_send

        conn_node_map = {"conn-w1": "worker-1", "conn-w2": "worker-2"}
        orchestrator = DistributedOrchestrator(
            tcp_server=mock_tcp,
            conn_node_map=conn_node_map,
        )

        nodes = {
            NodeId("worker-1"): NodeInfo(node_id=NodeId("worker-1"), ip="10.0.0.1", port=52415),
            NodeId("worker-2"): NodeInfo(node_id=NodeId("worker-2"), ip="10.0.0.2", port=52415),
        }
        node_resources = {
            NodeId("worker-1"): make_resources(gpus=[make_gpu(free=12000)]),
            NodeId("worker-2"): make_resources(gpus=[make_gpu(free=12000)]),
        }

        mock_engine = MagicMock()
        mock_engine.close = MagicMock()

        with patch.object(orchestrator, "_start_llama_server", return_value=mock_engine):
            result = await orchestrator.create_instance(
                instance_id=InstanceId("inst-retry"),
                model_id="llama-3.1-8b",
                model_path="/models/llama.gguf",
                model_vram_required_mb=10000,
                nodes=nodes,
                node_resources=node_resources,
                strategy=PlacementStrategy.TENSOR,
                rpc_timeout=0.5,
            )

        # 重试后应该用 worker-1 单节点成功（VRAM 足够）
        assert result.success


# ---------------------------------------------------------------------------
# Scenario 5: End-to-end MasterNode + Orchestrator chain
# ---------------------------------------------------------------------------


class TestEndToEndMasterOrchestratorChain:
    """端到端：MasterNode + DistributedOrchestrator 完整链路。"""

    @pytest.mark.asyncio
    async def test_create_and_delete_instance(self):
        """MasterNode 通过 Orchestrator 创建和删除实例。"""
        mock_tcp = MagicMock()

        async def mock_send(conn_id, envelope):
            from asc.network.protocol import Channel, Envelope, Message, MessageType
            request_id = envelope.message.payload.get("request_id", "")
            if envelope.message.type == MessageType.RPC_STOP:
                ack = Envelope(
                    channel=Channel.COMMANDS,
                    message=Message(
                        type=MessageType.RPC_STOP_ACK,
                        sender_id="worker",
                        payload={"request_id": request_id, "status": "ok"},
                    ),
                )
                await orchestrator.handle_rpc_stop_ack(ack)
            return True

        mock_tcp.send = mock_send

        orchestrator = DistributedOrchestrator(
            tcp_server=mock_tcp,
            conn_node_map={"conn-w1": "worker-1", "conn-w2": "worker-2"},
        )
        master = MasterNode(node_id="master", orchestrator=orchestrator)

        # 注册节点（含 GPU 资源）
        master.process_node_joined(NodeId("worker-1"), "10.0.0.1", 52415)
        master.process_node_joined(NodeId("worker-2"), "10.0.0.2", 52415)

        # 创建实例
        cmd = CreateInstance(model_id="llama-3.1-8b", sharding="tensor")
        events = master.process_create_instance(cmd)
        assert len(events) == 1
        assert isinstance(events[0], InstanceCreated)

        inst_id = list(master.state.instances.keys())[0]
        assert inst_id in master.state.instances

        # 删除实例
        del_cmd = DeleteInstance(instance_id=inst_id)
        events = await master.process_delete_instance(del_cmd)

        # 验证实例已删除
        assert inst_id not in master.state.instances
        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_orchestrator_engines_cleaned_after_delete(self):
        """删除实例后 orchestrator 的 engines dict 被清理。"""
        mock_tcp = MagicMock()

        async def mock_send(conn_id, envelope):
            from asc.network.protocol import Channel, Envelope, Message, MessageType
            request_id = envelope.message.payload.get("request_id", "")
            if envelope.message.type == MessageType.RPC_STOP:
                ack = Envelope(
                    channel=Channel.COMMANDS,
                    message=Message(
                        type=MessageType.RPC_STOP_ACK,
                        sender_id="worker-1",
                        payload={"request_id": request_id, "status": "ok"},
                    ),
                )
                await orchestrator.handle_rpc_stop_ack(ack)
            return True

        mock_tcp.send = mock_send

        orchestrator = DistributedOrchestrator(
            tcp_server=mock_tcp,
            conn_node_map={"conn-w1": "worker-1"},
        )
        master = MasterNode(node_id="master", orchestrator=orchestrator)

        master.process_node_joined(NodeId("worker-1"), "10.0.0.1", 52415)
        cmd = CreateInstance(model_id="llama-3.1-8b", sharding="tensor")
        master.process_create_instance(cmd)

        inst_id = list(master.state.instances.keys())[0]

        # 手动向 orchestrator 注入 mock engine（模拟 create_instance 后的状态）
        mock_engine = MagicMock()
        orchestrator._engines[inst_id] = mock_engine

        del_cmd = DeleteInstance(instance_id=inst_id)
        await master.process_delete_instance(del_cmd)

        # 验证 orchestrator 的 engines 已清理
        assert inst_id not in orchestrator._engines
        mock_engine.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_full_chain_with_orchestrator_create(self):
        """完整链路：MasterNode 使用真实 Orchestrator 创建实例（mock llama-server）。"""
        mock_tcp = MagicMock()

        async def mock_send(conn_id, envelope):
            from asc.network.protocol import Channel, Envelope, Message, MessageType
            request_id = envelope.message.payload.get("request_id", "")
            if envelope.message.type == MessageType.RPC_START:
                ack = Envelope(
                    channel=Channel.COMMANDS,
                    message=Message(
                        type=MessageType.RPC_START_ACK,
                        sender_id="worker-1",
                        payload={"request_id": request_id, "status": "ok", "endpoint": "10.0.0.1:50052", "port": 50052},
                    ),
                )
                await orchestrator.handle_rpc_start_ack(ack)
            elif envelope.message.type == MessageType.RPC_STOP:
                ack = Envelope(
                    channel=Channel.COMMANDS,
                    message=Message(
                        type=MessageType.RPC_STOP_ACK,
                        sender_id="worker-1",
                        payload={"request_id": request_id, "status": "ok"},
                    ),
                )
                await orchestrator.handle_rpc_stop_ack(ack)
            return True

        mock_tcp.send = mock_send

        conn_node_map = {"conn-master": "master", "conn-w1": "worker-1"}
        orchestrator = DistributedOrchestrator(
            tcp_server=mock_tcp,
            conn_node_map=conn_node_map,
        )
        master = MasterNode(node_id="master", orchestrator=orchestrator)

        # 注册节点
        master.process_node_joined(NodeId("master"), "127.0.0.1", 52415)
        master.process_node_joined(NodeId("worker-1"), "10.0.0.1", 52415)

        # 准备资源信息
        node_resources = {
            NodeId("master"): make_resources(gpus=[make_gpu(free=8000)]),
            NodeId("worker-1"): make_resources(gpus=[make_gpu(free=12000)]),
        }

        mock_engine = MagicMock()
        mock_engine.close = MagicMock()

        with patch.object(orchestrator, "_start_llama_server", return_value=mock_engine):
            result = await orchestrator.create_instance(
                instance_id=InstanceId("inst-e2e"),
                model_id="llama-3.1-8b",
                model_path="/models/llama.gguf",
                model_vram_required_mb=15000,
                nodes=master.state.nodes,
                node_resources=node_resources,
                strategy=PlacementStrategy.TENSOR,
            )

        # 验证编排结果
        assert result.success
        assert result.instance_id == InstanceId("inst-e2e")
        assert len(result.node_ids) >= 1
        assert InstanceId("inst-e2e") in orchestrator._engines

        # 清理：删除实例
        await orchestrator.delete_instance(
            instance_id=InstanceId("inst-e2e"),
            node_ids=result.node_ids,
            nodes=master.state.nodes,
        )

        assert InstanceId("inst-e2e") not in orchestrator._engines
        mock_engine.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_node_left_during_instance_lifecycle(self):
        """实例生命周期中节点离开的处理。"""
        mock_orchestrator = MagicMock(spec=DistributedOrchestrator)
        master = MasterNode(node_id="master", orchestrator=mock_orchestrator)

        master.process_node_joined(NodeId("worker-1"), "10.0.0.1", 52415)
        master.process_node_joined(NodeId("worker-2"), "10.0.0.2", 52415)

        cmd = CreateInstance(model_id="llama-3.1-8b", sharding="tensor")
        master.process_create_instance(cmd)

        # 节点离开
        master.process_node_left(NodeId("worker-2"))

        # 验证节点已从状态中移除
        assert NodeId("worker-2") not in master.state.nodes
        assert NodeId("worker-1") in master.state.nodes

        # 实例仍然存在（但 node_ids 可能需要更新——当前实现不自动更新）
        assert len(master.state.instances) == 1


# ---------------------------------------------------------------------------
# 补充：事件溯源确定性验证
# ---------------------------------------------------------------------------


class TestEventSourcingDeterminism:
    """验证事件溯源的确定性：相同事件序列产生相同状态。"""

    def test_same_events_same_state(self):
        """相同事件序列在两个 MasterNode 上产生相同状态。"""
        log1 = MemoryEventLog()
        log2 = MemoryEventLog()

        master1 = MasterNode(node_id="m1", event_log=log1)
        master2 = MasterNode(node_id="m2", event_log=log2)

        # 两个 Master 处理完全相同的事件序列
        for master in [master1, master2]:
            master.process_node_joined(NodeId("n1"), "10.0.0.1", 52415)
            master.process_node_joined(NodeId("n2"), "10.0.0.2", 52415)
            cmd = CreateInstance(model_id="llama-3.1-8b", sharding="tensor")
            master.process_create_instance(cmd)
            inst_id = list(master.state.instances.keys())[0]
            inf_cmd = StartInference(instance_id=inst_id, prompt="test")
            master.process_start_inference(inf_cmd)

        # 验证节点数、实例数、任务数一致
        assert len(master1.state.nodes) == len(master2.state.nodes)
        assert len(master1.state.instances) == len(master2.state.instances)
        assert len(master1.state.tasks) == len(master2.state.tasks)

        # 验证 event_index 一致
        assert master1.state.event_index == master2.state.event_index

    def test_apply_is_idempotent_for_same_index(self):
        """相同 index 的事件重复 apply 不改变状态。"""
        state = empty_state()
        event = NodeJoined(node_id=NodeId("n1"), ip="10.0.0.1", port=52415)
        ie = IndexedEvent(event=event, index=1)

        state1 = apply(state, ie)
        state2 = apply(state1, ie)  # 重复应用

        # event_index 不变（因为 index 相同）
        assert state2.event_index == 1
        assert len(state2.nodes) == 1


class TestMasterDiskPersistence:
    """Master 节点磁盘持久化集成测试。"""

    def test_data_dir_creates_snapshot_event_log(self, tmp_path):
        """指定 data_dir 时 MasterNode 使用 SnapshotEventLog。"""
        from asc.core.event_log import SnapshotEventLog
        from asc.master.main import MasterNode

        master = MasterNode(node_id="master-1", data_dir=str(tmp_path))
        assert isinstance(master.event_log, SnapshotEventLog)

    def test_events_persisted_to_disk(self, tmp_path):
        """事件持久化到磁盘文件。"""
        from asc.master.main import MasterNode

        master = MasterNode(node_id="master-1", data_dir=str(tmp_path))
        master.process_node_joined(NodeId("n1"), "10.0.0.1", 52415)
        master.process_node_joined(NodeId("n2"), "10.0.0.2", 52415)

        # 刷新缓冲区
        master.event_log.flush()

        # 验证文件存在且有内容
        log_file = tmp_path / "events.jsonl"
        assert log_file.exists()
        content = log_file.read_text(encoding="utf-8").strip()
        assert len(content) > 0
        lines = [l for l in content.split("\n") if l.strip()]
        assert len(lines) == 2

    def test_state_recovery_from_disk(self, tmp_path):
        """从磁盘事件日志恢复状态。"""
        from asc.master.main import MasterNode

        # 第一个 Master 写入事件
        master1 = MasterNode(node_id="master-1", data_dir=str(tmp_path))
        master1.process_node_joined(NodeId("n1"), "10.0.0.1", 52415)
        master1.process_node_joined(NodeId("n2"), "10.0.0.2", 52415)
        master1.event_log.flush()

        # 第二个 Master 从同一目录恢复
        master2 = MasterNode(node_id="master-2", data_dir=str(tmp_path))
        # 从日志重放状态
        state = empty_state()
        for ie in master2.event_log.read_from(0):
            state = apply(state, ie)

        assert len(state.nodes) == 2
        assert state.nodes[NodeId("n1")].ip == "10.0.0.1"
        assert state.nodes[NodeId("n2")].ip == "10.0.0.2"

    def test_no_data_dir_uses_memory_log(self):
        """不指定 data_dir 时使用 MemoryEventLog。"""
        from asc.core.event_log import MemoryEventLog
        from asc.master.main import MasterNode

        master = MasterNode(node_id="master-1")
        assert isinstance(master.event_log, MemoryEventLog)

    def test_explicit_event_log_takes_priority(self, tmp_path):
        """显式传入 event_log 优先于 data_dir。"""
        from asc.core.event_log import MemoryEventLog
        from asc.master.main import MasterNode

        log = MemoryEventLog()
        master = MasterNode(node_id="master-1", event_log=log, data_dir=str(tmp_path))
        assert master.event_log is log
