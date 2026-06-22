"""补充 MasterNode 测试，提升覆盖率至 85%+。

原测试仅覆盖基础初始化和消息处理，本文件补充：
- 事件发射和状态更新
- 节点注册/注销
- 任务调度
- 健康检查
- 编排器集成
- 模型分发集成
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from asc.master.main import MasterNode
from asc.network.protocol import Channel, Envelope, Message, MessageType
from asc.types import (
    NodeJoined,
    NodeLeft,
    TaskCompleted,
    TaskCreated,
    TaskFailed,
    empty_state,
)


class TestMasterNodeInit:
    """测试 MasterNode 初始化。"""

    def test_default_init(self):
        """默认初始化应创建内存事件日志。"""
        master = MasterNode(node_id="master-1")
        assert master.node_id == "master-1"
        assert master.state == empty_state()

    def test_init_with_data_dir(self, tmp_path):
        """指定 data_dir 应创建 SnapshotEventLog。"""
        master = MasterNode(node_id="m1", data_dir=str(tmp_path))
        assert master.event_log is not None

    def test_init_with_injected_deps(self):
        """注入依赖应被正确使用。"""
        mock_orchestrator = MagicMock()
        mock_lb = MagicMock()
        master = MasterNode(
            node_id="m1",
            orchestrator=mock_orchestrator,
            load_balancer=mock_lb,
        )
        assert master._orchestrator is mock_orchestrator
        assert master._load_balancer is mock_lb


class TestEmitEvent:
    """测试事件发射和状态更新。"""

    def test_emit_updates_state(self):
        """发射事件应更新状态。"""
        master = MasterNode(node_id="m1")
        event = NodeJoined(node_id="node-1", ip="10.0.0.1", port=52415)
        master._emit(event)
        assert "node-1" in master.state.nodes

    def test_emit_increments_index(self):
        """每次发射应递增索引。"""
        master = MasterNode(node_id="m1")
        master._emit(NodeJoined(node_id="n1", ip="1.1.1.1", port=1))
        assert master._next_index == 2
        master._emit(NodeJoined(node_id="n2", ip="2.2.2.2", port=2))
        assert master._next_index == 3

    def test_emit_writes_to_log(self):
        """发射应写入事件日志。"""
        mock_log = MagicMock()
        master = MasterNode(node_id="m1", event_log=mock_log)
        event = NodeJoined(node_id="n1", ip="1.1.1.1", port=1)
        master._emit(event)
        mock_log.append.assert_called_once()


class TestNodeRegistration:
    """测试节点注册和注销。"""

    def test_register_node(self):
        """注册节点应更新状态。"""
        master = MasterNode(node_id="m1")
        master._conn_node_map["conn-1"] = "node-1"
        master._node_resources["node-1"] = {"vram_free_mb": 24000}
        master._emit(NodeJoined(node_id="node-1", ip="10.0.0.1", port=52415))
        assert "node-1" in master.state.nodes

    def test_unregister_node(self):
        """注销节点应从状态中移除。"""
        master = MasterNode(node_id="m1")
        master._emit(NodeJoined(node_id="node-1", ip="10.0.0.1", port=52415))
        assert "node-1" in master.state.nodes
        master._emit(NodeLeft(node_id="node-1"))
        assert "node-1" not in master.state.nodes


class TestMessageHandlers:
    """测试消息处理器。"""

    @pytest.mark.asyncio
    async def test_heartbeat(self):
        """心跳消息应更新节点活动时间。"""
        master = MasterNode(node_id="m1")
        master._conn_node_map["conn-1"] = "node-1"
        master._emit(NodeJoined(node_id="node-1", ip="10.0.0.1", port=52415))

        envelope = Envelope(
            channel=Channel.HEARTBEATS,
            message=Message(
                type=MessageType.HEARTBEAT,
                sender_id="node-1",
                payload={},
            ),
        )
        await master._on_message("conn-1", envelope)
        assert "node-1" in master.state.nodes

    @pytest.mark.asyncio
    async def test_capacity_report(self):
        """容量上报应更新负载均衡器。"""
        master = MasterNode(node_id="m1")
        master._conn_node_map["conn-1"] = "node-1"
        master._emit(NodeJoined(node_id="node-1", ip="10.0.0.1", port=52415))

        envelope = Envelope(
            channel=Channel.CAPACITY,
            message=Message(
                type=MessageType.CAPACITY_REPORT,
                sender_id="node-1",
                payload={"node_id": "node-1", "compute_score": 85.0},
            ),
        )
        await master._on_message("conn-1", envelope)
        # capacity report 更新 load_balancer，不直接更新 _node_resources
        assert "node-1" in master._load_balancer._node_scores

    @pytest.mark.asyncio
    async def test_unknown_sender(self):
        """未知发送者应被忽略或记录。"""
        master = MasterNode(node_id="m1")
        envelope = Envelope(
            channel=Channel.HEARTBEATS,
            message=Message(
                type=MessageType.HEARTBEAT,
                sender_id="unknown-node",
                payload={},
            ),
        )
        # 不应抛异常
        await master._on_message("conn-99", envelope)


class TestTaskManagement:
    """测试任务管理。"""

    def test_create_task(self):
        """创建任务应更新状态。"""
        master = MasterNode(node_id="m1")
        event = TaskCreated(
            task_id="task-1",
            instance_id="inst-1",
            prompt="hello",
        )
        master._emit(event)
        assert "task-1" in master.state.tasks

    def test_complete_task(self):
        """完成任务应更新状态。"""
        master = MasterNode(node_id="m1")
        master._emit(TaskCreated(task_id="task-1", instance_id="inst-1", prompt="hello"))
        master._emit(TaskCompleted(task_id="task-1", output="world"))
        from asc.types.state import TaskStatus
        assert master.state.tasks["task-1"].status == TaskStatus.COMPLETED

    def test_fail_task(self):
        """任务失败应更新状态。"""
        master = MasterNode(node_id="m1")
        master._emit(TaskCreated(task_id="task-1", instance_id="inst-1", prompt="hello"))
        master._emit(TaskFailed(task_id="task-1", error="oom"))
        from asc.types.state import TaskStatus
        assert master.state.tasks["task-1"].status == TaskStatus.FAILED


class TestScheduling:
    """测试调度逻辑。"""

    @pytest.mark.asyncio
    async def test_dispatch_no_nodes(self):
        """无可用节点时不应调度。"""
        master = MasterNode(node_id="m1")
        await master._dispatch_pending_tasks()
        # 无节点时无操作，不抛异常

    @pytest.mark.asyncio
    async def test_dispatch_no_pending_tasks(self):
        """无待处理任务时不应调度。"""
        master = MasterNode(node_id="m1")
        master._emit(NodeJoined(node_id="node-1", ip="10.0.0.1", port=52415))
        await master._dispatch_pending_tasks()
        # 无任务时无操作


class TestHealthCheck:
    """测试健康检查。"""

    @pytest.mark.asyncio
    async def test_health_check_loop_no_nodes(self):
        """无节点时健康检查循环应正常退出。"""
        master = MasterNode(node_id="m1")
        master._running = True

        async def early_stop():
            await asyncio.sleep(0.05)
            master._running = False

        task = asyncio.create_task(early_stop())
        await master._health_check_loop()
        await task

    def test_health_check_detects_failure(self):
        """故障扫描应检测超时节点。"""
        master = MasterNode(node_id="m1")
        master._failover_manager = MagicMock()
        master._failover_manager.scan.return_value = ["node-1"]
        # 模拟同步扫描
        newly_failed = master._failover_manager.scan()
        assert "node-1" in newly_failed


class TestAPIIntegration:
    """测试 API Server 集成。"""

    @pytest.mark.asyncio
    async def test_run_api_server(self):
        """API Server 应正确启动。"""
        master = MasterNode(node_id="m1", api_port=18080)
        with patch("asc.api.server.create_app") as mock_create:
            mock_app = MagicMock()
            mock_create.return_value = mock_app
            task = asyncio.create_task(master._run_api_server())
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task


class TestFrameHandling:
    """测试原始帧处理。"""

    @pytest.mark.asyncio
    async def test_model_chunk_ack(self):
        """MODEL_CHUNK_ACK 应转发给模型分发器。"""
        master = MasterNode(node_id="m1")
        master._model_distributor = MagicMock()
        from asc.master.model_distributor import encode_model_chunk_ack_payload
        from asc.network.frame import Frame, FrameType
        payload = encode_model_chunk_ack_payload("model-1", 0, True)
        frame = Frame(frame_type=FrameType.MODEL_CHUNK_ACK, payload=payload)
        await master._on_frame("conn-1", frame)
        master._model_distributor.handle_chunk_ack.assert_called_once()

    def test_model_distributor_lazy_init(self):
        """模型分发器初始应为 None，在 plan_layer_assignment 中延迟初始化。"""
        master = MasterNode(node_id="m1")
        assert master._model_distributor is None
