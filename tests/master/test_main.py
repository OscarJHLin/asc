"""测试 Master 主循环。

Master 负责处理 Command、产生 Event、维护状态、调度任务。
"""

import pytest

from asc.core.event_log import MemoryEventLog
from asc.master.main import MasterNode
from asc.types import (
    NodeId,
)
from asc.types.commands import CreateInstance, DeleteInstance, StartInference
from asc.types.events import event_type


class TestMasterNode:
    """Master 节点。"""

    def test_create(self):
        master = MasterNode(node_id="master")
        assert master.node_id == "master"
        assert master.state.event_index == 0

    def test_process_node_joined(self):
        master = MasterNode(node_id="master")
        events = master.process_node_joined(
            node_id=NodeId("worker-1"),
            ip="10.0.0.2",
            port=52415,
        )
        assert len(events) == 1
        assert event_type(events[0]) == "node_joined"
        assert master.state.event_index == 1
        assert NodeId("worker-1") in master.state.nodes

    def test_process_create_instance(self):
        master = MasterNode(node_id="master")
        # 先加入节点
        master.process_node_joined(NodeId("n1"), "10.0.0.1", 52415)

        # 创建实例
        cmd = CreateInstance(model_id="llama-3.1-8b", sharding="tensor")
        events = master.process_create_instance(cmd)
        assert len(events) == 1
        assert event_type(events[0]) == "instance_created"
        assert len(master.state.instances) == 1

    def test_process_start_inference(self):
        master = MasterNode(node_id="master")
        master.process_node_joined(NodeId("n1"), "10.0.0.1", 52415)
        master.process_create_instance(CreateInstance(model_id="m", sharding="tensor"))

        inst_id = list(master.state.instances.keys())[0]
        cmd = StartInference(instance_id=inst_id, prompt="Hello")
        events = master.process_start_inference(cmd)
        assert len(events) == 1
        assert event_type(events[0]) == "task_created"

    @pytest.mark.asyncio
    async def test_process_delete_instance(self):
        master = MasterNode(node_id="master")
        master.process_node_joined(NodeId("n1"), "10.0.0.1", 52415)
        master.process_create_instance(CreateInstance(model_id="m", sharding="tensor"))

        inst_id = list(master.state.instances.keys())[0]
        cmd = DeleteInstance(instance_id=inst_id)
        events = await master.process_delete_instance(cmd)
        assert len(events) == 1
        assert event_type(events[0]) == "instance_deleted"
        assert len(master.state.instances) == 0

    def test_event_log_integration(self):
        """Master 事件写入日志。"""
        master = MasterNode(node_id="master", event_log=MemoryEventLog())
        master.process_node_joined(NodeId("n1"), "10.0.0.1", 52415)
        assert len(master.event_log) == 1
        assert master.event_log.last_index == 1

    def test_multiple_events(self):
        master = MasterNode(node_id="master")
        master.process_node_joined(NodeId("n1"), "10.0.0.1", 52415)
        master.process_node_joined(NodeId("n2"), "10.0.0.2", 52415)
        assert len(master.state.nodes) == 2
        assert master.state.event_index == 2
