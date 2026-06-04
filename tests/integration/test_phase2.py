"""Phase 2 集成测试：验证分布式系统的完整协作。

测试事件溯源 + 选举 + 推理引擎 + Worker Agent + 放置算法的端到端流程。
"""

from asc.core.election import BullyElection, ElectionMessage, ElectionMessageType
from asc.core.event_log import MemoryEventLog
from asc.network.protocol import Channel, Envelope, Message, MessageType
from asc.network.router import MessageRouter
from asc.scheduler.placement import PlacementEngine, PlacementStrategy
from asc.scheduler.splitter import TensorSplitCalculator
from asc.scheduler.topology import build_topology
from asc.types import (
    IndexedEvent,
    InstanceCreated,
    InstanceId,
    NodeId,
    NodeJoined,
    RunnerStatusUpdated,
    TaskCompleted,
    TaskCreated,
    TaskId,
    apply,
    empty_state,
)
from asc.worker.agent import GPUInfo, NodeResources
from asc.worker.runner import Runner, RunnerCommand, RunnerState


class TestDistributedInferenceWorkflow:
    """完整分布式推理工作流。"""

    def test_full_workflow(self):
        """完整流程：选举 -> 节点加入 -> 放置 -> 推理。"""

        # 1. 选举 Master
        election = BullyElection(
            node_id="master",
            all_node_ids=["master", "worker-1", "worker-2"],
        )
        election.start_election()
        # master 不是最高 ID，等待
        assert not election.should_become_master()

        # 模拟 worker-2 成为 Master
        msg = ElectionMessage(
            type=ElectionMessageType.COORDINATOR,
            sender_id="worker-2",
            election_clock=1,
        )
        election.handle_message(msg)
        assert election.master_id == "worker-2"

        # 2. 节点加入集群（事件溯源）
        state = empty_state()
        n1 = NodeId("master")
        n2 = NodeId("worker-1")
        state = apply(
            state, IndexedEvent(event=NodeJoined(node_id=n1, ip="10.0.0.1", port=52415), index=1)
        )
        state = apply(
            state, IndexedEvent(event=NodeJoined(node_id=n2, ip="10.0.0.2", port=52415), index=2)
        )
        state = apply(
            state, IndexedEvent(event=RunnerStatusUpdated(node_id=n1, status="ready"), index=3)
        )
        state = apply(
            state, IndexedEvent(event=RunnerStatusUpdated(node_id=n2, status="ready"), index=4)
        )
        assert len(state.nodes) == 2

        # 3. 放置算法
        resources = {
            "master": NodeResources(
                cpu_count=8,
                cpu_percent=25.0,
                memory_total_mb=32768,
                memory_free_mb=24000,
                gpus=[GPUInfo(index=0, name="RTX 4090", vram_total_mb=24564, vram_free_mb=8000)],
            ),
            "worker-1": NodeResources(
                cpu_count=4,
                cpu_percent=10.0,
                memory_total_mb=16384,
                memory_free_mb=12000,
                gpus=[GPUInfo(index=0, name="RTX 3060", vram_total_mb=12288, vram_free_mb=6000)],
            ),
        }
        topo = build_topology("master", resources, {"worker-1": "10.0.0.2:52415"})

        engine = PlacementEngine()
        placement = engine.place(
            model_vram_required_mb=12000,
            topology=topo,
            strategy=PlacementStrategy.TENSOR,
        )
        assert placement.success
        assert len(placement.selected_nodes) == 2

        # 4. 计算 tensor split
        split_result = TensorSplitCalculator.calculate(
            local_vram_free_mb=8000,
            workers_vram_free_mb={"worker-1": 6000},
            worker_addresses={"worker-1": "10.0.0.2:50052"},
        )
        assert split_result.is_distributed
        assert len(split_result.splits) == 2

        # 5. 创建实例（事件溯源）
        inst_id = InstanceId("i1")
        state = apply(
            state,
            IndexedEvent(
                event=InstanceCreated(
                    instance_id=inst_id,
                    model_id="llama-3.1-8b",
                    node_ids=[n1, n2],
                    sharding="tensor",
                ),
                index=5,
            ),
        )
        assert len(state.instances) == 1

        # 6. 推理任务
        task_id = TaskId("t1")
        state = apply(
            state,
            IndexedEvent(
                event=TaskCreated(task_id=task_id, instance_id=inst_id, prompt="Hello"), index=6
            ),
        )
        state = apply(
            state, IndexedEvent(event=TaskCompleted(task_id=task_id, output="Hi there!"), index=7)
        )
        assert state.tasks[task_id].output == "Hi there!"


class TestEventLogWithRouter:
    """事件日志 + 消息路由协作。"""

    def test_events_propagated_through_router(self):
        """事件通过路由器传播到订阅者。"""
        MemoryEventLog()
        router = MessageRouter(node_id="master")
        received = []

        router.subscribe(Channel.EVENTS, lambda env: received.append(env))

        # 模拟收到远程事件
        msg = Message(
            type=MessageType.NODE_JOINED,
            sender_id="worker-1",
            payload={"ip": "10.0.0.2", "port": 52415},
        )
        env = Envelope(channel=Channel.EVENTS, message=msg)
        router.handle_incoming(env)

        assert len(received) == 1
        assert received[0].message.sender_id == "worker-1"

    def test_own_messages_ignored(self):
        """自己发出的消息不应回环。"""
        router = MessageRouter(node_id="master")
        received = []

        router.subscribe(Channel.EVENTS, lambda env: received.append(env))

        msg = Message(
            type=MessageType.NODE_JOINED,
            sender_id="master",
            payload={},
        )
        env = Envelope(channel=Channel.EVENTS, message=msg)
        router.handle_incoming(env)

        assert len(received) == 0


class TestRunnerWithState:
    """Runner 状态机与 ClusterState 协作。"""

    def test_runner_error_reflected_in_state(self):
        """Runner 错误状态反映到 ClusterState。"""
        node_id = NodeId("n1")
        state = empty_state()
        state = apply(
            state,
            IndexedEvent(event=NodeJoined(node_id=node_id, ip="10.0.0.1", port=52415), index=1),
        )

        runner = Runner(node_id=node_id)
        runner.transition(RunnerCommand.LOAD)
        runner.transition(RunnerCommand.ERROR, error_msg="OOM")

        state = apply(
            state, IndexedEvent(event=RunnerStatusUpdated(node_id=node_id, status="error"), index=2)
        )

        assert state.nodes[node_id].runner_status == "error"
        assert runner.last_error == "OOM"

        # 恢复
        runner.transition(RunnerCommand.RESET)
        state = apply(
            state, IndexedEvent(event=RunnerStatusUpdated(node_id=node_id, status="idle"), index=3)
        )
        assert state.nodes[node_id].runner_status == "idle"
        assert runner.state == RunnerState.IDLE
