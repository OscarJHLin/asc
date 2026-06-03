"""集成测试：验证类型系统、Runner、配置的协作。"""

from asc.types import (
    ClusterState,
    InstanceId,
    NodeId,
    TaskId,
    IndexedEvent,
    InstanceCreated,
    NodeJoined,
    NodeLeft,
    RunnerStatusUpdated,
    TaskCreated,
    TaskCompleted,
    apply,
    empty_state,
    generate_instance_id,
    generate_node_id,
    generate_task_id,
)
from asc.worker.runner import Runner, RunnerCommand, RunnerState
from asc.core.config import AscConfig


class TestClusterStateWithRunner:
    """验证 Runner 状态变更与 ClusterState 的事件溯源协作。"""

    def test_runner_lifecycle_reflected_in_state(self):
        """Runner 状态变更应通过事件反映到 ClusterState。"""
        node_id = generate_node_id()

        # 1. 节点加入
        state = empty_state()
        state = apply(state, IndexedEvent(
            event=NodeJoined(node_id=node_id, ip="10.0.0.1", port=52415), index=1
        ))
        assert state.nodes[node_id].runner_status == "unknown"

        # 2. Runner 启动加载
        runner = Runner(node_id=node_id)
        runner.transition(RunnerCommand.LOAD)
        state = apply(state, IndexedEvent(
            event=RunnerStatusUpdated(node_id=node_id, status="loading"), index=2
        ))
        assert state.nodes[node_id].runner_status == "loading"

        # 3. Runner 就绪
        runner.transition(RunnerCommand.LOAD_COMPLETE)
        state = apply(state, IndexedEvent(
            event=RunnerStatusUpdated(node_id=node_id, status="ready"), index=3
        ))
        assert state.nodes[node_id].runner_status == "ready"
        assert runner.can_accept_inference()

        # 4. Runner 运行中
        runner.transition(RunnerCommand.START_INFERENCE)
        state = apply(state, IndexedEvent(
            event=RunnerStatusUpdated(node_id=node_id, status="running"), index=4
        ))
        assert state.nodes[node_id].runner_status == "running"
        assert not runner.can_accept_inference()

    def test_node_removal_cleans_up(self):
        """节点离开后，状态中不再有该节点。"""
        n1 = NodeId("n1")
        n2 = NodeId("n2")
        state = empty_state()
        state = apply(state, IndexedEvent(event=NodeJoined(node_id=n1, ip="10.0.0.1", port=52415), index=1))
        state = apply(state, IndexedEvent(event=NodeJoined(node_id=n2, ip="10.0.0.2", port=52415), index=2))

        # 创建实例在 n1 上
        state = apply(state, IndexedEvent(event=InstanceCreated(
            instance_id=InstanceId("i1"), model_id="m", node_ids=[n1], sharding="tensor"
        ), index=3))

        # n1 离开
        state = apply(state, IndexedEvent(event=NodeLeft(node_id=n1), index=4))
        assert n1 not in state.nodes
        assert n2 in state.nodes
        # 实例仍在（需要 Master 的 plan 循环来清理）


class TestConfigWithTypes:
    """验证配置与类型系统的协作。"""

    def test_config_port_matches_event(self):
        cfg = AscConfig()
        port = cfg.get("node", "port")
        state = empty_state()
        state = apply(state, IndexedEvent(
            event=NodeJoined(node_id=NodeId("n1"), ip="10.0.0.1", port=port), index=1
        ))
        assert state.nodes[NodeId("n1")].port == port

    def test_model_mapping_with_instance(self):
        cfg = AscConfig()
        cfg.add_model_mapping("llama-3.1-8b", "/models/llama-3.1-8b-q4.gguf")

        model_path = cfg.resolve_model_path("llama-3.1-8b")
        assert model_path is not None

        # 创建实例使用解析后的路径
        state = empty_state()
        state = apply(state, IndexedEvent(event=InstanceCreated(
            instance_id=InstanceId("i1"),
            model_id="llama-3.1-8b",
            node_ids=[NodeId("n1")],
            sharding="tensor",
        ), index=1))
        assert state.instances[InstanceId("i1")].model_id == "llama-3.1-8b"


class TestFullWorkflow:
    """完整工作流：节点加入 -> 创建实例 -> 推理 -> 完成。"""

    def test_complete_inference_workflow(self):
        # 初始状态
        state = empty_state()

        # 1. 两个节点加入
        n1, n2 = NodeId("n1"), NodeId("n2")
        state = apply(state, IndexedEvent(event=NodeJoined(node_id=n1, ip="10.0.0.1", port=52415), index=1))
        state = apply(state, IndexedEvent(event=NodeJoined(node_id=n2, ip="10.0.0.2", port=52415), index=2))

        # 2. Runner 就绪
        state = apply(state, IndexedEvent(event=RunnerStatusUpdated(node_id=n1, status="ready"), index=3))
        state = apply(state, IndexedEvent(event=RunnerStatusUpdated(node_id=n2, status="ready"), index=4))

        # 3. 创建分布式实例
        inst_id = InstanceId("i1")
        state = apply(state, IndexedEvent(event=InstanceCreated(
            instance_id=inst_id,
            model_id="llama-3.1-8b",
            node_ids=[n1, n2],
            sharding="tensor",
        ), index=5))

        # 4. 提交推理任务
        task_id = TaskId("t1")
        state = apply(state, IndexedEvent(event=TaskCreated(
            task_id=task_id, instance_id=inst_id, prompt="Hello"
        ), index=6))
        assert state.tasks[task_id].status.value == "pending"

        # 5. 推理完成
        state = apply(state, IndexedEvent(event=TaskCompleted(
            task_id=task_id, output="Hi there!"
        ), index=7))
        assert state.tasks[task_id].status.value == "completed"
        assert state.tasks[task_id].output == "Hi there!"

        # 6. 最终状态验证
        assert len(state.nodes) == 2
        assert len(state.instances) == 1
        assert len(state.tasks) == 1
        assert state.event_index == 7
