"""测试命令类型定义。

Command 表达"意图"，可被拒绝。与 Event（表达"已发生的事实"）严格分离。
"""

from asc.types.common import InstanceId, NodeId, TaskId
from asc.types.commands import (
    Command,
    CreateInstance,
    DeleteInstance,
    StartInference,
    CancelTask,
    ShutdownRunner,
    command_type,
)
from asc.types.events import InstanceCreated


class TestCommandType:
    """command_type 返回命令的类型标签。"""

    def test_create_instance(self):
        c = CreateInstance(model_id="llama-3.1-8b", sharding="tensor")
        assert command_type(c) == "create_instance"

    def test_delete_instance(self):
        c = DeleteInstance(instance_id=InstanceId("i1"))
        assert command_type(c) == "delete_instance"

    def test_start_inference(self):
        c = StartInference(instance_id=InstanceId("i1"), prompt="hello")
        assert command_type(c) == "start_inference"

    def test_cancel_task(self):
        c = CancelTask(task_id=TaskId("t1"))
        assert command_type(c) == "cancel_task"

    def test_shutdown_runner(self):
        c = ShutdownRunner(node_id=NodeId("n1"))
        assert command_type(c) == "shutdown_runner"


class TestCommandFields:
    """验证命令字段正确性。"""

    def test_create_instance_fields(self):
        c = CreateInstance(model_id="llama-3.1-8b", sharding="pipeline")
        assert c.model_id == "llama-3.1-8b"
        assert c.sharding == "pipeline"

    def test_delete_instance_fields(self):
        c = DeleteInstance(instance_id=InstanceId("i1"))
        assert c.instance_id == InstanceId("i1")

    def test_start_inference_fields(self):
        c = StartInference(
            instance_id=InstanceId("i1"),
            prompt="What is AI?",
            max_tokens=256,
            temperature=0.7,
        )
        assert c.instance_id == InstanceId("i1")
        assert c.prompt == "What is AI?"
        assert c.max_tokens == 256
        assert c.temperature == 0.7

    def test_start_inference_defaults(self):
        c = StartInference(instance_id=InstanceId("i1"), prompt="hello")
        assert c.max_tokens == 128
        assert c.temperature == 0.7

    def test_cancel_task_fields(self):
        c = CancelTask(task_id=TaskId("t1"))
        assert c.task_id == TaskId("t1")

    def test_shutdown_runner_fields(self):
        c = ShutdownRunner(node_id=NodeId("n1"))
        assert c.node_id == NodeId("n1")


class TestCommandIsNotEvent:
    """Command 和 Event 是不同的类型体系，不应混淆。"""

    def test_command_not_event_instance(self):
        from asc.types.events import InstanceCreated
        c = CreateInstance(model_id="m", sharding="tensor")
        assert not isinstance(c, InstanceCreated)

    def test_event_not_command_instance(self):
        e = InstanceCreated(
            instance_id=InstanceId("i1"),
            model_id="m",
            node_ids=[NodeId("n1")],
            sharding="tensor",
        )
        assert not isinstance(e, CreateInstance)
