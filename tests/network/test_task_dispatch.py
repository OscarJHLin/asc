"""测试 ASC Cluster Link Protocol - 任务分派协议。"""


from asc.network.task_dispatch import (
    InferenceTask,
    TaskAcceptMessage,
    TaskCancelMessage,
    TaskDispatchMessage,
    TaskProgressMessage,
    TaskResultMessage,
    TaskStatus,
)


class TestInferenceTask:
    """推理任务。"""

    def test_create(self):
        task = InferenceTask(
            task_id="task-001",
            model_id="qwen-7b",
            prompt="Hello",
            max_tokens=128,
            temperature=0.7,
        )
        assert task.task_id == "task-001"
        assert task.model_id == "qwen-7b"
        assert task.prompt == "Hello"

    def test_default_values(self):
        task = InferenceTask(
            task_id="task-001",
            model_id="qwen-7b",
            prompt="Hi",
            max_tokens=256,
            temperature=0.5,
        )
        assert task.priority == 0
        assert task.stream is False


class TestTaskStatus:
    """任务状态。"""

    def test_values(self):
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.CANCELLED.value == "cancelled"


class TestTaskDispatchMessage:
    """任务分派消息。"""

    def test_create(self):
        task = InferenceTask(
            task_id="t1", model_id="m1", prompt="p",
            max_tokens=100, temperature=0.7,
        )
        msg = TaskDispatchMessage(
            task=task,
            target_node_id="worker-1",
            instance_id="inst-001",
        )
        assert msg.task.task_id == "t1"
        assert msg.target_node_id == "worker-1"
        assert msg.instance_id == "inst-001"

    def test_to_dict_from_dict(self):
        task = InferenceTask(
            task_id="t1", model_id="m1", prompt="hello",
            max_tokens=100, temperature=0.7,
        )
        msg = TaskDispatchMessage(
            task=task,
            target_node_id="w1",
            instance_id="i1",
        )
        d = msg.to_dict()
        restored = TaskDispatchMessage.from_dict(d)
        assert restored.task.task_id == "t1"
        assert restored.target_node_id == "w1"


class TestTaskAcceptMessage:
    """任务接受消息。"""

    def test_create(self):
        msg = TaskAcceptMessage(
            task_id="t1",
            node_id="w1",
            estimated_latency_ms=100.0,
        )
        assert msg.task_id == "t1"
        assert msg.node_id == "w1"
        assert msg.accepted is True

    def test_to_dict_from_dict(self):
        msg = TaskAcceptMessage(
            task_id="t1",
            node_id="w1",
            estimated_latency_ms=50.0,
        )
        d = msg.to_dict()
        restored = TaskAcceptMessage.from_dict(d)
        assert restored.task_id == "t1"
        assert restored.accepted is True


class TestTaskProgressMessage:
    """任务进度消息。"""

    def test_create(self):
        msg = TaskProgressMessage(
            task_id="t1",
            node_id="w1",
            tokens_generated=50,
            tokens_per_second=30.0,
            progress_fraction=0.5,
        )
        assert msg.task_id == "t1"
        assert msg.tokens_generated == 50
        assert msg.progress_fraction == 0.5

    def test_to_dict_from_dict(self):
        msg = TaskProgressMessage(
            task_id="t1",
            node_id="w1",
            tokens_generated=100,
            tokens_per_second=40.0,
            progress_fraction=1.0,
        )
        d = msg.to_dict()
        restored = TaskProgressMessage.from_dict(d)
        assert restored.tokens_generated == 100


class TestTaskResultMessage:
    """任务结果消息。"""

    def test_create_success(self):
        msg = TaskResultMessage(
            task_id="t1",
            node_id="w1",
            status=TaskStatus.COMPLETED,
            output_text="Hello world",
            tokens_generated=10,
            tokens_per_second=50.0,
            latency_ms=200.0,
        )
        assert msg.status == TaskStatus.COMPLETED
        assert msg.output_text == "Hello world"

    def test_create_failure(self):
        msg = TaskResultMessage(
            task_id="t1",
            node_id="w1",
            status=TaskStatus.FAILED,
            output_text="",
            tokens_generated=0,
            tokens_per_second=0.0,
            latency_ms=0.0,
            error="OOM",
        )
        assert msg.status == TaskStatus.FAILED
        assert msg.error == "OOM"

    def test_to_dict_from_dict(self):
        msg = TaskResultMessage(
            task_id="t1",
            node_id="w1",
            status=TaskStatus.COMPLETED,
            output_text="result",
            tokens_generated=5,
            tokens_per_second=25.0,
            latency_ms=200.0,
        )
        d = msg.to_dict()
        restored = TaskResultMessage.from_dict(d)
        assert restored.output_text == "result"
        assert restored.status == TaskStatus.COMPLETED


class TestTaskCancelMessage:
    """任务取消消息。"""

    def test_create(self):
        msg = TaskCancelMessage(
            task_id="t1",
            reason="timeout",
        )
        assert msg.task_id == "t1"
        assert msg.reason == "timeout"

    def test_to_dict_from_dict(self):
        msg = TaskCancelMessage(task_id="t1", reason="user_cancel")
        d = msg.to_dict()
        restored = TaskCancelMessage.from_dict(d)
        assert restored.reason == "user_cancel"
