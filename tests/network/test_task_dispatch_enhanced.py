"""ASC 任务分派协议增强测试 - 覆盖边界值、生命周期、不可变性与向后兼容。"""

import dataclasses

import pytest

from asc.network.task_dispatch import (
    InferenceTask,
    TaskAcceptMessage,
    TaskCancelMessage,
    TaskDispatchMessage,
    TaskProgressMessage,
    TaskResultMessage,
    TaskStatus,
)


# ---------------------------------------------------------------------------
# 1. InferenceTask 边界值
# ---------------------------------------------------------------------------
class TestInferenceTaskBoundary:
    """InferenceTask 边界值测试。"""

    def test_max_tokens_zero(self):
        """max_tokens=0 是边界值，应能正常创建。"""
        task = InferenceTask(
            task_id="t0", model_id="m", prompt="hi",
            max_tokens=0, temperature=0.7,
        )
        assert task.max_tokens == 0

    def test_max_tokens_one(self):
        """max_tokens=1 是最小有意义的值。"""
        task = InferenceTask(
            task_id="t1", model_id="m", prompt="hi",
            max_tokens=1, temperature=0.7,
        )
        assert task.max_tokens == 1

    def test_max_tokens_very_large(self):
        """max_tokens 非常大时应能正常创建。"""
        task = InferenceTask(
            task_id="t2", model_id="m", prompt="hi",
            max_tokens=100000, temperature=0.7,
        )
        assert task.max_tokens == 100000

    def test_temperature_zero_deterministic(self):
        """temperature=0.0 表示确定性推理。"""
        task = InferenceTask(
            task_id="t3", model_id="m", prompt="hi",
            max_tokens=10, temperature=0.0,
        )
        assert task.temperature == 0.0

    def test_temperature_two_max_typical(self):
        """temperature=2.0 是典型最大值。"""
        task = InferenceTask(
            task_id="t4", model_id="m", prompt="hi",
            max_tokens=10, temperature=2.0,
        )
        assert task.temperature == 2.0

    def test_priority_negative(self):
        """priority 可以为负数。"""
        task = InferenceTask(
            task_id="t5", model_id="m", prompt="hi",
            max_tokens=10, temperature=0.7, priority=-5,
        )
        assert task.priority == -5

    def test_priority_very_large(self):
        """priority 非常大时应能正常创建。"""
        task = InferenceTask(
            task_id="t6", model_id="m", prompt="hi",
            max_tokens=10, temperature=0.7, priority=999999,
        )
        assert task.priority == 999999

    def test_stream_true(self):
        """stream=True 启用流式推理。"""
        task = InferenceTask(
            task_id="t7", model_id="m", prompt="hi",
            max_tokens=10, temperature=0.7, stream=True,
        )
        assert task.stream is True

    def test_empty_prompt(self):
        """空字符串 prompt 是边界值。"""
        task = InferenceTask(
            task_id="t8", model_id="m", prompt="",
            max_tokens=10, temperature=0.7,
        )
        assert task.prompt == ""

    def test_very_long_prompt(self):
        """超长 prompt（10000+ 字符）应能正常创建。"""
        long_prompt = "A" * 15000
        task = InferenceTask(
            task_id="t9", model_id="m", prompt=long_prompt,
            max_tokens=10, temperature=0.7,
        )
        assert len(task.prompt) == 15000

    def test_prompt_with_unicode_and_special_chars(self):
        """prompt 包含 Unicode 和特殊字符。"""
        prompt = "你好世界 🌍 éàü ñ \u00e9\U0001f600"
        task = InferenceTask(
            task_id="t10", model_id="m", prompt=prompt,
            max_tokens=10, temperature=0.7,
        )
        assert task.prompt == prompt

    def test_prompt_with_newlines_and_tabs(self):
        """prompt 包含换行符和制表符。"""
        prompt = "line1\nline2\ttab\r\nwindows"
        task = InferenceTask(
            task_id="t11", model_id="m", prompt=prompt,
            max_tokens=10, temperature=0.7,
        )
        assert task.prompt == prompt


# ---------------------------------------------------------------------------
# 2. TaskAcceptMessage 拒绝场景
# ---------------------------------------------------------------------------
class TestTaskAcceptMessageRejection:
    """TaskAcceptMessage 拒绝场景与边界值。"""

    def test_rejected_with_reason(self):
        """accepted=False 并附带拒绝原因。"""
        msg = TaskAcceptMessage(
            task_id="t1", node_id="w1",
            estimated_latency_ms=100.0, accepted=False,
        )
        assert msg.accepted is False

    def test_estimated_latency_zero(self):
        """estimated_latency_ms=0.0 是边界值。"""
        msg = TaskAcceptMessage(
            task_id="t1", node_id="w1",
            estimated_latency_ms=0.0,
        )
        assert msg.estimated_latency_ms == 0.0

    def test_estimated_latency_negative(self):
        """estimated_latency_ms 为负数（异常但应能序列化）。"""
        msg = TaskAcceptMessage(
            task_id="t1", node_id="w1",
            estimated_latency_ms=-10.0,
        )
        assert msg.estimated_latency_ms == -10.0

    def test_estimated_latency_very_large(self):
        """estimated_latency_ms 非常大。"""
        msg = TaskAcceptMessage(
            task_id="t1", node_id="w1",
            estimated_latency_ms=1e9,
        )
        assert msg.estimated_latency_ms == 1e9

    def test_rejection_roundtrip(self):
        """拒绝消息的 to_dict/from_dict 往返。"""
        msg = TaskAcceptMessage(
            task_id="t1", node_id="w1",
            estimated_latency_ms=200.0, accepted=False,
        )
        restored = TaskAcceptMessage.from_dict(msg.to_dict())
        assert restored.accepted is False
        assert restored.estimated_latency_ms == 200.0


# ---------------------------------------------------------------------------
# 3. TaskProgressMessage 边界场景
# ---------------------------------------------------------------------------
class TestTaskProgressMessageEdge:
    """TaskProgressMessage 边界场景。"""

    def test_tokens_generated_zero(self):
        """tokens_generated=0 表示尚未生成任何 token。"""
        msg = TaskProgressMessage(
            task_id="t1", node_id="w1",
            tokens_generated=0, tokens_per_second=0.0,
            progress_fraction=0.0,
        )
        assert msg.tokens_generated == 0

    def test_tokens_per_second_zero(self):
        """tokens_per_second=0.0 表示速度为零。"""
        msg = TaskProgressMessage(
            task_id="t1", node_id="w1",
            tokens_generated=10, tokens_per_second=0.0,
            progress_fraction=0.1,
        )
        assert msg.tokens_per_second == 0.0

    def test_progress_fraction_zero_just_started(self):
        """progress_fraction=0.0 表示刚启动。"""
        msg = TaskProgressMessage(
            task_id="t1", node_id="w1",
            tokens_generated=0, tokens_per_second=0.0,
            progress_fraction=0.0,
        )
        assert msg.progress_fraction == 0.0

    def test_progress_fraction_one_complete(self):
        """progress_fraction=1.0 表示完成。"""
        msg = TaskProgressMessage(
            task_id="t1", node_id="w1",
            tokens_generated=100, tokens_per_second=50.0,
            progress_fraction=1.0,
        )
        assert msg.progress_fraction == 1.0

    def test_progress_fraction_over_one(self):
        """progress_fraction>1.0 是边界情况，应仍能正常序列化。"""
        msg = TaskProgressMessage(
            task_id="t1", node_id="w1",
            tokens_generated=120, tokens_per_second=50.0,
            progress_fraction=1.2,
        )
        d = msg.to_dict()
        restored = TaskProgressMessage.from_dict(d)
        assert restored.progress_fraction == 1.2


# ---------------------------------------------------------------------------
# 4. TaskResultMessage 所有状态值
# ---------------------------------------------------------------------------
class TestTaskResultMessageAllStatus:
    """TaskResultMessage 遍历所有 TaskStatus。"""

    @pytest.mark.parametrize("status", list(TaskStatus))
    def test_roundtrip_each_status(self, status):
        """每种 TaskStatus 的 to_dict/from_dict 往返。"""
        msg = TaskResultMessage(
            task_id="t1", node_id="w1",
            status=status, output_text="out",
            tokens_generated=5, tokens_per_second=25.0,
            latency_ms=100.0,
        )
        restored = TaskResultMessage.from_dict(msg.to_dict())
        assert restored.status == status

    def test_error_default_empty_string(self):
        """error 字段默认为空字符串。"""
        msg = TaskResultMessage(
            task_id="t1", node_id="w1",
            status=TaskStatus.COMPLETED, output_text="ok",
            tokens_generated=5, tokens_per_second=25.0,
            latency_ms=100.0,
        )
        assert msg.error == ""

    def test_error_long_message(self):
        """error 字段包含长错误信息。"""
        long_error = "Error: " + "x" * 5000
        msg = TaskResultMessage(
            task_id="t1", node_id="w1",
            status=TaskStatus.FAILED, output_text="",
            tokens_generated=0, tokens_per_second=0.0,
            latency_ms=0.0, error=long_error,
        )
        restored = TaskResultMessage.from_dict(msg.to_dict())
        assert restored.error == long_error

    def test_failed_task_tokens_generated_zero(self):
        """失败任务的 tokens_generated 应为 0。"""
        msg = TaskResultMessage(
            task_id="t1", node_id="w1",
            status=TaskStatus.FAILED, output_text="",
            tokens_generated=0, tokens_per_second=0.0,
            latency_ms=50.0, error="OOM",
        )
        assert msg.tokens_generated == 0
        assert msg.status == TaskStatus.FAILED


# ---------------------------------------------------------------------------
# 5. TaskCancelMessage 边界场景
# ---------------------------------------------------------------------------
class TestTaskCancelMessageEdge:
    """TaskCancelMessage 边界场景。"""

    def test_reason_empty_default(self):
        """reason 默认为空字符串。"""
        msg = TaskCancelMessage(task_id="t1")
        assert msg.reason == ""

    def test_reason_detailed(self):
        """reason 包含详细取消信息。"""
        reason = "用户主动取消：任务超时 300s，资源不足 node-3"
        msg = TaskCancelMessage(task_id="t1", reason=reason)
        restored = TaskCancelMessage.from_dict(msg.to_dict())
        assert restored.reason == reason


# ---------------------------------------------------------------------------
# 6. 完整任务生命周期序列化
# ---------------------------------------------------------------------------
class TestFullLifecycleSerialization:
    """完整任务生命周期：创建 -> 分派 -> 接受 -> 进度 -> 结果，
    每步 to_dict/from_dict 保持所有字段。"""

    def test_lifecycle_roundtrip(self):
        """模拟完整任务生命周期，验证每一步序列化往返。"""
        # 1. 创建推理任务
        task = InferenceTask(
            task_id="life-001", model_id="qwen-7b",
            prompt="What is AI?", max_tokens=256,
            temperature=0.8, priority=3, stream=True,
        )

        # 2. 分派消息
        dispatch = TaskDispatchMessage(
            task=task, target_node_id="worker-1", instance_id="inst-001",
        )
        d_dispatch = dispatch.to_dict()
        r_dispatch = TaskDispatchMessage.from_dict(d_dispatch)
        assert r_dispatch.task.task_id == "life-001"
        assert r_dispatch.task.prompt == "What is AI?"
        assert r_dispatch.task.priority == 3
        assert r_dispatch.task.stream is True
        assert r_dispatch.target_node_id == "worker-1"
        assert r_dispatch.instance_id == "inst-001"

        # 3. 接受消息
        accept = TaskAcceptMessage(
            task_id="life-001", node_id="worker-1",
            estimated_latency_ms=150.0, accepted=True,
        )
        d_accept = accept.to_dict()
        r_accept = TaskAcceptMessage.from_dict(d_accept)
        assert r_accept.task_id == "life-001"
        assert r_accept.accepted is True
        assert r_accept.estimated_latency_ms == 150.0

        # 4. 进度消息
        progress = TaskProgressMessage(
            task_id="life-001", node_id="worker-1",
            tokens_generated=64, tokens_per_second=32.0,
            progress_fraction=0.25,
        )
        d_progress = progress.to_dict()
        r_progress = TaskProgressMessage.from_dict(d_progress)
        assert r_progress.tokens_generated == 64
        assert r_progress.tokens_per_second == 32.0
        assert r_progress.progress_fraction == 0.25

        # 5. 结果消息
        result = TaskResultMessage(
            task_id="life-001", node_id="worker-1",
            status=TaskStatus.COMPLETED, output_text="AI is ...",
            tokens_generated=256, tokens_per_second=32.0,
            latency_ms=8000.0,
        )
        d_result = result.to_dict()
        r_result = TaskResultMessage.from_dict(d_result)
        assert r_result.status == TaskStatus.COMPLETED
        assert r_result.output_text == "AI is ..."
        assert r_result.tokens_generated == 256
        assert r_result.error == ""


# ---------------------------------------------------------------------------
# 7. Frozen dataclass 不可变性验证
# ---------------------------------------------------------------------------
class TestFrozenDataclass:
    """所有消息类型应为不可变 dataclass。"""

    @pytest.mark.parametrize("cls,args,attr,val", [
        (
            InferenceTask,
            dict(task_id="t", model_id="m", prompt="p", max_tokens=1, temperature=0.5),
            "task_id", "changed",
        ),
        (
            TaskAcceptMessage,
            dict(task_id="t", node_id="w", estimated_latency_ms=1.0),
            "task_id", "changed",
        ),
        (
            TaskProgressMessage,
            dict(task_id="t", node_id="w", tokens_generated=1,
                 tokens_per_second=1.0, progress_fraction=0.5),
            "task_id", "changed",
        ),
        (
            TaskResultMessage,
            dict(task_id="t", node_id="w", status=TaskStatus.COMPLETED,
                 output_text="", tokens_generated=1, tokens_per_second=1.0, latency_ms=1.0),
            "task_id", "changed",
        ),
        (TaskCancelMessage, dict(task_id="t"), "task_id", "changed"),
    ])
    def test_frozen_immutable(self, cls, args, attr, val):
        """frozen dataclass 不允许修改属性。"""
        obj = cls(**args)
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(obj, attr, val)

    def test_task_dispatch_message_frozen_param(self):
        """TaskDispatchMessage frozen 不可修改。"""
        task = InferenceTask(task_id="t", model_id="m", prompt="p", max_tokens=1, temperature=0.5)
        msg = TaskDispatchMessage(task=task, target_node_id="w", instance_id="i")
        with pytest.raises(dataclasses.FrozenInstanceError):
            msg.target_node_id = "changed"

    def test_inference_task_frozen(self):
        """InferenceTask 不可修改。"""
        task = InferenceTask(task_id="t", model_id="m", prompt="p", max_tokens=1, temperature=0.5)
        with pytest.raises(dataclasses.FrozenInstanceError):
            task.task_id = "changed"

    def test_task_dispatch_message_frozen(self):
        """TaskDispatchMessage 不可修改。"""
        task = InferenceTask(task_id="t", model_id="m", prompt="p", max_tokens=1, temperature=0.5)
        msg = TaskDispatchMessage(task=task, target_node_id="w", instance_id="i")
        with pytest.raises(dataclasses.FrozenInstanceError):
            msg.target_node_id = "changed"

    def test_task_accept_message_frozen(self):
        """TaskAcceptMessage 不可修改。"""
        msg = TaskAcceptMessage(task_id="t", node_id="w", estimated_latency_ms=1.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            msg.accepted = False

    def test_task_progress_message_frozen(self):
        """TaskProgressMessage 不可修改。"""
        msg = TaskProgressMessage(
            task_id="t", node_id="w", tokens_generated=1,
            tokens_per_second=1.0, progress_fraction=0.5,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            msg.progress_fraction = 0.9

    def test_task_result_message_frozen(self):
        """TaskResultMessage 不可修改。"""
        msg = TaskResultMessage(
            task_id="t", node_id="w", status=TaskStatus.COMPLETED,
            output_text="", tokens_generated=1, tokens_per_second=1.0, latency_ms=1.0,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            msg.error = "changed"

    def test_task_cancel_message_frozen(self):
        """TaskCancelMessage 不可修改。"""
        msg = TaskCancelMessage(task_id="t")
        with pytest.raises(dataclasses.FrozenInstanceError):
            msg.reason = "changed"


# ---------------------------------------------------------------------------
# 8. from_dict 向后兼容
# ---------------------------------------------------------------------------
class TestFromDictBackwardCompatibility:
    """from_dict 在缺少可选字段时应使用默认值。"""

    def test_task_dispatch_missing_priority_and_stream(self):
        """TaskDispatchMessage.from_dict 缺少 priority 和 stream 时使用默认值。"""
        d = {
            "task": {
                "task_id": "t1",
                "model_id": "m1",
                "prompt": "hi",
                "max_tokens": 10,
                "temperature": 0.7,
                # priority 和 stream 缺失
            },
            "target_node_id": "w1",
            "instance_id": "i1",
        }
        msg = TaskDispatchMessage.from_dict(d)
        assert msg.task.priority == 0
        assert msg.task.stream is False

    def test_task_dispatch_missing_stream_only(self):
        """TaskDispatchMessage.from_dict 仅缺少 stream。"""
        d = {
            "task": {
                "task_id": "t1",
                "model_id": "m1",
                "prompt": "hi",
                "max_tokens": 10,
                "temperature": 0.7,
                "priority": 5,
            },
            "target_node_id": "w1",
            "instance_id": "i1",
        }
        msg = TaskDispatchMessage.from_dict(d)
        assert msg.task.priority == 5
        assert msg.task.stream is False

    def test_task_accept_missing_accepted(self):
        """TaskAcceptMessage.from_dict 缺少 accepted 时默认为 True。"""
        d = {
            "task_id": "t1",
            "node_id": "w1",
            "estimated_latency_ms": 100.0,
        }
        msg = TaskAcceptMessage.from_dict(d)
        assert msg.accepted is True

    def test_task_result_missing_error(self):
        """TaskResultMessage.from_dict 缺少 error 时默认为空字符串。"""
        d = {
            "task_id": "t1",
            "node_id": "w1",
            "status": "completed",
            "output_text": "ok",
            "tokens_generated": 5,
            "tokens_per_second": 25.0,
            "latency_ms": 100.0,
        }
        msg = TaskResultMessage.from_dict(d)
        assert msg.error == ""

    def test_task_cancel_missing_reason(self):
        """TaskCancelMessage.from_dict 缺少 reason 时默认为空字符串。"""
        d = {
            "task_id": "t1",
        }
        msg = TaskCancelMessage.from_dict(d)
        assert msg.reason == ""

    def test_task_dispatch_with_all_fields(self):
        """TaskDispatchMessage.from_dict 包含所有字段时正常工作。"""
        d = {
            "task": {
                "task_id": "t1",
                "model_id": "m1",
                "prompt": "hi",
                "max_tokens": 10,
                "temperature": 0.7,
                "priority": 10,
                "stream": True,
            },
            "target_node_id": "w1",
            "instance_id": "i1",
        }
        msg = TaskDispatchMessage.from_dict(d)
        assert msg.task.priority == 10
        assert msg.task.stream is True
