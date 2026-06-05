"""测试请求批处理模块。"""

import time

from asc.engine.base import InferenceRequest, InferenceResult
from asc.engine.batch import BatchConfig, BatchedRequest, BatchProcessor


class TestBatchConfig:
    """BatchConfig 测试。"""

    def test_default_values(self):
        cfg = BatchConfig()
        assert cfg.max_batch_size == 8
        assert cfg.max_wait_ms == 10.0
        assert cfg.max_prompt_length == 32000

    def test_custom_values(self):
        cfg = BatchConfig(max_batch_size=16, max_wait_ms=50.0)
        assert cfg.max_batch_size == 16
        assert cfg.max_wait_ms == 50.0


class TestBatchProcessorSubmit:
    """提交请求测试。"""

    def test_submit_increases_pending(self):
        processor = BatchProcessor()
        assert processor.pending_count == 0
        processor.submit(InferenceRequest(prompt="hello"), model_id="m1")
        assert processor.pending_count == 1

    def test_submit_multiple(self):
        processor = BatchProcessor()
        for i in range(5):
            processor.submit(
                InferenceRequest(prompt=f"q{i}"), model_id="m1"
            )
        assert processor.pending_count == 5


class TestBatchProcessorShouldFlush:
    """should_flush 判断测试。"""

    def test_empty_not_flush(self):
        processor = BatchProcessor()
        assert processor.should_flush() is False

    def test_flush_when_max_batch_size_reached(self):
        processor = BatchProcessor(
            config=BatchConfig(max_batch_size=3, max_wait_ms=1000.0)
        )
        for i in range(3):
            processor.submit(
                InferenceRequest(prompt=f"q{i}"), model_id="m1"
            )
        assert processor.should_flush() is True

    def test_flush_when_wait_time_exceeded(self):
        processor = BatchProcessor(
            config=BatchConfig(max_batch_size=10, max_wait_ms=5.0)
        )
        processor.submit(InferenceRequest(prompt="hello"), model_id="m1")
        time.sleep(0.01)
        assert processor.should_flush() is True

    def test_not_flush_when_under_limit(self):
        processor = BatchProcessor(
            config=BatchConfig(max_batch_size=10, max_wait_ms=1000.0)
        )
        processor.submit(InferenceRequest(prompt="hello"), model_id="m1")
        assert processor.should_flush() is False


class TestBatchProcessorFlush:
    """flush 执行测试。"""

    def test_flush_empty_returns_empty(self):
        processor = BatchProcessor()
        results = processor.flush(lambda _reqs: [])
        assert results == []

    def test_flush_calls_infer_fn(self):
        processor = BatchProcessor()
        processor.submit(InferenceRequest(prompt="a"), model_id="m1")
        processor.submit(InferenceRequest(prompt="b"), model_id="m1")

        def infer_fn(reqs: list[InferenceRequest]) -> list[InferenceResult]:
            return [
                InferenceResult(
                    text=r.prompt.upper(),
                    tokens_generated=1,
                    tokens_per_second=10.0,
                )
                for r in reqs
            ]

        results = processor.flush(infer_fn)
        assert len(results) == 2
        assert results[0].text == "A"
        assert results[1].text == "B"
        assert processor.pending_count == 0

    def test_flush_respects_max_batch_size(self):
        processor = BatchProcessor(
            config=BatchConfig(max_batch_size=2, max_wait_ms=1000.0)
        )
        for i in range(5):
            processor.submit(
                InferenceRequest(prompt=f"q{i}"), model_id="m1"
            )

        def infer_fn(reqs: list[InferenceRequest]) -> list[InferenceResult]:
            return [
                InferenceResult(
                    text=r.prompt,
                    tokens_generated=1,
                    tokens_per_second=10.0,
                )
                for r in reqs
            ]

        results = processor.flush(infer_fn)
        assert len(results) == 2
        assert processor.pending_count == 3

    def test_flush_invokes_callbacks(self):
        processor = BatchProcessor()
        callbacks = []

        def cb(result: InferenceResult) -> None:
            callbacks.append(result.text)

        processor.submit(
            InferenceRequest(prompt="x"), model_id="m1", callback=cb
        )

        def infer_fn(reqs: list[InferenceRequest]) -> list[InferenceResult]:
            return [
                InferenceResult(
                    text=r.prompt * 2,
                    tokens_generated=1,
                    tokens_per_second=10.0,
                )
                for r in reqs
            ]

        processor.flush(infer_fn)
        assert callbacks == ["xx"]

    def test_flush_no_callback_ok(self):
        processor = BatchProcessor()
        processor.submit(InferenceRequest(prompt="y"), model_id="m1")

        def infer_fn(reqs: list[InferenceRequest]) -> list[InferenceResult]:
            return [
                InferenceResult(
                    text=r.prompt,
                    tokens_generated=1,
                    tokens_per_second=10.0,
                )
                for r in reqs
            ]

        results = processor.flush(infer_fn)
        assert len(results) == 1


class TestBatchProcessorGroupByModel:
    """按模型分组测试。"""

    def test_group_by_model(self):
        processor = BatchProcessor()
        processor.submit(InferenceRequest(prompt="a"), model_id="m1")
        processor.submit(InferenceRequest(prompt="b"), model_id="m2")
        processor.submit(InferenceRequest(prompt="c"), model_id="m1")

        groups = processor.group_by_model()
        assert set(groups.keys()) == {"m1", "m2"}
        assert len(groups["m1"]) == 2
        assert len(groups["m2"]) == 1
        assert [r.request.prompt for r in groups["m1"]] == ["a", "c"]

    def test_group_empty(self):
        processor = BatchProcessor()
        groups = processor.group_by_model()
        assert groups == {}


class TestBatchProcessorClear:
    """清空队列测试。"""

    def test_clear_returns_all(self):
        processor = BatchProcessor()
        processor.submit(InferenceRequest(prompt="a"), model_id="m1")
        processor.submit(InferenceRequest(prompt="b"), model_id="m2")

        cleared = processor.clear()
        assert len(cleared) == 2
        assert processor.pending_count == 0

    def test_clear_empty(self):
        processor = BatchProcessor()
        cleared = processor.clear()
        assert cleared == []


class TestBatchedRequest:
    """BatchedRequest 数据类测试。"""

    def test_default_submit_time(self):
        before = time.time()
        br = BatchedRequest(
            request=InferenceRequest(prompt="test"), model_id="m1"
        )
        after = time.time()
        assert before <= br.submit_time <= after

    def test_fields(self):
        req = InferenceRequest(prompt="hello", max_tokens=64)
        br = BatchedRequest(
            request=req, model_id="m2", callback=None
        )
        assert br.request == req
        assert br.model_id == "m2"
        assert br.callback is None
