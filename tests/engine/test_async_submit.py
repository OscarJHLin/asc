"""测试 LlamaServerEngine.submit_async 异步推理。"""

import asyncio
import time
from unittest.mock import MagicMock

import httpx
import pytest

from asc.engine.base import EngineStatus, InferenceRequest
from asc.engine.llama_server import LlamaServerEngine


def _make_mock_process():
    p = MagicMock()
    p.poll.return_value = None
    return p


def _mock_httpx_post_return(content: str = "Hello async"):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": content}}]
    }
    return mock_resp


def _make_ready_engine() -> LlamaServerEngine:
    mock_client = MagicMock()
    mock_resp = _mock_httpx_post_return("async result")
    mock_client.post.return_value = mock_resp
    engine = LlamaServerEngine(
        model_path="/m.gguf",
        base_url="http://127.0.0.1:8081",
        process=_make_mock_process(),
        http_client=mock_client,
    )
    engine._set_status(EngineStatus.READY)
    return engine


class TestSubmitAsyncReturnsSameResult:
    """submit_async 返回结果与 submit 一致。"""

    def test_submit_async_returns_text(self):
        engine = _make_ready_engine()
        request = InferenceRequest(prompt="Hi", max_tokens=32, temperature=0.5)

        result = asyncio.run(engine.submit_async(request))
        assert result == "async result"
        assert engine.status() == EngineStatus.READY

    def test_submit_async_and_submit_produce_same_result(self):
        mock_client = MagicMock()
        mock_client.post.return_value = _mock_httpx_post_return("same result")
        engine = LlamaServerEngine(
            model_path="/m.gguf",
            base_url="http://127.0.0.1:8081",
            process=_make_mock_process(),
            http_client=mock_client,
        )
        engine._set_status(EngineStatus.READY)
        request = InferenceRequest(prompt="Hi", max_tokens=32, temperature=0.5)

        sync_result = engine.submit(request)

        engine._set_status(EngineStatus.READY)
        async_result = asyncio.run(engine.submit_async(request))
        assert sync_result == async_result

    def test_submit_async_http_error(self):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500",
            request=MagicMock(),
            response=MagicMock(),
        )
        mock_client.post.return_value = mock_resp
        engine = LlamaServerEngine(
            model_path="/m.gguf",
            base_url="http://127.0.0.1:8081",
            process=_make_mock_process(),
            http_client=mock_client,
        )
        engine._set_status(EngineStatus.READY)
        request = InferenceRequest(prompt="Hi")

        with pytest.raises(RuntimeError, match="推理请求失败"):
            asyncio.run(engine.submit_async(request))
        assert engine.status() == EngineStatus.ERROR

    def test_submit_async_not_ready_raises(self):
        engine = LlamaServerEngine(
            model_path="/m.gguf",
            base_url="http://127.0.0.1:8081",
            process=_make_mock_process(),
            http_client=MagicMock(),
        )
        with pytest.raises(RuntimeError, match="无法提交请求"):
            asyncio.run(engine.submit_async(InferenceRequest(prompt="Hi")))


class TestSubmitAsyncDoesNotBlockEventLoop:
    """submit_async 不阻塞事件循环。"""

    def test_submit_async_allows_concurrent_tasks(self):
        """验证在 submit_async 等待期间，其他协程可以执行。"""
        mock_client = MagicMock()

        def slow_post(*args, **kwargs):
            call_times.append(time.monotonic())
            time.sleep(0.2)
            return _mock_httpx_post_return("done")

        mock_client.post.side_effect = slow_post
        engine = LlamaServerEngine(
            model_path="/m.gguf",
            base_url="http://127.0.0.1:8081",
            process=_make_mock_process(),
            http_client=mock_client,
        )
        engine._set_status(EngineStatus.READY)
        request = InferenceRequest(prompt="Hi")

        call_times: list[float] = []

        async def marker_task():
            await asyncio.sleep(0.05)
            call_times.append(time.monotonic())

        async def run():
            submit_task = asyncio.create_task(engine.submit_async(request))
            marker = asyncio.create_task(marker_task())
            await asyncio.gather(submit_task, marker)

        asyncio.run(run())

        # marker_task 应该在 slow_post 完成之前就执行了
        # call_times[0] = slow_post 开始, call_times[1] = marker_task 执行
        assert len(call_times) == 2
        assert call_times[1] < call_times[0] + 0.2


class TestSyncSubmitBackwardCompatibility:
    """同步 submit() 保持向后兼容。"""

    def test_submit_still_works(self):
        mock_client = MagicMock()
        mock_client.post.return_value = _mock_httpx_post_return("sync result")
        engine = LlamaServerEngine(
            model_path="/m.gguf",
            base_url="http://127.0.0.1:8081",
            process=_make_mock_process(),
            http_client=mock_client,
        )
        engine._set_status(EngineStatus.READY)
        request = InferenceRequest(prompt="Hi", max_tokens=32, temperature=0.5)

        result = engine.submit(request)
        assert result == "sync result"
        assert engine.status() == EngineStatus.READY

    def test_submit_not_ready_raises(self):
        engine = LlamaServerEngine(
            model_path="/m.gguf",
            base_url="http://127.0.0.1:8081",
            process=_make_mock_process(),
            http_client=MagicMock(),
        )
        with pytest.raises(RuntimeError, match="无法提交请求"):
            engine.submit(InferenceRequest(prompt="Hi"))
