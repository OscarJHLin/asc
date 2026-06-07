"""流式推理增强测试。

覆盖审计中发现的关键缺口：
- _sync_infer_stream 方法（mock HTTP 流式响应）
- submit_async_stream 方法的 sentinel-based StopIteration 处理
- 流式响应中的 [DONE] 标记
- 流式响应中的 JSON 解析错误（应被跳过）
- 流式响应中的 HTTP 错误
"""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from asc.engine.base import EngineStatus, InferenceRequest
from asc.engine.llama_server import LlamaServerEngine


def _make_engine(status: EngineStatus = EngineStatus.READY) -> LlamaServerEngine:
    """创建一个用于测试的 LlamaServerEngine 实例。"""
    mock_process = MagicMock()
    mock_process.poll.return_value = None
    mock_client = MagicMock()
    engine = LlamaServerEngine(
        model_path="/models/test.gguf",
        base_url="http://127.0.0.1:8081",
        process=mock_process,
        http_client=mock_client,
    )
    engine._status = status
    return engine


class TestSyncInferStream:
    """_sync_infer_stream 方法测试。"""

    def test_stream_with_valid_tokens(self):
        """正常流式响应应逐个 yield token。"""
        engine = _make_engine()

        lines = [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            'data: {"choices":[{"delta":{"content":" World"}}]}',
            'data: [DONE]',
        ]

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.iter_lines.return_value = lines

        @contextmanager
        def mock_stream(*args, **kwargs):
            yield mock_response

        engine.http_client.stream = mock_stream

        request = InferenceRequest(prompt="Hi", max_tokens=10)
        tokens = list(engine._sync_infer_stream(request))

        assert tokens == ["Hello", " World"]
        assert engine._status == EngineStatus.READY

    def test_stream_with_done_marker(self):
        """遇到 [DONE] 标记应停止迭代。"""
        engine = _make_engine()

        lines = [
            'data: {"choices":[{"delta":{"content":"Hi"}}]}',
            'data: [DONE]',
            'data: {"choices":[{"delta":{"content":"Should not appear"}}]}',
        ]

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.iter_lines.return_value = lines

        @contextmanager
        def mock_stream(*args, **kwargs):
            yield mock_response

        engine.http_client.stream = mock_stream

        request = InferenceRequest(prompt="Hi", max_tokens=10)
        tokens = list(engine._sync_infer_stream(request))

        assert tokens == ["Hi"]

    def test_stream_with_json_parse_errors(self):
        """JSON 解析错误的行应被跳过。"""
        engine = _make_engine()

        lines = [
            'data: {"choices":[{"delta":{"content":"OK"}}]}',
            'data: not valid json',
            'data: {"choices":[]}',  # 缺少 delta
            'data: {"choices":[{"delta":{}}]}',  # delta 无 content
            'data: {"choices":[{"delta":{"content":"End"}}]}',
            'data: [DONE]',
        ]

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.iter_lines.return_value = lines

        @contextmanager
        def mock_stream(*args, **kwargs):
            yield mock_response

        engine.http_client.stream = mock_stream

        request = InferenceRequest(prompt="Hi", max_tokens=10)
        tokens = list(engine._sync_infer_stream(request))

        assert tokens == ["OK", "End"]

    def test_stream_with_http_error(self):
        """HTTP 错误应设置引擎状态为 ERROR 并抛出 RuntimeError。"""
        engine = _make_engine()

        @contextmanager
        def mock_stream(*args, **kwargs):
            raise httpx.HTTPStatusError(
                "Server Error",
                request=MagicMock(),
                response=MagicMock(status_code=500),
            )
            yield  # never reached

        import httpx
        engine.http_client.stream = mock_stream

        request = InferenceRequest(prompt="Hi", max_tokens=10)
        with pytest.raises(RuntimeError, match="流式推理请求失败"):
            list(engine._sync_infer_stream(request))

        assert engine._status == EngineStatus.ERROR

    def test_stream_ignores_non_data_lines(self):
        """非 data: 开头的行应被忽略。"""
        engine = _make_engine()

        lines = [
            '',
            ': comment',
            'data: {"choices":[{"delta":{"content":"Token"}}]}',
            'data: [DONE]',
        ]

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.iter_lines.return_value = lines

        @contextmanager
        def mock_stream(*args, **kwargs):
            yield mock_response

        engine.http_client.stream = mock_stream

        request = InferenceRequest(prompt="Hi", max_tokens=10)
        tokens = list(engine._sync_infer_stream(request))

        assert tokens == ["Token"]


class TestSubmitAsyncStream:
    """submit_async_stream 方法测试。"""

    @pytest.mark.asyncio
    async def test_async_stream_yields_tokens(self):
        """异步流式推理应逐个 yield token。"""
        engine = _make_engine()

        def fake_sync_stream(request):
            yield "Hello"
            yield " World"

        with patch.object(engine, "_sync_infer_stream", side_effect=fake_sync_stream):
            request = InferenceRequest(prompt="Hi", max_tokens=10)
            tokens = []
            async for token in engine.submit_async_stream(request):
                tokens.append(token)

        assert tokens == ["Hello", " World"]
        assert engine._status == EngineStatus.READY

    @pytest.mark.asyncio
    async def test_async_stream_sentinel_stopiteration(self):
        """生成器耗尽时 sentinel 应正确处理 StopIteration。"""
        engine = _make_engine()

        def fake_sync_stream(request):
            yield "Only"
            # 生成器自然结束

        with patch.object(engine, "_sync_infer_stream", side_effect=fake_sync_stream):
            request = InferenceRequest(prompt="Hi", max_tokens=10)
            tokens = []
            async for token in engine.submit_async_stream(request):
                tokens.append(token)

        assert tokens == ["Only"]
        assert engine._status == EngineStatus.READY

    @pytest.mark.asyncio
    async def test_async_stream_empty_generator(self):
        """空生成器应正常结束，不抛出异常。"""
        engine = _make_engine()

        def fake_sync_stream(request):
            return
            yield  # make it a generator

        with patch.object(engine, "_sync_infer_stream", side_effect=fake_sync_stream):
            request = InferenceRequest(prompt="Hi", max_tokens=10)
            tokens = []
            async for token in engine.submit_async_stream(request):
                tokens.append(token)

        assert tokens == []
        assert engine._status == EngineStatus.READY

    @pytest.mark.asyncio
    async def test_async_stream_error_sets_status(self):
        """流式推理出错时应设置 ERROR 状态并抛出 RuntimeError。"""
        engine = _make_engine()

        def fake_sync_stream(request):
            raise RuntimeError("Connection lost")
            yield  # make it a generator

        with patch.object(engine, "_sync_infer_stream", side_effect=fake_sync_stream):
            request = InferenceRequest(prompt="Hi", max_tokens=10)
            with pytest.raises(RuntimeError, match="流式推理失败"):
                async for _ in engine.submit_async_stream(request):
                    pass

        assert engine._status == EngineStatus.ERROR

    @pytest.mark.asyncio
    async def test_async_stream_rejects_invalid_status(self):
        """引擎状态不合法时应抛出 RuntimeError。"""
        engine = _make_engine(status=EngineStatus.IDLE)

        request = InferenceRequest(prompt="Hi", max_tokens=10)
        with pytest.raises(RuntimeError, match="引擎状态为"):
            async for _ in engine.submit_async_stream(request):
                pass
