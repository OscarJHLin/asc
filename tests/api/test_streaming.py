"""测试 SSE 流式推理。"""

import json
import os
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from asc.api.auth import _init_default_store
from asc.api.openai_adapter import ChatCompletionRequest, OpenAIAdapter
from asc.api.server import _stream_chat, create_app

# --- _stream_chat SSE 格式测试 ---


class TestStreamChatSSEFormat:
    """SSE 格式输出。"""

    def _make_request(self, **kwargs):
        defaults = {
            "model": "m",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 32,
            "temperature": 0.7,
            "stream": True,
        }
        defaults.update(kwargs)
        return ChatCompletionRequest(**defaults)

    def _collect_stream(self, gen):
        """收集异步生成器的所有输出。"""
        import asyncio

        chunks = []

        async def _collect():
            async for chunk in gen:
                chunks.append(chunk)

        asyncio.run(_collect())
        return chunks

    def _make_async_engine(self, return_value="Hello"):
        """创建支持 await 的 mock 引擎。"""
        mock_engine = MagicMock()
        mock_engine.submit.return_value = return_value
        mock_engine.submit_async = AsyncMock(return_value=return_value)

        # 模拟异步流式生成器
        async def mock_stream(request):
            for char in return_value:
                yield char

        mock_engine.submit_async_stream = mock_stream
        return mock_engine

    def test_sse_data_prefix(self):
        """每个 chunk 以 'data: ' 开头。"""
        adapter = OpenAIAdapter()
        request = self._make_request()
        mock_engine = self._make_async_engine("Hello")

        chunks = self._collect_stream(_stream_chat(adapter, request, "Hi", mock_engine))
        for chunk in chunks[:-1]:  # 排除 [DONE]
            if "[DONE]" not in chunk:
                assert chunk.startswith("data: ")

    def test_sse_double_newline(self):
        """每个 chunk 以双换行结尾。"""
        adapter = OpenAIAdapter()
        request = self._make_request()
        mock_engine = self._make_async_engine("Hello")

        chunks = self._collect_stream(_stream_chat(adapter, request, "Hi", mock_engine))
        for chunk in chunks:
            assert chunk.endswith("\n\n")

    def test_sse_json_format(self):
        """data 行的 payload 是合法 JSON，包含 object=chat.completion.chunk。"""
        adapter = OpenAIAdapter()
        request = self._make_request()
        mock_engine = self._make_async_engine("Hello")

        chunks = self._collect_stream(_stream_chat(adapter, request, "Hi", mock_engine))
        for chunk in chunks:
            if chunk.strip() == "data: [DONE]":
                continue
            payload_str = chunk.removeprefix("data: ").removesuffix("\n\n")
            payload = json.loads(payload_str)
            assert payload["object"] == "chat.completion.chunk"
            assert "choices" in payload

    def test_done_terminator(self):
        """最后一个 chunk 是 'data: [DONE]\n\n'。"""
        adapter = OpenAIAdapter()
        request = self._make_request()
        mock_engine = self._make_async_engine("Hello")

        chunks = self._collect_stream(_stream_chat(adapter, request, "Hi", mock_engine))
        assert chunks[-1] == "data: [DONE]\n\n"

    def test_finish_reason_stop(self):
        """倒数第二个 chunk 包含 finish_reason='stop'。"""
        adapter = OpenAIAdapter()
        request = self._make_request()
        mock_engine = self._make_async_engine("Hello")

        chunks = self._collect_stream(_stream_chat(adapter, request, "Hi", mock_engine))
        # 倒数第二个（[DONE] 之前）应该是 finish_reason=stop
        final_data_chunk = chunks[-2]
        payload_str = final_data_chunk.removeprefix("data: ").removesuffix("\n\n")
        payload = json.loads(payload_str)
        assert payload["choices"][0]["finish_reason"] == "stop"

    def test_content_chunks_have_no_finish_reason(self):
        """内容 chunk 的 finish_reason 为 null。"""
        adapter = OpenAIAdapter()
        request = self._make_request()
        mock_engine = self._make_async_engine("Hello world")

        chunks = self._collect_stream(_stream_chat(adapter, request, "Hi", mock_engine))
        # 排除 final chunk 和 [DONE]
        for chunk in chunks[:-2]:
            payload_str = chunk.removeprefix("data: ").removesuffix("\n\n")
            payload = json.loads(payload_str)
            assert payload["choices"][0].get("finish_reason") is None

    def test_engine_called_with_correct_params(self):
        """引擎使用正确的参数调用。"""
        adapter = OpenAIAdapter()
        request = self._make_request(max_tokens=64, temperature=0.3)
        mock_engine = MagicMock()
        mock_engine.submit.return_value = "Hi"
        mock_engine.submit_async = AsyncMock(return_value="Hi")

        # 使用 MagicMock 模拟异步流式生成器，以便追踪调用
        async def mock_stream(request):
            yield "H"
            yield "i"

        mock_engine.submit_async_stream = MagicMock(side_effect=mock_stream)

        self._collect_stream(_stream_chat(adapter, request, "test prompt", mock_engine))
        # 验证 submit_async_stream 被调用了一次
        assert mock_engine.submit_async_stream.call_count == 1
        call_args = mock_engine.submit_async_stream.call_args[0][0]
        assert call_args.prompt == "test prompt"
        assert call_args.max_tokens == 64
        assert call_args.temperature == 0.3

    def test_cjk_streaming_char_by_char(self):
        """CJK 文本逐字输出。"""
        adapter = OpenAIAdapter()
        request = self._make_request()
        mock_engine = self._make_async_engine("你好世界")

        chunks = self._collect_stream(_stream_chat(adapter, request, "Hi", mock_engine))
        # 排除 final chunk 和 [DONE]，提取内容
        content_chunks = []
        for chunk in chunks[:-2]:
            payload_str = chunk.removeprefix("data: ").removesuffix("\n\n")
            payload = json.loads(payload_str)
            delta = payload["choices"][0]["delta"]
            if "content" in delta:
                content_chunks.append(delta["content"])

        # CJK 应该逐字输出
        assert content_chunks == ["你", "好", "世", "界"]

    def test_first_chunk_includes_role(self):
        """第一个 SSE chunk 包含 role: 'assistant'（OpenAI 标准兼容）。"""
        adapter = OpenAIAdapter()
        request = self._make_request()
        mock_engine = self._make_async_engine("Hello")

        chunks = self._collect_stream(_stream_chat(adapter, request, "Hi", mock_engine))
        # 第一个 chunk 应包含 role: "assistant"
        first_payload_str = chunks[0].removeprefix("data: ").removesuffix("\n\n")
        first_payload = json.loads(first_payload_str)
        delta = first_payload["choices"][0]["delta"]
        assert delta.get("role") == "assistant"
        assert "content" in delta

        # 后续 chunk 不应包含 role
        for chunk in chunks[1:-2]:  # 排除第一个、final chunk 和 [DONE]
            payload_str = chunk.removeprefix("data: ").removesuffix("\n\n")
            payload = json.loads(payload_str)
            delta = payload["choices"][0]["delta"]
            assert "role" not in delta


# --- 通过 HTTP 端点集成测试 ---


class TestStreamingEndpoint:
    """通过 FastAPI TestClient 测试流式端点。"""

    def _make_client(self, **kwargs):
        os.environ["ASC_API_KEY"] = "test-key"
        _init_default_store()
        app = create_app(**kwargs)
        return TestClient(app, headers={"X-API-Key": "test-key"})

    def _make_async_engine(self, return_value="Hello"):
        """创建支持 await 的 mock 引擎。"""
        mock_engine = MagicMock()
        mock_engine.submit.return_value = return_value
        mock_engine.submit_async = AsyncMock(return_value=return_value)

        # 模拟异步流式生成器
        async def mock_stream(request):
            for char in return_value:
                yield char

        mock_engine.submit_async_stream = mock_stream
        return mock_engine

    def test_streaming_endpoint_returns_200(self):
        """流式端点返回 200。"""
        mock_engine = self._make_async_engine("Hello")
        client = self._make_client(model_mappings={"m": "/m.gguf"}, engine=mock_engine)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200

    def test_streaming_endpoint_sse_format(self):
        """流式端点输出 SSE 格式。"""
        mock_engine = self._make_async_engine("Hello world")
        client = self._make_client(model_mappings={"m": "/m.gguf"}, engine=mock_engine)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )
        text = resp.text
        assert "data:" in text
        assert "[DONE]" in text

        # 解析 SSE 行
        lines = [line for line in text.strip().split("\n") if line.startswith("data:")]
        assert len(lines) >= 3  # 至少: 内容 + stop + [DONE]

        # 最后一行是 [DONE]
        assert lines[-1] == "data: [DONE]"

    def test_streaming_endpoint_content_reconstructable(self):
        """流式输出的内容可以重新拼接为完整文本。"""
        mock_engine = self._make_async_engine("Hello world")
        client = self._make_client(model_mappings={"m": "/m.gguf"}, engine=mock_engine)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )
        text = resp.text

        # 提取所有内容 chunk 的 delta content
        content_parts = []
        for line in text.strip().split("\n"):
            if line.startswith("data: ") and line != "data: [DONE]":
                payload_str = line.removeprefix("data: ")
                payload = json.loads(payload_str)
                delta = payload["choices"][0]["delta"]
                if "content" in delta:
                    content_parts.append(delta["content"])

        reconstructed = "".join(content_parts)
        assert reconstructed == "Hello world"

    def test_streaming_with_cjk_content(self):
        """流式端点处理 CJK 内容。"""
        mock_engine = self._make_async_engine("你好世界")
        client = self._make_client(model_mappings={"m": "/m.gguf"}, engine=mock_engine)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "你好"}],
                "stream": True,
            },
        )
        text = resp.text
        assert "你好世界" in "".join(
            json.loads(line.removeprefix("data: "))["choices"][0]["delta"].get("content", "")
            for line in text.strip().split("\n")
            if line.startswith("data: ") and line != "data: [DONE]"
        )
