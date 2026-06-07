"""OpenAI 适配器增强测试。

覆盖审计中发现的关键缺口：
- ChatCompletionChunk.to_dict is_first=True（修复后包含 role）
- ChatCompletionChunk.to_dict is_first=False 且有 content
- ChatCompletionChunk.to_dict 有 finish_reason
- OpenAIAdapter.create_chunk is_first 参数
"""

from asc.api.openai_adapter import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    OpenAIAdapter,
)


class TestChatCompletionChunkToDict:
    """ChatCompletionChunk.to_dict 测试。"""

    def test_is_first_true_includes_role(self):
        """is_first=True 时 delta 应包含 role 字段。"""
        chunk = ChatCompletionChunk(
            id="chatcmpl-123",
            model="m",
            delta_content="Hello",
            is_first=True,
        )
        d = chunk.to_dict()
        assert d["choices"][0]["delta"]["role"] == "assistant"
        assert d["choices"][0]["delta"]["content"] == "Hello"

    def test_is_first_true_no_content(self):
        """is_first=True 且无 content 时 delta 应包含 role。"""
        chunk = ChatCompletionChunk(
            id="chatcmpl-123",
            model="m",
            delta_content="",
            is_first=True,
        )
        d = chunk.to_dict()
        assert d["choices"][0]["delta"]["role"] == "assistant"

    def test_is_first_false_with_content(self):
        """is_first=False 且有 content 时 delta 不应包含 role。"""
        chunk = ChatCompletionChunk(
            id="chatcmpl-123",
            model="m",
            delta_content="World",
            is_first=False,
        )
        d = chunk.to_dict()
        assert "role" not in d["choices"][0]["delta"]
        assert d["choices"][0]["delta"]["content"] == "World"

    def test_is_first_false_no_content_no_finish(self):
        """is_first=False、无 content、无 finish_reason 时 delta 应包含 role。"""
        chunk = ChatCompletionChunk(
            id="chatcmpl-123",
            model="m",
            delta_content="",
            is_first=False,
            finish_reason=None,
        )
        d = chunk.to_dict()
        # 空内容且非首 chunk 且无 finish_reason 时，delta 应包含 role
        assert d["choices"][0]["delta"]["role"] == "assistant"

    def test_with_finish_reason(self):
        """有 finish_reason 时 choice 应包含 finish_reason。"""
        chunk = ChatCompletionChunk(
            id="chatcmpl-123",
            model="m",
            delta_content="",
            finish_reason="stop",
        )
        d = chunk.to_dict()
        assert d["choices"][0]["finish_reason"] == "stop"

    def test_without_finish_reason(self):
        """无 finish_reason 时 choice 不应包含 finish_reason。"""
        chunk = ChatCompletionChunk(
            id="chatcmpl-123",
            model="m",
            delta_content="Hi",
            finish_reason=None,
        )
        d = chunk.to_dict()
        assert "finish_reason" not in d["choices"][0]

    def test_object_type(self):
        """object 字段应为 chat.completion.chunk。"""
        chunk = ChatCompletionChunk(
            id="chatcmpl-123",
            model="m",
            delta_content="Hi",
        )
        d = chunk.to_dict()
        assert d["object"] == "chat.completion.chunk"

    def test_is_first_true_with_finish_reason(self):
        """is_first=True 且有 finish_reason 时应同时包含 role 和 finish_reason。"""
        chunk = ChatCompletionChunk(
            id="chatcmpl-123",
            model="m",
            delta_content="",
            is_first=True,
            finish_reason="stop",
        )
        d = chunk.to_dict()
        assert d["choices"][0]["delta"]["role"] == "assistant"
        assert d["choices"][0]["finish_reason"] == "stop"


class TestOpenAIAdapterCreateChunk:
    """OpenAIAdapter.create_chunk 测试。"""

    def test_create_chunk_with_is_first(self):
        """create_chunk 应正确传递 is_first 参数。"""
        adapter = OpenAIAdapter(model_mappings={"m": "/m.gguf"})
        request = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "Hi"}],
        )

        chunk = adapter.create_chunk(
            request=request,
            delta="Hello",
            is_first=True,
        )

        assert chunk.is_first is True
        assert chunk.delta_content == "Hello"
        assert chunk.model == "m"

    def test_create_chunk_without_is_first(self):
        """不传 is_first 时默认为 False。"""
        adapter = OpenAIAdapter(model_mappings={"m": "/m.gguf"})
        request = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "Hi"}],
        )

        chunk = adapter.create_chunk(
            request=request,
            delta="World",
        )

        assert chunk.is_first is False

    def test_create_chunk_with_finish_reason(self):
        """create_chunk 应正确传递 finish_reason。"""
        adapter = OpenAIAdapter(model_mappings={"m": "/m.gguf"})
        request = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "Hi"}],
        )

        chunk = adapter.create_chunk(
            request=request,
            delta="",
            finish_reason="stop",
        )

        assert chunk.finish_reason == "stop"

    def test_create_chunk_id_format(self):
        """chunk id 应以 chatcmpl- 开头。"""
        adapter = OpenAIAdapter(model_mappings={"m": "/m.gguf"})
        request = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "Hi"}],
        )

        chunk = adapter.create_chunk(request=request, delta="Hi")

        assert chunk.id.startswith("chatcmpl-")
