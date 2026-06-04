"""测试 OpenAI 兼容 API 适配器。

适配器只负责格式转换，不涉及业务逻辑。
- Chat Completions（流式 + 非流式）
- Models 列表
- Embeddings
"""

from asc.api.openai_adapter import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelInfo,
    OpenAIAdapter,
    TokenUsage,
    messages_to_prompt,
)


class TestMessagesToPrompt:
    """OpenAI messages 转 llama.cpp prompt。"""

    def test_single_user_message(self):
        messages = [{"role": "user", "content": "Hello"}]
        prompt = messages_to_prompt(messages)
        assert "Hello" in prompt

    def test_system_and_user(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        prompt = messages_to_prompt(messages)
        assert "You are helpful." in prompt
        assert "Hi" in prompt

    def test_conversation(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        prompt = messages_to_prompt(messages)
        assert "Hello" in prompt
        assert "Hi there!" in prompt
        assert "How are you?" in prompt


class TestChatCompletionRequest:
    """请求值对象。"""

    def test_create(self):
        req = ChatCompletionRequest(
            model="llama-3.1-8b",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=64,
            temperature=0.7,
            stream=False,
        )
        assert req.model == "llama-3.1-8b"
        assert len(req.messages) == 1
        assert req.max_tokens == 64
        assert req.stream is False

    def test_defaults(self):
        req = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert req.max_tokens == 128
        assert req.temperature == 0.7
        assert req.stream is False


class TestChatCompletionResponse:
    """响应值对象。"""

    def test_create(self):
        resp = ChatCompletionResponse(
            id="chatcmpl-123",
            model="llama-3.1-8b",
            content="Hi there!",
            usage=TokenUsage(prompt_tokens=5, completion_tokens=10, total_tokens=15),
            finish_reason="stop",
        )
        assert resp.content == "Hi there!"
        assert resp.usage.total_tokens == 15
        assert resp.finish_reason == "stop"

    def test_to_dict(self):
        resp = ChatCompletionResponse(
            id="chatcmpl-123",
            model="m",
            content="Hello",
            usage=TokenUsage(prompt_tokens=2, completion_tokens=3, total_tokens=5),
            finish_reason="stop",
        )
        d = resp.to_dict()
        assert d["object"] == "chat.completion"
        assert d["choices"][0]["message"]["content"] == "Hello"
        assert d["usage"]["total_tokens"] == 5


class TestChatCompletionChunk:
    """流式 chunk。"""

    def test_create(self):
        chunk = ChatCompletionChunk(
            id="chatcmpl-123",
            model="m",
            delta_content="Hi",
            finish_reason=None,
        )
        assert chunk.delta_content == "Hi"
        assert chunk.finish_reason is None

    def test_to_dict(self):
        chunk = ChatCompletionChunk(
            id="chatcmpl-123",
            model="m",
            delta_content="Hi",
            finish_reason=None,
        )
        d = chunk.to_dict()
        assert d["object"] == "chat.completion.chunk"
        assert d["choices"][0]["delta"]["content"] == "Hi"

    def test_final_chunk(self):
        chunk = ChatCompletionChunk(
            id="chatcmpl-123",
            model="m",
            delta_content="",
            finish_reason="stop",
        )
        d = chunk.to_dict()
        assert d["choices"][0]["finish_reason"] == "stop"


class TestModelInfo:
    """模型信息。"""

    def test_create(self):
        info = ModelInfo(id="llama-3.1-8b", owned_by="asc")
        assert info.id == "llama-3.1-8b"

    def test_to_dict(self):
        info = ModelInfo(id="llama-3.1-8b", owned_by="asc")
        d = info.to_dict()
        assert d["object"] == "model"
        assert d["id"] == "llama-3.1-8b"


class TestOpenAIAdapter:
    """适配器主类。"""

    def test_create(self):
        adapter = OpenAIAdapter(model_mappings={"llama-3.1-8b": "/models/llama.gguf"})
        assert "llama-3.1-8b" in adapter.model_mappings

    def test_list_models(self):
        adapter = OpenAIAdapter(
            model_mappings={
                "llama-3.1-8b": "/models/llama.gguf",
                "qwen-2.5-7b": "/models/qwen.gguf",
            }
        )
        models = adapter.list_models()
        assert len(models) == 2
        ids = [m.id for m in models]
        assert "llama-3.1-8b" in ids
        assert "qwen-2.5-7b" in ids

    def test_format_request(self):
        adapter = OpenAIAdapter(model_mappings={"m": "/m.gguf"})
        req = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "Hello"}],
        )
        prompt = adapter.format_prompt(req)
        assert "Hello" in prompt

    def test_format_request_unknown_model(self):
        adapter = OpenAIAdapter(model_mappings={"m": "/m.gguf"})
        req = ChatCompletionRequest(
            model="unknown",
            messages=[{"role": "user", "content": "Hello"}],
        )
        # 未知模型仍应返回 prompt（不做严格校验）
        prompt = adapter.format_prompt(req)
        assert "Hello" in prompt
