"""测试 Ollama 兼容 API 适配器。

适配器只负责格式转换，不涉及业务逻辑。
- Generate（非流式 + 流式）
- Chat（非流式 + 流式）
- Models 列表
- Tags（模型详情）
"""

from asc.api.ollama_adapter import (
    OllamaAdapter,
    OllamaChatRequest,
    OllamaChatResponse,
    OllamaGenerateRequest,
    OllamaGenerateResponse,
    OllamaModelInfo,
)


class TestOllamaGenerateRequest:
    """Generate 请求。"""

    def test_create(self):
        req = OllamaGenerateRequest(model="llama3", prompt="Hello")
        assert req.model == "llama3"
        assert req.prompt == "Hello"

    def test_defaults(self):
        req = OllamaGenerateRequest(model="m", prompt="Hi")
        assert req.stream is False
        assert req.options == {}


class TestOllamaGenerateResponse:
    """Generate 响应。"""

    def test_create(self):
        resp = OllamaGenerateResponse(
            model="llama3",
            response="Hi there!",
            done=True,
            prompt_eval_count=5,
            eval_count=10,
        )
        assert resp.response == "Hi there!"
        assert resp.done is True

    def test_to_dict(self):
        resp = OllamaGenerateResponse(
            model="llama3",
            response="Hello",
            done=True,
            prompt_eval_count=3,
            eval_count=7,
        )
        d = resp.to_dict()
        assert d["model"] == "llama3"
        assert d["response"] == "Hello"
        assert d["done"] is True
        assert d["prompt_eval_count"] == 3
        assert d["eval_count"] == 7


class TestOllamaChatRequest:
    """Chat 请求。"""

    def test_create(self):
        req = OllamaChatRequest(
            model="llama3",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert req.model == "llama3"
        assert len(req.messages) == 1

    def test_defaults(self):
        req = OllamaChatRequest(model="m", messages=[])
        assert req.stream is False
        assert req.options == {}


class TestOllamaChatResponse:
    """Chat 响应。"""

    def test_create(self):
        resp = OllamaChatResponse(
            model="llama3",
            message={"role": "assistant", "content": "Hi!"},
            done=True,
            prompt_eval_count=5,
            eval_count=10,
        )
        assert resp.message["content"] == "Hi!"
        assert resp.done is True

    def test_to_dict(self):
        resp = OllamaChatResponse(
            model="llama3",
            message={"role": "assistant", "content": "Hi!"},
            done=True,
            prompt_eval_count=5,
            eval_count=10,
        )
        d = resp.to_dict()
        assert d["message"]["content"] == "Hi!"
        assert d["done"] is True


class TestOllamaModelInfo:
    """模型信息。"""

    def test_create(self):
        info = OllamaModelInfo(name="llama3", size=4700000000)
        assert info.name == "llama3"
        assert info.size == 4700000000

    def test_to_dict(self):
        info = OllamaModelInfo(name="llama3", size=4700000000)
        d = info.to_dict()
        assert d["name"] == "llama3"
        assert d["size"] == 4700000000


class TestOllamaAdapter:
    """适配器主类。"""

    def test_create(self):
        adapter = OllamaAdapter(model_mappings={"llama3": "/models/llama.gguf"})
        assert "llama3" in adapter.model_mappings

    def test_list_models(self):
        adapter = OllamaAdapter(
            model_mappings={
                "llama3": "/models/llama.gguf",
                "qwen2.5": "/models/qwen.gguf",
            }
        )
        models = adapter.list_models()
        assert len(models) == 2
        names = [m.name for m in models]
        assert "llama3" in names
        assert "qwen2.5" in names

    def test_format_generate_prompt(self):
        adapter = OllamaAdapter(model_mappings={"m": "/m.gguf"})
        req = OllamaGenerateRequest(model="m", prompt="Hello")
        prompt = adapter.format_prompt_from_generate(req)
        assert "Hello" in prompt

    def test_format_chat_prompt(self):
        adapter = OllamaAdapter(model_mappings={"m": "/m.gguf"})
        req = OllamaChatRequest(
            model="m",
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
            ],
        )
        prompt = adapter.format_prompt_from_chat(req)
        assert "You are helpful." in prompt
        assert "Hi" in prompt

    def test_create_generate_response(self):
        adapter = OllamaAdapter(model_mappings={"m": "/m.gguf"})
        req = OllamaGenerateRequest(model="m", prompt="Hello")
        resp = adapter.create_generate_response(
            request=req,
            output="Hi there!",
            prompt_eval_count=5,
            eval_count=10,
        )
        assert resp.model == "m"
        assert resp.response == "Hi there!"
        assert resp.done is True

    def test_create_chat_response(self):
        adapter = OllamaAdapter(model_mappings={"m": "/m.gguf"})
        req = OllamaChatRequest(model="m", messages=[{"role": "user", "content": "Hi"}])
        resp = adapter.create_chat_response(
            request=req,
            output="Hello!",
            prompt_eval_count=3,
            eval_count=8,
        )
        assert resp.model == "m"
        assert resp.message["content"] == "Hello!"
        assert resp.done is True
