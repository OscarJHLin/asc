"""测试推理引擎抽象接口和 llama-server 管理。

核心改进：从每次推理启动新进程 -> llama-server 常驻进程 + HTTP API。
"""

from unittest.mock import MagicMock

from asc.engine.base import (
    Engine,
    EngineBuilder,
    EngineStatus,
    InferenceRequest,
    InferenceResult,
    LoadProgress,
)
from asc.engine.llama_server import LlamaServerBuilder, LlamaServerEngine


class TestInferenceRequest:
    """推理请求值对象。"""

    def test_create(self):
        req = InferenceRequest(prompt="Hello", max_tokens=64, temperature=0.7)
        assert req.prompt == "Hello"
        assert req.max_tokens == 64
        assert req.temperature == 0.7

    def test_defaults(self):
        req = InferenceRequest(prompt="Hi")
        assert req.max_tokens == 128
        assert req.temperature == 0.7

    def test_frozen(self):
        req = InferenceRequest(prompt="Hi")
        try:
            req.prompt = "Bye"  # type: ignore[misc]
            raise AssertionError("Should be immutable")
        except (AttributeError, TypeError):
            pass


class TestInferenceResult:
    """推理结果值对象。"""

    def test_success_result(self):
        result = InferenceResult(text="World", tokens_generated=5, tokens_per_second=30.0)
        assert result.text == "World"
        assert result.tokens_generated == 5
        assert result.tokens_per_second == 30.0
        assert result.error is None

    def test_error_result(self):
        result = InferenceResult(text="", tokens_generated=0, tokens_per_second=0.0, error="OOM")
        assert result.error == "OOM"

    def test_frozen(self):
        result = InferenceResult(text="x", tokens_generated=1, tokens_per_second=1.0)
        try:
            result.text = "y"  # type: ignore[misc]
            raise AssertionError("Should be immutable")
        except (AttributeError, TypeError):
            pass


class TestLoadProgress:
    """模型加载进度。"""

    def test_create(self):
        p = LoadProgress(current=5, total=10, message="Loading layer 5/10")
        assert p.current == 5
        assert p.total == 10
        assert p.message == "Loading layer 5/10"

    def test_progress_fraction(self):
        p = LoadProgress(current=5, total=10, message="")
        assert p.fraction == 0.5

    def test_progress_complete(self):
        p = LoadProgress(current=10, total=10, message="Done")
        assert p.fraction == 1.0


class TestEngineStatus:
    """引擎状态枚举。"""

    def test_statuses(self):
        assert EngineStatus.IDLE.value == "idle"
        assert EngineStatus.LOADING.value == "loading"
        assert EngineStatus.READY.value == "ready"
        assert EngineStatus.RUNNING.value == "running"
        assert EngineStatus.ERROR.value == "error"
        assert EngineStatus.SHUTDOWN.value == "shutdown"


class TestEngineBuilderAbstract:
    """EngineBuilder 抽象接口验证。"""

    def test_builder_is_abstract(self):
        """EngineBuilder 不能直接实例化（抽象方法未实现）。"""
        # EngineBuilder 是 Protocol 或 ABC，这里验证接口存在
        assert hasattr(EngineBuilder, "load")
        assert hasattr(EngineBuilder, "build")


class TestEngineAbstract:
    """Engine 抽象接口验证。"""

    def test_engine_has_submit(self):
        assert hasattr(Engine, "submit")

    def test_engine_has_step(self):
        assert hasattr(Engine, "step")

    def test_engine_has_close(self):
        assert hasattr(Engine, "close")


class TestLlamaServerBuilder:
    """LlamaServerBuilder 测试。"""

    def test_create_with_model_path(self):
        builder = LlamaServerBuilder(
            model_path="/path/to/model.gguf",
            host="127.0.0.1",
            port=8081,
        )
        assert builder.model_path == "/path/to/model.gguf"
        assert builder.host == "127.0.0.1"
        assert builder.port == 8081

    def test_default_host_port(self):
        builder = LlamaServerBuilder(model_path="/path/to/model.gguf")
        assert builder.host == "127.0.0.1"
        assert builder.port == 8081


class TestLlamaServerEngine:
    """LlamaServerEngine 测试。"""

    @staticmethod
    def _make_mock_process():
        p = MagicMock()
        p.poll.return_value = None  # 进程仍在运行
        return p

    def test_create(self):
        engine = LlamaServerEngine(
            model_path="/path/to/model.gguf",
            base_url="http://127.0.0.1:8081",
            process=self._make_mock_process(),
        )
        assert engine.model_path == "/path/to/model.gguf"
        assert engine.base_url == "http://127.0.0.1:8081"
        assert engine.status() == EngineStatus.IDLE

    def test_status_transitions(self):
        engine = LlamaServerEngine(
            model_path="/path/to/model.gguf",
            base_url="http://127.0.0.1:8081",
            process=self._make_mock_process(),
        )
        assert engine.status() == EngineStatus.IDLE
        engine._set_status(EngineStatus.READY)
        assert engine.status() == EngineStatus.READY
