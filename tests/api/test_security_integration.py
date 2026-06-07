"""测试安全中间件（RateLimiter + InputValidator）集成到 API 服务器。"""

import os
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from asc.api.security import InputValidator, RateLimiter
from asc.api.server import create_app

# ---------------------------------------------------------------------------
# FastAPI 服务器安全集成测试
# ---------------------------------------------------------------------------


class TestFastAPIRateLimiting:
    """FastAPI 限流集成测试。"""

    def _make_client_with_limiter(self, max_requests: int = 3):
        """创建带低限流阈值的测试客户端。"""
        os.environ["ASC_ALLOW_NO_AUTH"] = "1"
        limiter = RateLimiter(max_requests=max_requests, window_seconds=60.0)
        app = create_app(model_mappings={"m": "/m.gguf"}, rate_limiter=limiter)
        return TestClient(app)

    def teardown_method(self):
        os.environ.pop("ASC_ALLOW_NO_AUTH", None)

    def test_requests_within_limit_succeed(self):
        """限流内的请求正常通过。"""
        client = self._make_client_with_limiter(max_requests=5)
        for _ in range(5):
            resp = client.get("/health")
            assert resp.status_code == 200

    def test_requests_beyond_limit_blocked(self):
        """超过限流阈值后返回 429。"""
        client = self._make_client_with_limiter(max_requests=2)
        # 前两次正常
        client.get("/health")
        client.get("/health")
        # 第三次被限流
        resp = client.get("/health")
        assert resp.status_code == 429
        assert "频繁" in resp.json()["detail"]

    def test_rate_limit_header_present(self):
        """响应中包含 X-RateLimit-Remaining 头。"""
        client = self._make_client_with_limiter(max_requests=10)
        resp = client.get("/health")
        assert "X-RateLimit-Remaining" in resp.headers

    def test_rate_limit_header_decreases(self):
        """X-RateLimit-Remaining 随请求递减。"""
        client = self._make_client_with_limiter(max_requests=5)
        resp1 = client.get("/health")
        remaining1 = int(resp1.headers["X-RateLimit-Remaining"])
        resp2 = client.get("/health")
        remaining2 = int(resp2.headers["X-RateLimit-Remaining"])
        assert remaining2 < remaining1


class TestFastAPIInputValidation:
    """FastAPI 输入验证集成测试。"""

    def _make_client(self, engine=None):
        """创建测试客户端。"""
        from unittest.mock import AsyncMock

        os.environ["ASC_ALLOW_NO_AUTH"] = "1"
        mock_engine = engine or MagicMock()
        mock_engine.submit.return_value = "test output"
        mock_engine.submit_async = AsyncMock(return_value="test output")

        # 模拟异步流式生成器
        async def mock_stream(request):
            for char in "test output":
                yield char

        mock_engine.submit_async_stream = mock_stream

        app = create_app(model_mappings={"m": "/m.gguf"}, engine=mock_engine)
        return TestClient(app)

    def teardown_method(self):
        os.environ.pop("ASC_ALLOW_NO_AUTH", None)

    def test_valid_request_passes(self):
        """合法请求通过验证并返回 200。"""
        client = self._make_client()
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 128,
                "temperature": 0.7,
            },
        )
        assert resp.status_code == 200

    def test_empty_prompt_rejected(self):
        """空 prompt 被拒绝（422）。"""
        client = self._make_client()
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": ""}],
            },
        )
        assert resp.status_code == 422
        assert "prompt" in resp.json()["detail"]

    def test_max_tokens_out_of_range_rejected(self):
        """max_tokens 超出范围被拒绝（422）。"""
        client = self._make_client()
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 99999,
            },
        )
        assert resp.status_code == 422
        assert "max_tokens" in resp.json()["detail"]

    def test_temperature_out_of_range_rejected(self):
        """temperature 超出范围被拒绝（422）。"""
        client = self._make_client()
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "Hi"}],
                "temperature": 5.0,
            },
        )
        assert resp.status_code == 422
        assert "temperature" in resp.json()["detail"]

    def test_top_p_out_of_range_rejected(self):
        """top_p 超出范围被拒绝（422）。"""
        client = self._make_client()
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "Hi"}],
                "top_p": 2.0,
            },
        )
        assert resp.status_code == 422
        assert "top_p" in resp.json()["detail"]

    def test_empty_model_rejected(self):
        """空 model 被拒绝（422）。"""
        client = self._make_client()
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        assert resp.status_code == 422

    def test_very_long_prompt_rejected(self):
        """超长 prompt 被拒绝（422）。"""
        client = self._make_client()
        long_content = "x" * (InputValidator.MAX_PROMPT_LENGTH + 1)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": long_content}],
            },
        )
        assert resp.status_code == 422
        assert "prompt" in resp.json()["detail"]

    def test_boundary_max_tokens_passes(self):
        """边界 max_tokens 值通过验证。"""
        client = self._make_client()
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": InputValidator.MAX_TOKENS_MAX,
            },
        )
        assert resp.status_code == 200

    def test_boundary_temperature_passes(self):
        """边界 temperature 值通过验证。"""
        client = self._make_client()
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "Hi"}],
                "temperature": InputValidator.TEMPERATURE_MAX,
            },
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Flask UI 服务器安全集成测试
# ---------------------------------------------------------------------------


class TestFlaskUIRateLimiting:
    """Flask UI 限流集成测试。"""

    @pytest.fixture(autouse=True)
    def _check_flask(self):
        pytest.importorskip("flask")

    def _make_client(self, max_requests: int = 3):
        """创建带低限流阈值的 Flask 测试客户端。"""
        from asc.ui.server import create_ui_app

        limiter = RateLimiter(max_requests=max_requests, window_seconds=60.0)
        app = create_ui_app(engine=None, limiter=limiter)
        app.config["TESTING"] = True
        return app.test_client()

    def test_requests_within_limit_succeed(self):
        """限流内的请求正常通过。"""
        client = self._make_client(max_requests=5)
        for _ in range(5):
            resp = client.get("/health")
            assert resp.status_code == 200

    def test_requests_beyond_limit_blocked(self):
        """超过限流阈值后返回 429。"""
        client = self._make_client(max_requests=2)
        client.get("/health")
        client.get("/health")
        resp = client.get("/health")
        assert resp.status_code == 429
        data = resp.get_json()
        assert data["success"] is False
        assert "频繁" in data["error"]


class TestFlaskUIInputValidation:
    """Flask UI 输入验证集成测试。"""

    @pytest.fixture(autouse=True)
    def _check_flask(self):
        pytest.importorskip("flask")

    def _make_client(self, engine=None):
        """创建 Flask 测试客户端。"""
        from asc.ui.server import create_ui_app

        mock_engine = engine or MagicMock()
        mock_engine.infer.return_value = {"success": True, "output": "test"}
        app = create_ui_app(engine=mock_engine)
        app.config["TESTING"] = True
        return app.test_client()

    def test_valid_request_passes(self):
        """合法请求通过验证。"""
        client = self._make_client()
        resp = client.post(
            "/api/infer",
            json={"prompt": "Hello", "max_tokens": 128, "temperature": 0.7},
        )
        assert resp.status_code == 200

    def test_empty_prompt_rejected(self):
        """空 prompt 被拒绝（422）。"""
        client = self._make_client()
        resp = client.post(
            "/api/infer",
            json={"prompt": "", "max_tokens": 128, "temperature": 0.7},
        )
        assert resp.status_code == 422
        data = resp.get_json()
        assert data["success"] is False
        assert "prompt" in data["error"]

    def test_max_tokens_out_of_range_rejected(self):
        """max_tokens 超出范围被拒绝（422）。"""
        client = self._make_client()
        resp = client.post(
            "/api/infer",
            json={"prompt": "Hi", "max_tokens": 99999, "temperature": 0.7},
        )
        assert resp.status_code == 422
        data = resp.get_json()
        assert data["success"] is False
        assert "max_tokens" in data["error"]

    def test_temperature_out_of_range_rejected(self):
        """temperature 超出范围被拒绝（422）。"""
        client = self._make_client()
        resp = client.post(
            "/api/infer",
            json={"prompt": "Hi", "max_tokens": 128, "temperature": 5.0},
        )
        assert resp.status_code == 422
        data = resp.get_json()
        assert data["success"] is False
        assert "temperature" in data["error"]

    def test_top_p_out_of_range_rejected(self):
        """top_p 超出范围被拒绝（422）。"""
        client = self._make_client()
        resp = client.post(
            "/api/infer",
            json={"prompt": "Hi", "max_tokens": 128, "temperature": 0.7, "top_p": 2.0},
        )
        assert resp.status_code == 422
        data = resp.get_json()
        assert data["success"] is False
        assert "top_p" in data["error"]

    def test_no_engine_returns_503(self):
        """无引擎时返回 503。"""
        from asc.ui.server import create_ui_app

        app = create_ui_app(engine=None)
        app.config["TESTING"] = True
        client = app.test_client()
        resp = client.post(
            "/api/infer",
            json={"prompt": "Hi", "max_tokens": 128, "temperature": 0.7},
        )
        assert resp.status_code == 503
