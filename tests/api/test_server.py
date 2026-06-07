"""测试 FastAPI 服务器 + 认证。"""

import os
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from asc.api.auth import require_api_key
from asc.api.server import create_app


class TestAuthMiddleware:
    """API Key 认证。"""

    def test_no_key_configured_rejects(self):
        """S-02/S-03: 未配置 API Key 时拒绝请求。"""
        os.environ.pop("ASC_API_KEY", None)
        os.environ.pop("ASC_ALLOW_NO_AUTH", None)
        assert require_api_key(api_key=None) is False

    def test_no_key_with_allow_no_auth(self):
        """ASC_ALLOW_NO_AUTH=1 时允许免认证。"""
        os.environ.pop("ASC_API_KEY", None)
        os.environ["ASC_ALLOW_NO_AUTH"] = "1"
        try:
            assert require_api_key(api_key=None) is True
        finally:
            del os.environ["ASC_ALLOW_NO_AUTH"]

    def test_valid_key_passes(self):
        """正确的 API Key 通过。"""
        os.environ["ASC_API_KEY"] = "test-key-123"
        try:
            assert require_api_key(api_key="test-key-123") is True
        finally:
            del os.environ["ASC_API_KEY"]

    def test_invalid_key_fails(self):
        """错误的 API Key 被拒绝。"""
        os.environ["ASC_API_KEY"] = "correct-key"
        try:
            assert require_api_key(api_key="wrong-key") is False
        finally:
            del os.environ["ASC_API_KEY"]

    def test_missing_key_fails(self):
        """需要 key 但未提供时被拒绝。"""
        os.environ["ASC_API_KEY"] = "required-key"
        try:
            assert require_api_key(api_key=None) is False
        finally:
            del os.environ["ASC_API_KEY"]


class TestFastAPIApp:
    """FastAPI 应用。"""

    def _make_client_with_auth(self, **kwargs):
        """创建带认证的测试客户端。"""
        os.environ["ASC_ALLOW_NO_AUTH"] = "1"
        app = create_app(**kwargs)
        client = TestClient(app)
        return client

    def test_create_app(self):
        app = create_app()
        assert app is not None
        assert app.title == "Asc API"

    def test_health_endpoint(self):
        app = create_app()
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_openai_models_endpoint(self):
        app = create_app(model_mappings={"llama-3.1-8b": "/models/llama.gguf"})
        client = TestClient(app)
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1

    def test_openai_chat_completions_endpoint(self):
        """Chat completions 端点存在（不测试实际推理）。"""
        client = self._make_client_with_auth(model_mappings={"m": "/m.gguf"})
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        assert resp.status_code in (200, 503)

    def test_admin_nodes_endpoint(self):
        """S-05: 管理端点需要管理员认证。"""
        # 无认证时应返回 403（管理端点需要管理员权限）
        os.environ.pop("ASC_API_KEY", None)
        os.environ.pop("ASC_ALLOW_NO_AUTH", None)
        os.environ.pop("ASC_ADMIN_API_KEY", None)
        app = create_app()
        client = TestClient(app)
        resp = client.get("/admin/nodes")
        assert resp.status_code == 403

        # 有普通 API Key 但无管理员 Key 时，回退到普通 Key 验证（兼容模式）
        os.environ["ASC_API_KEY"] = "test-key"
        try:
            app = create_app()
            client = TestClient(app)
            resp = client.get("/admin/nodes", headers={"X-API-Key": "test-key"})
            assert resp.status_code == 200
        finally:
            del os.environ["ASC_API_KEY"]

        # 有管理员 Key 时，使用管理员 Key 验证
        os.environ["ASC_ADMIN_API_KEY"] = "admin-key"
        try:
            app = create_app()
            client = TestClient(app)
            # 普通 Key 被拒绝
            resp = client.get("/admin/nodes", headers={"X-API-Key": "wrong-key"})
            assert resp.status_code == 403
            # 管理员 Key 通过
            resp = client.get("/admin/nodes", headers={"X-API-Key": "admin-key"})
            assert resp.status_code == 200
        finally:
            del os.environ["ASC_ADMIN_API_KEY"]

    def test_admin_config_endpoint(self):
        """S-05: 管理端点需要管理员认证。"""
        os.environ.pop("ASC_API_KEY", None)
        os.environ.pop("ASC_ALLOW_NO_AUTH", None)
        os.environ.pop("ASC_ADMIN_API_KEY", None)
        app = create_app()
        client = TestClient(app)
        resp = client.get("/admin/config")
        assert resp.status_code == 403

        os.environ["ASC_API_KEY"] = "test-key"
        try:
            app = create_app()
            client = TestClient(app)
            resp = client.get("/admin/config", headers={"X-API-Key": "test-key"})
            assert resp.status_code == 200
        finally:
            del os.environ["ASC_API_KEY"]

    def test_chat_completions_with_engine_success(self):
        """模拟引擎可用时的成功推理。"""
        from unittest.mock import AsyncMock

        mock_engine = MagicMock()
        mock_engine.submit.return_value = "Hello world"
        mock_engine.submit_async = AsyncMock(return_value="Hello world")

        client = self._make_client_with_auth(model_mappings={"m": "/m.gguf"}, engine=mock_engine)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 32,
                "temperature": 0.5,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "Hello world"
        assert data["model"] == "m"

    def test_chat_completions_engine_error_returns_500(self):
        """引擎抛出异常时返回 500。"""
        from unittest.mock import AsyncMock

        mock_engine = MagicMock()
        mock_engine.submit.return_value = "Hello world"
        mock_engine.submit_async = AsyncMock(side_effect=RuntimeError("GPU OOM"))

        client = self._make_client_with_auth(model_mappings={"m": "/m.gguf"}, engine=mock_engine)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        assert resp.status_code == 500

    def test_chat_completions_streaming(self):
        """流式响应测试。"""
        from unittest.mock import AsyncMock

        mock_engine = MagicMock()
        mock_engine.submit_async = AsyncMock(return_value="Hello world")

        # 模拟异步流式生成器
        async def mock_stream(request):
            for char in "Hello world":
                yield char

        mock_engine.submit_async_stream = mock_stream

        client = self._make_client_with_auth(model_mappings={"m": "/m.gguf"}, engine=mock_engine)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200
        text = resp.text
        assert "data:" in text
        assert "[DONE]" in text

    def test_api_key_required_rejects_without_key(self):
        """配置 API Key 后未提供 Key 应返回 401。"""
        os.environ["ASC_API_KEY"] = "secret-key"
        try:
            app = create_app(model_mappings={"m": "/m.gguf"})
            client = TestClient(app)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "m",
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )
            assert resp.status_code == 401
        finally:
            del os.environ["ASC_API_KEY"]

    def test_api_key_required_accepts_valid_key(self):
        """提供正确的 API Key 应通过认证。"""
        os.environ["ASC_API_KEY"] = "secret-key"
        try:
            app = create_app(model_mappings={"m": "/m.gguf"})
            client = TestClient(app)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "m",
                    "messages": [{"role": "user", "content": "Hi"}],
                },
                headers={"X-API-Key": "secret-key"},
            )
            # 503 = 引擎未启动，但认证已通过
            assert resp.status_code in (200, 503)
        finally:
            del os.environ["ASC_API_KEY"]

    def test_no_auth_env_allows_requests(self):
        """ASC_ALLOW_NO_AUTH=1 允许无认证请求。"""
        os.environ.pop("ASC_API_KEY", None)
        os.environ["ASC_ALLOW_NO_AUTH"] = "1"
        try:
            app = create_app(model_mappings={"m": "/m.gguf"})
            client = TestClient(app)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "m",
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )
            assert resp.status_code in (200, 503)
        finally:
            del os.environ["ASC_ALLOW_NO_AUTH"]
