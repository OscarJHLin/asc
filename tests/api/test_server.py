"""测试 FastAPI 服务器 + 认证。"""

import os

from fastapi.testclient import TestClient

from asc.api.auth import require_api_key
from asc.api.server import create_app


class TestAuthMiddleware:
    """API Key 认证。"""

    def test_no_key_configured_passes(self):
        """未配置 API Key 时跳过认证。"""
        os.environ.pop("ASC_API_KEY", None)
        # 无 key 时应该通过
        assert require_api_key(api_key=None) is True

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
        app = create_app(model_mappings={"m": "/m.gguf"})
        client = TestClient(app)
        # 发送请求但引擎未启动，应返回 503 或错误
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        # 引擎未启动，预期 503
        assert resp.status_code in (200, 503)

    def test_admin_nodes_endpoint(self):
        app = create_app()
        client = TestClient(app)
        resp = client.get("/admin/nodes")
        assert resp.status_code == 200
