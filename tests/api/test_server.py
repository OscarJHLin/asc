"""测试 FastAPI 服务器 + 认证。"""

import os
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from asc.api.auth import _init_default_store, require_api_key
from asc.api.server import create_app


class TestAuthMiddleware:
    """API Key 认证。"""

    def test_no_key_configured_rejects(self):
        """S-02/S-03: 未配置 API Key 时拒绝请求。"""
        os.environ.pop("ASC_API_KEY", None)
        os.environ.pop("ASC_ALLOW_NO_AUTH", None)
        _init_default_store()
        assert require_api_key(api_key=None) is False

    def test_no_key_with_allow_no_auth_allows(self):
        """ASC_ALLOW_NO_AUTH=1 允许免认证访问（仅用于开发/测试环境）。"""
        os.environ.pop("ASC_API_KEY", None)
        os.environ["ASC_ALLOW_NO_AUTH"] = "1"
        try:
            _init_default_store()
            assert require_api_key(api_key=None) is True
        finally:
            del os.environ["ASC_ALLOW_NO_AUTH"]

    def test_valid_key_passes(self):
        """正确的 API Key 通过。"""
        os.environ["ASC_API_KEY"] = "test-key-123"
        try:
            _init_default_store()
            assert require_api_key(api_key="test-key-123") is True
        finally:
            del os.environ["ASC_API_KEY"]

    def test_invalid_key_fails(self):
        """错误的 API Key 被拒绝。"""
        os.environ["ASC_API_KEY"] = "correct-key"
        try:
            _init_default_store()
            assert require_api_key(api_key="wrong-key") is False
        finally:
            del os.environ["ASC_API_KEY"]

    def test_missing_key_fails(self):
        """需要 key 但未提供时被拒绝。"""
        os.environ["ASC_API_KEY"] = "required-key"
        try:
            _init_default_store()
            assert require_api_key(api_key=None) is False
        finally:
            del os.environ["ASC_API_KEY"]


class TestFastAPIApp:
    """FastAPI 应用。"""

    def _make_client_with_auth(self, **kwargs):
        """创建带认证的测试客户端。"""
        os.environ["ASC_API_KEY"] = "test-key"
        _init_default_store()
        app = create_app(**kwargs)
        client = TestClient(app, headers={"X-API-Key": "test-key"})
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
        # /v1/models 现在需要认证
        resp = client.get("/v1/models")
        assert resp.status_code == 403
        # 带认证的请求应返回 200
        os.environ["ASC_API_KEY"] = "test-key"
        _init_default_store()
        app2 = create_app(model_mappings={"llama-3.1-8b": "/models/llama.gguf"})
        client2 = TestClient(app2)
        resp2 = client2.get("/v1/models", headers={"X-API-Key": "test-key"})
        assert resp2.status_code == 200
        data = resp2.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        del os.environ["ASC_API_KEY"]

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
        """S-05: 管理端点需要管理员认证（RBAC）。"""
        # 无认证时应返回 403（管理端点需要管理员权限）
        os.environ.pop("ASC_API_KEY", None)
        os.environ.pop("ASC_ALLOW_NO_AUTH", None)
        os.environ.pop("ASC_ADMIN_API_KEY", None)
        _init_default_store()
        app = create_app()
        client = TestClient(app)
        resp = client.get("/admin/nodes")
        assert resp.status_code == 403

        # 有普通 API Key 但无管理员 Key 时，RBAC 拒绝访问
        os.environ["ASC_API_KEY"] = "test-key"
        try:
            _init_default_store()
            app = create_app()
            client = TestClient(app)
            resp = client.get("/admin/nodes", headers={"X-API-Key": "test-key"})
            assert resp.status_code == 403
        finally:
            del os.environ["ASC_API_KEY"]

        # 有管理员 Key 时，使用管理员 Key 验证
        os.environ["ASC_API_KEY"] = "test-key"
        os.environ["ASC_ADMIN_API_KEY"] = "admin-key"
        try:
            _init_default_store()
            app = create_app()
            client = TestClient(app)
            # 普通 Key 被拒绝
            resp = client.get("/admin/nodes", headers={"X-API-Key": "test-key"})
            assert resp.status_code == 403
            # 管理员 Key 通过
            resp = client.get("/admin/nodes", headers={"X-API-Key": "admin-key"})
            assert resp.status_code == 200
        finally:
            del os.environ["ASC_API_KEY"]
            del os.environ["ASC_ADMIN_API_KEY"]

    def test_admin_config_endpoint(self):
        """S-05: 管理端点需要管理员认证（RBAC）。"""
        os.environ.pop("ASC_API_KEY", None)
        os.environ.pop("ASC_ALLOW_NO_AUTH", None)
        os.environ.pop("ASC_ADMIN_API_KEY", None)
        _init_default_store()
        app = create_app()
        client = TestClient(app)
        resp = client.get("/admin/config")
        assert resp.status_code == 403

        # 普通 API Key 不能访问 admin 端点
        os.environ["ASC_API_KEY"] = "test-key"
        try:
            _init_default_store()
            app = create_app()
            client = TestClient(app)
            resp = client.get("/admin/config", headers={"X-API-Key": "test-key"})
            assert resp.status_code == 403
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
            _init_default_store()
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
            _init_default_store()
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
        """ASC_ALLOW_NO_AUTH=1 允许无认证请求（仅用于开发/测试环境）。"""
        os.environ.pop("ASC_API_KEY", None)
        os.environ["ASC_ALLOW_NO_AUTH"] = "1"
        try:
            _init_default_store()
            app = create_app(model_mappings={"m": "/m.gguf"})
            client = TestClient(app)
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "m",
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )
            # ASC_ALLOW_NO_AUTH=1 时不会因认证被拒绝
            # 但可能因无引擎而 503，不会是 401
            assert resp.status_code != 401
        finally:
            del os.environ["ASC_ALLOW_NO_AUTH"]


class TestMetricsEndpoint:
    """Prometheus /metrics 端点测试。"""

    def _make_client_with_auth(self, **kwargs):
        """创建带认证的测试客户端。"""
        os.environ["ASC_API_KEY"] = "test-key"
        _init_default_store()
        app = create_app(**kwargs)
        client = TestClient(app)
        return client

    def test_metrics_endpoint_returns_200(self):
        """GET /metrics 带认证返回 200。"""
        client = self._make_client_with_auth()
        resp = client.get("/metrics", headers={"X-API-Key": "test-key"})
        assert resp.status_code == 200
        del os.environ["ASC_API_KEY"]

    def test_metrics_content_type(self):
        """响应 Content-Type 为 Prometheus 文本格式。"""
        client = self._make_client_with_auth()
        resp = client.get("/metrics", headers={"X-API-Key": "test-key"})
        assert "text/plain" in resp.headers["content-type"]
        del os.environ["ASC_API_KEY"]

    def test_metrics_empty_collector(self):
        """空 collector 返回空字符串。"""
        from asc.core.monitoring import MetricsCollector

        collector = MetricsCollector()
        client = self._make_client_with_auth(metrics_collector=collector)
        resp = client.get("/metrics", headers={"X-API-Key": "test-key"})
        assert resp.status_code == 200
        # 空 collector 导出为空字符串
        assert resp.text.strip() == ""
        del os.environ["ASC_API_KEY"]

    def test_metrics_with_counter(self):
        """包含 Counter 指标的 Prometheus 格式。"""
        from asc.core.monitoring import MetricsCollector

        collector = MetricsCollector()
        collector.counter("requests_total", "总请求数").inc(5)
        client = self._make_client_with_auth(metrics_collector=collector)
        resp = client.get("/metrics", headers={"X-API-Key": "test-key"})
        text = resp.text
        assert "# HELP requests_total 总请求数" in text
        assert "# TYPE requests_total counter" in text
        assert "requests_total 5" in text
        del os.environ["ASC_API_KEY"]

    def test_metrics_with_gauge(self):
        """包含 Gauge 指标的 Prometheus 格式。"""
        from asc.core.monitoring import MetricsCollector

        collector = MetricsCollector()
        collector.gauge("nodes_online", "在线节点数").set(3)
        client = self._make_client_with_auth(metrics_collector=collector)
        resp = client.get("/metrics", headers={"X-API-Key": "test-key"})
        text = resp.text
        assert "# HELP nodes_online 在线节点数" in text
        assert "# TYPE nodes_online gauge" in text
        assert "nodes_online 3" in text
        del os.environ["ASC_API_KEY"]

    def test_metrics_with_histogram(self):
        """包含 Histogram 指标的 Prometheus 格式。"""
        from asc.core.monitoring import MetricsCollector

        collector = MetricsCollector()
        h = collector.histogram("request_latency_ms", "请求延迟")
        h.observe(45.2)
        h.observe(120.0)
        client = self._make_client_with_auth(metrics_collector=collector)
        resp = client.get("/metrics", headers={"X-API-Key": "test-key"})
        text = resp.text
        assert "# HELP request_latency_ms 请求延迟" in text
        assert "# TYPE request_latency_ms histogram" in text
        assert "request_latency_ms_count 2" in text
        assert "request_latency_ms_sum" in text
        del os.environ["ASC_API_KEY"]

    def test_metrics_requires_auth(self):
        """Prometheus /metrics 端点现在需要认证。"""
        os.environ.pop("ASC_API_KEY", None)
        os.environ.pop("ASC_ALLOW_NO_AUTH", None)
        _init_default_store()
        app = create_app()
        client = TestClient(app)
        resp = client.get("/metrics")
        assert resp.status_code == 403
