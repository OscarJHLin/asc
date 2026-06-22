"""测试 API 服务器 uvicorn 启动入口 + SSL 配置。

TDD 测试覆盖：
- run_api_server() 函数启动 uvicorn
- SSL 证书参数传递到 uvicorn
- 无 SSL 时默认 HTTP
"""

from unittest.mock import patch

from asc.api.server import create_app


class TestRunApiServer:
    """API 服务器启动测试。"""

    def test_run_api_server_function_exists(self):
        """run_api_server 函数存在。"""
        from asc.api.server import run_api_server
        assert callable(run_api_server)

    @patch("uvicorn.run")
    def test_run_api_server_starts_uvicorn(self, mock_run):
        """run_api_server 调用 uvicorn.run。"""
        from asc.api.server import run_api_server
        run_api_server(host="0.0.0.0", port=8000)
        mock_run.assert_called_once()

    @patch("uvicorn.run")
    def test_run_api_server_default_params(self, mock_run):
        """默认参数：0.0.0.0:8000，无 SSL。"""
        from asc.api.server import run_api_server
        run_api_server()
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["host"] == "0.0.0.0"
        assert call_kwargs["port"] == 8000
        assert call_kwargs.get("ssl_certfile") is None
        assert call_kwargs.get("ssl_keyfile") is None

    @patch("uvicorn.run")
    def test_run_api_server_with_ssl(self, mock_run):
        """SSL 参数传递到 uvicorn。"""
        from asc.api.server import run_api_server
        run_api_server(
            host="0.0.0.0",
            port=8443,
            ssl_certfile="/path/to/cert.pem",
            ssl_keyfile="/path/to/key.pem",
        )
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["ssl_certfile"] == "/path/to/cert.pem"
        assert call_kwargs["ssl_keyfile"] == "/path/to/key.pem"
        assert call_kwargs["port"] == 8443

    @patch("uvicorn.run")
    def test_run_api_server_with_custom_app(self, mock_run):
        """传入自定义 app 时使用该 app。"""
        from asc.api.server import run_api_server
        app = create_app()
        run_api_server(app=app, host="127.0.0.1", port=9000)
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["app"] is app
