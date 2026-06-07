"""测试 RpcServer 模块。"""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from asc.worker.rpc_server import RpcServer


class TestRpcServerInit:
    """初始化。"""

    def test_default_state(self):
        rpc = RpcServer()
        assert rpc.is_running is False
        assert rpc.port is None
        assert rpc.endpoint is None


class TestRpcServerFindExecutable:
    """查找可执行文件。"""

    @patch("asc.utils.system.shutil.which", return_value="/usr/bin/rpc-server")
    @patch("pathlib.Path.exists", return_value=False)
    def test_find_from_path(self, mock_exists, mock_which):
        rpc = RpcServer()
        exe = rpc._find_executable()
        assert exe is not None
        assert "rpc-server" in exe

    @patch("asc.utils.system.shutil.which", return_value=None)
    @patch("pathlib.Path.exists", return_value=False)
    def test_not_found(self, mock_exists, mock_which):
        rpc = RpcServer()
        assert rpc._find_executable() is None


class TestRpcServerPortAllocation:
    """端口分配。"""

    def test_find_free_port(self):
        rpc = RpcServer()
        port = rpc._find_free_port()
        assert RpcServer.DEFAULT_PORT_START <= port <= RpcServer.DEFAULT_PORT_END

    @patch.object(RpcServer, "_is_port_free", return_value=False)
    def test_no_free_port(self, mock_free):
        rpc = RpcServer()
        with pytest.raises(RuntimeError, match="无法找到可用端口"):
            rpc._find_free_port()

    def test_is_port_free_true(self):
        rpc = RpcServer()
        # 使用一个不太可能占用的端口
        assert rpc._is_port_free(0) is True


class TestRpcServerStartStop:
    """启动和停止。"""

    @patch.object(RpcServer, "_find_executable", return_value="/fake/rpc-server")
    @patch("subprocess.Popen")
    def test_start_with_auto_port(self, mock_popen, mock_find):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        rpc = RpcServer()
        port = rpc.start()

        assert port is not None
        assert RpcServer.DEFAULT_PORT_START <= port <= RpcServer.DEFAULT_PORT_END
        assert rpc.is_running is True
        assert rpc.port == port
        assert rpc.endpoint == f"0.0.0.0:{port}"

        mock_popen.assert_called_once()
        cmd = mock_popen.call_args[0][0]
        assert cmd[0] == "/fake/rpc-server"
        assert "--rpc-server-bind" in cmd
        assert f"0.0.0.0:{port}" in cmd

    @patch.object(RpcServer, "_find_executable", return_value="/fake/rpc-server")
    @patch("subprocess.Popen")
    def test_start_with_specific_port(self, mock_popen, mock_find):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        rpc = RpcServer()
        port = rpc.start(port=55555)

        assert port == 55555
        assert rpc.port == 55555

    @patch.object(RpcServer, "_find_executable", return_value="/fake/rpc-server")
    @patch("subprocess.Popen")
    def test_start_already_running(self, mock_popen, mock_find):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        rpc = RpcServer()
        rpc.start()

        with pytest.raises(RuntimeError, match="已在运行"):
            rpc.start()

    def test_start_executable_not_found(self):
        rpc = RpcServer()
        with (
            patch.object(rpc, "_find_executable", return_value=None),
            pytest.raises(FileNotFoundError, match="未找到"),
        ):
            rpc.start()

    @patch.object(RpcServer, "_find_executable", return_value="/fake/rpc-server")
    @patch("subprocess.Popen")
    def test_stop(self, mock_popen, mock_find):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        rpc = RpcServer()
        rpc.start()
        rpc.stop()

        assert rpc.is_running is False
        assert rpc.port is None
        assert rpc.endpoint is None
        mock_proc.terminate.assert_called_once()

    @patch.object(RpcServer, "_find_executable", return_value="/fake/rpc-server")
    @patch("subprocess.Popen")
    def test_stop_kill_fallback(self, mock_popen, mock_find):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.wait.side_effect = subprocess.TimeoutExpired("cmd", 5)
        mock_popen.return_value = mock_proc

        rpc = RpcServer()
        rpc.start()
        rpc.stop()

        mock_proc.kill.assert_called_once()

    def test_stop_not_running(self):
        rpc = RpcServer()
        rpc.stop()  # 不应抛出异常
        assert rpc.is_running is False


class TestRpcServerReady:
    """就绪检测。"""

    @patch.object(RpcServer, "_find_executable", return_value="/fake/rpc-server")
    @patch("subprocess.Popen")
    @patch("socket.create_connection")
    def test_is_ready_success(self, mock_conn, mock_popen, mock_find):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc
        mock_conn.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_conn.return_value.__exit__ = MagicMock(return_value=False)

        rpc = RpcServer()
        rpc.start()
        assert rpc.is_ready(timeout=1.0) is True

    @patch.object(RpcServer, "_find_executable", return_value="/fake/rpc-server")
    @patch("subprocess.Popen")
    @patch("socket.create_connection", side_effect=ConnectionRefusedError)
    def test_is_ready_timeout(self, mock_conn, mock_popen, mock_find):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        rpc = RpcServer()
        rpc.start()
        assert rpc.is_ready(timeout=0.5) is False

    def test_is_ready_not_running(self):
        rpc = RpcServer()
        assert rpc.is_ready() is False


class TestRpcServerStatus:
    """状态查询。"""

    @patch.object(RpcServer, "_find_executable", return_value="/fake/rpc-server")
    @patch("subprocess.Popen")
    def test_status_running(self, mock_popen, mock_find):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        rpc = RpcServer()
        rpc.start(port=50055)
        status = rpc.status()

        assert status["running"] is True
        assert status["host"] == "0.0.0.0"
        assert status["port"] == 50055
        assert status["endpoint"] == "0.0.0.0:50055"

    def test_status_not_running(self):
        rpc = RpcServer()
        status = rpc.status()
        assert status["running"] is False
        assert status["host"] is None
        assert status["port"] is None
        assert status["endpoint"] is None
