"""测试 CLI 主入口模块。

覆盖 asc.cli.main 中所有公开和内部函数：
- _generate_node_id: 节点 ID 生成格式
- _get_local_ip: 本地 IP 获取与异常回退
- _cmd_start: Worker 节点启动流程
- _cmd_status: 节点状态查看
- _cmd_discover: 网络节点发现
- _cmd_master: Master 节点启动
- main: argparse 入口与子命令分发

注意：CLI 模块使用延迟导入（函数内 import），因此 patch 目标必须是
源模块而非 asc.cli.main，例如 patch("time.sleep") 而非
patch("asc.cli.main.time")。
"""

import argparse
from unittest.mock import MagicMock, patch

import pytest

from asc.cli.main import (
    _cmd_discover,
    _cmd_master,
    _cmd_start,
    _cmd_status,
    _cmd_worker,
    _generate_node_id,
    _get_local_ip,
    _set_active_role,
    main,
)


class TestGenerateNodeId:
    """_generate_node_id 返回 {hostname}-{hex4} 格式。"""

    def test_format_contains_hostname(self):
        node_id = _generate_node_id()
        # 节点 ID 格式为 {hostname}-{hex4}
        assert "-" in node_id

    def test_format_hex_length(self):
        node_id = _generate_node_id()
        # 最后一段为 4 位十六进制
        hex_part = node_id.rsplit("-", 1)[1]
        assert len(hex_part) == 4

    def test_hex_is_valid(self):
        node_id = _generate_node_id()
        hex_part = node_id.rsplit("-", 1)[1]
        int(hex_part, 16)  # 不抛异常即为合法十六进制

    def test_uniqueness(self):
        ids = {_generate_node_id() for _ in range(50)}
        # 4 位 hex = 65536 种可能，50 次调用碰撞概率极低
        assert len(ids) == 50

    def test_deterministic_with_mocked_uuid(self):
        with patch("asc.cli.main.uuid") as mock_uuid, \
             patch("asc.cli.main.socket") as mock_socket:
            mock_uuid.uuid4.return_value.hex = "abcdef0123456789"
            mock_socket.gethostname.return_value = "testhost"
            assert _generate_node_id() == "testhost-abcd"


class TestGetLocalIp:
    """_get_local_ip 获取本机 IP，异常时回退到 127.0.0.1。"""

    def test_returns_string(self):
        result = _get_local_ip()
        assert isinstance(result, str)

    def test_fallback_on_socket_error(self):
        with patch("asc.cli.main.socket") as mock_socket:
            mock_socket.socket.side_effect = OSError("no network")
            assert _get_local_ip() == "127.0.0.1"

    def test_returns_ip_from_socket(self):
        with patch("asc.cli.main.socket") as mock_socket:
            mock_sock = MagicMock()
            mock_socket.socket.return_value.__enter__ = MagicMock(return_value=mock_sock)
            mock_socket.socket.return_value.__exit__ = MagicMock(return_value=False)
            mock_socket.AF_INET = 2
            mock_socket.SOCK_DGRAM = 2
            mock_sock.getsockname.return_value = ("192.168.1.100", 12345)
            assert _get_local_ip() == "192.168.1.100"


class TestCmdStart:
    """_cmd_start 启动节点（支持首次运行引导）。"""

    def _make_args(self, port=52415, host="0.0.0.0", api_port=None, reconfigure=False):
        return argparse.Namespace(port=port, host=host, api_port=api_port, reconfigure=reconfigure)

    @patch("asc.cli.main._is_first_run", return_value=False)
    @patch("asc.cli.main._load_node_config", return_value={
        "node_id": "testhost-abcd", "role": "worker",
        "auto_discover": False, "master_host": "192.168.1.1", "master_port": 52414,
    })
    @patch("asc.cli.main._get_local_ip", return_value="192.168.1.10")
    def test_worker_start(self, mock_get_ip, mock_config, mock_first_run, capsys):
        """正常启动 Worker，打印节点信息后 KeyboardInterrupt 退出。"""
        mock_agent = MagicMock()
        mock_agent.get_resources.return_value = MagicMock(
            cpu_count=8,
            memory_free_mb=16000,
            memory_total_mb=32000,
            gpus=[],
        )
        # agent.run() 和 agent.stop() 需要返回协程
        async def mock_run(*args, **kwargs):
            raise KeyboardInterrupt()
        async def mock_stop(*args, **kwargs):
            pass
        mock_agent.run = mock_run
        mock_agent.stop = mock_stop

        with patch("asc.worker.agent.WorkerAgent", return_value=mock_agent), \
             patch("asc.worker.node_gui.run_node_gui"):
            _cmd_start(self._make_args())

        output = capsys.readouterr().out
        assert "testhost-abcd" in output

    @patch("asc.cli.main._is_first_run", return_value=False)
    @patch("asc.cli.main._load_node_config", return_value={
        "node_id": "testhost-abcd", "role": "master",
        "password": "test123", "tcp_port": 52414, "api_port": 8080,
        "api_config": {"enabled": True, "api_key": "test-key"},
    })
    @patch("asc.cli.main._get_local_ip", return_value="192.168.1.10")
    def test_master_start_from_config(self, mock_get_ip, mock_config, mock_first_run, capsys):
        """从已保存配置启动 Master。"""
        mock_master = MagicMock()

        with patch("asc.master.main.MasterNode", return_value=mock_master), \
             patch("asyncio.run"):
            _cmd_start(self._make_args())

        output = capsys.readouterr().out
        assert "Master" in output


class TestCmdStatus:
    """_cmd_status 显示节点状态信息。"""

    def test_prints_status(self, capsys):
        mock_agent = MagicMock()
        mock_agent.get_resources.return_value = MagicMock(
            cpu_count=16,
            cpu_percent=35.5,
            memory_free_mb=24000,
            memory_total_mb=32768,
            gpus=[],
            compute_score=42.5,
        )
        mock_agent.rpc_status.return_value = {"running": True, "host": "0.0.0.0", "port": 52415}

        with patch("asc.worker.agent.WorkerAgent", return_value=mock_agent):
            _cmd_status(argparse.Namespace())

        output = capsys.readouterr().out
        assert "16" in output
        assert "35.5" in output
        assert "42.50" in output
        assert "运行中" in output
        assert "0.0.0.0:52415" in output

    def test_rpc_not_running(self, capsys):
        mock_agent = MagicMock()
        mock_agent.get_resources.return_value = MagicMock(
            cpu_count=4,
            cpu_percent=10.0,
            memory_free_mb=8000,
            memory_total_mb=16000,
            gpus=[],
            compute_score=0.0,
        )
        mock_agent.rpc_status.return_value = {"running": False}

        with patch("asc.worker.agent.WorkerAgent", return_value=mock_agent):
            _cmd_status(argparse.Namespace())

        output = capsys.readouterr().out
        assert "未运行" in output

    def test_gpu_status_printed(self, capsys):
        mock_gpu = MagicMock()
        mock_gpu.name = "A100"
        mock_gpu.vram_free_mb = 40000
        mock_gpu.vram_total_mb = 81920
        mock_agent = MagicMock()
        mock_agent.get_resources.return_value = MagicMock(
            cpu_count=8,
            cpu_percent=20.0,
            memory_free_mb=16000,
            memory_total_mb=32000,
            gpus=[mock_gpu],
            compute_score=100.0,
        )
        mock_agent.rpc_status.return_value = {"running": False}

        with patch("asc.worker.agent.WorkerAgent", return_value=mock_agent):
            _cmd_status(argparse.Namespace())

        output = capsys.readouterr().out
        assert "A100" in output
        assert "40000" in output
        assert "81920" in output


class TestCmdDiscover:
    """_cmd_discover 发现网络节点。"""

    @patch("asc.cli.main._get_local_ip", return_value="192.168.1.10")
    @patch("asc.cli.main._generate_node_id", return_value="node-disc0001")
    def test_discover_prints_info(self, mock_gen_id, mock_get_ip, capsys):
        mock_discovery = MagicMock()
        mock_discovery.broadcast_address = ("255.255.255.255", 52415)
        mock_discovery.scan_addresses.return_value = [f"192.168.1.{i}" for i in range(1, 255)]

        with patch("asc.network.discovery.NodeDiscovery", return_value=mock_discovery):
            _cmd_discover(argparse.Namespace())

        output = capsys.readouterr().out
        assert "255.255.255.255:52415" in output
        assert "192.168.1" in output
        assert "254" in output
        mock_discovery.scan_addresses.assert_called_once_with("192.168.1")


class TestCmdMaster:
    """_cmd_master 启动 Master 节点。"""

    def _make_args(self, host="0.0.0.0", port=52414, api_host="0.0.0.0", api_port=None, node_id=None):
        return argparse.Namespace(host=host, port=port, api_host=api_host, api_port=api_port, node_id=node_id)

    @patch("asyncio.run")
    @patch("asc.cli.main._is_first_run", return_value=False)
    @patch("asc.cli.main._load_node_config", return_value={
        "node_id": "testhost-mst0", "role": "master",
        "password": "test123", "tcp_port": 52414, "api_port": 8080,
    })
    @patch("asc.cli.main._get_local_ip", return_value="192.168.1.10")
    def test_master_start(self, mock_get_ip, mock_config, mock_first_run, mock_asyncio_run, capsys):
        mock_master = MagicMock()

        with patch("asc.master.main.MasterNode", return_value=mock_master):
            _cmd_master(self._make_args())

        mock_asyncio_run.assert_called_once()
        output = capsys.readouterr().out
        assert "Master" in output

    @patch("asyncio.run")
    @patch("asc.cli.main._is_first_run", return_value=False)
    @patch("asc.cli.main._load_node_config", return_value={
        "node_id": "my-custom-id", "role": "master",
        "password": "test123", "tcp_port": 52414, "api_port": 8080,
    })
    @patch("asc.cli.main._get_local_ip", return_value="192.168.1.10")
    def test_master_with_custom_node_id(self, mock_get_ip, mock_config, mock_first_run, mock_asyncio_run, capsys):
        mock_master = MagicMock()

        with patch("asc.master.main.MasterNode", return_value=mock_master) as mock_cls:
            _cmd_master(self._make_args())

        # 验证 MasterNode 被传入正确的 node_id
        assert mock_cls.call_args is not None
        _, call_kwargs = mock_cls.call_args
        assert call_kwargs.get("node_id") == "my-custom-id"
        assert "api_host" in call_kwargs
        assert "api_port" in call_kwargs


class TestMain:
    """main() argparse 入口与子命令分发。"""

    def test_no_command_prints_help_and_exits(self):
        """无子命令时打印帮助并退出。"""
        with patch("sys.argv", ["asc"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_unknown_command_exits_with_error(self):
        """未知命令退出码为 2（argparse 默认行为）。"""
        with patch("sys.argv", ["asc", "nonexistent"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 2

    @patch("asc.cli.main._cmd_start")
    def test_start_dispatch(self, mock_cmd_start):
        with patch("sys.argv", ["asc", "start", "--port", "6000"]):
            main()
        mock_cmd_start.assert_called_once()
        args = mock_cmd_start.call_args[0][0]
        assert args.port == 6000

    @patch("asc.cli.main._cmd_status")
    def test_status_dispatch(self, mock_cmd_status):
        with patch("sys.argv", ["asc", "status"]):
            main()
        mock_cmd_status.assert_called_once()

    @patch("asc.cli.main._cmd_discover")
    def test_discover_dispatch(self, mock_cmd_discover):
        with patch("sys.argv", ["asc", "discover"]):
            main()
        mock_cmd_discover.assert_called_once()

    @patch("asc.cli.main._cmd_master")
    def test_master_dispatch(self, mock_cmd_master):
        with patch("sys.argv", ["asc", "master", "--host", "0.0.0.0", "--port", "52414"]):
            main()
        mock_cmd_master.assert_called_once()
        args = mock_cmd_master.call_args[0][0]
        assert args.host == "0.0.0.0"
        assert args.port == 52414


class TestArgparseConfig:
    """argparse 配置：默认端口、--version 等。"""

    def test_start_default_port(self):
        with patch("sys.argv", ["asc", "start"]):
            with patch("asc.cli.main._cmd_start") as mock_start:
                main()
        args = mock_start.call_args[0][0]
        assert args.port == 52415

    def test_master_default_port(self):
        with patch("sys.argv", ["asc", "master"]):
            with patch("asc.cli.main._cmd_master") as mock_master:
                main()
        args = mock_master.call_args[0][0]
        assert args.port == 52414

    def test_master_default_host(self):
        with patch("sys.argv", ["asc", "master"]):
            with patch("asc.cli.main._cmd_master") as mock_master:
                main()
        args = mock_master.call_args[0][0]
        assert args.host == "0.0.0.0"

    def test_version_flag(self):
        with patch("sys.argv", ["asc", "--version"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_start_reconfigure_flag(self):
        with patch("sys.argv", ["asc", "start", "--reconfigure"]):
            with patch("asc.cli.main._cmd_start") as mock_start:
                main()
        args = mock_start.call_args[0][0]
        assert args.reconfigure is True

    def test_start_api_port_flag(self):
        with patch("sys.argv", ["asc", "start", "--api-port", "9090"]):
            with patch("asc.cli.main._cmd_start") as mock_start:
                main()
        args = mock_start.call_args[0][0]
        assert args.api_port == 9090

    def test_master_node_id_flag(self):
        with patch("sys.argv", ["asc", "master", "--node-id", "custom-id"]):
            with patch("asc.cli.main._cmd_master") as mock_master:
                main()
        args = mock_master.call_args[0][0]
        assert args.node_id == "custom-id"


class TestCmdStartRoleArg:
    """_cmd_start --role 参数覆盖配置文件中的角色。"""

    def _make_args(self, port=52415, host="0.0.0.0", api_port=None, reconfigure=False,
                   role=None, master_host=None, master_port=None, config=None,
                   non_interactive=False, skip_benchmark=False):
        return argparse.Namespace(
            port=port, host=host, api_port=api_port, reconfigure=reconfigure,
            role=role, master_host=master_host, master_port=master_port,
            config=config, non_interactive=non_interactive, skip_benchmark=skip_benchmark,
        )

    @patch("asc.cli.main._is_first_run", return_value=False)
    @patch("asc.cli.main._load_node_config", return_value={
        "node_id": "testhost-abcd", "role": "master",
        "password": "test123", "tcp_port": 52414, "api_port": 8080,
    })
    @patch("asc.cli.main._get_local_ip", return_value="192.168.1.10")
    def test_role_arg_overrides_config(self, mock_get_ip, mock_config, mock_first_run):
        """--role worker 覆盖配置文件中的 role: master。"""
        mock_agent = MagicMock()
        async def mock_run(*args, **kwargs):
            raise KeyboardInterrupt()
        async def mock_stop(*args, **kwargs):
            pass
        mock_agent.run = mock_run
        mock_agent.stop = mock_stop

        with patch("asc.worker.agent.WorkerAgent", return_value=mock_agent), \
             patch("asc.worker.node_gui.run_node_gui"):
            _cmd_start(self._make_args(role="worker", master_host="192.168.1.1"))

    @patch("asc.cli.main._is_first_run", return_value=False)
    @patch("asc.cli.main._load_node_config", return_value={
        "node_id": "testhost-abcd", "role": "worker",
        "auto_discover": False, "master_host": "192.168.1.1", "master_port": 52414,
    })
    @patch("asc.cli.main._get_local_ip", return_value="192.168.1.10")
    def test_role_arg_master_overrides_worker_config(self, mock_get_ip, mock_config, mock_first_run):
        """--role master 覆盖配置文件中的 role: worker。"""
        mock_master = MagicMock()

        with patch("asc.master.main.MasterNode", return_value=mock_master), \
             patch("asyncio.run"):
            _cmd_start(self._make_args(role="master"))


class TestCmdWorker:
    """_cmd_worker 直接启动 Worker 节点。"""

    def _make_args(self, host="0.0.0.0", port=52415, master_host=None, master_port=None,
                   config=None, non_interactive=False, skip_benchmark=False, reconfigure=False):
        return argparse.Namespace(
            host=host, port=port, master_host=master_host, master_port=master_port,
            config=config, non_interactive=non_interactive, skip_benchmark=skip_benchmark,
            reconfigure=reconfigure,
        )

    @patch("asc.cli.main._is_first_run", return_value=False)
    @patch("asc.cli.main._load_node_config", return_value={
        "node_id": "testhost-abcd", "role": "worker",
        "auto_discover": False, "master_host": "192.168.1.1", "master_port": 52414,
    })
    @patch("asc.cli.main._get_local_ip", return_value="192.168.1.10")
    def test_worker_start(self, mock_get_ip, mock_config, mock_first_run, capsys):
        """使用 worker 子命令启动 Worker。"""
        mock_agent = MagicMock()
        async def mock_run(*args, **kwargs):
            raise KeyboardInterrupt()
        async def mock_stop(*args, **kwargs):
            pass
        mock_agent.run = mock_run
        mock_agent.stop = mock_stop

        with patch("asc.worker.agent.WorkerAgent", return_value=mock_agent), \
             patch("asc.worker.node_gui.run_node_gui"):
            _cmd_worker(self._make_args())

        output = capsys.readouterr().out
        assert "Worker" in output

    @patch("asc.cli.main._is_first_run", return_value=False)
    @patch("asc.cli.main._load_node_config", return_value={
        "node_id": "testhost-abcd", "role": "worker",
        "auto_discover": True,
    })
    @patch("asc.cli.main._get_local_ip", return_value="192.168.1.10")
    def test_worker_with_master_host_arg(self, mock_get_ip, mock_config, mock_first_run):
        """worker 子命令的 --master-host 参数覆盖配置。"""
        mock_agent = MagicMock()
        async def mock_run(*args, **kwargs):
            raise KeyboardInterrupt()
        async def mock_stop(*args, **kwargs):
            pass
        mock_agent.run = mock_run
        mock_agent.stop = mock_stop

        with patch("asc.worker.agent.WorkerAgent", return_value=mock_agent), \
             patch("asc.worker.node_gui.run_node_gui"):
            _cmd_worker(self._make_args(master_host="10.0.0.1", master_port=52414))


class TestWorkerSubcommand:
    """worker 子命令的 argparse 分发。"""

    @patch("asc.cli.main._cmd_worker")
    def test_worker_dispatch(self, mock_cmd_worker):
        with patch("sys.argv", ["asc", "worker"]):
            main()
        mock_cmd_worker.assert_called_once()

    @patch("asc.cli.main._cmd_worker")
    def test_worker_with_master_host(self, mock_cmd_worker):
        with patch("sys.argv", ["asc", "worker", "--master-host", "10.0.0.1"]):
            main()
        args = mock_cmd_worker.call_args[0][0]
        assert args.master_host == "10.0.0.1"

    @patch("asc.cli.main._cmd_worker")
    def test_worker_with_master_port(self, mock_cmd_worker):
        with patch("sys.argv", ["asc", "worker", "--master-port", "6000"]):
            main()
        args = mock_cmd_worker.call_args[0][0]
        assert args.master_port == 6000


class TestRoleSpecificConfig:
    """角色专属配置文件路径测试。"""

    def test_set_active_role_affects_config_path(self, tmp_path):
        """设定 _active_role 后，配置文件路径变为 node_config_{role}.json。"""
        from asc.cli.main import _get_node_config_path

        # 保存原始状态
        import asc.cli.main as cli_mod
        orig_custom = cli_mod._custom_config_path
        orig_role = cli_mod._active_role

        try:
            # 清除自定义路径
            cli_mod._custom_config_path = None
            # 设定角色
            _set_active_role("worker")
            path = _get_node_config_path()
            assert path.name == "node_config_worker.json"

            _set_active_role("master")
            path = _get_node_config_path()
            assert path.name == "node_config_master.json"

            # 清除角色
            _set_active_role(None)
            path = _get_node_config_path()
            assert path.name == "node_config.json"
        finally:
            # 恢复原始状态
            cli_mod._custom_config_path = orig_custom
            cli_mod._active_role = orig_role

    def test_start_role_flag_parsed(self):
        """asc start --role worker 被正确解析。"""
        with patch("sys.argv", ["asc", "start", "--role", "worker"]):
            with patch("asc.cli.main._cmd_start") as mock_start:
                main()
        args = mock_start.call_args[0][0]
        assert args.role == "worker"

    def test_start_master_host_flag_parsed(self):
        """asc start --master-host 10.0.0.1 被正确解析。"""
        with patch("sys.argv", ["asc", "start", "--master-host", "10.0.0.1"]):
            with patch("asc.cli.main._cmd_start") as mock_start:
                main()
        args = mock_start.call_args[0][0]
        assert args.master_host == "10.0.0.1"
