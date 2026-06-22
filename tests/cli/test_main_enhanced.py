"""补充 CLI main 测试，提升覆盖率至 85%+。

原测试仅覆盖基础命令解析，本文件补充：
- 首次运行引导流程
- 配置文件读写
- Master/Worker 启动逻辑
- 状态查看和发现命令
- 边界条件（空配置、无效输入等）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from asc.cli.main import (
    _cmd_discover,
    _cmd_master,
    _cmd_start,
    _cmd_status,
    _generate_node_id,
    _get_config_dir,
    _get_first_run_flag,
    _get_local_ip,
    _is_first_run,
    _load_node_config,
    _mark_first_run_complete,
    _print_master_info,
    _print_worker_info,
    _prompt_master_config,
    _prompt_role,
    _prompt_worker_config,
    _run_first_time_setup,
    _save_node_config,
    main,
)


class TestUtilityFunctions:
    """测试工具函数。"""

    def test_generate_node_id(self):
        """节点 ID 应包含主机名和 UUID 前缀。"""
        node_id = _generate_node_id()
        assert "-" in node_id
        assert len(node_id.split("-")[-1]) == 4

    def test_get_local_ip(self):
        """应返回合法 IP 格式。"""
        ip = _get_local_ip()
        parts = ip.split(".")
        assert len(parts) == 4
        for p in parts:
            assert p.isdigit()
            assert 0 <= int(p) <= 255

    def test_config_dir(self):
        """配置目录应为 ~/.asc。"""
        config_dir = _get_config_dir()
        assert config_dir.name == ".asc"
        assert config_dir.exists()

    def test_first_run_flag(self):
        """首次运行标志文件路径应包含配置文件名以区分角色。"""
        flag = _get_first_run_flag()
        assert flag.name.startswith(".") and flag.name.endswith("_first_run")

    def test_is_first_run(self, tmp_path: Path):
        """标志文件不存在时应返回 True。"""
        with patch("asc.cli.main._get_first_run_flag", return_value=tmp_path / "not_exists"):
            assert _is_first_run() is True

    def test_mark_first_run_complete(self, tmp_path: Path):
        """标记后应创建标志文件。"""
        flag = tmp_path / "first_run_complete"
        with patch("asc.cli.main._get_first_run_flag", return_value=flag):
            _mark_first_run_complete()
            assert flag.exists()

    def test_load_node_config_exists(self, tmp_path: Path):
        """配置文件存在时应正确加载。"""
        config_path = tmp_path / "node_config.json"
        config_path.write_text(json.dumps({"role": "master"}), encoding="utf-8")
        with patch("asc.cli.main._get_node_config_path", return_value=config_path):
            config = _load_node_config()
            assert config["role"] == "master"

    def test_load_node_config_not_exists(self, tmp_path: Path):
        """配置文件不存在时应返回空字典。"""
        with patch("asc.cli.main._get_node_config_path", return_value=tmp_path / "not_exists"):
            assert _load_node_config() == {}

    def test_save_node_config(self, tmp_path: Path):
        """应正确保存配置。"""
        config_path = tmp_path / "node_config.json"
        with patch("asc.cli.main._get_node_config_path", return_value=config_path):
            _save_node_config({"role": "worker", "node_id": "n1"})
            loaded = json.loads(config_path.read_text(encoding="utf-8"))
            assert loaded["role"] == "worker"


class TestPromptFunctions:
    """测试交互式提示函数。"""

    def test_prompt_role_master(self):
        """选择 1 应返回 master。"""
        with patch("builtins.input", return_value="1"):
            assert _prompt_role() == "master"

    def test_prompt_role_worker(self):
        """选择 2 应返回 worker。"""
        with patch("builtins.input", return_value="2"):
            assert _prompt_role() == "worker"

    def test_prompt_role_invalid_then_valid(self):
        """无效输入后应重新提示。"""
        with patch("builtins.input", side_effect=["3", "abc", "1"]):
            assert _prompt_role() == "master"

    def test_prompt_master_config_default(self):
        """Master 配置默认流程。"""
        inputs = ["n", "n"]  # 不自定义密码, 不启用 API
        with patch("builtins.input", side_effect=inputs):
            with patch("asc.cli.main.getpass.getpass", return_value=""):
                config = _prompt_master_config()
                assert "password" in config
                assert len(config["password"]) > 0  # 自动生成的密码
                assert config["api_config"]["enabled"] is False

    def test_prompt_master_config_custom(self):
        """Master 配置自定义流程。"""
        inputs = ["y", "Y", "custom-key"]  # 自定义密码, 启用 API, 输入 key
        with patch("builtins.input", side_effect=inputs):
            with patch("asc.cli.main.getpass.getpass", return_value="mypassword"):
                config = _prompt_master_config()
                assert config["password"] == "mypassword"
                assert config["api_config"]["enabled"] is True
                assert config["api_config"]["api_key"] == "custom-key"

    def test_prompt_worker_config_auto_discover(self):
        """Worker 自动发现配置。"""
        with patch("builtins.input", return_value=""):
            config = _prompt_worker_config()
            assert config["auto_discover"] is True
            assert config["master_host"] is None

    def test_prompt_worker_config_manual(self):
        """Worker 手动配置。"""
        inputs = ["n", "10.0.0.1", ""]
        with patch("builtins.input", side_effect=inputs):
            config = _prompt_worker_config()
            assert config["auto_discover"] is False
            assert config["master_host"] == "10.0.0.1"
            assert config["master_port"] == 52414

    def test_prompt_worker_config_custom_port(self):
        """Worker 自定义端口。"""
        inputs = ["n", "10.0.0.1", "9999"]
        with patch("builtins.input", side_effect=inputs):
            config = _prompt_worker_config()
            assert config["master_port"] == 9999

    def test_prompt_worker_config_invalid_port(self):
        """Worker 无效端口应回退默认值。"""
        inputs = ["n", "10.0.0.1", "not_a_port"]
        with patch("builtins.input", side_effect=inputs):
            config = _prompt_worker_config()
            assert config["master_port"] == 52414


class TestPrintInfo:
    """测试信息打印函数。"""

    def test_print_master_info(self, capsys):
        """Master 启动信息应包含关键字段，且密码被掩码。"""
        config = {"password": "secret123"}
        _print_master_info(config, "0.0.0.0", 52414, 8080)
        captured = capsys.readouterr()
        assert "Master" in captured.out
        assert "********" in captured.out
        assert "secret123" not in captured.out
        assert "8080" in captured.out

    def test_print_worker_info(self, capsys):
        """Worker 启动信息应包含关键字段。"""
        config = {"node_id": "worker-1", "auto_discover": True}
        _print_worker_info(config, 52415)
        captured = capsys.readouterr()
        assert "Worker" in captured.out
        assert "worker-1" in captured.out
        assert "自动发现" in captured.out

    def test_print_worker_info_with_master(self, capsys):
        """Worker 启动信息应显示 Master 地址。"""
        config = {"node_id": "w1", "auto_discover": False, "master_host": "10.0.0.5", "master_port": 52414}
        _print_worker_info(config, 52415)
        captured = capsys.readouterr()
        assert "10.0.0.5:52414" in captured.out


class TestRunFirstTimeSetup:
    """测试首次运行设置。"""

    def test_run_first_time_setup_master(self, tmp_path: Path):
        """首次运行选择 Master。"""
        with patch("asc.cli.main._is_first_run", return_value=True):
            with patch("asc.cli.main._prompt_role", return_value="master"):
                with patch("asc.cli.main._prompt_master_config", return_value={"password": "p"}):
                    with patch("asc.worker.node_setup.NodeSetup") as MockSetup:
                        mock_setup = MagicMock()
                        mock_setup.run.return_value.success = True
                        mock_setup.run.return_value.benchmark_score = 100.0
                        mock_setup.run.return_value.to_dict.return_value = {}
                        mock_setup.run.return_value.steps = []
                        MockSetup.return_value = mock_setup
                        with patch("asc.cli.main._save_node_config") as mock_save:
                            config = _run_first_time_setup()
                            assert config["role"] == "master"
                            mock_save.assert_called_once()

    def test_run_first_time_setup_worker(self, tmp_path: Path):
        """首次运行选择 Worker。"""
        with patch("asc.cli.main._is_first_run", return_value=True):
            with patch("asc.cli.main._prompt_role", return_value="worker"):
                with patch("asc.cli.main._prompt_worker_config", return_value={"auto_discover": True}):
                    with patch("asc.worker.node_setup.NodeSetup") as MockSetup:
                        mock_setup = MagicMock()
                        mock_setup.run.return_value.success = True
                        mock_setup.run.return_value.benchmark_score = 50.0
                        mock_setup.run.return_value.to_dict.return_value = {}
                        mock_setup.run.return_value.steps = []
                        MockSetup.return_value = mock_setup
                        with patch("asc.cli.main._save_node_config") as mock_save:
                            config = _run_first_time_setup()
                            assert config["role"] == "worker"
                            mock_save.assert_called_once()


class TestCommands:
    """测试 CLI 命令处理。"""

    def test_cmd_discover(self, capsys):
        """discover 命令应打印扫描信息。"""
        args = argparse.Namespace()
        _cmd_discover(args)
        captured = capsys.readouterr()
        assert "发现" in captured.out

    def test_cmd_status(self, capsys):
        """status 命令应打印状态信息。"""
        args = argparse.Namespace()
        with patch("asc.worker.agent.WorkerAgent") as MockAgent:
            mock_agent = MagicMock()
            mock_agent.get_resources.return_value.cpu_count = 8
            mock_agent.get_resources.return_value.cpu_percent = 10.0
            mock_agent.get_resources.return_value.memory_free_mb = 8000
            mock_agent.get_resources.return_value.memory_total_mb = 16000
            mock_agent.get_resources.return_value.gpus = []
            mock_agent.get_resources.return_value.compute_score = 60.0
            mock_agent.rpc_status.return_value = {"running": False}
            MockAgent.return_value = mock_agent
            _cmd_status(args)
        captured = capsys.readouterr()
        assert "CPU" in captured.out
        assert "8" in captured.out

    def test_cmd_start_first_run_master(self):
        """首次运行启动 Master。"""
        args = argparse.Namespace(port=52414, host="0.0.0.0", api_port=None, reconfigure=False)
        with patch("asc.cli.main._is_first_run", return_value=True):
            with patch("asc.cli.main._run_first_time_setup", return_value={
                "role": "master", "node_id": "m1", "password": "p"
            }):
                with patch("asc.cli.main._start_master") as mock_start:
                    _cmd_start(args)
                    mock_start.assert_called_once()

    def test_cmd_start_existing_worker(self):
        """已有配置启动 Worker。"""
        args = argparse.Namespace(port=52415, host="0.0.0.0", api_port=None, reconfigure=False)
        with patch("asc.cli.main._load_node_config", return_value={
            "role": "worker", "node_id": "w1", "master_host": "10.0.0.1"
        }), patch("asc.cli.main._is_first_run", return_value=False):
            with patch("asc.cli.main._start_worker") as mock_start:
                _cmd_start(args)
                mock_start.assert_called_once()

    def test_cmd_master_no_config(self):
        """master 命令无配置时应引导。"""
        args = argparse.Namespace(host="0.0.0.0", port=52414, api_host="0.0.0.0", api_port=8080, node_id=None)
        with patch("asc.cli.main._load_node_config", return_value={}):
            with patch("asc.cli.main._run_first_time_setup", return_value={
                "role": "master", "node_id": "m1"
            }):
                with patch("asc.cli.main._start_master") as mock_start:
                    _cmd_master(args)
                    mock_start.assert_called_once()


class TestMainEntry:
    """测试 main 入口。"""

    def test_main_no_args(self, capsys):
        """无参数时应打印帮助。"""
        with patch("sys.argv", ["asc"]):
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 0
        captured = capsys.readouterr()
        assert "usage" in captured.out.lower()

    def test_main_version(self, capsys):
        """--version 应打印版本。"""
        with patch("sys.argv", ["asc", "--version"]):
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 0
        captured = capsys.readouterr()
        assert "0.1.0" in captured.out

    def test_main_unknown_command(self, capsys):
        """未知命令应提示错误。"""
        with patch("sys.argv", ["asc", "unknown"]):
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 2  # argparse invalid choice
        captured = capsys.readouterr()
        assert "unknown" in captured.err.lower() or "invalid choice" in captured.err.lower()
