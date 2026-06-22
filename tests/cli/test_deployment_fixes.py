"""测试部署问题修复。

覆盖 deployment_issues_oscarlin.md 中记录的根因修复：
- 问题3：Master 启动时自动注入配置文件中的 api_key 到 ApiKeyStore
- 问题4：基准测试失败日志降级 + --skip-benchmark 参数
- 问题5：Worker 注册使用异步 get_resources_async + TCPClient.connect 日志
- 问题6：ASC_CONFIG_DIR 环境变量 + --config 启动参数
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from asc.api.auth import ApiKeyStore, set_key_store, require_admin, _default_store
from asc.cli.main import (
    _cmd_master,
    _cmd_start,
    _get_config_dir,
    _load_node_config,
    _save_node_config,
    _set_config_path,
    _custom_config_path,
    main,
)
from asc.worker.agent import WorkerAgent


# ---------------------------------------------------------------------------
# 问题3：Master 启动时自动注入配置文件中的 api_key 到 ApiKeyStore
# ---------------------------------------------------------------------------


class TestMasterApiKeyInjection:
    """_start_master 应从配置文件中注入 api_key 到 ApiKeyStore。"""

    def _make_args(self, host="0.0.0.0", port=52414, api_host="0.0.0.0", api_port=8080, node_id=None, config=None):
        return argparse.Namespace(host=host, port=port, api_host=api_host, api_port=api_port, node_id=node_id, config=config)

    @patch("asyncio.run")
    @patch("asc.cli.main._get_local_ip", return_value="192.168.1.10")
    @patch("asc.cli.main._save_node_config")
    @patch("asc.cli.main._load_node_config")
    def test_api_key_from_config_injected(self, mock_load, mock_save, mock_ip, mock_asyncio):
        """配置文件中的 api_key 应被注入到 ApiKeyStore。"""
        config = {
            "node_id": "test-master",
            "role": "master",
            "api_config": {"enabled": True, "api_key": "config-api-key-123"},
        }
        mock_load.return_value = config
        mock_master = MagicMock()

        with patch("asc.master.main.MasterNode", return_value=mock_master), \
             patch.dict(os.environ, {}, clear=True):
            _cmd_master(self._make_args())

        # 验证 ApiKeyStore 中包含配置文件的 api_key
        from asc.api.auth import _get_default_store
        store = _get_default_store()
        assert store.require_admin("config-api-key-123") is True

    @patch("asyncio.run")
    @patch("asc.cli.main._get_local_ip", return_value="192.168.1.10")
    @patch("asc.cli.main._save_node_config")
    @patch("asc.cli.main._load_node_config")
    def test_env_key_and_config_key_merged(self, mock_load, mock_save, mock_ip, mock_asyncio):
        """环境变量和配置文件中的 api_key 应合并。"""
        config = {
            "node_id": "test-master",
            "role": "master",
            "api_config": {"enabled": True, "api_key": "config-key"},
        }
        mock_load.return_value = config
        mock_master = MagicMock()

        with patch("asc.master.main.MasterNode", return_value=mock_master), \
             patch.dict(os.environ, {"ASC_ADMIN_API_KEY": "env-key"}, clear=False):
            _cmd_master(self._make_args())

        from asc.api.auth import _get_default_store
        store = _get_default_store()
        assert store.require_admin("config-key") is True
        assert store.require_admin("env-key") is True

    @patch("asyncio.run")
    @patch("asc.cli.main._get_local_ip", return_value="192.168.1.10")
    @patch("asc.cli.main._save_node_config")
    @patch("asc.cli.main._load_node_config")
    def test_no_api_key_in_config_uses_env(self, mock_load, mock_save, mock_ip, mock_asyncio):
        """配置文件中无 api_key 时，仅使用环境变量。"""
        config = {
            "node_id": "test-master",
            "role": "master",
            "api_config": {"enabled": True},
        }
        mock_load.return_value = config
        mock_master = MagicMock()

        with patch("asc.master.main.MasterNode", return_value=mock_master), \
             patch.dict(os.environ, {"ASC_ADMIN_API_KEY": "env-only-key"}, clear=False):
            _cmd_master(self._make_args())

        from asc.api.auth import _get_default_store
        store = _get_default_store()
        assert store.require_admin("env-only-key") is True


# ---------------------------------------------------------------------------
# 问题4：基准测试失败日志降级 + --skip-benchmark 参数
# ---------------------------------------------------------------------------


class TestSkipBenchmark:
    """WorkerAgent 应支持 --skip-benchmark 参数。"""

    def test_skip_benchmark_flag(self):
        """skip_benchmark=True 时 get_resources 不调用 run_benchmark。"""
        agent = WorkerAgent(node_id="test", port=0, skip_benchmark=True)

        with patch.object(agent._benchmark_score, "run_benchmark") as mock_bench, \
             patch("psutil.cpu_count", return_value=4), \
             patch("psutil.cpu_percent", return_value=10.0), \
             patch("psutil.virtual_memory") as mock_vm, \
             patch.object(agent._hardware_detector, "detect_cpu") as mock_cpu, \
             patch.object(agent._hardware_detector, "detect_gpus", return_value=[]), \
             patch.object(agent._hardware_detector, "detect_disk"), \
             patch.object(agent._hardware_detector, "detect_network"):
            mock_vm.return_value = MagicMock(total=16 * 1024**3, available=12 * 1024**3)
            mock_cpu.return_value = MagicMock(logical_count=4, physical_count=2, freq_mhz=2400.0, brand="TestCPU")
            res = agent.get_resources()
            mock_bench.assert_not_called()
            assert res.compute_score == 0.0

    def test_benchmark_file_not_found_no_traceback(self):
        """基准模型文件缺失时应只打 warning 而非 traceback。"""
        agent = WorkerAgent(node_id="test", port=0)

        with patch.object(agent._benchmark_score, "run_benchmark", side_effect=FileNotFoundError("model not found")), \
             patch("psutil.cpu_count", return_value=4), \
             patch("psutil.cpu_percent", return_value=10.0), \
             patch("psutil.virtual_memory") as mock_vm, \
             patch.object(agent._hardware_detector, "detect_cpu") as mock_cpu, \
             patch.object(agent._hardware_detector, "detect_gpus", return_value=[]), \
             patch.object(agent._hardware_detector, "detect_disk"), \
             patch.object(agent._hardware_detector, "detect_network"):
            mock_vm.return_value = MagicMock(total=16 * 1024**3, available=12 * 1024**3)
            mock_cpu.return_value = MagicMock(logical_count=4, physical_count=2, freq_mhz=2400.0, brand="TestCPU")
            res = agent.get_resources()
            assert res.compute_score == 0.0

    def test_skip_benchmark_cli_flag(self):
        """--skip-benchmark 命令行参数应被正确解析。"""
        with patch("sys.argv", ["asc", "start", "--skip-benchmark"]):
            with patch("asc.cli.main._cmd_start") as mock_start:
                main()
        args = mock_start.call_args[0][0]
        assert args.skip_benchmark is True


# ---------------------------------------------------------------------------
# 问题5：Worker 注册使用异步方法 + TCPClient.connect 日志
# ---------------------------------------------------------------------------


class TestWorkerRegistrationAsync:
    """_register_to_master 应使用异步 get_resources_async。"""

    @pytest.mark.asyncio
    async def test_register_uses_async_resources(self):
        """_register_to_master 应调用 get_resources_async 而非 get_resources。"""
        agent = WorkerAgent(node_id="test-node", port=53414)

        mock_client = MagicMock()
        mock_client.send = MagicMock(return_value=True)
        # 让 MagicMock 的 send 返回协程
        async def mock_send(envelope):
            return True
        mock_client.send = mock_send
        agent._client = mock_client

        with patch.object(agent, "get_resources_async") as mock_async:
            from asc.worker.agent import NodeResources
            mock_async.return_value = NodeResources(
                cpu_count=8, cpu_percent=25.0, memory_total_mb=32768, memory_free_mb=24000,
            )
            await agent._register_to_master()
            mock_async.assert_called_once()


class TestTCPClientConnectLogging:
    """TCPClient.connect 应记录连接失败原因。"""

    @pytest.mark.asyncio
    async def test_connect_failure_logs_oserror(self):
        """连接失败时应记录 OSError 日志。"""
        from asc.network.transport import TCPClient

        client = TCPClient(host="192.0.2.1", port=1, node_id="test")

        with patch("asyncio.open_connection", side_effect=OSError("Connection refused")):
            with patch("asc.network.transport.logger") as mock_logger:
                result = await client.connect()
                assert result is False
                mock_logger.warning.assert_called_once()
                # 验证日志消息中包含错误信息
                call_args = mock_logger.warning.call_args
                assert "Connection refused" in str(call_args)


# ---------------------------------------------------------------------------
# 问题6：ASC_CONFIG_DIR 环境变量 + --config 启动参数
# ---------------------------------------------------------------------------


class TestConfigDirEnvVar:
    """_get_config_dir 应支持 ASC_CONFIG_DIR 环境变量。"""

    def test_default_config_dir(self):
        """默认配置目录为 ~/.asc。"""
        with patch.dict(os.environ, {}, clear=False):
            if "ASC_CONFIG_DIR" in os.environ:
                del os.environ["ASC_CONFIG_DIR"]
            config_dir = _get_config_dir()
            assert config_dir == Path.home() / ".asc"

    def test_custom_config_dir_from_env(self):
        """ASC_CONFIG_DIR 环境变量应覆盖默认路径。"""
        with patch.dict(os.environ, {"ASC_CONFIG_DIR": "/tmp/asc-test-config"}):
            config_dir = _get_config_dir()
            assert config_dir == Path("/tmp/asc-test-config")


class TestConfigFilePath:
    """--config 参数应覆盖配置文件路径。"""

    def test_set_config_path(self):
        """_set_config_path 应设置自定义配置路径。"""
        import asc.cli.main as main_mod
        # 保存原始值
        original = main_mod._custom_config_path
        try:
            main_mod._custom_config_path = None
            _set_config_path("/tmp/test_config.json")
            assert main_mod._custom_config_path == Path("/tmp/test_config.json").resolve()
        finally:
            main_mod._custom_config_path = original

    def test_load_config_from_custom_path(self):
        """--config 指定的路径应被 _load_node_config 使用。"""
        import asc.cli.main as main_mod
        original = main_mod._custom_config_path
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump({"node_id": "custom-path-node", "role": "worker"}, f)
                f.flush()
                temp_path = f.name

            main_mod._custom_config_path = Path(temp_path).resolve()
            config = _load_node_config()
            assert config["node_id"] == "custom-path-node"
            os.unlink(temp_path)
        finally:
            main_mod._custom_config_path = original

    def test_save_config_to_custom_path(self):
        """_save_node_config 应保存到 --config 指定的路径。"""
        import asc.cli.main as main_mod
        original = main_mod._custom_config_path
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                config_path = Path(tmpdir) / "custom_config.json"
                main_mod._custom_config_path = config_path
                _save_node_config({"node_id": "saved-node", "role": "master"})
                assert config_path.exists()
                data = json.loads(config_path.read_text(encoding="utf-8"))
                assert data["node_id"] == "saved-node"
        finally:
            main_mod._custom_config_path = original

    def test_start_config_cli_flag(self):
        """--config 命令行参数应被正确解析。"""
        with patch("sys.argv", ["asc", "start", "--config", "/tmp/my_config.json"]):
            with patch("asc.cli.main._cmd_start") as mock_start:
                main()
        args = mock_start.call_args[0][0]
        assert args.config == "/tmp/my_config.json"

    def test_master_config_cli_flag(self):
        """master --config 命令行参数应被正确解析。"""
        with patch("sys.argv", ["asc", "master", "--config", "/tmp/master_config.json"]):
            with patch("asc.cli.main._cmd_master") as mock_master:
                main()
        args = mock_master.call_args[0][0]
        assert args.config == "/tmp/master_config.json"
