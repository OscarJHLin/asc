"""测试 NodeSetup 首次安装自动检测与配置模块。"""

from __future__ import annotations

import json
import platform
from pathlib import Path
from unittest.mock import MagicMock, patch

from asc.worker.node_setup import (
    NodeSetup,
    NodeSetupResult,
    PlatformInfo,
    SetupStep,
)

# ------------------------------------------------------------------
# PlatformInfo 测试
# ------------------------------------------------------------------


class TestPlatformInfo:
    """测试平台信息检测。"""

    def test_detect_current_platform(self):
        """能检测当前运行平台的 OS、架构、Python 版本。"""
        info = PlatformInfo.detect()
        assert info.os in ("Linux", "Windows", "Darwin")
        assert info.arch in ("x86_64", "AMD64", "aarch64", "arm64")
        assert info.python_version == platform.python_version()

    @patch("platform.system", return_value="Linux")
    @patch("platform.machine", return_value="x86_64")
    def test_detect_linux(self, _mock_machine, _mock_system):
        info = PlatformInfo.detect()
        assert info.os == "Linux"
        assert info.arch == "x86_64"
        assert info.is_linux is True
        assert info.is_windows is False
        assert info.is_macos is False

    @patch("platform.system", return_value="Windows")
    @patch("platform.machine", return_value="AMD64")
    def test_detect_windows(self, _mock_machine, _mock_system):
        info = PlatformInfo.detect()
        assert info.os == "Windows"
        assert info.is_windows is True

    @patch("platform.system", return_value="Darwin")
    @patch("platform.machine", return_value="arm64")
    def test_detect_macos(self, _mock_machine, _mock_system):
        info = PlatformInfo.detect()
        assert info.os == "Darwin"
        assert info.is_macos is True

    def test_has_gpu_with_nvidia(self):
        """检测到 NVIDIA GPU 时 has_gpu 为 True。"""
        info = PlatformInfo(os="Linux", arch="x86_64", python_version="3.12.0")
        info._gpu_available = True
        assert info.has_gpu is True

    def test_has_gpu_without_gpu(self):
        """无 GPU 时 has_gpu 为 False。"""
        info = PlatformInfo(os="Linux", arch="x86_64", python_version="3.12.0")
        info._gpu_available = False
        assert info.has_gpu is False

    def test_to_dict(self):
        """能序列化为字典。"""
        info = PlatformInfo(os="Linux", arch="x86_64", python_version="3.12.0")
        d = info.to_dict()
        assert d["os"] == "Linux"
        assert d["arch"] == "x86_64"
        assert "python_version" in d


# ------------------------------------------------------------------
# SetupStep 测试
# ------------------------------------------------------------------


class TestSetupStep:
    """测试安装步骤状态。"""

    def test_step_pending(self):
        step = SetupStep(name="detect_os", status="pending")
        assert step.name == "detect_os"
        assert step.status == "pending"
        assert step.message == ""

    def test_step_completed(self):
        step = SetupStep(name="detect_os", status="completed", message="Linux x86_64")
        assert step.status == "completed"
        assert step.message == "Linux x86_64"

    def test_step_failed(self):
        step = SetupStep(name="download_model", status="failed", message="Network error")
        assert step.status == "failed"
        assert step.message == "Network error"


# ------------------------------------------------------------------
# NodeSetupResult 测试
# ------------------------------------------------------------------


class TestNodeSetupResult:
    """测试安装结果。"""

    def test_success_result(self):
        result = NodeSetupResult(
            success=True,
            platform_info=PlatformInfo(os="Linux", arch="x86_64", python_version="3.12.0"),
            steps=[
                SetupStep(name="detect_os", status="completed", message="Linux x86_64"),
                SetupStep(name="download_model", status="completed", message="OK"),
                SetupStep(name="benchmark", status="completed", message="score=1.5"),
            ],
            benchmark_score=1.5,
        )
        assert result.success is True
        assert result.benchmark_score == 1.5
        assert len(result.steps) == 3

    def test_partial_failure_result(self):
        result = NodeSetupResult(
            success=False,
            platform_info=PlatformInfo(os="Linux", arch="x86_64", python_version="3.12.0"),
            steps=[
                SetupStep(name="detect_os", status="completed", message="OK"),
                SetupStep(name="download_model", status="failed", message="No network"),
            ],
            benchmark_score=0.0,
        )
        assert result.success is False
        assert result.benchmark_score == 0.0

    def test_to_dict(self):
        result = NodeSetupResult(
            success=True,
            platform_info=PlatformInfo(os="Linux", arch="x86_64", python_version="3.12.0"),
            steps=[SetupStep(name="detect_os", status="completed")],
            benchmark_score=1.5,
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["benchmark_score"] == 1.5
        assert "platform_info" in d
        assert "steps" in d


# ------------------------------------------------------------------
# NodeSetup 核心流程测试
# ------------------------------------------------------------------


class TestNodeSetup:
    """测试 NodeSetup 首次安装流程。"""

    def test_init_default_models_dir(self, tmp_path):
        """默认 models 目录使用 ASC_MODELS_PATH 或 ./models。"""
        setup = NodeSetup(node_id="test-node", models_dir=tmp_path / "models")
        assert setup.models_dir == tmp_path / "models"

    @patch("asc.worker.node_setup.PlatformInfo.detect")
    def test_step_detect_platform(self, mock_detect, tmp_path):
        """第一步：检测平台信息。"""
        mock_detect.return_value = PlatformInfo(
            os="Linux", arch="x86_64", python_version="3.12.0"
        )
        setup = NodeSetup(node_id="test-node", models_dir=tmp_path / "models")
        step = setup._step_detect_platform()
        assert step.status == "completed"
        assert "Linux" in step.message

    @patch("asc.worker.node_setup.PlatformInfo.detect")
    def test_step_detect_platform_with_gpu(self, mock_detect, tmp_path):
        """检测平台时同时检测 GPU。"""
        info = PlatformInfo(os="Linux", arch="x86_64", python_version="3.12.0")
        info._gpu_available = True
        mock_detect.return_value = info
        setup = NodeSetup(node_id="test-node", models_dir=tmp_path / "models")
        step = setup._step_detect_platform()
        assert "GPU" in step.message

    def test_step_check_llama_server_found(self, tmp_path):
        """llama-server 已安装时跳过。"""
        setup = NodeSetup(node_id="test-node", models_dir=tmp_path / "models")
        with patch("asc.worker.node_setup.find_executable", return_value=Path("/usr/bin/llama-server")):
            step = setup._step_check_llama_server()
        assert step.status == "completed"
        assert "found" in step.message.lower() or "已" in step.message

    def test_step_check_llama_server_not_found(self, tmp_path):
        """llama-server 未安装且 llama-cpp-python 也不可用时标记为 failed。"""
        setup = NodeSetup(node_id="test-node", models_dir=tmp_path / "models")
        with patch("asc.worker.node_setup.find_executable", return_value=None):
            with patch.dict("sys.modules", {"llama_cpp": None}):
                step = setup._step_check_llama_server()
        assert step.status == "failed"

    def test_step_check_llama_server_fallback_llama_cpp(self, tmp_path):
        """llama-server 未安装但 llama-cpp-python 可用时标记为 completed。"""
        setup = NodeSetup(node_id="test-node", models_dir=tmp_path / "models")
        with patch("asc.worker.node_setup.find_executable", return_value=None):
            with patch.dict("sys.modules", {"llama_cpp": MagicMock()}):
                step = setup._step_check_llama_server()
        assert step.status == "completed"
        assert "fallback" in step.message.lower()

    def test_step_download_model_already_exists(self, tmp_path):
        """模型文件已存在时跳过下载。"""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        model_file = models_dir / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
        model_file.write_text("fake model")

        setup = NodeSetup(node_id="test-node", models_dir=models_dir)
        step = setup._step_download_model()
        assert step.status == "completed"
        assert "已存在" in step.message or "skip" in step.message.lower()

    @patch("asc.worker.node_setup.ModelDownloader")
    def test_step_download_model_from_hf(self, mock_downloader_cls, tmp_path):
        """模型不存在时从 HuggingFace 下载。"""
        models_dir = tmp_path / "models"

        mock_downloader = MagicMock()
        mock_downloader.models_dir = models_dir
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.file_path = str(models_dir / "qwen2.5-0.5b-instruct-q4_k_m.gguf")
        mock_result.file_size_mb = 300

        # Generator that yields progress then returns result
        def fake_download(request):
            yield {"stage": "downloading", "progress": 0.5, "detail": "Downloading..."}
            return mock_result

        mock_downloader.download = fake_download
        mock_downloader_cls.return_value = mock_downloader

        setup = NodeSetup(node_id="test-node", models_dir=models_dir)
        step = setup._step_download_model()
        assert step.status == "completed"

    @patch("asc.worker.node_setup.BenchmarkScore")
    def test_step_run_benchmark_success(self, mock_benchmark_cls, tmp_path):
        """基准测试成功时返回评分。"""
        mock_benchmark = MagicMock()
        mock_report = MagicMock()
        mock_report.relative_score = 2.5
        mock_benchmark.run_benchmark.return_value = mock_report
        mock_benchmark_cls.return_value = mock_benchmark

        setup = NodeSetup(node_id="test-node", models_dir=tmp_path / "models")
        step = setup._step_run_benchmark()
        assert step.status == "completed"
        assert "2.5" in step.message

    @patch("asc.worker.node_setup.BenchmarkScore")
    def test_step_run_benchmark_failure(self, mock_benchmark_cls, tmp_path):
        """基准测试失败时标记为 skipped（不影响整体安装）。"""
        mock_benchmark = MagicMock()
        mock_benchmark.run_benchmark.side_effect = FileNotFoundError("No model")
        mock_benchmark_cls.return_value = mock_benchmark

        setup = NodeSetup(node_id="test-node", models_dir=tmp_path / "models")
        step = setup._step_run_benchmark()
        assert step.status == "skipped"

    @patch("asc.worker.node_setup.BenchmarkScore")
    @patch("asc.worker.node_setup.ModelDownloader")
    @patch("asc.worker.node_setup.find_executable", return_value=Path("/usr/bin/llama-server"))
    @patch("asc.worker.node_setup.PlatformInfo.detect")
    def test_run_full_setup_success(self, mock_detect, mock_find, mock_dl_cls, mock_bm_cls, tmp_path):
        """完整安装流程成功。"""
        mock_detect.return_value = PlatformInfo(
            os="Linux", arch="x86_64", python_version="3.12.0"
        )

        models_dir = tmp_path / "models"
        mock_downloader = MagicMock()
        mock_downloader.models_dir = models_dir
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.file_path = str(models_dir / "qwen2.5-0.5b-instruct-q4_k_m.gguf")
        mock_result.file_size_mb = 300

        def fake_download(request):
            yield {"stage": "downloading", "progress": 0.5, "detail": "Downloading..."}
            return mock_result

        mock_downloader.download = fake_download
        mock_dl_cls.return_value = mock_downloader

        mock_benchmark = MagicMock()
        mock_report = MagicMock()
        mock_report.relative_score = 2.5
        mock_benchmark.run_benchmark.return_value = mock_report
        mock_bm_cls.return_value = mock_benchmark

        setup = NodeSetup(node_id="test-node", models_dir=models_dir)
        result = setup.run()

        assert result.success is True
        assert result.benchmark_score == 2.5
        assert len(result.steps) >= 3

    @patch("asc.worker.node_setup.find_executable", return_value=None)
    @patch("asc.worker.node_setup.PlatformInfo.detect")
    def test_run_setup_llama_server_missing(self, mock_detect, mock_find, tmp_path):
        """llama-server 缺失时不阻塞安装，标记为 skipped。"""
        mock_detect.return_value = PlatformInfo(
            os="Linux", arch="x86_64", python_version="3.12.0"
        )

        setup = NodeSetup(node_id="test-node", models_dir=tmp_path / "models", skip_benchmark=True)
        result = setup.run()

        # llama-server 缺失不再阻塞，success=True
        assert result.success is True
        assert any(s.name == "check_llama_server" and s.status == "skipped" for s in result.steps)

    def test_save_setup_result(self, tmp_path):
        """安装结果能持久化到 JSON 文件。"""
        result = NodeSetupResult(
            success=True,
            platform_info=PlatformInfo(os="Linux", arch="x86_64", python_version="3.12.0"),
            steps=[SetupStep(name="detect_os", status="completed", message="OK")],
            benchmark_score=1.5,
        )
        output_path = tmp_path / "setup_result.json"
        result.save(output_path)

        assert output_path.exists()
        data = json.loads(output_path.read_text(encoding="utf-8"))
        assert data["success"] is True
        assert data["benchmark_score"] == 1.5

    def test_load_setup_result(self, tmp_path):
        """能从 JSON 文件加载安装结果。"""
        data = {
            "success": True,
            "platform_info": {"os": "Linux", "arch": "x86_64", "python_version": "3.12.0"},
            "steps": [{"name": "detect_os", "status": "completed", "message": "OK"}],
            "benchmark_score": 1.5,
        }
        output_path = tmp_path / "setup_result.json"
        output_path.write_text(json.dumps(data), encoding="utf-8")

        result = NodeSetupResult.load(output_path)
        assert result.success is True
        assert result.benchmark_score == 1.5

    def test_is_first_run_no_result_file(self, tmp_path):
        """无结果文件时判定为首次运行。"""
        result_path = tmp_path / ".asc" / "setup_test-node.json"
        setup = NodeSetup(node_id="test-node", models_dir=tmp_path / "models", result_path=result_path)
        assert setup.is_first_run() is True

    def test_is_first_run_with_result_file(self, tmp_path):
        """有结果文件时判定为非首次运行。"""
        result_path = tmp_path / ".asc" / "setup_test-node.json"
        setup = NodeSetup(node_id="test-node", models_dir=tmp_path / "models", result_path=result_path)
        result = NodeSetupResult(
            success=True,
            platform_info=PlatformInfo(os="Linux", arch="x86_64", python_version="3.12.0"),
            steps=[],
            benchmark_score=1.0,
        )
        result.save(setup._result_path)
        assert setup.is_first_run() is False
