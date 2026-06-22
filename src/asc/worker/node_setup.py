"""NodeSetup 首次安装自动检测与配置模块。

首次运行时自动完成以下步骤：
1. 检测操作系统和硬件平台
2. 检查 llama-server 可执行文件
3. 下载基准测试模型（Qwen2.5-0.5B-Instruct Q4_K_M，仅 0.3GB）
4. 运行基准性能测试，计算节点算力评分
5. 保存安装结果，后续启动时跳过

设计原则：
- 幂等：重复运行不会重复下载或测试
- 容错：基准测试失败不影响节点启动（评分默认 0.0）
- 跨平台：支持 Linux / Windows / macOS
"""

from __future__ import annotations

import json
import logging
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from asc.core.model_downloader import DownloadRequest, DownloadSource, ModelDownloader
from asc.utils.system import find_executable
from asc.worker.benchmark_score import BenchmarkScore
from asc.worker.hardware import HardwareDetector

logger = logging.getLogger(__name__)

# 默认基准模型 HuggingFace 仓库和文件
DEFAULT_BENCHMARK_MODEL_REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
DEFAULT_BENCHMARK_MODEL_FILE = "qwen2.5-0.5b-instruct-q4_k_m.gguf"

# 基准模型候选列表（与 BenchmarkScore._BENCHMARK_MODEL_CANDIDATES 保持一致）
_BENCHMARK_MODEL_CANDIDATES = [
    "gemma-4-E2B-it-UD-Q2_K_XL.gguf",
    "qwen2.5-0.5b-instruct-q4_k_m.gguf",
    "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
]


@dataclass
class PlatformInfo:
    """平台信息。"""

    os: str
    arch: str
    python_version: str
    _gpu_available: bool = field(default=False, repr=False)

    @classmethod
    def detect(cls) -> PlatformInfo:
        """检测当前平台信息。"""
        info = cls(
            os=platform.system(),
            arch=platform.machine(),
            python_version=platform.python_version(),
        )
        # 检测 GPU
        detector = HardwareDetector()
        gpus = detector.detect_gpus()
        info._gpu_available = len(gpus) > 0
        return info

    @property
    def is_linux(self) -> bool:
        return self.os == "Linux"

    @property
    def is_windows(self) -> bool:
        return self.os == "Windows"

    @property
    def is_macos(self) -> bool:
        return self.os == "Darwin"

    @property
    def has_gpu(self) -> bool:
        return self._gpu_available

    def to_dict(self) -> dict[str, Any]:
        return {
            "os": self.os,
            "arch": self.arch,
            "python_version": self.python_version,
            "has_gpu": self._gpu_available,
        }


@dataclass
class SetupStep:
    """安装步骤状态。"""

    name: str
    status: str  # pending / completed / failed / skipped
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
        }


@dataclass
class NodeSetupResult:
    """安装结果。"""

    success: bool
    platform_info: PlatformInfo
    steps: list[SetupStep]
    benchmark_score: float
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "platform_info": self.platform_info.to_dict(),
            "steps": [s.to_dict() for s in self.steps],
            "benchmark_score": self.benchmark_score,
            "timestamp": self.timestamp,
        }

    def save(self, path: Path) -> None:
        """保存安装结果到 JSON 文件。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> NodeSetupResult:
        """从 JSON 文件加载安装结果。"""
        data = json.loads(path.read_text(encoding="utf-8"))
        pi = data["platform_info"]
        platform_info = PlatformInfo(
            os=pi["os"],
            arch=pi["arch"],
            python_version=pi["python_version"],
            _gpu_available=pi.get("has_gpu", False),
        )
        steps = [
            SetupStep(name=s["name"], status=s["status"], message=s.get("message", ""))
            for s in data.get("steps", [])
        ]
        return cls(
            success=data["success"],
            platform_info=platform_info,
            steps=steps,
            benchmark_score=data.get("benchmark_score", 0.0),
            timestamp=data.get("timestamp", ""),
        )


class NodeSetup:
    """首次安装自动检测与配置。

    使用流程：
        setup = NodeSetup(node_id="worker-01")
        if setup.is_first_run():
            result = setup.run()
            if result.success:
                print(f"安装完成，性能评分: {result.benchmark_score}")
    """

    def __init__(
        self,
        node_id: str,
        models_dir: Path | None = None,
        result_path: Path | None = None,
        skip_benchmark: bool = False,
        non_interactive: bool = False,
    ) -> None:
        self.node_id = node_id
        self.models_dir = models_dir or Path("./models")
        self._result_path = result_path or Path.home() / ".asc" / f"setup_{node_id}.json"
        self._platform_info: PlatformInfo | None = None
        self._skip_benchmark = skip_benchmark
        self._non_interactive = non_interactive

    def is_first_run(self) -> bool:
        """判断是否为首次运行。"""
        return not self._result_path.exists()

    def run(self) -> NodeSetupResult:
        """执行完整的安装流程。"""
        steps: list[SetupStep] = []
        benchmark_score = 0.0

        # Step 1: 检测平台
        step = self._step_detect_platform()
        steps.append(step)
        if step.status == "failed":
            return NodeSetupResult(
                success=False,
                platform_info=self._platform_info or PlatformInfo(os="Unknown", arch="Unknown", python_version=""),
                steps=steps,
                benchmark_score=0.0,
            )

        # Step 2: 检查 llama-server
        step = self._step_check_llama_server()
        steps.append(step)
        if step.status == "failed":
            # llama-server 缺失不阻塞，标记为 skipped 继续
            step = SetupStep(name="check_llama_server", status="skipped", message="llama-server 未找到（可稍后安装）")
            steps[-1] = step

        if self._skip_benchmark:
            # 跳过基准测试相关步骤
            steps.append(SetupStep(name="download_model", status="skipped", message="用户指定跳过"))
            steps.append(SetupStep(name="run_benchmark", status="skipped", message="用户指定跳过"))
        elif self._non_interactive:
            # 非交互模式下，仅当模型文件已存在时才运行基准测试
            model_exists = any(
                (self.models_dir / c).exists() for c in _BENCHMARK_MODEL_CANDIDATES
            ) or bool(os.getenv("ASC_BENCHMARK_MODEL"))
            if model_exists:
                step = self._step_download_model()
                steps.append(step)
                step = self._step_run_benchmark()
                steps.append(step)
                if step.status == "completed":
                    try:
                        benchmark_score = float(step.message.split("=")[-1].strip())
                    except (ValueError, IndexError):
                        benchmark_score = 0.0
            else:
                steps.append(SetupStep(name="download_model", status="skipped", message="非交互模式下跳过模型下载"))
                steps.append(SetupStep(name="run_benchmark", status="skipped", message="非交互模式下跳过基准测试"))
        else:
            # Step 3: 下载基准模型
            step = self._step_download_model()
            steps.append(step)
            # 模型下载失败不阻塞，基准测试会跳过

            # Step 4: 运行基准测试
            step = self._step_run_benchmark()
            steps.append(step)
            if step.status == "completed":
                # 从 message 中解析评分
                try:
                    benchmark_score = float(step.message.split("=")[-1].strip())
                except (ValueError, IndexError):
                    benchmark_score = 0.0

        # 判断整体是否成功：至少平台检测和 llama-server 检查通过
        success = all(
            s.status in ("completed", "skipped")
            for s in steps[:2]  # 前2步必须成功
        )

        result = NodeSetupResult(
            success=success,
            platform_info=self._platform_info,
            steps=steps,
            benchmark_score=benchmark_score,
        )

        # 保存结果（无论成功与否都保存，避免反复重试首次设置）
        result.save(self._result_path)
        if success:
            logger.info("Node setup completed. Score: %.2f", benchmark_score)
        else:
            logger.warning("Node setup completed with issues. Score: %.2f", benchmark_score)

        return result

    def _step_detect_platform(self) -> SetupStep:
        """Step 1: 检测操作系统和硬件。"""
        try:
            self._platform_info = PlatformInfo.detect()
            gpu_info = "with GPU" if self._platform_info.has_gpu else "CPU only"
            msg = f"{self._platform_info.os} {self._platform_info.arch} ({gpu_info})"
            logger.info("Platform: %s", msg)
            return SetupStep(name="detect_platform", status="completed", message=msg)
        except Exception as e:
            return SetupStep(name="detect_platform", status="failed", message=str(e))

    def _step_check_llama_server(self) -> SetupStep:
        """Step 2: 检查 llama-server 可执行文件或 llama-cpp-python。"""
        exe = find_executable("llama-server")
        if exe is not None:
            msg = f"llama-server found: {exe}"
            logger.info(msg)
            return SetupStep(name="check_llama_server", status="completed", message=msg)

        # 检查 llama-cpp-python 作为 fallback
        try:
            from llama_cpp import Llama  # noqa: F401
            msg = "llama-server not found, using llama-cpp-python as fallback"
            logger.info(msg)
            return SetupStep(name="check_llama_server", status="completed", message=msg)
        except ImportError:
            pass

        msg = "llama-server not found and llama-cpp-python not installed. Install llama.cpp or: pip install llama-cpp-python"
        logger.warning(msg)
        return SetupStep(name="check_llama_server", status="failed", message=msg)

    def _step_download_model(self) -> SetupStep:
        """Step 3: 下载基准测试模型。"""
        # 检查模型是否已存在
        model_candidates = [self.models_dir / c for c in _BENCHMARK_MODEL_CANDIDATES]
        for candidate in model_candidates:
            if candidate.exists():
                msg = f"模型已存在: {candidate}"
                logger.info(msg)
                return SetupStep(name="download_model", status="completed", message=msg)

        # 从 HuggingFace 下载
        self.models_dir.mkdir(parents=True, exist_ok=True)
        downloader = ModelDownloader(self.models_dir)

        request = DownloadRequest(
            source=DownloadSource.HUGGINGFACE,
            model_uri=DEFAULT_BENCHMARK_MODEL_REPO,
            filename=DEFAULT_BENCHMARK_MODEL_FILE,
        )

        try:
            # ModelDownloader.download 是 Generator，返回值通过 StopIteration.value 获取
            gen = downloader.download(request)
            result = None
            while True:
                try:
                    progress = next(gen)
                    logger.debug("Download progress: %s", progress)
                except StopIteration as e:
                    result = e.value
                    break

            if result and result.success:
                msg = f"模型下载完成: {result.file_path} ({result.file_size_mb}MB)"
                logger.info(msg)
                return SetupStep(name="download_model", status="completed", message=msg)
            else:
                error = result.error if result else "Unknown error"
                msg = f"模型下载失败: {error}"
                logger.warning(msg)
                return SetupStep(name="download_model", status="failed", message=msg)
        except Exception as e:
            msg = f"模型下载异常: {e}"
            logger.warning(msg)
            return SetupStep(name="download_model", status="failed", message=msg)

    def _step_run_benchmark(self) -> SetupStep:
        """Step 4: 运行基准性能测试。"""
        try:
            benchmark = BenchmarkScore(node_id=self.node_id)
            report = benchmark.run_benchmark()
            msg = f"score={report.relative_score:.4f}"
            logger.info("Benchmark completed: avg_tps=%.2f, %s", report.avg_tps, msg)
            return SetupStep(name="benchmark", status="completed", message=msg)
        except FileNotFoundError as e:
            msg = f"基准测试跳过: {e}"
            logger.warning(msg)
            return SetupStep(name="benchmark", status="skipped", message=msg)
        except Exception as e:
            msg = f"基准测试失败: {e}"
            logger.warning(msg)
            return SetupStep(name="benchmark", status="skipped", message=msg)
