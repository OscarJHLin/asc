"""Asc 模型下载器。

支持三种下载方式：
- Hugging Face：通过 huggingface_hub 下载
- ModelScope：通过 modelscope CLI 下载（首次自动安装）
- 手动下载：用户自行下载到指定目录

所有下载的模型统一保存到 models_dir 目录，
并自动扫描注册。
"""

from __future__ import annotations

import contextlib
import enum
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator


class DownloadSource(enum.Enum):
    """下载来源。"""

    HUGGINGFACE = "huggingface"
    MODELSCOPE = "modelscope"
    MANUAL = "manual"


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DownloadRequest:
    """下载请求。"""

    source: DownloadSource
    model_uri: str  # HF repo_id 或 ModelScope 模型 ID
    filename: str | None = None  # 指定文件名（HF 可选）
    target_dir: Path | None = None  # 覆盖默认目录


@dataclass(frozen=True)
class DownloadResult:
    """下载结果。"""

    success: bool
    source: DownloadSource
    model_id: str
    file_path: str  # 下载后的本地路径
    file_size_mb: int
    error: str = ""


class ModelDownloader:
    """统一模型下载器。

    支持 HuggingFace / ModelScope / 手动下载三种方式。
    所有模型统一保存到 models_dir，自动扫描发现。
    """

    def __init__(self, models_dir: Path) -> None:
        self._models_dir = models_dir
        self._models_dir.mkdir(parents=True, exist_ok=True)

    @property
    def models_dir(self) -> Path:
        """模型保存目录。"""
        return self._models_dir

    def download(
        self, request: DownloadRequest
    ) -> Generator[dict[str, Any], None, DownloadResult]:
        """执行模型下载。

        Args:
            request: 下载请求

        Yields:
            进度信息字典 {"stage": str, "progress": float, "detail": str}

        Returns:
            DownloadResult
        """
        target_dir = request.target_dir or self._models_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        if request.source == DownloadSource.HUGGINGFACE:
            return self._download_huggingface(request, target_dir)
        elif request.source == DownloadSource.MODELSCOPE:
            return self._download_modelscope(request, target_dir)
        elif request.source == DownloadSource.MANUAL:
            return self._download_manual(request, target_dir)
        else:
            return DownloadResult(
                success=False,
                source=request.source,
                model_id=request.model_uri,
                file_path="",
                file_size_mb=0,
                error=f"不支持的下载来源: {request.source}",
            )

    def _download_huggingface(
        self,
        request: DownloadRequest,
        target_dir: Path,
    ) -> Generator[dict[str, Any], None, DownloadResult]:
        """从 HuggingFace 下载模型。"""
        try:
            from huggingface_hub import hf_hub_download, list_repo_files
        except ImportError:
            return DownloadResult(
                success=False,
                source=DownloadSource.HUGGINGFACE,
                model_id=request.model_uri,
                file_path="",
                file_size_mb=0,
                error="请安装 huggingface_hub: pip install huggingface_hub",
            )

        repo_id = request.model_uri
        filename = request.filename

        # 自动查找 .gguf 文件
        if filename is None:
            yield {
                "stage": "scanning", "progress": 0.0,
                "detail": f"扫描 {repo_id} 中的 GGUF 文件...",
            }
            try:
                files = list_repo_files(repo_id)
                gguf_files = [f for f in files if f.endswith(".gguf")]
                if gguf_files:
                    filename = gguf_files[0]
                    yield {"stage": "scanning", "progress": 1.0, "detail": f"找到: {filename}"}
                else:
                    return DownloadResult(
                        success=False,
                        source=DownloadSource.HUGGINGFACE,
                        model_id=repo_id,
                        file_path="",
                        file_size_mb=0,
                        error=f"未在 {repo_id} 中找到 .gguf 文件",
                    )
            except Exception as e:
                return DownloadResult(
                    success=False,
                    source=DownloadSource.HUGGINGFACE,
                    model_id=repo_id,
                    file_path="",
                    file_size_mb=0,
                    error=f"扫描仓库失败: {e}",
                )

        yield {
            "stage": "downloading", "progress": 0.0,
            "detail": f"从 HuggingFace 下载 {filename}...",
        }

        try:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(target_dir),
            )
        except Exception as e:
            return DownloadResult(
                success=False,
                source=DownloadSource.HUGGINGFACE,
                model_id=repo_id,
                file_path="",
                file_size_mb=0,
                error=f"下载失败: {e}",
            )

        # 如果下载到了子目录，移动到 models_dir 根目录
        local_file = Path(local_path)
        if local_file.exists() and local_file.parent != target_dir:
            dest = target_dir / local_file.name
            if not dest.exists():
                shutil.move(str(local_file), str(dest))
                local_file = dest
            else:
                local_file = dest

        size_mb = int(local_file.stat().st_size // (1024 * 1024)) if local_file.exists() else 0
        model_id = Path(filename).stem

        yield {"stage": "complete", "progress": 1.0, "detail": f"下载完成: {local_file}"}

        return DownloadResult(
            success=True,
            source=DownloadSource.HUGGINGFACE,
            model_id=model_id,
            file_path=str(local_file),
            file_size_mb=size_mb,
        )

    def _download_modelscope(
        self,
        request: DownloadRequest,
        target_dir: Path,
    ) -> Generator[dict[str, Any], None, DownloadResult]:
        """从 ModelScope 下载模型。

        首次使用自动安装 modelscope。
        下载命令: modelscope download --model <model_id> --local_dir <dir>
        """
        # 检查并安装 modelscope
        yield {"stage": "checking", "progress": 0.0, "detail": "检查 ModelScope CLI..."}
        if not self._is_modelscope_installed():
            yield {
                "stage": "installing", "progress": 0.1,
                "detail": "首次使用，正在安装 modelscope...",
            }
            install_ok = self._install_modelscope()
            if not install_ok:
                return DownloadResult(
                    success=False,
                    source=DownloadSource.MODELSCOPE,
                    model_id=request.model_uri,
                    file_path="",
                    file_size_mb=0,
                    error="安装 modelscope 失败，请手动执行: pip install modelscope",
                )
            yield {"stage": "installing", "progress": 0.3, "detail": "modelscope 安装成功"}

        model_id = request.model_uri

        # 解析 ModelScope 链接或直接使用模型 ID
        resolved_id = self._parse_modelscope_uri(model_id)
        yield {
            "stage": "downloading", "progress": 0.3,
            "detail": f"从 ModelScope 下载 {resolved_id}...",
        }

        # 执行下载命令
        cmd = [
            sys_executable(), "-m", "modelscope", "download",
            "--model", resolved_id,
            "--local_dir", str(target_dir),
        ]

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            try:
                if process.stdout is not None:
                    for line in process.stdout:
                        line = line.strip()
                        if line:
                            yield {"stage": "downloading", "progress": 0.5, "detail": line}
            finally:
                if process.stdout is not None:
                    with contextlib.suppress(Exception):
                        process.stdout.close()

            returncode = process.wait()
            if returncode != 0:
                return DownloadResult(
                    success=False,
                    source=DownloadSource.MODELSCOPE,
                    model_id=model_id,
                    file_path="",
                    file_size_mb=0,
                    error=f"modelscope 下载失败 (exit code {returncode})",
                )
        except FileNotFoundError:
            return DownloadResult(
                success=False,
                source=DownloadSource.MODELSCOPE,
                model_id=model_id,
                file_path="",
                file_size_mb=0,
                error="未找到 modelscope 命令，请手动执行: pip install modelscope",
            )
        except Exception as e:
            return DownloadResult(
                success=False,
                source=DownloadSource.MODELSCOPE,
                model_id=model_id,
                file_path="",
                file_size_mb=0,
                error=f"下载异常: {e}",
            )

        # 扫描下载的 .gguf 文件
        gguf_files = list(target_dir.glob("**/*.gguf"))
        if not gguf_files:
            # 可能下载了非 gguf 文件，仍然标记成功
            yield {"stage": "complete", "progress": 1.0, "detail": f"下载完成，保存到 {target_dir}"}
            return DownloadResult(
                success=True,
                source=DownloadSource.MODELSCOPE,
                model_id=model_id,
                file_path=str(target_dir),
                file_size_mb=0,
            )

        # 如果 gguf 在子目录中，移动到 models_dir 根目录
        main_gguf = gguf_files[0]
        if main_gguf.parent != self._models_dir:
            dest = self._models_dir / main_gguf.name
            if not dest.exists():
                shutil.move(str(main_gguf), str(dest))
                main_gguf = dest

        size_mb = int(main_gguf.stat().st_size // (1024 * 1024))
        resolved_model_id = main_gguf.stem

        yield {"stage": "complete", "progress": 1.0, "detail": f"下载完成: {main_gguf}"}

        return DownloadResult(
            success=True,
            source=DownloadSource.MODELSCOPE,
            model_id=resolved_model_id,
            file_path=str(main_gguf),
            file_size_mb=size_mb,
        )

    def _download_manual(
        self,
        request: DownloadRequest,
        target_dir: Path,
    ) -> Generator[dict[str, Any], None, DownloadResult]:
        """手动下载：扫描指定目录中的模型文件。"""
        yield {
            "stage": "scanning", "progress": 0.0,
            "detail": f"扫描目录 {target_dir} 中的模型文件...",
        }

        gguf_files = list(target_dir.glob("*.gguf"))
        if not gguf_files:
            return DownloadResult(
                success=False,
                source=DownloadSource.MANUAL,
                model_id=request.model_uri,
                file_path="",
                file_size_mb=0,
                error=f"未在 {target_dir} 中找到 .gguf 文件",
            )

        # 返回第一个找到的 gguf 文件
        gguf = gguf_files[0]
        size_mb = int(gguf.stat().st_size // (1024 * 1024))
        model_id = gguf.stem

        yield {"stage": "complete", "progress": 1.0, "detail": f"发现模型: {gguf.name}"}

        return DownloadResult(
            success=True,
            source=DownloadSource.MANUAL,
            model_id=model_id,
            file_path=str(gguf),
            file_size_mb=size_mb,
        )

    def scan_models(self) -> list[dict[str, Any]]:
        """扫描 models_dir 中所有可用的 GGUF 模型文件。

        Returns:
            模型信息列表 [{"model_id": str, "file_path": str, "file_size_mb": int}]
        """
        models: list[dict[str, Any]] = []
        if not self._models_dir.exists():
            return models

        for gguf_file in sorted(self._models_dir.glob("*.gguf")):
            size_mb = int(gguf_file.stat().st_size // (1024 * 1024))
            models.append({
                "model_id": gguf_file.stem,
                "file_path": str(gguf_file.resolve()),
                "file_size_mb": size_mb,
            })

        return models

    def _is_modelscope_installed(self) -> bool:
        """检查 modelscope 是否已安装。"""
        try:
            import modelscope  # noqa: F401
            return True
        except ImportError:
            pass
        # 也检查 CLI
        return shutil.which("modelscope") is not None

    def _install_modelscope(self) -> bool:
        """通过 pip 安装 modelscope。"""
        try:
            result = subprocess.run(
                [sys_executable(), "-m", "pip", "install", "modelscope"],
                capture_output=True,
                text=True,
                timeout=300,
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            logger.warning("pip install modelscope 超时，子进程可能未完全清理")
            return False
        except Exception:
            return False

    @staticmethod
    def _parse_modelscope_uri(uri: str) -> str:
        """解析 ModelScope URI。

        支持格式：
        - 直接模型 ID: "unsloth/Qwen3.6-27B-GGUF"
        - 社区链接: "https://modelscope.cn/models/unsloth/Qwen3.6-27B-GGUF"
        - 下载链接: "https://modelscope.cn/models/unsloth/Qwen3.6-27B-GGUF/files"
        """
        if uri.startswith("https://modelscope.cn/models/"):
            # 提取模型 ID
            path = uri.replace("https://modelscope.cn/models/", "")
            # 去掉尾部路径如 /files, /download 等
            parts = path.split("/")
            if len(parts) >= 2:
                return f"{parts[0]}/{parts[1]}"
            return parts[0]
        return uri


def sys_executable() -> str:
    """获取当前 Python 解释器路径。"""
    return os.sys.executable or "python"
