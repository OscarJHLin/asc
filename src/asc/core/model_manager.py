"""Asc 模型管理器。

支持：
- 本地模型注册/注销
- 自动发现 models 目录下的 .gguf 文件
- HuggingFace 下载（通过 huggingface_hub）
- 下载进度追踪
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator


@dataclass(frozen=True)
class ModelMetadata:
    """模型元数据。"""

    model_id: str
    file_path: str
    file_size_mb: int
    source: str = "local"  # "local" | "huggingface"


@dataclass(frozen=True)
class DownloadProgress:
    """下载进度。"""

    model_id: str
    downloaded_mb: int
    total_mb: int
    speed_mbps: float

    @property
    def fraction(self) -> float:
        if self.total_mb == 0:
            return 0.0
        return self.downloaded_mb / self.total_mb


class ModelManager:
    """模型管理器。"""

    def __init__(self, models_dir: Path) -> None:
        self._models_dir = models_dir
        self._registry: dict[str, ModelMetadata] = {}

    def register(self, model_id: str, file_path: str) -> None:
        """注册本地模型。"""
        path = Path(file_path)
        size_mb = 0
        if path.exists():
            size_mb = int(path.stat().st_size // (1024 * 1024))

        self._registry[model_id] = ModelMetadata(
            model_id=model_id,
            file_path=file_path,
            file_size_mb=size_mb,
            source="local",
        )

    def unregister(self, model_id: str) -> None:
        """注销模型。"""
        self._registry.pop(model_id, None)

    def resolve(self, model_id: str) -> str | None:
        """解析模型路径。"""
        meta = self._registry.get(model_id)
        if meta is not None:
            return meta.file_path
        return None

    def list_models(self) -> list[ModelMetadata]:
        """列出所有已注册模型。"""
        return list(self._registry.values())

    def auto_discover(self) -> None:
        """自动发现 models 目录下的 .gguf 文件。"""
        if not self._models_dir.exists():
            return

        for gguf_file in self._models_dir.glob("*.gguf"):
            model_id = gguf_file.stem  # 文件名去掉 .gguf
            if model_id not in self._registry:
                self.register(model_id, str(gguf_file.resolve()))

    def download(
        self, repo_id: str, filename: str | None = None
    ) -> Generator[DownloadProgress, None, str]:
        """从 HuggingFace 下载模型。

        Args:
            repo_id: HuggingFace 仓库 ID（如 "TheBloke/Llama-3.1-8B-Q4_K_M"）
            filename: 仓库中的文件名（默认自动查找 .gguf）

        Yields:
            DownloadProgress 下载进度

        Returns:
            下载后的本地文件路径
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError("请安装 huggingface_hub: pip install huggingface_hub") from None

        if filename is None:
            filename = self._find_gguf_filename(repo_id)

        yield DownloadProgress(
            model_id=repo_id,
            downloaded_mb=0,
            total_mb=0,
            speed_mbps=0,
        )

        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(self._models_dir),
        )

        # 注册下载的模型
        model_id = Path(filename).stem
        self.register(model_id, local_path)

        return local_path

    def _find_gguf_filename(self, repo_id: str) -> str:
        """在 HuggingFace 仓库中查找 .gguf 文件。"""
        try:
            from huggingface_hub import list_repo_files

            files = list_repo_files(repo_id)
            gguf_files = [f for f in files if f.endswith(".gguf")]
            if gguf_files:
                return gguf_files[0]
        except Exception:
            pass
        raise FileNotFoundError(f"未在 {repo_id} 中找到 .gguf 文件")
