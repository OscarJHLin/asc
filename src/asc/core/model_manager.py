"""Asc 模型管理器。

支持：
- 本地模型注册/注销
- 自动发现 models 目录下的 .gguf 文件
- 多源下载：HuggingFace / ModelScope / 手动
- 下载进度追踪
- 模型分发到集群节点
- 远程模型同步（通过 ModelSyncProtocol）
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Generator

from asc.core.model_distributor import DistributionTarget, ModelDistributor
from asc.core.model_downloader import (
    DownloadRequest,
    DownloadResult,
    DownloadSource,
    ModelDownloader,
)
from asc.core.model_registry import ModelRegistry
from asc.network.sync import ModelSyncProtocol, SyncProgress


@dataclass(frozen=True)
class ModelMetadata:
    """模型元数据。"""

    model_id: str
    file_path: str
    file_size_mb: int
    source: str = "local"  # "local" | "huggingface" | "modelscope" | "manual" | "remote"


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
    """模型管理器。

    统一管理模型下载、注册、发现和分发。
    """

    def __init__(
        self,
        models_dir: Path,
        node_id: str | None = None,
        registry: ModelRegistry | None = None,
        sync_protocol: ModelSyncProtocol | None = None,
    ) -> None:
        self._models_dir = models_dir
        self._node_id = node_id or "local"
        self._registry = registry
        self._sync = sync_protocol
        self._registry_local: dict[str, ModelMetadata] = {}
        self._downloader = ModelDownloader(models_dir)
        self._distributor = ModelDistributor(models_dir)

    @property
    def models_dir(self) -> Path:
        """模型保存目录。"""
        return self._models_dir

    @property
    def downloader(self) -> ModelDownloader:
        """模型下载器。"""
        return self._downloader

    @property
    def distributor(self) -> ModelDistributor:
        """模型分发器。"""
        return self._distributor

    def register(self, model_id: str, file_path: str, source: str = "local") -> None:
        """注册本地模型。"""
        path = Path(file_path)
        size_mb = 0
        if path.exists():
            size_mb = int(path.stat().st_size // (1024 * 1024))

        self._registry_local[model_id] = ModelMetadata(
            model_id=model_id,
            file_path=file_path,
            file_size_mb=size_mb,
            source=source,
        )

        # 同步到集群注册表
        if self._registry is not None:
            self._registry.register(
                node_id=self._node_id,
                model_id=model_id,
                path=file_path,
                size_mb=size_mb,
            )

    def unregister(self, model_id: str) -> None:
        """注销模型。"""
        self._registry_local.pop(model_id, None)
        if self._registry is not None:
            self._registry.unregister(self._node_id, model_id)

    def resolve(self, model_id: str) -> str | None:
        """解析模型路径。"""
        meta = self._registry_local.get(model_id)
        if meta is not None:
            return meta.file_path
        return None

    def list_models(self) -> list[ModelMetadata]:
        """列出所有已注册模型。"""
        return list(self._registry_local.values())

    def auto_discover(self) -> list[str]:
        """自动发现 models 目录下的 .gguf 文件。

        Returns:
            新发现的模型 ID 列表
        """
        discovered: list[str] = []
        if not self._models_dir.exists():
            return discovered

        for gguf_file in self._models_dir.glob("*.gguf"):
            model_id = gguf_file.stem  # 文件名去掉 .gguf
            if model_id not in self._registry_local:
                self.register(model_id, str(gguf_file.resolve()), source="local")
                discovered.append(model_id)

        return discovered

    def scan_available_models(self) -> list[dict[str, Any]]:
        """扫描 models 目录中所有可用的 GGUF 模型文件。

        不依赖注册表，直接扫描文件系统。

        Returns:
            模型信息列表
        """
        return self._downloader.scan_models()

    def download_model(
        self,
        source: DownloadSource,
        model_uri: str,
        filename: str | None = None,
        target_dir: Path | None = None,
    ) -> Generator[dict[str, Any], None, DownloadResult]:
        """统一模型下载入口。

        Args:
            source: 下载来源 (HuggingFace / ModelScope / Manual)
            model_uri: 模型标识 (HF repo_id / ModelScope 模型 ID 或链接)
            filename: 指定文件名（HF 可选）
            target_dir: 覆盖默认保存目录

        Yields:
            进度信息字典

        Returns:
            DownloadResult
        """
        request = DownloadRequest(
            source=source,
            model_uri=model_uri,
            filename=filename,
            target_dir=target_dir,
        )

        gen = self._downloader.download(request)
        result = yield from gen

        # 下载成功后自动注册
        if result.success and result.file_path:
            self.register(
                model_id=result.model_id,
                file_path=result.file_path,
                source=result.source.value,
            )

        return result

    def distribute_model(
        self,
        model_id: str,
        targets: list[DistributionTarget],
        send_chunk_fn: Any | None = None,
    ) -> Generator[dict[str, Any], None, list]:
        """将模型分发到集群节点。

        Args:
            model_id: 模型 ID
            targets: 目标节点列表
            send_chunk_fn: 发送分片的回调函数

        Yields:
            进度信息字典
        """
        gen = self._distributor.distribute(model_id, targets, send_chunk_fn)
        results = yield from gen
        return results

    def download(
        self, repo_id: str, filename: str | None = None
    ) -> Generator[DownloadProgress, None, str]:
        """从 HuggingFace 下载模型（兼容旧接口）。

        Args:
            repo_id: HuggingFace 仓库 ID
            filename: 仓库中的文件名

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
        self.register(model_id, local_path, source="huggingface")

        return local_path

    def sync_from_node(
        self,
        model_id: str,
        source_node_id: str,
        get_chunk_fn: Callable[[str, Any], Any],
    ) -> Generator[SyncProgress, None, str]:
        """从远程节点同步模型。

        Args:
            model_id: 模型 ID
            source_node_id: 源节点 ID
            get_chunk_fn: 获取分片数据的回调函数

        Yields:
            SyncProgress 同步进度

        Returns:
            同步后的本地文件路径
        """
        if self._sync is None:
            raise RuntimeError("未配置 ModelSyncProtocol")

        result = get_chunk_fn("prepare", model_id)
        chunks, file_sha256 = result

        total_bytes = sum(c.size for c in chunks)
        state = self._sync.init_receive(
            model_id=model_id,
            total_bytes=total_bytes,
            chunks=chunks,
            file_sha256=file_sha256,
        )

        state.load_progress()

        for chunk in state.missing_chunks:
            data = get_chunk_fn("chunk", chunk)
            state.receive_chunk(chunk, data)
            yield SyncProgress(
                model_id=model_id,
                total_bytes=total_bytes,
                downloaded_bytes=sum(
                    c.size for c in chunks if c.chunk_index in state._completed
                ),
                total_chunks=len(chunks),
                completed_chunks=len(state._completed),
            )

        if not state.verify_file():
            raise RuntimeError("模型文件校验失败")

        progress_file = state.file_path.with_suffix(".gguf.progress")
        if progress_file.exists():
            progress_file.unlink()

        self.register(model_id, str(state.file_path), source="remote")

        return str(state.file_path)

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
