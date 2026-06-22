"""模型同步协议。

实现节点间模型文件的分片传输：
- 分片下载（每片 10MB）
- 断点续传支持
- SHA256 校验完整性
- 下载进度实时上报
- 带宽控制（令牌桶）
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from asc.api.auth import validate_model_id as _validate_model_id_bool

CHUNK_SIZE: int = 10 * 1024 * 1024  # 10MB


def _validate_model_id(model_id: str) -> None:
    """验证 model_id 安全性，防止路径遍历。

    Raises:
        ValueError: model_id 包含路径遍历字符
    """
    if not _validate_model_id_bool(model_id):
        raise ValueError(f"model_id 包含非法字符: {model_id!r}")


def _safe_model_path(models_dir: Path, model_id: str) -> Path:
    """安全拼接模型路径，验证解析后路径仍在 models_dir 内。

    Raises:
        ValueError: 路径遍历攻击
    """
    _validate_model_id(model_id)
    file_path = models_dir / f"{model_id}.gguf"
    # 验证解析后路径仍在 models_dir 内
    resolved = file_path.resolve()
    models_dir_resolved = models_dir.resolve()
    try:
        resolved.relative_to(models_dir_resolved)
    except ValueError:
        raise ValueError(f"路径遍历检测: {model_id}") from None
    return resolved


@dataclass(frozen=True)
class ChunkInfo:
    """分片信息。"""

    chunk_index: int
    offset: int
    size: int
    sha256: str


@dataclass(frozen=True)
class SyncProgress:
    """同步进度。"""

    model_id: str
    total_bytes: int
    downloaded_bytes: int
    total_chunks: int
    completed_chunks: int

    @property
    def fraction(self) -> float:
        if self.total_bytes == 0:
            return 0.0
        return self.downloaded_bytes / self.total_bytes


class BandwidthLimiter:
    """简单带宽限制器（令牌桶）。"""

    def __init__(self, max_bytes_per_sec: int) -> None:
        self._max_rate = max_bytes_per_sec
        self._tokens = float(max_bytes_per_sec)
        self._last_time = 0.0

    def acquire(self, bytes_needed: int) -> float:
        """获取传输许可，返回需要等待的秒数。"""
        import time

        now = time.time()
        elapsed = now - self._last_time
        self._tokens = min(self._max_rate, self._tokens + elapsed * self._max_rate)
        self._last_time = now

        if self._tokens >= bytes_needed:
            self._tokens -= bytes_needed
            return 0.0

        deficit = bytes_needed - self._tokens
        wait_time = deficit / self._max_rate
        self._tokens = 0.0
        return wait_time


def compute_file_sha256(file_path: str | Path) -> str:
    """计算文件 SHA256。"""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_chunk_sha256(data: bytes) -> str:
    """计算分片 SHA256。"""
    return hashlib.sha256(data).hexdigest()


def split_file_into_chunks_with_data(
    file_path: str | Path,
) -> list[tuple[ChunkInfo, bytes]]:
    """将文件分割为分片信息列表，同时返回分片数据（单次 I/O）。"""
    path = Path(file_path)
    total_size = path.stat().st_size
    result: list[tuple[ChunkInfo, bytes]] = []

    offset = 0
    index = 0
    with open(path, "rb") as f:
        while offset < total_size:
            size = min(CHUNK_SIZE, total_size - offset)
            f.seek(offset)
            data = f.read(size)
            sha = compute_chunk_sha256(data)
            chunk_info = ChunkInfo(
                chunk_index=index,
                offset=offset,
                size=size,
                sha256=sha,
            )
            result.append((chunk_info, data))
            offset += size
            index += 1

    return result


def split_file_into_chunks(file_path: str | Path) -> list[ChunkInfo]:
    """将文件分割为分片信息列表（不预读数据，不计算 SHA256）。

    分片信息中的 sha256 字段为空字符串，调用方应在传输/验证时
    使用 compute_chunk_sha256() 自行计算。若需预计算 SHA256，
    请使用 split_file_into_chunks_with_data()。
    """
    path = Path(file_path)
    total_size = path.stat().st_size
    result: list[ChunkInfo] = []

    offset = 0
    index = 0
    while offset < total_size:
        size = min(CHUNK_SIZE, total_size - offset)
        chunk_info = ChunkInfo(
            chunk_index=index,
            offset=offset,
            size=size,
            sha256="",  # 延迟计算，调用方应自行验证
        )
        result.append(chunk_info)
        offset += size
        index += 1

    return result


def read_chunk(file_path: str | Path, chunk: ChunkInfo) -> bytes:
    """从文件读取指定分片。"""
    with open(file_path, "rb") as f:
        f.seek(chunk.offset)
        return f.read(chunk.size)


def write_chunk(
    file_path: str | Path,
    chunk: ChunkInfo,
    data: bytes,
) -> None:
    """将分片写入文件（支持断点续传）。"""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 预分配文件空间
    if not path.exists():
        path.touch()

    with open(path, "r+b") as f:
        f.seek(chunk.offset)
        f.write(data)


def verify_chunk(chunk: ChunkInfo, data: bytes) -> bool:
    """验证分片数据完整性。空 sha256 跳过验证。"""
    if not chunk.sha256:
        return True
    return compute_chunk_sha256(data) == chunk.sha256


class ModelSyncProtocol:
    """模型同步协议。

    负责分片下载的发起、接收和进度追踪。
    """

    def __init__(
        self,
        models_dir: Path,
        bandwidth_limit_mbps: float = 100.0,
        progress_callback: Callable[[SyncProgress], None] | None = None,
    ) -> None:
        self._models_dir = models_dir
        self._limiter = BandwidthLimiter(
            max_bytes_per_sec=int(bandwidth_limit_mbps * 1024 * 1024)
        )
        self._progress_callback = progress_callback

    def prepare_send(
        self,
        model_id: str,
    ) -> tuple[list[ChunkInfo], str]:
        """准备发送模型，返回分片列表和文件 SHA256。

        Args:
            model_id: 模型 ID

        Returns:
            (分片列表, 文件 SHA256)
        """
        file_path = _safe_model_path(self._models_dir, model_id)
        if not file_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {file_path}")

        chunks_with_data = split_file_into_chunks_with_data(file_path)
        chunks = [info for info, _data in chunks_with_data]
        # 利用已读取的数据计算文件 SHA256，避免二次 I/O
        h = hashlib.sha256()
        for _info, data in chunks_with_data:
            h.update(data)
        file_sha = h.hexdigest()
        return chunks, file_sha

    def get_chunk_data(
        self,
        model_id: str,
        chunk: ChunkInfo,
    ) -> bytes:
        """获取指定分片的数据。"""
        file_path = _safe_model_path(self._models_dir, model_id)
        return read_chunk(file_path, chunk)

    def init_receive(
        self,
        model_id: str,
        total_bytes: int,
        chunks: list[ChunkInfo],
        file_sha256: str,
    ) -> "ReceiveState":
        """初始化接收状态。

        Args:
            model_id: 模型 ID
            total_bytes: 文件总大小
            chunks: 分片列表
            file_sha256: 文件 SHA256

        Returns:
            ReceiveState 接收状态对象
        """
        file_path = _safe_model_path(self._models_dir, model_id)
        return ReceiveState(
            model_id=model_id,
            file_path=file_path,
            total_bytes=total_bytes,
            chunks=chunks,
            file_sha256=file_sha256,
            protocol=self,
        )

    def _report_progress(self, progress: SyncProgress) -> None:
        if self._progress_callback is not None:
            self._progress_callback(progress)


class ReceiveState:
    """接收状态（支持断点续传）。"""

    def __init__(
        self,
        model_id: str,
        file_path: Path,
        total_bytes: int,
        chunks: list[ChunkInfo],
        file_sha256: str,
        protocol: ModelSyncProtocol,
    ) -> None:
        self.model_id = model_id
        self.file_path = file_path
        self.total_bytes = total_bytes
        self.chunks = chunks
        self.file_sha256 = file_sha256
        self._protocol = protocol
        self._completed: set[int] = set()
        self._received_bytes: int = 0

        # 预分配文件（保留已有数据以支持断点续传）
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if not file_path.exists():
            # 新文件：预分配空间
            with open(file_path, "wb") as f:
                f.seek(total_bytes - 1)
                f.write(b"\x00")
        elif file_path.stat().st_size < total_bytes:
            # 已有文件但不够大：扩展到目标大小
            with open(file_path, "r+b") as f:
                f.seek(total_bytes - 1)
                f.write(b"\x00")

    @property
    def is_complete(self) -> bool:
        return len(self._completed) == len(self.chunks)

    @property
    def completed_chunks(self) -> set[int]:
        """已完成的分片索引集合（只读副本）。"""
        return self._completed.copy()

    @property
    def completed_count(self) -> int:
        """已完成的分片数量。"""
        return len(self._completed)

    @property
    def missing_chunks(self) -> list[ChunkInfo]:
        return [c for c in self.chunks if c.chunk_index not in self._completed]

    def receive_chunk(self, chunk: ChunkInfo, data: bytes) -> bool:
        """接收一个分片。

        Returns:
            True 如果分片验证通过并写入成功
        """
        if not verify_chunk(chunk, data):
            return False

        write_chunk(self.file_path, chunk, data)
        self._completed.add(chunk.chunk_index)
        self._received_bytes += chunk.size

        self._protocol._report_progress(
            SyncProgress(
                model_id=self.model_id,
                total_bytes=self.total_bytes,
                downloaded_bytes=self._received_bytes,
                total_chunks=len(self.chunks),
                completed_chunks=len(self._completed),
            )
        )
        return True

    def verify_file(self) -> bool:
        """验证完整文件 SHA256。

        Returns:
            True 如果文件完整且校验通过
        """
        if not self.is_complete:
            return False
        actual = compute_file_sha256(self.file_path)
        return actual == self.file_sha256

    def save_progress(self, progress_file: Path | None = None) -> None:
        """保存下载进度到文件。"""
        path = progress_file or self.file_path.with_suffix(".gguf.progress")
        with open(path, "w") as f:
            f.write(
                struct.pack(
                    "!QQ",
                    self.total_bytes,
                    len(self._completed),
                ).hex()
                + "\n"
            )
            for idx in sorted(self._completed):
                f.write(f"{idx}\n")

    def load_progress(self, progress_file: Path | None = None) -> bool:
        """从文件加载下载进度。

        Returns:
            True 如果进度文件存在且有效
        """
        path = progress_file or self.file_path.with_suffix(".gguf.progress")
        if not path.exists():
            return False

        try:
            with open(path) as f:
                header = bytes.fromhex(f.readline().strip())
                total_bytes, completed_count = struct.unpack("!QQ", header)
                if total_bytes != self.total_bytes:
                    return False

                self._completed.clear()
                for _ in range(completed_count):
                    line = f.readline().strip()
                    if line:
                        self._completed.add(int(line))
                # 从已完成的分片重新计算 _received_bytes
                self._received_bytes = sum(
                    c.size for c in self.chunks if c.chunk_index in self._completed
                )
            return True
        except (ValueError, struct.error, OSError):
            return False
