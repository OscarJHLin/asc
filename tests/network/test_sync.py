"""模型同步协议测试。"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from asc.network.sync import (
    CHUNK_SIZE,
    BandwidthLimiter,
    ChunkInfo,
    ModelSyncProtocol,
    SyncProgress,
    compute_chunk_sha256,
    compute_file_sha256,
    read_chunk,
    split_file_into_chunks,
    verify_chunk,
    write_chunk,
)


class TestBandwidthLimiter:
    def test_acquire_immediate(self):
        limiter = BandwidthLimiter(max_bytes_per_sec=1024 * 1024)
        wait = limiter.acquire(100)
        assert wait == 0.0

    def test_acquire_waits_when_over_limit(self):
        limiter = BandwidthLimiter(max_bytes_per_sec=100)
        limiter.acquire(100)  # 用完令牌
        wait = limiter.acquire(100)
        assert wait > 0.0


class TestSha256:
    def test_compute_file_sha256(self, tmp_path: Path):
        file_path = tmp_path / "test.bin"
        file_path.write_bytes(b"hello world")
        sha = compute_file_sha256(file_path)
        expected = hashlib.sha256(b"hello world").hexdigest()
        assert sha == expected

    def test_compute_chunk_sha256(self):
        data = b"test data"
        sha = compute_chunk_sha256(data)
        expected = hashlib.sha256(data).hexdigest()
        assert sha == expected


class TestChunkOperations:
    def test_split_file_into_chunks(self, tmp_path: Path):
        file_path = tmp_path / "test.bin"
        data = b"a" * (CHUNK_SIZE + 100)
        file_path.write_bytes(data)

        chunks = split_file_into_chunks(file_path)
        assert len(chunks) == 2
        assert chunks[0].size == CHUNK_SIZE
        assert chunks[1].size == 100
        assert chunks[0].offset == 0
        assert chunks[1].offset == CHUNK_SIZE

    def test_read_chunk(self, tmp_path: Path):
        file_path = tmp_path / "test.bin"
        file_path.write_bytes(b"hello world")
        chunk = ChunkInfo(chunk_index=0, offset=0, size=5, sha256="")
        data = read_chunk(file_path, chunk)
        assert data == b"hello"

    def test_write_chunk(self, tmp_path: Path):
        file_path = tmp_path / "test.bin"
        chunk = ChunkInfo(chunk_index=0, offset=0, size=5, sha256="")
        write_chunk(file_path, chunk, b"hello")
        assert file_path.read_bytes() == b"hello"

    def test_verify_chunk(self):
        data = b"test data"
        sha = compute_chunk_sha256(data)
        chunk = ChunkInfo(chunk_index=0, offset=0, size=len(data), sha256=sha)
        assert verify_chunk(chunk, data)

    def test_verify_chunk_invalid(self):
        data = b"test data"
        chunk = ChunkInfo(chunk_index=0, offset=0, size=len(data), sha256="invalid")
        assert not verify_chunk(chunk, data)


class TestModelSyncProtocol:
    def test_prepare_send(self, tmp_path: Path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        file_path = models_dir / "model1.gguf"
        file_path.write_bytes(b"x" * 1000)

        protocol = ModelSyncProtocol(models_dir=models_dir)
        chunks, file_sha = protocol.prepare_send("model1")

        assert len(chunks) == 1
        assert chunks[0].size == 1000
        assert file_sha == compute_file_sha256(file_path)

    def test_prepare_send_not_found(self, tmp_path: Path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        protocol = ModelSyncProtocol(models_dir=models_dir)
        with pytest.raises(FileNotFoundError):
            protocol.prepare_send("model1")

    def test_get_chunk_data(self, tmp_path: Path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        file_path = models_dir / "model1.gguf"
        file_path.write_bytes(b"hello world")

        protocol = ModelSyncProtocol(models_dir=models_dir)
        chunk = ChunkInfo(chunk_index=0, offset=0, size=5, sha256="")
        data = protocol.get_chunk_data("model1", chunk)
        assert data == b"hello"

    def test_init_receive(self, tmp_path: Path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        protocol = ModelSyncProtocol(models_dir=models_dir)

        chunks = [ChunkInfo(chunk_index=0, offset=0, size=10, sha256="abc")]
        state = protocol.init_receive(
            model_id="model1",
            total_bytes=10,
            chunks=chunks,
            file_sha256="xyz",
        )

        assert state.model_id == "model1"
        assert state.total_bytes == 10
        assert not state.is_complete

    def test_progress_callback(self, tmp_path: Path):
        progress_calls: list[SyncProgress] = []

        def callback(p: SyncProgress) -> None:
            progress_calls.append(p)

        models_dir = tmp_path / "models"
        models_dir.mkdir()
        protocol = ModelSyncProtocol(
            models_dir=models_dir, progress_callback=callback
        )

        chunks = [ChunkInfo(chunk_index=0, offset=0, size=10, sha256="")]
        state = protocol.init_receive(
            model_id="model1",
            total_bytes=10,
            chunks=chunks,
            file_sha256="xyz",
        )
        state.receive_chunk(chunks[0], b"x" * 10)

        assert len(progress_calls) == 1
        assert progress_calls[0].downloaded_bytes == 10


class TestReceiveState:
    def test_receive_and_verify(self, tmp_path: Path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        protocol = ModelSyncProtocol(models_dir=models_dir)

        data = b"hello world test data"
        file_sha = compute_file_sha256_from_bytes(data)
        chunks = split_file_into_chunks_from_bytes(data)

        state = protocol.init_receive(
            model_id="model1",
            total_bytes=len(data),
            chunks=chunks,
            file_sha256=file_sha,
        )

        for chunk in chunks:
            chunk_data = data[chunk.offset : chunk.offset + chunk.size]
            assert state.receive_chunk(chunk, chunk_data)

        assert state.is_complete
        assert state.verify_file()

    def test_receive_invalid_chunk(self, tmp_path: Path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        protocol = ModelSyncProtocol(models_dir=models_dir)

        chunk = ChunkInfo(chunk_index=0, offset=0, size=5, sha256="invalid")
        state = protocol.init_receive(
            model_id="model1",
            total_bytes=5,
            chunks=[chunk],
            file_sha256="xyz",
        )
        assert not state.receive_chunk(chunk, b"hello")

    def test_save_and_load_progress(self, tmp_path: Path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        protocol = ModelSyncProtocol(models_dir=models_dir)

        chunks = [
            ChunkInfo(chunk_index=0, offset=0, size=5, sha256=""),
            ChunkInfo(chunk_index=1, offset=5, size=5, sha256=""),
        ]
        state = protocol.init_receive(
            model_id="model1",
            total_bytes=10,
            chunks=chunks,
            file_sha256="xyz",
        )
        state.receive_chunk(chunks[0], b"hello")

        progress_file = tmp_path / "progress.txt"
        state.save_progress(progress_file)

        # 新状态加载进度
        state2 = protocol.init_receive(
            model_id="model1",
            total_bytes=10,
            chunks=chunks,
            file_sha256="xyz",
        )
        assert state2.load_progress(progress_file)
        assert 0 in state2._completed
        assert 1 not in state2._completed

    def test_load_progress_invalid_total(self, tmp_path: Path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        protocol = ModelSyncProtocol(models_dir=models_dir)

        chunks = [ChunkInfo(chunk_index=0, offset=0, size=5, sha256="")]
        state = protocol.init_receive(
            model_id="model1",
            total_bytes=10,
            chunks=chunks,
            file_sha256="xyz",
        )

        # 伪造一个错误的进度文件
        progress_file = tmp_path / "bad.progress"
        with open(progress_file, "w") as f:
            import struct
            f.write(struct.pack("!QQ", 99, 0).hex() + "\n")

        assert not state.load_progress(progress_file)

    def test_missing_chunks(self, tmp_path: Path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        protocol = ModelSyncProtocol(models_dir=models_dir)

        chunks = [
            ChunkInfo(chunk_index=0, offset=0, size=5, sha256=""),
            ChunkInfo(chunk_index=1, offset=5, size=5, sha256=""),
        ]
        state = protocol.init_receive(
            model_id="model1",
            total_bytes=10,
            chunks=chunks,
            file_sha256="xyz",
        )
        state.receive_chunk(chunks[0], b"hello")

        missing = state.missing_chunks
        assert len(missing) == 1
        assert missing[0].chunk_index == 1


# 辅助函数

def compute_file_sha256_from_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def split_file_into_chunks_from_bytes(data: bytes) -> list[ChunkInfo]:
    chunks: list[ChunkInfo] = []
    offset = 0
    index = 0
    while offset < len(data):
        size = min(CHUNK_SIZE, len(data) - offset)
        chunk_data = data[offset : offset + size]
        sha = compute_chunk_sha256(chunk_data)
        chunks.append(
            ChunkInfo(
                chunk_index=index,
                offset=offset,
                size=size,
                sha256=sha,
            )
        )
        offset += size
        index += 1
    return chunks
