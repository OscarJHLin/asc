"""增强测试模型同步协议。

覆盖分片边界条件、进度计算、带宽限制器边界、
ReceiveState 高级场景及 ModelSyncProtocol 细节。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from asc.network.sync import (
    CHUNK_SIZE,
    BandwidthLimiter,
    ChunkInfo,
    ModelSyncProtocol,
    SyncProgress,
    compute_chunk_sha256,
    split_file_into_chunks,
    verify_chunk,
)

# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _sha256(data: bytes) -> str:
    """计算 bytes 的 SHA256。"""
    return hashlib.sha256(data).hexdigest()


def _make_chunks_from_data(data: bytes) -> list[ChunkInfo]:
    """从 bytes 数据生成 ChunkInfo 列表。"""
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


# ---------------------------------------------------------------------------
# split_file_into_chunks 边界条件
# ---------------------------------------------------------------------------


class TestSplitFileIntoChunksEdgeCases:
    """split_file_into_chunks 边界条件测试。"""

    def test_empty_file(self, tmp_path: Path):
        """空文件（0字节）应返回空分片列表。"""
        file_path = tmp_path / "empty.bin"
        file_path.write_bytes(b"")
        chunks = split_file_into_chunks(file_path)
        assert chunks == []

    def test_file_exactly_chunk_size(self, tmp_path: Path):
        """文件恰好等于 CHUNK_SIZE 字节时应只有1个分片。"""
        file_path = tmp_path / "exact.bin"
        data = b"\x42" * CHUNK_SIZE
        file_path.write_bytes(data)
        chunks = split_file_into_chunks(file_path)
        assert len(chunks) == 1
        assert chunks[0].size == CHUNK_SIZE
        assert chunks[0].offset == 0
        assert chunks[0].chunk_index == 0

    def test_file_chunk_size_plus_one(self, tmp_path: Path):
        """文件 CHUNK_SIZE+1 字节应产生2个分片，第二个为1字节。"""
        file_path = tmp_path / "plus_one.bin"
        data = b"\x42" * (CHUNK_SIZE + 1)
        file_path.write_bytes(data)
        chunks = split_file_into_chunks(file_path)
        assert len(chunks) == 2
        assert chunks[0].size == CHUNK_SIZE
        assert chunks[1].size == 1
        assert chunks[1].offset == CHUNK_SIZE

    def test_file_much_smaller_than_chunk_size(self, tmp_path: Path):
        """文件远小于 CHUNK_SIZE 时应只有1个分片。"""
        file_path = tmp_path / "tiny.bin"
        data = b"hello"
        file_path.write_bytes(data)
        chunks = split_file_into_chunks(file_path)
        assert len(chunks) == 1
        assert chunks[0].size == 5
        assert chunks[0].sha256 == ""


# ---------------------------------------------------------------------------
# ChunkInfo 边界条件
# ---------------------------------------------------------------------------


class TestChunkInfoEdgeCases:
    """ChunkInfo 边界条件测试。"""

    def test_sha256_empty_string(self):
        """sha256 为空字符串时应正常创建。"""
        chunk = ChunkInfo(chunk_index=0, offset=0, size=10, sha256="")
        assert chunk.sha256 == ""

    def test_very_large_offset_and_size(self):
        """非常大的 offset/size 值应正常创建。"""
        chunk = ChunkInfo(
            chunk_index=999,
            offset=2**63 - 1,
            size=2**63 - 1,
            sha256="ab" * 32,
        )
        assert chunk.offset == 2**63 - 1
        assert chunk.size == 2**63 - 1


# ---------------------------------------------------------------------------
# SyncProgress 边界条件
# ---------------------------------------------------------------------------


class TestSyncProgressEdgeCases:
    """SyncProgress 边界条件测试。"""

    def test_fraction_total_bytes_zero(self):
        """total_bytes=0 时 fraction 应为 0.0。"""
        progress = SyncProgress(
            model_id="m1",
            total_bytes=0,
            downloaded_bytes=0,
            total_chunks=0,
            completed_chunks=0,
        )
        assert progress.fraction == 0.0

    def test_fraction_downloaded_bytes_zero(self):
        """downloaded_bytes=0 时 fraction 应为 0.0。"""
        progress = SyncProgress(
            model_id="m1",
            total_bytes=1000,
            downloaded_bytes=0,
            total_chunks=10,
            completed_chunks=0,
        )
        assert progress.fraction == 0.0

    def test_fraction_fully_downloaded(self):
        """完全下载时 fraction 应为 1.0。"""
        progress = SyncProgress(
            model_id="m1",
            total_bytes=1000,
            downloaded_bytes=1000,
            total_chunks=1,
            completed_chunks=1,
        )
        assert progress.fraction == 1.0


# ---------------------------------------------------------------------------
# BandwidthLimiter 边界条件
# ---------------------------------------------------------------------------


class TestBandwidthLimiterEdgeCases:
    """BandwidthLimiter 边界条件测试。"""

    def test_very_slow_rate(self):
        """max_bytes_per_sec=1 时，获取1字节后令牌耗尽。"""
        limiter = BandwidthLimiter(max_bytes_per_sec=1)
        wait = limiter.acquire(1)
        assert wait == 0.0
        # 第二次获取需要等待
        wait2 = limiter.acquire(1)
        assert wait2 > 0.0

    def test_acquire_zero_bytes(self):
        """获取0字节应返回0.0。"""
        limiter = BandwidthLimiter(max_bytes_per_sec=1024)
        wait = limiter.acquire(0)
        assert wait == 0.0

    def test_multiple_rapid_acquires(self):
        """多次快速获取应逐步耗尽令牌桶。"""
        limiter = BandwidthLimiter(max_bytes_per_sec=1000)
        # 首次获取部分令牌
        w1 = limiter.acquire(500)
        assert w1 == 0.0
        # 再次获取，令牌桶还有500
        w2 = limiter.acquire(500)
        assert w2 == 0.0
        # 第三次获取，令牌耗尽
        w3 = limiter.acquire(1)
        assert w3 > 0.0


# ---------------------------------------------------------------------------
# verify_chunk 边界条件
# ---------------------------------------------------------------------------


class TestVerifyChunkEdgeCases:
    """verify_chunk 边界条件测试。"""

    def test_empty_sha256_skips_verification(self):
        """sha256 为空字符串时应跳过验证，返回 True。"""
        chunk = ChunkInfo(chunk_index=0, offset=0, size=5, sha256="")
        assert verify_chunk(chunk, b"hello") is True

    def test_correct_sha256_passes(self):
        """正确的 sha256 应通过验证。"""
        data = b"test data"
        sha = compute_chunk_sha256(data)
        chunk = ChunkInfo(chunk_index=0, offset=0, size=len(data), sha256=sha)
        assert verify_chunk(chunk, data) is True

    def test_wrong_sha256_fails(self):
        """错误的 sha256 应验证失败。"""
        chunk = ChunkInfo(chunk_index=0, offset=0, size=9, sha256="00" * 32)
        assert verify_chunk(chunk, b"test data") is False


# ---------------------------------------------------------------------------
# ReceiveState 高级场景
# ---------------------------------------------------------------------------


class TestReceiveStateAdvanced:
    """ReceiveState 高级场景测试。"""

    def test_verify_file_when_incomplete(self, tmp_path: Path):
        """文件未接收完成时 verify_file 应返回 False。"""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        protocol = ModelSyncProtocol(models_dir=models_dir)

        chunks = [
            ChunkInfo(chunk_index=0, offset=0, size=5, sha256=""),
            ChunkInfo(chunk_index=1, offset=5, size=5, sha256=""),
        ]
        state = protocol.init_receive(
            model_id="m1",
            total_bytes=10,
            chunks=chunks,
            file_sha256="xyz",
        )
        # 只接收了第一个分片
        state.receive_chunk(chunks[0], b"hello")
        assert not state.is_complete
        assert state.verify_file() is False

    def test_receive_chunk_with_corrupted_data(self, tmp_path: Path):
        """接收损坏数据（sha256不匹配）应返回 False 且不标记完成。"""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        protocol = ModelSyncProtocol(models_dir=models_dir)

        data = b"hello"
        sha = compute_chunk_sha256(data)
        chunk = ChunkInfo(chunk_index=0, offset=0, size=5, sha256=sha)

        state = protocol.init_receive(
            model_id="m1",
            total_bytes=5,
            chunks=[chunk],
            file_sha256="xyz",
        )
        # 传入错误数据
        result = state.receive_chunk(chunk, b"world")
        assert result is False
        assert not state.is_complete

    def test_load_progress_with_corrupted_hex(self, tmp_path: Path):
        """加载损坏的进度文件（无效十六进制）应返回 False。"""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        protocol = ModelSyncProtocol(models_dir=models_dir)

        chunks = [ChunkInfo(chunk_index=0, offset=0, size=5, sha256="")]
        state = protocol.init_receive(
            model_id="m1",
            total_bytes=5,
            chunks=chunks,
            file_sha256="xyz",
        )
        # 写入无效的十六进制数据
        progress_file = tmp_path / "bad.progress"
        progress_file.write_text("ZZZZZZZZ\n0\n")
        assert state.load_progress(progress_file) is False

    def test_load_progress_from_nonexistent_file(self, tmp_path: Path):
        """从不存在的文件加载进度应返回 False。"""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        protocol = ModelSyncProtocol(models_dir=models_dir)

        chunks = [ChunkInfo(chunk_index=0, offset=0, size=5, sha256="")]
        state = protocol.init_receive(
            model_id="m1",
            total_bytes=5,
            chunks=chunks,
            file_sha256="xyz",
        )
        progress_file = tmp_path / "nonexistent.progress"
        assert state.load_progress(progress_file) is False

    def test_receive_chunks_out_of_order(self, tmp_path: Path):
        """乱序接收多个分片应全部成功。"""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        protocol = ModelSyncProtocol(models_dir=models_dir)

        data = b"0123456789"
        chunks = _make_chunks_from_data(data)
        # 确保有2个分片
        assert len(chunks) == 1  # 10 bytes < CHUNK_SIZE，只有1个分片

        # 使用更大的数据来产生多个分片
        # 改为手动创建2个分片
        c0_data = b"hello"
        c1_data = b"world"
        full_data = c0_data + c1_data
        full_sha = _sha256(full_data)
        c0 = ChunkInfo(
            chunk_index=0,
            offset=0,
            size=5,
            sha256=compute_chunk_sha256(c0_data),
        )
        c1 = ChunkInfo(
            chunk_index=1,
            offset=5,
            size=5,
            sha256=compute_chunk_sha256(c1_data),
        )

        state = protocol.init_receive(
            model_id="m1",
            total_bytes=10,
            chunks=[c0, c1],
            file_sha256=full_sha,
        )
        # 先接收第二个分片
        assert state.receive_chunk(c1, c1_data) is True
        assert not state.is_complete
        # 再接收第一个分片
        assert state.receive_chunk(c0, c0_data) is True
        assert state.is_complete
        assert state.verify_file() is True


# ---------------------------------------------------------------------------
# ModelSyncProtocol 细节
# ---------------------------------------------------------------------------


class TestModelSyncProtocolDetails:
    """ModelSyncProtocol 细节测试。"""

    def test_get_chunk_data_with_middle_offset(self, tmp_path: Path):
        """从文件中间偏移位置读取分片数据。"""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        file_path = models_dir / "model1.gguf"
        data = b"0123456789ABCDEF"
        file_path.write_bytes(data)

        protocol = ModelSyncProtocol(models_dir=models_dir)
        chunk = ChunkInfo(chunk_index=0, offset=5, size=5, sha256="")
        result = protocol.get_chunk_data("model1", chunk)
        assert result == b"56789"

    def test_init_receive_creates_file_with_correct_size(self, tmp_path: Path):
        """init_receive 应创建正确大小的预分配文件。"""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        protocol = ModelSyncProtocol(models_dir=models_dir)

        chunks = [
            ChunkInfo(chunk_index=0, offset=0, size=50, sha256=""),
            ChunkInfo(chunk_index=1, offset=50, size=50, sha256=""),
        ]
        protocol.init_receive(
            model_id="m1",
            total_bytes=100,
            chunks=chunks,
            file_sha256="xyz",
        )
        # 文件应已预分配
        file_path = models_dir / "m1.gguf"
        assert file_path.exists()
        assert file_path.stat().st_size == 100
