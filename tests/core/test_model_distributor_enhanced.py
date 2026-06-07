"""模型分发器增强测试。

覆盖审计中发现的关键缺口：
- 异步发送回调
- 分片传输失败 + 重试成功
- 分片传输失败 + 重试也失败
- asyncio.gather return_exceptions=True（修复后行为）
- 滑动窗口不同窗口大小
- list_local_models 不存在的目录
"""

import tempfile
from pathlib import Path

import pytest

from asc.core.model_distributor import (
    DistributionTarget,
    ModelDistributor,
)
from asc.network.sync import ChunkInfo


class TestAsyncSendCallback:
    """异步发送回调测试。"""

    @pytest.mark.asyncio
    async def test_async_send_callback(self):
        """异步发送回调应被正确调用。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir)
            (models_dir / "test-model.gguf").write_bytes(b"\x00" * 100)

            dist = ModelDistributor(models_dir)
            targets = [
                DistributionTarget(node_id="w1", ip="10.0.0.1", port=52415),
            ]

            call_log = []

            async def async_send(node_id: str, chunk: ChunkInfo, data: bytes) -> bool:
                call_log.append((node_id, chunk.chunk_index))
                return True

            results = await dist.distribute("test-model", targets, send_chunk_fn=async_send)

            assert len(results) == 1
            assert results[0].success is True
            assert len(call_log) == 1

    @pytest.mark.asyncio
    async def test_sync_send_callback_wrapped(self):
        """同步发送回调应被自动包装为异步。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir)
            (models_dir / "test-model.gguf").write_bytes(b"\x00" * 100)

            dist = ModelDistributor(models_dir)
            targets = [
                DistributionTarget(node_id="w1", ip="10.0.0.1", port=52415),
            ]

            call_count = 0

            def sync_send(node_id: str, chunk: ChunkInfo, data: bytes) -> bool:
                nonlocal call_count
                call_count += 1
                return True

            results = await dist.distribute("test-model", targets, send_chunk_fn=sync_send)

            assert len(results) == 1
            assert results[0].success is True
            assert call_count == 1


class TestChunkTransmissionRetry:
    """分片传输失败与重试测试。"""

    @pytest.mark.asyncio
    async def test_retry_success_after_failure(self):
        """分片传输失败后重试成功应返回成功。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir)
            (models_dir / "test-model.gguf").write_bytes(b"\x00" * 100)

            dist = ModelDistributor(models_dir)
            targets = [
                DistributionTarget(node_id="w1", ip="10.0.0.1", port=52415),
            ]

            call_count = 0

            async def flaky_send(node_id: str, chunk: ChunkInfo, data: bytes) -> bool:
                nonlocal call_count
                call_count += 1
                # 第一次失败，后续成功
                return call_count != 1

            results = await dist.distribute(
                "test-model", targets,
                send_chunk_fn=flaky_send,
                max_retries=3,
            )

            assert len(results) == 1
            assert results[0].success is True

    @pytest.mark.asyncio
    async def test_retry_also_fails(self):
        """分片传输失败后重试也失败应返回失败。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir)
            (models_dir / "test-model.gguf").write_bytes(b"\x00" * 100)

            dist = ModelDistributor(models_dir)
            targets = [
                DistributionTarget(node_id="w1", ip="10.0.0.1", port=52415),
            ]

            async def always_fail(node_id: str, chunk: ChunkInfo, data: bytes) -> bool:
                return False

            results = await dist.distribute(
                "test-model", targets,
                send_chunk_fn=always_fail,
                max_retries=2,
            )

            assert len(results) == 1
            assert results[0].success is False
            assert "传输失败" in results[0].error

    @pytest.mark.asyncio
    async def test_exception_in_send_treated_as_failure(self):
        """发送回调抛出异常应被视为失败（return_exceptions=True 修复）。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir)
            (models_dir / "test-model.gguf").write_bytes(b"\x00" * 100)

            dist = ModelDistributor(models_dir)
            targets = [
                DistributionTarget(node_id="w1", ip="10.0.0.1", port=52415),
            ]

            async def error_send(node_id: str, chunk: ChunkInfo, data: bytes) -> bool:
                raise ConnectionError("Network down")

            results = await dist.distribute(
                "test-model", targets,
                send_chunk_fn=error_send,
                max_retries=1,
            )

            assert len(results) == 1
            assert results[0].success is False


class TestSlidingWindow:
    """滑动窗口不同窗口大小测试。"""

    @pytest.mark.asyncio
    async def test_window_size_one(self):
        """窗口大小为 1 时应串行传输。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir)
            # 创建足够大的文件以产生多个分片
            (models_dir / "test-model.gguf").write_bytes(b"\x00" * (11 * 1024 * 1024))

            dist = ModelDistributor(models_dir)
            targets = [
                DistributionTarget(node_id="w1", ip="10.0.0.1", port=52415),
            ]

            active_count = 0
            max_active = 0

            async def tracked_send(node_id: str, chunk: ChunkInfo, data: bytes) -> bool:
                nonlocal active_count, max_active
                active_count += 1
                max_active = max(max_active, active_count)
                # 模拟一些延迟
                import asyncio
                await asyncio.sleep(0.01)
                active_count -= 1
                return True

            results = await dist.distribute(
                "test-model", targets,
                send_chunk_fn=tracked_send,
                window_size=1,
            )

            assert len(results) == 1
            assert results[0].success is True
            assert max_active <= 1

    @pytest.mark.asyncio
    async def test_window_size_five(self):
        """窗口大小为 5 时应允许最多 5 个并发。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir)
            (models_dir / "test-model.gguf").write_bytes(b"\x00" * (11 * 1024 * 1024))

            dist = ModelDistributor(models_dir)
            targets = [
                DistributionTarget(node_id="w1", ip="10.0.0.1", port=52415),
            ]

            active_count = 0
            max_active = 0

            async def tracked_send(node_id: str, chunk: ChunkInfo, data: bytes) -> bool:
                nonlocal active_count, max_active
                active_count += 1
                max_active = max(max_active, active_count)
                import asyncio
                await asyncio.sleep(0.01)
                active_count -= 1
                return True

            results = await dist.distribute(
                "test-model", targets,
                send_chunk_fn=tracked_send,
                window_size=5,
            )

            assert len(results) == 1
            assert results[0].success is True
            assert max_active <= 5


class TestListLocalModels:
    """list_local_models 边界情况测试。"""

    def test_nonexistent_directory(self):
        """不存在的目录应返回空列表。"""
        dist = ModelDistributor(Path("/nonexistent/path"))
        assert dist.list_local_models() == []

    def test_empty_directory(self, tmp_path: Path):
        """空目录应返回空列表。"""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        dist = ModelDistributor(models_dir)
        assert dist.list_local_models() == []

    def test_non_gguf_files_ignored(self, tmp_path: Path):
        """非 .gguf 文件应被忽略。"""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "readme.txt").write_text("not a model")
        (models_dir / "config.json").write_text("{}")

        dist = ModelDistributor(models_dir)
        assert dist.list_local_models() == []
