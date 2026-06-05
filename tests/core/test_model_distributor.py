"""测试 ModelDistributor 模型分发器。"""

import tempfile
from pathlib import Path

from asc.core.model_distributor import (
    DistributionResult,
    DistributionTarget,
    ModelDistributor,
)
from asc.worker.agent import NodeResources


class TestDistributionTarget:
    """分发目标。"""

    def test_create(self):
        target = DistributionTarget(
            node_id="worker-1",
            ip="192.168.1.10",
            port=52415,
        )
        assert target.node_id == "worker-1"
        assert target.ip == "192.168.1.10"

    def test_with_resources(self):
        res = NodeResources(
            cpu_count=8,
            cpu_percent=10.0,
            memory_total_mb=16000,
            memory_free_mb=8000,
        )
        target = DistributionTarget(
            node_id="worker-1",
            ip="192.168.1.10",
            port=52415,
            resources=res,
        )
        assert target.resources is not None
        assert target.resources.cpu_count == 8


class TestDistributionResult:
    """分发结果。"""

    def test_success(self):
        result = DistributionResult(
            model_id="llama-3.1-8b",
            target_node_id="worker-1",
            success=True,
            file_path="/models/llama-3.1-8b.gguf",
            file_size_mb=4900,
        )
        assert result.success

    def test_failure(self):
        result = DistributionResult(
            model_id="llama-3.1-8b",
            target_node_id="worker-1",
            success=False,
            file_path="",
            file_size_mb=0,
            error="传输失败",
        )
        assert not result.success
        assert result.error == "传输失败"


class TestModelDistributor:
    """模型分发器。"""

    def test_init(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dist = ModelDistributor(Path(tmpdir))
            assert dist._models_dir == Path(tmpdir)

    def test_list_local_models_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dist = ModelDistributor(Path(tmpdir))
            assert dist.list_local_models() == []

    def test_list_local_models(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir)
            (models_dir / "llama-3.1-8b.gguf").write_bytes(b"\x00" * (3 * 1024 * 1024))
            (models_dir / "qwen-2.5-7b.gguf").write_bytes(b"\x00" * (5 * 1024 * 1024))

            dist = ModelDistributor(models_dir)
            models = dist.list_local_models()
            assert len(models) == 2
            ids = [m["model_id"] for m in models]
            assert "llama-3.1-8b" in ids
            assert "qwen-2.5-7b" in ids

    def test_distribute_model_not_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dist = ModelDistributor(Path(tmpdir))
            targets = [
                DistributionTarget(node_id="w1", ip="10.0.0.1", port=52415),
            ]
            gen = dist.distribute("nonexistent", targets)
            try:
                while True:
                    next(gen)
            except StopIteration as e:
                results = e.value

            assert len(results) == 1
            assert not results[0].success
            assert "未找到模型文件" in results[0].error

    def test_distribute_no_send_callback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir)
            (models_dir / "test-model.gguf").write_bytes(b"\x00" * 100)

            dist = ModelDistributor(models_dir)
            targets = [
                DistributionTarget(node_id="w1", ip="10.0.0.1", port=52415),
            ]
            gen = dist.distribute("test-model", targets, send_chunk_fn=None)
            try:
                while True:
                    next(gen)
            except StopIteration as e:
                results = e.value

            assert len(results) == 1
            assert not results[0].success
            assert "未配置发送回调" in results[0].error

    def test_distribute_with_callback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir)
            # 创建一个大于 10MB 的文件以确保有分片
            (models_dir / "big-model.gguf").write_bytes(b"\x00" * (11 * 1024 * 1024))

            dist = ModelDistributor(models_dir)
            targets = [
                DistributionTarget(node_id="w1", ip="10.0.0.1", port=52415),
                DistributionTarget(node_id="w2", ip="10.0.0.2", port=52415),
            ]

            # Mock 发送回调
            def send_chunk(node_id, chunk_info, data):
                return True

            gen = dist.distribute("big-model", targets, send_chunk_fn=send_chunk)
            progress = []
            try:
                while True:
                    progress.append(next(gen))
            except StopIteration as e:
                results = e.value

            assert len(results) == 2
            assert all(r.success for r in results)
            assert results[0].target_node_id == "w1"
            assert results[1].target_node_id == "w2"

    def test_find_model_file_exact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir)
            (models_dir / "my-model.gguf").write_bytes(b"\x00" * 100)

            dist = ModelDistributor(models_dir)
            found = dist._find_model_file("my-model")
            assert found is not None
            assert found.name == "my-model.gguf"

    def test_find_model_file_not_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dist = ModelDistributor(Path(tmpdir))
            assert dist._find_model_file("nonexistent") is None

    def test_find_model_file_case_insensitive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir)
            (models_dir / "My-Model.gguf").write_bytes(b"\x00" * 100)

            dist = ModelDistributor(models_dir)
            found = dist._find_model_file("my-model")
            assert found is not None
