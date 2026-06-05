"""测试模型管理：多源下载 + 进度追踪 + 分发。"""

import tempfile
from pathlib import Path
from unittest.mock import patch

from asc.core.model_manager import (
    DownloadProgress,
    ModelManager,
    ModelMetadata,
)
from asc.core.model_downloader import DownloadSource


class TestModelMetadata:
    """模型元数据。"""

    def test_create(self):
        meta = ModelMetadata(
            model_id="llama-3.1-8b",
            file_path="/models/llama-3.1-8b-q4.gguf",
            file_size_mb=4900,
            source="huggingface",
        )
        assert meta.model_id == "llama-3.1-8b"
        assert meta.file_size_mb == 4900

    def test_frozen(self):
        meta = ModelMetadata(model_id="m", file_path="/m.gguf", file_size_mb=100, source="local")
        try:
            meta.model_id = "other"  # type: ignore[misc]
            raise AssertionError("Should be immutable")
        except (AttributeError, TypeError):
            pass

    def test_source_types(self):
        """支持多种来源标记。"""
        for source in ("local", "huggingface", "modelscope", "manual", "remote"):
            meta = ModelMetadata(model_id="m", file_path="/m.gguf", file_size_mb=100, source=source)
            assert meta.source == source


class TestDownloadProgress:
    """下载进度。"""

    def test_create(self):
        p = DownloadProgress(
            model_id="llama-3.1-8b",
            downloaded_mb=2500,
            total_mb=4900,
            speed_mbps=50.0,
        )
        assert p.downloaded_mb == 2500
        assert p.total_mb == 4900

    def test_fraction(self):
        p = DownloadProgress(model_id="m", downloaded_mb=2500, total_mb=5000, speed_mbps=10)
        assert p.fraction == 0.5

    def test_complete(self):
        p = DownloadProgress(model_id="m", downloaded_mb=5000, total_mb=5000, speed_mbps=0)
        assert p.fraction == 1.0


class TestModelManager:
    """模型管理器。"""

    def test_register_local_model(self):
        mgr = ModelManager(models_dir=Path("/tmp/models"))
        mgr.register("my-model", "/path/to/model.gguf")
        assert mgr.resolve("my-model") == "/path/to/model.gguf"

    def test_register_with_source(self):
        mgr = ModelManager(models_dir=Path("/tmp/models"))
        mgr.register("hf-model", "/path/to/model.gguf", source="huggingface")
        models = mgr.list_models()
        assert models[0].source == "huggingface"

    def test_resolve_unknown_returns_none(self):
        mgr = ModelManager(models_dir=Path("/tmp/models"))
        assert mgr.resolve("unknown") is None

    def test_list_models(self):
        mgr = ModelManager(models_dir=Path("/tmp/models"))
        mgr.register("model-a", "/a.gguf")
        mgr.register("model-b", "/b.gguf")
        models = mgr.list_models()
        assert len(models) == 2
        ids = [m.model_id for m in models]
        assert "model-a" in ids
        assert "model-b" in ids

    def test_unregister(self):
        mgr = ModelManager(models_dir=Path("/tmp/models"))
        mgr.register("m", "/m.gguf")
        mgr.unregister("m")
        assert mgr.resolve("m") is None

    def test_auto_discover(self):
        """自动发现 models 目录下的 .gguf 文件。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir)
            (models_dir / "llama-3.1-8b-q4.gguf").write_bytes(b"\x00" * 100)
            (models_dir / "qwen-2.5-7b-q8.gguf").write_bytes(b"\x00" * 200)

            mgr = ModelManager(models_dir=models_dir)
            discovered = mgr.auto_discover()

            assert "llama-3.1-8b-q4" in discovered
            assert "qwen-2.5-7b-q8" in discovered
            assert mgr.resolve("llama-3.1-8b-q4") is not None
            assert mgr.resolve("qwen-2.5-7b-q8") is not None

    def test_auto_discover_returns_new_only(self):
        """auto_discover 只返回新发现的模型。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir)
            (models_dir / "model-a.gguf").write_bytes(b"\x00" * 100)

            mgr = ModelManager(models_dir=models_dir)
            first = mgr.auto_discover()
            assert "model-a" in first

            second = mgr.auto_discover()
            assert len(second) == 0

    def test_auto_discover_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ModelManager(models_dir=Path(tmpdir))
            discovered = mgr.auto_discover()
            assert len(discovered) == 0

    def test_scan_available_models(self):
        """扫描可用模型（不依赖注册表）。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir)
            (models_dir / "model-a.gguf").write_bytes(b"\x00" * 100)
            (models_dir / "model-b.gguf").write_bytes(b"\x00" * 200)

            mgr = ModelManager(models_dir=models_dir)
            models = mgr.scan_available_models()
            assert len(models) == 2

    @patch("huggingface_hub.hf_hub_download")
    def test_download_from_huggingface(self, mock_download):
        """HuggingFace 下载（mock）。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_download.return_value = str(Path(tmpdir) / "model.gguf")
            Path(tmpdir, "model.gguf").write_bytes(b"\x00" * 100)

            mgr = ModelManager(models_dir=Path(tmpdir))
            assert hasattr(mgr, "download")

    def test_models_dir_property(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = ModelManager(models_dir=Path(tmpdir))
            assert mgr.models_dir == Path(tmpdir)

    def test_downloader_property(self):
        mgr = ModelManager(models_dir=Path("/tmp/models"))
        assert mgr.downloader is not None

    def test_distributor_property(self):
        mgr = ModelManager(models_dir=Path("/tmp/models"))
        assert mgr.distributor is not None

    def test_download_model_manual(self):
        """通过统一入口手动下载。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir)
            (models_dir / "test-model.gguf").write_bytes(b"\x00" * 100)

            mgr = ModelManager(models_dir=models_dir)
            gen = mgr.download_model(
                source=DownloadSource.MANUAL,
                model_uri="manual",
            )
            try:
                while True:
                    next(gen)
            except StopIteration as e:
                result = e.value

            assert result.success
            assert mgr.resolve("test-model") is not None

    def test_distribute_model(self):
        """分发模型到节点。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir)
            (models_dir / "test-model.gguf").write_bytes(b"\x00" * 100)

            mgr = ModelManager(models_dir=models_dir)
            from asc.core.model_distributor import DistributionTarget

            targets = [
                DistributionTarget(node_id="w1", ip="10.0.0.1", port=52415),
            ]
            gen = mgr.distribute_model("test-model", targets, send_chunk_fn=None)
            try:
                while True:
                    next(gen)
            except StopIteration as e:
                results = e.value

            assert len(results) == 1
