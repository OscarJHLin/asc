"""测试 ModelDownloader 多源下载器。"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from asc.core.model_downloader import (
    DownloadRequest,
    DownloadResult,
    DownloadSource,
    ModelDownloader,
)


class TestDownloadSource:
    """下载来源枚举。"""

    def test_values(self):
        assert DownloadSource.HUGGINGFACE.value == "huggingface"
        assert DownloadSource.MODELSCOPE.value == "modelscope"
        assert DownloadSource.MANUAL.value == "manual"


class TestDownloadRequest:
    """下载请求。"""

    def test_create_huggingface(self):
        req = DownloadRequest(
            source=DownloadSource.HUGGINGFACE,
            model_uri="TheBloke/Llama-3.1-8B-Q4_K_M",
        )
        assert req.source == DownloadSource.HUGGINGFACE
        assert req.model_uri == "TheBloke/Llama-3.1-8B-Q4_K_M"
        assert req.filename is None
        assert req.target_dir is None

    def test_create_modelscope(self):
        req = DownloadRequest(
            source=DownloadSource.MODELSCOPE,
            model_uri="unsloth/Qwen3.6-27B-GGUF",
        )
        assert req.source == DownloadSource.MODELSCOPE

    def test_create_manual(self):
        req = DownloadRequest(
            source=DownloadSource.MANUAL,
            model_uri="manual",
            target_dir=Path("/custom/models"),
        )
        assert req.target_dir == Path("/custom/models")


class TestDownloadResult:
    """下载结果。"""

    def test_success(self):
        result = DownloadResult(
            success=True,
            source=DownloadSource.HUGGINGFACE,
            model_id="llama-3.1-8b",
            file_path="/models/llama-3.1-8b.gguf",
            file_size_mb=4900,
        )
        assert result.success
        assert result.model_id == "llama-3.1-8b"

    def test_failure(self):
        result = DownloadResult(
            success=False,
            source=DownloadSource.MODELSCOPE,
            model_id="test",
            file_path="",
            file_size_mb=0,
            error="下载失败",
        )
        assert not result.success
        assert result.error == "下载失败"


class TestModelDownloader:
    """模型下载器。"""

    def test_init_creates_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir) / "models"
            dl = ModelDownloader(models_dir)
            assert dl.models_dir.exists()

    def test_models_dir_property(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir) / "models"
            models_dir.mkdir()
            dl = ModelDownloader(models_dir)
            assert dl.models_dir == models_dir

    def test_scan_models_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dl = ModelDownloader(Path(tmpdir))
            assert dl.scan_models() == []

    def test_scan_models_finds_gguf(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir)
            (models_dir / "llama-3.1-8b-q4.gguf").write_bytes(b"\x00" * 100)
            (models_dir / "qwen-2.5-7b-q8.gguf").write_bytes(b"\x00" * 200)
            (models_dir / "readme.txt").write_text("not a model")

            dl = ModelDownloader(models_dir)
            models = dl.scan_models()
            assert len(models) == 2
            ids = [m["model_id"] for m in models]
            assert "llama-3.1-8b-q4" in ids
            assert "qwen-2.5-7b-q8" in ids

    def test_scan_models_includes_size(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir)
            (models_dir / "model.gguf").write_bytes(b"\x00" * (2 * 1024 * 1024))

            dl = ModelDownloader(models_dir)
            models = dl.scan_models()
            assert len(models) == 1
            assert models[0]["file_size_mb"] == 2

    def test_manual_download_finds_gguf(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir)
            (models_dir / "test-model.gguf").write_bytes(b"\x00" * 100)

            dl = ModelDownloader(models_dir)
            request = DownloadRequest(
                source=DownloadSource.MANUAL,
                model_uri="manual",
            )
            results = list(dl.download(request))
            result = results[-1] if results else None
            # 最后一个 yield 是 generator 的 return value
            # 需要通过不同方式获取
            gen = dl.download(request)
            progress = []
            try:
                while True:
                    progress.append(next(gen))
            except StopIteration as e:
                result = e.value

            assert result.success
            assert result.source == DownloadSource.MANUAL
            assert result.model_id == "test-model"

    def test_manual_download_no_gguf(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dl = ModelDownloader(Path(tmpdir))
            request = DownloadRequest(
                source=DownloadSource.MANUAL,
                model_uri="manual",
            )
            gen = dl.download(request)
            try:
                while True:
                    next(gen)
            except StopIteration as e:
                result = e.value

            assert not result.success
            assert ".gguf" in result.error

    @patch("asc.core.model_downloader.ModelDownloader._is_modelscope_installed")
    @patch("asc.core.model_downloader.ModelDownloader._install_modelscope")
    @patch("subprocess.Popen")
    def test_modelscope_download_success(self, mock_popen, mock_install, mock_installed):
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir)
            (models_dir / "Qwen-7B-Q4.gguf").write_bytes(b"\x00" * 100)

            mock_installed.return_value = True
            mock_install.return_value = True

            process_mock = MagicMock()
            process_mock.stdout = iter(["Downloading...", "Done"])
            process_mock.wait.return_value = 0
            mock_popen.return_value = process_mock

            dl = ModelDownloader(models_dir)
            request = DownloadRequest(
                source=DownloadSource.MODELSCOPE,
                model_uri="unsloth/Qwen-7B-GGUF",
            )
            gen = dl.download(request)
            progress = []
            try:
                while True:
                    progress.append(next(gen))
            except StopIteration as e:
                result = e.value

            assert result.success
            assert result.source == DownloadSource.MODELSCOPE

    @patch("asc.core.model_downloader.ModelDownloader._is_modelscope_installed")
    @patch("asc.core.model_downloader.ModelDownloader._install_modelscope")
    def test_modelscope_auto_install(self, mock_install, mock_installed):
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_installed.return_value = False
            mock_install.return_value = True

            # 还需要 mock Popen
            with patch("subprocess.Popen") as mock_popen:
                (Path(tmpdir) / "model.gguf").write_bytes(b"\x00" * 100)
                process_mock = MagicMock()
                process_mock.stdout = iter([])
                process_mock.wait.return_value = 0
                mock_popen.return_value = process_mock

                dl = ModelDownloader(Path(tmpdir))
                request = DownloadRequest(
                    source=DownloadSource.MODELSCOPE,
                    model_uri="test/model",
                )
                gen = dl.download(request)
                try:
                    while True:
                        next(gen)
                except StopIteration as e:
                    result = e.value

                mock_install.assert_called_once()

    @patch("asc.core.model_downloader.ModelDownloader._is_modelscope_installed")
    @patch("asc.core.model_downloader.ModelDownloader._install_modelscope")
    def test_modelscope_install_failure(self, mock_install, mock_installed):
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_installed.return_value = False
            mock_install.return_value = False

            dl = ModelDownloader(Path(tmpdir))
            request = DownloadRequest(
                source=DownloadSource.MODELSCOPE,
                model_uri="test/model",
            )
            gen = dl.download(request)
            try:
                while True:
                    next(gen)
            except StopIteration as e:
                result = e.value

            assert not result.success
            assert "安装 modelscope 失败" in result.error


class TestParseModelscopeUri:
    """ModelScope URI 解析。"""

    def test_direct_model_id(self):
        assert ModelDownloader._parse_modelscope_uri("unsloth/Qwen3.6-27B-GGUF") == "unsloth/Qwen3.6-27B-GGUF"

    def test_community_link(self):
        uri = "https://modelscope.cn/models/unsloth/Qwen3.6-27B-GGUF"
        assert ModelDownloader._parse_modelscope_uri(uri) == "unsloth/Qwen3.6-27B-GGUF"

    def test_files_link(self):
        uri = "https://modelscope.cn/models/unsloth/Qwen3.6-27B-GGUF/files"
        assert ModelDownloader._parse_modelscope_uri(uri) == "unsloth/Qwen3.6-27B-GGUF"

    def test_download_link(self):
        uri = "https://modelscope.cn/models/unsloth/Qwen3.6-27B-GGUF/download"
        assert ModelDownloader._parse_modelscope_uri(uri) == "unsloth/Qwen3.6-27B-GGUF"


class TestHuggingFaceDownload:
    """HuggingFace 下载测试。"""

    @patch("huggingface_hub.hf_hub_download")
    @patch("huggingface_hub.list_repo_files")
    def test_hf_download_success(self, mock_list_files, mock_download):
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir)
            model_file = models_dir / "llama-3.1-8b-q4.gguf"
            model_file.write_bytes(b"\x00" * 100)

            mock_list_files.return_value = ["llama-3.1-8b-q4.gguf"]
            mock_download.return_value = str(model_file)

            dl = ModelDownloader(models_dir)
            request = DownloadRequest(
                source=DownloadSource.HUGGINGFACE,
                model_uri="TheBloke/Llama-3.1-8B-Q4_K_M",
            )
            gen = dl.download(request)
            try:
                while True:
                    next(gen)
            except StopIteration as e:
                result = e.value

            assert result.success
            assert result.source == DownloadSource.HUGGINGFACE
            assert result.model_id == "llama-3.1-8b-q4"

    def test_hf_import_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dl = ModelDownloader(Path(tmpdir))
            request = DownloadRequest(
                source=DownloadSource.HUGGINGFACE,
                model_uri="test/model",
            )
            with patch.dict("sys.modules", {"huggingface_hub": None}):
                gen = dl.download(request)
                try:
                    while True:
                        next(gen)
                except StopIteration as e:
                    result = e.value

                assert not result.success
                assert "huggingface_hub" in result.error
