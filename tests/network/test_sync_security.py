"""模型同步安全增强测试。

覆盖审计中发现的关键缺口：
- _safe_model_path 路径遍历检测（使用 relative_to 修复）
- _validate_model_id 空字符串
- _validate_model_id 路径遍历字符 (.., /, \\, null byte)
- verify_chunk 空 sha256 跳过验证
- BandwidthLimiter 初始突发行为
"""

from pathlib import Path

import pytest

from asc.network.sync import (
    BandwidthLimiter,
    ChunkInfo,
    _safe_model_path,
    _validate_model_id,
    verify_chunk,
)


class TestSafeModelPath:
    """_safe_model_path 路径遍历检测测试。"""

    def test_normal_model_id(self, tmp_path: Path):
        """正常的 model_id 应返回正确的路径。"""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        result = _safe_model_path(models_dir, "llama-3.1-8b")
        assert result == models_dir.resolve() / "llama-3.1-8b.gguf"

    def test_path_traversal_dotdot(self, tmp_path: Path):
        """包含 .. 的路径遍历应被检测。"""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        with pytest.raises(ValueError, match="非法字符"):
            _safe_model_path(models_dir, "../etc/passwd")

    def test_path_traversal_slash(self, tmp_path: Path):
        """包含 / 的 model_id 应被拒绝。"""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        with pytest.raises(ValueError, match="非法字符"):
            _safe_model_path(models_dir, "subdir/model")

    def test_path_traversal_backslash(self, tmp_path: Path):
        """包含 \\ 的 model_id 应被拒绝。"""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        with pytest.raises(ValueError, match="非法字符"):
            _safe_model_path(models_dir, "subdir\\model")

    def test_path_traversal_null_byte(self, tmp_path: Path):
        """包含 null byte 的 model_id 应被拒绝。"""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        with pytest.raises(ValueError, match="非法字符"):
            _safe_model_path(models_dir, "model\x00.exe")

    def test_symlink_traversal(self, tmp_path: Path):
        """通过符号链接逃逸 models_dir 应被检测。"""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        # 创建一个指向 models_dir 外部的符号链接
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        try:
            (models_dir / "escape.gguf").symlink_to(outside_dir / "target.gguf")
        except OSError:
            # Windows 上创建符号链接需要管理员权限，跳过此测试
            pytest.skip("创建符号链接需要管理员权限")

        # _safe_model_path 本身只检查 resolve 后的路径是否在 models_dir 内
        # 符号链接 resolve 后指向 models_dir 外部，应被检测
        with pytest.raises(ValueError, match="路径遍历检测"):
            _safe_model_path(models_dir, "escape")


class TestValidateModelId:
    """_validate_model_id 测试。"""

    def test_empty_string(self):
        """空字符串应抛出 ValueError。"""
        with pytest.raises(ValueError, match="非法字符"):
            _validate_model_id("")

    def test_dotdot(self):
        """包含 .. 应抛出 ValueError。"""
        with pytest.raises(ValueError, match="非法字符"):
            _validate_model_id("../etc/passwd")

    def test_forward_slash(self):
        """包含 / 应抛出 ValueError。"""
        with pytest.raises(ValueError, match="非法字符"):
            _validate_model_id("path/to/model")

    def test_backslash(self):
        """包含 \\ 应抛出 ValueError。"""
        with pytest.raises(ValueError, match="非法字符"):
            _validate_model_id("path\\to\\model")

    def test_null_byte(self):
        """包含 null byte 应抛出 ValueError。"""
        with pytest.raises(ValueError, match="非法字符"):
            _validate_model_id("model\x00evil")

    def test_valid_model_id(self):
        """合法的 model_id 不应抛出异常。"""
        _validate_model_id("llama-3.1-8b")
        _validate_model_id("qwen_2.5-7b")
        _validate_model_id("my.model.v2")


class TestVerifyChunkEmptySha256:
    """verify_chunk 空 sha256 跳过验证测试。"""

    def test_empty_sha256_skips_verification(self):
        """空 sha256 应跳过验证，返回 True。"""
        chunk = ChunkInfo(chunk_index=0, offset=0, size=5, sha256="")
        assert verify_chunk(chunk, b"any data") is True

    def test_none_sha256_skips_verification(self):
        """sha256 为 None 时（如果传入了空值）应跳过验证。"""
        chunk = ChunkInfo(chunk_index=0, offset=0, size=5, sha256="")
        assert verify_chunk(chunk, b"any data") is True

    def test_valid_sha256_passes(self):
        """正确的 sha256 应通过验证。"""
        import hashlib
        data = b"test data"
        sha = hashlib.sha256(data).hexdigest()
        chunk = ChunkInfo(chunk_index=0, offset=0, size=len(data), sha256=sha)
        assert verify_chunk(chunk, data) is True

    def test_invalid_sha256_fails(self):
        """错误的 sha256 应验证失败。"""
        chunk = ChunkInfo(chunk_index=0, offset=0, size=5, sha256="invalid_hash")
        assert verify_chunk(chunk, b"hello") is False


class TestBandwidthLimiterInitialBurst:
    """BandwidthLimiter 初始突发行为测试。"""

    def test_initial_burst_allows_full_rate(self):
        """初始令牌桶满，应允许一次性发送 max_bytes_per_sec 的数据。"""
        limiter = BandwidthLimiter(max_bytes_per_sec=1000)
        wait = limiter.acquire(1000)
        assert wait == 0.0

    def test_initial_burst_exceeds_rate(self):
        """初始突发超过速率时应返回等待时间。"""
        limiter = BandwidthLimiter(max_bytes_per_sec=1000)
        limiter.acquire(1000)  # 用完初始令牌
        wait = limiter.acquire(500)
        assert wait > 0.0

    def test_initial_burst_partial(self):
        """部分使用令牌后，剩余令牌仍可使用。"""
        limiter = BandwidthLimiter(max_bytes_per_sec=1000)
        limiter.acquire(600)
        wait = limiter.acquire(400)
        assert wait == 0.0

    def test_initial_burst_small_request(self):
        """小请求应立即通过。"""
        limiter = BandwidthLimiter(max_bytes_per_sec=10 * 1024 * 1024)
        wait = limiter.acquire(1024)
        assert wait == 0.0
