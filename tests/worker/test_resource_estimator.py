"""资源估算器测试。"""
import os
import tempfile

from asc.worker.resource_estimator import ResourceEstimate, estimate_model_resources


def _create_temp_gguf(size_mb: int = 100) -> tuple[str, str]:
    """创建临时 GGUF 文件，返回 (文件路径, 临时目录路径)。

    Windows 上 NamedTemporaryFile 在 with 块内无法删除，
    因此先创建临时目录再在目录中创建文件，确保文件句柄关闭后可删除。
    """
    tmpdir = tempfile.mkdtemp()
    fpath = os.path.join(tmpdir, "test.gguf")
    with open(fpath, "wb") as f:
        f.write(b"\0" * (size_mb * 1024 * 1024))
    return fpath, tmpdir


def _cleanup(tmpdir: str) -> None:
    """清理临时目录。"""
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


def test_estimate_with_file_size():
    """有文件大小时置信度为 high。"""
    fpath, tmpdir = _create_temp_gguf(100)
    try:
        result = estimate_model_resources(fpath)
        assert result.confidence == "high"
        assert result.model_bytes > 0
        assert result.total_bytes > 0
    finally:
        _cleanup(tmpdir)


def test_estimate_gpu_offload_max():
    """gpu_offload_ratio="max" 时全部使用 VRAM。"""
    fpath, tmpdir = _create_temp_gguf(100)
    try:
        result = estimate_model_resources(fpath, {"gpu_offload_ratio": "max"})
        assert result.model_vram_bytes > 0
        assert result.model_ram_bytes == 0
    finally:
        _cleanup(tmpdir)


def test_estimate_gpu_offload_off():
    """gpu_offload_ratio="off" 时全部使用 RAM。"""
    fpath, tmpdir = _create_temp_gguf(100)
    try:
        result = estimate_model_resources(fpath, {"gpu_offload_ratio": "off"})
        assert result.model_vram_bytes == 0
        assert result.model_ram_bytes > 0
    finally:
        _cleanup(tmpdir)


def test_guardrails_vram_exceeded():
    """VRAM 不足时 passes_guardrails=False。"""
    fpath, tmpdir = _create_temp_gguf(100)
    try:
        result = estimate_model_resources(
            fpath,
            {"gpu_offload_ratio": "max"},
            available_vram_mb=10,  # 只有 10MB VRAM
        )
        assert not result.passes_guardrails
        assert any("VRAM" in w for w in result.warnings)
    finally:
        _cleanup(tmpdir)


def test_guardrails_ram_exceeded():
    """RAM 不足时 passes_guardrails=False。"""
    fpath, tmpdir = _create_temp_gguf(100)
    try:
        result = estimate_model_resources(
            fpath,
            {"gpu_offload_ratio": "off"},
            available_ram_mb=10,  # 只有 10MB RAM
        )
        assert not result.passes_guardrails
        assert any("RAM" in w for w in result.warnings)
    finally:
        _cleanup(tmpdir)


def test_estimate_no_file():
    """文件不存在时置信度为 low。"""
    result = estimate_model_resources("/nonexistent/model.gguf")
    assert result.confidence == "low"
    assert result.model_bytes == 0


def test_resource_estimate_dataclass():
    """ResourceEstimate 默认值正确。"""
    est = ResourceEstimate()
    assert est.confidence == "low"
    assert est.passes_guardrails is True
    assert est.warnings == []
