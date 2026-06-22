"""模型资源估算器。

在加载模型前估算 VRAM/RAM 使用量，配合安全护栏防止 OOM。

估算方法：
- 基于 GGUF 文件大小估算模型内存使用量（模型权重 + 上下文 KV 缓存）
- 模型权重内存 ≈ 文件大小 × 1.1（考虑运行时开销）
- KV 缓存内存 ≈ context_length × 2 bytes × num_layers × 2 (K+V) / 1024 / 1024 MB（粗略估算）
- GPU 卸载比例影响 VRAM vs RAM 的分配
"""
import os
from dataclasses import dataclass
from typing import Any


@dataclass
class ResourceEstimate:
    """资源估算结果。"""
    model_vram_bytes: int = 0       # 模型 VRAM 使用量
    context_vram_bytes: int = 0     # 上下文 VRAM 使用量
    total_vram_bytes: int = 0       # 总 VRAM 使用量
    model_ram_bytes: int = 0        # 模型 RAM 使用量
    context_ram_bytes: int = 0      # 上下文 RAM 使用量
    total_ram_bytes: int = 0        # 总 RAM 使用量
    model_bytes: int = 0            # 模型总内存
    context_bytes: int = 0          # 上下文总内存
    total_bytes: int = 0            # 总内存
    confidence: str = "low"         # "high" / "low"
    passes_guardrails: bool = True  # 是否通过安全护栏
    warnings: list[str] = None      # 警告列表

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


def estimate_model_resources(
    model_path: str,
    config: dict[str, Any] | None = None,
    available_vram_mb: int = 0,
    available_ram_mb: int = 0,
) -> ResourceEstimate:
    """估算模型资源使用量。

    Args:
        model_path: GGUF 模型文件路径
        config: 模型配置（context_length, gpu_offload_ratio 等）
        available_vram_mb: 可用 VRAM (MB)
        available_ram_mb: 可用 RAM (MB)

    Returns:
        ResourceEstimate 估算结果
    """
    config = config or {}
    context_length = config.get("context_length", 4096)
    gpu_offload_ratio = config.get("gpu_offload_ratio", "max")

    # 获取模型文件大小
    model_size_bytes = 0
    try:
        model_size_bytes = os.path.getsize(model_path)
    except OSError:
        pass

    # 模型运行时开销（约 10%）
    model_runtime_bytes = int(model_size_bytes * 1.1)

    # KV 缓存估算（粗略）
    # 每个 token 的 KV 缓存 ≈ 2 bytes × hidden_dim × 2 (K+V) × num_layers
    # 简化估算：每 token 约 1MB（对 7B 模型），按文件大小比例缩放
    kv_bytes_per_token = max(1, model_size_bytes // 7000000000) * 1024 * 1024  # 粗略
    context_bytes = kv_bytes_per_token * context_length

    # GPU 卸载比例
    if gpu_offload_ratio == "max":
        gpu_ratio = 1.0
    elif gpu_offload_ratio == "off":
        gpu_ratio = 0.0
    else:
        try:
            gpu_ratio = float(gpu_offload_ratio)
            gpu_ratio = max(0.0, min(1.0, gpu_ratio))
        except (ValueError, TypeError):
            gpu_ratio = 1.0

    # 分配 VRAM/RAM
    model_vram_bytes = int(model_runtime_bytes * gpu_ratio)
    model_ram_bytes = model_runtime_bytes - model_vram_bytes
    context_vram_bytes = int(context_bytes * gpu_ratio)
    context_ram_bytes = context_bytes - context_vram_bytes

    total_vram_bytes = model_vram_bytes + context_vram_bytes
    total_ram_bytes = model_ram_bytes + context_ram_bytes
    total_bytes = model_runtime_bytes + context_bytes

    # 安全护栏检查
    warnings = []
    passes_guardrails = True

    available_vram_bytes = available_vram_mb * 1024 * 1024
    available_ram_bytes = available_ram_mb * 1024 * 1024

    if available_vram_bytes > 0 and total_vram_bytes > available_vram_bytes:
        passes_guardrails = False
        vram_over = (total_vram_bytes - available_vram_bytes) // (1024 * 1024)
        warnings.append(f"VRAM 不足: 估算需要 {total_vram_bytes // (1024*1024)} MB，可用 {available_vram_mb} MB，超出 {vram_over} MB")

    if available_ram_bytes > 0 and total_ram_bytes > available_ram_bytes:
        passes_guardrails = False
        ram_over = (total_ram_bytes - available_ram_bytes) // (1024 * 1024)
        warnings.append(f"RAM 不足: 估算需要 {total_ram_bytes // (1024*1024)} MB，可用 {available_ram_mb} MB，超出 {ram_over} MB")

    # 置信度：如果有文件大小则为 high，否则为 low
    confidence = "high" if model_size_bytes > 0 else "low"

    return ResourceEstimate(
        model_vram_bytes=model_vram_bytes,
        context_vram_bytes=context_vram_bytes,
        total_vram_bytes=total_vram_bytes,
        model_ram_bytes=model_ram_bytes,
        context_ram_bytes=context_ram_bytes,
        total_ram_bytes=total_ram_bytes,
        model_bytes=model_runtime_bytes,
        context_bytes=context_bytes,
        total_bytes=total_bytes,
        confidence=confidence,
        passes_guardrails=passes_guardrails,
        warnings=warnings,
    )
