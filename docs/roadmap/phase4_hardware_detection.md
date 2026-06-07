# Phase 4: 完整硬件检测与资源监控

## 目标

实现跨平台的完整硬件检测，包括 CPU、GPU、内存、磁盘等，为差异化任务分配提供数据基础。

## 现状

当前 `NodeResources` 仅检测 VRAM：
```python
@dataclass
class NodeResources:
    total_vram_free_mb: int = 0
```

## 需求

### 4.1 硬件信息检测

```python
@dataclass
class HardwareInfo:
    # CPU
    cpu_count: int           # 逻辑核心数
    cpu_physical_count: int  # 物理核心数
    cpu_freq_mhz: float      # 频率
    cpu_brand: str           # 品牌
    
    # GPU
    gpus: list[GPUInfo]      # 多 GPU 支持
    
    # 内存
    total_ram_mb: int
    free_ram_mb: int
    
    # 磁盘
    total_disk_mb: int
    free_disk_mb: int
    
    # 网络
    network_speed_mbps: float  # 预估带宽

@dataclass
class GPUInfo:
    index: int
    name: str                # "NVIDIA RTX 4090"
    vendor: str              # "nvidia" | "amd" | "apple"
    vram_total_mb: int
    vram_free_mb: int
    compute_capability: str  # CUDA 版本或 Metal 版本
```

### 4.2 跨平台实现

| 平台 | CPU | GPU | 内存 | 磁盘 |
|------|-----|-----|------|------|
| Linux | `psutil` | `pynvml` / `rocm-smi` | `psutil` | `psutil` |
| Windows | `psutil` | `pynvml` / `wmi` | `psutil` | `psutil` |
| macOS | `psutil` | `torch.mps` / `system_profiler` | `psutil` | `psutil` |

### 4.3 实时资源监控

- 每秒刷新一次资源状态
- 资源变化超过阈值时上报 Master
- 支持资源预警（VRAM < 10% 时告警）

### 4.4 任务分配权重

```python
def calculate_compute_score(hw: HardwareInfo) -> float:
    """计算节点算力评分。"""
    cpu_score = hw.cpu_count * hw.cpu_freq_mhz / 1000
    gpu_score = sum(g.vram_total_mb / 1024 for g in hw.gpus) * 10
    return cpu_score + gpu_score
```

## 实现计划

1. **Day 1-2**: 集成 `psutil`，实现 CPU/内存/磁盘检测
2. **Day 3**: 集成 `pynvml`（NVIDIA）和 `torch.mps`（Apple）
3. **Day 4**: 实现资源监控循环和变化上报
4. **Day 5**: 更新调度器，使用完整硬件信息做任务分配

## 文件变更

- `src/asc/worker/hardware.py` - 新增硬件检测模块
- `src/asc/worker/agent.py` - 集成硬件检测
- `src/asc/scheduler/placement.py` - 使用硬件评分
- `tests/worker/test_hardware.py` - 新增测试

## 依赖

```toml
[project.optional-dependencies]
hardware = [
    "psutil>=5.9.0",
    "pynvml>=11.5.0",      # NVIDIA
    "pyamdgpuinfo>=2.1.0",  # AMD (Linux)
]
```
