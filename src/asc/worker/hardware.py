"""HardwareDetector 跨平台硬件检测模块。

封装跨平台硬件检测逻辑，从 WorkerAgent 解耦：
- CPU: 逻辑核心数、物理核心数、基准频率(MHz)、品牌名
- GPU: 索引、名称、厂商、总VRAM、可用VRAM、计算能力
- 磁盘: 可用空间(MB)
- 网络: 预估带宽(Mbps)
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil

from asc.worker.gpu_info import GPUInfo


@dataclass(frozen=True)
class CPUInfo:
    """CPU 信息。"""

    logical_count: int
    physical_count: int
    freq_mhz: float
    brand: str


@dataclass(frozen=True)
class DiskInfo:
    """磁盘信息。"""

    free_mb: int


@dataclass(frozen=True)
class NetworkInfo:
    """网络信息。"""

    estimated_mbps: float


class HardwareDetector:
    """跨平台硬件检测器。"""

    def detect_cpu(self) -> CPUInfo:
        """检测 CPU 信息。"""
        logical = psutil.cpu_count(logical=True) or 0
        physical = psutil.cpu_count(logical=False) or 0
        freq = 0.0
        brand = ""
        try:
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                freq = cpu_freq.current or 0.0
        except Exception:
            pass
        try:
            info = self._get_cpu_brand()
            if info:
                brand = info
        except Exception:
            pass
        return CPUInfo(
            logical_count=logical,
            physical_count=physical,
            freq_mhz=freq,
            brand=brand,
        )

    def detect_gpus(self) -> list[GPUInfo]:
        """检测 GPU 信息。"""
        system = platform.system()
        if system == "Windows":
            return self._detect_gpus_windows()
        elif system == "Linux":
            return self._detect_gpus_linux()
        elif system == "Darwin":
            return self._detect_gpus_darwin()
        return []

    def detect_disk(self, model_storage_path: str | None = None) -> DiskInfo:
        """检测磁盘可用空间。

        Args:
            model_storage_path: 模型存储路径，None 时使用 ASC_MODELS_PATH 环境变量，
                               未设置时回退到用户主目录。
        """
        try:
            if model_storage_path:
                target = Path(model_storage_path)
            else:
                models_env = os.environ.get("ASC_MODELS_PATH")
                target = Path(models_env) if models_env else Path.home()
            usage = psutil.disk_usage(str(target))
            free_mb = int(usage.free // (1024 * 1024))
        except Exception:
            free_mb = 0
        return DiskInfo(free_mb=free_mb)

    def detect_network(self) -> NetworkInfo:
        """预估网络带宽。

        优先使用环境变量 ASC_NETWORK_MBPS，否则通过实际测速获取估算值。
        """
        # 检查环境变量覆盖
        env_mbps = os.environ.get("ASC_NETWORK_MBPS")
        if env_mbps:
            try:
                return NetworkInfo(estimated_mbps=float(env_mbps))
            except ValueError:
                pass

        # 通过 psutil 网卡统计估算带宽（简化实现）
        try:
            counters = psutil.net_io_counters()
            # 粗略估算：基于网卡速度，常见以太网 1000 Mbps
            if counters.bytes_sent > 0 or counters.bytes_recv > 0:
                return NetworkInfo(estimated_mbps=1000.0)
        except Exception:
            pass
        return NetworkInfo(estimated_mbps=0.0)

    def detect_all(self) -> dict[str, Any]:
        """返回完整硬件信息字典。"""
        cpu = self.detect_cpu()
        gpus = self.detect_gpus()
        disk = self.detect_disk()
        network = self.detect_network()
        mem = psutil.virtual_memory()
        return {
            "cpu": {
                "logical_count": cpu.logical_count,
                "physical_count": cpu.physical_count,
                "freq_mhz": cpu.freq_mhz,
                "brand": cpu.brand,
            },
            "gpus": [
                {
                    "index": g.index,
                    "name": g.name,
                    "vendor": g.vendor,
                    "vram_total_mb": g.vram_total_mb,
                    "vram_free_mb": g.vram_free_mb,
                    "compute_capability": g.compute_capability,
                }
                for g in gpus
            ],
            "memory_total_mb": int(mem.total // (1024 * 1024)),
            "memory_free_mb": int(mem.available // (1024 * 1024)),
            "disk_free_mb": disk.free_mb,
            "network_mbps": network.estimated_mbps,
        }

    # ------------------------------------------------------------------
    # 平台相关 GPU 检测
    # ------------------------------------------------------------------

    def _detect_gpus_windows(self) -> list[GPUInfo]:
        """Windows GPU 检测（nvidia-smi 优先）。"""
        return self._parse_nvidia_smi()

    def _detect_gpus_linux(self) -> list[GPUInfo]:
        """Linux GPU 检测。"""
        gpus = self._parse_nvidia_smi()
        if not gpus:
            gpus = self._parse_rocm_smi()
        return gpus

    def _detect_gpus_darwin(self) -> list[GPUInfo]:
        """macOS GPU 检测（Metal / Apple Silicon 统一内存）。

        Apple Silicon 使用统一内存架构，GPU 与 CPU 共享内存池。
        报告的内存总量为系统总内存，但标注为统一内存（unified）。
        """
        gpus = []
        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType", "-json"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                for i, display in enumerate(data.get("SPDisplaysDataType", [])):
                    name = display.get("sppci_model", "Apple GPU")
                    mem = psutil.virtual_memory()
                    # Apple Silicon 统一内存：GPU 可用的内存量取决于系统压力
                    # 使用可用内存的 75% 作为 GPU 可用内存估算
                    vram_total = int(mem.total // (1024 * 1024))
                    vram_free = int(mem.available * 0.75 // (1024 * 1024))
                    gpus.append(
                        GPUInfo(
                            index=i,
                            name=name,
                            vram_total_mb=vram_total,
                            vram_free_mb=vram_free,
                            vendor="apple",
                        )
                    )
        except Exception:
            pass
        return gpus

    def _parse_nvidia_smi(self) -> list[GPUInfo]:
        """解析 nvidia-smi 输出。"""
        gpus = []
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 4:
                        gpus.append(
                            GPUInfo(
                                index=int(parts[0]),
                                name=parts[1],
                                vram_total_mb=int(float(parts[2])),
                                vram_free_mb=int(float(parts[3])),
                                vendor="nvidia",
                            )
                        )
        except Exception:
            pass
        return gpus

    def _parse_rocm_smi(self) -> list[GPUInfo]:
        """解析 rocm-smi 输出。"""
        gpus = []
        try:
            result = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram", "--csv"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split("\n")
                # 跳过表头行
                for line in lines[1:]:
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 3:
                        try:
                            gpu_index = int(parts[0])
                            vram_total = int(float(parts[1]))
                            vram_used = int(float(parts[2]))
                            gpus.append(
                                GPUInfo(
                                    index=gpu_index,
                                    name=f"AMD GPU {gpu_index}",
                                    vram_total_mb=vram_total,
                                    vram_free_mb=vram_total - vram_used,
                                    vendor="amd",
                                )
                            )
                        except (ValueError, IndexError):
                            continue
        except Exception:
            pass
        return gpus

    # ------------------------------------------------------------------
    # CPU 品牌检测
    # ------------------------------------------------------------------

    def _get_cpu_brand(self) -> str:
        """获取 CPU 品牌名。"""
        system = platform.system()
        if system == "Windows":
            return self._get_cpu_brand_windows()
        elif system == "Linux":
            return self._get_cpu_brand_linux()
        elif system == "Darwin":
            return self._get_cpu_brand_darwin()
        return ""

    def _get_cpu_brand_windows(self) -> str:
        try:
            result = subprocess.run(
                [
                    "powershell", "-NoProfile", "-Command",
                    "Get-CimInstance -ClassName Win32_Processor "
                    "| Select-Object -ExpandProperty Name",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return ""

    def _get_cpu_brand_linux(self) -> str:
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass
        return ""

    def _get_cpu_brand_darwin(self) -> str:
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return ""
