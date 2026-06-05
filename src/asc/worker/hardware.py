"""HardwareDetector 跨平台硬件检测模块。

封装跨平台硬件检测逻辑，从 WorkerAgent 解耦：
- CPU: 逻辑核心数、物理核心数、基准频率(MHz)、品牌名
- GPU: 索引、名称、厂商、总VRAM、可用VRAM、计算能力
- 磁盘: 可用空间(MB)
- 网络: 预估带宽(Mbps)
"""

from __future__ import annotations

import contextlib
import json
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

    def detect_disk(self) -> DiskInfo:
        """检测磁盘可用空间。"""
        try:
            usage = psutil.disk_usage(str(Path.home()))
            free_mb = int(usage.free // (1024 * 1024))
        except Exception:
            free_mb = 0
        return DiskInfo(free_mb=free_mb)

    def detect_network(self) -> NetworkInfo:
        """预估网络带宽（简化实现）。"""
        # 预留扩展接口：后续可通过 iperf3 或实际测速获取
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
        """macOS GPU 检测（Metal）。"""
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
                    vram_total = int(mem.total // (1024 * 1024))
                    vram_free = int(mem.available // (1024 * 1024))
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
        with contextlib.suppress(Exception):
            subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram", "--csv"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            # 简化解析，预留完整实现
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
                ["wmic", "cpu", "get", "Name", "/value"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if line.startswith("Name="):
                        return line.split("=", 1)[1].strip()
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
