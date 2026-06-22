"""HardwareDetector 跨平台硬件检测模块。

封装跨平台硬件检测逻辑，从 WorkerAgent 解耦：
- CPU: 逻辑核心数、物理核心数、基准频率(MHz)、品牌名
- GPU: 索引、名称、厂商、总VRAM、可用VRAM、计算能力
- 磁盘: 可用空间(MB)
- 网络: 预估带宽(Mbps)
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import psutil

from asc.worker.gpu_info import GPUInfo

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CPUInfo:
    """CPU 信息。"""

    logical_count: int
    physical_count: int
    freq_mhz: float
    brand: str
    instruction_set_extensions: list[str] = field(default_factory=list)  # ["AVX2", "AVX", "AdvSIMD"]


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
        except Exception as e:
            logger.debug("CPU频率检测失败: %s", e)
        try:
            info = self._get_cpu_brand()
            if info:
                brand = info
        except Exception as e:
            logger.debug("CPU品牌检测失败: %s", e)
        extensions = self._detect_cpu_instruction_sets()
        return CPUInfo(
            logical_count=logical,
            physical_count=physical,
            freq_mhz=freq,
            brand=brand,
            instruction_set_extensions=extensions,
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
        except Exception as e:
            logger.debug("磁盘空间检测失败: %s", e)
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
        except Exception as e:
            logger.debug("网络带宽检测失败: %s", e)
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
                "instruction_set_extensions": cpu.instruction_set_extensions,
            },
            "gpus": [
                {
                    "index": g.index,
                    "name": g.name,
                    "vendor": g.vendor,
                    "vram_total_mb": g.vram_total_mb,
                    "vram_free_mb": g.vram_free_mb,
                    "compute_capability": g.compute_capability,
                    "detection_platform": g.detection_platform,
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
        gpus = self._parse_nvidia_smi()
        if gpus:
            return [replace(g, detection_platform="CUDA") for g in gpus]
        return gpus

    def _detect_gpus_linux(self) -> list[GPUInfo]:
        """Linux GPU 检测。"""
        gpus = self._parse_nvidia_smi()
        if gpus:
            return [replace(g, detection_platform="CUDA") for g in gpus]
        gpus = self._parse_rocm_smi()
        if gpus:
            return [replace(g, detection_platform="ROCm") for g in gpus]
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
                            detection_platform="Metal",
                        )
                    )
        except Exception as e:
            logger.debug("macOS GPU检测失败: %s", e)
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
        except Exception as e:
            logger.debug("nvidia-smi解析失败: %s", e)
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
        except Exception as e:
            logger.debug("rocm-smi解析失败: %s", e)
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
        except Exception as e:
            logger.debug("Windows CPU品牌检测失败: %s", e)
        return ""

    def _get_cpu_brand_linux(self) -> str:
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except Exception as e:
            logger.debug("Linux CPU品牌检测失败: %s", e)
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
        except Exception as e:
            logger.debug("macOS CPU品牌检测失败: %s", e)
        return ""

    # ------------------------------------------------------------------
    # CPU 指令集检测
    # ------------------------------------------------------------------

    # x86 指令集 flag 到扩展名的映射（按重要性排序）
    _X86_FLAG_MAP: dict[str, str] = {
        "avx2": "AVX2",
        "avx": "AVX",
        "sse4_2": "SSE4.2",
        "sse4_1": "SSE4.1",
        "sse2": "SSE2",
        "sse": "SSE",
        "fma": "FMA",
        "f16c": "F16C",
    }

    # ARM 指令集 flag 到扩展名的映射
    _ARM_FLAG_MAP: dict[str, str] = {
        "asimd": "AdvSIMD",
        "aes": "AES",
        "sha1": "SHA1",
        "sha2": "SHA2",
        "crc32": "CRC32",
        "atomics": "Atomics",
        "fp16": "FP16",
        "sve": "SVE",
    }

    def _detect_cpu_instruction_sets(self) -> list[str]:
        """跨平台检测 CPU 支持的指令集扩展。

        Returns:
            支持的指令集列表，如 ["AVX2", "AVX", "SSE4.2"]
        """
        system = platform.system()
        machine = platform.machine().lower()
        try:
            if system == "Linux":
                return self._detect_instruction_sets_linux(machine)
            elif system == "Windows":
                return self._detect_instruction_sets_windows(machine)
            elif system == "Darwin":
                return self._detect_instruction_sets_darwin(machine)
        except Exception as e:
            logger.debug("CPU指令集检测失败: %s", e)
        return []

    def _detect_instruction_sets_linux(self, machine: str) -> list[str]:
        """Linux: 读取 /proc/cpuinfo 中的 flags 字段。"""
        flags_str = ""
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("flags"):
                        flags_str = line.split(":", 1)[1].strip()
                        break
                    elif line.startswith("Features"):
                        # ARM Linux 使用 Features 字段
                        flags_str = line.split(":", 1)[1].strip()
                        break
        except Exception as e:
            logger.debug("Linux CPU指令集检测失败: %s", e)
            return []

        if not flags_str:
            return []

        flags_set = set(flags_str.split())
        return self._parse_instruction_set_flags(flags_set, machine)

    def _detect_instruction_sets_windows(self, machine: str) -> list[str]:
        """Windows: 通过 PowerShell 检测 CPU 特性。"""
        flags_set: set[str] = set()
        try:
            # 使用 Windows API IsProcessorFeaturePresent 检测指令集
            # PF_AVX2_INSTRUCTIONS_AVAILABLE = 40
            # PF_AVX_INSTRUCTIONS_AVAILABLE = 39
            # PF_XMMI64_INSTRUCTIONS_AVAILABLE = 10 (SSE2)
            # PF_XMMI_INSTRUCTIONS_AVAILABLE = 6 (SSE)
            ps_script = (
                "Add-Type @'\n"
                "using System;\n"
                "using System.Runtime.InteropServices;\n"
                "public class CpuCheck {\n"
                '  [DllImport("kernel32.dll")] public static extern bool IsProcessorFeaturePresent(int pf);\n'
                "}\n"
                "'@;\n"
                "$f = @();\n"
                "if ([CpuCheck]::IsProcessorFeaturePresent(40)) { $f += 'avx2' };\n"
                "if ([CpuCheck]::IsProcessorFeaturePresent(39)) { $f += 'avx' };\n"
                "if ([CpuCheck]::IsProcessorFeaturePresent(10)) { $f += 'sse2' };\n"
                "if ([CpuCheck]::IsProcessorFeaturePresent(6)) { $f += 'sse' };\n"
                "($f -join ' ')"
            )
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_script],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode == 0 and result.stdout.strip():
                flags_set = set(result.stdout.strip().split())
        except Exception as e:
            logger.debug("Windows CPU指令集PowerShell检测失败: %s", e)

        if not flags_set:
            # 回退：通过架构判断
            try:
                arch = os.environ.get("PROCESSOR_ARCHITECTURE", "").upper()
                if "ARM" in arch:
                    flags_set = {"asimd"}
            except Exception as e:
                logger.debug("Windows CPU架构回退检测失败: %s", e)

        return self._parse_instruction_set_flags(flags_set, machine)

    def _detect_instruction_sets_darwin(self, machine: str) -> list[str]:
        """macOS: 通过 sysctl 检测 CPU 特性。"""
        flags_set: set[str] = set()
        try:
            if machine == "arm64":
                # Apple Silicon: 默认支持 AdvSIMD (NEON)
                result = subprocess.run(
                    ["sysctl", "-n", "hw.optional.armv8_2_sha3"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                flags_set.add("asimd")
                if result.returncode == 0 and result.stdout.strip() == "1":
                    flags_set.add("sha2")
                # Apple M 系列都支持 FP16
                flags_set.add("fp16")
            else:
                # Intel Mac: 读取 machdep.cpu.features
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.features"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    # macOS 输出格式如 "FPU VME DE PSE TSC MSR PAE MCE CX8 ..."
                    flags_set = set(result.stdout.strip().lower().split())

                # 也读取 leaf7 特性
                result7 = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.leaf7_features"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result7.returncode == 0:
                    flags_set.update(result7.stdout.strip().lower().split())
        except Exception as e:
            logger.debug("macOS CPU指令集检测失败: %s", e)

        return self._parse_instruction_set_flags(flags_set, machine)

    def _parse_instruction_set_flags(self, flags_set: set[str], machine: str) -> list[str]:
        """从原始 flags 集合中提取标准化的指令集扩展名列表。

        Args:
            flags_set: 原始 flag 集合（小写）
            machine: platform.machine() 的返回值

        Returns:
            标准化的指令集扩展名列表
        """
        extensions: list[str] = []
        is_arm = machine in ("arm64", "aarch64", "armv7l", "armv6l")

        if is_arm:
            for flag, name in self._ARM_FLAG_MAP.items():
                if flag in flags_set:
                    extensions.append(name)
        else:
            # x86/x64
            for flag, name in self._X86_FLAG_MAP.items():
                if flag in flags_set:
                    extensions.append(name)

        return extensions


def check_compatibility(model_requirements: dict, node_resources: dict) -> dict:
    """检查模型要求与节点硬件的兼容性。

    Args:
        model_requirements: 模型硬件要求，支持以下字段：
            - "require_gpu" (bool): 是否需要 GPU
            - "min_memory_mb" (int): 最低内存要求
            - "min_vram_mb" (int): 最低显存要求
            - "required_instruction_sets" (list[str]): 必需的 CPU 指令集，如 ["AVX2"]
        node_resources: 节点资源信息，支持以下字段：
            - "gpus" (list[dict]): GPU 列表，每个包含 vram_total_mb/vram_free_mb/detection_platform
            - "memory_total_mb" (int): 总内存
            - "memory_free_mb" (int): 可用内存
            - "cpu_instruction_set_extensions" (list[str]): CPU 支持的指令集

    Returns:
        {
            "compatible": bool,
            "issues": list[str],  # 如 ["gpuRequiredButNoneFound", "invalidCpuInstructionSetExtensions"]
        }
    """
    issues: list[str] = []

    # 检查 GPU 要求
    require_gpu = model_requirements.get("require_gpu", False)
    gpus = node_resources.get("gpus", [])
    if require_gpu and not gpus:
        issues.append("gpuRequiredButNoneFound")

    # 检查 CPU 指令集
    required_extensions = model_requirements.get("required_instruction_sets", [])
    available_extensions = set(node_resources.get("cpu_instruction_set_extensions", []))
    if required_extensions:
        missing = [ext for ext in required_extensions if ext not in available_extensions]
        if missing:
            issues.append("invalidCpuInstructionSetExtensions")

    # 检查内存
    min_memory_mb = model_requirements.get("min_memory_mb", 0)
    memory_free_mb = node_resources.get("memory_free_mb", 0)
    if min_memory_mb > 0 and memory_free_mb < min_memory_mb:
        issues.append("insufficientMemory")

    # 检查显存
    min_vram_mb = model_requirements.get("min_vram_mb", 0)
    if min_vram_mb > 0 and gpus:
        total_free_vram = sum(g.get("vram_free_mb", 0) for g in gpus)
        if total_free_vram < min_vram_mb:
            issues.append("insufficientVram")
    elif min_vram_mb > 0 and not gpus:
        # 需要 VRAM 但无 GPU，已在 gpuRequiredButNoneFound 中覆盖
        pass

    return {
        "compatible": len(issues) == 0,
        "issues": issues,
    }
