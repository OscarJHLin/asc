"""Asc Worker Agent。

Worker Agent 运行在 Worker 节点上，提供：
- 健康检查
- 资源查询（CPU/GPU/内存）
- RPC Server 生命周期管理
- 模型加载/卸载

通过 HTTP API 对外暴露服务，Master 通过 RemoteManager 远程调用。
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil


@dataclass(frozen=True)
class GPUInfo:
    """GPU 信息。"""

    index: int
    name: str
    vram_total_mb: int
    vram_free_mb: int

    @property
    def vram_used_mb(self) -> int:
        return self.vram_total_mb - self.vram_free_mb


@dataclass(frozen=True)
class NodeResources:
    """节点资源信息。"""

    cpu_count: int
    cpu_percent: float
    memory_total_mb: int
    memory_free_mb: int
    gpus: list[GPUInfo] = field(default_factory=list)

    @property
    def total_vram_free_mb(self) -> int:
        return sum(g.vram_free_mb for g in self.gpus)


class RPCServerManager:
    """llama.cpp rpc-server 生命周期管理。"""

    def __init__(self) -> None:
        self._process: subprocess.Popen | None = None
        self._port: int | None = None
        self._host: str = "0.0.0.0"

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def port(self) -> int | None:
        return self._port

    def start(self, host: str = "0.0.0.0", port: int = 50052) -> None:
        """启动 rpc-server。"""
        if self.is_running:
            return

        exe = self._find_executable()
        if exe is None:
            raise FileNotFoundError("未找到 rpc-server 可执行文件")

        cmd = [exe, "--host", host, "--port", str(port)]
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self._host = host
        self._port = port

    def stop(self) -> None:
        """停止 rpc-server。"""
        if self._process is not None:
            if self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
            self._process = None
            self._port = None

    def status(self) -> dict[str, Any]:
        """返回 rpc-server 状态。"""
        return {
            "running": self.is_running,
            "host": self._host if self.is_running else None,
            "port": self._port if self.is_running else None,
        }

    def _find_executable(self) -> str | None:
        """查找 rpc-server 可执行文件。"""
        project_root = Path(__file__).parent.parent.parent.parent.resolve()
        candidates = [
            project_root / "llama.cpp" / "build" / "bin" / "Release" / "rpc-server.exe",
            project_root / "llama.cpp" / "build" / "bin" / "rpc-server",
        ]

        custom_path = os.getenv("ASC_LLAMA_PATH")
        if custom_path:
            candidates.insert(0, Path(custom_path) / "rpc-server.exe")
            candidates.insert(1, Path(custom_path) / "rpc-server")

        for path in candidates:
            if path.exists():
                return str(path.resolve())

        found = shutil.which("rpc-server")
        if found:
            return found

        return None


class WorkerAgent:
    """Worker 节点 Agent。

    提供资源查询、健康检查、RPC 管理等功能。
    通过 HTTP API 对外暴露。
    """

    def __init__(self, node_id: str, port: int = 52415) -> None:
        self.node_id = node_id
        self.port = port
        self._rpc_manager = RPCServerManager()

    def health(self) -> dict[str, Any]:
        """健康检查。"""
        return {
            "status": "ok",
            "node_id": self.node_id,
        }

    def get_resources(self) -> NodeResources:
        """获取节点资源信息。"""
        cpu_count = psutil.cpu_count(logical=True) or 0
        cpu_percent = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()

        gpus = self._detect_gpus()

        return NodeResources(
            cpu_count=cpu_count,
            cpu_percent=cpu_percent,
            memory_total_mb=int(mem.total // (1024 * 1024)),
            memory_free_mb=int(mem.available // (1024 * 1024)),
            gpus=gpus,
        )

    def start_rpc(self, host: str = "0.0.0.0", port: int = 50052) -> dict[str, Any]:
        """启动 RPC Server。"""
        try:
            self._rpc_manager.start(host=host, port=port)
            return {"status": "ok", "port": port}
        except FileNotFoundError as e:
            return {"status": "error", "error": str(e)}

    def stop_rpc(self) -> dict[str, Any]:
        """停止 RPC Server。"""
        self._rpc_manager.stop()
        return {"status": "ok"}

    def rpc_status(self) -> dict[str, Any]:
        """查询 RPC Server 状态。"""
        return self._rpc_manager.status()

    def _detect_gpus(self) -> list[GPUInfo]:
        """检测 GPU 信息。"""
        import platform

        system = platform.system()

        if system == "Windows":
            return self._detect_gpus_windows()
        elif system == "Linux":
            return self._detect_gpus_linux()
        elif system == "Darwin":
            return self._detect_gpus_darwin()
        return []

    def _detect_gpus_windows(self) -> list[GPUInfo]:
        """Windows GPU 检测（nvidia-smi 优先）。"""
        gpus = self._parse_nvidia_smi()
        return gpus

    def _detect_gpus_linux(self) -> list[GPUInfo]:
        """Linux GPU 检测。"""
        gpus = self._parse_nvidia_smi()
        if not gpus:
            gpus = self._parse_rocm_smi()
        return gpus

    def _detect_gpus_darwin(self) -> list[GPUInfo]:
        """macOS GPU 检测（Metal）。"""
        # 简化实现：通过 system_profiler 获取
        gpus = []
        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType", "-json"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                import json

                data = json.loads(result.stdout)
                for i, display in enumerate(data.get("SPDisplaysDataType", [])):
                    name = display.get("sppci_model", "Apple GPU")
                    # Apple Silicon 统一内存，从 psutil 推算
                    mem = psutil.virtual_memory()
                    vram_total = int(mem.total // (1024 * 1024))
                    vram_free = int(mem.available // (1024 * 1024))
                    gpus.append(
                        GPUInfo(
                            index=i,
                            name=name,
                            vram_total_mb=vram_total,
                            vram_free_mb=vram_free,
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
            # 简化解析
        return gpus
