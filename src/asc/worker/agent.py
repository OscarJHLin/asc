"""Asc Worker Agent。

Worker Agent 运行在 Worker 节点上，提供：
- 健康检查
- 资源查询（CPU/GPU/内存）
- RPC Server 生命周期管理
- 模型加载/卸载

通过 HTTP API 对外暴露服务，Master 通过 RemoteManager 远程调用。
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil

from asc.worker.benchmark_score import BenchmarkScore
from asc.worker.gpu_info import GPUInfo
from asc.worker.hardware import HardwareDetector
from asc.worker.rpc_server import RpcServer


@dataclass(frozen=True)
class NodeResources:
    """节点资源信息。"""

    cpu_count: int
    cpu_percent: float
    memory_total_mb: int
    memory_free_mb: int
    gpus: list[GPUInfo] = field(default_factory=list)
    compute_score: float = 0.0
    cpu_physical_count: int = 0
    cpu_freq_mhz: float = 0.0
    cpu_brand: str = ""
    disk_free_mb: int = 0
    network_mbps: float = 0.0

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

    def __init__(
        self,
        node_id: str,
        port: int = 52415,
        benchmark_score: BenchmarkScore | None = None,
        hardware_detector: HardwareDetector | None = None,
        rpc_server: RpcServer | None = None,
    ) -> None:
        self.node_id = node_id
        self.port = port
        self._rpc_manager = RPCServerManager()
        self._rpc_server = rpc_server or RpcServer()
        self._benchmark_score = benchmark_score or BenchmarkScore(node_id=node_id)
        self._hardware_detector = hardware_detector or HardwareDetector()

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

        cpu_info = self._hardware_detector.detect_cpu()
        gpus = self._hardware_detector.detect_gpus()
        disk = self._hardware_detector.detect_disk()
        network = self._hardware_detector.detect_network()

        try:
            report = self._benchmark_score.run_benchmark()
            compute_score = report.relative_score
        except Exception:
            compute_score = 0.0

        return NodeResources(
            cpu_count=cpu_count,
            cpu_percent=cpu_percent,
            memory_total_mb=int(mem.total // (1024 * 1024)),
            memory_free_mb=int(mem.available // (1024 * 1024)),
            gpus=gpus,
            compute_score=compute_score,
            cpu_physical_count=cpu_info.physical_count,
            cpu_freq_mhz=cpu_info.freq_mhz,
            cpu_brand=cpu_info.brand,
            disk_free_mb=disk.free_mb,
            network_mbps=network.estimated_mbps,
        )

    def start_rpc(self, port: int | None = None) -> dict[str, Any]:
        """启动 RPC Server。

        Args:
            port: 指定端口，None 则自动分配
        """
        try:
            assigned_port = self._rpc_server.start(port=port)
            return {"status": "ok", "port": assigned_port, "endpoint": self._rpc_server.endpoint}
        except (FileNotFoundError, RuntimeError) as e:
            return {"status": "error", "error": str(e)}

    def stop_rpc(self) -> dict[str, Any]:
        """停止 RPC Server。"""
        self._rpc_server.stop()
        return {"status": "ok"}

    def rpc_status(self) -> dict[str, Any]:
        """查询 RPC Server 状态。"""
        return self._rpc_server.status()




