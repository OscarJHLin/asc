"""RpcServer 模块。

封装 Worker 节点上的 RPC 服务端生命周期：
- 启动 rpc-server 子进程
- 动态端口分配
- 就绪检测（端口监听）
- 优雅停止
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
from pathlib import Path
from typing import Any


class RpcServer:
    """RPC 服务端管理器。

    负责启动/停止 llama.cpp 的 rpc-server 进程。
    """

    DEFAULT_PORT_START: int = 50052
    DEFAULT_PORT_END: int = 50152
    DEFAULT_HOST: str = "0.0.0.0"

    def __init__(self) -> None:
        self._process: subprocess.Popen | None = None
        self._host: str = self.DEFAULT_HOST
        self._port: int | None = None

    @property
    def is_running(self) -> bool:
        """RPC 服务是否正在运行。"""
        return self._process is not None and self._process.poll() is None

    @property
    def port(self) -> int | None:
        """当前绑定的端口。"""
        return self._port

    @property
    def endpoint(self) -> str | None:
        """返回 RPC 端点地址（host:port）。"""
        if self._port is None:
            return None
        return f"{self._host}:{self._port}"

    def start(self, port: int | None = None) -> int:
        """启动 rpc-server。

        Args:
            port: 指定端口，None 则自动分配

        Returns:
            实际绑定的端口号

        Raises:
            FileNotFoundError: 未找到 rpc-server 可执行文件
            RuntimeError: 无法分配可用端口或启动失败
        """
        if self.is_running:
            raise RuntimeError("RPC Server 已在运行")

        exe = self._find_executable()
        if exe is None:
            raise FileNotFoundError("未找到 rpc-server 可执行文件")

        assigned_port = port or self._find_free_port()
        self._port = assigned_port

        cmd = [
            exe,
            "--rpc-server-bind",
            f"{self._host}:{assigned_port}",
        ]

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        return assigned_port

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

    def is_ready(self, timeout: float = 5.0) -> bool:
        """检测 RPC 服务是否就绪（端口监听）。

        Args:
            timeout: 检测超时时间（秒）
        """
        if not self.is_running or self._port is None:
            return False

        import time

        start = time.time()
        while time.time() - start < timeout:
            try:
                with socket.create_connection(
                    (self._host, self._port), timeout=1.0
                ):
                    return True
            except (OSError, ConnectionRefusedError):
                time.sleep(0.2)
        return False

    def status(self) -> dict[str, Any]:
        """返回 RPC 服务状态。"""
        return {
            "running": self.is_running,
            "host": self._host if self.is_running else None,
            "port": self._port,
            "endpoint": self.endpoint,
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

    def _find_free_port(self) -> int:
        """在默认范围内查找可用端口。"""
        for port in range(self.DEFAULT_PORT_START, self.DEFAULT_PORT_END + 1):
            if self._is_port_free(port):
                return port
        raise RuntimeError(
            f"无法找到可用端口（范围 {self.DEFAULT_PORT_START}-{self.DEFAULT_PORT_END}）"
        )

    @staticmethod
    def _is_port_free(port: int) -> bool:
        """检查端口是否可用。"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return True
            except OSError:
                return False
