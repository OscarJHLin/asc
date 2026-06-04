"""llama-server 推理引擎实现。

核心改进：使用 llama-server 常驻进程 + OpenAI 兼容 HTTP API。
- 避免每次推理启动新进程的冷启动开销
- 天然支持流式 SSE 输出
- 支持分布式推理（--rpc + --tensor-split）
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

import httpx

from asc.engine.base import (
    Engine,
    EngineBuilder,
    EngineStatus,
    InferenceRequest,
    LoadProgress,
)


@dataclass
class LlamaServerBuilder(EngineBuilder):
    """llama-server 引擎构建器。

    构建流程：
    1. 查找 llama-server 可执行文件
    2. 启动 llama-server 子进程
    3. 等待服务就绪（健康检查）
    4. 返回 LlamaServerEngine
    """

    model_path: str
    host: str = "127.0.0.1"
    port: int = 8081
    rpc_servers: list[str] = field(default_factory=list)
    tensor_split: list[float] = field(default_factory=list)
    n_gpu_layers: int = -1  # -1 = 全部加载到 GPU
    _process: subprocess.Popen | None = field(default=None, init=False, repr=False)

    def load(self) -> Generator[LoadProgress, None, None]:
        """启动 llama-server 并等待就绪。"""
        exe = self._find_executable()
        if exe is None:
            raise FileNotFoundError("未找到 llama-server 可执行文件")

        yield LoadProgress(current=0, total=3, message="启动 llama-server 进程")
        self._start_server(exe)

        yield LoadProgress(current=1, total=3, message="等待服务就绪")
        self._wait_for_ready()

        yield LoadProgress(current=3, total=3, message="服务就绪")

    def build(self) -> LlamaServerEngine:
        """返回可用的 Engine 实例。"""
        if self._process is None:
            raise RuntimeError("必须先调用 load()")

        return LlamaServerEngine(
            model_path=self.model_path,
            base_url=f"http://{self.host}:{self.port}",
            process=self._process,
        )

    def _find_executable(self) -> str | None:
        """查找 llama-server 可执行文件。"""
        # 项目内路径
        project_root = Path(__file__).parent.parent.parent.parent.resolve()
        candidates = [
            project_root / "llama.cpp" / "build" / "bin" / "Release" / "llama-server.exe",
            project_root / "llama.cpp" / "build" / "bin" / "llama-server",
            project_root / "llama.cpp" / "build-linux" / "bin" / "llama-server",
        ]

        # 环境变量自定义路径
        import os

        custom_path = os.getenv("ASC_LLAMA_PATH")
        if custom_path:
            candidates.insert(0, Path(custom_path) / "llama-server.exe")
            candidates.insert(1, Path(custom_path) / "llama-server")

        for path in candidates:
            if path.exists():
                return str(path.resolve())

        # 系统 PATH
        import shutil

        found = shutil.which("llama-server")
        if found:
            return found

        return None

    def _start_server(self, exe: str) -> None:
        """启动 llama-server 子进程。"""
        cmd = [
            exe,
            "-m",
            self.model_path,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "-ngl",
            str(self.n_gpu_layers),
        ]

        # 分布式推理参数
        if self.rpc_servers:
            cmd.extend(["--rpc", ",".join(self.rpc_servers)])
        if self.tensor_split:
            cmd.extend(["--tensor-split", ",".join(str(s) for s in self.tensor_split)])

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def _wait_for_ready(self, timeout: float = 60.0) -> None:
        """等待 llama-server HTTP 服务就绪。"""
        start = time.time()
        while time.time() - start < timeout:
            try:
                resp = httpx.get(f"http://{self.host}:{self.port}/health", timeout=2.0)
                if resp.status_code == 200:
                    return
            except httpx.ConnectError:
                pass
            time.sleep(0.5)

        if self._process and self._process.poll() is not None:
            stderr = self._process.stderr.read() if self._process.stderr else ""
            raise RuntimeError(f"llama-server 启动失败: {stderr}")

        raise TimeoutError(f"llama-server 在 {timeout} 秒内未就绪")


@dataclass
class LlamaServerEngine(Engine):
    """llama-server 推理引擎。

    通过 HTTP API 与常驻的 llama-server 进程通信。
    支持 OpenAI 兼容的 /v1/chat/completions 端点。
    """

    model_path: str
    base_url: str
    process: subprocess.Popen = field(repr=False)
    _status: EngineStatus = field(default=EngineStatus.IDLE, init=False)

    def _set_status(self, status: EngineStatus) -> None:
        self._status = status

    def status(self) -> EngineStatus:
        # 检查进程是否还活着
        if (
            self._status in (EngineStatus.READY, EngineStatus.RUNNING)
            and self.process.poll() is not None
        ):
            self._status = EngineStatus.ERROR
        return self._status

    def submit(self, request: InferenceRequest) -> str:
        """提交推理请求（同步，通过 HTTP POST）。"""
        if self.status() not in (EngineStatus.READY, EngineStatus.RUNNING):
            raise RuntimeError(f"引擎状态为 {self.status()}，无法提交请求")

        self._status = EngineStatus.RUNNING
        return self._sync_infer(request)

    def _sync_infer(self, request: InferenceRequest) -> str:
        """同步推理，返回完整结果文本。"""
        payload = {
            "messages": [{"role": "user", "content": request.prompt}],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": False,
        }

        try:
            resp = httpx.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=300.0,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            self._status = EngineStatus.READY
            return text
        except Exception as e:
            self._status = EngineStatus.ERROR
            raise RuntimeError(f"推理请求失败: {e}") from e

    def step(self) -> list[tuple[str, str]]:
        """llama-server 模式下不需要 step，推理在 submit 中完成。"""
        return []

    def close(self) -> None:
        """关闭引擎，终止 llama-server 进程。"""
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self._status = EngineStatus.SHUTDOWN
