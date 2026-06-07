"""llama-server 推理引擎实现。

基于 llama.cpp 的 llama-server 构建的推理引擎，通过 HTTP API 与常驻进程通信。

核心改进：
- 常驻进程：避免每次推理启动新进程的冷启动开销（从数秒降至毫秒级）
- 流式输出：天然支持 SSE 流式输出，提升用户体验
- 分布式推理：通过 --rpc 和 --tensor-split 参数支持多节点张量并行
- 连接池复用：使用 httpx.Client 保持长连接，减少 TCP 握手开销

架构说明：
    LlamaServerBuilder 负责启动 llama-server 子进程并等待就绪，
    LlamaServerEngine 负责提交推理请求和管理进程生命周期。
    两者分离使得构建逻辑可被复用（如预热、健康检查），而运行逻辑保持简洁。

线程安全：
    submit_async() 使用 asyncio.to_thread() 将同步 HTTP 请求放到线程池，
    避免阻塞事件循环。submit() 为同步方法，仅在非 asyncio 环境使用。

资源管理：
    必须在程序退出时调用 engine.close()，否则 llama-server 子进程可能成为孤儿进程。
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Generator

import httpx

from asc.engine.base import (
    Engine,
    EngineBuilder,
    EngineStatus,
    InferenceRequest,
    LoadProgress,
)
from asc.utils.system import find_executable


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
    _http_client: httpx.Client | None = field(default=None, init=False, repr=False)

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

        self._http_client = httpx.Client(
            base_url=f"http://{self.host}:{self.port}",
            timeout=300.0,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )

        return LlamaServerEngine(
            model_path=self.model_path,
            base_url=f"http://{self.host}:{self.port}",
            process=self._process,
            http_client=self._http_client,
        )

    def _find_executable(self) -> str | None:
        """查找 llama-server 可执行文件。"""
        result = find_executable("llama-server")
        return str(result) if result is not None else None

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
        with httpx.Client(timeout=2.0) as client:
            while time.time() - start < timeout:
                try:
                    resp = client.get(f"http://{self.host}:{self.port}/health")
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
    使用 httpx.Client 连接池复用 TCP 连接。
    """

    model_path: str
    base_url: str
    process: subprocess.Popen = field(repr=False)
    http_client: httpx.Client = field(repr=False)
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

    async def submit_async(self, request: InferenceRequest) -> str:
        """提交推理请求（异步，不阻塞事件循环）。

        使用 asyncio.to_thread() 将同步的 submit() 调用放到线程池中执行，
        避免长时间 HTTP 请求阻塞事件循环。这是 FastAPI 端点中的推荐调用方式。

        Args:
            request: 推理请求，包含 prompt、max_tokens、temperature

        Returns:
            生成的完整文本

        Raises:
            RuntimeError: 引擎状态不为 READY 或 RUNNING

        性能注意：
            asyncio.to_thread() 会将任务提交到默认线程池（ThreadPoolExecutor），
            适合 IO 密集型操作（如 HTTP 请求）。若需更精细控制，可自定义 executor。
        """
        if self.status() not in (EngineStatus.READY, EngineStatus.RUNNING):
            raise RuntimeError(f"引擎状态为 {self.status()}，无法提交请求")

        self._status = EngineStatus.RUNNING
        return await asyncio.to_thread(self._sync_infer, request)

    def _sync_infer(self, request: InferenceRequest) -> str:
        """同步推理，返回完整结果文本。"""
        payload = {
            "messages": [{"role": "user", "content": request.prompt}],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": False,
        }

        try:
            resp = self.http_client.post(
                "/v1/chat/completions",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            self._status = EngineStatus.READY
            return text
        except Exception as e:
            self._status = EngineStatus.ERROR
            raise RuntimeError(f"推理请求失败: {e}") from e

    def _sync_infer_stream(self, request: InferenceRequest) -> Generator[str, None, None]:
        """同步流式推理，逐个 yield 生成的 token。"""
        payload = {
            "messages": [{"role": "user", "content": request.prompt}],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": True,
        }

        try:
            with self.http_client.stream(
                "POST",
                "/v1/chat/completions",
                json=payload,
                timeout=300.0,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]  # 去掉 "data: " 前缀
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            delta = data["choices"][0]["delta"]
                            if "content" in delta:
                                yield delta["content"]
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue
            self._status = EngineStatus.READY
        except Exception as e:
            self._status = EngineStatus.ERROR
            raise RuntimeError(f"流式推理请求失败: {e}") from e

    async def submit_async_stream(
        self, request: InferenceRequest
    ) -> AsyncGenerator[str, None]:
        """异步流式推理，逐个 yield 生成的 token。"""
        if self.status() not in (EngineStatus.READY, EngineStatus.RUNNING):
            raise RuntimeError(f"引擎状态为 {self.status()}，无法提交请求")

        self._status = EngineStatus.RUNNING
        loop = asyncio.get_running_loop()
        stream_gen = self._sync_infer_stream(request)
        _sentinel = object()

        try:
            while True:
                token = await loop.run_in_executor(None, next, stream_gen, _sentinel)
                if token is _sentinel:
                    # 生成器耗尽
                    self._status = EngineStatus.READY
                    return
                yield token
        except Exception as e:
            self._status = EngineStatus.ERROR
            raise RuntimeError(f"流式推理失败: {e}") from e

    def step(self) -> list[tuple[str, str]]:
        """llama-server 模式下不需要 step，推理在 submit 中完成。"""
        return []

    def close(self) -> None:
        """关闭引擎，终止 llama-server 进程并释放连接池。"""
        if self.http_client is not None:
            self.http_client.close()
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self._status = EngineStatus.SHUTDOWN
