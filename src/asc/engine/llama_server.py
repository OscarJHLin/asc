"""llama-server 推理引擎实现。

基于 llama.cpp 的 llama-server 构建的推理引擎，通过 HTTP API 与常驻进程通信。

核心改进：
- 常驻进程：避免每次推理启动新进程的冷启动开销（从数秒降至毫秒级）
- 流式输出：天然支持 SSE 流式输出，提升用户体验
- 分布式推理：通过 --rpc 和 --tensor-split 参数支持多节点张量并行
- 连接池复用：使用 httpx.Client 保持长连接，减少 TCP 握手开销
- 异步启动：aload()/abuild() 使用 asyncio.create_subprocess_exec，
  避免同步 subprocess.Popen 阻塞事件循环

架构说明：
    LlamaServerBuilder 负责启动 llama-server 子进程并等待就绪，
    LlamaServerEngine 负责提交推理请求和管理进程生命周期。
    两者分离使得构建逻辑可被复用（如预热、健康检查），而运行逻辑保持简洁。

    推荐使用异步路径（aload + abuild），同步路径（load + build）保留向后兼容。

线程安全：
    submit_async() 使用 asyncio.to_thread() 将同步 HTTP 请求放到线程池，
    避免阻塞事件循环。submit() 为同步方法，仅在非 asyncio 环境使用。

资源管理：
    必须在程序退出时调用 engine.close()，否则 llama-server 子进程可能成为孤儿进程。
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

import logging

import httpx

from asc.engine.base import (
    Engine,
    EngineBuilder,
    EngineStatus,
    InferenceRequest,
    LoadProgress,
)
from asc.utils.system import find_executable

logger = logging.getLogger(__name__)


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
    # 推理参数
    context_length: int = 4096       # -c
    cpu_threads: int = 0             # -t, 0 = auto
    batch_size: int = 512            # -b
    flash_attention: bool = True     # -fa
    use_mmap: bool = True            # --no-mmap (inverted)
    seed: int = -1                   # --seed, -1 = random
    rope_freq_base: float = 0.0      # --rope-freq-base, 0 = auto
    rope_freq_scale: float = 0.0     # --rope-freq-scale, 0 = auto
    keep_in_memory: bool = True      # 常驻内存（不释放）
    offload_kv_to_gpu: bool = True   # -nkvo (inverted: offload KV to GPU)
    kv_quantization: str = ""        # --cache-type-k
    # GPU 管理策略
    gpu_offload_ratio: str | float = "max"  # "max"/"off"/0-1
    gpu_split_strategy: str = "evenly"  # "evenly"/"priorityOrder"/"custom"
    gpu_custom_ratio: list[float] = field(default_factory=list)
    disabled_gpus: list[int] = field(default_factory=list)
    # 新增模型加载参数
    physical_batch_size: int = 0  # -ub, 0 = auto
    use_direct_io: bool = False  # --directio
    gpu_strict_vram_cap: bool = False  # 严格 VRAM 限制
    use_fp16_kv_cache: bool = True  # --cache-type-k f16/f32
    keep_model_in_memory: bool = True  # 常驻内存
    # 新增推理参数
    min_p: float = 0.0  # --min-p, 0 = disabled
    repeat_penalty: float = 0.0  # --repeat-penalty, 0 = disabled
    presence_penalty: float = 0.0  # --presence-penalty, 0 = disabled
    frequency_penalty: float = 0.0  # --frequency-penalty, 0 = disabled
    # 推测解码
    draft_model: str = ""  # -draft 模型路径
    draft_max_tokens: int = 0  # 草稿最大 token 数
    _process: subprocess.Popen | None = field(default=None, init=False, repr=False)
    _async_process: asyncio.subprocess.Process | None = field(
        default=None, init=False, repr=False
    )
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

        engine = LlamaServerEngine(
            model_path=self.model_path,
            base_url=f"http://{self.host}:{self.port}",
            process=self._process,
            http_client=self._http_client,
        )
        engine._status = EngineStatus.READY
        return engine

    def _find_executable(self) -> str | None:
        """查找 llama-server 可执行文件。"""
        result = find_executable("llama-server")
        return str(result) if result is not None else None

    @staticmethod
    def _get_lib_dirs(exe: str) -> list[str]:
        """获取 llama-server 可执行文件相关的库搜索目录。

        新版 llama-server (pip 安装) 将 .so 文件放在与 bin 同级的 lib/ 目录下，
        例如 ~/.local/bin/llama-server -> ~/.local/lib/libllama-server-impl.so。
        编译版则将 .so 放在 bin/ 同目录下。
        同时搜索 CUDA 运行时库路径。
        """
        exe_path = Path(exe).resolve()
        dirs: list[str] = []
        # 可执行文件所在目录（编译版 .so 通常在此）
        dirs.append(str(exe_path.parent))
        # pip 安装版：bin/ -> ../lib/
        parent = exe_path.parent.parent
        lib_dir = parent / "lib"
        if lib_dir.is_dir():
            dirs.append(str(lib_dir))
        # 也检查 lib64
        lib64_dir = parent / "lib64"
        if lib64_dir.is_dir():
            dirs.append(str(lib64_dir))
        # CUDA Toolkit（用户安装）
        cuda_toolkit = Path.home() / ".local" / "cuda-toolkit" / "targets" / "x86_64-linux" / "lib"
        if cuda_toolkit.is_dir():
            dirs.append(str(cuda_toolkit))
        # CUDA Toolkit（系统安装）
        for cuda_path in Path("/usr/local").glob("cuda*/lib64"):
            if cuda_path.is_dir():
                dirs.append(str(cuda_path))
        # Ollama 内置 CUDA 库
        ollama_cuda = Path("/usr/lib/ollama/cuda_v12")
        if ollama_cuda.is_dir():
            dirs.append(str(ollama_cuda))
        return dirs

    def _build_cmd(self, exe: str) -> list[str]:
        """构建 llama-server 启动命令。

        统一命令构建逻辑，供 _start_server 和 _start_server_async 共用。
        """
        cmd = [
            exe,
            "-m", self.model_path,
            "--host", self.host,
            "--port", str(self.port),
            "-c", str(self.context_length),
            "-b", str(self.batch_size),
        ]

        # --- GPU offload ratio → -ngl 转换 ---
        if self.gpu_offload_ratio == "max":
            cmd.extend(["-ngl", "999"])
        elif self.gpu_offload_ratio == "off":
            cmd.extend(["-ngl", "0"])
        elif isinstance(self.gpu_offload_ratio, (int, float)):
            ratio = float(self.gpu_offload_ratio)
            ngl = int(999 * ratio)
            cmd.extend(["-ngl", str(ngl)])
        else:
            # 兜底：使用 n_gpu_layers
            cmd.extend(["-ngl", str(self.n_gpu_layers)])

        # --- GPU split strategy → -sm/-ts ---
        if self.gpu_split_strategy == "evenly":
            cmd.extend(["-sm", "row"])
        elif self.gpu_split_strategy == "priorityOrder":
            cmd.extend(["-sm", "layer"])
        elif self.gpu_split_strategy == "custom":
            cmd.extend(["-sm", "row"])
            if self.gpu_custom_ratio:
                cmd.extend(["-ts", ",".join(str(s) for s in self.gpu_custom_ratio)])

        # --- disabled_gpus 过滤 → --device ---
        if self.disabled_gpus:
            try:
                from asc.worker.hardware import HardwareDetector
                all_gpus = HardwareDetector().detect_gpus()
                enabled_gpus = [
                    i for i in range(len(all_gpus)) if i not in self.disabled_gpus
                ]
                if enabled_gpus:
                    for gpu_idx in enabled_gpus:
                        cmd.extend(["--device", f"CUDA{gpu_idx}"])
            except ImportError as e:
                logger.warning("硬件检测模块不可用，无法过滤禁用 GPU: %s", e)
            except (OSError, RuntimeError) as e:
                logger.warning(
                    "GPU 检测失败，无法过滤禁用 GPU: %s [类型: %s]",
                    e, type(e).__name__,
                )
            except Exception as e:
                logger.error(
                    "GPU 过滤时发生未预期错误: %s [类型: %s]",
                    e, type(e).__name__,
                )

        # CPU 线程数（0 = auto，不传参数）
        if self.cpu_threads > 0:
            cmd.extend(["-t", str(self.cpu_threads)])

        # Flash Attention
        if self.flash_attention:
            cmd.extend(["-fa", "on"])
        else:
            cmd.extend(["-fa", "off"])

        # mmap（默认启用，--no-mmap 禁用）
        if not self.use_mmap:
            cmd.append("--no-mmap")

        # 随机种子（-1 = random，不传参数）
        if self.seed >= 0:
            cmd.extend(["--seed", str(self.seed)])

        # RoPE 频率参数（0 = auto，不传参数）
        if self.rope_freq_base > 0:
            cmd.extend(["--rope-freq-base", str(self.rope_freq_base)])
        if self.rope_freq_scale > 0:
            cmd.extend(["--rope-freq-scale", str(self.rope_freq_scale)])

        # KV 缓存不卸载到 GPU（默认卸载，-nkvo 禁用）
        if not self.offload_kv_to_gpu:
            cmd.append("-nkvo")

        # KV 缓存量化（use_fp16_kv_cache 或 kv_quantization）
        if self.kv_quantization:
            cmd.extend(["--cache-type-k", self.kv_quantization])
        elif self.use_fp16_kv_cache:
            cmd.extend(["--cache-type-k", "f16"])
        else:
            cmd.extend(["--cache-type-k", "f32"])

        # 分布式推理参数
        if self.rpc_servers:
            cmd.extend(["--rpc", ",".join(self.rpc_servers)])
        if self.tensor_split:
            cmd.extend(["--tensor-split", ",".join(str(s) for s in self.tensor_split)])

        # 多 GPU 环境 fallback：仅在未设置 gpu_split_strategy 且无 rpc/tensor_split 时
        if (
            self.gpu_split_strategy == "evenly"
            and not self.rpc_servers
            and not self.tensor_split
            and not self.gpu_custom_ratio
            and not self.disabled_gpus
        ):
            try:
                from asc.worker.hardware import HardwareDetector
                gpu_count = len(HardwareDetector().detect_gpus())
                if gpu_count > 1:
                    cmd.extend(["-sm", "none", "--device", "CUDA0"])
                    logger.info("检测到 %d 个 GPU，使用单设备模式避免崩溃", gpu_count)
            except ImportError as e:
                logger.warning("硬件检测模块不可用，将使用默认 GPU 配置: %s", e)
            except (OSError, RuntimeError) as e:
                logger.warning(
                    "GPU 检测失败，将使用默认配置: %s [类型: %s]",
                    e, type(e).__name__,
                )
            except Exception as e:
                logger.error(
                    "GPU 检测时发生未预期错误: %s [类型: %s]",
                    e, type(e).__name__,
                )

        # --- 新增模型加载参数 ---
        if self.physical_batch_size > 0:
            cmd.extend(["-ub", str(self.physical_batch_size)])

        if self.use_direct_io:
            cmd.append("--directio")

        # --- 新增推理参数 ---
        if self.min_p > 0:
            cmd.extend(["--min-p", str(self.min_p)])
        if self.repeat_penalty > 0:
            cmd.extend(["--repeat-penalty", str(self.repeat_penalty)])
        if self.presence_penalty > 0:
            cmd.extend(["--presence-penalty", str(self.presence_penalty)])
        if self.frequency_penalty > 0:
            cmd.extend(["--frequency-penalty", str(self.frequency_penalty)])

        # --- 推测解码 ---
        if self.draft_model:
            cmd.extend(["-draft", self.draft_model])
            if self.draft_max_tokens > 0:
                cmd.extend(["--draft-max-tokens", str(self.draft_max_tokens)])

        return cmd

    def _start_server(self, exe: str) -> None:
        """启动 llama-server 子进程。"""
        cmd = self._build_cmd(exe)

        # 设置 LD_LIBRARY_PATH，确保 llama-server 能找到所需的 .so 文件
        env = os.environ.copy()
        lib_dirs = self._get_lib_dirs(exe)
        ld_path = env.get("LD_LIBRARY_PATH", "")
        for d in lib_dirs:
            ld_path = f"{d}:{ld_path}" if ld_path else d
        if lib_dirs:
            env["LD_LIBRARY_PATH"] = ld_path

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
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

    # --- 异步路径（推荐） ---

    async def aload(self) -> AsyncGenerator[LoadProgress, None]:
        """异步启动 llama-server 并等待就绪。

        使用 asyncio.create_subprocess_exec 替代 subprocess.Popen，
        使用 asyncio.sleep + httpx.AsyncClient 替代 time.sleep + httpx.Client，
        避免阻塞事件循环。
        """
        exe = self._find_executable()
        if exe is None:
            raise FileNotFoundError("未找到 llama-server 可执行文件")

        yield LoadProgress(current=0, total=3, message="启动 llama-server 进程（异步）")
        await self._start_server_async(exe)

        yield LoadProgress(current=1, total=3, message="等待服务就绪（异步）")
        await self._wait_for_ready_async()

        yield LoadProgress(current=3, total=3, message="服务就绪")

    async def _start_server_async(self, exe: str) -> None:
        """使用 asyncio.create_subprocess_exec 启动 llama-server 子进程。"""
        cmd = self._build_cmd(exe)

        # 设置 LD_LIBRARY_PATH，确保 llama-server 能找到所需的 .so 文件
        env = os.environ.copy()
        lib_dirs = self._get_lib_dirs(exe)
        ld_path = env.get("LD_LIBRARY_PATH", "")
        for d in lib_dirs:
            ld_path = f"{d}:{ld_path}" if ld_path else d
        if lib_dirs:
            env["LD_LIBRARY_PATH"] = ld_path

        self._async_process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

    async def _wait_for_ready_async(self, timeout: float = 60.0) -> None:
        """异步等待 llama-server HTTP 服务就绪。

        使用 asyncio.sleep 替代 time.sleep，httpx.AsyncClient 替代 httpx.Client，
        完全不阻塞事件循环。
        """
        start = time.time()
        async with httpx.AsyncClient(timeout=2.0) as client:
            while time.time() - start < timeout:
                try:
                    resp = await client.get(f"http://{self.host}:{self.port}/health")
                    if resp.status_code == 200:
                        return
                except httpx.ConnectError:
                    pass
                await asyncio.sleep(0.5)

        if self._async_process and self._async_process.returncode is not None:
            stderr = ""
            if self._async_process.stderr:
                stderr_bytes = await self._async_process.stderr.read()
                stderr = stderr_bytes.decode("utf-8", errors="replace")
            raise RuntimeError(f"llama-server 启动失败: {stderr}")

        raise TimeoutError(f"llama-server 在 {timeout} 秒内未就绪")

    async def abuild(self) -> LlamaServerEngine:
        """异步构建，返回可用的 Engine 实例。

        与 build() 不同，abuild() 返回的引擎使用 asyncio.subprocess.Process，
        close() 会正确处理异步进程的生命周期。
        """
        if self._async_process is None:
            raise RuntimeError("必须先调用 aload()")

        self._http_client = httpx.Client(
            base_url=f"http://{self.host}:{self.port}",
            timeout=300.0,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )

        engine = LlamaServerEngine(
            model_path=self.model_path,
            base_url=f"http://{self.host}:{self.port}",
            process=self._async_process,
            http_client=self._http_client,
        )
        engine._status = EngineStatus.READY
        engine._is_async_process = True
        return engine


@dataclass
class LlamaServerEngine(Engine):
    """llama-server 推理引擎。

    通过 HTTP API 与常驻的 llama-server 进程通信。
    支持 OpenAI 兼容的 /v1/chat/completions 端点。
    使用 httpx.Client 连接池复用 TCP 连接。

    兼容两种进程类型：
    - subprocess.Popen（同步路径 load+build 产生）
    - asyncio.subprocess.Process（异步路径 aload+abuild 产生）
    """

    model_path: str
    base_url: str
    process: subprocess.Popen | asyncio.subprocess.Process = field(repr=False)
    http_client: httpx.Client = field(repr=False)
    _status: EngineStatus = field(default=EngineStatus.IDLE, init=False)
    _is_async_process: bool = field(default=False, init=False)

    def _set_status(self, status: EngineStatus) -> None:
        self._status = status

    def _is_process_alive(self) -> bool:
        """检查进程是否仍在运行，兼容同步和异步进程。"""
        if self._is_async_process:
            return self.process.returncode is None
        return self.process.poll() is None

    def status(self) -> EngineStatus:
        # 检查进程是否还活着
        if (
            self._status in (EngineStatus.READY, EngineStatus.RUNNING)
            and not self._is_process_alive()
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
            message = data["choices"][0]["message"]
            text = message.get("content", "") or ""
            # 思考模型（如 Gemma 4）的推理过程在 reasoning_content 中
            reasoning = message.get("reasoning_content", "") or ""
            if not text and reasoning:
                text = reasoning
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
        """关闭引擎，终止 llama-server 进程并释放连接池。

        兼容同步进程（subprocess.Popen）和异步进程（asyncio.subprocess.Process）：
        - 同步进程：terminate → wait(timeout=5) → kill
        - 异步进程：terminate → 同步等待最多3秒 → kill
        """
        if self.http_client is not None:
            self.http_client.close()
        if self._is_async_process:
            # asyncio.subprocess.Process 路径
            if self.process.returncode is None:
                self.process.terminate()
                # 给进程优雅退出的时间（同步等待是 close() 的已知限制）
                # 在异步上下文中应使用 aclose() 代替
                import time
                time.sleep(3)
                if self.process.returncode is None:
                    self.process.kill()
        else:
            # subprocess.Popen 路径
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
        self._status = EngineStatus.SHUTDOWN

    async def aclose(self) -> None:
        """异步关闭引擎，给进程优雅退出的时间。

        优先使用此方法替代同步 close()，可正确等待异步进程退出。
        """
        if self.http_client is not None:
            self.http_client.close()
        if self._is_async_process:
            # asyncio.subprocess.Process 路径
            if self.process.returncode is None:
                self.process.terminate()
                try:
                    await asyncio.wait_for(self.process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    self.process.kill()
                    try:
                        await self.process.wait()
                    except (OSError, ProcessLookupError) as e:
                        logger.debug("等待进程退出时出错: %s", e)
        else:
            # subprocess.Popen 路径
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
        self._status = EngineStatus.SHUTDOWN
