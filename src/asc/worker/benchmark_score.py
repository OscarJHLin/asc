"""BenchmarkScore 标准化性能测试模块。

基于小参数大语言模型的节点性能测试方案：
- 使用统一轻量级基准模型（如 Qwen2.5-0.5B-Instruct Q4_K_M）
- 标准测试问题集覆盖不同复杂度
- 精确测量 TPS（tokens/秒）并计算相对性能评分
- 支持评分缓存，避免重复测试
- 预留 HardwareFactor 扩展接口
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import psutil

from asc.utils.system import find_executable
from asc.worker.hardware import HardwareDetector


@dataclass
class BenchmarkQuestion:
    """单个基准测试问题。"""

    id: str
    category: str
    difficulty: str
    prompt: str
    expected_tokens_range: list[int]
    description: str


@dataclass
class QuestionResult:
    """单个问题的测试结果。"""

    question_id: str
    prompt: str
    output_text: str
    output_tokens: int
    elapsed_ms: float
    tps: float


@dataclass
class BenchmarkReport:
    """标准化测试报告。"""

    timestamp: str
    node_id: str
    node_hardware_summary: dict[str, Any]
    model_name: str
    quantization: str
    question_results: list[QuestionResult]
    avg_elapsed_ms: float
    avg_tps: float
    theoretical_score: float
    standard_score: float
    relative_score: float

    def to_dict(self) -> dict[str, Any]:
        """转换为可序列化的字典。"""
        return {
            "timestamp": self.timestamp,
            "node_id": self.node_id,
            "node_hardware_summary": self.node_hardware_summary,
            "model_name": self.model_name,
            "quantization": self.quantization,
            "question_results": [
                {
                    "question_id": r.question_id,
                    "prompt": r.prompt,
                    "output_text": r.output_text,
                    "output_tokens": r.output_tokens,
                    "elapsed_ms": round(r.elapsed_ms, 2),
                    "tps": round(r.tps, 2),
                }
                for r in self.question_results
            ],
            "avg_elapsed_ms": round(self.avg_elapsed_ms, 2),
            "avg_tps": round(self.avg_tps, 2),
            "theoretical_score": round(self.theoretical_score, 2),
            "standard_score": round(self.standard_score, 2),
            "relative_score": round(self.relative_score, 4),
        }


class HardwareFactor:
    """硬件因素加权扩展接口（预留）。

    后续可扩展：
    - VRAM 容量加权
    - 内存带宽加权
    - CPU 核心数加权
    - 网络带宽加权
    """

    def apply(self, base_score: float, hardware_info: dict[str, Any]) -> float:
        """将硬件因素应用到基础评分上。

        Args:
            base_score: 基础 TPS 评分
            hardware_info: 节点硬件信息摘要

        Returns:
            加权后的评分
        """
        # 默认不做任何加权，直接返回基础分数
        return base_score


class BenchmarkScore:
    """标准化性能测试核心类。

    使用流程：
        bs = BenchmarkScore(node_id="worker-1")
        report = bs.run_benchmark()
        score = report.relative_score
    """

    # 默认标准性能分数（TPS），由参考机器测定或预设
    DEFAULT_STANDARD_SCORE: float = 50.0

    # 默认基准模型配置
    DEFAULT_CONTEXT_LENGTH: int = 8192

    # 基准模型候选列表（按优先级排列，第一个找到的即为基准模型）
    _BENCHMARK_MODEL_CANDIDATES: list[str] = [
        "gemma-4-E2B-it-UD-Q2_K_XL.gguf",
        "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
    ]

    # 默认模型名和量化方式（向后兼容，用于测试等场景）
    DEFAULT_MODEL_NAME: str = "Gemma-4-E2B-it-UD-Q2_K_XL"
    DEFAULT_QUANTIZATION: str = "UD-Q2_K_XL"

    # 模型文件名到模型名的映射
    _MODEL_NAME_MAP: dict[str, str] = {
        "gemma-4-E2B-it-UD-Q2_K_XL.gguf": "Gemma-4-E2B-it-UD-Q2_K_XL",
        "qwen2.5-0.5b-instruct-q4_k_m.gguf": "Qwen2.5-0.5B-Instruct",
        "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf": "Qwen2.5-0.5B-Instruct",
    }

    # 模型文件名到量化方式的映射
    _MODEL_QUANT_MAP: dict[str, str] = {
        "gemma-4-E2B-it-UD-Q2_K_XL.gguf": "UD-Q2_K_XL",
        "qwen2.5-0.5b-instruct-q4_k_m.gguf": "Q4_K_M",
        "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf": "Q4_K_M",
    }

    def __init__(
        self,
        node_id: str,
        model_path: str | None = None,
        questions_path: str | None = None,
        cache_path: str | None = None,
        llama_server_port: int = 18081,
        standard_score: float | None = None,
        hardware_factor: HardwareFactor | None = None,
    ) -> None:
        """初始化 BenchmarkScore。

        Args:
            node_id: 节点唯一标识
            model_path: 基准模型 GGUF 文件路径，None 时自动查找
            questions_path: 测试问题 JSON 路径，None 时使用内置默认
            cache_path: 评分缓存文件路径
            llama_server_port: llama-server 服务端口
            standard_score: 标准性能分数（TPS），None 时使用默认值
            hardware_factor: 硬件因素加权器（预留扩展）
        """
        self.node_id = node_id
        self.model_path = model_path or self._find_default_model()
        self._model_name = self._infer_model_name()
        self._quantization = self._infer_quantization()
        self.questions_path = questions_path or self._default_questions_path()
        self.cache_path = cache_path or self._default_cache_path()
        self.llama_server_port = llama_server_port
        self.standard_score = (
            standard_score if standard_score is not None else self.DEFAULT_STANDARD_SCORE
        )
        self.hardware_factor = hardware_factor or HardwareFactor()

        self._process: subprocess.Popen | None = None
        self._questions: list[BenchmarkQuestion] = []
        self._hardware_detector = HardwareDetector()

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def run_benchmark(self) -> BenchmarkReport:
        """运行完整的基准测试并返回报告。

        如果缓存有效且未过期，直接返回缓存结果。
        """
        cached = self._load_cache()
        if cached is not None:
            return cached

        self._load_questions()
        self._deploy_benchmark_model()
        try:
            results: list[QuestionResult] = []
            for q in self._questions:
                result = self._run_single_question(q)
                results.append(result)

            if not results:
                raise RuntimeError("基准测试未产生任何结果，请检查问题集和模型配置")

            avg_elapsed = sum(r.elapsed_ms for r in results) / len(results)
            avg_tps = sum(r.tps for r in results) / len(results)
            theoretical = avg_tps
            relative = self.calculate_score(theoretical)

            report = BenchmarkReport(
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
                node_id=self.node_id,
                node_hardware_summary=self._get_hardware_summary(),
                model_name=self._model_name,
                quantization=self._quantization,
                question_results=results,
                avg_elapsed_ms=avg_elapsed,
                avg_tps=avg_tps,
                theoretical_score=theoretical,
                standard_score=self.standard_score,
                relative_score=relative,
            )
            self._save_cache(report)
            return report
        finally:
            self._shutdown_server()

    def calculate_score(self, theoretical_score: float) -> float:
        """计算节点性能评分（相对值）。

        评分 = 节点理论分数 / 标准分数

        Args:
            theoretical_score: 节点实测平均 TPS

        Returns:
            相对性能评分（无量纲，>=0）
        """
        if self.standard_score <= 0:
            return 0.0
        base = theoretical_score / self.standard_score
        # 预留扩展：应用硬件因素加权
        return self.hardware_factor.apply(base, self._get_hardware_summary())

    def generate_report(self, report: BenchmarkReport, output_path: str | None = None) -> str:
        """生成标准化 JSON 测试报告文件。

        Args:
            report: 测试报告对象
            output_path: 输出文件路径，None 时使用默认路径

        Returns:
            生成的报告文件路径
        """
        path = Path(output_path or f"benchmark_report_{self.node_id}_{int(time.time())}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        return str(path)

    def clear_cache(self) -> None:
        """清除本地评分缓存。"""
        path = Path(self.cache_path)
        if path.exists():
            path.unlink()

    # ------------------------------------------------------------------
    # 模型部署
    # ------------------------------------------------------------------

    def _deploy_benchmark_model(self) -> None:
        """在节点上启动 llama-server 加载基准模型。

        优先使用独立的 llama-server 可执行文件，
        若不存在则回退到 llama-cpp-python 的 Python API。
        """
        if self._process is not None and self._process.poll() is None:
            # 已有运行中的 server，复用
            return

        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"基准模型文件不存在: {self.model_path}")

        exe = self._find_llama_server()
        if exe is not None:
            cmd = [
                exe,
                "-m", self.model_path,
                "--host", "127.0.0.1",
                "--port", str(self.llama_server_port),
                "-ngl", "-1",  # 全部加载到 GPU（若可用），CPU 节点会自动回退
                "-c", str(self.DEFAULT_CONTEXT_LENGTH),
            ]

            # 多 GPU 环境：使用单设备模式避免 GGML_SCHED_MAX_SPLIT_INPUTS 崩溃
            # 当检测到多个 GPU 时，-ngl -1 会让 llama.cpp 将模型分割到多个 GPU，
            # 可能触发 GGML_ASSERT(n_inputs < GGML_SCHED_MAX_SPLIT_INPUTS)。
            # 解决方案：使用 -sm none --device CUDA0 只用一个 GPU 运行基准测试。
            gpu_count = len(self._hardware_detector.detect_gpus())
            if gpu_count > 1:
                cmd.extend(["-sm", "none", "--device", "CUDA0"])

            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self._wait_for_ready()
        else:
            # 回退：使用 llama-cpp-python 启动 HTTP server
            self._start_python_server()

    def _start_python_server(self) -> None:
        """使用 llama-cpp-python 直接加载模型并启动 HTTP server。

        使用 llama-cpp-python 的 Llama 类加载模型，
        然后通过 uvicorn 暴露 OpenAI 兼容 API。
        """
        try:
            from llama_cpp import Llama
        except ImportError:
            raise FileNotFoundError(
                "未找到 llama-server 可执行文件，也未安装 llama-cpp-python。"
                "请安装 llama.cpp 或运行: pip install llama-cpp-python"
            )

        # 在子进程中启动 llama-cpp-python server
        import sys

        server_script = f"""
import sys
import os
os.environ['LLAMA_ARG_N_CTX'] = '{self.DEFAULT_CONTEXT_LENGTH}'
from llama_cpp import Llama
llm = Llama(model_path={repr(self.model_path)}, n_ctx={self.DEFAULT_CONTEXT_LENGTH}, n_gpu_layers=-1, verbose=False)

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
import json
import time

app = FastAPI()

@app.get('/health')
async def health():
    return {{'status': 'ok'}}

@app.post('/v1/chat/completions')
async def chat_completions(request: dict):
    messages = request.get('messages', [])
    prompt = ''
    for m in messages:
        prompt += m.get('content', '') + '\\n'
    max_tokens = request.get('max_tokens', 128)
    temperature = request.get('temperature', 0.7)

    start = time.perf_counter()
    result = llm(prompt, max_tokens=max_tokens, temperature=temperature, echo=False)
    elapsed = (time.perf_counter() - start) * 1000

    text = result['choices'][0]['text'] if result.get('choices') else ''
    usage = result.get('usage', {{}})
    return JSONResponse({{
        'choices': [{{'message': {{'content': text}}, 'index': 0}}],
        'usage': usage,
    }})

uvicorn.run(app, host='127.0.0.1', port={self.llama_server_port})
"""
        self._process = subprocess.Popen(
            [sys.executable, "-c", server_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self._wait_for_ready()

    def _shutdown_server(self) -> None:
        """关闭 llama-server 进程。"""
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
        self._process = None

    def _wait_for_ready(self, timeout: float = 60.0) -> None:
        """等待 llama-server HTTP 服务就绪。"""
        url = f"http://127.0.0.1:{self.llama_server_port}/health"
        start = time.time()
        while time.time() - start < timeout:
            try:
                resp = httpx.get(url, timeout=2.0)
                if resp.status_code == 200:
                    return
            except httpx.ConnectError:
                pass
            time.sleep(0.5)

        if self._process and self._process.poll() is not None:
            stderr = self._process.stderr.read() if self._process.stderr else ""
            raise RuntimeError(f"llama-server 启动失败: {stderr}")
        raise TimeoutError(f"llama-server 在 {timeout} 秒内未就绪")

    # ------------------------------------------------------------------
    # 单问题执行
    # ------------------------------------------------------------------

    def _run_single_question(self, question: BenchmarkQuestion) -> QuestionResult:
        """运行单个问题并记录时间/TPS。

        使用 OpenAI 兼容 API 请求，精确测量首 token 到末 token 的总时间。
        """
        url = f"http://127.0.0.1:{self.llama_server_port}/v1/chat/completions"
        payload = {
            "messages": [{"role": "user", "content": question.prompt}],
            "max_tokens": max(question.expected_tokens_range),
            "temperature": 0.7,
            "stream": False,
        }

        start = time.perf_counter()
        resp = httpx.post(url, json=payload, timeout=300.0)
        resp.raise_for_status()
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        # 使用简单空格分词估算 token 数（llama-server 未返回 usage 时的兜底）
        usage = data.get("usage", {})
        output_tokens = usage.get("completion_tokens", 0)
        if output_tokens == 0:
            output_tokens = len(text.split())

        tps = (output_tokens / elapsed_ms) * 1000.0 if elapsed_ms > 0 else 0.0
        return QuestionResult(
            question_id=question.id,
            prompt=question.prompt,
            output_text=text,
            output_tokens=output_tokens,
            elapsed_ms=elapsed_ms,
            tps=tps,
        )

    # ------------------------------------------------------------------
    # 问题加载
    # ------------------------------------------------------------------

    def _load_questions(self) -> None:
        """加载标准测试问题集合。"""
        path = Path(self.questions_path)
        if not path.exists():
            raise FileNotFoundError(f"测试问题文件不存在: {path}")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self._questions = [
            BenchmarkQuestion(
                id=q["id"],
                category=q["category"],
                difficulty=q["difficulty"],
                prompt=q["prompt"],
                expected_tokens_range=q["expected_tokens_range"],
                description=q["description"],
            )
            for q in data.get("questions", [])
        ]
        if not self._questions:
            raise ValueError("测试问题集为空")

    # ------------------------------------------------------------------
    # 缓存机制
    # ------------------------------------------------------------------

    def _load_cache(self) -> BenchmarkReport | None:
        """从本地文件加载缓存的评分结果。"""
        path = Path(self.cache_path)
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            # 简单校验：node_id 和模型名匹配
            if data.get("node_id") != self.node_id:
                return None
            if data.get("model_name") != self._model_name:
                return None
            # 重建 report（简化版，不重建完整 QuestionResult 列表）
            return BenchmarkReport(
                timestamp=data["timestamp"],
                node_id=data["node_id"],
                node_hardware_summary=data.get("node_hardware_summary", {}),
                model_name=data["model_name"],
                quantization=data.get("quantization", self._quantization),
                question_results=[],
                avg_elapsed_ms=data.get("avg_elapsed_ms", 0.0),
                avg_tps=data.get("avg_tps", 0.0),
                theoretical_score=data.get("theoretical_score", 0.0),
                standard_score=data.get("standard_score", self.standard_score),
                relative_score=data.get("relative_score", 0.0),
            )
        except Exception:
            return None

    def _is_cache_valid(self) -> bool:
        """检查缓存是否有效（文件存在且内容可解析）。"""
        return self._load_cache() is not None

    def _save_cache(self, report: BenchmarkReport) -> None:
        """保存评分结果到本地文件。"""
        path = Path(self.cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # 硬件摘要
    # ------------------------------------------------------------------

    def _get_hardware_summary(self) -> dict[str, Any]:
        """获取节点硬件信息摘要。"""
        mem = psutil.virtual_memory()
        gpus = self._hardware_detector.detect_gpus()
        return {
            "cpu_count": psutil.cpu_count(logical=True) or 0,
            "memory_total_mb": int(mem.total // (1024 * 1024)),
            "memory_free_mb": int(mem.available // (1024 * 1024)),
            "gpu_count": len(gpus),
            "gpus": [
                {
                    "index": g.index,
                    "name": g.name,
                    "vram_total_mb": g.vram_total_mb,
                    "vram_free_mb": g.vram_free_mb,
                }
                for g in gpus
            ],
        }

    # ------------------------------------------------------------------
    # 路径/可执行文件查找
    # ------------------------------------------------------------------

    def _find_llama_server(self) -> str | None:
        """查找 llama-server 可执行文件。"""
        result = find_executable("llama-server")
        return str(result) if result is not None else None

    def _find_default_model(self) -> str:
        """查找默认基准模型路径。

        优先级：
        1. ASC_BENCHMARK_MODEL 环境变量（完整文件路径）
        2. ASC_MODELS_PATH 目录下的候选模型文件
        3. 项目根目录下的 models/ 目录
        4. 当前工作目录下的 ./models 目录
        """
        # 环境变量指定完整路径
        env_model = os.getenv("ASC_BENCHMARK_MODEL")
        if env_model and Path(env_model).exists():
            return str(Path(env_model).resolve())

        # 候选 models 目录列表
        project_root = Path(__file__).resolve().parent.parent.parent  # asc/src/asc/worker -> asc/
        candidate_dirs = [
            Path(os.getenv("ASC_MODELS_PATH", "")) if os.getenv("ASC_MODELS_PATH") else None,
            project_root / "models",
            Path("./models"),
        ]

        for models_dir in candidate_dirs:
            if models_dir is None:
                continue
            for candidate in self._BENCHMARK_MODEL_CANDIDATES:
                path = models_dir / candidate
                if path.exists():
                    return str(path.resolve())

        # 未找到任何模型，返回项目根目录下第一个候选的预期路径（会触发 FileNotFoundError）
        fallback_dir = project_root / "models"
        return str((fallback_dir / self._BENCHMARK_MODEL_CANDIDATES[0]).resolve())

    def _infer_model_name(self) -> str:
        """根据模型文件路径推断模型名称。"""
        filename = Path(self.model_path).name
        return self._MODEL_NAME_MAP.get(filename, Path(self.model_path).stem)

    def _infer_quantization(self) -> str:
        """根据模型文件路径推断量化方式。"""
        filename = Path(self.model_path).name
        return self._MODEL_QUANT_MAP.get(filename, "unknown")

    def _default_questions_path(self) -> str:
        """默认测试问题文件路径。"""
        return str(Path(__file__).with_name("benchmark_questions.json").resolve())

    def _default_cache_path(self) -> str:
        """默认缓存文件路径。"""
        cache_dir = Path.home() / ".asc" / "benchmark_cache"
        return str(cache_dir / f"{self.node_id}.json")
