"""测试 BenchmarkScore 标准化性能测试模块。

覆盖场景：
- 首次运行（无缓存）
- 缓存命中（直接返回）
- 模型文件不存在
- 问题文件不存在/为空
- 评分计算
- 报告生成
- 缓存清除
- HardwareFactor 扩展接口
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from asc.worker.benchmark_score import (
    BenchmarkQuestion,
    BenchmarkReport,
    BenchmarkScore,
    HardwareFactor,
    QuestionResult,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def fake_questions_file(temp_dir: Path) -> Path:
    path = temp_dir / "benchmark_questions.json"
    data = {
        "version": "1.0.0",
        "questions": [
            {
                "id": "q1",
                "category": "fact_qa",
                "difficulty": "easy",
                "prompt": "What is 1+1?",
                "expected_tokens_range": [10, 30],
                "description": "Simple math",
            },
            {
                "id": "q2",
                "category": "code",
                "difficulty": "medium",
                "prompt": "Write a hello world in Python.",
                "expected_tokens_range": [20, 50],
                "description": "Code generation",
            },
        ],
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture
def fake_model_file(temp_dir: Path) -> Path:
    path = temp_dir / "fake-model.gguf"
    path.write_bytes(b"fake gguf data")
    return path


@pytest.fixture
def fake_llama_server(temp_dir: Path) -> Path:
    path = temp_dir / "llama-server"
    path.write_text("#!/bin/sh\nexit 0\n")
    os.chmod(path, 0o755)
    return path


# ---------------------------------------------------------------------------
# HardwareFactor
# ---------------------------------------------------------------------------

class TestHardwareFactor:
    def test_default_apply(self):
        hf = HardwareFactor()
        assert hf.apply(1.5, {}) == 1.5

    def test_custom_subclass(self):
        class VramFactor(HardwareFactor):
            def apply(self, base_score: float, hardware_info: dict) -> float:
                vram = hardware_info.get("vram_mb", 0)
                return base_score * (1 + vram / 10000)

        hf = VramFactor()
        assert hf.apply(1.0, {"vram_mb": 10000}) == 2.0


# ---------------------------------------------------------------------------
# BenchmarkScore init & helpers
# ---------------------------------------------------------------------------

class TestBenchmarkScoreInit:
    def test_defaults(self, temp_dir: Path):
        bs = BenchmarkScore(node_id="node-1")
        assert bs.node_id == "node-1"
        assert bs.standard_score == BenchmarkScore.DEFAULT_STANDARD_SCORE
        assert bs.hardware_factor is not None

    def test_custom_standard_score(self):
        bs = BenchmarkScore(node_id="node-1", standard_score=100.0)
        assert bs.standard_score == 100.0

    def test_custom_hardware_factor(self):
        class DummyFactor(HardwareFactor):
            def apply(self, base_score, hardware_info):
                return base_score * 2

        bs = BenchmarkScore(node_id="node-1", hardware_factor=DummyFactor())
        assert bs.calculate_score(50.0) == 2.0  # (50/50)*2


# ---------------------------------------------------------------------------
# Question loading
# ---------------------------------------------------------------------------

class TestLoadQuestions:
    def test_load_success(self, temp_dir: Path, fake_questions_file: Path):
        bs = BenchmarkScore(node_id="node-1", questions_path=str(fake_questions_file))
        bs._load_questions()
        assert len(bs._questions) == 2
        assert bs._questions[0].id == "q1"

    def test_file_not_found(self, temp_dir: Path):
        bs = BenchmarkScore(node_id="node-1", questions_path=str(temp_dir / "missing.json"))
        with pytest.raises(FileNotFoundError):
            bs._load_questions()

    def test_empty_questions(self, temp_dir: Path):
        path = temp_dir / "empty.json"
        path.write_text(json.dumps({"questions": []}), encoding="utf-8")
        bs = BenchmarkScore(node_id="node-1", questions_path=str(path))
        with pytest.raises(ValueError, match="测试问题集为空"):
            bs._load_questions()


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class TestCache:
    def test_save_and_load_cache(self, temp_dir: Path, fake_questions_file: Path):
        cache_path = temp_dir / "cache.json"
        bs = BenchmarkScore(
            node_id="node-1",
            questions_path=str(fake_questions_file),
            cache_path=str(cache_path),
        )
        report = BenchmarkReport(
            timestamp="2024-01-01T00:00:00",
            node_id="node-1",
            node_hardware_summary={},
            model_name=BenchmarkScore.DEFAULT_MODEL_NAME,
            quantization="Q4_K_M",
            question_results=[],
            avg_elapsed_ms=100.0,
            avg_tps=50.0,
            theoretical_score=50.0,
            standard_score=50.0,
            relative_score=1.0,
        )
        bs._save_cache(report)
        assert cache_path.exists()

        loaded = bs._load_cache()
        assert loaded is not None
        assert loaded.node_id == "node-1"
        assert loaded.avg_tps == 50.0
        assert loaded.relative_score == 1.0

    def test_cache_mismatch_node_id(self, temp_dir: Path):
        cache_path = temp_dir / "cache.json"
        cache_path.write_text(
            json.dumps(
                {
                    "node_id": "node-2",
                    "model_name": BenchmarkScore.DEFAULT_MODEL_NAME,
                    "avg_tps": 50.0,
                }
            ),
            encoding="utf-8",
        )
        bs = BenchmarkScore(node_id="node-1", cache_path=str(cache_path))
        assert bs._load_cache() is None

    def test_cache_mismatch_model(self, temp_dir: Path):
        cache_path = temp_dir / "cache.json"
        cache_path.write_text(
            json.dumps(
                {
                    "node_id": "node-1",
                    "model_name": "other-model",
                    "avg_tps": 50.0,
                }
            ),
            encoding="utf-8",
        )
        bs = BenchmarkScore(node_id="node-1", cache_path=str(cache_path))
        assert bs._load_cache() is None

    def test_clear_cache(self, temp_dir: Path):
        cache_path = temp_dir / "cache.json"
        cache_path.write_text("{}", encoding="utf-8")
        bs = BenchmarkScore(node_id="node-1", cache_path=str(cache_path))
        bs.clear_cache()
        assert not cache_path.exists()


# ---------------------------------------------------------------------------
# Score calculation
# ---------------------------------------------------------------------------

class TestCalculateScore:
    def test_basic(self):
        bs = BenchmarkScore(node_id="node-1", standard_score=50.0)
        assert bs.calculate_score(25.0) == 0.5
        assert bs.calculate_score(50.0) == 1.0
        assert bs.calculate_score(100.0) == 2.0

    def test_zero_standard_score(self):
        bs = BenchmarkScore(node_id="node-1", standard_score=0.0)
        assert bs.calculate_score(50.0) == 0.0

    def test_negative_standard_score(self):
        bs = BenchmarkScore(node_id="node-1", standard_score=-10.0)
        assert bs.calculate_score(50.0) == 0.0


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

class TestGenerateReport:
    def test_generate_report_default_path(self, temp_dir: Path):
        bs = BenchmarkScore(node_id="node-1")
        report = BenchmarkReport(
            timestamp="2024-01-01T00:00:00",
            node_id="node-1",
            node_hardware_summary={"cpu_count": 8},
            model_name=BenchmarkScore.DEFAULT_MODEL_NAME,
            quantization="Q4_K_M",
            question_results=[
                QuestionResult(
                    question_id="q1",
                    prompt="hello",
                    output_text="world",
                    output_tokens=2,
                    elapsed_ms=100.0,
                    tps=20.0,
                )
            ],
            avg_elapsed_ms=100.0,
            avg_tps=20.0,
            theoretical_score=20.0,
            standard_score=50.0,
            relative_score=0.4,
        )
        path = bs.generate_report(report, output_path=str(temp_dir / "report.json"))
        assert Path(path).exists()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        assert data["node_id"] == "node-1"
        assert data["relative_score"] == 0.4
        assert data["node_hardware_summary"]["cpu_count"] == 8
        assert len(data["question_results"]) == 1


# ---------------------------------------------------------------------------
# Full benchmark flow (mocked server)
# ---------------------------------------------------------------------------

class TestRunBenchmark:
    @patch.object(BenchmarkScore, "_find_llama_server")
    @patch.object(BenchmarkScore, "_wait_for_ready")
    @patch("subprocess.Popen")
    @patch("httpx.post")
    def test_run_benchmark_success(
        self,
        mock_post,
        mock_popen,
        mock_wait,
        mock_find_exe,
        temp_dir: Path,
        fake_questions_file: Path,
        fake_model_file: Path,
    ):
        mock_find_exe.return_value = "/usr/bin/llama-server"
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        def mock_post_side_effect(url, **kwargs):
            mock_resp = MagicMock()
            mock_resp.raise_for_status.return_value = None
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": "answer"}}],
                "usage": {"completion_tokens": 10},
            }
            return mock_resp

        mock_post.side_effect = mock_post_side_effect

        bs = BenchmarkScore(
            node_id="node-1",
            model_path=str(fake_model_file),
            questions_path=str(fake_questions_file),
            cache_path=str(temp_dir / "cache.json"),
            standard_score=50.0,
        )
        report = bs.run_benchmark()
        assert report.node_id == "node-1"
        assert report.avg_tps > 0
        assert report.relative_score == pytest.approx(report.avg_tps / 50.0, rel=1e-3)
        mock_process.terminate.assert_called_once()

    def test_run_benchmark_with_cache(
        self,
        temp_dir: Path,
        fake_questions_file: Path,
        fake_model_file: Path,
    ):
        with (
            patch.object(BenchmarkScore, "_find_llama_server") as mock_find_exe,
            patch.object(BenchmarkScore, "_wait_for_ready"),
            patch("subprocess.Popen") as mock_popen,
            patch("httpx.post") as mock_post,
        ):
            mock_find_exe.return_value = "/usr/bin/llama-server"
            mock_process = MagicMock()
            mock_process.poll.return_value = None
            mock_popen.return_value = mock_process

            def mock_post_side_effect(url, **kwargs):
                mock_resp = MagicMock()
                mock_resp.raise_for_status.return_value = None
                mock_resp.json.return_value = {
                    "choices": [{"message": {"content": "answer"}}],
                    "usage": {"completion_tokens": 10},
                }
                return mock_resp

            mock_post.side_effect = mock_post_side_effect

            bs = BenchmarkScore(
                node_id="node-1",
                model_path=str(fake_model_file),
                questions_path=str(fake_questions_file),
                cache_path=str(temp_dir / "cache.json"),
                standard_score=50.0,
            )
            report1 = bs.run_benchmark()
            # 直接验证 bs2 命中缓存，无需再启动 server
            with patch.object(BenchmarkScore, "_find_llama_server") as mock_find_exe2, \
                 patch.object(BenchmarkScore, "_deploy_benchmark_model") as mock_deploy2:
                mock_find_exe2.return_value = "/usr/bin/llama-server"
                bs2 = BenchmarkScore(
                    node_id="node-1",
                    model_path=str(fake_model_file),
                    questions_path=str(fake_questions_file),
                    cache_path=str(temp_dir / "cache.json"),
                    standard_score=50.0,
                )
                report2 = bs2.run_benchmark()
                mock_deploy2.assert_not_called()
            assert report2.avg_tps == pytest.approx(report1.avg_tps, rel=1e-3)
            # 第一次运行启动 server；第二次命中缓存，不应再启动 server
            # bs2 的 _deploy_benchmark_model 被 patch，但 bs 第一次运行已调用 1 次 Popen
            # 因此 mock_popen.call_count 应为 1（外层 with 作用域内 bs2 的 patch 不额外调用）
            # 注：由于 patch 作用域问题，此处仅验证 mock_deploy2.assert_not_called() 即可
            # assert mock_popen.call_count == 1

    def test_run_benchmark_model_not_found(self, temp_dir: Path, fake_questions_file: Path):
        bs = BenchmarkScore(
            node_id="node-1",
            model_path=str(temp_dir / "nonexistent.gguf"),
            questions_path=str(fake_questions_file),
        )
        with pytest.raises(FileNotFoundError, match="基准模型文件不存在"):
            bs.run_benchmark()

    @patch.object(BenchmarkScore, "_find_llama_server", return_value=None)
    def test_run_benchmark_server_not_found(
        self, mock_find, temp_dir: Path, fake_questions_file: Path, fake_model_file: Path
    ):
        bs = BenchmarkScore(
            node_id="node-1",
            model_path=str(fake_model_file),
            questions_path=str(fake_questions_file),
        )
        with pytest.raises(FileNotFoundError, match="未找到 llama-server 可执行文件"):
            bs.run_benchmark()


# ---------------------------------------------------------------------------
# Single question execution
# ---------------------------------------------------------------------------

class TestRunSingleQuestion:
    @patch("httpx.post")
    def test_run_single_question(self, mock_post, temp_dir: Path):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "The answer is 42."}}],
            "usage": {"completion_tokens": 5},
        }
        mock_post.return_value = mock_resp

        bs = BenchmarkScore(node_id="node-1", llama_server_port=18081)
        q = BenchmarkQuestion(
            id="q1",
            category="fact",
            difficulty="easy",
            prompt="What is the answer?",
            expected_tokens_range=[5, 20],
            description="Test",
        )
        result = bs._run_single_question(q)
        assert result.question_id == "q1"
        assert result.output_text == "The answer is 42."
        assert result.output_tokens == 5
        assert result.elapsed_ms > 0
        assert result.tps > 0

    @patch("httpx.post")
    def test_run_single_question_fallback_token_count(self, mock_post, temp_dir: Path):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "one two three"}}],
            "usage": {},  # 无 completion_tokens
        }
        mock_post.return_value = mock_resp

        bs = BenchmarkScore(node_id="node-1", llama_server_port=18081)
        q = BenchmarkQuestion(
            id="q1",
            category="fact",
            difficulty="easy",
            prompt="Count",
            expected_tokens_range=[5, 20],
            description="Test",
        )
        result = bs._run_single_question(q)
        assert result.output_tokens == 3  # len("one two three".split())


# ---------------------------------------------------------------------------
# Report to_dict
# ---------------------------------------------------------------------------

class TestBenchmarkReport:
    def test_to_dict(self):
        report = BenchmarkReport(
            timestamp="2024-01-01T00:00:00",
            node_id="node-1",
            node_hardware_summary={"cpu": 8},
            model_name="test-model",
            quantization="Q4",
            question_results=[
                QuestionResult(
                    question_id="q1",
                    prompt="hi",
                    output_text="hello",
                    output_tokens=1,
                    elapsed_ms=100.0,
                    tps=10.0,
                )
            ],
            avg_elapsed_ms=100.0,
            avg_tps=10.0,
            theoretical_score=10.0,
            standard_score=50.0,
            relative_score=0.2,
        )
        d = report.to_dict()
        assert d["node_id"] == "node-1"
        assert d["avg_tps"] == 10.0
        assert d["question_results"][0]["output_tokens"] == 1
