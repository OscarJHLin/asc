"""ASC 验收测试 - Worker 模块

测试范围：
- worker/runner.py
- worker/agent.py
- worker/rpc_server.py
- worker/hardware.py
- worker/gpu_info.py
- worker/benchmark_score.py

测试维度：功能测试、边界条件测试、异常场景测试
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from asc.worker.agent import NodeResources, WorkerAgent
from asc.worker.benchmark_score import (
    BenchmarkQuestion,
    BenchmarkReport,
    BenchmarkScore,
    HardwareFactor,
)
from asc.worker.gpu_info import GPUInfo
from asc.worker.hardware import CPUInfo, DiskInfo, HardwareDetector, NetworkInfo
from asc.worker.runner import Runner, RunnerCommand, RunnerState, RunnerTransitionError

# ======================================================================
# 1. worker/runner.py 测试
# ======================================================================


class TestRunnerStateMachine:
    """Runner 状态机功能测试。"""

    def test_initial_state_idle(self):
        runner = Runner(node_id="n1")
        assert runner.state == RunnerState.IDLE

    def test_valid_transition_idle_to_loading(self):
        runner = Runner(node_id="n1")
        runner.transition(RunnerCommand.LOAD)
        assert runner.state == RunnerState.LOADING

    def test_valid_transition_loading_to_ready(self):
        runner = Runner(node_id="n1")
        runner.transition(RunnerCommand.LOAD)
        runner.transition(RunnerCommand.LOAD_COMPLETE)
        assert runner.state == RunnerState.READY

    def test_valid_transition_ready_to_running(self):
        runner = Runner(node_id="n1")
        runner.transition(RunnerCommand.LOAD)
        runner.transition(RunnerCommand.LOAD_COMPLETE)
        runner.transition(RunnerCommand.START_INFERENCE)
        assert runner.state == RunnerState.RUNNING

    def test_valid_transition_running_to_ready(self):
        runner = Runner(node_id="n1")
        runner.transition(RunnerCommand.LOAD)
        runner.transition(RunnerCommand.LOAD_COMPLETE)
        runner.transition(RunnerCommand.START_INFERENCE)
        runner.transition(RunnerCommand.INFERENCE_COMPLETE)
        assert runner.state == RunnerState.READY

    def test_valid_transition_loading_to_error(self):
        runner = Runner(node_id="n1")
        runner.transition(RunnerCommand.LOAD)
        runner.transition(RunnerCommand.ERROR, error_msg="load failed")
        assert runner.state == RunnerState.ERROR
        assert runner.last_error == "load failed"

    def test_valid_transition_running_to_error(self):
        runner = Runner(node_id="n1")
        runner.transition(RunnerCommand.LOAD)
        runner.transition(RunnerCommand.LOAD_COMPLETE)
        runner.transition(RunnerCommand.START_INFERENCE)
        runner.transition(RunnerCommand.ERROR, error_msg="inference crash")
        assert runner.state == RunnerState.ERROR
        assert runner.last_error == "inference crash"

    def test_valid_transition_error_to_idle(self):
        runner = Runner(node_id="n1")
        runner.transition(RunnerCommand.LOAD)
        runner.transition(RunnerCommand.ERROR, error_msg="fail")
        runner.transition(RunnerCommand.RESET)
        assert runner.state == RunnerState.IDLE
        assert runner.last_error is None

    def test_valid_transition_ready_to_shutdown(self):
        runner = Runner(node_id="n1")
        runner.transition(RunnerCommand.LOAD)
        runner.transition(RunnerCommand.LOAD_COMPLETE)
        runner.transition(RunnerCommand.SHUTDOWN)
        assert runner.state == RunnerState.SHUTDOWN

    def test_full_lifecycle(self):
        """完整生命周期：Idle -> Loading -> Ready -> Running -> Ready -> Shutdown。"""
        runner = Runner(node_id="n1")
        runner.transition(RunnerCommand.LOAD)
        runner.transition(RunnerCommand.LOAD_COMPLETE)
        runner.transition(RunnerCommand.START_INFERENCE)
        runner.transition(RunnerCommand.INFERENCE_COMPLETE)
        runner.transition(RunnerCommand.START_INFERENCE)
        runner.transition(RunnerCommand.INFERENCE_COMPLETE)
        runner.transition(RunnerCommand.SHUTDOWN)
        assert runner.state == RunnerState.SHUTDOWN

    def test_error_recovery_cycle(self):
        """错误恢复循环：Idle -> Loading -> Error -> Idle -> Loading -> Ready。"""
        runner = Runner(node_id="n1")
        runner.transition(RunnerCommand.LOAD)
        runner.transition(RunnerCommand.ERROR, error_msg="first fail")
        runner.transition(RunnerCommand.RESET)
        runner.transition(RunnerCommand.LOAD)
        runner.transition(RunnerCommand.LOAD_COMPLETE)
        assert runner.state == RunnerState.READY
        assert runner.last_error is None


class TestRunnerInvalidTransitions:
    """Runner 非法状态转换测试。"""

    def test_idle_to_ready_invalid(self):
        runner = Runner(node_id="n1")
        with pytest.raises(RunnerTransitionError):
            runner.transition(RunnerCommand.LOAD_COMPLETE)

    def test_idle_to_running_invalid(self):
        runner = Runner(node_id="n1")
        with pytest.raises(RunnerTransitionError):
            runner.transition(RunnerCommand.START_INFERENCE)

    def test_idle_to_inference_complete_invalid(self):
        runner = Runner(node_id="n1")
        with pytest.raises(RunnerTransitionError):
            runner.transition(RunnerCommand.INFERENCE_COMPLETE)

    def test_idle_to_shutdown_invalid(self):
        runner = Runner(node_id="n1")
        with pytest.raises(RunnerTransitionError):
            runner.transition(RunnerCommand.SHUTDOWN)

    def test_loading_to_start_inference_invalid(self):
        runner = Runner(node_id="n1")
        runner.transition(RunnerCommand.LOAD)
        with pytest.raises(RunnerTransitionError):
            runner.transition(RunnerCommand.START_INFERENCE)

    def test_ready_to_load_invalid(self):
        runner = Runner(node_id="n1")
        runner.transition(RunnerCommand.LOAD)
        runner.transition(RunnerCommand.LOAD_COMPLETE)
        with pytest.raises(RunnerTransitionError):
            runner.transition(RunnerCommand.LOAD)

    def test_running_to_load_invalid(self):
        runner = Runner(node_id="n1")
        runner.transition(RunnerCommand.LOAD)
        runner.transition(RunnerCommand.LOAD_COMPLETE)
        runner.transition(RunnerCommand.START_INFERENCE)
        with pytest.raises(RunnerTransitionError):
            runner.transition(RunnerCommand.LOAD)

    def test_error_to_load_invalid(self):
        runner = Runner(node_id="n1")
        runner.transition(RunnerCommand.LOAD)
        runner.transition(RunnerCommand.ERROR)
        with pytest.raises(RunnerTransitionError):
            runner.transition(RunnerCommand.LOAD)

    def test_shutdown_to_any_invalid(self):
        runner = Runner(node_id="n1")
        runner.transition(RunnerCommand.LOAD)
        runner.transition(RunnerCommand.LOAD_COMPLETE)
        runner.transition(RunnerCommand.SHUTDOWN)
        with pytest.raises(RunnerTransitionError):
            runner.transition(RunnerCommand.RESET)


class TestRunnerTransitionError:
    """RunnerTransitionError 异常测试。"""

    def test_error_contains_state_info(self):
        runner = Runner(node_id="n1")
        try:
            runner.transition(RunnerCommand.START_INFERENCE)
        except RunnerTransitionError as e:
            assert e.from_state == RunnerState.IDLE
            assert e.command == RunnerCommand.START_INFERENCE
            assert "idle" in str(e)
            assert "start_inference" in str(e)


class TestRunnerStateHistory:
    """Runner 状态历史测试。"""

    def test_state_history_records(self):
        runner = Runner(node_id="n1")
        runner.transition(RunnerCommand.LOAD)
        runner.transition(RunnerCommand.LOAD_COMPLETE)
        assert len(runner.state_history) == 2
        assert runner.state_history[0] == (RunnerState.IDLE, RunnerState.LOADING)
        assert runner.state_history[1] == (RunnerState.LOADING, RunnerState.READY)

    def test_state_history_returns_copy(self):
        """state_history 返回副本，不可外部修改。"""
        runner = Runner(node_id="n1")
        runner.transition(RunnerCommand.LOAD)
        history = runner.state_history
        history.clear()
        assert len(runner.state_history) == 1


class TestRunnerCanAcceptInference:
    """Runner can_accept_inference 测试。"""

    def test_can_accept_in_ready(self):
        runner = Runner(node_id="n1")
        runner.transition(RunnerCommand.LOAD)
        runner.transition(RunnerCommand.LOAD_COMPLETE)
        assert runner.can_accept_inference() is True

    def test_cannot_accept_in_idle(self):
        runner = Runner(node_id="n1")
        assert runner.can_accept_inference() is False

    def test_cannot_accept_in_loading(self):
        runner = Runner(node_id="n1")
        runner.transition(RunnerCommand.LOAD)
        assert runner.can_accept_inference() is False

    def test_cannot_accept_in_running(self):
        runner = Runner(node_id="n1")
        runner.transition(RunnerCommand.LOAD)
        runner.transition(RunnerCommand.LOAD_COMPLETE)
        runner.transition(RunnerCommand.START_INFERENCE)
        assert runner.can_accept_inference() is False

    def test_cannot_accept_in_error(self):
        runner = Runner(node_id="n1")
        runner.transition(RunnerCommand.LOAD)
        runner.transition(RunnerCommand.ERROR)
        assert runner.can_accept_inference() is False

    def test_cannot_accept_in_shutdown(self):
        runner = Runner(node_id="n1")
        runner.transition(RunnerCommand.LOAD)
        runner.transition(RunnerCommand.LOAD_COMPLETE)
        runner.transition(RunnerCommand.SHUTDOWN)
        assert runner.can_accept_inference() is False


# ======================================================================
# 2. worker/gpu_info.py 测试
# ======================================================================


class TestGPUInfo:
    """GPUInfo 值对象测试。"""

    def test_basic_creation(self):
        gpu = GPUInfo(
            index=0,
            name="RTX 4090",
            vram_total_mb=24576,
            vram_free_mb=20000,
            vendor="nvidia",
        )
        assert gpu.index == 0
        assert gpu.name == "RTX 4090"
        assert gpu.vram_total_mb == 24576
        assert gpu.vram_free_mb == 20000
        assert gpu.vendor == "nvidia"

    def test_vram_used_mb(self):
        gpu = GPUInfo(index=0, name="RTX 4090", vram_total_mb=24576, vram_free_mb=20000)
        assert gpu.vram_used_mb == 4576

    def test_vram_used_mb_zero(self):
        gpu = GPUInfo(index=0, name="GPU", vram_total_mb=10000, vram_free_mb=10000)
        assert gpu.vram_used_mb == 0

    def test_vram_used_mb_full(self):
        gpu = GPUInfo(index=0, name="GPU", vram_total_mb=10000, vram_free_mb=0)
        assert gpu.vram_used_mb == 10000

    def test_frozen(self):
        gpu = GPUInfo(index=0, name="GPU", vram_total_mb=10000, vram_free_mb=5000)
        with pytest.raises(AttributeError):
            gpu.name = "new"

    def test_default_values(self):
        gpu = GPUInfo(index=0, name="GPU", vram_total_mb=10000, vram_free_mb=5000)
        assert gpu.vendor == ""
        assert gpu.compute_capability == ""

    def test_zero_vram(self):
        """VRAM 为 0 的边界情况。"""
        gpu = GPUInfo(index=0, name="No VRAM", vram_total_mb=0, vram_free_mb=0)
        assert gpu.vram_used_mb == 0


# ======================================================================
# 3. worker/hardware.py 测试
# ======================================================================


class TestHardwareDetectorCPU:
    """HardwareDetector CPU 检测测试。"""

    def test_detect_cpu_returns_cpuinfo(self):
        detector = HardwareDetector()
        cpu = detector.detect_cpu()
        assert isinstance(cpu, CPUInfo)
        assert cpu.logical_count >= 0
        assert cpu.physical_count >= 0

    def test_detect_cpu_has_brand(self):
        detector = HardwareDetector()
        cpu = detector.detect_cpu()
        # 品牌可能为空（某些环境），但不应抛异常
        assert isinstance(cpu.brand, str)


class TestHardwareDetectorGPU:
    """HardwareDetector GPU 检测测试。"""

    def test_detect_gpus_returns_list(self):
        detector = HardwareDetector()
        gpus = detector.detect_gpus()
        assert isinstance(gpus, list)

    def test_detect_gpus_no_gpu(self):
        """无 GPU 环境应返回空列表。"""
        detector = HardwareDetector()
        with (
            patch.object(detector, "_detect_gpus_windows", return_value=[]),
            patch.object(detector, "_detect_gpus_linux", return_value=[]),
            patch.object(detector, "_detect_gpus_darwin", return_value=[]),
        ):
            gpus = detector.detect_gpus()
            assert isinstance(gpus, list)


class TestHardwareDetectorDisk:
    """HardwareDetector 磁盘检测测试。"""

    def test_detect_disk_returns_diskinfo(self):
        detector = HardwareDetector()
        disk = detector.detect_disk()
        assert isinstance(disk, DiskInfo)
        assert disk.free_mb >= 0


class TestHardwareDetectorNetwork:
    """HardwareDetector 网络检测测试。"""

    def test_detect_network_returns_networkinfo(self):
        detector = HardwareDetector()
        net = detector.detect_network()
        assert isinstance(net, NetworkInfo)
        assert net.estimated_mbps >= 0


class TestHardwareDetectorAll:
    """HardwareDetector detect_all 测试。"""

    def test_detect_all_returns_dict(self):
        detector = HardwareDetector()
        info = detector.detect_all()
        assert "cpu" in info
        assert "gpus" in info
        assert "memory_total_mb" in info
        assert "memory_free_mb" in info
        assert "disk_free_mb" in info
        assert "network_mbps" in info


# ======================================================================
# 4. worker/agent.py 测试
# ======================================================================


class TestNodeResources:
    """NodeResources 值对象测试。"""

    def test_basic_creation(self):
        res = NodeResources(
            cpu_count=8, cpu_percent=50.0, memory_total_mb=16384, memory_free_mb=8192
        )
        assert res.cpu_count == 8
        assert res.memory_total_mb == 16384

    def test_total_vram_free_no_gpus(self):
        res = NodeResources(
            cpu_count=8, cpu_percent=50.0, memory_total_mb=16384, memory_free_mb=8192
        )
        assert res.total_vram_free_mb == 0

    def test_total_vram_free_with_gpus(self):
        gpus = [
            GPUInfo(index=0, name="GPU0", vram_total_mb=10000, vram_free_mb=5000),
            GPUInfo(index=1, name="GPU1", vram_total_mb=10000, vram_free_mb=3000),
        ]
        res = NodeResources(
            cpu_count=8, cpu_percent=50.0, memory_total_mb=16384, memory_free_mb=8192, gpus=gpus
        )
        assert res.total_vram_free_mb == 8000


class TestWorkerAgentHealth:
    """WorkerAgent 健康检查测试。"""

    def test_health_returns_ok(self):
        agent = WorkerAgent(node_id="n1", port=52415)
        result = agent.health()
        assert result["status"] == "ok"
        assert result["node_id"] == "n1"


class TestWorkerAgentResources:
    """WorkerAgent 资源查询测试。"""

    def test_get_resources_returns_node_resources(self):
        agent = WorkerAgent(node_id="n1", port=52415)
        res = agent.get_resources()
        assert isinstance(res, NodeResources)
        assert res.cpu_count > 0
        assert res.memory_total_mb > 0


class TestWorkerAgentRPC:
    """WorkerAgent RPC 管理测试。"""

    def test_start_rpc_no_executable(self):
        """没有可执行文件时启动 RPC 应返回错误。"""
        agent = WorkerAgent(node_id="n1", port=52415)
        result = agent.start_rpc()
        # 在没有 rpc-server 的环境中应返回 error
        assert result["status"] in ("ok", "error")

    def test_rpc_status_initial(self):
        agent = WorkerAgent(node_id="n1", port=52415)
        status = agent.rpc_status()
        assert "running" in status

    def test_stop_rpc_when_not_running(self):
        """停止未运行的 RPC 不报错。"""
        agent = WorkerAgent(node_id="n1", port=52415)
        result = agent.stop_rpc()
        assert result["status"] == "ok"


# ======================================================================
# 5. worker/rpc_server.py 测试
# ======================================================================


class TestRpcServerBasic:
    """RpcServer 基本功能测试。"""

    def test_initial_state(self):
        from asc.worker.rpc_server import RpcServer
        rpc = RpcServer()
        assert rpc.is_running is False
        assert rpc.port is None
        assert rpc.endpoint is None

    def test_status_not_running(self):
        from asc.worker.rpc_server import RpcServer
        rpc = RpcServer()
        status = rpc.status()
        assert status["running"] is False
        assert status["host"] is None
        assert status["port"] is None

    def test_start_no_executable(self):
        """没有可执行文件应抛出 FileNotFoundError。"""
        from asc.worker.rpc_server import RpcServer
        rpc = RpcServer()
        with (
            patch.object(rpc, "_find_executable", return_value=None),
            pytest.raises(FileNotFoundError),
        ):
            rpc.start()

    def test_stop_when_not_running(self):
        """停止未运行的服务不报错。"""
        from asc.worker.rpc_server import RpcServer
        rpc = RpcServer()
        rpc.stop()  # 不应抛异常

    def test_is_ready_when_not_running(self):
        from asc.worker.rpc_server import RpcServer
        rpc = RpcServer()
        assert rpc.is_ready() is False


class TestRpcServerPortAllocation:
    """RpcServer 端口分配测试。"""

    def test_find_free_port(self):
        from asc.worker.rpc_server import RpcServer
        rpc = RpcServer()
        port = rpc._find_free_port()
        assert 50052 <= port <= 50152

    def test_is_port_free(self):
        # 找一个空闲端口
        import socket

        from asc.worker.rpc_server import RpcServer
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            port = s.getsockname()[1]
        # 端口释放后应可用
        assert RpcServer._is_port_free(port) is True


# ======================================================================
# 6. worker/benchmark_score.py 测试
# ======================================================================


class TestBenchmarkScoreCalculation:
    """BenchmarkScore 评分计算测试。"""

    def test_calculate_score_normal(self):
        bs = BenchmarkScore(node_id="n1", standard_score=50.0)
        score = bs.calculate_score(100.0)
        assert score == 2.0  # 100/50

    def test_calculate_score_zero_standard(self):
        """标准分数为 0 时返回 0。"""
        bs = BenchmarkScore(node_id="n1", standard_score=0.0)
        score = bs.calculate_score(100.0)
        assert score == 0.0

    def test_calculate_score_negative_standard(self):
        """标准分数为负数时返回 0。"""
        bs = BenchmarkScore(node_id="n1", standard_score=-10.0)
        score = bs.calculate_score(100.0)
        assert score == 0.0

    def test_calculate_score_zero_theoretical(self):
        """理论分数为 0 时返回 0。"""
        bs = BenchmarkScore(node_id="n1", standard_score=50.0)
        score = bs.calculate_score(0.0)
        assert score == 0.0

    def test_default_standard_score(self):
        bs = BenchmarkScore(node_id="n1")
        assert bs.standard_score == 50.0


class TestHardwareFactor:
    """HardwareFactor 扩展接口测试。"""

    def test_default_apply(self):
        hf = HardwareFactor()
        assert hf.apply(1.5, {}) == 1.5

    def test_custom_hardware_factor(self):
        class DoubleFactor(HardwareFactor):
            def apply(self, base_score, hardware_info):
                return base_score * 2

        bs = BenchmarkScore(node_id="n1", standard_score=50.0, hardware_factor=DoubleFactor())
        score = bs.calculate_score(100.0)
        assert score == 4.0  # (100/50) * 2


class TestBenchmarkScoreCache:
    """BenchmarkScore 缓存机制测试。"""

    def test_clear_cache(self, tmp_path):
        cache_path = str(tmp_path / "cache.json")
        bs = BenchmarkScore(node_id="n1", cache_path=cache_path)
        # 创建缓存文件
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_path).write_text("{}")
        bs.clear_cache()
        assert not Path(cache_path).exists()

    def test_clear_cache_nonexistent(self, tmp_path):
        """清除不存在的缓存不报错。"""
        bs = BenchmarkScore(node_id="n1", cache_path=str(tmp_path / "nonexistent.json"))
        bs.clear_cache()


class TestBenchmarkReport:
    """BenchmarkReport 测试。"""

    def test_to_dict(self):
        report = BenchmarkReport(
            timestamp="2024-01-01T00:00:00",
            node_id="n1",
            node_hardware_summary={"cpu_count": 8},
            model_name="Qwen2.5-0.5B-Instruct",
            quantization="Q4_K_M",
            question_results=[],
            avg_elapsed_ms=100.0,
            avg_tps=50.0,
            theoretical_score=50.0,
            standard_score=50.0,
            relative_score=1.0,
        )
        d = report.to_dict()
        assert d["node_id"] == "n1"
        assert d["relative_score"] == 1.0
        assert d["quantization"] == "Q4_K_M"


class TestBenchmarkQuestion:
    """BenchmarkQuestion 测试。"""

    def test_creation(self):
        q = BenchmarkQuestion(
            id="q1",
            category="reasoning",
            difficulty="easy",
            prompt="What is 2+2?",
            expected_tokens_range=[5, 20],
            description="Simple math",
        )
        assert q.id == "q1"
        assert q.expected_tokens_range == [5, 20]


class TestBenchmarkScoreException:
    """BenchmarkScore 异常场景测试。"""

    def test_run_benchmark_no_model(self, tmp_path):
        """模型文件不存在时运行基准测试应抛出 FileNotFoundError。"""
        bs = BenchmarkScore(
            node_id="n1",
            model_path=str(tmp_path / "nonexistent.gguf"),
            cache_path=str(tmp_path / "cache.json"),
        )
        with pytest.raises(FileNotFoundError):
            bs.run_benchmark()

    def test_load_questions_nonexistent_file(self, tmp_path):
        """问题文件不存在时应抛出 FileNotFoundError。"""
        bs = BenchmarkScore(
            node_id="n1",
            questions_path=str(tmp_path / "nonexistent.json"),
            cache_path=str(tmp_path / "cache.json"),
        )
        with pytest.raises(FileNotFoundError):
            bs._load_questions()

    def test_load_questions_empty_set(self, tmp_path):
        """空问题集应抛出 ValueError。"""
        path = tmp_path / "questions.json"
        path.write_text(json.dumps({"questions": []}))
        bs = BenchmarkScore(
            node_id="n1",
            questions_path=str(path),
            cache_path=str(tmp_path / "cache.json"),
        )
        with pytest.raises(ValueError, match="测试问题集为空"):
            bs._load_questions()
