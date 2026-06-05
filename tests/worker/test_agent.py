"""测试 Worker Agent。

Worker Agent 运行在 Worker 节点上，提供：
- 健康检查
- 资源查询（CPU/GPU/内存）
- RPC Server 生命周期管理
- 模型加载/卸载
"""

from unittest.mock import MagicMock, patch

from asc.worker.agent import (
    NodeResources,
    RPCServerManager,
    WorkerAgent,
)
from asc.worker.benchmark_score import BenchmarkReport, BenchmarkScore
from asc.worker.gpu_info import GPUInfo
from asc.worker.hardware import CPUInfo, DiskInfo, HardwareDetector, NetworkInfo
from asc.worker.rpc_server import RpcServer


class TestGPUInfo:
    """GPU 信息值对象。"""

    def test_create(self):
        gpu = GPUInfo(index=0, name="RTX 4090", vram_total_mb=24564, vram_free_mb=20000)
        assert gpu.index == 0
        assert gpu.name == "RTX 4090"
        assert gpu.vram_total_mb == 24564
        assert gpu.vram_free_mb == 20000
        assert gpu.vendor == ""
        assert gpu.compute_capability == ""

    def test_create_with_vendor(self):
        gpu = GPUInfo(
            index=0,
            name="RTX 4090",
            vram_total_mb=24564,
            vram_free_mb=20000,
            vendor="nvidia",
            compute_capability="8.9",
        )
        assert gpu.vendor == "nvidia"
        assert gpu.compute_capability == "8.9"

    def test_vram_used(self):
        gpu = GPUInfo(index=0, name="RTX 4090", vram_total_mb=24564, vram_free_mb=20000)
        assert gpu.vram_used_mb == 4564

    def test_frozen(self):
        gpu = GPUInfo(index=0, name="RTX 4090", vram_total_mb=24564, vram_free_mb=20000)
        try:
            gpu.name = "other"  # type: ignore[misc]
            raise AssertionError("Should be immutable")
        except (AttributeError, TypeError):
            pass


class TestNodeResources:
    """节点资源信息。"""

    def test_create(self):
        res = NodeResources(
            cpu_count=8,
            cpu_percent=25.0,
            memory_total_mb=32768,
            memory_free_mb=24000,
            gpus=[
                GPUInfo(index=0, name="RTX 4090", vram_total_mb=24564, vram_free_mb=20000),
            ],
        )
        assert res.cpu_count == 8
        assert len(res.gpus) == 1
        assert res.compute_score == 0.0
        assert res.cpu_physical_count == 0
        assert res.cpu_freq_mhz == 0.0
        assert res.cpu_brand == ""
        assert res.disk_free_mb == 0
        assert res.network_mbps == 0.0

    def test_create_with_extended_fields(self):
        res = NodeResources(
            cpu_count=8,
            cpu_percent=25.0,
            memory_total_mb=32768,
            memory_free_mb=24000,
            gpus=[],
            compute_score=1.5,
            cpu_physical_count=4,
            cpu_freq_mhz=3200.0,
            cpu_brand="Intel(R) Core(TM) i7",
            disk_free_mb=102400,
            network_mbps=1000.0,
        )
        assert res.compute_score == 1.5
        assert res.cpu_physical_count == 4
        assert res.cpu_freq_mhz == 3200.0
        assert res.cpu_brand == "Intel(R) Core(TM) i7"
        assert res.disk_free_mb == 102400
        assert res.network_mbps == 1000.0

    def test_total_vram(self):
        res = NodeResources(
            cpu_count=8,
            cpu_percent=25.0,
            memory_total_mb=32768,
            memory_free_mb=24000,
            gpus=[
                GPUInfo(index=0, name="RTX 4090", vram_total_mb=24564, vram_free_mb=20000),
                GPUInfo(index=1, name="RTX 4090", vram_total_mb=24564, vram_free_mb=15000),
            ],
        )
        assert res.total_vram_free_mb == 35000

    def test_no_gpus(self):
        res = NodeResources(
            cpu_count=4,
            cpu_percent=10.0,
            memory_total_mb=16384,
            memory_free_mb=12000,
            gpus=[],
        )
        assert res.total_vram_free_mb == 0


class TestRPCServerManager:
    """RPC Server 生命周期管理。"""

    def test_initial_state(self):
        mgr = RPCServerManager()
        assert not mgr.is_running
        assert mgr.port is None

    @patch.object(RPCServerManager, "_find_executable", return_value="/usr/bin/rpc-server")
    @patch("subprocess.Popen")
    def test_start(self, mock_popen, mock_find):
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        mgr = RPCServerManager()
        mgr.start(host="0.0.0.0", port=50052)
        assert mgr.is_running
        assert mgr.port == 50052

    @patch.object(RPCServerManager, "_find_executable", return_value="/usr/bin/rpc-server")
    @patch("subprocess.Popen")
    def test_stop(self, mock_popen, mock_find):
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        mgr = RPCServerManager()
        mgr.start(host="0.0.0.0", port=50052)
        mgr.stop()
        assert not mgr.is_running
        mock_process.terminate.assert_called_once()

    def test_stop_when_not_running(self):
        mgr = RPCServerManager()
        mgr.stop()  # 不应抛异常

    @patch.object(RPCServerManager, "_find_executable", return_value="/usr/bin/rpc-server")
    @patch("subprocess.Popen")
    def test_status(self, mock_popen, mock_find):
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        mgr = RPCServerManager()
        mgr.start(host="0.0.0.0", port=50052)
        status = mgr.status()
        assert status["running"] is True
        assert status["port"] == 50052


class TestWorkerAgent:
    """Worker Agent 主类。"""

    def test_create(self):
        agent = WorkerAgent(node_id="node-1", port=52415)
        assert agent.node_id == "node-1"
        assert agent.port == 52415

    @patch("psutil.cpu_count", return_value=8)
    @patch("psutil.cpu_percent", return_value=25.0)
    @patch("psutil.virtual_memory")
    @patch.object(HardwareDetector, "detect_cpu")
    @patch.object(HardwareDetector, "detect_gpus", return_value=[])
    @patch.object(HardwareDetector, "detect_disk")
    @patch.object(HardwareDetector, "detect_network")
    @patch.object(BenchmarkScore, "run_benchmark")
    def test_get_resources(
        self,
        mock_benchmark,
        mock_net,
        mock_disk,
        mock_gpus,
        mock_cpu,
        mock_vm,
        mock_cpu_pct,
        mock_cpu_cnt,
    ):
        mock_vm.return_value = MagicMock(total=32 * 1024**3, available=24 * 1024**3)
        mock_cpu.return_value = CPUInfo(
            logical_count=8, physical_count=4, freq_mhz=3200.0, brand="Intel i7"
        )
        mock_disk.return_value = DiskInfo(free_mb=102400)
        mock_net.return_value = NetworkInfo(estimated_mbps=1000.0)
        mock_benchmark.return_value = BenchmarkReport(
            timestamp="2024-01-01T00:00:00",
            node_id="node-1",
            node_hardware_summary={},
            model_name="Qwen2.5-1.5B-Instruct",
            quantization="Q4_K_M",
            question_results=[],
            avg_elapsed_ms=100.0,
            avg_tps=75.0,
            theoretical_score=75.0,
            standard_score=50.0,
            relative_score=1.5,
        )
        agent = WorkerAgent(node_id="node-1", port=52415)
        res = agent.get_resources()
        assert res.cpu_count == 8
        assert res.cpu_percent == 25.0
        assert res.memory_total_mb == 32 * 1024
        assert res.gpus == []
        assert res.compute_score == 1.5
        assert res.cpu_physical_count == 4
        assert res.cpu_freq_mhz == 3200.0
        assert res.cpu_brand == "Intel i7"
        assert res.disk_free_mb == 102400
        assert res.network_mbps == 1000.0

    @patch("psutil.cpu_count", return_value=8)
    @patch("psutil.cpu_percent", return_value=25.0)
    @patch("psutil.virtual_memory")
    @patch.object(HardwareDetector, "detect_cpu")
    @patch.object(HardwareDetector, "detect_gpus", return_value=[])
    @patch.object(HardwareDetector, "detect_disk")
    @patch.object(HardwareDetector, "detect_network")
    @patch.object(BenchmarkScore, "run_benchmark")
    def test_get_resources_benchmark_fallback(
        self,
        mock_benchmark,
        mock_net,
        mock_disk,
        mock_gpus,
        mock_cpu,
        mock_vm,
        mock_cpu_pct,
        mock_cpu_cnt,
    ):
        mock_vm.return_value = MagicMock(total=32 * 1024**3, available=24 * 1024**3)
        mock_cpu.return_value = CPUInfo(
            logical_count=8, physical_count=4, freq_mhz=3200.0, brand="Intel i7"
        )
        mock_disk.return_value = DiskInfo(free_mb=102400)
        mock_net.return_value = NetworkInfo(estimated_mbps=1000.0)
        mock_benchmark.side_effect = FileNotFoundError("model not found")
        agent = WorkerAgent(node_id="node-1", port=52415)
        res = agent.get_resources()
        assert res.compute_score == 0.0
        assert res.cpu_brand == "Intel i7"

    def test_health_check(self):
        agent = WorkerAgent(node_id="node-1", port=52415)
        health = agent.health()
        assert health["status"] == "ok"
        assert health["node_id"] == "node-1"

    @patch.object(RpcServer, "start", return_value=50055)
    def test_start_rpc(self, mock_start):
        agent = WorkerAgent(node_id="node-1", port=52415)
        # 手动设置 endpoint，因为 mock 的 start 不会设置 _port
        agent._rpc_server._port = 50055
        result = agent.start_rpc()
        assert result["status"] == "ok"
        assert result["port"] == 50055
        assert result["endpoint"] == "0.0.0.0:50055"

    @patch.object(RpcServer, "start", return_value=50055)
    def test_stop_rpc(self, mock_start):
        agent = WorkerAgent(node_id="node-1", port=52415)
        agent.start_rpc()
        result = agent.stop_rpc()
        assert result["status"] == "ok"

    def test_rpc_status(self):
        agent = WorkerAgent(node_id="node-1", port=52415)
        status = agent.rpc_status()
        assert "running" in status
