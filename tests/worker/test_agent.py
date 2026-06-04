"""测试 Worker Agent。

Worker Agent 运行在 Worker 节点上，提供：
- 健康检查
- 资源查询（CPU/GPU/内存）
- RPC Server 生命周期管理
- 模型加载/卸载
"""

from unittest.mock import MagicMock, patch

from asc.worker.agent import (
    GPUInfo,
    NodeResources,
    RPCServerManager,
    WorkerAgent,
)


class TestGPUInfo:
    """GPU 信息值对象。"""

    def test_create(self):
        gpu = GPUInfo(index=0, name="RTX 4090", vram_total_mb=24564, vram_free_mb=20000)
        assert gpu.index == 0
        assert gpu.name == "RTX 4090"
        assert gpu.vram_total_mb == 24564
        assert gpu.vram_free_mb == 20000

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
    @patch.object(WorkerAgent, "_detect_gpus", return_value=[])
    def test_get_resources(self, mock_gpus, mock_vm, mock_cpu_pct, mock_cpu_cnt):
        mock_vm.return_value = MagicMock(total=32 * 1024**3, available=24 * 1024**3)
        agent = WorkerAgent(node_id="node-1", port=52415)
        res = agent.get_resources()
        assert res.cpu_count == 8
        assert res.cpu_percent == 25.0
        assert res.memory_total_mb == 32 * 1024
        assert res.gpus == []

    def test_health_check(self):
        agent = WorkerAgent(node_id="node-1", port=52415)
        health = agent.health()
        assert health["status"] == "ok"
        assert health["node_id"] == "node-1"
