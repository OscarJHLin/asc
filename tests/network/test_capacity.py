"""测试 ASC Cluster Link Protocol - 节点容量协议。"""

from asc.network.capacity import (
    CapacityMetrics,
    CapacityQuery,
    CapacityReport,
    CapacityResponse,
    GpuMetrics,
)


class TestGpuMetrics:
    """GPU 指标。"""

    def test_create(self):
        gpu = GpuMetrics(
            index=0,
            name="RTX 4060",
            vram_total_mb=8192,
            vram_free_mb=4096,
            utilization_percent=75.0,
            temperature_c=65,
        )
        assert gpu.index == 0
        assert gpu.vram_free_mb == 4096

    def test_to_dict_from_dict(self):
        gpu = GpuMetrics(
            index=0, name="RTX 4060",
            vram_total_mb=8192, vram_free_mb=4096,
            utilization_percent=75.0, temperature_c=65,
        )
        d = gpu.to_dict()
        restored = GpuMetrics.from_dict(d)
        assert restored.name == "RTX 4060"
        assert restored.temperature_c == 65


class TestCapacityMetrics:
    """节点容量指标。"""

    def test_create(self):
        metrics = CapacityMetrics(
            cpu_count=8,
            cpu_percent=30.0,
            memory_total_mb=16000,
            memory_free_mb=8000,
            gpus=[
                GpuMetrics(0, "RTX 4060", 8192, 4096, 75.0, 65),
            ],
            active_tasks=2,
            max_concurrent_tasks=4,
            compute_score=85.5,
            network_latency_ms=1.2,
        )
        assert metrics.cpu_count == 8
        assert metrics.active_tasks == 2
        assert len(metrics.gpus) == 1

    def test_available_capacity(self):
        """可用容量计算。"""
        metrics = CapacityMetrics(
            cpu_count=8,
            cpu_percent=30.0,
            memory_total_mb=16000,
            memory_free_mb=8000,
            gpus=[
                GpuMetrics(0, "RTX 4060", 8192, 4096, 75.0, 65),
            ],
            active_tasks=2,
            max_concurrent_tasks=4,
            compute_score=85.5,
            network_latency_ms=1.2,
        )
        # 可用并发 = 最大并发 - 活跃任务
        assert metrics.available_slots == 2

    def test_total_vram_free(self):
        """总空闲 VRAM。"""
        metrics = CapacityMetrics(
            cpu_count=8, cpu_percent=30.0,
            memory_total_mb=16000, memory_free_mb=8000,
            gpus=[
                GpuMetrics(0, "GPU0", 8192, 4096, 50.0, 60),
                GpuMetrics(1, "GPU1", 8192, 2048, 80.0, 70),
            ],
            active_tasks=1, max_concurrent_tasks=4,
            compute_score=80.0, network_latency_ms=1.0,
        )
        assert metrics.total_vram_free_mb == 6144

    def test_to_dict_from_dict(self):
        metrics = CapacityMetrics(
            cpu_count=8, cpu_percent=30.0,
            memory_total_mb=16000, memory_free_mb=8000,
            gpus=[GpuMetrics(0, "GPU0", 8192, 4096, 50.0, 60)],
            active_tasks=1, max_concurrent_tasks=4,
            compute_score=80.0, network_latency_ms=1.0,
        )
        d = metrics.to_dict()
        restored = CapacityMetrics.from_dict(d)
        assert restored.cpu_count == 8
        assert len(restored.gpus) == 1
        assert restored.gpus[0].name == "GPU0"


class TestCapacityReport:
    """容量上报消息。"""

    def test_create(self):
        metrics = CapacityMetrics(
            cpu_count=8, cpu_percent=30.0,
            memory_total_mb=16000, memory_free_mb=8000,
            gpus=[], active_tasks=0, max_concurrent_tasks=4,
            compute_score=80.0, network_latency_ms=1.0,
        )
        report = CapacityReport(
            node_id="worker-1",
            metrics=metrics,
        )
        assert report.node_id == "worker-1"
        assert report.metrics.cpu_count == 8

    def test_to_dict_from_dict(self):
        metrics = CapacityMetrics(
            cpu_count=4, cpu_percent=10.0,
            memory_total_mb=8000, memory_free_mb=4000,
            gpus=[], active_tasks=0, max_concurrent_tasks=2,
            compute_score=50.0, network_latency_ms=2.0,
        )
        report = CapacityReport(node_id="w1", metrics=metrics)
        d = report.to_dict()
        restored = CapacityReport.from_dict(d)
        assert restored.node_id == "w1"
        assert restored.metrics.compute_score == 50.0


class TestCapacityQuery:
    """容量查询消息。"""

    def test_create(self):
        query = CapacityQuery(requester_id="master")
        assert query.requester_id == "master"

    def test_to_dict_from_dict(self):
        query = CapacityQuery(requester_id="master")
        d = query.to_dict()
        restored = CapacityQuery.from_dict(d)
        assert restored.requester_id == "master"


class TestCapacityResponse:
    """容量响应消息。"""

    def test_create(self):
        metrics = CapacityMetrics(
            cpu_count=8, cpu_percent=30.0,
            memory_total_mb=16000, memory_free_mb=8000,
            gpus=[], active_tasks=1, max_concurrent_tasks=4,
            compute_score=80.0, network_latency_ms=1.0,
        )
        resp = CapacityResponse(
            node_id="w1",
            metrics=metrics,
        )
        assert resp.node_id == "w1"

    def test_to_dict_from_dict(self):
        metrics = CapacityMetrics(
            cpu_count=8, cpu_percent=30.0,
            memory_total_mb=16000, memory_free_mb=8000,
            gpus=[], active_tasks=0, max_concurrent_tasks=4,
            compute_score=80.0, network_latency_ms=1.0,
        )
        resp = CapacityResponse(node_id="w1", metrics=metrics)
        d = resp.to_dict()
        restored = CapacityResponse.from_dict(d)
        assert restored.metrics.cpu_count == 8
