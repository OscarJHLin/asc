"""ASC 容量协议增强测试 - 覆盖边界条件、属性计算、序列化链和不可变性。"""

import pytest

from asc.network.capacity import (
    CapacityMetrics,
    CapacityQuery,
    CapacityReport,
    CapacityResponse,
    GpuMetrics,
)

# ---------------------------------------------------------------------------
# GpuMetrics 边界用例
# ---------------------------------------------------------------------------

class TestGpuMetricsEdgeCases:
    """GpuMetrics 边界条件测试。"""

    def test_zero_utilization(self):
        """利用率为 0% 时应正常创建。"""
        gpu = GpuMetrics(
            index=0, name="RTX 4060",
            vram_total_mb=8192, vram_free_mb=8192,
            utilization_percent=0.0, temperature_c=35,
        )
        assert gpu.utilization_percent == 0.0

    def test_full_utilization(self):
        """利用率为 100% 时应正常创建。"""
        gpu = GpuMetrics(
            index=0, name="RTX 4060",
            vram_total_mb=8192, vram_free_mb=0,
            utilization_percent=100.0, temperature_c=90,
        )
        assert gpu.utilization_percent == 100.0

    def test_temperature_zero(self):
        """温度为 0°C 时应正常创建。"""
        gpu = GpuMetrics(
            index=0, name="GPU",
            vram_total_mb=8192, vram_free_mb=4096,
            utilization_percent=50.0, temperature_c=0,
        )
        assert gpu.temperature_c == 0

    def test_very_high_temperature(self):
        """温度超过 100°C 时应正常创建（极端过热场景）。"""
        gpu = GpuMetrics(
            index=0, name="GPU",
            vram_total_mb=8192, vram_free_mb=4096,
            utilization_percent=99.0, temperature_c=105,
        )
        assert gpu.temperature_c == 105

    def test_vram_fully_used(self):
        """VRAM 完全用尽（vram_free_mb=0）。"""
        gpu = GpuMetrics(
            index=0, name="GPU",
            vram_total_mb=8192, vram_free_mb=0,
            utilization_percent=100.0, temperature_c=85,
        )
        assert gpu.vram_free_mb == 0

    def test_vram_fully_free(self):
        """VRAM 完全空闲（vram_free_mb == vram_total_mb）。"""
        gpu = GpuMetrics(
            index=0, name="GPU",
            vram_total_mb=8192, vram_free_mb=8192,
            utilization_percent=0.0, temperature_c=30,
        )
        assert gpu.vram_free_mb == gpu.vram_total_mb

    def test_gpu_name_with_special_characters(self):
        """GPU 名称包含特殊字符（中文、符号、空格）。"""
        name = "NVIDIA GeForce RTX 4090 🎮 / 特供版"
        gpu = GpuMetrics(
            index=0, name=name,
            vram_total_mb=24576, vram_free_mb=12288,
            utilization_percent=50.0, temperature_c=60,
        )
        assert gpu.name == name
        # 序列化往返后特殊字符不变
        restored = GpuMetrics.from_dict(gpu.to_dict())
        assert restored.name == name

    def test_negative_index(self):
        """GPU index 为负数时应正常创建（不限制语义）。"""
        gpu = GpuMetrics(
            index=-1, name="GPU",
            vram_total_mb=8192, vram_free_mb=4096,
            utilization_percent=50.0, temperature_c=60,
        )
        assert gpu.index == -1

    def test_very_large_index(self):
        """GPU index 为极大值时应正常创建。"""
        gpu = GpuMetrics(
            index=999999, name="GPU",
            vram_total_mb=8192, vram_free_mb=4096,
            utilization_percent=50.0, temperature_c=60,
        )
        assert gpu.index == 999999

    def test_roundtrip_preserves_all_fields(self):
        """to_dict -> from_dict 往返后所有字段一致。"""
        original = GpuMetrics(
            index=3, name="A100",
            vram_total_mb=81920, vram_free_mb=40960,
            utilization_percent=42.5, temperature_c=72,
        )
        restored = GpuMetrics.from_dict(original.to_dict())
        assert restored.index == original.index
        assert restored.name == original.name
        assert restored.vram_total_mb == original.vram_total_mb
        assert restored.vram_free_mb == original.vram_free_mb
        assert restored.utilization_percent == original.utilization_percent
        assert restored.temperature_c == original.temperature_c


# ---------------------------------------------------------------------------
# CapacityMetrics 边界用例
# ---------------------------------------------------------------------------

class TestCapacityMetricsEdgeCases:
    """CapacityMetrics 边界条件测试。"""

    def test_zero_gpus(self):
        """无 GPU 时 gpus 为空列表。"""
        metrics = CapacityMetrics(
            cpu_count=4, cpu_percent=10.0,
            memory_total_mb=8000, memory_free_mb=4000,
            gpus=[], active_tasks=0, max_concurrent_tasks=2,
            compute_score=30.0, network_latency_ms=1.0,
        )
        assert metrics.gpus == []
        assert metrics.total_vram_free_mb == 0

    def test_multiple_gpus(self):
        """4 块以上 GPU 的场景。"""
        gpus = [
            GpuMetrics(i, f"GPU-{i}", 8192, 2048 * (i + 1), 50.0, 60 + i)
            for i in range(4)
        ]
        metrics = CapacityMetrics(
            cpu_count=32, cpu_percent=45.0,
            memory_total_mb=64000, memory_free_mb=32000,
            gpus=gpus, active_tasks=3, max_concurrent_tasks=8,
            compute_score=95.0, network_latency_ms=0.5,
        )
        assert len(metrics.gpus) == 4
        # 2048*(1+2+3+4) = 2048*10 = 20480
        assert metrics.total_vram_free_mb == 20480

    def test_available_slots_equal(self):
        """active_tasks == max_concurrent_tasks 时 available_slots 为 0。"""
        metrics = CapacityMetrics(
            cpu_count=4, cpu_percent=80.0,
            memory_total_mb=8000, memory_free_mb=1000,
            gpus=[], active_tasks=4, max_concurrent_tasks=4,
            compute_score=60.0, network_latency_ms=1.0,
        )
        assert metrics.available_slots == 0

    def test_available_slots_overflow_protection(self):
        """active_tasks > max_concurrent_tasks 时 available_slots 不为负数。"""
        metrics = CapacityMetrics(
            cpu_count=4, cpu_percent=99.0,
            memory_total_mb=8000, memory_free_mb=100,
            gpus=[], active_tasks=10, max_concurrent_tasks=4,
            compute_score=60.0, network_latency_ms=1.0,
        )
        assert metrics.available_slots == 0

    def test_total_vram_free_no_gpus(self):
        """无 GPU 时 total_vram_free_mb 应为 0。"""
        metrics = CapacityMetrics(
            cpu_count=2, cpu_percent=20.0,
            memory_total_mb=4000, memory_free_mb=2000,
            gpus=[], active_tasks=0, max_concurrent_tasks=1,
            compute_score=10.0, network_latency_ms=5.0,
        )
        assert metrics.total_vram_free_mb == 0

    def test_total_vram_free_multiple_gpus(self):
        """多 GPU 时 total_vram_free_mb 为所有 GPU 空闲 VRAM 之和。"""
        gpus = [
            GpuMetrics(0, "GPU0", 8192, 4096, 50.0, 60),
            GpuMetrics(1, "GPU1", 16384, 8192, 30.0, 55),
            GpuMetrics(2, "GPU2", 24576, 0, 100.0, 90),
        ]
        metrics = CapacityMetrics(
            cpu_count=16, cpu_percent=40.0,
            memory_total_mb=32000, memory_free_mb=16000,
            gpus=gpus, active_tasks=1, max_concurrent_tasks=4,
            compute_score=88.0, network_latency_ms=0.8,
        )
        assert metrics.total_vram_free_mb == 4096 + 8192 + 0

    def test_cpu_percent_zero(self):
        """cpu_percent 为 0.0（CPU 完全空闲）。"""
        metrics = CapacityMetrics(
            cpu_count=8, cpu_percent=0.0,
            memory_total_mb=16000, memory_free_mb=16000,
            gpus=[], active_tasks=0, max_concurrent_tasks=4,
            compute_score=70.0, network_latency_ms=1.0,
        )
        assert metrics.cpu_percent == 0.0

    def test_cpu_percent_hundred(self):
        """cpu_percent 为 100.0（CPU 满载）。"""
        metrics = CapacityMetrics(
            cpu_count=8, cpu_percent=100.0,
            memory_total_mb=16000, memory_free_mb=0,
            gpus=[], active_tasks=4, max_concurrent_tasks=4,
            compute_score=70.0, network_latency_ms=1.0,
        )
        assert metrics.cpu_percent == 100.0

    def test_memory_free_zero(self):
        """memory_free_mb 为 0（内存耗尽）。"""
        metrics = CapacityMetrics(
            cpu_count=8, cpu_percent=90.0,
            memory_total_mb=16000, memory_free_mb=0,
            gpus=[], active_tasks=4, max_concurrent_tasks=4,
            compute_score=40.0, network_latency_ms=1.0,
        )
        assert metrics.memory_free_mb == 0

    def test_network_latency_zero(self):
        """network_latency_ms 为 0.0（本机通信场景）。"""
        metrics = CapacityMetrics(
            cpu_count=8, cpu_percent=30.0,
            memory_total_mb=16000, memory_free_mb=8000,
            gpus=[], active_tasks=0, max_concurrent_tasks=4,
            compute_score=80.0, network_latency_ms=0.0,
        )
        assert metrics.network_latency_ms == 0.0

    def test_compute_score_zero(self):
        """compute_score 为 0.0（算力极低或未评测）。"""
        metrics = CapacityMetrics(
            cpu_count=1, cpu_percent=0.0,
            memory_total_mb=1024, memory_free_mb=512,
            gpus=[], active_tasks=0, max_concurrent_tasks=1,
            compute_score=0.0, network_latency_ms=10.0,
        )
        assert metrics.compute_score == 0.0

    def test_compute_score_very_high(self):
        """compute_score 为极高值。"""
        metrics = CapacityMetrics(
            cpu_count=128, cpu_percent=50.0,
            memory_total_mb=512000, memory_free_mb=256000,
            gpus=[], active_tasks=0, max_concurrent_tasks=64,
            compute_score=99999.99, network_latency_ms=0.1,
        )
        assert metrics.compute_score == 99999.99


# ---------------------------------------------------------------------------
# CapacityMetrics from_dict 缺失 gpus 键
# ---------------------------------------------------------------------------

class TestCapacityMetricsFromDictEdgeCases:
    """CapacityMetrics.from_dict 边界测试。"""

    def test_missing_gpus_key_defaults_to_empty(self):
        """from_dict 中缺少 'gpus' 键应默认为空列表。"""
        d = {
            "cpu_count": 4,
            "cpu_percent": 20.0,
            "memory_total_mb": 8000,
            "memory_free_mb": 4000,
            "active_tasks": 0,
            "max_concurrent_tasks": 2,
            "compute_score": 50.0,
            "network_latency_ms": 1.0,
        }
        metrics = CapacityMetrics.from_dict(d)
        assert metrics.gpus == []
        assert metrics.total_vram_free_mb == 0

    def test_empty_gpus_list(self):
        """from_dict 中 gpus 为空列表。"""
        d = {
            "cpu_count": 4,
            "cpu_percent": 20.0,
            "memory_total_mb": 8000,
            "memory_free_mb": 4000,
            "gpus": [],
            "active_tasks": 0,
            "max_concurrent_tasks": 2,
            "compute_score": 50.0,
            "network_latency_ms": 1.0,
        }
        metrics = CapacityMetrics.from_dict(d)
        assert metrics.gpus == []


# ---------------------------------------------------------------------------
# 完整序列化链
# ---------------------------------------------------------------------------

class TestFullSerializationChain:
    """完整序列化链测试：to_dict -> from_dict -> 字段验证。"""

    def _make_complex_metrics(self):
        """构造包含多 GPU 的复杂指标。"""
        return CapacityMetrics(
            cpu_count=64, cpu_percent=55.5,
            memory_total_mb=256000, memory_free_mb=128000,
            gpus=[
                GpuMetrics(0, "A100-80GB", 81920, 40960, 60.0, 65),
                GpuMetrics(1, "A100-80GB", 81920, 0, 100.0, 88),
                GpuMetrics(2, "H100-SXM", 98304, 65536, 30.0, 50),
            ],
            active_tasks=5, max_concurrent_tasks=8,
            compute_score=97.5, network_latency_ms=0.3,
        )

    def test_capacity_report_roundtrip(self):
        """CapacityReport 复杂指标完整往返序列化。"""
        metrics = self._make_complex_metrics()
        report = CapacityReport(node_id="worker-gpu-01", metrics=metrics)

        d = report.to_dict()
        restored = CapacityReport.from_dict(d)

        assert restored.node_id == "worker-gpu-01"
        assert restored.metrics.cpu_count == 64
        assert restored.metrics.cpu_percent == 55.5
        assert restored.metrics.memory_total_mb == 256000
        assert restored.metrics.memory_free_mb == 128000
        assert len(restored.metrics.gpus) == 3
        assert restored.metrics.gpus[0].name == "A100-80GB"
        assert restored.metrics.gpus[1].vram_free_mb == 0
        assert restored.metrics.gpus[2].vram_total_mb == 98304
        assert restored.metrics.active_tasks == 5
        assert restored.metrics.max_concurrent_tasks == 8
        assert restored.metrics.available_slots == 3
        assert restored.metrics.compute_score == 97.5
        assert restored.metrics.network_latency_ms == 0.3
        assert restored.metrics.total_vram_free_mb == 40960 + 0 + 65536

    def test_capacity_query_roundtrip(self):
        """CapacityQuery 完整往返序列化。"""
        query = CapacityQuery(requester_id="master-node-01")
        d = query.to_dict()
        restored = CapacityQuery.from_dict(d)
        assert restored.requester_id == "master-node-01"

    def test_capacity_response_multiple_gpus_roundtrip(self):
        """CapacityResponse 多 GPU 完整往返序列化。"""
        metrics = self._make_complex_metrics()
        resp = CapacityResponse(node_id="worker-gpu-01", metrics=metrics)

        d = resp.to_dict()
        restored = CapacityResponse.from_dict(d)

        assert restored.node_id == "worker-gpu-01"
        assert len(restored.metrics.gpus) == 3
        assert restored.metrics.gpus[2].name == "H100-SXM"
        assert restored.metrics.total_vram_free_mb == 40960 + 0 + 65536
        assert restored.metrics.available_slots == 3

    def test_nested_dict_structure(self):
        """验证 to_dict 输出的嵌套结构正确。"""
        metrics = self._make_complex_metrics()
        report = CapacityReport(node_id="w1", metrics=metrics)
        d = report.to_dict()

        assert "node_id" in d
        assert "metrics" in d
        assert "gpus" in d["metrics"]
        assert len(d["metrics"]["gpus"]) == 3
        assert d["metrics"]["gpus"][0]["index"] == 0
        assert d["metrics"]["gpus"][1]["name"] == "A100-80GB"
        assert d["metrics"]["gpus"][2]["vram_free_mb"] == 65536


# ---------------------------------------------------------------------------
# Frozen dataclass 不可变性验证
# ---------------------------------------------------------------------------

class TestFrozenDataclassImmutability:
    """验证所有 frozen dataclass 实例不可修改。"""

    def test_gpu_metrics_immutable(self):
        """GpuMetrics 为 frozen dataclass，属性不可修改。"""
        gpu = GpuMetrics(
            index=0, name="GPU",
            vram_total_mb=8192, vram_free_mb=4096,
            utilization_percent=50.0, temperature_c=60,
        )
        with pytest.raises(AttributeError):
            gpu.index = 1

    def test_gpu_metrics_name_immutable(self):
        """GpuMetrics.name 不可修改。"""
        gpu = GpuMetrics(
            index=0, name="GPU",
            vram_total_mb=8192, vram_free_mb=4096,
            utilization_percent=50.0, temperature_c=60,
        )
        with pytest.raises(AttributeError):
            gpu.name = "new-name"

    def test_capacity_metrics_immutable(self):
        """CapacityMetrics 为 frozen dataclass，属性不可修改。"""
        metrics = CapacityMetrics(
            cpu_count=8, cpu_percent=30.0,
            memory_total_mb=16000, memory_free_mb=8000,
            gpus=[], active_tasks=0, max_concurrent_tasks=4,
            compute_score=80.0, network_latency_ms=1.0,
        )
        with pytest.raises(AttributeError):
            metrics.cpu_count = 16

    def test_capacity_metrics_active_tasks_immutable(self):
        """CapacityMetrics.active_tasks 不可修改。"""
        metrics = CapacityMetrics(
            cpu_count=8, cpu_percent=30.0,
            memory_total_mb=16000, memory_free_mb=8000,
            gpus=[], active_tasks=0, max_concurrent_tasks=4,
            compute_score=80.0, network_latency_ms=1.0,
        )
        with pytest.raises(AttributeError):
            metrics.active_tasks = 99

    def test_capacity_report_immutable(self):
        """CapacityReport 为 frozen dataclass，属性不可修改。"""
        report = CapacityReport(
            node_id="w1",
            metrics=CapacityMetrics(
                cpu_count=4, cpu_percent=10.0,
                memory_total_mb=8000, memory_free_mb=4000,
                gpus=[], active_tasks=0, max_concurrent_tasks=2,
                compute_score=50.0, network_latency_ms=2.0,
            ),
        )
        with pytest.raises(AttributeError):
            report.node_id = "w2"

    def test_capacity_query_immutable(self):
        """CapacityQuery 为 frozen dataclass，属性不可修改。"""
        query = CapacityQuery(requester_id="master")
        with pytest.raises(AttributeError):
            query.requester_id = "other"

    def test_capacity_response_immutable(self):
        """CapacityResponse 为 frozen dataclass，属性不可修改。"""
        resp = CapacityResponse(
            node_id="w1",
            metrics=CapacityMetrics(
                cpu_count=4, cpu_percent=10.0,
                memory_total_mb=8000, memory_free_mb=4000,
                gpus=[], active_tasks=0, max_concurrent_tasks=2,
                compute_score=50.0, network_latency_ms=2.0,
            ),
        )
        with pytest.raises(AttributeError):
            resp.node_id = "w2"


# ---------------------------------------------------------------------------
# 属性计算专项测试
# ---------------------------------------------------------------------------

class TestPropertyCalculations:
    """属性计算专项测试。"""

    def test_available_slots_normal(self):
        """正常情况：available_slots = max_concurrent_tasks - active_tasks。"""
        metrics = CapacityMetrics(
            cpu_count=8, cpu_percent=30.0,
            memory_total_mb=16000, memory_free_mb=8000,
            gpus=[], active_tasks=2, max_concurrent_tasks=6,
            compute_score=80.0, network_latency_ms=1.0,
        )
        assert metrics.available_slots == 4

    def test_available_slots_zero_active(self):
        """active_tasks 为 0 时 available_slots == max_concurrent_tasks。"""
        metrics = CapacityMetrics(
            cpu_count=8, cpu_percent=0.0,
            memory_total_mb=16000, memory_free_mb=16000,
            gpus=[], active_tasks=0, max_concurrent_tasks=8,
            compute_score=80.0, network_latency_ms=1.0,
        )
        assert metrics.available_slots == 8

    def test_available_slots_overflow(self):
        """active_tasks > max_concurrent_tasks 时 available_slots 为 0（不溢出为负）。"""
        metrics = CapacityMetrics(
            cpu_count=8, cpu_percent=100.0,
            memory_total_mb=16000, memory_free_mb=0,
            gpus=[], active_tasks=15, max_concurrent_tasks=4,
            compute_score=30.0, network_latency_ms=5.0,
        )
        assert metrics.available_slots == 0

    def test_total_vram_free_single_gpu(self):
        """单 GPU 时 total_vram_free_mb 等于该 GPU 的 vram_free_mb。"""
        gpu = GpuMetrics(0, "GPU", 8192, 3000, 60.0, 70)
        metrics = CapacityMetrics(
            cpu_count=4, cpu_percent=30.0,
            memory_total_mb=8000, memory_free_mb=4000,
            gpus=[gpu], active_tasks=1, max_concurrent_tasks=4,
            compute_score=70.0, network_latency_ms=1.0,
        )
        assert metrics.total_vram_free_mb == 3000

    def test_total_vram_free_all_zero(self):
        """所有 GPU 的 vram_free_mb 为 0 时 total_vram_free_mb 为 0。"""
        gpus = [
            GpuMetrics(0, "GPU0", 8192, 0, 100.0, 90),
            GpuMetrics(1, "GPU1", 8192, 0, 100.0, 92),
        ]
        metrics = CapacityMetrics(
            cpu_count=8, cpu_percent=90.0,
            memory_total_mb=16000, memory_free_mb=100,
            gpus=gpus, active_tasks=4, max_concurrent_tasks=4,
            compute_score=50.0, network_latency_ms=1.0,
        )
        assert metrics.total_vram_free_mb == 0

    def test_total_vram_free_all_max(self):
        """所有 GPU 的 vram_free_mb == vram_total_mb 时总和正确。"""
        gpus = [
            GpuMetrics(0, "GPU0", 8192, 8192, 0.0, 30),
            GpuMetrics(1, "GPU1", 16384, 16384, 0.0, 28),
        ]
        metrics = CapacityMetrics(
            cpu_count=8, cpu_percent=5.0,
            memory_total_mb=16000, memory_free_mb=15000,
            gpus=gpus, active_tasks=0, max_concurrent_tasks=4,
            compute_score=90.0, network_latency_ms=0.5,
        )
        assert metrics.total_vram_free_mb == 8192 + 16384
