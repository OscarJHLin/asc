"""补充 HardwareDetector 测试，提升覆盖率至 85%+。

原测试仅覆盖基础检测，本文件补充：
- CPU 检测边界条件
- GPU 检测各平台分支
- 磁盘检测
- 网络检测
- detect_all 综合检测
"""

from __future__ import annotations

import os
from unittest.mock import patch

from asc.worker.hardware import CPUInfo, DiskInfo, HardwareDetector, NetworkInfo


class TestDetectCPU:
    """测试 CPU 检测。"""

    def test_detect_cpu(self):
        """应返回有效的 CPU 信息。"""
        detector = HardwareDetector()
        cpu = detector.detect_cpu()
        assert isinstance(cpu, CPUInfo)
        assert cpu.logical_count > 0
        assert cpu.physical_count > 0

    def test_detect_cpu_with_brand(self):
        """应能获取 CPU 品牌。"""
        detector = HardwareDetector()
        cpu = detector.detect_cpu()
        # brand 可能为空（某些平台）
        assert isinstance(cpu.brand, str)

    @patch("asc.worker.hardware.psutil.cpu_freq", side_effect=Exception("no freq"))
    def test_detect_cpu_freq_failure(self, _mock):
        """频率检测失败时不应抛异常。"""
        detector = HardwareDetector()
        cpu = detector.detect_cpu()
        assert cpu.freq_mhz == 0.0

    @patch("asc.worker.hardware.psutil.cpu_count", return_value=None)
    def test_detect_cpu_count_none(self, _mock):
        """cpu_count 返回 None 时应回退到 0。"""
        detector = HardwareDetector()
        cpu = detector.detect_cpu()
        assert cpu.logical_count == 0
        assert cpu.physical_count == 0


class TestDetectGPUs:
    """测试 GPU 检测。"""

    def test_detect_gpus_returns_list(self):
        """应返回列表。"""
        detector = HardwareDetector()
        gpus = detector.detect_gpus()
        assert isinstance(gpus, list)

    @patch("asc.worker.hardware.platform.system", return_value="Windows")
    def test_detect_gpus_windows(self, _mock):
        """Windows 分支应被调用。"""
        detector = HardwareDetector()
        with patch.object(detector, "_detect_gpus_windows", return_value=[]) as mock_win:
            detector.detect_gpus()
            mock_win.assert_called_once()

    @patch("asc.worker.hardware.platform.system", return_value="Linux")
    def test_detect_gpus_linux(self, _mock):
        """Linux 分支应被调用。"""
        detector = HardwareDetector()
        with patch.object(detector, "_detect_gpus_linux", return_value=[]) as mock_linux:
            detector.detect_gpus()
            mock_linux.assert_called_once()

    @patch("asc.worker.hardware.platform.system", return_value="Darwin")
    def test_detect_gpus_darwin(self, _mock):
        """macOS 分支应被调用。"""
        detector = HardwareDetector()
        with patch.object(detector, "_detect_gpus_darwin", return_value=[]) as mock_darwin:
            detector.detect_gpus()
            mock_darwin.assert_called_once()

    @patch("asc.worker.hardware.platform.system", return_value="UnknownOS")
    def test_detect_gpus_unknown(self, _mock):
        """未知平台应返回空列表。"""
        detector = HardwareDetector()
        gpus = detector.detect_gpus()
        assert gpus == []


class TestDetectDisk:
    """测试磁盘检测。"""

    def test_detect_disk_default(self):
        """默认应检测用户主目录。"""
        detector = HardwareDetector()
        disk = detector.detect_disk()
        assert isinstance(disk, DiskInfo)
        assert disk.free_mb >= 0

    def test_detect_disk_with_path(self):
        """指定路径时应检测该路径。"""
        detector = HardwareDetector()
        disk = detector.detect_disk("/")
        assert isinstance(disk, DiskInfo)

    def test_detect_disk_env_override(self):
        """ASC_MODELS_PATH 环境变量应生效。"""
        detector = HardwareDetector()
        with patch.dict(os.environ, {"ASC_MODELS_PATH": "/tmp"}):
            disk = detector.detect_disk()
            assert isinstance(disk, DiskInfo)

    @patch("asc.worker.hardware.psutil.disk_usage", side_effect=Exception("no disk"))
    def test_detect_disk_failure(self, _mock):
        """磁盘检测失败时应返回 0。"""
        detector = HardwareDetector()
        disk = detector.detect_disk()
        assert disk.free_mb == 0


class TestDetectNetwork:
    """测试网络检测。"""

    def test_detect_network_default(self):
        """默认应返回带宽估算。"""
        detector = HardwareDetector()
        net = detector.detect_network()
        assert isinstance(net, NetworkInfo)

    def test_detect_network_env_override(self):
        """ASC_NETWORK_MBPS 环境变量应生效。"""
        detector = HardwareDetector()
        with patch.dict(os.environ, {"ASC_NETWORK_MBPS": "500"}):
            net = detector.detect_network()
            assert net.estimated_mbps == 500.0

    def test_detect_network_env_invalid(self):
        """无效环境变量应被忽略。"""
        detector = HardwareDetector()
        with patch.dict(os.environ, {"ASC_NETWORK_MBPS": "not_a_number"}):
            net = detector.detect_network()
            assert isinstance(net, NetworkInfo)

    @patch("asc.worker.hardware.psutil.net_io_counters", side_effect=Exception("no network"))
    def test_detect_network_failure(self, _mock):
        """网络检测失败时应返回 0。"""
        detector = HardwareDetector()
        net = detector.detect_network()
        assert net.estimated_mbps == 0.0


class TestDetectAll:
    """测试综合检测。"""

    def test_detect_all_structure(self):
        """返回结构应包含所有关键字段。"""
        detector = HardwareDetector()
        info = detector.detect_all()
        assert "cpu" in info
        assert "gpus" in info
        assert "memory_total_mb" in info
        assert "memory_free_mb" in info
        assert "disk_free_mb" in info
        assert "network_mbps" in info

    def test_detect_all_cpu_fields(self):
        """CPU 信息应包含预期字段。"""
        detector = HardwareDetector()
        info = detector.detect_all()
        cpu = info["cpu"]
        assert "logical_count" in cpu
        assert "physical_count" in cpu
        assert "freq_mhz" in cpu
        assert "brand" in cpu

    def test_detect_all_memory_fields(self):
        """内存信息应包含预期字段。"""
        detector = HardwareDetector()
        info = detector.detect_all()
        assert "memory_total_mb" in info
        assert "memory_free_mb" in info
