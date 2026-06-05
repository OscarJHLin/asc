"""测试 HardwareDetector 跨平台硬件检测模块。"""

from unittest.mock import MagicMock, patch

import pytest

from asc.worker.gpu_info import GPUInfo
from asc.worker.hardware import CPUInfo, DiskInfo, HardwareDetector, NetworkInfo


class TestCPUInfo:
    """CPU 信息值对象。"""

    def test_create(self):
        cpu = CPUInfo(logical_count=8, physical_count=4, freq_mhz=3200.0, brand="Intel i7")
        assert cpu.logical_count == 8
        assert cpu.physical_count == 4
        assert cpu.freq_mhz == 3200.0
        assert cpu.brand == "Intel i7"

    def test_frozen(self):
        cpu = CPUInfo(logical_count=8, physical_count=4, freq_mhz=3200.0, brand="Intel i7")
        with pytest.raises((AttributeError, TypeError)):
            cpu.brand = "AMD"  # type: ignore[misc]


class TestDiskInfo:
    """磁盘信息值对象。"""

    def test_create(self):
        disk = DiskInfo(free_mb=102400)
        assert disk.free_mb == 102400


class TestNetworkInfo:
    """网络信息值对象。"""

    def test_create(self):
        net = NetworkInfo(estimated_mbps=1000.0)
        assert net.estimated_mbps == 1000.0


class TestHardwareDetectorCPU:
    """CPU 检测。"""

    @patch("psutil.cpu_count")
    @patch("psutil.cpu_freq")
    @patch.object(HardwareDetector, "_get_cpu_brand", return_value="Intel(R) Core(TM) i7")
    def test_detect_cpu(self, mock_brand, mock_freq, mock_count):
        mock_count.side_effect = lambda logical: 8 if logical else 4
        mock_freq.return_value = MagicMock(current=3200.0)

        detector = HardwareDetector()
        cpu = detector.detect_cpu()
        assert cpu.logical_count == 8
        assert cpu.physical_count == 4
        assert cpu.freq_mhz == 3200.0
        assert cpu.brand == "Intel(R) Core(TM) i7"

    @patch("psutil.cpu_count", return_value=None)
    @patch("psutil.cpu_freq", side_effect=Exception("no freq"))
    @patch.object(HardwareDetector, "_get_cpu_brand", return_value="")
    def test_detect_cpu_fallback(self, mock_brand, mock_freq, mock_count):
        detector = HardwareDetector()
        cpu = detector.detect_cpu()
        assert cpu.logical_count == 0
        assert cpu.physical_count == 0
        assert cpu.freq_mhz == 0.0
        assert cpu.brand == ""


class TestHardwareDetectorGPU:
    """GPU 检测。"""

    @patch("platform.system", return_value="Windows")
    @patch.object(HardwareDetector, "_parse_nvidia_smi")
    def test_detect_gpus_windows(self, mock_parse, mock_system):
        mock_parse.return_value = [
            GPUInfo(
                index=0,
                name="RTX 4090",
                vram_total_mb=24564,
                vram_free_mb=20000,
                vendor="nvidia",
            ),
        ]
        detector = HardwareDetector()
        gpus = detector.detect_gpus()
        assert len(gpus) == 1
        assert gpus[0].vendor == "nvidia"

    @patch("platform.system", return_value="Linux")
    @patch.object(HardwareDetector, "_parse_nvidia_smi", return_value=[])
    @patch.object(HardwareDetector, "_parse_rocm_smi")
    def test_detect_gpus_linux_fallback_rocm(self, mock_rocm, mock_nvidia, mock_system):
        mock_rocm.return_value = [
            GPUInfo(
                index=0,
                name="RX 7900 XTX",
                vram_total_mb=24564,
                vram_free_mb=20000,
                vendor="amd",
            ),
        ]
        detector = HardwareDetector()
        gpus = detector.detect_gpus()
        assert len(gpus) == 1
        assert gpus[0].vendor == "amd"

    @patch("platform.system", return_value="Darwin")
    @patch("subprocess.run")
    @patch("psutil.virtual_memory")
    def test_detect_gpus_darwin(self, mock_vm, mock_run, mock_system):
        mock_vm.return_value = MagicMock(total=16 * 1024**3, available=8 * 1024**3)
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"SPDisplaysDataType":[{"sppci_model":"Apple M2"}]}',
        )
        detector = HardwareDetector()
        gpus = detector.detect_gpus()
        assert len(gpus) == 1
        assert gpus[0].name == "Apple M2"
        assert gpus[0].vendor == "apple"

    @patch("platform.system", return_value="Unknown")
    def test_detect_gpus_unknown_os(self, mock_system):
        detector = HardwareDetector()
        assert detector.detect_gpus() == []


class TestHardwareDetectorDisk:
    """磁盘检测。"""

    @patch("psutil.disk_usage")
    def test_detect_disk(self, mock_usage):
        mock_usage.return_value = MagicMock(free=100 * 1024 * 1024)
        detector = HardwareDetector()
        disk = detector.detect_disk()
        assert disk.free_mb == 100

    @patch("psutil.disk_usage", side_effect=Exception("fail"))
    def test_detect_disk_fallback(self, mock_usage):
        detector = HardwareDetector()
        disk = detector.detect_disk()
        assert disk.free_mb == 0


class TestHardwareDetectorNetwork:
    """网络检测。"""

    def test_detect_network(self):
        detector = HardwareDetector()
        net = detector.detect_network()
        assert net.estimated_mbps == 0.0


class TestHardwareDetectorAll:
    """完整硬件信息检测。"""

    @patch.object(HardwareDetector, "detect_cpu")
    @patch.object(HardwareDetector, "detect_gpus")
    @patch.object(HardwareDetector, "detect_disk")
    @patch.object(HardwareDetector, "detect_network")
    @patch("psutil.virtual_memory")
    def test_detect_all(self, mock_vm, mock_net, mock_disk, mock_gpus, mock_cpu):
        mock_cpu.return_value = CPUInfo(
            logical_count=8, physical_count=4, freq_mhz=3200.0, brand="Intel i7"
        )
        mock_gpus.return_value = [
            GPUInfo(
                index=0,
                name="RTX 4090",
                vram_total_mb=24564,
                vram_free_mb=20000,
                vendor="nvidia",
            ),
        ]
        mock_disk.return_value = DiskInfo(free_mb=102400)
        mock_net.return_value = NetworkInfo(estimated_mbps=1000.0)
        mock_vm.return_value = MagicMock(total=32 * 1024**3, available=24 * 1024**3)

        detector = HardwareDetector()
        info = detector.detect_all()
        assert info["cpu"]["logical_count"] == 8
        assert info["cpu"]["brand"] == "Intel i7"
        assert len(info["gpus"]) == 1
        assert info["gpus"][0]["vendor"] == "nvidia"
        assert info["memory_total_mb"] == 32 * 1024
        assert info["disk_free_mb"] == 102400
        assert info["network_mbps"] == 1000.0


class TestCPUBrandDetection:
    """CPU 品牌名跨平台检测。"""

    @patch("subprocess.run")
    @patch("platform.system", return_value="Windows")
    def test_get_cpu_brand_windows(self, mock_system, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Name=Intel(R) Core(TM) i7-9700K CPU @ 3.60GHz\n",
        )
        detector = HardwareDetector()
        assert detector._get_cpu_brand() == "Intel(R) Core(TM) i7-9700K CPU @ 3.60GHz"

    @patch("builtins.open")
    @patch("platform.system", return_value="Linux")
    def test_get_cpu_brand_linux(self, mock_system, mock_open):
        mock_open.return_value.__enter__.return_value = iter([
            "processor\t: 0\n",
            "model name\t: AMD Ryzen 9 5900X\n",
        ])
        detector = HardwareDetector()
        assert detector._get_cpu_brand() == "AMD Ryzen 9 5900X"

    @patch("subprocess.run")
    @patch("platform.system", return_value="Darwin")
    def test_get_cpu_brand_darwin(self, mock_system, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="Apple M2\n")
        detector = HardwareDetector()
        assert detector._get_cpu_brand() == "Apple M2"
