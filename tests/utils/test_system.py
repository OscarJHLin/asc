"""测试 utils/system.py 共享系统工具函数。"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from asc.utils.system import find_executable


class TestFindExecutable:
    """find_executable 可执行文件查找逻辑。"""

    def test_not_found(self):
        with (
            patch("asc.utils.system.shutil.which", return_value=None),
            patch("pathlib.Path.exists", return_value=False),
        ):
            result = find_executable("nonexistent-binary-xyz")
            assert result is None

    def test_find_via_system_path(self):
        with (
            patch("asc.utils.system.shutil.which", return_value="/usr/bin/python3"),
            patch("pathlib.Path.exists", return_value=False),
        ):
            result = find_executable("python3")
            assert result is not None
            assert "python3" in str(result)

    def test_returns_none_when_nothing_found(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("asc.utils.system.shutil.which", return_value=None),
            patch("pathlib.Path.exists", return_value=False),
        ):
            result = find_executable("missing-tool")
            assert result is None

    def test_custom_path_takes_priority_over_project(self, tmp_path: Path):
        custom_dir = tmp_path / "custom"
        custom_dir.mkdir()
        exe_path = custom_dir / "llama-server.exe"
        exe_path.write_text("fake")

        with (
            patch.dict(os.environ, {"ASC_LLAMA_PATH": str(custom_dir)}),
            # 让项目内路径不存在，确保走自定义路径
            patch("pathlib.Path.exists", side_effect=lambda: "custom" in str(self)),
        ):
            pass

        # 简化：直接验证 env var 路径被优先检查
        with (
            patch.dict(os.environ, {"ASC_LLAMA_PATH": str(custom_dir)}),
            patch("asc.utils.system.shutil.which", return_value=None),
        ):
            # exists 在自定义路径下返回 True

            def selective_exists(self):
                return bool("custom" in str(self) and "llama-server" in str(self))

            with patch.object(Path, "exists", selective_exists):
                result = find_executable("llama-server")
                assert result is not None
                assert "llama-server" in str(result)

    def test_project_build_release_path(self, tmp_path: Path):
        exe_path = tmp_path / "llama.cpp" / "build" / "bin" / "Release" / "rpc-server.exe"
        exe_path.parent.mkdir(parents=True, exist_ok=True)
        exe_path.write_text("fake")

        with patch("asc.utils.system.shutil.which", return_value=None):

            def selective_exists(self):
                return bool("rpc-server" in str(self) and "Release" in str(self))

            with patch.object(Path, "exists", selective_exists):
                result = find_executable("rpc-server")
                assert result is not None
                assert "rpc-server" in str(result)

    def test_project_build_bin_path(self, tmp_path: Path):
        with patch("asc.utils.system.shutil.which", return_value=None):

            def selective_exists(self):
                # 匹配 build/bin/llama-server（非 Release）
                s = str(self)
                return bool(
                    "llama-server" in s and "build" in s
                    and "bin" in s and "Release" not in s
                )

            with patch.object(Path, "exists", selective_exists):
                result = find_executable("llama-server")
                assert result is not None
                assert "llama-server" in str(result)

    def test_project_build_linux_path(self, tmp_path: Path):
        with patch("asc.utils.system.shutil.which", return_value=None):
            def selective_exists(self):
                s = str(self)
                return bool("build-linux" in s and "llama-server" in s)

            with patch.object(Path, "exists", selective_exists):
                result = find_executable("llama-server")
                assert result is not None
                assert "llama-server" in str(result)

    def test_returns_path_object(self):
        with (
            patch("asc.utils.system.shutil.which", return_value="/usr/bin/test-bin"),
            patch("pathlib.Path.exists", return_value=False),
        ):
            result = find_executable("test-bin")
            assert isinstance(result, Path)
