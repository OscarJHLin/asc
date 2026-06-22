"""系统工具函数。

提取自 agent.py / rpc_server.py / llama_server.py / benchmark_score.py 的重复逻辑。
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def find_executable(name: str) -> Path | None:
    """在项目目录和环境变量路径中查找可执行文件。

    查找顺序：
    1. ASC_LLAMA_PATH 环境变量指定目录下的 {name}.exe / {name}
    2. 项目根目录下 llama.cpp/build/bin/Release/{name}.exe
    3. 项目根目录下 llama.cpp/build/bin/{name}
    4. 项目根目录下 llama.cpp/build-linux/bin/{name}
    5. 用户本地安装目录 (~/.local/bin, ~/bin)
    6. LM Studio 内置 llama-server (Windows)
    7. 系统 PATH（shutil.which）

    Args:
        name: 可执行文件名称（如 "rpc-server"、"llama-server"）。

    Returns:
        找到的可执行文件路径，未找到返回 None。
    """
    import sys

    project_root = Path(__file__).parent.parent.parent.parent.resolve()
    home = Path.home()
    candidates = [
        project_root / "llama.cpp" / "build" / "bin" / "Release" / f"{name}.exe",
        project_root / "llama.cpp" / "build" / "bin" / name,
        project_root / "llama.cpp" / "build-linux" / "bin" / name,
    ]

    custom_path = os.getenv("ASC_LLAMA_PATH")
    if custom_path:
        candidates.insert(0, Path(custom_path) / f"{name}.exe")
        candidates.insert(1, Path(custom_path) / name)

    # Linux/macOS 用户本地安装路径（nohup/SSH 环境可能缺少 ~/.local/bin）
    if sys.platform != "win32":
        candidates.extend([
            home / ".local" / "bin" / name,
            home / "bin" / name,
        ])
    else:
        # Windows: LM Studio 内置的 llama-server
        lm_studio_base = home / ".lmstudio" / "extensions" / "backends"
        if lm_studio_base.exists():
            # 优先使用 CUDA 版本，按版本号降序排列
            cuda_dirs = sorted(
                [d for d in lm_studio_base.iterdir() if d.is_dir() and "cuda" in d.name.lower()],
                key=lambda d: d.name,
                reverse=True,
            )
            for d in cuda_dirs:
                candidates.append(d / f"{name}.exe")
            # 非 CUDA 版本
            non_cuda_dirs = sorted(
                [d for d in lm_studio_base.iterdir() if d.is_dir() and "cuda" not in d.name.lower()],
                key=lambda d: d.name,
                reverse=True,
            )
            for d in non_cuda_dirs:
                candidates.append(d / f"{name}.exe")

    for path in candidates:
        if path.exists():
            return path.resolve()

    found = shutil.which(name)
    if found:
        return Path(found)

    return None
