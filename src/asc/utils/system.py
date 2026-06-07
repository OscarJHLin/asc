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
    5. 系统 PATH（shutil.which）

    Args:
        name: 可执行文件名称（如 "rpc-server"、"llama-server"）。

    Returns:
        找到的可执行文件路径，未找到返回 None。
    """
    project_root = Path(__file__).parent.parent.parent.parent.resolve()
    candidates = [
        project_root / "llama.cpp" / "build" / "bin" / "Release" / f"{name}.exe",
        project_root / "llama.cpp" / "build" / "bin" / name,
        project_root / "llama.cpp" / "build-linux" / "bin" / name,
    ]

    custom_path = os.getenv("ASC_LLAMA_PATH")
    if custom_path:
        candidates.insert(0, Path(custom_path) / f"{name}.exe")
        candidates.insert(1, Path(custom_path) / name)

    for path in candidates:
        if path.exists():
            return path.resolve()

    found = shutil.which(name)
    if found:
        return Path(found)

    return None
