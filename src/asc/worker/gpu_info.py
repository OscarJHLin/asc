"""GPU 信息值对象。

从 agent.py 独立出来，避免循环导入。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GPUInfo:
    """GPU 信息。"""

    index: int
    name: str
    vram_total_mb: int
    vram_free_mb: int
    vendor: str = ""
    compute_capability: str = ""

    @property
    def vram_used_mb(self) -> int:
        return self.vram_total_mb - self.vram_free_mb
