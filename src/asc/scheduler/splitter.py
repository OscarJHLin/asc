"""Asc 张量分割算法。

基于各节点空闲 VRAM 比例计算 --tensor-split 参数。
支持本地权重加成和网络惩罚系数。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TensorSplitResult:
    """张量分割结果。"""

    splits: list[float]
    rpc_endpoints: list[str]
    is_distributed: bool

    def format_tensor_split(self) -> str:
        """格式化为 llama-cli --tensor-split 参数。"""
        return ",".join(f"{s:.2f}" for s in self.splits)


class TensorSplitCalculator:
    """张量分割计算器。"""

    @staticmethod
    def calculate(
        local_vram_free_mb: int,
        workers_vram_free_mb: dict[str, int],
        worker_addresses: dict[str, str] | None = None,
        local_weight: float = 1.0,
        network_penalty: float = 0.95,
    ) -> TensorSplitResult:
        """计算张量分割方案。

        Args:
            local_vram_free_mb: 本地空闲 VRAM (MB)
            workers_vram_free_mb: Worker 空闲 VRAM 映射 {worker_id: vram_mb}
            worker_addresses: Worker RPC 地址映射 {worker_id: "ip:port"}
            local_weight: 本地权重加成（本地通信更快）
            network_penalty: 网络惩罚系数（0-1，越低惩罚越大）

        Returns:
            TensorSplitResult
        """
        worker_addresses = worker_addresses or {}

        # 过滤 VRAM 为 0 的 Worker
        active_workers = {k: v for k, v in workers_vram_free_mb.items() if v > 0}

        if not active_workers:
            return TensorSplitResult(
                splits=[1.0],
                rpc_endpoints=[],
                is_distributed=False,
            )

        # 计算加权 VRAM
        weighted_local = local_vram_free_mb * local_weight
        weighted_workers = {k: v * network_penalty for k, v in active_workers.items()}

        # 归一化
        all_weights = [weighted_local] + list(weighted_workers.values())
        total = sum(all_weights)
        if total == 0:
            splits = [1.0 / len(all_weights)] * len(all_weights)
        else:
            splits = [w / total for w in all_weights]

        # RPC 端点
        rpc_endpoints = [worker_addresses.get(wid, f"{wid}:50052") for wid in active_workers]

        return TensorSplitResult(
            splits=splits,
            rpc_endpoints=rpc_endpoints,
            is_distributed=True,
        )
