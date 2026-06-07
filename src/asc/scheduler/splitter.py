"""Asc 张量分割算法。

Tensor Parallelism（张量并行）将模型的每一层横向切分到多个节点，
每个节点负责一部分张量的计算。llama.cpp 通过 --tensor-split 参数
指定各节点承担的权重比例。

本模块基于各节点空闲 VRAM 比例计算 --tensor-split 参数，并支持：
- 本地权重加成（local_weight）：本地节点通信更快，可适当提高其比例
- 网络惩罚系数（network_penalty）：远程节点通信有延迟，适当降低其比例

使用场景：
    当单节点 VRAM 不足以容纳完整模型的一层时，将张量分布到多个节点，
    各节点同时计算部分结果，最后聚合。适合节点间带宽较高的环境。

与 Pipeline Parallelism 的区别：
    - Tensor：每层横向切分，通信量大（每层的激活值都要同步），但无气泡
    - Pipeline：按层纵向切分，通信量小，但存在流水线气泡
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

        算法步骤：
        1. 过滤 VRAM 为 0 的 Worker（无资源或不可用）
        2. 对本地 VRAM 乘以 local_weight 加成，对远程 VRAM 乘以 network_penalty 惩罚
        3. 将所有加权 VRAM 归一化为比例（总和为 1.0）
        4. 生成 RPC 端点列表供 llama-server 的 --rpc 参数使用

        Args:
            local_vram_free_mb: 本地空闲 VRAM (MB)
            workers_vram_free_mb: Worker 空闲 VRAM 映射 {worker_id: vram_mb}
            worker_addresses: Worker RPC 地址映射 {worker_id: "ip:port"}
            local_weight: 本地权重加成（默认 1.0，无加成；建议 1.1-1.3）
            network_penalty: 网络惩罚系数（0-1，默认 0.95；越低惩罚越大）

        Returns:
            TensorSplitResult，包含 splits（比例列表）和 rpc_endpoints

        边界情况：
            - 无活跃 Worker：返回单节点模式（splits=[1.0], is_distributed=False）
            - 总权重为 0：平均分配，避免除零错误

        示例：
            local_vram=16000, workers={"w1": 8000}, local_weight=1.2, penalty=0.9
            -> weighted_local=19200, weighted_w1=7200
            -> splits=[0.727, 0.273]
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

        # 计算加权 VRAM：本地节点通信更快，给予权重加成；
        # 远程节点受网络延迟影响，给予惩罚系数
        weighted_local = local_vram_free_mb * local_weight
        weighted_workers = {k: v * network_penalty for k, v in active_workers.items()}

        # 归一化：将加权 VRAM 转换为比例，总和严格为 1.0
        all_weights = [weighted_local] + list(weighted_workers.values())
        total = sum(all_weights)
        if total == 0:
            # 防御性编程：避免除零，平均分配
            splits = [1.0 / len(all_weights)] * len(all_weights)
        else:
            splits = [w / total for w in all_weights]

        # RPC 端点：按 active_workers 的顺序生成，与 splits 一一对应
        # 默认端口 50052 为 llama.cpp RPC Server 的标准端口
        rpc_endpoints = [worker_addresses.get(wid, f"{wid}:50052") for wid in active_workers]

        return TensorSplitResult(
            splits=splits,
            rpc_endpoints=rpc_endpoints,
            is_distributed=True,
        )
