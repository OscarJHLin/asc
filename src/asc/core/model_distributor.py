"""Asc 模型分发器。

将本地模型文件推送到集群中的 Worker 节点：
- 通过内网分片传输模型文件
- 支持断点续传
- 进度追踪
- 多节点并行分发
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator

from asc.network.sync import (
    compute_file_sha256,
    read_chunk,
    split_file_into_chunks,
)
from asc.worker.agent import NodeResources


@dataclass(frozen=True)
class DistributionTarget:
    """分发目标节点。"""

    node_id: str
    ip: str
    port: int
    resources: NodeResources | None = None


@dataclass(frozen=True)
class DistributionResult:
    """分发结果。"""

    model_id: str
    target_node_id: str
    success: bool
    file_path: str
    file_size_mb: int
    error: str = ""


class ModelDistributor:
    """模型分发器。

    将本地模型文件推送到集群中的 Worker 节点。
    使用 ModelSyncProtocol 进行分片传输。
    """

    def __init__(
        self,
        models_dir: Path,
        bandwidth_limit_mbps: float = 100.0,
    ) -> None:
        self._models_dir = models_dir
        self._bandwidth_limit = bandwidth_limit_mbps

    def distribute(
        self,
        model_id: str,
        targets: list[DistributionTarget],
        send_chunk_fn: Any | None = None,
    ) -> Generator[dict[str, Any], None, list[DistributionResult]]:
        """将模型分发到目标节点。

        Args:
            model_id: 模型 ID（对应 models_dir 下的 <model_id>.gguf）
            targets: 目标节点列表
            send_chunk_fn: 发送分片到远程节点的回调函数
                签名: (node_id, chunk_info, chunk_data) -> bool

        Yields:
            进度信息 {"node_id": str, "stage": str, "progress": float, "detail": str}

        Returns:
            每个节点的分发结果列表
        """
        # 查找模型文件
        model_path = self._find_model_file(model_id)
        if model_path is None:
            return [
                DistributionResult(
                    model_id=model_id,
                    target_node_id=t.node_id,
                    success=False,
                    file_path="",
                    file_size_mb=0,
                    error=f"未找到模型文件: {model_id}",
                )
                for t in targets
            ]

        file_size_mb = int(model_path.stat().st_size // (1024 * 1024))

        # 准备分片
        yield {
            "stage": "preparing",
            "progress": 0.0,
            "detail": f"准备分片: {model_path.name} ({file_size_mb} MB)",
        }
        chunks = split_file_into_chunks(model_path)
        file_sha256 = compute_file_sha256(model_path)  # noqa: F841 - 用于未来校验

        results: list[DistributionResult] = []

        for i, target in enumerate(targets):
            node_progress = (i + 1) / len(targets)
            yield {
                "node_id": target.node_id,
                "stage": "distributing",
                "progress": node_progress * 0.5,
                "detail": f"分发到 {target.node_id} ({target.ip}:{target.port})",
            }

            if send_chunk_fn is None:
                # 无发送回调，标记为需要手动同步
                results.append(
                    DistributionResult(
                        model_id=model_id,
                        target_node_id=target.node_id,
                        success=False,
                        file_path=str(model_path),
                        file_size_mb=file_size_mb,
                        error="未配置发送回调，无法实际传输",
                    )
                )
                continue

            # 逐片发送
            success = True
            error_msg = ""
            for j, chunk in enumerate(chunks):
                chunk_progress = (j + 1) / len(chunks)
                yield {
                    "node_id": target.node_id,
                    "stage": "transferring",
                    "progress": node_progress * 0.5 + chunk_progress * 0.5 / len(targets),
                    "detail": f"分片 {j + 1}/{len(chunks)} -> {target.node_id}",
                }

                # 读取分片数据
                data = read_chunk(model_path, chunk)
                ok = send_chunk_fn(target.node_id, chunk, data)
                if not ok:
                    success = False
                    error_msg = f"分片 {chunk.chunk_index} 传输失败"
                    break

            results.append(
                DistributionResult(
                    model_id=model_id,
                    target_node_id=target.node_id,
                    success=success,
                    file_path=str(model_path),
                    file_size_mb=file_size_mb,
                    error=error_msg,
                )
            )

            if success:
                yield {
                    "node_id": target.node_id,
                    "stage": "complete",
                    "progress": node_progress,
                    "detail": f"分发完成: {target.node_id}",
                }

        return results

    def list_local_models(self) -> list[dict[str, Any]]:
        """列出本地可分发的模型。"""
        models: list[dict[str, Any]] = []
        if not self._models_dir.exists():
            return models

        for gguf in sorted(self._models_dir.glob("*.gguf")):
            size_mb = int(gguf.stat().st_size // (1024 * 1024))
            models.append({
                "model_id": gguf.stem,
                "file_path": str(gguf.resolve()),
                "file_size_mb": size_mb,
            })

        return models

    def _find_model_file(self, model_id: str) -> Path | None:
        """查找模型文件。"""
        # 精确匹配
        exact = self._models_dir / f"{model_id}.gguf"
        if exact.exists():
            return exact

        # 模糊匹配（忽略大小写）
        for gguf in self._models_dir.glob("*.gguf"):
            if gguf.stem.lower() == model_id.lower():
                return gguf

        return None
