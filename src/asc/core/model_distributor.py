"""Asc 模型分发器。

将本地模型文件推送到集群中的 Worker 节点：
- 通过内网分片传输模型文件
- 支持断点续传
- 进度追踪
- 多节点并行分发
- 滑动窗口并发传输，提升带宽利用率
- 单分片失败重试，不中断整个传输
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from asc.network.sync import (
    ChunkInfo,
    read_chunk,
    split_file_into_chunks,
)
from asc.worker.agent import NodeResources

# 异步发送回调类型
AsyncSendChunkFn = Callable[[str, ChunkInfo, bytes], Awaitable[bool]]
# 同步发送回调类型（向后兼容）
SyncSendChunkFn = Callable[[str, ChunkInfo, bytes], bool]

# 默认滑动窗口大小
DEFAULT_WINDOW_SIZE = 5
# 默认重试次数
DEFAULT_MAX_RETRIES = 3


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
    支持滑动窗口并发传输，提升带宽利用率。
    """

    def __init__(
        self,
        models_dir: Path,
        bandwidth_limit_mbps: float = 100.0,
    ) -> None:
        self._models_dir = models_dir
        self._bandwidth_limit = bandwidth_limit_mbps

    async def distribute(
        self,
        model_id: str,
        targets: list[DistributionTarget],
        send_chunk_fn: Any | None = None,
        on_progress: Any | None = None,
        window_size: int = DEFAULT_WINDOW_SIZE,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> list[DistributionResult]:
        """将模型分发到目标节点。

        Args:
            model_id: 模型 ID（对应 models_dir 下的 <model_id>.gguf）
            targets: 目标节点列表
            send_chunk_fn: 发送分片到远程节点的回调函数
                支持同步签名: (node_id, chunk_info, chunk_data) -> bool
                支持异步签名: (node_id, chunk_info, chunk_data) -> Awaitable[bool]
            on_progress: 进度回调函数
                签名: (progress_info: dict) -> None
            window_size: 滑动窗口大小（并发分片数），默认5
            max_retries: 单分片最大重试次数，默认3

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
        if on_progress:
            on_progress({
                "stage": "preparing",
                "progress": 0.0,
                "detail": f"准备分片: {model_path.name} ({file_size_mb} MB)",
            })
        chunks = split_file_into_chunks(model_path)

        # 包装发送回调为异步函数
        async_send = self._wrap_send_fn(send_chunk_fn)

        # 并行分发到所有目标节点
        async def _distribute_to_target(
            idx: int, target: DistributionTarget
        ) -> DistributionResult:
            if send_chunk_fn is None:
                return DistributionResult(
                    model_id=model_id,
                    target_node_id=target.node_id,
                    success=False,
                    file_path=str(model_path),
                    file_size_mb=file_size_mb,
                    error="未配置发送回调，无法实际传输",
                )

            # 滑动窗口并发传输
            failed_chunks: set[int] = set()
            pending: dict[int, asyncio.Task] = {}

            for chunk in chunks:
                # 等待窗口有空位
                while len(pending) >= window_size:
                    done, _ = await asyncio.wait(
                        pending.values(),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in done:
                        task_idx = next(k for k, v in pending.items() if v is task)
                        del pending[task_idx]
                        try:
                            if not task.result():
                                failed_chunks.add(task_idx)
                        except BaseException:
                            failed_chunks.add(task_idx)

                # 发送新分片
                data = read_chunk(model_path, chunk)
                task = asyncio.create_task(
                    async_send(target.node_id, chunk, data)
                )
                pending[chunk.chunk_index] = task

            # 等待剩余分片完成
            if pending:
                results = await asyncio.gather(*pending.values(), return_exceptions=True)
                for idx_r, result in zip(pending.keys(), results, strict=False):
                    if isinstance(result, BaseException) or not result:
                        failed_chunks.add(idx_r)

            # 重传失败分片
            for _attempt in range(max_retries):
                if not failed_chunks:
                    break
                to_retry = list(failed_chunks)
                failed_chunks.clear()
                retry_tasks: dict[int, asyncio.Task] = {}
                for chunk_idx in to_retry:
                    chunk = chunks[chunk_idx]
                    data = read_chunk(model_path, chunk)
                    task = asyncio.create_task(
                        async_send(target.node_id, chunk, data)
                    )
                    retry_tasks[chunk_idx] = task

                if retry_tasks:
                    retry_results = await asyncio.gather(
                        *retry_tasks.values(), return_exceptions=True
                    )
                    for idx_r, result in zip(retry_tasks.keys(), retry_results, strict=False):
                        if isinstance(result, BaseException) or not result:
                            failed_chunks.add(idx_r)

            success = len(failed_chunks) == 0
            error_msg = ""
            if not success:
                error_msg = f"分片 {sorted(failed_chunks)} 传输失败（已重试{max_retries}次）"

            return DistributionResult(
                model_id=model_id,
                target_node_id=target.node_id,
                success=success,
                file_path=str(model_path),
                file_size_mb=file_size_mb,
                error=error_msg,
            )

        # 使用 asyncio.gather 并行分发
        tasks = [
            _distribute_to_target(i, target)
            for i, target in enumerate(targets)
        ]
        results = await asyncio.gather(*tasks)

        # 报告进度
        if on_progress:
            for i, (target, result) in enumerate(zip(targets, results, strict=True)):
                node_progress = (i + 1) / len(targets)
                on_progress({
                    "node_id": target.node_id,
                    "stage": "distributing",
                    "progress": node_progress * 0.5,
                    "detail": (
                        f"分发到 {target.node_id} "
                        f"({target.ip}:{target.port})"
                    ),
                })
                if result.success:
                    on_progress({
                        "node_id": target.node_id,
                        "stage": "complete",
                        "progress": node_progress,
                        "detail": f"分发完成: {target.node_id}",
                    })

        return list(results)

    @staticmethod
    def _wrap_send_fn(
        send_chunk_fn: Any | None,
    ) -> AsyncSendChunkFn:
        """将发送回调统一包装为异步函数。"""
        if send_chunk_fn is None:
            async def _none(node_id: str, chunk: ChunkInfo, data: bytes) -> bool:
                return False
            return _none

        if asyncio.iscoroutinefunction(send_chunk_fn):
            return send_chunk_fn

        # 同步回调包装为异步
        fn = send_chunk_fn
        async def _sync_wrapper(node_id: str, chunk: ChunkInfo, data: bytes) -> bool:
            return fn(node_id, chunk, data)
        return _sync_wrapper

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
