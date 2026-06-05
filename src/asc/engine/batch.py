"""请求批处理模块。

提供：
- 按模型分组的请求合并
- 动态批处理窗口（最大等待时间）
- 批处理结果拆分与返回
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from asc.engine.base import InferenceRequest, InferenceResult


@dataclass
class BatchedRequest:
    """已加入批处理队列的请求。"""

    request: InferenceRequest
    model_id: str
    submit_time: float = field(default_factory=time.time)
    callback: Callable[[InferenceResult], None] | None = None


@dataclass
class BatchConfig:
    """批处理配置。"""

    max_batch_size: int = 8
    max_wait_ms: float = 10.0
    max_prompt_length: int = 32000


class BatchProcessor:
    """请求批处理器。

    将多个相同模型的请求合并为一批处理，提高吞吐量。
    """

    def __init__(self, config: BatchConfig | None = None) -> None:
        self._config = config or BatchConfig()
        self._pending: list[BatchedRequest] = []

    @property
    def pending_count(self) -> int:
        """当前待处理请求数。"""
        return len(self._pending)

    def submit(
        self,
        request: InferenceRequest,
        model_id: str,
        callback: Callable[[InferenceResult], None] | None = None,
    ) -> None:
        """提交请求到批处理队列。"""
        self._pending.append(
            BatchedRequest(
                request=request,
                model_id=model_id,
                callback=callback,
            )
        )

    def should_flush(self) -> bool:
        """判断是否应该立即处理当前批次。"""
        if not self._pending:
            return False

        if len(self._pending) >= self._config.max_batch_size:
            return True

        oldest = min(r.submit_time for r in self._pending)
        return (time.time() - oldest) * 1000 >= self._config.max_wait_ms

    def flush(
        self,
        infer_fn: Callable[[list[InferenceRequest]], list[InferenceResult]],
    ) -> list[InferenceResult]:
        """执行批处理推理并返回结果。

        Args:
            infer_fn: 接收请求列表，返回结果列表的推理函数

        Returns:
            与 pending 请求顺序对应的结果列表
        """
        if not self._pending:
            return []

        batch = self._pending[: self._config.max_batch_size]
        self._pending = self._pending[self._config.max_batch_size :]

        requests = [r.request for r in batch]
        results = infer_fn(requests)

        for batched, result in zip(batch, results, strict=True):
            if batched.callback is not None:
                batched.callback(result)

        return results

    def group_by_model(self) -> dict[str, list[BatchedRequest]]:
        """将待处理请求按模型分组。"""
        groups: dict[str, list[BatchedRequest]] = {}
        for req in self._pending:
            groups.setdefault(req.model_id, []).append(req)
        return groups

    def clear(self) -> list[BatchedRequest]:
        """清空所有待处理请求并返回。"""
        pending = self._pending[:]
        self._pending.clear()
        return pending
