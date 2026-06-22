"""ASC Cluster Link Protocol - 任务分派协议。

定义推理任务从 Master 分派到 Worker 的完整生命周期：
1. Master -> Worker: TaskDispatchMessage (分派任务)
2. Worker -> Master: TaskAcceptMessage (接受/拒绝)
3. Worker -> Master: TaskProgressMessage (进度上报，可选)
4. Worker -> Master: TaskResultMessage (返回结果)
5. Master -> Worker: TaskCancelMessage (取消任务)

支持并行推理：同一节点可同时处理多个任务。
"""

from __future__ import annotations

from dataclasses import dataclass

from asc.types.state import TaskStatus


@dataclass(frozen=True)
class InferenceTask:
    """推理任务。"""

    task_id: str
    model_id: str
    prompt: str
    max_tokens: int
    temperature: float
    priority: int = 0
    stream: bool = False


@dataclass(frozen=True)
class TaskDispatchMessage:
    """任务分派消息: Master -> Worker。"""

    task: InferenceTask
    target_node_id: str
    instance_id: str

    def to_dict(self) -> dict:
        return {
            "task": {
                "task_id": self.task.task_id,
                "model_id": self.task.model_id,
                "prompt": self.task.prompt,
                "max_tokens": self.task.max_tokens,
                "temperature": self.task.temperature,
                "priority": self.task.priority,
                "stream": self.task.stream,
            },
            "target_node_id": self.target_node_id,
            "instance_id": self.instance_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TaskDispatchMessage:
        td = d["task"]
        return cls(
            task=InferenceTask(
                task_id=td["task_id"],
                model_id=td["model_id"],
                prompt=td["prompt"],
                max_tokens=td["max_tokens"],
                temperature=td["temperature"],
                priority=td.get("priority", 0),
                stream=td.get("stream", False),
            ),
            target_node_id=d["target_node_id"],
            instance_id=d["instance_id"],
        )


@dataclass(frozen=True)
class TaskAcceptMessage:
    """任务接受消息: Worker -> Master。"""

    task_id: str
    node_id: str
    estimated_latency_ms: float
    accepted: bool = True

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "node_id": self.node_id,
            "estimated_latency_ms": self.estimated_latency_ms,
            "accepted": self.accepted,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TaskAcceptMessage:
        return cls(
            task_id=d["task_id"],
            node_id=d["node_id"],
            estimated_latency_ms=d["estimated_latency_ms"],
            accepted=d.get("accepted", True),
        )


@dataclass(frozen=True)
class TaskProgressMessage:
    """任务进度消息: Worker -> Master (可选，流式推理时使用)。"""

    task_id: str
    node_id: str
    tokens_generated: int
    tokens_per_second: float
    progress_fraction: float

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "node_id": self.node_id,
            "tokens_generated": self.tokens_generated,
            "tokens_per_second": self.tokens_per_second,
            "progress_fraction": self.progress_fraction,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TaskProgressMessage:
        return cls(
            task_id=d["task_id"],
            node_id=d["node_id"],
            tokens_generated=d["tokens_generated"],
            tokens_per_second=d["tokens_per_second"],
            progress_fraction=d["progress_fraction"],
        )


@dataclass(frozen=True)
class TaskResultMessage:
    """任务结果消息: Worker -> Master。"""

    task_id: str
    node_id: str
    status: TaskStatus
    output_text: str
    tokens_generated: int
    tokens_per_second: float
    latency_ms: float
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "node_id": self.node_id,
            "status": self.status.value,
            "output_text": self.output_text,
            "tokens_generated": self.tokens_generated,
            "tokens_per_second": self.tokens_per_second,
            "latency_ms": self.latency_ms,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TaskResultMessage:
        return cls(
            task_id=d["task_id"],
            node_id=d["node_id"],
            status=TaskStatus(d["status"]),
            output_text=d["output_text"],
            tokens_generated=d["tokens_generated"],
            tokens_per_second=d["tokens_per_second"],
            latency_ms=d["latency_ms"],
            error=d.get("error", ""),
        )


@dataclass(frozen=True)
class TaskCancelMessage:
    """任务取消消息: Master -> Worker。"""

    task_id: str
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TaskCancelMessage:
        return cls(
            task_id=d["task_id"],
            reason=d.get("reason", ""),
        )
