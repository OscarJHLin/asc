"""ASC Cluster Link Protocol - 节点容量协议。

定义节点实时容量上报机制：
1. Worker -> Master: CapacityReport (周期性主动上报)
2. Master -> Worker: CapacityQuery (按需查询)
3. Worker -> Master: CapacityResponse (查询响应)

容量信息用于：
- 算力平衡：高算力节点承担更多层
- 动态调度：根据实时负载分配任务
- 并行推理：判断节点可用并发槽位
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GpuMetrics:
    """GPU 指标。"""

    index: int
    name: str
    vram_total_mb: int
    vram_free_mb: int
    utilization_percent: float
    temperature_c: int

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "name": self.name,
            "vram_total_mb": self.vram_total_mb,
            "vram_free_mb": self.vram_free_mb,
            "utilization_percent": self.utilization_percent,
            "temperature_c": self.temperature_c,
        }

    @classmethod
    def from_dict(cls, d: dict) -> GpuMetrics:
        return cls(
            index=d["index"],
            name=d["name"],
            vram_total_mb=d["vram_total_mb"],
            vram_free_mb=d["vram_free_mb"],
            utilization_percent=d["utilization_percent"],
            temperature_c=d["temperature_c"],
        )


@dataclass(frozen=True)
class CapacityMetrics:
    """节点容量指标。"""

    cpu_count: int
    cpu_percent: float
    memory_total_mb: int
    memory_free_mb: int
    gpus: list[GpuMetrics]
    active_tasks: int
    max_concurrent_tasks: int
    compute_score: float
    network_latency_ms: float

    @property
    def available_slots(self) -> int:
        """可用并发槽位。"""
        return max(0, self.max_concurrent_tasks - self.active_tasks)

    @property
    def total_vram_free_mb(self) -> int:
        """总空闲 VRAM。"""
        return sum(g.vram_free_mb for g in self.gpus)

    def to_dict(self) -> dict:
        return {
            "cpu_count": self.cpu_count,
            "cpu_percent": self.cpu_percent,
            "memory_total_mb": self.memory_total_mb,
            "memory_free_mb": self.memory_free_mb,
            "gpus": [g.to_dict() for g in self.gpus],
            "active_tasks": self.active_tasks,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "compute_score": self.compute_score,
            "network_latency_ms": self.network_latency_ms,
        }

    @classmethod
    def from_dict(cls, d: dict) -> CapacityMetrics:
        return cls(
            cpu_count=d["cpu_count"],
            cpu_percent=d["cpu_percent"],
            memory_total_mb=d["memory_total_mb"],
            memory_free_mb=d["memory_free_mb"],
            gpus=[GpuMetrics.from_dict(g) for g in d.get("gpus", [])],
            active_tasks=d["active_tasks"],
            max_concurrent_tasks=d["max_concurrent_tasks"],
            compute_score=d["compute_score"],
            network_latency_ms=d["network_latency_ms"],
        )


@dataclass(frozen=True)
class CapacityReport:
    """容量上报消息: Worker -> Master (周期性)。"""

    node_id: str
    metrics: CapacityMetrics

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "metrics": self.metrics.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> CapacityReport:
        return cls(
            node_id=d["node_id"],
            metrics=CapacityMetrics.from_dict(d["metrics"]),
        )


@dataclass(frozen=True)
class CapacityQuery:
    """容量查询消息: Master -> Worker。"""

    requester_id: str

    def to_dict(self) -> dict:
        return {"requester_id": self.requester_id}

    @classmethod
    def from_dict(cls, d: dict) -> CapacityQuery:
        return cls(requester_id=d["requester_id"])


@dataclass(frozen=True)
class CapacityResponse:
    """容量响应消息: Worker -> Master。"""

    node_id: str
    metrics: CapacityMetrics

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "metrics": self.metrics.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> CapacityResponse:
        return cls(
            node_id=d["node_id"],
            metrics=CapacityMetrics.from_dict(d["metrics"]),
        )
