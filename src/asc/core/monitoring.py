"""监控与指标收集模块。

提供可观测性基础设施，支持性能监控、健康检查和指标导出。

核心组件：
- Counter：单调递增计数器，适合记录请求总数、错误次数等
- Histogram：直方图，记录数值分布（如延迟、响应大小），自动分桶
- Gauge：仪表盘，记录当前值（如在线节点数、活跃请求数）
- MetricsCollector：统一管理所有指标，支持 Prometheus 格式导出
- HealthChecker：基于节点状态计算集群健康等级

设计原则：
- 低开销：指标收集不应显著影响主流程性能
- 防内存泄漏：Histogram 使用 deque(maxlen=10000)，自动丢弃旧样本
- 可导出：支持 Prometheus 文本格式，便于接入 Grafana 等监控平台

使用示例：
    collector = MetricsCollector()
    collector.counter("requests_total", "总请求数").inc()
    collector.histogram("request_latency_ms", "请求延迟").observe(45.2)
    prometheus_text = collector.to_prometheus()

健康检查：
    HealthChecker 根据在线节点比例判断集群状态：
    - 0 节点在线 -> unhealthy
    - 部分节点在线 -> degraded
    - 全部节点在线 -> healthy
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class Counter:
    """计数器指标。"""

    name: str
    description: str
    _value: int = field(default=0, repr=False)
    _labels: dict[str, str] = field(default_factory=dict, repr=False)

    def inc(self, amount: int = 1) -> None:
        """增加计数。"""
        self._value += amount

    @property
    def value(self) -> int:
        return self._value


@dataclass
class Histogram:
    """直方图指标（记录数值分布）。"""

    name: str
    description: str
    buckets: list[float] = field(default_factory=lambda: [10, 50, 100, 500, 1000, 5000])
    max_samples: int = 10000
    _values: deque[float] = field(default_factory=deque, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self._values, deque):
            self._values = deque(self._values, maxlen=self.max_samples)
        else:
            self._values = deque(self._values, maxlen=self.max_samples)

    def observe(self, value: float) -> None:
        """记录一个观测值。"""
        self._values.append(value)

    @property
    def count(self) -> int:
        return len(self._values)

    @property
    def sum(self) -> float:
        return sum(self._values)

    @property
    def avg(self) -> float:
        if not self._values:
            return 0.0
        return sum(self._values) / len(self._values)

    def bucket_counts(self) -> dict[str, int]:
        """返回各桶的计数。"""
        result: dict[str, int] = {}
        for bucket in self.buckets:
            result[f"le_{bucket}"] = sum(1 for v in self._values if v <= bucket)
        result["le_inf"] = len(self._values)
        return result


@dataclass
class Gauge:
    """仪表盘指标（当前值）。"""

    name: str
    description: str
    _value: float = field(default=0.0, repr=False)

    def set(self, value: float) -> None:
        """设置当前值。"""
        self._value = value

    @property
    def value(self) -> float:
        return self._value


class MetricsCollector:
    """指标收集器。

    统一管理所有指标，支持导出为 Prometheus 格式。
    """

    def __init__(self) -> None:
        self._counters: dict[str, Counter] = {}
        self._histograms: dict[str, Histogram] = {}
        self._gauges: dict[str, Gauge] = {}

    def counter(self, name: str, description: str = "") -> Counter:
        """获取或创建计数器。"""
        if name not in self._counters:
            self._counters[name] = Counter(name=name, description=description)
        return self._counters[name]

    def histogram(self, name: str, description: str = "") -> Histogram:
        """获取或创建直方图。"""
        if name not in self._histograms:
            self._histograms[name] = Histogram(name=name, description=description)
        return self._histograms[name]

    def gauge(self, name: str, description: str = "") -> Gauge:
        """获取或创建仪表盘。"""
        if name not in self._gauges:
            self._gauges[name] = Gauge(name=name, description=description)
        return self._gauges[name]

    def to_prometheus(self) -> str:
        """导出为 Prometheus 文本格式。"""
        lines: list[str] = []

        for c in self._counters.values():
            lines.append(f"# HELP {c.name} {c.description}")
            lines.append(f"# TYPE {c.name} counter")
            lines.append(f"{c.name} {c.value}")

        for h in self._histograms.values():
            lines.append(f"# HELP {h.name} {h.description}")
            lines.append(f"# TYPE {h.name} histogram")
            for bucket, count in h.bucket_counts().items():
                lines.append(f'{h.name}_bucket{{le="{bucket.replace("le_", "")}"}} {count}')
            lines.append(f"{h.name}_sum {h.sum}")
            lines.append(f"{h.name}_count {h.count}")

        for g in self._gauges.values():
            lines.append(f"# HELP {g.name} {g.description}")
            lines.append(f"# TYPE {g.name} gauge")
            lines.append(f"{g.name} {g.value}")

        return "\n".join(lines)


@dataclass
class HealthStatus:
    """健康状态。"""

    status: str  # "healthy" | "degraded" | "unhealthy"
    nodes_online: int
    nodes_total: int
    models_loaded: int
    uptime_seconds: float
    active_requests: int


class HealthChecker:
    """健康检查器。"""

    def __init__(self) -> None:
        self._start_time = time.time()
        self._active_requests = 0
        self._lock = asyncio.Lock()

    async def start_request(self) -> None:
        async with self._lock:
            self._active_requests += 1

    async def finish_request(self) -> None:
        async with self._lock:
            self._active_requests = max(0, self._active_requests - 1)

    def check(
        self,
        nodes_online: int,
        nodes_total: int,
        models_loaded: int,
    ) -> HealthStatus:
        """执行健康检查。"""
        if nodes_online == 0:
            status = "unhealthy"
        elif nodes_online < nodes_total:
            status = "degraded"
        else:
            status = "healthy"

        return HealthStatus(
            status=status,
            nodes_online=nodes_online,
            nodes_total=nodes_total,
            models_loaded=models_loaded,
            uptime_seconds=time.time() - self._start_time,
            active_requests=self._active_requests,
        )
