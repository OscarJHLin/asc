"""测试监控与指标收集模块。"""

from asc.core.monitoring import (
    Counter,
    Gauge,
    HealthChecker,
    HealthStatus,
    Histogram,
    MetricsCollector,
)


class TestCounter:
    """计数器指标测试。"""

    def test_inc_default(self):
        c = Counter(name="test_counter", description="A test counter")
        c.inc()
        assert c.value == 1

    def test_inc_with_amount(self):
        c = Counter(name="test_counter", description="A test counter")
        c.inc(5)
        assert c.value == 5

    def test_inc_multiple_times(self):
        c = Counter(name="test_counter", description="A test counter")
        c.inc(2)
        c.inc(3)
        assert c.value == 5

    def test_initial_value(self):
        c = Counter(name="test_counter", description="A test counter")
        assert c.value == 0


class TestHistogram:
    """直方图指标测试。"""

    def test_observe_single_value(self):
        h = Histogram(name="test_histogram", description="A test histogram")
        h.observe(42.0)
        assert h.count == 1
        assert h.sum == 42.0
        assert h.avg == 42.0

    def test_observe_multiple_values(self):
        h = Histogram(name="test_histogram", description="A test histogram")
        h.observe(10.0)
        h.observe(20.0)
        h.observe(30.0)
        assert h.count == 3
        assert h.sum == 60.0
        assert h.avg == 20.0

    def test_avg_empty(self):
        h = Histogram(name="test_histogram", description="A test histogram")
        assert h.avg == 0.0

    def test_sum_empty(self):
        h = Histogram(name="test_histogram", description="A test histogram")
        assert h.sum == 0.0

    def test_count_empty(self):
        h = Histogram(name="test_histogram", description="A test histogram")
        assert h.count == 0

    def test_bucket_counts(self):
        h = Histogram(
            name="test_histogram",
            description="A test histogram",
            buckets=[10, 50, 100],
        )
        h.observe(5.0)
        h.observe(20.0)
        h.observe(80.0)
        h.observe(150.0)
        buckets = h.bucket_counts()
        assert buckets["le_10"] == 1
        assert buckets["le_50"] == 2
        assert buckets["le_100"] == 3
        assert buckets["le_inf"] == 4

    def test_default_buckets(self):
        h = Histogram(name="test_histogram", description="A test histogram")
        assert h.buckets == [10, 50, 100, 500, 1000, 5000]


class TestGauge:
    """仪表盘指标测试。"""

    def test_set_value(self):
        g = Gauge(name="test_gauge", description="A test gauge")
        g.set(3.14)
        assert g.value == 3.14

    def test_set_overwrite(self):
        g = Gauge(name="test_gauge", description="A test gauge")
        g.set(1.0)
        g.set(2.0)
        assert g.value == 2.0

    def test_initial_value(self):
        g = Gauge(name="test_gauge", description="A test gauge")
        assert g.value == 0.0


class TestMetricsCollector:
    """指标收集器测试。"""

    def test_counter_creation_and_retrieval(self):
        collector = MetricsCollector()
        c1 = collector.counter("requests_total", "Total requests")
        c2 = collector.counter("requests_total")
        assert c1 is c2
        assert c1.name == "requests_total"

    def test_histogram_creation_and_retrieval(self):
        collector = MetricsCollector()
        h1 = collector.histogram("latency_seconds", "Request latency")
        h2 = collector.histogram("latency_seconds")
        assert h1 is h2
        assert h1.name == "latency_seconds"

    def test_gauge_creation_and_retrieval(self):
        collector = MetricsCollector()
        g1 = collector.gauge("memory_usage", "Memory usage")
        g2 = collector.gauge("memory_usage")
        assert g1 is g2
        assert g1.name == "memory_usage"

    def test_to_prometheus_counter(self):
        collector = MetricsCollector()
        c = collector.counter("requests_total", "Total requests")
        c.inc(5)
        output = collector.to_prometheus()
        assert "# HELP requests_total Total requests" in output
        assert "# TYPE requests_total counter" in output
        assert "requests_total 5" in output

    def test_to_prometheus_histogram(self):
        collector = MetricsCollector()
        h = collector.histogram("latency_seconds", "Request latency")
        h.observe(25.0)
        h.observe(75.0)
        output = collector.to_prometheus()
        assert "# HELP latency_seconds Request latency" in output
        assert "# TYPE latency_seconds histogram" in output
        assert "latency_seconds_sum 100.0" in output
        assert "latency_seconds_count 2" in output
        assert "latency_seconds_bucket" in output

    def test_to_prometheus_gauge(self):
        collector = MetricsCollector()
        g = collector.gauge("memory_usage", "Memory usage")
        g.set(1024.0)
        output = collector.to_prometheus()
        assert "# HELP memory_usage Memory usage" in output
        assert "# TYPE memory_usage gauge" in output
        assert "memory_usage 1024.0" in output

    def test_to_prometheus_multiple_metrics(self):
        collector = MetricsCollector()
        collector.counter("c1", "Counter 1").inc(1)
        collector.gauge("g1", "Gauge 1").set(2.0)
        output = collector.to_prometheus()
        lines = output.splitlines()
        assert len(lines) == 6
        assert "c1 1" in lines
        assert "g1 2.0" in lines


class TestHealthChecker:
    """健康检查器测试。"""

    def test_healthy_status(self):
        checker = HealthChecker()
        status = checker.check(nodes_online=3, nodes_total=3, models_loaded=5)
        assert isinstance(status, HealthStatus)
        assert status.status == "healthy"
        assert status.nodes_online == 3
        assert status.nodes_total == 3
        assert status.models_loaded == 5

    def test_degraded_status(self):
        checker = HealthChecker()
        status = checker.check(nodes_online=2, nodes_total=3, models_loaded=5)
        assert status.status == "degraded"

    def test_unhealthy_status(self):
        checker = HealthChecker()
        status = checker.check(nodes_online=0, nodes_total=3, models_loaded=0)
        assert status.status == "unhealthy"

    def test_active_requests_tracking(self):
        checker = HealthChecker()
        checker.start_request()
        checker.start_request()
        status = checker.check(nodes_online=1, nodes_total=1, models_loaded=1)
        assert status.active_requests == 2
        checker.finish_request()
        status = checker.check(nodes_online=1, nodes_total=1, models_loaded=1)
        assert status.active_requests == 1

    def test_finish_request_does_not_go_negative(self):
        checker = HealthChecker()
        checker.finish_request()
        status = checker.check(nodes_online=1, nodes_total=1, models_loaded=1)
        assert status.active_requests == 0

    def test_uptime_increases(self):
        checker = HealthChecker()
        import time

        status1 = checker.check(nodes_online=1, nodes_total=1, models_loaded=1)
        time.sleep(0.01)
        status2 = checker.check(nodes_online=1, nodes_total=1, models_loaded=1)
        assert status2.uptime_seconds > status1.uptime_seconds
