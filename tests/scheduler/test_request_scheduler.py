"""RequestScheduler 测试。"""

from __future__ import annotations

from asc.scheduler.load_balancer import LoadBalancer
from asc.scheduler.request_scheduler import InferenceRequest, RequestScheduler


class TestRequestScheduler:
    def test_schedule_round_robin(self):
        lb = LoadBalancer(strategy="round_robin")
        lb.register_node("n1")
        lb.register_node("n2")
        scheduler = RequestScheduler(load_balancer=lb)

        req = InferenceRequest(
            request_id="r1", model_id="m1", prompt="hello"
        )
        result = scheduler.schedule(req, ["n1", "n2"])

        assert result is not None
        assert result.selected_node == "n1"
        assert result.strategy == "round_robin"
        assert result.request.request_id == "r1"

    def test_schedule_least_connections(self):
        lb = LoadBalancer(strategy="least_connections")
        lb.register_node("n1")
        lb.register_node("n2")
        lb.start_request("n1")
        scheduler = RequestScheduler(load_balancer=lb)

        req = InferenceRequest(
            request_id="r1", model_id="m1", prompt="hello"
        )
        result = scheduler.schedule(req, ["n1", "n2"])

        assert result is not None
        assert result.selected_node == "n2"

    def test_complete(self):
        lb = LoadBalancer(strategy="round_robin")
        lb.register_node("n1")
        scheduler = RequestScheduler(load_balancer=lb)

        req = InferenceRequest(
            request_id="r1", model_id="m1", prompt="hello"
        )
        scheduler.schedule(req, ["n1"])
        assert lb._active_requests["n1"] == 1

        scheduler.complete("n1")
        assert lb._active_requests["n1"] == 0

    def test_set_strategy(self):
        lb = LoadBalancer(strategy="round_robin")
        lb.register_node("n1")
        lb.register_node("n2")
        scheduler = RequestScheduler(load_balancer=lb)

        scheduler.set_strategy("least_connections")
        assert lb.strategy == "least_connections"

    def test_schedule_no_candidates(self):
        scheduler = RequestScheduler()
        req = InferenceRequest(
            request_id="r1", model_id="m1", prompt="hello"
        )
        result = scheduler.schedule(req, [])
        assert result is None

    def test_schedule_with_latency_estimator(self):
        lb = LoadBalancer(strategy="round_robin")
        lb.register_node("n1")
        scheduler = RequestScheduler(
            load_balancer=lb,
            latency_estimator=lambda n: 50.0 if n == "n1" else 100.0,
        )

        req = InferenceRequest(
            request_id="r1", model_id="m1", prompt="hello"
        )
        result = scheduler.schedule(req, ["n1"])

        assert result is not None
        assert result.estimated_latency_ms == 50.0
