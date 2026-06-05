"""LoadBalancer 测试。"""

from __future__ import annotations

from asc.scheduler.load_balancer import LoadBalancer


class TestLoadBalancer:
    def test_register_node(self):
        lb = LoadBalancer()
        lb.register_node("n1", compute_score=2.0)
        assert "n1" in lb._active_requests
        assert lb._node_scores["n1"] == 2.0

    def test_unregister_node(self):
        lb = LoadBalancer()
        lb.register_node("n1")
        lb.unregister_node("n1")
        assert "n1" not in lb._active_requests

    def test_start_finish_request(self):
        lb = LoadBalancer()
        lb.register_node("n1")
        lb.start_request("n1")
        assert lb._active_requests["n1"] == 1
        lb.finish_request("n1")
        assert lb._active_requests["n1"] == 0

    def test_round_robin(self):
        lb = LoadBalancer(strategy="round_robin")
        lb.register_node("n1")
        lb.register_node("n2")

        s1 = lb.select(["n1", "n2"])
        s2 = lb.select(["n1", "n2"])
        s3 = lb.select(["n1", "n2"])

        assert s1.node_id == "n1"
        assert s2.node_id == "n2"
        assert s3.node_id == "n1"
        assert s1.reason == "round_robin"

    def test_least_connections(self):
        lb = LoadBalancer(strategy="least_connections")
        lb.register_node("n1")
        lb.register_node("n2")
        lb.start_request("n1")

        s = lb.select(["n1", "n2"])
        assert s.node_id == "n2"  # n2 连接更少
        assert s.reason == "least_connections"

    def test_weighted(self):
        lb = LoadBalancer(strategy="weighted")
        lb.register_node("n1", compute_score=10.0)
        lb.register_node("n2", compute_score=1.0)

        # 多次选择，n1 应被选中更多次
        counts = {"n1": 0, "n2": 0}
        for _ in range(100):
            s = lb.select(["n1", "n2"])
            counts[s.node_id] += 1

        assert counts["n1"] > counts["n2"]

    def test_locality_fallback(self):
        lb = LoadBalancer(strategy="locality")
        lb.register_node("n1")
        lb.register_node("n2")

        s = lb.select(["n1", "n2"])
        # 无集群状态时 fallback 到 least_connections
        assert s.reason == "least_connections"

    def test_select_empty_candidates(self):
        lb = LoadBalancer()
        assert lb.select([]) is None

    def test_select_unregistered_candidates(self):
        lb = LoadBalancer()
        lb.register_node("n1")
        assert lb.select(["n2"]) is None

    def test_rebalance_plan_no_imbalance(self):
        lb = LoadBalancer()
        lb.register_node("n1", compute_score=1.0)
        lb.register_node("n2", compute_score=1.0)
        lb.start_request("n1")
        lb.start_request("n2")

        plan = lb.rebalance_plan(threshold_ratio=3.0)
        assert plan == []

    def test_rebalance_plan_with_imbalance(self):
        lb = LoadBalancer()
        lb.register_node("n1", compute_score=1.0)
        lb.register_node("n2", compute_score=1.0)
        for _ in range(10):
            lb.start_request("n1")
        lb.start_request("n2")

        plan = lb.rebalance_plan(threshold_ratio=2.0)
        assert len(plan) == 1
        assert plan[0] == ("n1", "n2")

    def test_rebalance_plan_single_node(self):
        lb = LoadBalancer()
        lb.register_node("n1")
        plan = lb.rebalance_plan()
        assert plan == []
