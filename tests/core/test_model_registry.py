"""ModelRegistry 测试。"""

from __future__ import annotations

import pytest

from asc.core.model_registry import ModelLocation, ModelRegistry, NodeStorage


class TestModelLocation:
    def test_model_location_creation(self):
        loc = ModelLocation(
            node_id="n1", path="/models/m1.gguf", size_mb=100, network_latency_ms=5.0
        )
        assert loc.node_id == "n1"
        assert loc.path == "/models/m1.gguf"
        assert loc.size_mb == 100
        assert loc.network_latency_ms == 5.0


class TestNodeStorage:
    def test_free_mb(self):
        node = NodeStorage(node_id="n1", total_mb=1000, used_mb=300)
        assert node.free_mb == 700

    def test_can_store(self):
        node = NodeStorage(node_id="n1", total_mb=1000, used_mb=300)
        assert node.can_store(700)
        assert not node.can_store(701)


class TestModelRegistry:
    def test_register_node(self):
        reg = ModelRegistry()
        reg.register_node("n1", total_mb=1000)
        storage = reg.get_storage("n1")
        assert storage is not None
        assert storage.total_mb == 1000

    def test_unregister_node(self):
        reg = ModelRegistry()
        reg.register_node("n1", total_mb=1000)
        reg.unregister_node("n1")
        assert reg.get_storage("n1") is None

    def test_register_model(self):
        reg = ModelRegistry()
        reg.register_node("n1", total_mb=1000)
        reg.register("n1", "m1", "/models/m1.gguf", 100)

        locations = reg.find_model("m1")
        assert len(locations) == 1
        assert locations[0].node_id == "n1"
        assert locations[0].size_mb == 100

    def test_register_model_without_node_raises(self):
        reg = ModelRegistry()
        with pytest.raises(ValueError, match="未注册"):
            reg.register("n1", "m1", "/models/m1.gguf", 100)

    def test_unregister_model(self):
        reg = ModelRegistry()
        reg.register_node("n1", total_mb=1000)
        reg.register("n1", "m1", "/models/m1.gguf", 100)
        reg.unregister("n1", "m1")
        assert reg.find_model("m1") == []

    def test_find_model_multiple_nodes(self):
        reg = ModelRegistry()
        reg.register_node("n1", total_mb=1000)
        reg.register_node("n2", total_mb=1000)
        reg.register("n1", "m1", "/models/m1.gguf", 100)
        reg.register("n2", "m1", "/models/m1.gguf", 100)

        locations = reg.find_model("m1")
        assert len(locations) == 2

    def test_get_best_source_local(self):
        reg = ModelRegistry()
        reg.register_node("n1", total_mb=1000)
        reg.register_node("n2", total_mb=1000)
        reg.register("n1", "m1", "/models/m1.gguf", 100)
        reg.register("n2", "m1", "/models/m1.gguf", 100)

        best = reg.get_best_source("m1", requester="n1")
        assert best is not None
        assert best.node_id == "n1"  # 优先本地

    def test_get_best_source_by_latency(self):
        latency_map = {
            ("n1", "n2"): 10.0,
            ("n1", "n3"): 5.0,
        }

        def latency_fn(a: str, b: str) -> float:
            return latency_map.get((a, b), 0.0)

        reg = ModelRegistry(latency_fn=latency_fn)
        reg.register_node("n1", total_mb=1000)
        reg.register_node("n2", total_mb=1000)
        reg.register_node("n3", total_mb=1000)
        reg.register("n2", "m1", "/models/m1.gguf", 100)
        reg.register("n3", "m1", "/models/m1.gguf", 100)

        best = reg.get_best_source("m1", requester="n1")
        assert best is not None
        assert best.node_id == "n3"  # 延迟更低

    def test_get_best_source_not_found(self):
        reg = ModelRegistry()
        assert reg.get_best_source("m1", requester="n1") is None

    def test_list_node_models(self):
        reg = ModelRegistry()
        reg.register_node("n1", total_mb=1000)
        reg.register("n1", "m1", "/models/m1.gguf", 100)
        reg.register("n1", "m2", "/models/m2.gguf", 200)

        models = reg.list_node_models("n1")
        assert len(models) == 2

    def test_update_storage_usage(self):
        reg = ModelRegistry()
        reg.register_node("n1", total_mb=1000)
        reg.update_storage_usage("n1", 500)
        assert reg.get_storage("n1").used_mb == 500

    def test_all_models(self):
        reg = ModelRegistry()
        reg.register_node("n1", total_mb=1000)
        reg.register_node("n2", total_mb=1000)
        reg.register("n1", "m1", "/models/m1.gguf", 100)
        reg.register("n2", "m1", "/models/m1.gguf", 100)
        reg.register("n2", "m2", "/models/m2.gguf", 200)

        all_models = reg.all_models()
        assert len(all_models["m1"]) == 2
        assert len(all_models["m2"]) == 1
