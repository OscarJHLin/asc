"""测试管理员集群配置 API 端点。"""

from __future__ import annotations

import os

from fastapi.testclient import TestClient

from asc.api.auth import _init_default_store
from asc.api.server import create_app
from asc.core.cluster_config import ClusterConfig, ModelConfig, NodeResourcePolicy


def _make_client(config: ClusterConfig | None = None) -> TestClient:
    """创建带管理员认证的测试客户端。"""
    os.environ["ASC_ADMIN_API_KEY"] = "admin-test-key"
    os.environ["ASC_API_KEY"] = "user-test-key"
    os.environ.pop("ASC_ALLOW_NO_AUTH", None)
    _init_default_store()
    app = create_app(cluster_config=config or ClusterConfig())
    client = TestClient(app)
    return client


def _cleanup_env():
    """清理测试环境变量。"""
    os.environ.pop("ASC_ADMIN_API_KEY", None)
    os.environ.pop("ASC_API_KEY", None)
    os.environ.pop("ASC_ALLOW_NO_AUTH", None)
    _init_default_store()


ADMIN_HEADERS = {"X-API-Key": "admin-test-key"}
USER_HEADERS = {"X-API-Key": "user-test-key"}


class TestClusterConfigAPI:
    """测试集群配置管理端点。"""

    def teardown_method(self):
        _cleanup_env()

    def test_get_cluster_config(self):
        """GET /admin/cluster-config 返回完整配置。"""
        client = _make_client()
        resp = client.get("/admin/cluster-config", headers=ADMIN_HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert "default_resource_policy" in data
        assert "model_configs" in data
        assert "node_policies" in data
        assert "heartbeat_interval" in data

    def test_update_cluster_config(self):
        """PUT /admin/cluster-config 更新集群参数。"""
        client = _make_client()
        resp = client.put(
            "/admin/cluster-config",
            json={"heartbeat_interval": 10, "node_timeout": 60},
            headers=ADMIN_HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["config"]["heartbeat_interval"] == 10
        assert data["config"]["node_timeout"] == 60

    def test_cluster_config_requires_admin(self):
        """非管理员无法访问配置端点。"""
        client = _make_client()
        resp = client.get("/admin/cluster-config", headers=USER_HEADERS)
        assert resp.status_code == 403

    def test_cluster_config_requires_auth(self):
        """无认证无法访问配置端点。"""
        client = _make_client()
        resp = client.get("/admin/cluster-config")
        assert resp.status_code in (401, 403)


class TestResourcePolicyAPI:
    """测试资源策略端点。"""

    def teardown_method(self):
        _cleanup_env()

    def test_get_default_resource_policy(self):
        """GET /admin/resource-policy/default 返回默认策略。"""
        client = _make_client()
        resp = client.get("/admin/resource-policy/default", headers=ADMIN_HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert "memory_limit_mb" in data
        assert "vram_reserve_mb" in data

    def test_update_default_resource_policy(self):
        """PUT /admin/resource-policy/default 更新默认策略。"""
        client = _make_client()
        resp = client.put(
            "/admin/resource-policy/default",
            json={"memory_limit_mb": 32768, "memory_offload_ratio": 0.3},
            headers=ADMIN_HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["policy"]["memory_limit_mb"] == 32768
        assert data["policy"]["memory_offload_ratio"] == 0.3

    def test_update_default_policy_invalid_offload_ratio(self):
        """无效的 offload_ratio 返回 422。"""
        client = _make_client()
        resp = client.put(
            "/admin/resource-policy/default",
            json={"memory_offload_ratio": 1.5},
            headers=ADMIN_HEADERS,
        )
        assert resp.status_code == 422

    def test_list_node_policies(self):
        """GET /admin/resource-policy/nodes 列出所有节点策略。"""
        config = ClusterConfig()
        config.set_node_policy("worker-01", NodeResourcePolicy(vram_limit_mb=16384))
        client = _make_client(config)
        resp = client.get("/admin/resource-policy/nodes", headers=ADMIN_HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert "worker-01" in data["node_policies"]

    def test_get_node_policy(self):
        """GET /admin/resource-policy/nodes/{node_id} 返回节点策略。"""
        config = ClusterConfig()
        config.set_node_policy("worker-01", NodeResourcePolicy(vram_limit_mb=16384))
        client = _make_client(config)
        resp = client.get("/admin/resource-policy/nodes/worker-01", headers=ADMIN_HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert data["policy"]["vram_limit_mb"] == 16384

    def test_get_node_policy_fallback_to_default(self):
        """未配置节点策略时回退到默认策略。"""
        config = ClusterConfig()
        config.default_resource_policy = NodeResourcePolicy(memory_limit_mb=16384)
        client = _make_client(config)
        resp = client.get("/admin/resource-policy/nodes/unknown-node", headers=ADMIN_HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert data["policy"]["memory_limit_mb"] == 16384

    def test_update_node_policy(self):
        """PUT /admin/resource-policy/nodes/{node_id} 更新节点策略。"""
        client = _make_client()
        resp = client.put(
            "/admin/resource-policy/nodes/worker-01",
            json={"vram_limit_mb": 14336, "memory_offload_ratio": 0.2},
            headers=ADMIN_HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["policy"]["vram_limit_mb"] == 14336
        assert data["policy"]["memory_offload_ratio"] == 0.2

    def test_delete_node_policy(self):
        """DELETE /admin/resource-policy/nodes/{node_id} 删除节点策略。"""
        config = ClusterConfig()
        config.set_node_policy("worker-01", NodeResourcePolicy(vram_limit_mb=14336))
        client = _make_client(config)
        resp = client.delete("/admin/resource-policy/nodes/worker-01", headers=ADMIN_HEADERS)
        assert resp.status_code == 200


class TestModelConfigAPI:
    """测试模型配置端点。"""

    def teardown_method(self):
        _cleanup_env()

    def test_list_model_configs(self):
        """GET /admin/model-configs 列出所有模型配置。"""
        config = ClusterConfig()
        config.add_model_config(ModelConfig(model_id="qwen2.5-7b", context_length=8192))
        client = _make_client(config)
        resp = client.get("/admin/model-configs", headers=ADMIN_HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert "qwen2.5-7b" in data["model_configs"]

    def test_get_model_config(self):
        """GET /admin/model-configs/{model_id} 返回模型配置。"""
        config = ClusterConfig()
        config.add_model_config(ModelConfig(model_id="qwen2.5-7b", context_length=8192))
        client = _make_client(config)
        resp = client.get("/admin/model-configs/qwen2.5-7b", headers=ADMIN_HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert data["config"]["context_length"] == 8192

    def test_get_model_config_not_found(self):
        """GET 不存在的模型返回 404。"""
        client = _make_client()
        resp = client.get("/admin/model-configs/nonexistent", headers=ADMIN_HEADERS)
        assert resp.status_code == 404

    def test_update_model_config(self):
        """PUT /admin/model-configs/{model_id} 更新模型配置。"""
        client = _make_client()
        resp = client.put(
            "/admin/model-configs/qwen2.5-7b",
            json={"context_length": 16384, "temperature": 0.5, "gpu_layers": 40},
            headers=ADMIN_HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["config"]["context_length"] == 16384
        assert data["config"]["temperature"] == 0.5
        assert data["config"]["gpu_layers"] == 40

    def test_update_model_config_invalid_context_length(self):
        """无效的 context_length 返回 422。"""
        client = _make_client()
        resp = client.put(
            "/admin/model-configs/test-model",
            json={"context_length": 0},
            headers=ADMIN_HEADERS,
        )
        assert resp.status_code == 422

    def test_delete_model_config(self):
        """DELETE /admin/model-configs/{model_id} 删除模型配置。"""
        config = ClusterConfig()
        config.add_model_config(ModelConfig(model_id="qwen2.5-7b"))
        client = _make_client(config)
        resp = client.delete("/admin/model-configs/qwen2.5-7b", headers=ADMIN_HEADERS)
        assert resp.status_code == 200
        # 验证已删除
        resp2 = client.get("/admin/model-configs/qwen2.5-7b", headers=ADMIN_HEADERS)
        assert resp2.status_code == 404

    def test_model_domain_type_in_config(self):
        """model_domain_type 字段在模型配置中正确返回。"""
        config = ClusterConfig()
        config.add_model_config(ModelConfig(model_id="embed-model", model_domain_type="embedding"))
        client = _make_client(config)
        resp = client.get("/admin/model-configs/embed-model", headers=ADMIN_HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert data["config"]["model_domain_type"] == "embedding"

    def test_update_model_domain_type(self):
        """PUT 可以更新 model_domain_type。"""
        client = _make_client()
        resp = client.put(
            "/admin/model-configs/test-model",
            json={"model_domain_type": "embedding", "context_length": 2048},
            headers=ADMIN_HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["config"]["model_domain_type"] == "embedding"


class TestEstimateResourcesAPI:
    """测试资源估算端点。"""

    def teardown_method(self):
        _cleanup_env()

    def test_estimate_resources_requires_admin(self):
        """非管理员无法访问资源估算端点。"""
        client = _make_client()
        resp = client.post(
            "/admin/estimate-resources",
            json={"model_path": "test.gguf"},
            headers=USER_HEADERS,
        )
        assert resp.status_code == 403

    def test_estimate_resources_returns_estimate(self):
        """POST /admin/estimate-resources 返回资源估算。"""
        client = _make_client()
        resp = client.post(
            "/admin/estimate-resources",
            json={"model_path": "nonexistent.gguf", "config": {"context_length": 4096}},
            headers=ADMIN_HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "total_vram_mb" in data
        assert "total_ram_mb" in data
        assert "available_vram_mb" in data
        assert "available_ram_mb" in data
        assert "passes_guardrails" in data
        assert "confidence" in data
