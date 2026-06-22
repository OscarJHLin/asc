"""测试 api/server.py 管理端点覆盖率补充。

覆盖范围：
- /admin/cluster-config (GET/PUT)
- /admin/resource-policy/* (GET/PUT/DELETE)
- /admin/model-configs/* (GET/PUT/DELETE)
- /admin/broadcast-config
- /admin/deploy-model (SSRF/路径遍历校验)
- /admin/api-config
"""

import os
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from asc.api.auth import _init_default_store
from asc.api.server import create_app
from asc.core.cluster_config import ClusterConfig

ADMIN_HEADERS = {"X-API-Key": "admin-key"}
USER_HEADERS = {"X-API-Key": "user-key"}


def _make_client(config: ClusterConfig | None = None, **kwargs):
    """创建带管理员认证的测试客户端。"""
    os.environ["ASC_ADMIN_API_KEY"] = "admin-key"
    os.environ["ASC_API_KEY"] = "user-key"
    os.environ.pop("ASC_ALLOW_NO_AUTH", None)
    _init_default_store()
    app = create_app(cluster_config=config or ClusterConfig(), **kwargs)
    return TestClient(app)


def _cleanup_env():
    os.environ.pop("ASC_ADMIN_API_KEY", None)
    os.environ.pop("ASC_API_KEY", None)
    os.environ.pop("ASC_ALLOW_NO_AUTH", None)
    _init_default_store()


class TestClusterConfigEndpoints:
    """集群配置管理端点。"""

    def teardown_method(self):
        _cleanup_env()

    def test_get_cluster_config(self):
        """GET /admin/cluster-config 返回完整配置。"""
        client = _make_client()
        resp = client.get("/admin/cluster-config", headers=ADMIN_HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert "heartbeat_interval" in data
        assert "default_resource_policy" in data

    def test_update_cluster_config(self):
        """PUT /admin/cluster-config 更新集群参数。"""
        client = _make_client()
        resp = client.put(
            "/admin/cluster-config",
            headers=ADMIN_HEADERS,
            json={"heartbeat_interval": 10, "node_timeout": 60},
        )
        assert resp.status_code == 200
        assert resp.json()["config"]["heartbeat_interval"] == 10

    def test_update_cluster_config_invalid_value(self):
        """PUT /admin/cluster-config 传入非法值返回 422。"""
        client = _make_client()
        resp = client.put(
            "/admin/cluster-config",
            headers=ADMIN_HEADERS,
            json={"default_resource_policy": {"memory_offload_ratio": 2.0}},
        )
        assert resp.status_code == 422

    def test_cluster_config_forbidden_for_user(self):
        """普通用户访问管理端点返回 403。"""
        client = _make_client()
        resp = client.get("/admin/cluster-config", headers=USER_HEADERS)
        assert resp.status_code == 403


class TestResourcePolicyEndpoints:
    """资源策略管理端点。"""

    def teardown_method(self):
        _cleanup_env()

    def test_get_default_resource_policy(self):
        """GET /admin/resource-policy/default 返回默认策略。"""
        client = _make_client()
        resp = client.get("/admin/resource-policy/default", headers=ADMIN_HEADERS)
        assert resp.status_code == 200
        assert "memory_limit_mb" in resp.json()

    def test_update_default_resource_policy(self):
        """PUT /admin/resource-policy/default 更新默认策略。"""
        client = _make_client()
        resp = client.put(
            "/admin/resource-policy/default",
            headers=ADMIN_HEADERS,
            json={"memory_limit_mb": 16384, "vram_reserve_mb": 1024},
        )
        assert resp.status_code == 200
        assert resp.json()["policy"]["memory_limit_mb"] == 16384

    def test_list_node_policies_empty(self):
        """GET /admin/resource-policy/nodes 初始为空。"""
        client = _make_client()
        resp = client.get("/admin/resource-policy/nodes", headers=ADMIN_HEADERS)
        assert resp.status_code == 200
        assert resp.json()["node_policies"] == {}

    def test_update_and_get_node_policy(self):
        """PUT/GET /admin/resource-policy/nodes/{node_id}。"""
        client = _make_client()
        resp = client.put(
            "/admin/resource-policy/nodes/worker-1",
            headers=ADMIN_HEADERS,
            json={"memory_limit_mb": 8192},
        )
        assert resp.status_code == 200
        assert resp.json()["policy"]["memory_limit_mb"] == 8192

        resp = client.get("/admin/resource-policy/nodes/worker-1", headers=ADMIN_HEADERS)
        assert resp.status_code == 200
        assert resp.json()["policy"]["memory_limit_mb"] == 8192

    def test_delete_node_policy(self):
        """DELETE /admin/resource-policy/nodes/{node_id} 回退到默认。"""
        config = ClusterConfig()
        config.set_node_policy("worker-1", config.default_resource_policy)
        client = _make_client(config=config)
        resp = client.delete(
            "/admin/resource-policy/nodes/worker-1",
            headers=ADMIN_HEADERS,
        )
        assert resp.status_code == 200
        assert "removed" in resp.json()["message"]


class TestModelConfigEndpoints:
    """模型配置管理端点。"""

    def teardown_method(self):
        _cleanup_env()

    def test_list_model_configs_empty(self):
        """GET /admin/model-configs 初始为空。"""
        client = _make_client()
        resp = client.get("/admin/model-configs", headers=ADMIN_HEADERS)
        assert resp.status_code == 200
        assert resp.json()["model_configs"] == {}

    def test_update_and_get_model_config(self):
        """PUT/GET /admin/model-configs/{model_id}。"""
        client = _make_client()
        resp = client.put(
            "/admin/model-configs/qwen-7b",
            headers=ADMIN_HEADERS,
            json={"context_length": 8192, "gpu_layers": 32},
        )
        assert resp.status_code == 200
        assert resp.json()["config"]["context_length"] == 8192

        resp = client.get("/admin/model-configs/qwen-7b", headers=ADMIN_HEADERS)
        assert resp.status_code == 200
        assert resp.json()["config"]["gpu_layers"] == 32

    def test_get_model_config_not_found(self):
        """GET 不存在的模型配置返回 404。"""
        client = _make_client()
        resp = client.get("/admin/model-configs/not-exist", headers=ADMIN_HEADERS)
        assert resp.status_code == 404

    def test_delete_model_config(self):
        """DELETE /admin/model-configs/{model_id}。"""
        config = ClusterConfig()
        from asc.core.cluster_config import ModelConfig
        config.add_model_config(ModelConfig(model_id="qwen-7b"))
        client = _make_client(config=config)
        resp = client.delete("/admin/model-configs/qwen-7b", headers=ADMIN_HEADERS)
        assert resp.status_code == 200
        assert "removed" in resp.json()["message"]

    def test_update_model_config_invalid(self):
        """PUT 非法模型配置返回 422。"""
        client = _make_client()
        resp = client.put(
            "/admin/model-configs/qwen-7b",
            headers=ADMIN_HEADERS,
            json={"context_length": -1},
        )
        assert resp.status_code == 422


class TestBroadcastConfigEndpoint:
    """集群广播配置端点。"""

    def teardown_method(self):
        _cleanup_env()

    def test_broadcast_config_invalid_action(self):
        """POST /admin/broadcast-config 非法 action 返回 422。"""
        client = _make_client()
        resp = client.post(
            "/admin/broadcast-config",
            headers=ADMIN_HEADERS,
            json={"action": "invalid_action"},
        )
        assert resp.status_code == 422

    def test_broadcast_config_valid_action(self):
        """POST /admin/broadcast-config 合法 action 返回 200。"""
        client = _make_client()
        resp = client.post(
            "/admin/broadcast-config",
            headers=ADMIN_HEADERS,
            json={"action": "reload_config"},
        )
        assert resp.status_code == 200
        assert resp.json()["action"] == "reload_config"

    def test_broadcast_config_forbidden_for_user(self):
        """普通用户访问返回 403。"""
        client = _make_client()
        resp = client.post(
            "/admin/broadcast-config",
            headers=USER_HEADERS,
            json={"action": "reload_config"},
        )
        assert resp.status_code == 403


class TestDeployModelEndpoint:
    """模型部署端点安全校验。"""

    def teardown_method(self):
        _cleanup_env()

    def test_deploy_model_missing_fields(self):
        """缺少 model_id/uri 返回 422。"""
        client = _make_client()
        resp = client.post(
            "/admin/deploy-model",
            headers=ADMIN_HEADERS,
            json={"model_id": "", "uri": ""},
        )
        assert resp.status_code == 422

    def test_deploy_model_ssrf_rejected(self):
        """非 http/https scheme 的 URI 被拒绝（SSRF 防护）。"""
        client = _make_client()
        resp = client.post(
            "/admin/deploy-model",
            headers=ADMIN_HEADERS,
            json={"model_id": "m", "uri": "file:///etc/passwd"},
        )
        assert resp.status_code == 422
        assert "http or https" in resp.json()["detail"]

    def test_deploy_model_path_traversal_rejected(self):
        """包含路径遍历字符的文件名被拒绝。"""
        client = _make_client()
        resp = client.post(
            "/admin/deploy-model",
            headers=ADMIN_HEADERS,
            json={"model_id": "m", "uri": "http://example.com/m.gguf", "filename": "../etc/passwd"},
        )
        assert resp.status_code == 422
        assert "invalid filename" in resp.json()["detail"]


class TestApiConfigEndpoint:
    """API 配置端点。"""

    def teardown_method(self):
        _cleanup_env()

    def test_update_api_config(self):
        """PUT /admin/api-config 更新 API 配置。"""
        client = _make_client()
        resp = client.put(
            "/admin/api-config",
            headers=ADMIN_HEADERS,
            json={"enabled": True, "api_key": "new-key", "port": 8000},
        )
        assert resp.status_code == 200
        assert resp.json()["api_enabled"] is True
        assert resp.json()["api_port"] == 8000

    def test_update_api_config_forbidden_for_user(self):
        """普通用户访问返回 403。"""
        client = _make_client()
        resp = client.put(
            "/admin/api-config",
            headers=USER_HEADERS,
            json={"enabled": True},
        )
        assert resp.status_code == 403
