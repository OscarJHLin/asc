"""测试 Web UI 后端（WebSocket + 静态文件 + 广播）。"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from asc.api.auth import _init_default_store
from asc.api.server import create_app
from asc.core.cluster_config import ClusterConfig


def _make_client(config: ClusterConfig | None = None) -> TestClient:
    os.environ["ASC_ADMIN_API_KEY"] = "admin-test-key"
    os.environ["ASC_API_KEY"] = "user-test-key"
    os.environ.pop("ASC_ALLOW_NO_AUTH", None)
    _init_default_store()
    app = create_app(cluster_config=config or ClusterConfig())
    return TestClient(app)


ADMIN_HEADERS = {"X-API-Key": "admin-test-key"}


class TestWebUIStatic:
    """测试 Web UI 静态页面。"""

    def test_dashboard_page_exists(self):
        """Dashboard 页面可访问。"""
        client = _make_client()
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
        assert b"ASC" in resp.content

    def test_dashboard_contains_cluster_info(self):
        """Dashboard 包含集群管理相关元素。"""
        client = _make_client()
        resp = client.get("/")
        content = resp.content.decode()
        assert "cluster" in content.lower() or "集群" in content
        assert "nodes" in content.lower() or "节点" in content


class TestWebUIWebSocket:
    """测试 WebSocket 实时推送。"""

    def test_websocket_connect(self):
        """WebSocket 可以连接（带认证 token）。"""
        client = _make_client()
        with client.websocket_connect("/ws/cluster?token=admin-test-key") as ws:
            # 连接后应收到初始状态
            data = ws.receive_json()
            assert "type" in data

    def test_websocket_requires_auth(self):
        """WebSocket 需要认证。"""
        client = _make_client()
        # 不带 token 应被拒绝
        with pytest.raises(Exception), client.websocket_connect("/ws/cluster") as ws:
            ws.receive_json()

    def test_websocket_subscribe_with_token(self):
        """带 token 订阅集群状态。"""
        client = _make_client()
        with client.websocket_connect("/ws/cluster?token=admin-test-key") as ws:
            data = ws.receive_json()
            assert data["type"] == "cluster_state"

    def test_websocket_receive_node_updates(self):
        """WebSocket 能接收节点更新。"""
        client = _make_client()
        with client.websocket_connect("/ws/cluster?token=admin-test-key") as ws:
            # 第一条：cluster_state
            data = ws.receive_json()
            assert data["type"] == "cluster_state"
            assert "nodes_count" in data["data"]
            # 第二条：node_update
            data2 = ws.receive_json()
            assert data2["type"] == "node_update"
            assert "nodes" in data2["data"]


class TestClusterBroadcastAPI:
    """测试集群广播配置 API。"""

    def test_broadcast_config_requires_admin(self):
        """广播配置需要管理员权限。"""
        client = _make_client()
        resp = client.post("/admin/broadcast-config", json={"key": "value"})
        assert resp.status_code in (401, 403)

    def test_broadcast_config_to_workers(self):
        """管理员可以广播配置到所有 Worker。"""
        client = _make_client()
        resp = client.post(
            "/admin/broadcast-config",
            json={"action": "update_policy", "node_id": "worker-01", "policy": {"vram_limit_mb": 8192}},
            headers=ADMIN_HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "broadcasted_to" in data or "sent" in data

    def test_broadcast_config_invalid_action(self):
        """无效的广播 action 返回 422。"""
        client = _make_client()
        resp = client.post(
            "/admin/broadcast-config",
            json={"action": "invalid_action"},
            headers=ADMIN_HEADERS,
        )
        assert resp.status_code == 422
