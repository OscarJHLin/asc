"""补充 Node GUI 测试，提升覆盖率至 90%+。

原 test_node_gui.py 仅测试了基础导入和 Flask 不可用场景。
本文件补充：
- HTML 生成
- API 端点 (/api/resources, /api/config)
- 配置保存逻辑
- 异常边界
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from asc.worker.node_gui import (
    FLASK_AVAILABLE,
    _get_node_gui_html,
    create_node_gui_app,
    run_node_gui,
)

pytestmark = pytest.mark.skipif(not FLASK_AVAILABLE, reason="Flask not installed")


class TestGetNodeGuiHtml:
    """测试 HTML 页面生成。"""

    def test_contains_node_id(self):
        """HTML 应包含节点 ID。"""
        html = _get_node_gui_html("test-node-123")
        assert "test-node-123" in html
        assert "ASC Worker" in html

    def test_contains_all_sections(self):
        """HTML 应包含所有关键区域。"""
        html = _get_node_gui_html("node-1")
        assert "CPU" in html
        assert "内存" in html
        assert "GPU" in html
        assert "连接状态" in html
        assert "资源配置" in html
        assert "saveConfig" in html
        assert "/api/resources" in html
        assert "/api/config" in html


class TestCreateNodeGuiApp:
    """测试 Flask 应用创建和端点。"""

    @pytest.fixture
    def mock_agent(self):
        """创建模拟 WorkerAgent。"""
        agent = MagicMock()
        agent.node_id = "worker-1"
        agent._client = None
        agent._master_host = "10.0.0.1"
        agent._cluster_config = None

        # 模拟资源
        gpu = MagicMock()
        gpu.index = 0
        gpu.name = "RTX 4090"
        gpu.vendor = "nvidia"
        gpu.vram_total_mb = 24576
        gpu.vram_free_mb = 20480
        gpu.compute_capability = "8.9"

        resources = MagicMock()
        resources.cpu_count = 16
        resources.cpu_brand = "Intel i9"
        resources.cpu_physical_count = 8
        resources.cpu_freq_mhz = 3200
        resources.cpu_percent = 15.5
        resources.memory_total_mb = 65536
        resources.memory_free_mb = 32768
        resources.compute_score = 85.0
        resources.gpus = [gpu]
        agent.get_resources.return_value = resources
        return agent

    def test_index_route(self, mock_agent):
        """测试首页路由返回 HTML。"""
        app = create_node_gui_app(mock_agent)
        with app.test_client() as client:
            resp = client.get("/")
            assert resp.status_code == 200
            assert b"worker-1" in resp.data
            assert b"ASC Worker" in resp.data

    def test_api_resources(self, mock_agent):
        """测试 /api/resources 返回正确 JSON。"""
        app = create_node_gui_app(mock_agent)
        with app.test_client() as client:
            resp = client.get("/api/resources")
            assert resp.status_code == 200
            data = json.loads(resp.data)
            assert data["node_id"] == "worker-1"
            assert data["cpu_count"] == 16
            assert data["cpu_brand"] == "Intel i9"
            assert data["memory_total_mb"] == 65536
            assert data["compute_score"] == 85.0
            assert len(data["gpus"]) == 1
            assert data["gpus"][0]["name"] == "RTX 4090"
            assert data["connected"] is False
            assert data["master_host"] == "10.0.0.1"

    def test_api_resources_connected(self, mock_agent):
        """测试已连接状态下的 /api/resources。"""
        mock_agent._client = MagicMock()
        mock_agent._client.is_connected = True
        app = create_node_gui_app(mock_agent)
        with app.test_client() as client:
            resp = client.get("/api/resources")
            data = json.loads(resp.data)
            assert data["connected"] is True

    def test_api_resources_no_gpus(self, mock_agent):
        """测试无 GPU 时的 /api/resources。"""
        mock_agent.get_resources.return_value.gpus = []
        app = create_node_gui_app(mock_agent)
        with app.test_client() as client:
            resp = client.get("/api/resources")
            data = json.loads(resp.data)
            assert data["gpus"] == []

    def test_api_config_post(self, mock_agent):
        """测试 /api/config POST 保存配置。"""
        from asc.core.cluster_config import ClusterConfig

        config = ClusterConfig()
        mock_agent._cluster_config = config
        mock_agent.get_resources.return_value.gpus = [
            MagicMock(vram_total_mb=24576),
        ]
        mock_agent.get_resources.return_value.memory_total_mb = 65536

        app = create_node_gui_app(mock_agent)
        with app.test_client() as client:
            resp = client.post(
                "/api/config",
                data=json.dumps({
                    "vram_limit_percent": 80,
                    "memory_limit_percent": 90,
                    "offload_ratio": 0.3,
                }),
                content_type="application/json",
            )
            assert resp.status_code == 200
            data = json.loads(resp.data)
            assert data["success"] is True
            assert data["config"]["vram_limit_mb"] == 19660  # 24576 * 0.8
            assert data["config"]["memory_limit_mb"] == 58982  # 65536 * 0.9
            assert data["config"]["memory_offload_ratio"] == 0.3

    def test_api_config_post_no_cluster_config(self, mock_agent):
        """测试无 cluster_config 时的 /api/config。"""
        mock_agent._cluster_config = None
        mock_agent.get_resources.return_value.gpus = []
        mock_agent.get_resources.return_value.memory_total_mb = 32768

        app = create_node_gui_app(mock_agent)
        with app.test_client() as client:
            resp = client.post(
                "/api/config",
                data=json.dumps({"vram_limit_percent": 100, "memory_limit_percent": 100}),
                content_type="application/json",
            )
            assert resp.status_code == 200
            data = json.loads(resp.data)
            assert data["success"] is True

    def test_api_config_post_empty_json(self, mock_agent):
        """测试 POST 空 JSON 时的 /api/config。"""
        app = create_node_gui_app(mock_agent)
        with app.test_client() as client:
            resp = client.post("/api/config", data="{}", content_type="application/json")
            assert resp.status_code == 200
            data = json.loads(resp.data)
            assert data["success"] is True


class TestRunNodeGui:
    """测试 run_node_gui 函数。"""

    def test_run_node_gui_flask_not_available(self):
        """Flask 不可用时静默返回。"""
        agent = MagicMock()
        with patch("asc.worker.node_gui.FLASK_AVAILABLE", False):
            # 不应抛异常
            run_node_gui(agent, port=62415)

    def test_run_node_gui_success(self):
        """正常启动 GUI 服务器。"""
        agent = MagicMock()
        agent.node_id = "node-1"
        with patch("werkzeug.serving.run_simple") as mock_run:
            run_node_gui(agent, port=62415)
            mock_run.assert_called_once()
            args = mock_run.call_args[0]
            assert args[0] == "0.0.0.0"
            assert args[1] == 62415

    def test_run_node_gui_error(self):
        """启动失败时记录日志。"""
        agent = MagicMock()
        agent.node_id = "node-1"
        with patch("werkzeug.serving.run_simple", side_effect=OSError("bind error")):
            # 不应抛异常到上层
            run_node_gui(agent, port=62415)
