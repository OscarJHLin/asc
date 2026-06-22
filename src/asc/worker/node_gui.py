"""Worker 节点本地控制 GUI。

提供无需登录的节点配置界面，支持：
- 查看节点资源状态（CPU型号/占用、多GPU详情、内存详情）
- 调节显存和内存使用限制（通过滑动条）
- 配置 CPU 卸载比例
- 查看连接状态
"""

from __future__ import annotations

import logging

from asc.worker.agent import WorkerAgent

logger = logging.getLogger(__name__)

# 简单的 HTTP 服务器用于节点 GUI
try:
    from flask import Flask, jsonify, render_template_string, request

    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False


def _get_node_gui_html(node_id: str) -> str:
    """获取节点 GUI HTML 页面。"""
    return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ASC Worker 节点控制 - """ + node_id + """</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: #e2e8f0;
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 960px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            padding: 24px 0;
            border-bottom: 1px solid #334155;
            margin-bottom: 24px;
        }
        .header h1 {
            font-size: 26px;
            color: #60a5fa;
            margin-bottom: 6px;
        }
        .header .subtitle {
            color: #94a3b8;
            font-size: 13px;
        }
        .card {
            background: rgba(30, 41, 59, 0.8);
            border-radius: 14px;
            padding: 22px;
            margin-bottom: 18px;
            border: 1px solid #334155;
            backdrop-filter: blur(10px);
        }
        .card h2 {
            font-size: 16px;
            color: #94a3b8;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .info-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid rgba(51, 65, 85, 0.5);
        }
        .info-row:last-child { border-bottom: none; }
        .info-label {
            color: #94a3b8;
            font-size: 13px;
        }
        .info-value {
            color: #e2e8f0;
            font-weight: 600;
            font-size: 13px;
        }
        .info-value.highlight {
            color: #60a5fa;
        }
        .progress-bar {
            width: 120px;
            height: 6px;
            background: #334155;
            border-radius: 3px;
            overflow: hidden;
            display: inline-block;
            vertical-align: middle;
            margin-left: 8px;
        }
        .progress-fill {
            height: 100%;
            border-radius: 3px;
            transition: width 0.5s ease;
        }
        .progress-fill.green { background: #22c55e; }
        .progress-fill.yellow { background: #eab308; }
        .progress-fill.red { background: #ef4444; }
        .progress-fill.blue { background: #60a5fa; }
        .gpu-card {
            background: #0f172a;
            border-radius: 10px;
            padding: 14px;
            border: 1px solid #334155;
            margin-bottom: 10px;
        }
        .gpu-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .gpu-name {
            font-weight: 700;
            font-size: 14px;
            color: #e2e8f0;
        }
        .gpu-vendor {
            font-size: 11px;
            padding: 2px 8px;
            border-radius: 10px;
            background: #334155;
            color: #94a3b8;
        }
        .gpu-vendor.nvidia { background: #1a3a1a; color: #76b900; }
        .gpu-vendor.amd { background: #3a1a1a; color: #ed1c24; }
        .gpu-vendor.apple { background: #1a1a3a; color: #a3aaff; }
        .gpu-stats {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
        }
        .gpu-stat {
            font-size: 12px;
        }
        .gpu-stat-label {
            color: #64748b;
        }
        .gpu-stat-value {
            color: #e2e8f0;
            font-weight: 600;
        }
        .vram-bar {
            width: 100%;
            height: 8px;
            background: #334155;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 8px;
        }
        .vram-fill {
            height: 100%;
            border-radius: 4px;
            background: linear-gradient(90deg, #22c55e, #eab308);
            transition: width 0.5s ease;
        }
        .status-indicator {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 600;
        }
        .status-online {
            background: rgba(34, 197, 94, 0.2);
            color: #22c55e;
        }
        .status-offline {
            background: rgba(239, 68, 68, 0.2);
            color: #ef4444;
        }
        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: currentColor;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .slider-container {
            margin: 18px 0;
        }
        .slider-label {
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
            font-size: 14px;
        }
        .slider-label .value {
            color: #60a5fa;
            font-weight: 600;
        }
        input[type="range"] {
            width: 100%;
            height: 6px;
            border-radius: 3px;
            background: #334155;
            outline: none;
            -webkit-appearance: none;
        }
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #60a5fa;
            cursor: pointer;
            border: 2px solid #1e293b;
            box-shadow: 0 2px 6px rgba(96, 165, 250, 0.4);
        }
        input[type="range"]::-moz-range-thumb {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #60a5fa;
            cursor: pointer;
            border: 2px solid #1e293b;
        }
        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            transition: all 0.2s;
            width: 100%;
        }
        .btn-primary {
            background: linear-gradient(135deg, #2563eb, #1d4ed8);
            color: white;
        }
        .btn-primary:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4);
        }
        .info-text {
            font-size: 12px;
            color: #64748b;
            margin-top: 6px;
            line-height: 1.4;
        }
        .toast {
            position: fixed;
            bottom: 20px;
            right: 20px;
            padding: 14px 20px;
            border-radius: 10px;
            color: white;
            font-weight: 600;
            transform: translateY(100px);
            transition: transform 0.3s ease;
            z-index: 1000;
        }
        .toast.show { transform: translateY(0); }
        .toast-success { background: #16a34a; }
        .toast-error { background: #dc2626; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>ASC Worker 节点控制</h1>
            <div class="subtitle">节点: """ + node_id + """</div>
        </div>

        <!-- CPU 信息 -->
        <div class="card">
            <h2>CPU</h2>
            <div class="info-row">
                <span class="info-label">型号</span>
                <span class="info-value" id="cpu-brand">-</span>
            </div>
            <div class="info-row">
                <span class="info-label">核心</span>
                <span class="info-value" id="cpu-cores">-</span>
            </div>
            <div class="info-row">
                <span class="info-label">频率</span>
                <span class="info-value" id="cpu-freq">-</span>
            </div>
            <div class="info-row">
                <span class="info-label">占用</span>
                <span class="info-value">
                    <span id="cpu-percent">-</span>
                    <span class="progress-bar">
                        <span class="progress-fill" id="cpu-bar" style="width:0%"></span>
                    </span>
                </span>
            </div>
        </div>

        <!-- 内存信息 -->
        <div class="card">
            <h2>内存</h2>
            <div class="info-row">
                <span class="info-label">总量</span>
                <span class="info-value" id="mem-total">-</span>
            </div>
            <div class="info-row">
                <span class="info-label">可用</span>
                <span class="info-value highlight" id="mem-free">-</span>
            </div>
            <div class="info-row">
                <span class="info-label">使用率</span>
                <span class="info-value">
                    <span id="mem-percent">-</span>
                    <span class="progress-bar">
                        <span class="progress-fill" id="mem-bar" style="width:0%"></span>
                    </span>
                </span>
            </div>
        </div>

        <!-- GPU 信息 -->
        <div class="card">
            <h2>GPU</h2>
            <div id="gpu-list"></div>
        </div>

        <!-- 连接状态 -->
        <div class="card">
            <h2>连接状态</h2>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span id="connection-status" class="status-indicator status-offline">
                    <span class="status-dot"></span>
                    <span id="status-text">未连接</span>
                </span>
                <span id="master-info" style="font-size: 13px; color: #64748b;">-</span>
            </div>
        </div>

        <!-- 资源配置 -->
        <div class="card">
            <h2>资源配置</h2>

            <div class="slider-container">
                <div class="slider-label">
                    <span>显存使用限制</span>
                    <span class="value" id="vram-limit-value">不限制</span>
                </div>
                <input type="range" id="vram-limit" min="0" max="100" value="100">
                <div class="info-text">限制模型可使用的显存量，剩余显存保留给系统</div>
            </div>

            <div class="slider-container">
                <div class="slider-label">
                    <span>内存使用限制</span>
                    <span class="value" id="memory-limit-value">不限制</span>
                </div>
                <input type="range" id="memory-limit" min="0" max="100" value="100">
                <div class="info-text">限制模型可使用的内存量，影响 CPU 卸载层数</div>
            </div>

            <div class="slider-container">
                <div class="slider-label">
                    <span>CPU 卸载比例</span>
                    <span class="value" id="offload-value">0%</span>
                </div>
                <input type="range" id="offload-ratio" min="0" max="100" value="0">
                <div class="info-text">将部分模型层卸载到内存，降低显存占用但影响速度</div>
            </div>

            <button class="btn btn-primary" onclick="saveConfig()">保存配置</button>
        </div>
    </div>

    <div id="toast" class="toast"></div>

    <script>
        let config = {
            vram_limit_percent: 100,
            memory_limit_percent: 100,
            offload_ratio: 0
        };

        function showToast(msg, type = 'success') {
            const toast = document.getElementById('toast');
            toast.textContent = msg;
            toast.className = 'toast toast-' + type + ' show';
            setTimeout(() => toast.classList.remove('show'), 3000);
        }

        function formatMB(mb) {
            if (mb >= 1024) return (mb / 1024).toFixed(1) + ' GB';
            return mb + ' MB';
        }

        function getProgressColor(percent) {
            if (percent < 60) return 'green';
            if (percent < 85) return 'yellow';
            return 'red';
        }

        async function fetchResources() {
            try {
                const resp = await fetch('/api/resources');
                const data = await resp.json();
                updateUI(data);
            } catch (e) {
                console.error('获取资源失败:', e);
            }
        }

        function updateUI(data) {
            // CPU
            document.getElementById('cpu-brand').textContent = data.cpu_brand || '未知';
            document.getElementById('cpu-cores').textContent =
                (data.cpu_physical_count || 0) + ' 物理 / ' + (data.cpu_count || 0) + ' 逻辑';
            document.getElementById('cpu-freq').textContent =
                data.cpu_freq_mhz ? (data.cpu_freq_mhz / 1000).toFixed(2) + ' GHz' : '-';

            const cpuPercent = data.cpu_percent || 0;
            document.getElementById('cpu-percent').textContent = cpuPercent.toFixed(1) + '%';
            const cpuBar = document.getElementById('cpu-bar');
            cpuBar.style.width = cpuPercent + '%';
            cpuBar.className = 'progress-fill ' + getProgressColor(cpuPercent);

            // 内存
            const memTotal = data.memory_total_mb || 0;
            const memFree = data.memory_free_mb || 0;
            const memUsed = memTotal - memFree;
            const memPercent = memTotal > 0 ? (memUsed / memTotal * 100) : 0;

            document.getElementById('mem-total').textContent = formatMB(memTotal);
            document.getElementById('mem-free').textContent = formatMB(memFree);
            document.getElementById('mem-percent').textContent = memPercent.toFixed(1) + '%';
            const memBar = document.getElementById('mem-bar');
            memBar.style.width = memPercent + '%';
            memBar.className = 'progress-fill ' + getProgressColor(memPercent);

            // GPU
            const gpuList = document.getElementById('gpu-list');
            if (data.gpus && data.gpus.length > 0) {
                gpuList.innerHTML = data.gpus.map(gpu => {
                    const vramUsed = gpu.vram_total_mb - gpu.vram_free_mb;
                    const vramPercent = gpu.vram_total_mb > 0 ? (vramUsed / gpu.vram_total_mb * 100) : 0;
                    const vendorClass = (gpu.vendor || '').toLowerCase();
                    return `
                    <div class="gpu-card">
                        <div class="gpu-header">
                            <span class="gpu-name">GPU ${gpu.index}: ${gpu.name}</span>
                            <span class="gpu-vendor ${vendorClass}">${gpu.vendor || 'unknown'}</span>
                        </div>
                        <div class="gpu-stats">
                            <div class="gpu-stat">
                                <span class="gpu-stat-label">总显存: </span>
                                <span class="gpu-stat-value">${formatMB(gpu.vram_total_mb)}</span>
                            </div>
                            <div class="gpu-stat">
                                <span class="gpu-stat-label">可用显存: </span>
                                <span class="gpu-stat-value" style="color:#60a5fa">${formatMB(gpu.vram_free_mb)}</span>
                            </div>
                            <div class="gpu-stat">
                                <span class="gpu-stat-label">已用显存: </span>
                                <span class="gpu-stat-value">${formatMB(vramUsed)}</span>
                            </div>
                            <div class="gpu-stat">
                                <span class="gpu-stat-label">使用率: </span>
                                <span class="gpu-stat-value">${vramPercent.toFixed(1)}%</span>
                            </div>
                        </div>
                        <div class="vram-bar">
                            <div class="vram-fill" style="width:${vramPercent}%"></div>
                        </div>
                    </div>`;
                }).join('');
            } else {
                gpuList.innerHTML = '<div class="gpu-card"><div class="gpu-name">未检测到 GPU</div></div>';
            }

            // 连接状态
            const statusEl = document.getElementById('connection-status');
            const statusText = document.getElementById('status-text');
            if (data.connected) {
                statusEl.className = 'status-indicator status-online';
                statusText.textContent = '已连接';
                document.getElementById('master-info').textContent = 'Master: ' + (data.master_host || 'unknown');
            } else {
                statusEl.className = 'status-indicator status-offline';
                statusText.textContent = '未连接';
                document.getElementById('master-info').textContent = '独立运行模式';
            }
        }

        // 滑动条事件
        document.getElementById('vram-limit').addEventListener('input', function() {
            const val = this.value;
            config.vram_limit_percent = parseInt(val);
            document.getElementById('vram-limit-value').textContent = val == 100 ? '不限制' : val + '%';
        });

        document.getElementById('memory-limit').addEventListener('input', function() {
            const val = this.value;
            config.memory_limit_percent = parseInt(val);
            document.getElementById('memory-limit-value').textContent = val == 100 ? '不限制' : val + '%';
        });

        document.getElementById('offload-ratio').addEventListener('input', function() {
            const val = this.value;
            config.offload_ratio = parseInt(val) / 100;
            document.getElementById('offload-value').textContent = val + '%';
        });

        async function saveConfig() {
            try {
                const resp = await fetch('/api/config', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(config)
                });
                const data = await resp.json();
                if (data.success) {
                    showToast('配置已保存', 'success');
                } else {
                    showToast('保存失败: ' + data.error, 'error');
                }
            } catch (e) {
                showToast('保存失败', 'error');
            }
        }

        // 定时刷新
        fetchResources();
        setInterval(fetchResources, 5000);
    </script>
</body>
</html>"""


def create_node_gui_app(agent: WorkerAgent) -> Flask:
    """创建 Worker 节点 GUI Flask 应用。"""
    if not FLASK_AVAILABLE:
        raise RuntimeError("Flask 未安装，无法创建节点 GUI。安装: pip install flask")

    app = Flask(__name__)
    app.agent = agent

    @app.route("/")
    def index():
        return render_template_string(_get_node_gui_html(agent.node_id))

    @app.route("/api/resources")
    def api_resources():
        resources = agent.get_resources()
        return jsonify({
            "node_id": agent.node_id,
            "cpu_count": resources.cpu_count,
            "cpu_brand": resources.cpu_brand,
            "cpu_physical_count": resources.cpu_physical_count,
            "cpu_freq_mhz": resources.cpu_freq_mhz,
            "cpu_percent": resources.cpu_percent,
            "memory_total_mb": resources.memory_total_mb,
            "memory_free_mb": resources.memory_free_mb,
            "compute_score": resources.compute_score,
            "gpus": [
                {
                    "index": g.index,
                    "name": g.name,
                    "vendor": g.vendor,
                    "vram_total_mb": g.vram_total_mb,
                    "vram_free_mb": g.vram_free_mb,
                    "compute_capability": g.compute_capability,
                }
                for g in resources.gpus
            ],
            "connected": agent._client is not None and agent._client.is_connected if hasattr(agent, '_client') else False,
            "master_host": getattr(agent, '_master_host', None),
        })

    @app.route("/api/config", methods=["POST"])
    def api_config():
        data = request.get_json() or {}

        from asc.core.cluster_config import NodeResourcePolicy

        vram_percent = data.get("vram_limit_percent", 100)
        memory_percent = data.get("memory_limit_percent", 100)
        offload_ratio = data.get("offload_ratio", 0.0)

        resources = agent.get_resources()

        total_vram = sum(g.vram_total_mb for g in resources.gpus)
        total_memory = resources.memory_total_mb

        vram_limit = int(total_vram * vram_percent / 100) if vram_percent < 100 else None
        memory_limit = int(total_memory * memory_percent / 100) if memory_percent < 100 else None

        policy = NodeResourcePolicy(
            vram_limit_mb=vram_limit,
            memory_limit_mb=memory_limit,
            memory_offload_ratio=offload_ratio,
        )

        if agent._cluster_config is not None:
            agent._cluster_config.set_node_policy(agent.node_id, policy)

        return jsonify({"success": True, "config": policy.to_dict()})

    return app


def run_node_gui(agent: WorkerAgent, port: int = 62415) -> None:
    """运行节点 GUI 服务器。"""
    if not FLASK_AVAILABLE:
        try:
            logger.warning("Flask 未安装，跳过节点 GUI 启动")
        except OSError:
            pass
        return

    try:
        app = create_node_gui_app(agent)
        import werkzeug.serving

        werkzeug.serving.run_simple(
            "0.0.0.0", port, app,
            threaded=True,
            use_reloader=False,
            use_debugger=False,
        )
    except Exception as e:
        logger.error("节点 GUI 启动失败: %s", e)
