"""
Web UI 服务器模块

提供可视化控制台界面，支持实时监控和管理
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

try:
    from flask import Flask, render_template, jsonify, request
    from flask_socketio import SocketIO, emit
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("[UI] Flask 未安装，Web UI 功能不可用")
    print("[UI] 安装命令: pip install flask flask-socketio")

from ..core.cluster import Cluster
from ..core.config import Config


class WebUIServer:
    """
    Web UI 服务器类
    
    提供可视化监控和管理界面
    """
    
    def __init__(self, cluster: Cluster, config: Config = None):
        """
        初始化 Web UI 服务器
        
        Args:
            cluster: 集群对象
            config: 配置对象
        """
        self.cluster = cluster
        self.config = config or Config()
        
        self.port = self.config.get('ui', 'port', 8080)
        self.refresh_interval = self.config.get('ui', 'refresh_interval', 2)
        
        self.app: Optional['Flask'] = None
        self.socketio: Optional['SocketIO'] = None
        self._running = False
        
        if FLASK_AVAILABLE:
            self._init_app()
    
    def _init_app(self) -> None:
        """初始化 Flask 应用"""
        template_dir = Path(__file__).parent / 'templates'
        static_dir = Path(__file__).parent / 'static'
        
        self.app = Flask(
            __name__,
            template_folder=str(template_dir),
            static_folder=str(static_dir)
        )
        self.app.config['SECRET_KEY'] = 'exollama-secret-key'
        
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        self._register_routes()
        self._register_socketio_events()
        
        # 注册集群状态变更回调
        self.cluster.register_callback(self._on_cluster_change)
    
    def _register_routes(self) -> None:
        """注册 HTTP 路由"""
        
        @self.app.route('/')
        def index():
            """首页"""
            return render_template('index.html')
        
        @self.app.route('/api/cluster')
        def api_cluster():
            """集群信息 API"""
            return jsonify(self.cluster.get_cluster_info())
        
        @self.app.route('/api/nodes')
        def api_nodes():
            """节点列表 API"""
            nodes = [
                node.to_dict()
                for node in self.cluster.nodes.values()
            ]
            return jsonify({'nodes': nodes})
        
        @self.app.route('/api/node/<node_id>')
        def api_node(node_id):
            """单个节点信息 API"""
            node = self.cluster.get_node(node_id)
            if node:
                return jsonify(node.to_dict())
            return jsonify({'error': 'Node not found'}), 404
        
        @self.app.route('/api/infer', methods=['POST'])
        def api_infer():
            """推理 API"""
            data = request.get_json()
            prompt = data.get('prompt', '')
            max_tokens = data.get('max_tokens', 128)
            
            # 这里简化处理，实际应该调用推理引擎
            return jsonify({
                'success': True,
                'output': f'推理结果: {prompt}',
                'prompt': prompt,
            })
    
    def _register_socketio_events(self) -> None:
        """注册 WebSocket 事件"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """客户端连接"""
            print(f"[UI] 客户端已连接")
            emit('cluster_info', self.cluster.get_cluster_info())
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """客户端断开"""
            print(f"[UI] 客户端已断开")
        
        @self.socketio.on('request_cluster_info')
        def handle_request_cluster_info():
            """请求集群信息"""
            emit('cluster_info', self.cluster.get_cluster_info())
        
        @self.socketio.on('request_infer')
        def handle_request_infer(data):
            """请求推理"""
            prompt = data.get('prompt', '')
            # 实际应该调用推理引擎
            emit('infer_result', {
                'success': True,
                'output': f'推理结果: {prompt}',
            })
    
    def _on_cluster_change(self, event_type: str, data) -> None:
        """
        集群状态变更回调
        
        Args:
            event_type: 事件类型
            data: 事件数据
        """
        if self.socketio:
            self.socketio.emit('cluster_update', {
                'event': event_type,
                'data': str(data),
                'cluster_info': self.cluster.get_cluster_info(),
            })
    
    def start(self) -> bool:
        """
        启动 Web UI 服务器
        
        Returns:
            是否启动成功
        """
        if not FLASK_AVAILABLE:
            print("[UI] Flask 未安装，无法启动 Web UI")
            return False
        
        if self._running:
            return True
        
        self._running = True
        
        print(f"[UI] Web UI 服务器已启动")
        print(f"[UI] 访问地址: http://localhost:{self.port}")
        
        # 在后台线程运行服务器
        import threading
        server_thread = threading.Thread(
            target=self._run_server,
            daemon=True
        )
        server_thread.start()
        
        return True
    
    def _run_server(self) -> None:
        """运行服务器"""
        if self.socketio:
            self.socketio.run(
                self.app,
                host='0.0.0.0',
                port=self.port,
                debug=False,
                use_reloader=False
            )
    
    def stop(self) -> None:
        """停止 Web UI 服务器"""
        self._running = False
        print("[UI] Web UI 服务器已停止")
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取 UI 服务器状态
        
        Returns:
            状态信息字典
        """
        return {
            'enabled': self.config.get('ui', 'enabled', True),
            'running': self._running,
            'port': self.port,
            'flask_available': FLASK_AVAILABLE,
        }
