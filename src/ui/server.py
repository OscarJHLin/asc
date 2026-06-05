"""
Web UI 服务器模块

提供可视化控制台界面，支持实时监控和管理
"""

from pathlib import Path
from typing import Any, Dict, Optional

try:
    from flask import Flask, jsonify, render_template, request
    from flask_socketio import SocketIO, emit
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

from ..core.cluster import Cluster
from ..core.config import Config
from ..core.logging import get_logger
from ..inference.engine import InferenceEngine

logger = get_logger('ui')


class WebUIServer:
    """
    Web UI 服务器类
    
    提供可视化监控和管理界面
    """
    
    def __init__(self, cluster: Cluster, config: Config = None, engine: InferenceEngine = None):
        self.cluster = cluster
        self.config = config or Config()
        self.engine = engine or InferenceEngine(self.config)
        
        self.port = self.config.get('ui', 'port', 8080)
        self.refresh_interval = self.config.get('ui', 'refresh_interval', 2)
        
        self.app: Optional['Flask'] = None
        self.socketio: Optional['SocketIO'] = None
        self._running = False
        
        if FLASK_AVAILABLE:
            self._init_app()
        else:
            logger.warning("Flask 未安装，Web UI 不可用。安装: pip install flask flask-socketio")
    
    def _init_app(self) -> None:
        """初始化 Flask 应用"""
        template_dir = Path(__file__).parent / 'templates'
        static_dir = Path(__file__).parent / 'static'
        
        self.app = Flask(
            __name__,
            template_folder=str(template_dir),
            static_folder=str(static_dir) if static_dir.exists() else None
        )
        self.app.config['SECRET_KEY'] = 'asc-secret-key'
        
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        self._register_routes()
        self._register_socketio_events()
        
        self.cluster.register_callback(self._on_cluster_change)
    
    def _register_routes(self) -> None:
        """注册 HTTP 路由"""
        
        @self.app.route('/')
        def index():
            return render_template('index.html')
        
        @self.app.route('/api/cluster')
        def api_cluster():
            return jsonify(self.cluster.get_cluster_info())
        
        @self.app.route('/api/nodes')
        def api_nodes():
            nodes = [node.to_dict() for node in self.cluster.nodes.values()]
            return jsonify({'nodes': nodes})
        
        @self.app.route('/api/node/<node_id>')
        def api_node(node_id):
            node = self.cluster.get_node(node_id)
            if node:
                return jsonify(node.to_dict())
            return jsonify({'error': 'Node not found'}), 404
        
        @self.app.route('/api/infer', methods=['POST'])
        def api_infer():
            """推理 API - 真正调用推理引擎"""
            data = request.get_json()
            prompt = data.get('prompt', '')
            model = data.get('model', '')
            max_tokens = data.get('max_tokens', 128)
            temperature = data.get('temperature', 0.7)
            
            if not prompt:
                return jsonify({'success': False, 'error': '提示词不能为空'})
            
            # 加载模型
            if model and model != self.engine.current_model:
                if not self.engine.load_model(model):
                    return jsonify({'success': False, 'error': '模型加载失败'})
            
            if not self.engine.current_model:
                return jsonify({'success': False, 'error': '未加载模型'})
            
            # 执行推理
            result = self.engine.infer(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return jsonify(result)
        
        @self.app.route('/api/engine')
        def api_engine():
            """引擎状态 API"""
            return jsonify(self.engine.get_status())
        
        @self.app.route('/health')
        def health():
            """健康检查端点"""
            return jsonify({'status': 'ok', 'service': 'asc'})
    
    def _register_socketio_events(self) -> None:
        """注册 WebSocket 事件"""
        
        @self.socketio.on('connect')
        def handle_connect():
            logger.info("客户端已连接")
            emit('cluster_info', self.cluster.get_cluster_info())
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            logger.info("客户端已断开")
        
        @self.socketio.on('request_cluster_info')
        def handle_request_cluster_info():
            emit('cluster_info', self.cluster.get_cluster_info())
        
        @self.socketio.on('request_infer')
        def handle_request_infer(data):
            prompt = data.get('prompt', '')
            model = data.get('model', '')
            max_tokens = data.get('max_tokens', 128)
            
            if model and model != self.engine.current_model:
                self.engine.load_model(model)
            
            result = self.engine.infer(prompt=prompt, max_tokens=max_tokens)
            emit('infer_result', result)
    
    def _on_cluster_change(self, event_type: str, data) -> None:
        """集群状态变更回调"""
        if self.socketio:
            self.socketio.emit('cluster_update', {
                'event': event_type,
                'data': str(data),
                'cluster_info': self.cluster.get_cluster_info(),
            })
    
    def start(self) -> bool:
        """启动 Web UI 服务器"""
        if not FLASK_AVAILABLE:
            logger.warning("Flask 未安装，无法启动 Web UI")
            return False
        
        if self._running:
            return True
        
        self._running = True
        
        logger.info(f"Web UI 已启动: http://localhost:{self.port}")
        
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
        logger.info("Web UI 服务器已停止")
    
    def get_status(self) -> Dict[str, Any]:
        """获取 UI 服务器状态"""
        return {
            'enabled': self.config.get('ui', 'enabled', True),
            'running': self._running,
            'port': self.port,
            'flask_available': FLASK_AVAILABLE,
        }
