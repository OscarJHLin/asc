"""
节点管理模块

管理单个节点的状态、资源和通信
"""

import socket
import time
import platform
import psutil
from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


class NodeStatus(Enum):
    """节点状态枚举"""
    OFFLINE = "offline"
    STARTING = "starting"
    ONLINE = "online"
    BUSY = "busy"
    ERROR = "error"


@dataclass
class NodeResources:
    """节点资源信息"""
    cpu_count: int = 0
    cpu_percent: float = 0.0
    memory_total: int = 0  # MB
    memory_used: int = 0   # MB
    memory_percent: float = 0.0
    gpu_count: int = 0
    gpu_memory: Dict[str, int] = field(default_factory=dict)
    disk_free: int = 0     # MB
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'cpu_count': self.cpu_count,
            'cpu_percent': self.cpu_percent,
            'memory_total': self.memory_total,
            'memory_used': self.memory_used,
            'memory_percent': self.memory_percent,
            'gpu_count': self.gpu_count,
            'gpu_memory': self.gpu_memory,
            'disk_free': self.disk_free,
        }


class Node:
    """
    节点类
    
    表示集群中的一个节点，管理其状态和资源
    """
    
    def __init__(self, 
                 node_id: str = None,
                 name: str = None,
                 ip: str = None,
                 port: int = 52415,
                 rpc_port: int = 50052):
        """
        初始化节点
        
        Args:
            node_id: 节点唯一标识
            name: 节点名称
            ip: 节点IP地址
            port: HTTP服务端口
            rpc_port: RPC服务端口
        """
        self.node_id = node_id or self._generate_node_id()
        self.name = name or socket.gethostname()
        self.ip = ip or self._get_local_ip()
        self.port = port
        self.rpc_port = rpc_port
        
        self.status = NodeStatus.OFFLINE
        self.platform = platform.system()
        self.version = "1.0.0"
        
        self.resources = NodeResources()
        self.last_heartbeat = 0
        self.capabilities = []
        
        self._update_resources()
    
    def _generate_node_id(self) -> str:
        """生成节点ID"""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def _get_local_ip(self) -> str:
        """获取本地IP地址"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"
    
    def _update_resources(self) -> None:
        """更新资源信息"""
        self.resources.cpu_count = psutil.cpu_count()
        self.resources.cpu_percent = psutil.cpu_percent(interval=0.1)
        
        mem = psutil.virtual_memory()
        self.resources.memory_total = mem.total // (1024 * 1024)
        self.resources.memory_used = mem.used // (1024 * 1024)
        self.resources.memory_percent = mem.percent
        
        disk = psutil.disk_usage('/')
        self.resources.disk_free = disk.free // (1024 * 1024)
        
        # 检测GPU
        self._detect_gpu()
    
    def _detect_gpu(self) -> None:
        """检测GPU信息"""
        try:
            import torch
            if torch.cuda.is_available():
                self.resources.gpu_count = torch.cuda.device_count()
                for i in range(self.resources.gpu_count):
                    props = torch.cuda.get_device_properties(i)
                    self.resources.gpu_memory[f'gpu_{i}'] = props.total_memory // (1024 * 1024)
                self.capabilities.append('cuda')
        except ImportError:
            pass
        
        # 检测 Metal (macOS)
        if self.platform == 'Darwin':
            try:
                import torch
                if torch.backends.mps.is_available():
                    self.capabilities.append('metal')
            except:
                pass
    
    def update_status(self, status: NodeStatus) -> None:
        """
        更新节点状态
        
        Args:
            status: 新状态
        """
        self.status = status
        if status == NodeStatus.ONLINE:
            self.last_heartbeat = time.time()
    
    def heartbeat(self) -> None:
        """更新心跳时间"""
        self.last_heartbeat = time.time()
        self._update_resources()
    
    def is_alive(self, timeout: int = 30) -> bool:
        """
        检查节点是否存活
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            是否存活
        """
        if self.status != NodeStatus.ONLINE:
            return False
        return (time.time() - self.last_heartbeat) < timeout
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典
        
        Returns:
            节点信息字典
        """
        return {
            'node_id': self.node_id,
            'name': self.name,
            'ip': self.ip,
            'port': self.port,
            'rpc_port': self.rpc_port,
            'status': self.status.value,
            'platform': self.platform,
            'version': self.version,
            'resources': self.resources.to_dict(),
            'capabilities': self.capabilities,
            'last_heartbeat': self.last_heartbeat,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Node':
        """
        从字典创建节点
        
        Args:
            data: 节点信息字典
            
        Returns:
            节点实例
        """
        node = cls(
            node_id=data.get('node_id'),
            name=data.get('name'),
            ip=data.get('ip'),
            port=data.get('port', 52415),
            rpc_port=data.get('rpc_port', 50052)
        )
        node.status = NodeStatus(data.get('status', 'offline'))
        node.version = data.get('version', '1.0.0')
        node.capabilities = data.get('capabilities', [])
        node.last_heartbeat = data.get('last_heartbeat', 0)
        return node
    
    def __repr__(self) -> str:
        return f"Node({self.name}@{self.ip}:{self.port}, status={self.status.value})"
    
    def __eq__(self, other) -> bool:
        if isinstance(other, Node):
            return self.node_id == other.node_id
        return False
    
    def __hash__(self) -> int:
        return hash(self.node_id)
