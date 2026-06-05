"""
节点发现模块

实现UDP广播和HTTP健康检查的节点发现机制
"""

import json
import socket
import threading
import time
import urllib.request
from typing import Callable, Dict, List, Optional

from ..core.config import Config
from ..core.node import Node, NodeStatus


class NodeDiscovery:
    """
    节点发现类
    
    通过UDP广播和HTTP健康检查发现内网节点
    """
    
    DISCOVERY_MAGIC = "ASC_DISCOVER"
    RESPONSE_MAGIC = "ASC_RESPONSE"
    
    def __init__(self, config: Config = None, 
                 on_node_found: Callable[[Node], None] = None):
        """
        初始化节点发现
        
        Args:
            config: 配置对象
            on_node_found: 发现新节点时的回调
        """
        self.config = config or Config()
        self.on_node_found = on_node_found
        
        self.discovery_port = self.config.get('network', 'discovery_port', 52416)
        self.broadcast_interval = self.config.get('network', 'broadcast_interval', 3)
        
        self._running = False
        self._broadcast_thread: Optional[threading.Thread] = None
        self._listen_thread: Optional[threading.Thread] = None
        self._scan_thread: Optional[threading.Thread] = None
        
        self._local_ip = self._get_local_ip()
        self._broadcast_socket: Optional[socket.socket] = None
    
    def _get_local_ip(self) -> str:
        """获取本地IP"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"
    
    def start(self) -> None:
        """启动节点发现"""
        if self._running:
            return
        
        self._running = True
        
        # 创建广播socket
        self._broadcast_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._broadcast_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self._broadcast_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._broadcast_socket.bind(("0.0.0.0", self.discovery_port))
        self._broadcast_socket.settimeout(1.0)
        
        # 启动广播线程
        self._broadcast_thread = threading.Thread(
            target=self._broadcast_loop,
            daemon=True
        )
        self._broadcast_thread.start()
        
        # 启动监听线程
        self._listen_thread = threading.Thread(
            target=self._listen_loop,
            daemon=True
        )
        self._listen_thread.start()
        
        # 启动扫描线程
        self._scan_thread = threading.Thread(
            target=self._scan_loop,
            daemon=True
        )
        self._scan_thread.start()
        
        print(f"[Discovery] 节点发现已启动 (端口: {self.discovery_port})")
    
    def stop(self) -> None:
        """停止节点发现"""
        self._running = False
        
        if self._broadcast_socket:
            self._broadcast_socket.close()
        
        for thread in [self._broadcast_thread, self._listen_thread, self._scan_thread]:
            if thread:
                thread.join(timeout=2)
        
        print("[Discovery] 节点发现已停止")
    
    def _broadcast_loop(self) -> None:
        """广播循环"""
        message = json.dumps({
            'magic': self.DISCOVERY_MAGIC,
            'ip': self._local_ip,
            'port': self.config.get('node', 'port', 52415),
            'rpc_port': self.config.get('node', 'rpc_port', 50052),
            'timestamp': time.time(),
        })
        
        while self._running:
            try:
                self._broadcast_socket.sendto(
                    message.encode(),
                    ('<broadcast>', self.discovery_port)
                )
            except Exception as e:
                if self._running:
                    print(f"[Discovery] 广播失败: {e}")
            
            time.sleep(self.broadcast_interval)
    
    def _listen_loop(self) -> None:
        """监听循环"""
        while self._running:
            try:
                data, addr = self._broadcast_socket.recvfrom(1024)
                message = json.loads(data.decode())
                
                # 忽略自己的广播
                if message.get('ip') == self._local_ip:
                    continue
                
                # 检查是否是发现请求
                if message.get('magic') == self.DISCOVERY_MAGIC:
                    self._handle_discovery(message, addr)
                
                # 检查是否是响应
                elif message.get('magic') == self.RESPONSE_MAGIC:
                    self._handle_response(message)
                    
            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    print(f"[Discovery] 监听错误: {e}")
    
    def _handle_discovery(self, message: Dict, addr) -> None:
        """处理发现请求"""
        # 发送响应
        response = json.dumps({
            'magic': self.RESPONSE_MAGIC,
            'ip': self._local_ip,
            'port': self.config.get('node', 'port', 52415),
            'rpc_port': self.config.get('node', 'rpc_port', 50052),
            'name': socket.gethostname(),
            'timestamp': time.time(),
        })
        
        try:
            self._broadcast_socket.sendto(
                response.encode(),
                (addr[0], self.discovery_port)
            )
        except Exception as e:
            print(f"[Discovery] 响应失败: {e}")
    
    def _handle_response(self, message: Dict) -> None:
        """处理发现响应"""
        node_ip = message.get('ip')
        node_port = message.get('port', 52415)
        
        # 检查节点是否可达
        if self._check_node_health(node_ip, node_port):
            node = Node(
                name=message.get('name', 'unknown'),
                ip=node_ip,
                port=node_port,
                rpc_port=message.get('rpc_port', 50052)
            )
            node.update_status(NodeStatus.ONLINE)
            
            if self.on_node_found:
                self.on_node_found(node)
    
    def _scan_loop(self) -> None:
        """主动扫描循环"""
        while self._running:
            # 扫描已知网段
            subnet = '.'.join(self._local_ip.split('.')[:3])
            
            for i in range(1, 255):
                if not self._running:
                    break
                
                ip = f"{subnet}.{i}"
                if ip == self._local_ip:
                    continue
                
                # 快速检查端口是否开放
                if self._check_port_open(ip, self.config.get('node', 'port', 52415)):
                    # 进一步检查健康状态
                    if self._check_node_health(ip):
                        node = Node(ip=ip)
                        node.update_status(NodeStatus.ONLINE)
                        
                        if self.on_node_found:
                            self.on_node_found(node)
                
                # 避免扫描过快
                if i % 10 == 0:
                    time.sleep(0.1)
            
            # 每30秒扫描一次
            time.sleep(30)
    
    def _check_port_open(self, ip: str, port: int, timeout: float = 0.5) -> bool:
        """检查端口是否开放"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((ip, port))
            sock.close()
            return result == 0
        except Exception:
            return False
    
    def _check_node_health(self, ip: str, port: int = None) -> bool:
        """检查节点健康状态"""
        port = port or self.config.get('node', 'port', 52415)
        
        try:
            url = f"http://{ip}:{port}/health"
            req = urllib.request.Request(url, method='GET')
            with urllib.request.urlopen(req, timeout=2) as response:
                return response.status == 200
        except Exception:
            return False
    
    def discover_nodes(self) -> List[Node]:
        """
        手动触发节点发现
        
        Returns:
            发现的节点列表
        """
        nodes = []
        subnet = '.'.join(self._local_ip.split('.')[:3])
        
        for i in range(1, 255):
            ip = f"{subnet}.{i}"
            if ip == self._local_ip:
                continue
            
            if self._check_node_health(ip):
                node = Node(ip=ip)
                node.update_status(NodeStatus.ONLINE)
                nodes.append(node)
        
        return nodes
