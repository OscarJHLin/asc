"""
集群管理模块

管理多个节点的集群，实现节点发现、状态同步和任务调度
"""

import threading
import time
from typing import Callable, Dict, List, Optional

from .config import Config
from .node import Node, NodeStatus


class Cluster:
    """
    集群类
    
    管理节点集合，提供集群级别的操作
    """
    
    def __init__(self, config: Config = None):
        """
        初始化集群
        
        Args:
            config: 配置对象
        """
        self.config = config or Config()
        self.nodes: Dict[str, Node] = {}
        self.local_node: Optional[Node] = None
        
        self._lock = threading.RLock()
        self._running = False
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable] = []
        
        self._init_local_node()
    
    def _init_local_node(self) -> None:
        """初始化本地节点"""
        self.local_node = Node(
            name=self.config.get('node', 'name'),
            port=self.config.get('node', 'port', 52415),
            rpc_port=self.config.get('node', 'rpc_port', 50052)
        )
        self.local_node.update_status(NodeStatus.ONLINE)
        
        with self._lock:
            self.nodes[self.local_node.node_id] = self.local_node
    
    def start(self) -> None:
        """启动集群管理"""
        if self._running:
            return
        
        self._running = True
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True
        )
        self._heartbeat_thread.start()
        
        print(f"[Cluster] 集群管理已启动，本地节点: {self.local_node}")
    
    def stop(self) -> None:
        """停止集群管理"""
        self._running = False
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=5)
        
        print("[Cluster] 集群管理已停止")
    
    def _heartbeat_loop(self) -> None:
        """心跳检测循环"""
        interval = self.config.get('node', 'heartbeat_interval', 5)
        timeout = self.config.get('node', 'timeout', 30)
        
        while self._running:
            time.sleep(interval)
            
            with self._lock:
                # 更新本地节点心跳
                if self.local_node:
                    self.local_node.heartbeat()
                
                # 检查远程节点状态
                dead_nodes = []
                for node_id, node in self.nodes.items():
                    if node_id == self.local_node.node_id:
                        continue
                    
                    if not node.is_alive(timeout):
                        dead_nodes.append(node_id)
                        print(f"[Cluster] 节点离线: {node}")
                
                # 移除离线节点
                for node_id in dead_nodes:
                    del self.nodes[node_id]
                    self._notify_change('node_removed', node_id)
    
    def add_node(self, node: Node) -> bool:
        """
        添加节点到集群
        
        Args:
            node: 要添加的节点
            
        Returns:
            是否添加成功
        """
        with self._lock:
            if node.node_id in self.nodes:
                # 更新现有节点
                self.nodes[node.node_id] = node
                self._notify_change('node_updated', node)
                return True
            
            max_nodes = self.config.get('network', 'max_nodes', 16)
            if len(self.nodes) >= max_nodes:
                print(f"[Cluster] 集群已满，无法添加节点: {node}")
                return False
            
            self.nodes[node.node_id] = node
            print(f"[Cluster] 节点加入: {node}")
            self._notify_change('node_added', node)
            return True
    
    def remove_node(self, node_id: str) -> bool:
        """
        从集群移除节点
        
        Args:
            node_id: 节点ID
            
        Returns:
            是否移除成功
        """
        with self._lock:
            if node_id not in self.nodes:
                return False
            
            node = self.nodes.pop(node_id)
            print(f"[Cluster] 节点移除: {node}")
            self._notify_change('node_removed', node)
            return True
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """
        获取节点
        
        Args:
            node_id: 节点ID
            
        Returns:
            节点对象或None
        """
        with self._lock:
            return self.nodes.get(node_id)
    
    def get_online_nodes(self) -> List[Node]:
        """
        获取所有在线节点
        
        Returns:
            在线节点列表
        """
        with self._lock:
            return [
                node for node in self.nodes.values()
                if node.status == NodeStatus.ONLINE
            ]
    
    def get_available_nodes(self) -> List[Node]:
        """
        获取可用节点（在线且非忙碌）
        
        Returns:
            可用节点列表
        """
        with self._lock:
            return [
                node for node in self.nodes.values()
                if node.status in (NodeStatus.ONLINE, NodeStatus.BUSY)
            ]
    
    def select_node_for_task(self) -> Optional[Node]:
        """
        选择节点执行任务
        
        Returns:
            选中的节点或None
        """
        available = self.get_available_nodes()
        if not available:
            return None
        
        # 简单的负载均衡：选择CPU使用率最低的节点
        return min(available, key=lambda n: n.resources.cpu_percent)
    
    def get_cluster_info(self) -> Dict:
        """
        获取集群信息
        
        Returns:
            集群信息字典
        """
        with self._lock:
            online_count = sum(
                1 for n in self.nodes.values()
                if n.status == NodeStatus.ONLINE
            )
            
            total_cpu = sum(n.resources.cpu_count for n in self.nodes.values())
            total_memory = sum(n.resources.memory_total for n in self.nodes.values())
            
            return {
                'node_count': len(self.nodes),
                'online_count': online_count,
                'total_cpu_cores': total_cpu,
                'total_memory_mb': total_memory,
                'nodes': [n.to_dict() for n in self.nodes.values()],
            }
    
    def register_callback(self, callback: Callable) -> None:
        """
        注册状态变更回调
        
        Args:
            callback: 回调函数，接收(event_type, data)参数
        """
        self._callbacks.append(callback)
    
    def _notify_change(self, event_type: str, data) -> None:
        """通知状态变更"""
        for callback in self._callbacks:
            try:
                callback(event_type, data)
            except Exception as e:
                print(f"[Cluster] 回调执行失败: {e}")
    
    def __len__(self) -> int:
        """返回节点数量"""
        with self._lock:
            return len(self.nodes)
    
    def __contains__(self, node_id: str) -> bool:
        """检查节点是否在集群中"""
        with self._lock:
            return node_id in self.nodes
    
    def __repr__(self) -> str:
        with self._lock:
            return f"Cluster(nodes={len(self.nodes)}, online={len(self.get_online_nodes())})"
