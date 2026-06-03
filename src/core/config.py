"""
配置管理模块

管理全局配置，支持配置文件读写和环境变量
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """全局配置类"""
    
    DEFAULT_CONFIG = {
        'node': {
            'name': None,  # 自动获取主机名
            'port': 52415,
            'rpc_port': 50052,
            'heartbeat_interval': 5,
            'timeout': 10,
        },
        'network': {
            'discovery_port': 52416,
            'broadcast_interval': 3,
            'max_nodes': 16,
            'preferred_ip_prefix': ['192.168.', '10.', '172.'],
        },
        'inference': {
            'backend': 'llama.cpp',
            'max_tokens': 2048,
            'temperature': 0.7,
            'top_p': 0.9,
            'batch_size': 512,
        },
        'cluster': {
            'auto_discover': True,
            'load_balance': 'round_robin',
            'fault_tolerance': True,
            'replica_count': 1,
        },
        'ui': {
            'enabled': True,
            'port': 8080,
            'refresh_interval': 2,
        },
        'paths': {
            'models': './models',
            'logs': './logs',
            'temp': './temp',
        }
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置
        
        Args:
            config_path: 配置文件路径，默认使用内置配置
        """
        self._config = self._load_config(config_path)
        self._apply_env_overrides()
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """加载配置文件"""
        config = self.DEFAULT_CONFIG.copy()
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                self._deep_update(config, user_config)
        
        return config
    
    def _deep_update(self, base: Dict, update: Dict) -> None:
        """深度更新字典"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value
    
    def _apply_env_overrides(self) -> None:
        """应用环境变量覆盖"""
        env_mappings = {
            'EXOLLAMA_NODE_PORT': ('node', 'port'),
            'EXOLLAMA_NODE_NAME': ('node', 'name'),
            'EXOLLAMA_RPC_PORT': ('node', 'rpc_port'),
            'EXOLLAMA_DISCOVERY_PORT': ('network', 'discovery_port'),
            'EXOLLAMA_UI_PORT': ('ui', 'port'),
            'EXOLLAMA_MODELS_PATH': ('paths', 'models'),
        }
        
        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # 尝试转换为整数
                try:
                    value = int(value)
                except ValueError:
                    pass
                self._config[section][key] = value
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            section: 配置段
            key: 配置键
            default: 默认值
            
        Returns:
            配置值
        """
        return self._config.get(section, {}).get(key, default)
    
    def set(self, section: str, key: str, value: Any) -> None:
        """
        设置配置值
        
        Args:
            section: 配置段
            key: 配置键
            value: 配置值
        """
        if section not in self._config:
            self._config[section] = {}
        self._config[section][key] = value
    
    def save(self, config_path: str) -> None:
        """
        保存配置到文件
        
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self._config, f, indent=2, ensure_ascii=False)
    
    @property
    def all(self) -> Dict[str, Any]:
        """获取所有配置"""
        return self._config.copy()
    
    def __repr__(self) -> str:
        return f"Config({json.dumps(self._config, indent=2)})"
