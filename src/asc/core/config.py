"""Asc 统一配置管理。

合并原有 Config + ConfigStore 为单一 AscConfig 类，提供集中式的配置管理。
支持多种配置来源，按优先级覆盖：
1. 默认值（代码中硬编码）
2. JSON 配置文件（用户持久化设置）
3. 环境变量（运行时覆盖，适合容器化部署）
4. 运行时修改（程序动态调整）

环境变量映射：
    ASC_NODE_PORT       -> node.port
    ASC_API_PORT        -> api.port
    ASC_API_KEY         -> api.key
    ASC_MODELS_PATH     -> paths.models
    ASC_LLAMA_PATH      -> paths.llama_cpp

设计原则：
- 单一职责：所有配置相关逻辑集中在此模块，避免分散在多个文件中
- 来源透明：调用者无需关心配置来自默认值、文件还是环境变量
- 类型安全：环境变量自动尝试 int 转换，减少运行时类型错误
- 模型映射：支持别名 -> 路径映射，便于用户用简短名称引用模型

使用示例：
    config = AscConfig()
    config.load(Path("config.json"))  # 加载文件
    port = config.get("node", "port")  # 读取（自动包含环境变量覆盖）
    config.set("node", "port", 52416)  # 运行时修改
    config.save(Path("config.json"))   # 持久化
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# 默认配置
_DEFAULTS: dict[str, dict[str, Any]] = {
    "node": {
        "port": 52415,
        "name": "",
    },
    "network": {
        "discovery_port": 52416,
        "broadcast_interval": 3,
    },
    "inference": {
        "backend": "llama.cpp",
        "max_tokens": 128,
        "temperature": 0.7,
        "server_port": 8081,
    },
    "cluster": {
        "heartbeat_timeout": 30,
    },
    "api": {
        "port": 52415,
        "key": "",
    },
    "paths": {
        "models": "./models",
        "llama_cpp": "",
    },
}

# 环境变量映射：env_var -> (section, key)
_ENV_MAP: dict[str, tuple[str, str]] = {
    "ASC_NODE_PORT": ("node", "port"),
    "ASC_API_PORT": ("api", "port"),
    "ASC_API_KEY": ("api", "key"),
    "ASC_MODELS_PATH": ("paths", "models"),
    "ASC_LLAMA_PATH": ("paths", "llama_cpp"),
}


class AscConfig:
    """Asc 统一配置管理。"""

    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = {}
        self._model_mappings: dict[str, str] = {}
        self._load_defaults()
        self._apply_env_overrides()

    def _load_defaults(self) -> None:
        """加载默认配置。"""
        for section, values in _DEFAULTS.items():
            self._data[section] = dict(values)

    def _apply_env_overrides(self) -> None:
        """应用环境变量覆盖。"""
        for env_var, (section, key) in _ENV_MAP.items():
            value = os.getenv(env_var)
            if value is not None:
                # 尝试转换为整数
                try:
                    self._data.setdefault(section, {})[key] = int(value)
                except ValueError:
                    self._data.setdefault(section, {})[key] = value

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """获取配置值。"""
        return self._data.get(section, {}).get(key, default)

    def set(self, section: str, key: str, value: Any) -> None:
        """设置配置值。"""
        self._data.setdefault(section, {})[key] = value

    # --- 模型映射 ---

    def add_model_mapping(self, alias: str, path: str) -> None:
        """添加模型别名映射。"""
        self._model_mappings[alias] = path

    def remove_model_mapping(self, alias: str) -> None:
        """移除模型别名映射。"""
        self._model_mappings.pop(alias, None)

    def resolve_model_path(self, alias: str) -> str | None:
        """根据别名解析模型路径。"""
        return self._model_mappings.get(alias)

    def list_model_mappings(self) -> dict[str, str]:
        """列出所有模型映射。"""
        return dict(self._model_mappings)

    # --- 文件 I/O ---

    def save(self, path: Path) -> None:
        """保存配置到 JSON 文件。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "config": self._data,
            "model_mappings": self._model_mappings,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self, path: Path) -> None:
        """从 JSON 文件加载配置。"""
        if not path.exists():
            return
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        if "config" in data:
            for section, values in data["config"].items():
                self._data.setdefault(section, {}).update(values)

        if "model_mappings" in data:
            self._model_mappings.update(data["model_mappings"])

    def to_dict(self) -> dict[str, Any]:
        """导出为字典。"""
        return {
            "config": self._data,
            "model_mappings": self._model_mappings,
        }
