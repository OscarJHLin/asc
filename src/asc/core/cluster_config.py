"""管理员集群配置管理模块。

提供管理员对集群资源的统一配置能力：
- 节点资源策略：内存卸载比例、显存/存储限制与预留
- 模型配置：上下文长度、最大 token 数、GPU 层数、批处理大小等
- 集群参数：心跳间隔、容量上报间隔、节点超时等

设计原则：
- 层级覆盖：节点策略 > 默认策略 > 硬编码默认值
- 持久化：配置变更自动保存到 JSON 文件
- 校验：所有配置项均有合理范围约束
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class NodeResourcePolicy:
    """节点资源策略配置。

    管理员可控制每个节点的资源使用上限和预留量。

    Attributes:
        memory_limit_mb: 内存使用上限 (None=不限制)
        memory_offload_ratio: 内存卸载到磁盘的比例 [0.0, 1.0]
        vram_limit_mb: 显存使用上限 (None=不限制)
        vram_reserve_mb: 显存预留量 (MB)，不分配给模型
        disk_limit_mb: 磁盘使用上限 (None=不限制)
        disk_reserve_mb: 磁盘预留量 (MB)，保持可用空间
    """

    memory_limit_mb: int | None = None
    memory_offload_ratio: float = 0.0
    vram_limit_mb: int | None = None
    vram_reserve_mb: int = 512
    disk_limit_mb: int | None = None
    disk_reserve_mb: int = 1024

    def __post_init__(self) -> None:
        if not 0.0 <= self.memory_offload_ratio <= 1.0:
            raise ValueError(
                f"memory_offload_ratio 必须在 [0.0, 1.0] 范围内，当前值: {self.memory_offload_ratio}"
            )

    def effective_memory_mb(self) -> float | None:
        """计算有效可用内存 (MB)，考虑卸载比例。

        Returns:
            扣除卸载比例后的内存限制，若未设置限制则返回 None
        """
        if self.memory_limit_mb is None:
            return None
        return self.memory_limit_mb * (1.0 - self.memory_offload_ratio)

    def effective_vram_mb(self) -> int | None:
        """计算有效可用显存 (MB)，扣除预留量。

        Returns:
            扣除预留后的显存限制，若未设置限制则返回 None
        """
        if self.vram_limit_mb is None:
            return None
        return max(0, self.vram_limit_mb - self.vram_reserve_mb)

    def effective_disk_mb(self) -> int | None:
        """计算有效可用磁盘 (MB)，扣除预留量。"""
        if self.disk_limit_mb is None:
            return None
        return max(0, self.disk_limit_mb - self.disk_reserve_mb)

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_limit_mb": self.memory_limit_mb,
            "memory_offload_ratio": self.memory_offload_ratio,
            "vram_limit_mb": self.vram_limit_mb,
            "vram_reserve_mb": self.vram_reserve_mb,
            "disk_limit_mb": self.disk_limit_mb,
            "disk_reserve_mb": self.disk_reserve_mb,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NodeResourcePolicy:
        return cls(
            memory_limit_mb=d.get("memory_limit_mb"),
            memory_offload_ratio=d.get("memory_offload_ratio", 0.0),
            vram_limit_mb=d.get("vram_limit_mb"),
            vram_reserve_mb=d.get("vram_reserve_mb", 512),
            disk_limit_mb=d.get("disk_limit_mb"),
            disk_reserve_mb=d.get("disk_reserve_mb", 1024),
        )


@dataclass
class ModelConfig:
    """模型配置。

    管理员可为每个部署的模型设置推理参数。

    Attributes:
        model_id: 模型标识符
        context_length: 上下文窗口长度 (tokens)
        max_tokens: 最大生成 token 数
        temperature: 采样温度
        top_p: Top-p 采样参数
        gpu_layers: 加载到 GPU 的层数 (-1=全部)
        batch_size: 批处理大小
        num_parallel: 并行推理数
    """

    model_id: str
    model_domain_type: str = "llm"  # "llm" / "embedding"
    context_length: int = 4096
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    gpu_layers: int = -1
    batch_size: int = 512
    num_parallel: int = 1

    # GPU 管理策略
    gpu_offload_ratio: str | float = "max"  # "max"/"off"/0-1 数值
    gpu_split_strategy: str = "evenly"  # "evenly"/"priorityOrder"/"custom"
    gpu_custom_ratio: list[float] = field(default_factory=list)  # custom 策略的比例
    disabled_gpus: list[int] = field(default_factory=list)  # 禁用的 GPU 索引

    # 模型加载参数
    flash_attention: bool = True
    offload_kv_cache_to_gpu: bool = True
    use_fp16_for_kv_cache: bool = True
    try_mmap: bool = True
    try_direct_io: bool = False
    eval_batch_size: int = 512
    physical_batch_size: int = 0  # 0 = auto
    keep_model_in_memory: bool = True
    gpu_strict_vram_cap: bool = False

    # 推理参数
    min_p_sampling: float = 0.0  # 0 = disabled
    repeat_penalty: float = 0.0  # 0 = disabled
    presence_penalty: float = 0.0  # 0 = disabled
    frequency_penalty: float = 0.0  # 0 = disabled
    context_overflow_policy: str = "truncateMiddle"  # stopAtLimit/truncateMiddle/rollingWindow
    reasoning_parsing_enabled: bool = True
    reasoning_start_string: str = "<think>"
    reasoning_end_string: str = "</think>"
    # 推测解码
    speculative_draft_model: str = ""
    speculative_draft_max_tokens: int = 0
    speculative_draft_min_continue_probability: float = 0.0

    def __post_init__(self) -> None:
        if self.context_length <= 0:
            raise ValueError(f"context_length 必须 > 0，当前值: {self.context_length}")
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens 必须 > 0，当前值: {self.max_tokens}")
        # max_tokens 不应超过 context_length
        if self.max_tokens > self.context_length:
            logger.warning(
                "max_tokens(%d) > context_length(%d)，自动修正为 context_length",
                self.max_tokens, self.context_length,
            )
            self.max_tokens = self.context_length

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_domain_type": self.model_domain_type,
            "context_length": self.context_length,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "gpu_layers": self.gpu_layers,
            "batch_size": self.batch_size,
            "num_parallel": self.num_parallel,
            # GPU 管理策略
            "gpu_offload_ratio": self.gpu_offload_ratio,
            "gpu_split_strategy": self.gpu_split_strategy,
            "gpu_custom_ratio": self.gpu_custom_ratio,
            "disabled_gpus": self.disabled_gpus,
            # 模型加载参数
            "flash_attention": self.flash_attention,
            "offload_kv_cache_to_gpu": self.offload_kv_cache_to_gpu,
            "use_fp16_for_kv_cache": self.use_fp16_for_kv_cache,
            "try_mmap": self.try_mmap,
            "try_direct_io": self.try_direct_io,
            "eval_batch_size": self.eval_batch_size,
            "physical_batch_size": self.physical_batch_size,
            "keep_model_in_memory": self.keep_model_in_memory,
            "gpu_strict_vram_cap": self.gpu_strict_vram_cap,
            # 推理参数
            "min_p_sampling": self.min_p_sampling,
            "repeat_penalty": self.repeat_penalty,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "context_overflow_policy": self.context_overflow_policy,
            "reasoning_parsing_enabled": self.reasoning_parsing_enabled,
            "reasoning_start_string": self.reasoning_start_string,
            "reasoning_end_string": self.reasoning_end_string,
            # 推测解码
            "speculative_draft_model": self.speculative_draft_model,
            "speculative_draft_max_tokens": self.speculative_draft_max_tokens,
            "speculative_draft_min_continue_probability": self.speculative_draft_min_continue_probability,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelConfig:
        return cls(
            model_id=d["model_id"],
            model_domain_type=d.get("model_domain_type", "llm"),
            context_length=d.get("context_length", 4096),
            max_tokens=d.get("max_tokens", 2048),
            temperature=d.get("temperature", 0.7),
            top_p=d.get("top_p", 0.9),
            gpu_layers=d.get("gpu_layers", -1),
            batch_size=d.get("batch_size", 512),
            num_parallel=d.get("num_parallel", 1),
            # GPU 管理策略
            gpu_offload_ratio=d.get("gpu_offload_ratio", "max"),
            gpu_split_strategy=d.get("gpu_split_strategy", "evenly"),
            gpu_custom_ratio=d.get("gpu_custom_ratio", []),
            disabled_gpus=d.get("disabled_gpus", []),
            # 模型加载参数
            flash_attention=d.get("flash_attention", True),
            offload_kv_cache_to_gpu=d.get("offload_kv_cache_to_gpu", True),
            use_fp16_for_kv_cache=d.get("use_fp16_for_kv_cache", True),
            try_mmap=d.get("try_mmap", True),
            try_direct_io=d.get("try_direct_io", False),
            eval_batch_size=d.get("eval_batch_size", 512),
            physical_batch_size=d.get("physical_batch_size", 0),
            keep_model_in_memory=d.get("keep_model_in_memory", True),
            gpu_strict_vram_cap=d.get("gpu_strict_vram_cap", False),
            # 推理参数
            min_p_sampling=d.get("min_p_sampling", 0.0),
            repeat_penalty=d.get("repeat_penalty", 0.0),
            presence_penalty=d.get("presence_penalty", 0.0),
            frequency_penalty=d.get("frequency_penalty", 0.0),
            context_overflow_policy=d.get("context_overflow_policy", "truncateMiddle"),
            reasoning_parsing_enabled=d.get("reasoning_parsing_enabled", True),
            reasoning_start_string=d.get("reasoning_start_string", "<think>"),
            reasoning_end_string=d.get("reasoning_end_string", "</think>"),
            # 推测解码
            speculative_draft_model=d.get("speculative_draft_model", ""),
            speculative_draft_max_tokens=d.get("speculative_draft_max_tokens", 0),
            speculative_draft_min_continue_probability=d.get("speculative_draft_min_continue_probability", 0.0),
        )


@dataclass
class ClusterConfig:
    """集群全局配置。

    管理员可通过此配置控制：
    1. 默认资源策略（适用于所有节点）
    2. 特定节点的资源策略（覆盖默认）
    3. 每个模型的推理参数
    4. 集群运行参数

    使用方式：
        config = ClusterConfig()
        config.load(Path("cluster_config.json"))

        # 设置默认资源策略
        config.default_resource_policy = NodeResourcePolicy(memory_limit_mb=32768)

        # 为特定节点设置策略
        config.set_node_policy("worker-01", NodeResourcePolicy(vram_limit_mb=14336))

        # 添加模型配置
        config.add_model_config(ModelConfig(model_id="qwen2.5-7b", context_length=8192))

        # 持久化
        config.save(Path("cluster_config.json"))
    """

    default_resource_policy: NodeResourcePolicy = field(default_factory=NodeResourcePolicy)
    model_configs: dict[str, ModelConfig] = field(default_factory=dict)
    node_policies: dict[str, NodeResourcePolicy] = field(default_factory=dict)
    heartbeat_interval: int = 5
    capacity_report_interval: int = 30
    node_timeout: int = 30

    def get_node_policy(self, node_id: str) -> NodeResourcePolicy:
        """获取节点资源策略，未配置时回退到默认策略。"""
        return self.node_policies.get(node_id, self.default_resource_policy)

    def set_node_policy(self, node_id: str, policy: NodeResourcePolicy) -> None:
        """设置特定节点的资源策略。"""
        self.node_policies[node_id] = policy

    def remove_node_policy(self, node_id: str) -> None:
        """删除节点策略，回退到默认。"""
        self.node_policies.pop(node_id, None)

    def get_model_config(self, model_id: str) -> ModelConfig | None:
        """获取模型配置。"""
        return self.model_configs.get(model_id)

    def add_model_config(self, config: ModelConfig) -> None:
        """添加或更新模型配置。"""
        self.model_configs[config.model_id] = config

    def remove_model_config(self, model_id: str) -> None:
        """删除模型配置。"""
        self.model_configs.pop(model_id, None)

    def update(self, updates: dict[str, Any]) -> None:
        """通过 API 风格的部分更新修改集群参数。"""
        if "heartbeat_interval" in updates:
            self.heartbeat_interval = int(updates["heartbeat_interval"])
        if "capacity_report_interval" in updates:
            self.capacity_report_interval = int(updates["capacity_report_interval"])
        if "node_timeout" in updates:
            self.node_timeout = int(updates["node_timeout"])
        if "default_resource_policy" in updates:
            self.default_resource_policy = NodeResourcePolicy.from_dict(
                updates["default_resource_policy"]
            )

    def update_model(self, model_id: str, updates: dict[str, Any]) -> None:
        """通过 API 更新模型配置，不存在时创建。"""
        existing = self.model_configs.get(model_id)
        if existing is not None:
            d = existing.to_dict()
            d.update(updates)
            d["model_id"] = model_id
            self.model_configs[model_id] = ModelConfig.from_dict(d)
        else:
            updates_copy = dict(updates)
            updates_copy["model_id"] = model_id
            self.model_configs[model_id] = ModelConfig.from_dict(updates_copy)

    def update_node_policy(self, node_id: str, updates: dict[str, Any]) -> None:
        """通过 API 更新节点策略，不存在时创建。"""
        existing = self.node_policies.get(node_id)
        if existing is not None:
            d = existing.to_dict()
            d.update(updates)
            self.node_policies[node_id] = NodeResourcePolicy.from_dict(d)
        else:
            self.node_policies[node_id] = NodeResourcePolicy.from_dict(updates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "default_resource_policy": self.default_resource_policy.to_dict(),
            "model_configs": {k: v.to_dict() for k, v in self.model_configs.items()},
            "node_policies": {k: v.to_dict() for k, v in self.node_policies.items()},
            "heartbeat_interval": self.heartbeat_interval,
            "capacity_report_interval": self.capacity_report_interval,
            "node_timeout": self.node_timeout,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ClusterConfig:
        config = cls()
        if "default_resource_policy" in d:
            config.default_resource_policy = NodeResourcePolicy.from_dict(
                d["default_resource_policy"]
            )
        if "model_configs" in d:
            for model_id, model_d in d["model_configs"].items():
                config.model_configs[model_id] = ModelConfig.from_dict(model_d)
        if "node_policies" in d:
            for node_id, policy_d in d["node_policies"].items():
                config.node_policies[node_id] = NodeResourcePolicy.from_dict(policy_d)
        config.heartbeat_interval = d.get("heartbeat_interval", 5)
        config.capacity_report_interval = d.get("capacity_report_interval", 30)
        config.node_timeout = d.get("node_timeout", 30)
        return config

    def save(self, path: Path) -> None:
        """持久化到 JSON 文件。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Cluster config saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> ClusterConfig:
        """从 JSON 文件加载配置，文件不存在时返回默认配置。"""
        if not path.exists():
            logger.info("Config file not found: %s, using defaults", path)
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)
