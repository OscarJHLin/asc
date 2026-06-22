"""测试管理员集群配置管理模块。"""

from __future__ import annotations

import pytest

from asc.core.cluster_config import (
    ClusterConfig,
    ModelConfig,
    NodeResourcePolicy,
)

# ------------------------------------------------------------------
# NodeResourcePolicy 测试
# ------------------------------------------------------------------


class TestNodeResourcePolicy:
    """测试节点资源策略配置。"""

    def test_default_values(self):
        """默认值合理：不限制资源。"""
        policy = NodeResourcePolicy()
        assert policy.memory_limit_mb is None
        assert policy.memory_offload_ratio == 0.0
        assert policy.vram_limit_mb is None
        assert policy.vram_reserve_mb == 512
        assert policy.disk_limit_mb is None
        assert policy.disk_reserve_mb == 1024

    def test_custom_values(self):
        """自定义资源限制。"""
        policy = NodeResourcePolicy(
            memory_limit_mb=32768,
            memory_offload_ratio=0.3,
            vram_limit_mb=14336,
            vram_reserve_mb=1024,
            disk_limit_mb=500000,
            disk_reserve_mb=5000,
        )
        assert policy.memory_limit_mb == 32768
        assert policy.memory_offload_ratio == 0.3
        assert policy.vram_limit_mb == 14336
        assert policy.vram_reserve_mb == 1024
        assert policy.disk_limit_mb == 500000
        assert policy.disk_reserve_mb == 5000

    def test_offload_ratio_validation_valid(self):
        """offload_ratio 在 [0.0, 1.0] 范围内合法。"""
        policy = NodeResourcePolicy(memory_offload_ratio=0.0)
        assert policy.memory_offload_ratio == 0.0
        policy = NodeResourcePolicy(memory_offload_ratio=1.0)
        assert policy.memory_offload_ratio == 1.0
        policy = NodeResourcePolicy(memory_offload_ratio=0.5)
        assert policy.memory_offload_ratio == 0.5

    def test_offload_ratio_validation_invalid(self):
        """offload_ratio 超出 [0.0, 1.0] 范围抛异常。"""
        with pytest.raises(ValueError, match="memory_offload_ratio"):
            NodeResourcePolicy(memory_offload_ratio=-0.1)
        with pytest.raises(ValueError, match="memory_offload_ratio"):
            NodeResourcePolicy(memory_offload_ratio=1.5)

    def test_to_dict(self):
        """能序列化为字典。"""
        policy = NodeResourcePolicy(memory_limit_mb=16384, memory_offload_ratio=0.2)
        d = policy.to_dict()
        assert d["memory_limit_mb"] == 16384
        assert d["memory_offload_ratio"] == 0.2
        assert d["memory_offload_ratio"] is not None

    def test_from_dict(self):
        """能从字典反序列化。"""
        d = {
            "memory_limit_mb": 16384,
            "memory_offload_ratio": 0.2,
            "vram_limit_mb": None,
            "vram_reserve_mb": 512,
            "disk_limit_mb": None,
            "disk_reserve_mb": 1024,
        }
        policy = NodeResourcePolicy.from_dict(d)
        assert policy.memory_limit_mb == 16384
        assert policy.memory_offload_ratio == 0.2
        assert policy.vram_reserve_mb == 512

    def test_from_dict_missing_fields_use_defaults(self):
        """缺失字段使用默认值。"""
        d = {}
        policy = NodeResourcePolicy.from_dict(d)
        assert policy.vram_reserve_mb == 512
        assert policy.disk_reserve_mb == 1024
        assert policy.memory_offload_ratio == 0.0

    def test_effective_memory_limit_with_offload(self):
        """计算有效内存限制时考虑 offload 比例。"""
        policy = NodeResourcePolicy(
            memory_limit_mb=32768,
            memory_offload_ratio=0.3,
        )
        effective = policy.effective_memory_mb()
        # 32768 * (1 - 0.3) = 22937.6
        assert abs(effective - 22937.6) < 0.1

    def test_effective_memory_limit_no_limit(self):
        """无内存限制时返回 None。"""
        policy = NodeResourcePolicy()
        assert policy.effective_memory_mb() is None

    def test_effective_vram_limit(self):
        """计算有效显存限制。"""
        policy = NodeResourcePolicy(
            vram_limit_mb=16384,
            vram_reserve_mb=1024,
        )
        effective = policy.effective_vram_mb()
        assert effective == 15360  # 16384 - 1024 (reserve subtracted from limit)

    def test_effective_vram_no_limit(self):
        """无显存限制时返回 None。"""
        policy = NodeResourcePolicy()
        assert policy.effective_vram_mb() is None


# ------------------------------------------------------------------
# ModelConfig 测试
# ------------------------------------------------------------------


class TestModelConfig:
    """测试模型配置。"""

    def test_default_values(self):
        """默认值合理。"""
        config = ModelConfig(model_id="qwen2.5-7b")
        assert config.model_id == "qwen2.5-7b"
        assert config.model_domain_type == "llm"
        assert config.context_length == 4096
        assert config.max_tokens == 2048
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.gpu_layers == -1  # 全部加载到 GPU
        assert config.batch_size == 512
        assert config.num_parallel == 1

    def test_custom_values(self):
        """自定义模型配置。"""
        config = ModelConfig(
            model_id="qwen2.5-72b",
            context_length=32768,
            max_tokens=8192,
            temperature=0.5,
            gpu_layers=40,
            batch_size=1024,
            num_parallel=4,
        )
        assert config.context_length == 32768
        assert config.max_tokens == 8192
        assert config.gpu_layers == 40
        assert config.num_parallel == 4

    def test_context_length_validation(self):
        """context_length 必须 > 0。"""
        with pytest.raises(ValueError, match="context_length"):
            ModelConfig(model_id="test", context_length=0)
        with pytest.raises(ValueError, match="context_length"):
            ModelConfig(model_id="test", context_length=-1)

    def test_max_tokens_validation(self):
        """max_tokens 必须 > 0 且 <= context_length。"""
        with pytest.raises(ValueError, match="max_tokens"):
            ModelConfig(model_id="test", max_tokens=0)
        # max_tokens > context_length 时自动修正
        config = ModelConfig(model_id="test", context_length=2048, max_tokens=4096)
        assert config.max_tokens == 2048

    def test_to_dict(self):
        """能序列化为字典。"""
        config = ModelConfig(model_id="qwen2.5-7b", context_length=8192)
        d = config.to_dict()
        assert d["model_id"] == "qwen2.5-7b"
        assert d["context_length"] == 8192

    def test_from_dict(self):
        """能从字典反序列化。"""
        d = {
            "model_id": "qwen2.5-7b",
            "context_length": 8192,
            "max_tokens": 2048,
            "temperature": 0.7,
            "top_p": 0.9,
            "gpu_layers": -1,
            "batch_size": 512,
            "num_parallel": 1,
        }
        config = ModelConfig.from_dict(d)
        assert config.model_id == "qwen2.5-7b"
        assert config.context_length == 8192

    def test_from_dict_missing_fields_use_defaults(self):
        """缺失字段使用默认值。"""
        d = {"model_id": "test"}
        config = ModelConfig.from_dict(d)
        assert config.context_length == 4096
        assert config.gpu_layers == -1

    def test_model_domain_type_default(self):
        """默认 model_domain_type 为 llm。"""
        config = ModelConfig(model_id="test")
        assert config.model_domain_type == "llm"

    def test_model_domain_type_embedding(self):
        """可以设置为 embedding 类型。"""
        config = ModelConfig(model_id="test-embed", model_domain_type="embedding")
        assert config.model_domain_type == "embedding"

    def test_model_domain_type_serialization(self):
        """model_domain_type 能正确序列化和反序列化。"""
        config = ModelConfig(model_id="test-embed", model_domain_type="embedding")
        d = config.to_dict()
        assert d["model_domain_type"] == "embedding"
        loaded = ModelConfig.from_dict(d)
        assert loaded.model_domain_type == "embedding"

    def test_model_domain_type_from_dict_default(self):
        """from_dict 缺失 model_domain_type 时使用默认值 llm。"""
        d = {"model_id": "test"}
        config = ModelConfig.from_dict(d)
        assert config.model_domain_type == "llm"


# ------------------------------------------------------------------
# ClusterConfig 测试
# ------------------------------------------------------------------


class TestClusterConfig:
    """测试集群全局配置。"""

    def test_default_values(self):
        """默认配置合理。"""
        config = ClusterConfig()
        assert isinstance(config.default_resource_policy, NodeResourcePolicy)
        assert isinstance(config.model_configs, dict)
        assert len(config.model_configs) == 0
        assert config.heartbeat_interval == 5
        assert config.capacity_report_interval == 30
        assert config.node_timeout == 30

    def test_set_resource_policy(self):
        """设置默认资源策略。"""
        config = ClusterConfig()
        policy = NodeResourcePolicy(memory_limit_mb=32768, memory_offload_ratio=0.3)
        config.default_resource_policy = policy
        assert config.default_resource_policy.memory_limit_mb == 32768

    def test_set_node_resource_policy(self):
        """为特定节点设置资源策略。"""
        config = ClusterConfig()
        policy = NodeResourcePolicy(vram_limit_mb=14336)
        config.set_node_policy("worker-01", policy)
        assert config.get_node_policy("worker-01").vram_limit_mb == 14336

    def test_get_node_policy_fallback_to_default(self):
        """未配置特定节点策略时回退到默认策略。"""
        config = ClusterConfig()
        policy = NodeResourcePolicy(memory_limit_mb=16384)
        config.default_resource_policy = policy
        result = config.get_node_policy("unknown-node")
        assert result.memory_limit_mb == 16384

    def test_add_model_config(self):
        """添加模型配置。"""
        config = ClusterConfig()
        model = ModelConfig(model_id="qwen2.5-7b", context_length=8192)
        config.add_model_config(model)
        assert "qwen2.5-7b" in config.model_configs
        assert config.model_configs["qwen2.5-7b"].context_length == 8192

    def test_remove_model_config(self):
        """删除模型配置。"""
        config = ClusterConfig()
        model = ModelConfig(model_id="qwen2.5-7b")
        config.add_model_config(model)
        config.remove_model_config("qwen2.5-7b")
        assert "qwen2.5-7b" not in config.model_configs

    def test_get_model_config(self):
        """获取模型配置。"""
        config = ClusterConfig()
        model = ModelConfig(model_id="qwen2.5-7b", context_length=8192)
        config.add_model_config(model)
        result = config.get_model_config("qwen2.5-7b")
        assert result.context_length == 8192

    def test_get_model_config_not_found(self):
        """模型不存在时返回 None。"""
        config = ClusterConfig()
        assert config.get_model_config("nonexistent") is None

    def test_to_dict(self):
        """能序列化为字典。"""
        config = ClusterConfig()
        config.add_model_config(ModelConfig(model_id="qwen2.5-7b"))
        config.set_node_policy("worker-01", NodeResourcePolicy(vram_limit_mb=16384))

        d = config.to_dict()
        assert "default_resource_policy" in d
        assert "model_configs" in d
        assert "node_policies" in d
        assert "qwen2.5-7b" in d["model_configs"]
        assert "worker-01" in d["node_policies"]

    def test_from_dict(self):
        """能从字典反序列化。"""
        d = {
            "default_resource_policy": {
                "memory_limit_mb": 16384,
                "memory_offload_ratio": 0.2,
                "vram_limit_mb": None,
                "vram_reserve_mb": 512,
                "disk_limit_mb": None,
                "disk_reserve_mb": 1024,
            },
            "model_configs": {
                "qwen2.5-7b": {
                    "model_id": "qwen2.5-7b",
                    "context_length": 8192,
                    "max_tokens": 2048,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "gpu_layers": -1,
                    "batch_size": 512,
                    "num_parallel": 1,
                },
            },
            "node_policies": {
                "worker-01": {
                    "memory_limit_mb": None,
                    "memory_offload_ratio": 0.0,
                    "vram_limit_mb": 16384,
                    "vram_reserve_mb": 512,
                    "disk_limit_mb": None,
                    "disk_reserve_mb": 1024,
                },
            },
            "heartbeat_interval": 5,
            "capacity_report_interval": 30,
            "node_timeout": 30,
        }
        config = ClusterConfig.from_dict(d)
        assert config.default_resource_policy.memory_limit_mb == 16384
        assert "qwen2.5-7b" in config.model_configs
        assert config.get_node_policy("worker-01").vram_limit_mb == 16384

    def test_save_and_load(self, tmp_path):
        """能持久化到 JSON 文件并加载。"""
        config = ClusterConfig()
        config.add_model_config(ModelConfig(model_id="qwen2.5-7b", context_length=8192))
        config.set_node_policy("worker-01", NodeResourcePolicy(vram_limit_mb=16384))

        path = tmp_path / "cluster_config.json"
        config.save(path)

        assert path.exists()
        loaded = ClusterConfig.load(path)
        assert loaded.get_model_config("qwen2.5-7b").context_length == 8192
        assert loaded.get_node_policy("worker-01").vram_limit_mb == 16384

    def test_load_nonexistent_file(self, tmp_path):
        """加载不存在的文件返回默认配置。"""
        path = tmp_path / "nonexistent.json"
        config = ClusterConfig.load(path)
        assert isinstance(config, ClusterConfig)
        assert len(config.model_configs) == 0

    def test_update_from_api(self):
        """通过 API 风格的部分更新修改配置。"""
        config = ClusterConfig()
        config.update({
            "heartbeat_interval": 10,
            "node_timeout": 60,
        })
        assert config.heartbeat_interval == 10
        assert config.node_timeout == 60

    def test_update_model_config_from_api(self):
        """通过 API 更新模型配置。"""
        config = ClusterConfig()
        config.update_model("qwen2.5-7b", {"context_length": 16384, "temperature": 0.5})
        model = config.get_model_config("qwen2.5-7b")
        assert model.context_length == 16384
        assert model.temperature == 0.5
        # 未指定的字段使用默认值
        assert model.gpu_layers == -1

    def test_update_node_policy_from_api(self):
        """通过 API 更新节点策略。"""
        config = ClusterConfig()
        config.update_node_policy("worker-01", {"vram_limit_mb": 14336, "memory_offload_ratio": 0.3})
        policy = config.get_node_policy("worker-01")
        assert policy.vram_limit_mb == 14336
        assert policy.memory_offload_ratio == 0.3

    def test_remove_node_policy(self):
        """删除节点策略，回退到默认。"""
        config = ClusterConfig()
        config.set_node_policy("worker-01", NodeResourcePolicy(vram_limit_mb=14336))
        config.remove_node_policy("worker-01")
        # 应该回退到默认策略
        policy = config.get_node_policy("worker-01")
        assert policy.vram_limit_mb is None  # 默认值
