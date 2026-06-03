"""测试统一配置管理。

合并原有 Config + ConfigStore 为单一 AscConfig 类。
支持：默认值、JSON 文件加载/保存、环境变量覆盖、运行时修改。
"""

import json
import os
import tempfile
from pathlib import Path

from asc.core.config import AscConfig


class TestAscConfigDefaults:
    """默认配置值。"""

    def test_default_node_port(self):
        cfg = AscConfig()
        assert cfg.get("node", "port") == 52415

    def test_default_discovery_port(self):
        cfg = AscConfig()
        assert cfg.get("network", "discovery_port") == 52416

    def test_default_inference_backend(self):
        cfg = AscConfig()
        assert cfg.get("inference", "backend") == "llama.cpp"

    def test_default_inference_max_tokens(self):
        cfg = AscConfig()
        assert cfg.get("inference", "max_tokens") == 128

    def test_default_api_port(self):
        cfg = AscConfig()
        assert cfg.get("api", "port") == 52415


class TestAscConfigGetSet:
    """配置读写。"""

    def test_get_existing_key(self):
        cfg = AscConfig()
        assert cfg.get("node", "port") == 52415

    def test_get_with_default(self):
        cfg = AscConfig()
        assert cfg.get("node", "nonexistent", 42) == 42

    def test_set_value(self):
        cfg = AscConfig()
        cfg.set("node", "port", 8080)
        assert cfg.get("node", "port") == 8080

    def test_set_creates_new_key(self):
        cfg = AscConfig()
        cfg.set("custom", "key", "value")
        assert cfg.get("custom", "key") == "value"


class TestAscConfigEnvOverride:
    """环境变量覆盖。"""

    def test_env_override(self):
        os.environ["ASC_NODE_PORT"] = "9999"
        try:
            cfg = AscConfig()
            assert cfg.get("node", "port") == 9999
        finally:
            del os.environ["ASC_NODE_PORT"]

    def test_env_override_string(self):
        os.environ["ASC_API_KEY"] = "test-key-123"
        try:
            cfg = AscConfig()
            assert cfg.get("api", "key") == "test-key-123"
        finally:
            del os.environ["ASC_API_KEY"]


class TestAscConfigFileIO:
    """配置文件读写。"""

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            cfg = AscConfig()
            cfg.set("node", "port", 9090)
            cfg.save(path)

            cfg2 = AscConfig()
            cfg2.load(path)
            assert cfg2.get("node", "port") == 9090

    def test_load_nonexistent_file_is_noop(self):
        cfg = AscConfig()
        cfg.load(Path("/nonexistent/config.json"))
        # 应保持默认值
        assert cfg.get("node", "port") == 52415

    def test_save_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "deep" / "dir" / "config.json"
            cfg = AscConfig()
            cfg.save(path)
            assert path.exists()


class TestAscConfigModelMapping:
    """模型映射管理。"""

    def test_add_model_mapping(self):
        cfg = AscConfig()
        cfg.add_model_mapping("llama-3.1-8b", "/models/llama-3.1-8b-q4.gguf")
        assert cfg.resolve_model_path("llama-3.1-8b") == "/models/llama-3.1-8b-q4.gguf"

    def test_remove_model_mapping(self):
        cfg = AscConfig()
        cfg.add_model_mapping("llama-3.1-8b", "/models/llama.gguf")
        cfg.remove_model_mapping("llama-3.1-8b")
        assert cfg.resolve_model_path("llama-3.1-8b") is None

    def test_resolve_nonexistent_model(self):
        cfg = AscConfig()
        assert cfg.resolve_model_path("nonexistent") is None

    def test_list_model_mappings(self):
        cfg = AscConfig()
        cfg.add_model_mapping("model-a", "/a.gguf")
        cfg.add_model_mapping("model-b", "/b.gguf")
        mappings = cfg.list_model_mappings()
        assert len(mappings) == 2
        assert "model-a" in mappings
        assert "model-b" in mappings


class TestAscConfigPersistence:
    """模型映射持久化。"""

    def test_model_mappings_survive_save_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            cfg = AscConfig()
            cfg.add_model_mapping("llama", "/models/llama.gguf")
            cfg.save(path)

            cfg2 = AscConfig()
            cfg2.load(path)
            assert cfg2.resolve_model_path("llama") == "/models/llama.gguf"
