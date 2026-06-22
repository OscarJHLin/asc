"""测试 P0 安全修复。"""

import os
from unittest.mock import patch

from asc.api.auth import _init_default_store, require_api_key, validate_model_id


class TestRequireApiKey:
    """认证修复测试。"""

    def test_no_api_key_configured_rejects(self):
        """S-02/S-03: 未配置 ASC_API_KEY 时拒绝请求。"""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ASC_API_KEY", None)
            os.environ.pop("ASC_ALLOW_NO_AUTH", None)
            _init_default_store()
            assert require_api_key() is False

    def test_no_api_key_with_allow_no_auth_allows(self):
        """ASC_ALLOW_NO_AUTH=1 允许免认证访问（仅用于开发/测试环境）。"""
        with patch.dict(os.environ, {"ASC_ALLOW_NO_AUTH": "1"}, clear=True):
            os.environ.pop("ASC_API_KEY", None)
            _init_default_store()
            assert require_api_key() is True

    def test_correct_api_key_passes(self):
        """正确 API Key 通过认证。"""
        with patch.dict(os.environ, {"ASC_API_KEY": "test-key-123"}, clear=True):
            _init_default_store()
            assert require_api_key("test-key-123") is True

    def test_wrong_api_key_rejects(self):
        """错误 API Key 被拒绝。"""
        with patch.dict(os.environ, {"ASC_API_KEY": "test-key-123"}, clear=True):
            _init_default_store()
            assert require_api_key("wrong-key") is False

    def test_none_api_key_rejects(self):
        """无 API Key 被拒绝。"""
        with patch.dict(os.environ, {"ASC_API_KEY": "test-key-123"}, clear=True):
            _init_default_store()
            assert require_api_key(None) is False

    def test_empty_api_key_rejects(self):
        """空 API Key 被拒绝。"""
        with patch.dict(os.environ, {"ASC_API_KEY": "test-key-123"}, clear=True):
            _init_default_store()
            assert require_api_key("") is False


class TestTimingAttackPrevention:
    """S-04: 时序攻击防护测试。"""

    def test_hmac_compare_used(self):
        """验证使用 hmac.compare_digest 而非 ==。"""
        import inspect

        from asc.api.auth import ApiKeyStore

        source = inspect.getsource(ApiKeyStore.resolve_role)
        assert "hmac.compare_digest" in source

    def test_wrong_key_always_fails(self):
        """各种错误 Key 都被拒绝。"""
        with patch.dict(os.environ, {"ASC_API_KEY": "correct-key"}, clear=True):
            _init_default_store()
            assert require_api_key("wrong") is False
            assert require_api_key("correct-ke") is False
            assert require_api_key("correct-keyy") is False
            assert require_api_key("CORRECT-KEY") is False


class TestValidateModelId:
    """S-08: 路径遍历防护测试。"""

    def test_valid_model_id(self):
        assert validate_model_id("qwen-7b") is True

    def test_dot_dot_rejected(self):
        """.. 被拒绝。"""
        assert validate_model_id("../etc/passwd") is False

    def test_slash_rejected(self):
        """/ 被拒绝。"""
        assert validate_model_id("path/to/model") is False

    def test_backslash_rejected(self):
        """\\ 被拒绝。"""
        assert validate_model_id("path\\to\\model") is False

    def test_null_byte_rejected(self):
        """空字节被拒绝。"""
        assert validate_model_id("model\x00.gguf") is False

    def test_empty_rejected(self):
        """空 model_id 被拒绝。"""
        assert validate_model_id("") is False

    def test_normal_with_dashes(self):
        """正常带破折号的 model_id 通过。"""
        assert validate_model_id("Qwen2.5-7B-Instruct-Q4_K_M") is True

    def test_normal_with_dots(self):
        """正常带点（非..）的 model_id 通过。"""
        assert validate_model_id("Qwen2.5-7B") is True


class TestSyncPathTraversal:
    """S-08: sync.py 路径遍历防护测试。"""

    def test_path_traversal_in_prepare_send(self):
        """prepare_send 拒绝路径遍历 model_id。"""
        from pathlib import Path

        from asc.network.sync import ModelSyncProtocol

        protocol = ModelSyncProtocol(models_dir=Path("/tmp/models"))
        try:
            protocol.prepare_send("../../etc/passwd")
            raise AssertionError("Should raise ValueError")
        except ValueError as e:
            assert "非法字符" in str(e) or "路径遍历" in str(e)

    def test_path_traversal_in_get_chunk_data(self):
        """get_chunk_data 拒绝路径遍历 model_id。"""
        from pathlib import Path

        from asc.network.sync import ChunkInfo, ModelSyncProtocol

        protocol = ModelSyncProtocol(models_dir=Path("/tmp/models"))
        chunk = ChunkInfo(chunk_index=0, offset=0, size=10, sha256="abc")
        try:
            protocol.get_chunk_data("../../../etc/passwd", chunk)
            raise AssertionError("Should raise ValueError")
        except ValueError as e:
            assert "非法字符" in str(e) or "路径遍历" in str(e)

    def test_path_traversal_in_init_receive(self):
        """init_receive 拒绝路径遍历 model_id。"""
        from pathlib import Path

        from asc.network.sync import ModelSyncProtocol

        protocol = ModelSyncProtocol(models_dir=Path("/tmp/models"))
        try:
            protocol.init_receive("../../etc/passwd", 100, [], "sha256")
            raise AssertionError("Should raise ValueError")
        except ValueError as e:
            assert "非法字符" in str(e) or "路径遍历" in str(e)
