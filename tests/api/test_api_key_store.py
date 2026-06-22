"""测试 ApiKeyStore — 多密钥管理和密钥轮换。

TDD 测试优先于实现，覆盖：
- 多密钥支持（逗号分隔、文件加载）
- 角色解析（ADMIN/USER）
- 密钥轮换（热重载、动态增删）
- 时序攻击防护
- 向后兼容
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from asc.api.auth import Role


class TestApiKeyStoreBasic:
    """ApiKeyStore 基础功能测试。"""

    def test_create_with_direct_keys(self):
        """直接注入密钥创建 store。"""
        from asc.api.auth import ApiKeyStore

        store = ApiKeyStore(user_keys=["key1", "key2"], admin_keys=["admin1"])
        assert store.key_count == 3

    def test_empty_store(self):
        """空 store 拒绝所有密钥。"""
        from asc.api.auth import ApiKeyStore

        store = ApiKeyStore()
        assert store.key_count == 0
        assert store.resolve_role("any-key") is None

    def test_resolve_role_user_key(self):
        """用户密钥解析为 USER 角色。"""
        from asc.api.auth import ApiKeyStore

        store = ApiKeyStore(user_keys=["user-key"])
        assert store.resolve_role("user-key") == Role.USER

    def test_resolve_role_admin_key(self):
        """管理员密钥解析为 ADMIN 角色。"""
        from asc.api.auth import ApiKeyStore

        store = ApiKeyStore(admin_keys=["admin-key"])
        assert store.resolve_role("admin-key") == Role.ADMIN

    def test_resolve_role_unknown_key(self):
        """未知密钥返回 None。"""
        from asc.api.auth import ApiKeyStore

        store = ApiKeyStore(user_keys=["user-key"])
        assert store.resolve_role("unknown") is None

    def test_resolve_role_none_key(self):
        """None 密钥返回 None。"""
        from asc.api.auth import ApiKeyStore

        store = ApiKeyStore(user_keys=["user-key"])
        assert store.resolve_role(None) is None

    def test_multiple_user_keys(self):
        """多个用户密钥都能解析为 USER。"""
        from asc.api.auth import ApiKeyStore

        store = ApiKeyStore(user_keys=["key1", "key2", "key3"])
        assert store.resolve_role("key1") == Role.USER
        assert store.resolve_role("key2") == Role.USER
        assert store.resolve_role("key3") == Role.USER

    def test_multiple_admin_keys(self):
        """多个管理员密钥都能解析为 ADMIN。"""
        from asc.api.auth import ApiKeyStore

        store = ApiKeyStore(admin_keys=["admin1", "admin2"])
        assert store.resolve_role("admin1") == Role.ADMIN
        assert store.resolve_role("admin2") == Role.ADMIN


class TestApiKeyStoreRequire:
    """ApiKeyStore require_* 方法测试。"""

    def test_require_api_key_valid_user(self):
        """用户密钥通过 require_api_key。"""
        from asc.api.auth import ApiKeyStore

        store = ApiKeyStore(user_keys=["user-key"])
        assert store.require_api_key("user-key") is True

    def test_require_api_key_valid_admin(self):
        """管理员密钥也通过 require_api_key（ADMIN 继承 USER 权限）。"""
        from asc.api.auth import ApiKeyStore

        store = ApiKeyStore(admin_keys=["admin-key"])
        assert store.require_api_key("admin-key") is True

    def test_require_api_key_invalid(self):
        """无效密钥被拒绝。"""
        from asc.api.auth import ApiKeyStore

        store = ApiKeyStore(user_keys=["user-key"])
        assert store.require_api_key("wrong") is False

    def test_require_api_key_none(self):
        """None 密钥被拒绝。"""
        from asc.api.auth import ApiKeyStore

        store = ApiKeyStore(user_keys=["user-key"])
        assert store.require_api_key(None) is False

    def test_require_api_key_empty_store(self):
        """空 store 拒绝所有请求。"""
        from asc.api.auth import ApiKeyStore

        store = ApiKeyStore()
        assert store.require_api_key("any") is False

    def test_require_admin_valid(self):
        """管理员密钥通过 require_admin。"""
        from asc.api.auth import ApiKeyStore

        store = ApiKeyStore(admin_keys=["admin-key"])
        assert store.require_admin("admin-key") is True

    def test_require_admin_user_key_rejected(self):
        """用户密钥不能通过 require_admin。"""
        from asc.api.auth import ApiKeyStore

        store = ApiKeyStore(user_keys=["user-key"], admin_keys=["admin-key"])
        assert store.require_admin("user-key") is False

    def test_require_admin_invalid(self):
        """无效密钥不能通过 require_admin。"""
        from asc.api.auth import ApiKeyStore

        store = ApiKeyStore(admin_keys=["admin-key"])
        assert store.require_admin("wrong") is False


class TestApiKeyStoreFromEnv:
    """从环境变量加载密钥测试。"""

    def test_from_env_single_key(self):
        """从 ASC_API_KEY 加载单个用户密钥。"""
        from asc.api.auth import ApiKeyStore

        with patch.dict(os.environ, {"ASC_API_KEY": "my-key"}, clear=True):
            os.environ.pop("ASC_API_KEYS", None)
            os.environ.pop("ASC_ADMIN_API_KEY", None)
            os.environ.pop("ASC_ADMIN_API_KEYS", None)
            store = ApiKeyStore.from_env()
            assert store.resolve_role("my-key") == Role.USER

    def test_from_env_comma_separated_user_keys(self):
        """从 ASC_API_KEYS 加载逗号分隔的多个用户密钥。"""
        from asc.api.auth import ApiKeyStore

        with patch.dict(os.environ, {"ASC_API_KEYS": "key1,key2,key3"}, clear=True):
            os.environ.pop("ASC_API_KEY", None)
            os.environ.pop("ASC_ADMIN_API_KEY", None)
            os.environ.pop("ASC_ADMIN_API_KEYS", None)
            store = ApiKeyStore.from_env()
            assert store.resolve_role("key1") == Role.USER
            assert store.resolve_role("key2") == Role.USER
            assert store.resolve_role("key3") == Role.USER

    def test_from_env_comma_separated_admin_keys(self):
        """从 ASC_ADMIN_API_KEYS 加载逗号分隔的多个管理员密钥。"""
        from asc.api.auth import ApiKeyStore

        with patch.dict(
            os.environ,
            {"ASC_ADMIN_API_KEYS": "admin1,admin2"},
            clear=True,
        ):
            os.environ.pop("ASC_API_KEY", None)
            os.environ.pop("ASC_API_KEYS", None)
            os.environ.pop("ASC_ADMIN_API_KEY", None)
            store = ApiKeyStore.from_env()
            assert store.resolve_role("admin1") == Role.ADMIN
            assert store.resolve_role("admin2") == Role.ADMIN

    def test_from_env_single_admin_key(self):
        """从 ASC_ADMIN_API_KEY 加载单个管理员密钥。"""
        from asc.api.auth import ApiKeyStore

        with patch.dict(
            os.environ,
            {"ASC_API_KEY": "user1", "ASC_ADMIN_API_KEY": "admin1"},
            clear=True,
        ):
            os.environ.pop("ASC_API_KEYS", None)
            os.environ.pop("ASC_ADMIN_API_KEYS", None)
            store = ApiKeyStore.from_env()
            assert store.resolve_role("user1") == Role.USER
            assert store.resolve_role("admin1") == Role.ADMIN

    def test_from_env_both_single_and_multi(self):
        """同时设置 ASC_API_KEY 和 ASC_API_KEYS 时合并。"""
        from asc.api.auth import ApiKeyStore

        with patch.dict(
            os.environ,
            {"ASC_API_KEY": "single-key", "ASC_API_KEYS": "multi1,multi2"},
            clear=True,
        ):
            os.environ.pop("ASC_ADMIN_API_KEY", None)
            os.environ.pop("ASC_ADMIN_API_KEYS", None)
            store = ApiKeyStore.from_env()
            assert store.resolve_role("single-key") == Role.USER
            assert store.resolve_role("multi1") == Role.USER
            assert store.resolve_role("multi2") == Role.USER

    def test_from_env_empty(self):
        """无环境变量时创建空 store。"""
        from asc.api.auth import ApiKeyStore

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ASC_API_KEY", None)
            os.environ.pop("ASC_API_KEYS", None)
            os.environ.pop("ASC_ADMIN_API_KEY", None)
            os.environ.pop("ASC_ADMIN_API_KEYS", None)
            store = ApiKeyStore.from_env()
            assert store.key_count == 0

    def test_from_env_trims_whitespace(self):
        """逗号分隔密钥自动去除空白。"""
        from asc.api.auth import ApiKeyStore

        with patch.dict(os.environ, {"ASC_API_KEYS": " key1 , key2 , key3 "}, clear=True):
            os.environ.pop("ASC_API_KEY", None)
            os.environ.pop("ASC_ADMIN_API_KEY", None)
            os.environ.pop("ASC_ADMIN_API_KEYS", None)
            store = ApiKeyStore.from_env()
            assert store.resolve_role("key1") == Role.USER
            assert store.resolve_role("key2") == Role.USER


class TestApiKeyStoreFromFile:
    """从文件加载密钥测试。"""

    def test_from_file_user_keys(self):
        """从文件加载用户密钥（每行一个）。"""
        from asc.api.auth import ApiKeyStore

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("user-key-1\n")
            f.write("user-key-2\n")
            f.write("# 这是注释\n")
            f.write("\n")  # 空行
            f.write("user-key-3\n")
            path = Path(f.name)

        try:
            store = ApiKeyStore.from_file(path, role=Role.USER)
            assert store.resolve_role("user-key-1") == Role.USER
            assert store.resolve_role("user-key-2") == Role.USER
            assert store.resolve_role("user-key-3") == Role.USER
            assert store.key_count == 3  # 注释和空行不计
        finally:
            os.unlink(path)

    def test_from_file_admin_keys(self):
        """从文件加载管理员密钥。"""
        from asc.api.auth import ApiKeyStore

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("admin-key-1\n")
            f.write("admin-key-2\n")
            path = Path(f.name)

        try:
            store = ApiKeyStore.from_file(path, role=Role.ADMIN)
            assert store.resolve_role("admin-key-1") == Role.ADMIN
            assert store.resolve_role("admin-key-2") == Role.ADMIN
        finally:
            os.unlink(path)

    def test_from_file_nonexistent_raises(self):
        """文件不存在时抛出 FileNotFoundError。"""
        from asc.api.auth import ApiKeyStore

        with pytest.raises(FileNotFoundError):
            ApiKeyStore.from_file(Path("/nonexistent/keys.txt"), role=Role.USER)


class TestApiKeyStoreRotation:
    """密钥轮换测试。"""

    def test_add_user_key(self):
        """动态添加用户密钥。"""
        from asc.api.auth import ApiKeyStore

        store = ApiKeyStore(user_keys=["key1"])
        store.add_key("key2", Role.USER)
        assert store.resolve_role("key2") == Role.USER
        assert store.key_count == 2

    def test_add_admin_key(self):
        """动态添加管理员密钥。"""
        from asc.api.auth import ApiKeyStore

        store = ApiKeyStore()
        store.add_key("new-admin", Role.ADMIN)
        assert store.resolve_role("new-admin") == Role.ADMIN

    def test_remove_key(self):
        """动态删除密钥。"""
        from asc.api.auth import ApiKeyStore

        store = ApiKeyStore(user_keys=["key1", "key2"])
        store.remove_key("key1")
        assert store.resolve_role("key1") is None
        assert store.resolve_role("key2") == Role.USER
        assert store.key_count == 1

    def test_remove_nonexistent_key_noop(self):
        """删除不存在的密钥无副作用。"""
        from asc.api.auth import ApiKeyStore

        store = ApiKeyStore(user_keys=["key1"])
        store.remove_key("nonexistent")
        assert store.key_count == 1

    def test_add_duplicate_key_ignored(self):
        """添加重复密钥被忽略。"""
        from asc.api.auth import ApiKeyStore

        store = ApiKeyStore(user_keys=["key1"])
        store.add_key("key1", Role.USER)
        assert store.key_count == 1

    def test_reload_from_env(self):
        """从环境变量重新加载密钥（轮换场景）。"""
        from asc.api.auth import ApiKeyStore

        with patch.dict(os.environ, {"ASC_API_KEY": "old-key"}, clear=True):
            os.environ.pop("ASC_API_KEYS", None)
            os.environ.pop("ASC_ADMIN_API_KEY", None)
            os.environ.pop("ASC_ADMIN_API_KEYS", None)
            store = ApiKeyStore.from_env()
            assert store.resolve_role("old-key") == Role.USER

        # 模拟环境变量变更（轮换后）
        with patch.dict(os.environ, {"ASC_API_KEY": "new-key"}, clear=True):
            os.environ.pop("ASC_API_KEYS", None)
            os.environ.pop("ASC_ADMIN_API_KEY", None)
            os.environ.pop("ASC_ADMIN_API_KEYS", None)
            store.reload()
            assert store.resolve_role("old-key") is None
            assert store.resolve_role("new-key") == Role.USER

    def test_reload_from_file(self):
        """从文件重新加载密钥（轮换场景）。"""
        from asc.api.auth import ApiKeyStore

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("key-v1\n")
            f.flush()
            path = Path(f.name)

        try:
            store = ApiKeyStore.from_file(path, role=Role.USER)
            assert store.resolve_role("key-v1") == Role.USER

            # 更新文件（轮换）
            with open(path, "w") as f:
                f.write("key-v2\n")

            store.reload()
            assert store.resolve_role("key-v1") is None
            assert store.resolve_role("key-v2") == Role.USER
        finally:
            os.unlink(path)

    def test_rotation_grace_period(self):
        """密钥轮换时新旧密钥共存（宽限期）。

        实际场景：先添加新密钥，确认生效后再删除旧密钥。
        """
        from asc.api.auth import ApiKeyStore

        store = ApiKeyStore(user_keys=["old-key"])
        # 添加新密钥（宽限期开始）
        store.add_key("new-key", Role.USER)
        assert store.resolve_role("old-key") == Role.USER
        assert store.resolve_role("new-key") == Role.USER
        # 确认新密钥生效后删除旧密钥
        store.remove_key("old-key")
        assert store.resolve_role("old-key") is None
        assert store.resolve_role("new-key") == Role.USER


class TestApiKeyStoreTimingAttack:
    """时序攻击防护测试。"""

    def test_uses_hmac_compare(self):
        """ApiKeyStore 使用 hmac.compare_digest。"""
        import inspect

        from asc.api.auth import ApiKeyStore

        source = inspect.getsource(ApiKeyStore.resolve_role)
        assert "hmac.compare_digest" in source

    def test_wrong_keys_always_fail(self):
        """各种错误密钥都被拒绝。"""
        from asc.api.auth import ApiKeyStore

        store = ApiKeyStore(user_keys=["correct-key"])
        assert store.require_api_key("wrong") is False
        assert store.require_api_key("correct-ke") is False
        assert store.require_api_key("correct-keyy") is False
        assert store.require_api_key("CORRECT-KEY") is False


class TestApiKeyStoreBackwardCompat:
    """向后兼容测试 — auth.py 函数委托到 ApiKeyStore。"""

    def test_require_api_key_uses_store(self):
        """require_api_key 委托到全局 ApiKeyStore。"""
        from asc.api.auth import require_api_key

        with patch.dict(os.environ, {"ASC_API_KEY": "test-key"}, clear=True):
            os.environ.pop("ASC_API_KEYS", None)
            os.environ.pop("ASC_ALLOW_NO_AUTH", None)
            # 重新初始化全局 store
            from asc.api import auth
            auth._init_default_store()
            assert require_api_key("test-key") is True
            assert require_api_key("wrong") is False

    def test_resolve_role_uses_store(self):
        """resolve_role 委托到全局 ApiKeyStore。"""
        from asc.api.auth import resolve_role

        with patch.dict(
            os.environ,
            {"ASC_API_KEY": "user1", "ASC_ADMIN_API_KEY": "admin1"},
            clear=True,
        ):
            os.environ.pop("ASC_API_KEYS", None)
            os.environ.pop("ASC_ADMIN_API_KEYS", None)
            from asc.api import auth
            auth._init_default_store()
            assert resolve_role("user1") == Role.USER
            assert resolve_role("admin1") == Role.ADMIN

    def test_multi_key_env_works_with_require_api_key(self):
        """ASC_API_KEYS 环境变量通过 require_api_key 生效。"""
        from asc.api.auth import require_api_key

        with patch.dict(os.environ, {"ASC_API_KEYS": "key1,key2,key3"}, clear=True):
            os.environ.pop("ASC_API_KEY", None)
            os.environ.pop("ASC_ALLOW_NO_AUTH", None)
            from asc.api import auth
            auth._init_default_store()
            assert require_api_key("key1") is True
            assert require_api_key("key2") is True
            assert require_api_key("key3") is True
            assert require_api_key("key4") is False
