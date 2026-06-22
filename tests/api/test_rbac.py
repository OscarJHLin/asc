"""测试 RBAC 角色授权系统。

验证基于角色的访问控制：
- Role 枚举定义 admin 和 user 角色
- resolve_role() 根据 API Key 解析角色
- require_role() 检查 API Key 是否有所需角色
- 向后兼容 ASC_API_KEY 和 ASC_ADMIN_API_KEY
"""

import os
from unittest.mock import patch

from asc.api.auth import Role, _init_default_store, require_role, resolve_role


class TestRoleEnum:
    """Role 枚举测试。"""

    def test_admin_role(self):
        assert Role.ADMIN.value == "admin"

    def test_user_role(self):
        assert Role.USER.value == "user"


class TestResolveRole:
    """resolve_role() 角色解析测试。"""

    def test_admin_key_resolves_to_admin(self):
        """ASC_ADMIN_API_KEY 对应 admin 角色。"""
        with patch.dict(
            os.environ,
            {"ASC_API_KEY": "user-key", "ASC_ADMIN_API_KEY": "admin-key"},
            clear=True,
        ):
            _init_default_store()
            assert resolve_role("admin-key") == Role.ADMIN

    def test_user_key_resolves_to_user(self):
        """ASC_API_KEY 对应 user 角色。"""
        with patch.dict(
            os.environ,
            {"ASC_API_KEY": "user-key", "ASC_ADMIN_API_KEY": "admin-key"},
            clear=True,
        ):
            _init_default_store()
            assert resolve_role("user-key") == Role.USER

    def test_admin_key_also_has_user_role(self):
        """admin key 不应解析为 user（admin 是独立角色）。"""
        with patch.dict(
            os.environ,
            {"ASC_API_KEY": "user-key", "ASC_ADMIN_API_KEY": "admin-key"},
            clear=True,
        ):
            _init_default_store()
            assert resolve_role("admin-key") == Role.ADMIN
            assert resolve_role("admin-key") != Role.USER

    def test_unknown_key_resolves_to_none(self):
        """未知 key 解析为 None。"""
        with patch.dict(
            os.environ,
            {"ASC_API_KEY": "user-key", "ASC_ADMIN_API_KEY": "admin-key"},
            clear=True,
        ):
            _init_default_store()
            assert resolve_role("unknown-key") is None

    def test_none_key_resolves_to_none(self):
        """None key 解析为 None。"""
        with patch.dict(
            os.environ,
            {"ASC_API_KEY": "user-key"},
            clear=True,
        ):
            _init_default_store()
            assert resolve_role(None) is None

    def test_no_admin_key_user_key_still_works(self):
        """未配置 ASC_ADMIN_API_KEY 时，ASC_API_KEY 仍为 user 角色。"""
        with patch.dict(
            os.environ,
            {"ASC_API_KEY": "user-key"},
            clear=True,
        ):
            os.environ.pop("ASC_ADMIN_API_KEY", None)
            _init_default_store()
            assert resolve_role("user-key") == Role.USER

    def test_only_admin_key_configured(self):
        """仅配置 ASC_ADMIN_API_KEY 时，该 key 为 admin 角色。"""
        with patch.dict(
            os.environ,
            {"ASC_ADMIN_API_KEY": "admin-key"},
            clear=True,
        ):
            os.environ.pop("ASC_API_KEY", None)
            _init_default_store()
            assert resolve_role("admin-key") == Role.ADMIN


class TestRequireRole:
    """require_role() 角色检查测试。"""

    def test_admin_can_access_admin_endpoint(self):
        """admin 角色可以访问 admin 端点。"""
        with patch.dict(
            os.environ,
            {"ASC_API_KEY": "user-key", "ASC_ADMIN_API_KEY": "admin-key"},
            clear=True,
        ):
            _init_default_store()
            assert require_role("admin-key", Role.ADMIN) is True

    def test_user_cannot_access_admin_endpoint(self):
        """user 角色不能访问 admin 端点。"""
        with patch.dict(
            os.environ,
            {"ASC_API_KEY": "user-key", "ASC_ADMIN_API_KEY": "admin-key"},
            clear=True,
        ):
            _init_default_store()
            assert require_role("user-key", Role.ADMIN) is False

    def test_admin_can_access_user_endpoint(self):
        """admin 角色可以访问 user 端点（admin 继承 user 权限）。"""
        with patch.dict(
            os.environ,
            {"ASC_API_KEY": "user-key", "ASC_ADMIN_API_KEY": "admin-key"},
            clear=True,
        ):
            _init_default_store()
            assert require_role("admin-key", Role.USER) is True

    def test_user_can_access_user_endpoint(self):
        """user 角色可以访问 user 端点。"""
        with patch.dict(
            os.environ,
            {"ASC_API_KEY": "user-key", "ASC_ADMIN_API_KEY": "admin-key"},
            clear=True,
        ):
            _init_default_store()
            assert require_role("user-key", Role.USER) is True

    def test_unknown_key_cannot_access_any_endpoint(self):
        """未知 key 不能访问任何端点。"""
        with patch.dict(
            os.environ,
            {"ASC_API_KEY": "user-key", "ASC_ADMIN_API_KEY": "admin-key"},
            clear=True,
        ):
            _init_default_store()
            assert require_role("unknown-key", Role.USER) is False
            assert require_role("unknown-key", Role.ADMIN) is False

    def test_none_key_cannot_access_any_endpoint(self):
        """None key 不能访问任何端点。"""
        with patch.dict(
            os.environ,
            {"ASC_API_KEY": "user-key"},
            clear=True,
        ):
            _init_default_store()
            assert require_role(None, Role.USER) is False
            assert require_role(None, Role.ADMIN) is False


class TestRbacBackwardCompatibility:
    """RBAC 向后兼容性测试。"""

    def test_require_admin_still_works(self):
        """require_admin 仍能正常工作（内部使用 RBAC）。"""
        from asc.api.auth import require_admin

        with patch.dict(
            os.environ,
            {"ASC_API_KEY": "user-key", "ASC_ADMIN_API_KEY": "admin-key"},
            clear=True,
        ):
            _init_default_store()
            assert require_admin("admin-key") is True
            assert require_admin("user-key") is False

    def test_require_api_key_still_works(self):
        """require_api_key 仍能正常工作。"""
        from asc.api.auth import require_api_key

        with patch.dict(
            os.environ,
            {"ASC_API_KEY": "user-key"},
            clear=True,
        ):
            _init_default_store()
            assert require_api_key("user-key") is True
            assert require_api_key("wrong-key") is False

    def test_no_admin_key_fallback_to_user_key_for_admin(self):
        """未配置 ASC_ADMIN_API_KEY 时，admin 端点拒绝普通 user key。"""
        from asc.api.auth import require_admin

        with patch.dict(
            os.environ,
            {"ASC_API_KEY": "user-key"},
            clear=True,
        ):
            os.environ.pop("ASC_ADMIN_API_KEY", None)
            _init_default_store()
            # 没有 admin key，user key 不应获得 admin 权限
            assert require_admin("user-key") is False
