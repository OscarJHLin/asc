"""Asc API 认证中间件。

提供基于角色的访问控制（RBAC）：
- Role.USER: 普通用户，可访问推理端点
- Role.ADMIN: 管理员，可访问管理端点（继承 USER 权限）

角色映射：
- ASC_API_KEY / ASC_API_KEYS 对应 Role.USER
- ASC_ADMIN_API_KEY / ASC_ADMIN_API_KEYS 对应 Role.ADMIN

安全策略：
- 使用 hmac.compare_digest 防止时序攻击
- 未配置 Key 时拒绝所有请求
- 支持多密钥和密钥轮换（通过 ApiKeyStore）
"""

from __future__ import annotations

import enum
import hmac
import logging
import os
import secrets
from pathlib import Path

logger = logging.getLogger(__name__)


class Role(enum.Enum):
    """用户角色。"""

    ADMIN = "admin"
    USER = "user"


class ApiKeyStore:
    """集中式 API Key 管理，支持多密钥和密钥轮换。

    设计原则：
    - 所有密钥查找使用 hmac.compare_digest 防时序攻击
    - 支持从环境变量、文件、直接注入加载密钥
    - 支持 reload() 热重载，无需重启服务
    - 支持动态 add_key / remove_key 实现轮换宽限期
    """

    def __init__(
        self,
        user_keys: list[str] | None = None,
        admin_keys: list[str] | None = None,
    ) -> None:
        self._user_keys: set[str] = set(user_keys or [])
        self._admin_keys: set[str] = set(admin_keys or [])
        # 记录加载来源，用于 reload
        self._source: str = "direct"  # "direct" | "env" | "file"
        self._file_path: Path | None = None
        self._file_role: Role | None = None

    @property
    def key_count(self) -> int:
        """已注册的密钥总数。"""
        return len(self._user_keys) + len(self._admin_keys)

    def resolve_role(self, api_key: str | None) -> Role | None:
        """根据 API Key 解析用户角色。

        优先检查 admin 密钥，再检查 user 密钥。
        使用 hmac.compare_digest 防止时序攻击。
        """
        if api_key is None:
            return None

        # 优先检查 admin keys
        for key in self._admin_keys:
            if hmac.compare_digest(api_key, key):
                return Role.ADMIN

        # 检查 user keys
        for key in self._user_keys:
            if hmac.compare_digest(api_key, key):
                return Role.USER

        return None

    def require_api_key(self, api_key: str | None = None) -> bool:
        """验证 API Key 是否为有效的 user 或 admin 密钥。

        当未配置任何 Key 时：
        - 若 ASC_ALLOW_NO_AUTH=1，允许所有请求（不推荐用于生产）
        - 否则拒绝所有请求
        """
        if self.key_count == 0:
            return os.getenv("ASC_ALLOW_NO_AUTH", "0") == "1"

        if api_key is None:
            return False

        # 检查所有密钥（admin 也继承 user 权限）
        for key in self._admin_keys:
            if hmac.compare_digest(api_key, key):
                return True
        for key in self._user_keys:
            if hmac.compare_digest(api_key, key):
                return True

        return False

    def require_admin(self, api_key: str | None = None) -> bool:
        """验证管理员权限。

        当未配置任何 Key 时：
        - 若 ASC_ALLOW_NO_AUTH=1，允许所有请求（不推荐用于生产）
        - 否则拒绝所有请求
        """
        if self.key_count == 0:
            return os.getenv("ASC_ALLOW_NO_AUTH", "0") == "1"

        return self.resolve_role(api_key) == Role.ADMIN

    def add_key(self, key: str, role: Role) -> None:
        """动态添加密钥。"""
        target = self._admin_keys if role == Role.ADMIN else self._user_keys
        target.add(key)

    def remove_key(self, key: str) -> None:
        """动态删除密钥。"""
        self._user_keys.discard(key)
        self._admin_keys.discard(key)

    def reload(self) -> None:
        """从来源重新加载密钥（热轮换）。"""
        if self._source == "env":
            new_store = ApiKeyStore.from_env()
            self._user_keys = new_store._user_keys
            self._admin_keys = new_store._admin_keys
        elif self._source == "file" and self._file_path is not None and self._file_role is not None:
            new_store = ApiKeyStore.from_file(self._file_path, self._file_role)
            if self._file_role == Role.USER:
                self._user_keys = new_store._user_keys
            else:
                self._admin_keys = new_store._admin_keys

    @classmethod
    def from_env(cls) -> ApiKeyStore:
        """从环境变量加载密钥。

        支持的环境变量：
        - ASC_API_KEY: 单个用户密钥
        - ASC_API_KEYS: 逗号分隔的多个用户密钥
        - ASC_ADMIN_API_KEY: 单个管理员密钥
        - ASC_ADMIN_API_KEYS: 逗号分隔的多个管理员密钥
        """
        user_keys: list[str] = []
        admin_keys: list[str] = []

        # 加载用户密钥
        single_user = os.getenv("ASC_API_KEY")
        if single_user:
            user_keys.append(single_user.strip())

        multi_user = os.getenv("ASC_API_KEYS")
        if multi_user:
            user_keys.extend(k.strip() for k in multi_user.split(",") if k.strip())

        # 加载管理员密钥
        single_admin = os.getenv("ASC_ADMIN_API_KEY")
        if single_admin:
            admin_keys.append(single_admin.strip())

        multi_admin = os.getenv("ASC_ADMIN_API_KEYS")
        if multi_admin:
            admin_keys.extend(k.strip() for k in multi_admin.split(",") if k.strip())

        store = cls(user_keys=user_keys, admin_keys=admin_keys)
        store._source = "env"
        return store

    @classmethod
    def from_file(cls, path: Path, role: Role) -> ApiKeyStore:
        """从文件加载密钥（每行一个，# 开头为注释，空行忽略）。"""
        if not path.exists():
            raise FileNotFoundError(f"密钥文件不存在: {path}")

        keys: list[str] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    keys.append(line)

        store = cls(
            user_keys=keys if role == Role.USER else [],
            admin_keys=keys if role == Role.ADMIN else [],
        )
        store._source = "file"
        store._file_path = path
        store._file_role = role
        return store


# ---------------------------------------------------------------------------
# 全局默认 store — 向后兼容
# ---------------------------------------------------------------------------

_default_store: ApiKeyStore | None = None


def _init_default_store() -> None:
    """初始化全局默认 ApiKeyStore（从环境变量加载）。"""
    global _default_store
    _default_store = ApiKeyStore.from_env()


def _get_default_store() -> ApiKeyStore:
    """获取全局默认 store，懒初始化。"""
    global _default_store
    if _default_store is None:
        _init_default_store()
    return _default_store


def set_key_store(store: ApiKeyStore) -> None:
    """替换全局 ApiKeyStore（用于测试或自定义注入）。"""
    global _default_store
    _default_store = store


# ---------------------------------------------------------------------------
# 公共 API — 向后兼容，委托到 ApiKeyStore
# ---------------------------------------------------------------------------


def resolve_role(api_key: str | None) -> Role | None:
    """根据 API Key 解析用户角色。

    委托到全局 ApiKeyStore.resolve_role()。
    支持多密钥（ASC_API_KEYS / ASC_ADMIN_API_KEYS）。
    """
    return _get_default_store().resolve_role(api_key)


def require_role(api_key: str | None, required_role: Role) -> bool:
    """检查 API Key 是否具有所需角色。

    权限继承：ADMIN 角色自动拥有 USER 权限。
    """
    role = resolve_role(api_key)
    if role is None:
        return False

    if required_role == Role.ADMIN:
        return role == Role.ADMIN
    if required_role == Role.USER:
        return role in (Role.USER, Role.ADMIN)

    return False


def require_api_key(api_key: str | None = None) -> bool:
    """验证 API Key。

    委托到全局 ApiKeyStore.require_api_key()。
    支持多密钥。
    """
    return _get_default_store().require_api_key(api_key)


def require_admin(api_key: str | None = None) -> bool:
    """验证管理员权限（基于 RBAC）。

    委托到全局 ApiKeyStore.require_admin()。
    """
    return _get_default_store().require_admin(api_key)


def generate_secret_key() -> str:
    """生成随机密钥。"""
    return secrets.token_hex(32)


def get_secret_key() -> str:
    """获取 SECRET_KEY，优先从环境变量读取，否则生成随机密钥并警告。"""
    key = os.environ.get("ASC_SECRET_KEY")
    if key:
        return key

    key = generate_secret_key()
    logger.warning(
        "ASC_SECRET_KEY 未设置，已生成随机密钥。"
        "生产环境请设置 ASC_SECRET_KEY 环境变量。"
    )
    return key


def validate_model_id(model_id: str) -> bool:
    """验证 model_id 不含路径遍历字符。

    防止通过 model_id 拼接路径时出现路径遍历攻击。
    """
    if not model_id:
        return False
    # 禁止路径分隔符和遍历字符
    dangerous = ["..", "/", "\\", "\x00"]
    return all(char not in model_id for char in dangerous)
