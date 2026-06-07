"""Asc API 认证中间件。"""

from __future__ import annotations

import hmac
import os
import secrets
import warnings


def require_api_key(api_key: str | None = None) -> bool:
    """验证 API Key。

    安全策略：
    - 未配置 ASC_API_KEY 环境变量时拒绝所有请求（返回 False）
    - 未配置 Key 时可通过 ASC_ALLOW_NO_AUTH=1 开启内网免认证模式（不推荐）
    - 使用 hmac.compare_digest 防止时序攻击
    """
    expected = os.getenv("ASC_API_KEY")

    # 未配置 API Key 时的安全策略
    if not expected:
        if os.getenv("ASC_ALLOW_NO_AUTH") == "1":
            warnings.warn(
                "ASC_ALLOW_NO_AUTH=1 已启用免认证模式。"
                "这是不安全的配置，不应在生产环境使用。"
                "请设置 ASC_API_KEY 环境变量启用认证。",
                RuntimeWarning,
                stacklevel=2,
            )
            return True
        return False

    if api_key is None:
        return False

    # 使用 hmac.compare_digest 防止时序攻击
    return hmac.compare_digest(api_key, expected)


def require_admin(api_key: str | None = None) -> bool:
    """验证管理员权限。

    管理端点（/admin/*）需要额外的管理员 API Key 验证。
    安全策略：
    - 优先使用 ASC_ADMIN_API_KEY 环境变量
    - 未配置时回退到 ASC_API_KEY（兼容旧版）
    - 使用 hmac.compare_digest 防止时序攻击
    """
    admin_key = os.getenv("ASC_ADMIN_API_KEY")
    if admin_key:
        if api_key is None:
            return False
        return hmac.compare_digest(api_key, admin_key)

    # 未配置管理员 Key 时，回退到普通 API Key 验证
    # 但发出警告：管理端点应使用独立的管理员凭据
    if not admin_key:
        warnings.warn(
            "ASC_ADMIN_API_KEY 未设置，管理端点回退到 ASC_API_KEY 验证。"
            "生产环境建议设置独立的 ASC_ADMIN_API_KEY 环境变量以启用 RBAC。",
            RuntimeWarning,
            stacklevel=2,
        )
    return require_api_key(api_key)


def generate_secret_key() -> str:
    """生成随机密钥。"""
    return secrets.token_hex(32)


def get_secret_key() -> str:
    """获取 SECRET_KEY，优先从环境变量读取，否则生成随机密钥并警告。"""
    key = os.environ.get("ASC_SECRET_KEY")
    if key:
        return key

    key = generate_secret_key()
    warnings.warn(
        "ASC_SECRET_KEY 未设置，已生成随机密钥。"
        "生产环境请设置 ASC_SECRET_KEY 环境变量。",
        stacklevel=2,
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
