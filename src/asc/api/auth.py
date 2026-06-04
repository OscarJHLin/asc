"""Asc API 认证中间件。"""

from __future__ import annotations

import os


def require_api_key(api_key: str | None = None) -> bool:
    """验证 API Key。

    未配置 ASC_API_KEY 环境变量时跳过认证（内网模式）。
    """
    expected = os.getenv("ASC_API_KEY")
    if not expected:
        return True  # 内网模式，跳过认证
    if api_key is None:
        return False
    return api_key == expected
