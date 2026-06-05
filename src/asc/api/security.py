"""API 安全模块。

提供：
- 基于内存的 API 限流（滑动窗口）
- 输入长度限制
- 参数范围校验
- 内容安全过滤（可选）
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class RateLimitEntry:
    """限流条目。"""

    count: int = 0
    window_start: float = field(default_factory=time.time)


class RateLimiter:
    """基于内存的滑动窗口限流器。

    按 key（如 IP 地址或 API Key）统计请求频率。
    """

    def __init__(
        self,
        max_requests: int = 60,
        window_seconds: float = 60.0,
    ) -> None:
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._entries: dict[str, RateLimitEntry] = {}

    def is_allowed(self, key: str) -> bool:
        """检查 key 是否允许通过。"""
        now = time.time()
        entry = self._entries.get(key)

        if entry is None or now - entry.window_start > self._window_seconds:
            # 新窗口
            self._entries[key] = RateLimitEntry(count=1, window_start=now)
            return True

        if entry.count < self._max_requests:
            entry.count += 1
            return True

        return False

    def remaining(self, key: str) -> int:
        """返回 key 在当前窗口剩余的请求次数。"""
        now = time.time()
        entry = self._entries.get(key)
        if entry is None or now - entry.window_start > self._window_seconds:
            return self._max_requests
        return max(0, self._max_requests - entry.count)

    def reset(self, key: str) -> None:
        """重置 key 的限流计数。"""
        self._entries.pop(key, None)

    def cleanup(self) -> None:
        """清理过期的限流条目。"""
        now = time.time()
        expired = [
            k for k, v in self._entries.items()
            if now - v.window_start > self._window_seconds
        ]
        for k in expired:
            self._entries.pop(k, None)


class InputValidator:
    """输入验证器。"""

    MAX_PROMPT_LENGTH: int = 32000
    MAX_TOKENS_MIN: int = 1
    MAX_TOKENS_MAX: int = 8192
    TEMPERATURE_MIN: float = 0.0
    TEMPERATURE_MAX: float = 2.0
    TOP_P_MIN: float = 0.0
    TOP_P_MAX: float = 1.0

    @classmethod
    def validate_prompt(cls, prompt: str) -> tuple[bool, str]:
        """验证 prompt 长度。"""
        if not prompt:
            return False, "prompt 不能为空"
        if len(prompt) > cls.MAX_PROMPT_LENGTH:
            return False, f"prompt 长度超过限制 {cls.MAX_PROMPT_LENGTH}"
        return True, ""

    @classmethod
    def validate_max_tokens(cls, max_tokens: int) -> tuple[bool, str]:
        """验证 max_tokens 范围。"""
        if not cls.MAX_TOKENS_MIN <= max_tokens <= cls.MAX_TOKENS_MAX:
            return (
                False,
                f"max_tokens 必须在 {cls.MAX_TOKENS_MIN}-{cls.MAX_TOKENS_MAX} 之间",
            )
        return True, ""

    @classmethod
    def validate_temperature(cls, temperature: float) -> tuple[bool, str]:
        """验证 temperature 范围。"""
        if not cls.TEMPERATURE_MIN <= temperature <= cls.TEMPERATURE_MAX:
            return (
                False,
                f"temperature 必须在 {cls.TEMPERATURE_MIN}-{cls.TEMPERATURE_MAX} 之间",
            )
        return True, ""

    @classmethod
    def validate_top_p(cls, top_p: float) -> tuple[bool, str]:
        """验证 top_p 范围。"""
        if not cls.TOP_P_MIN <= top_p <= cls.TOP_P_MAX:
            return False, f"top_p 必须在 {cls.TOP_P_MIN}-{cls.TOP_P_MAX} 之间"
        return True, ""

    @classmethod
    def validate_request(
        cls,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 1.0,
    ) -> tuple[bool, str]:
        """完整验证请求参数。"""
        checks = [
            cls.validate_prompt(prompt),
            cls.validate_max_tokens(max_tokens),
            cls.validate_temperature(temperature),
            cls.validate_top_p(top_p),
        ]
        for ok, msg in checks:
            if not ok:
                return False, msg
        return True, ""


class ContentFilter:
    """简单内容安全过滤器。

    基于关键词列表进行基础过滤。
    生产环境应接入专业内容审核服务。
    """

    DEFAULT_BLOCKED_KEYWORDS: list[str] = []

    def __init__(self, blocked_keywords: list[str] | None = None) -> None:
        self._blocked = set(
            blocked_keywords or self.DEFAULT_BLOCKED_KEYWORDS
        )

    def is_safe(self, text: str) -> bool:
        """检查文本是否安全。"""
        text_lower = text.lower()
        return all(
            keyword.lower() not in text_lower for keyword in self._blocked
        )

    def check(self, text: str) -> tuple[bool, str]:
        """检查并返回原因。"""
        if self.is_safe(text):
            return True, ""
        return False, "内容包含敏感信息"
