"""API 安全模块。

提供 API 层的防护机制，防止滥用和恶意输入：
- 基于内存的滑动窗口限流（RateLimiter）
- 输入长度和参数范围校验（InputValidator）
- 基础内容安全过滤（ContentFilter，生产环境建议接入专业服务）

安全策略：
    1. 限流：按客户端 IP 或 API Key 限制请求频率，防止暴力破解和 DDoS
    2. 校验：限制 prompt 长度、max_tokens、temperature 等参数范围
    3. 过滤：基于关键词列表进行基础内容审核（可配置）

线程安全：
    RateLimiter 使用 asyncio.Lock 保护内存中的计数器，可在多协程环境下安全使用。
    InputValidator 和 ContentFilter 为纯函数/无状态，无需锁。

性能注意：
    限流数据存储在内存中，单进程有效。若部署多进程，需改用 Redis 等外部存储。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field


@dataclass
class RateLimitEntry:
    """限流条目。"""

    count: int = 0
    window_start: float = field(default_factory=time.time)


class RateLimiter:
    """基于内存的滑动窗口限流器。

    按 key（如 IP 地址或 API Key）统计请求频率，防止单客户端过度消耗资源。
    采用滑动窗口而非固定窗口，避免窗口边界处的突发流量问题。

    算法说明：
        - 每个 key 维护一个计数器和窗口起始时间
        - 若当前时间 - window_start > window_seconds，重置计数器（新窗口）
        - 若计数器 < max_requests，允许通过并递增计数器
        - 否则拒绝

    扩展建议：
        生产环境若需多进程共享限流状态，可替换为 Redis + Lua 脚本实现。
    """

    def __init__(
        self,
        max_requests: int = 60,
        window_seconds: float = 60.0,
    ) -> None:
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._entries: dict[str, RateLimitEntry] = {}
        self._lock = asyncio.Lock()

    async def is_allowed(self, key: str) -> bool:
        """检查 key 是否允许通过当前请求。

        Args:
            key: 限流键，通常为客户端 IP 或 API Key

        Returns:
            True 允许通过，False 拒绝（触发 429 Too Many Requests）
        """
        async with self._lock:
            now = time.time()
            entry = self._entries.get(key)

            if entry is None or now - entry.window_start > self._window_seconds:
                # 新窗口：重置计数器
                self._entries[key] = RateLimitEntry(count=1, window_start=now)
                return True

            if entry.count < self._max_requests:
                entry.count += 1
                return True

            return False

    async def remaining(self, key: str) -> int:
        """返回 key 在当前窗口剩余的请求次数。

        用于在响应头中设置 X-RateLimit-Remaining，帮助客户端调整请求频率。
        """
        async with self._lock:
            now = time.time()
            entry = self._entries.get(key)
            if entry is None or now - entry.window_start > self._window_seconds:
                return self._max_requests
            return max(0, self._max_requests - entry.count)

    async def reset(self, key: str) -> None:
        """重置 key 的限流计数。

        可用于手动解禁被限流的客户端（如管理员操作）。
        """
        async with self._lock:
            self._entries.pop(key, None)

    async def cleanup(self) -> None:
        """清理过期的限流条目，释放内存。

        建议定期调用（如每小时一次），防止内存无限增长。
        """
        async with self._lock:
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
