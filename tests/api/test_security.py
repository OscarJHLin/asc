"""测试 asc.api.security 模块。"""

import time

from asc.api.security import ContentFilter, InputValidator, RateLimiter


class TestRateLimiter:
    """RateLimiter 测试。"""

    async def test_is_allowed_up_to_limit(self):
        """允许请求直到达到上限。"""
        limiter = RateLimiter(max_requests=3, window_seconds=60.0)
        key = "test-key"

        assert await limiter.is_allowed(key) is True
        assert await limiter.is_allowed(key) is True
        assert await limiter.is_allowed(key) is True

    async def test_is_allowed_blocks_beyond_limit(self):
        """超过上限后拒绝请求。"""
        limiter = RateLimiter(max_requests=2, window_seconds=60.0)
        key = "test-key"

        await limiter.is_allowed(key)
        await limiter.is_allowed(key)

        assert await limiter.is_allowed(key) is False

    async def test_remaining_decreases(self):
        """remaining 随请求减少。"""
        limiter = RateLimiter(max_requests=5, window_seconds=60.0)
        key = "test-key"

        assert await limiter.remaining(key) == 5
        await limiter.is_allowed(key)
        assert await limiter.remaining(key) == 4
        await limiter.is_allowed(key)
        assert await limiter.remaining(key) == 3

    async def test_remaining_zero_at_limit(self):
        """达到上限后 remaining 为 0。"""
        limiter = RateLimiter(max_requests=2, window_seconds=60.0)
        key = "test-key"

        await limiter.is_allowed(key)
        await limiter.is_allowed(key)

        assert await limiter.remaining(key) == 0

    async def test_reset_clears_entry(self):
        """reset 后重新计数。"""
        limiter = RateLimiter(max_requests=2, window_seconds=60.0)
        key = "test-key"

        await limiter.is_allowed(key)
        await limiter.is_allowed(key)
        assert await limiter.is_allowed(key) is False

        await limiter.reset(key)
        assert await limiter.remaining(key) == 2
        assert await limiter.is_allowed(key) is True

    async def test_new_window_after_expiry(self):
        """窗口过期后允许新请求。"""
        limiter = RateLimiter(max_requests=1, window_seconds=0.1)
        key = "test-key"

        assert await limiter.is_allowed(key) is True
        assert await limiter.is_allowed(key) is False

        time.sleep(0.15)
        assert await limiter.is_allowed(key) is True

    async def test_remaining_returns_max_for_new_window(self):
        """新窗口返回完整剩余次数。"""
        limiter = RateLimiter(max_requests=3, window_seconds=0.1)
        key = "test-key"

        await limiter.is_allowed(key)
        await limiter.is_allowed(key)
        assert await limiter.remaining(key) == 1

        time.sleep(0.15)
        assert await limiter.remaining(key) == 3

    async def test_cleanup_removes_expired(self):
        """cleanup 清理过期条目。"""
        limiter = RateLimiter(max_requests=3, window_seconds=0.1)
        key = "expired-key"

        await limiter.is_allowed(key)
        assert key in limiter._entries

        time.sleep(0.15)
        await limiter.cleanup()
        assert key not in limiter._entries

    async def test_cleanup_keeps_active(self):
        """cleanup 保留未过期条目。"""
        limiter = RateLimiter(max_requests=3, window_seconds=60.0)
        key = "active-key"

        await limiter.is_allowed(key)
        await limiter.cleanup()
        assert key in limiter._entries

    async def test_different_keys_independent(self):
        """不同 key 互不影响。"""
        limiter = RateLimiter(max_requests=2, window_seconds=60.0)

        assert await limiter.is_allowed("key-a") is True
        assert await limiter.is_allowed("key-a") is True
        assert await limiter.is_allowed("key-b") is True
        assert await limiter.is_allowed("key-a") is False
        assert await limiter.is_allowed("key-b") is True


class TestInputValidator:
    """InputValidator 测试。"""

    def test_validate_prompt_empty(self):
        """空 prompt 被拒绝。"""
        ok, msg = InputValidator.validate_prompt("")
        assert ok is False
        assert "不能为空" in msg

    def test_validate_prompt_too_long(self):
        """超长 prompt 被拒绝。"""
        long_prompt = "x" * (InputValidator.MAX_PROMPT_LENGTH + 1)
        ok, msg = InputValidator.validate_prompt(long_prompt)
        assert ok is False
        assert "超过限制" in msg

    def test_validate_prompt_ok(self):
        """正常 prompt 通过。"""
        ok, msg = InputValidator.validate_prompt("hello")
        assert ok is True
        assert msg == ""

    def test_validate_prompt_boundary(self):
        """恰好达到最大长度的 prompt 通过。"""
        boundary_prompt = "x" * InputValidator.MAX_PROMPT_LENGTH
        ok, msg = InputValidator.validate_prompt(boundary_prompt)
        assert ok is True
        assert msg == ""

    def test_validate_max_tokens_too_low(self):
        """max_tokens 过小被拒绝。"""
        ok, msg = InputValidator.validate_max_tokens(0)
        assert ok is False
        assert "max_tokens" in msg

    def test_validate_max_tokens_too_high(self):
        """max_tokens 过大被拒绝。"""
        ok, msg = InputValidator.validate_max_tokens(
            InputValidator.MAX_TOKENS_MAX + 1
        )
        assert ok is False
        assert "max_tokens" in msg

    def test_validate_max_tokens_ok(self):
        """正常 max_tokens 通过。"""
        ok, msg = InputValidator.validate_max_tokens(128)
        assert ok is True
        assert msg == ""

    def test_validate_max_tokens_boundary(self):
        """边界 max_tokens 通过。"""
        ok, msg = InputValidator.validate_max_tokens(
            InputValidator.MAX_TOKENS_MIN
        )
        assert ok is True
        ok, msg = InputValidator.validate_max_tokens(
            InputValidator.MAX_TOKENS_MAX
        )
        assert ok is True

    def test_validate_temperature_too_low(self):
        """temperature 过小被拒绝。"""
        ok, msg = InputValidator.validate_temperature(-0.1)
        assert ok is False
        assert "temperature" in msg

    def test_validate_temperature_too_high(self):
        """temperature 过大被拒绝。"""
        ok, msg = InputValidator.validate_temperature(
            InputValidator.TEMPERATURE_MAX + 0.1
        )
        assert ok is False
        assert "temperature" in msg

    def test_validate_temperature_ok(self):
        """正常 temperature 通过。"""
        ok, msg = InputValidator.validate_temperature(0.7)
        assert ok is True
        assert msg == ""

    def test_validate_temperature_boundary(self):
        """边界 temperature 通过。"""
        ok, msg = InputValidator.validate_temperature(
            InputValidator.TEMPERATURE_MIN
        )
        assert ok is True
        ok, msg = InputValidator.validate_temperature(
            InputValidator.TEMPERATURE_MAX
        )
        assert ok is True

    def test_validate_top_p_too_low(self):
        """top_p 过小被拒绝。"""
        ok, msg = InputValidator.validate_top_p(-0.1)
        assert ok is False
        assert "top_p" in msg

    def test_validate_top_p_too_high(self):
        """top_p 过大被拒绝。"""
        ok, msg = InputValidator.validate_top_p(
            InputValidator.TOP_P_MAX + 0.1
        )
        assert ok is False
        assert "top_p" in msg

    def test_validate_top_p_ok(self):
        """正常 top_p 通过。"""
        ok, msg = InputValidator.validate_top_p(0.9)
        assert ok is True
        assert msg == ""

    def test_validate_top_p_boundary(self):
        """边界 top_p 通过。"""
        ok, msg = InputValidator.validate_top_p(InputValidator.TOP_P_MIN)
        assert ok is True
        ok, msg = InputValidator.validate_top_p(InputValidator.TOP_P_MAX)
        assert ok is True

    def test_validate_request_all_valid(self):
        """全部参数有效时通过。"""
        ok, msg = InputValidator.validate_request(
            prompt="hello", max_tokens=128, temperature=0.7, top_p=1.0
        )
        assert ok is True
        assert msg == ""

    def test_validate_request_invalid_prompt(self):
        """prompt 无效时返回错误。"""
        ok, msg = InputValidator.validate_request(prompt="")
        assert ok is False
        assert "prompt" in msg

    def test_validate_request_invalid_max_tokens(self):
        """max_tokens 无效时返回错误。"""
        ok, msg = InputValidator.validate_request(
            prompt="hello", max_tokens=0
        )
        assert ok is False
        assert "max_tokens" in msg

    def test_validate_request_invalid_temperature(self):
        """temperature 无效时返回错误。"""
        ok, msg = InputValidator.validate_request(
            prompt="hello", temperature=3.0
        )
        assert ok is False
        assert "temperature" in msg

    def test_validate_request_invalid_top_p(self):
        """top_p 无效时返回错误。"""
        ok, msg = InputValidator.validate_request(
            prompt="hello", top_p=1.5
        )
        assert ok is False
        assert "top_p" in msg

    def test_validate_request_uses_defaults(self):
        """使用默认参数通过验证。"""
        ok, msg = InputValidator.validate_request(prompt="hello")
        assert ok is True
        assert msg == ""


class TestContentFilter:
    """ContentFilter 测试。"""

    def test_is_safe_with_blocked_keyword(self):
        """包含被阻关键词的文本不安全。"""
        cf = ContentFilter(blocked_keywords=["badword"])
        assert cf.is_safe("this contains badword") is False

    def test_is_safe_without_blocked_keyword(self):
        """不包含被阻关键词的文本安全。"""
        cf = ContentFilter(blocked_keywords=["badword"])
        assert cf.is_safe("this is safe text") is True

    def test_is_safe_case_insensitive(self):
        """大小写不敏感匹配。"""
        cf = ContentFilter(blocked_keywords=["BadWord"])
        assert cf.is_safe("this contains BADWORD") is False

    def test_is_safe_empty_blocked_list(self):
        """空关键词列表允许所有文本。"""
        cf = ContentFilter(blocked_keywords=[])
        assert cf.is_safe("anything") is True

    def test_is_safe_default_blocked_list(self):
        """默认空关键词列表允许所有文本。"""
        cf = ContentFilter()
        assert cf.is_safe("anything") is True

    def test_is_safe_partial_match(self):
        """部分匹配也被拦截。"""
        cf = ContentFilter(blocked_keywords=["bad"])
        assert cf.is_safe("this is badword") is False

    def test_check_safe(self):
        """安全文本返回 True 和空原因。"""
        cf = ContentFilter(blocked_keywords=["bad"])
        ok, msg = cf.check("safe text")
        assert ok is True
        assert msg == ""

    def test_check_unsafe(self):
        """不安全文本返回 False 和原因。"""
        cf = ContentFilter(blocked_keywords=["bad"])
        ok, msg = cf.check("this is bad")
        assert ok is False
        assert "敏感" in msg

    def test_multiple_keywords(self):
        """多个关键词中匹配任一即拦截。"""
        cf = ContentFilter(blocked_keywords=["bad", "worse"])
        assert cf.is_safe("contains bad") is False
        assert cf.is_safe("contains worse") is False
        assert cf.is_safe("clean text") is True
