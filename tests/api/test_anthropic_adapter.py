"""测试 Anthropic 兼容 API 适配器。

Anthropic Messages API 格式与 OpenAI 不同：
- system 是顶层字段
- content 是数组格式
- 流式事件序列：message_start -> content_block_start -> content_block_delta -> content_block_stop -> message_delta -> message_stop
"""

from asc.api.anthropic_adapter import (
    AnthropicAdapter,
    AnthropicRequest,
    AnthropicResponse,
    ContentBlock,
    ContentBlockDelta,
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    MessageDeltaEvent,
    MessageStartEvent,
    MessageStopEvent,
    anthropic_messages_to_prompt,
)


class TestAnthropicMessagesToPrompt:
    """Anthropic messages 转 prompt。"""

    def test_single_user_message(self):
        messages = [{"role": "user", "content": "Hello"}]
        prompt = anthropic_messages_to_prompt(messages, system="You are helpful.")
        assert "Hello" in prompt
        assert "You are helpful." in prompt

    def test_conversation(self):
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"},
        ]
        prompt = anthropic_messages_to_prompt(messages)
        assert "Hi" in prompt
        assert "Hello!" in prompt
        assert "How are you?" in prompt

    def test_content_array(self):
        """Anthropic content 数组格式。"""
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
        ]
        prompt = anthropic_messages_to_prompt(messages)
        assert "Hello" in prompt


class TestAnthropicRequest:
    """请求值对象。"""

    def test_create(self):
        req = AnthropicRequest(
            model="claude-3",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=128,
            system="You are helpful.",
        )
        assert req.model == "claude-3"
        assert req.system == "You are helpful."
        assert req.max_tokens == 128

    def test_defaults(self):
        req = AnthropicRequest(
            model="claude-3",
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert req.system is None
        assert req.max_tokens == 128
        assert req.temperature == 0.7


class TestAnthropicResponse:
    """响应值对象。"""

    def test_create(self):
        resp = AnthropicResponse(
            id="msg_123",
            model="claude-3",
            content=[ContentBlock(type="text", text="Hello!")],
            stop_reason="end_turn",
            usage={"input_tokens": 5, "output_tokens": 10},
        )
        assert resp.content[0].text == "Hello!"
        assert resp.stop_reason == "end_turn"

    def test_to_dict(self):
        resp = AnthropicResponse(
            id="msg_123",
            model="claude-3",
            content=[ContentBlock(type="text", text="Hi")],
            stop_reason="end_turn",
            usage={"input_tokens": 2, "output_tokens": 3},
        )
        d = resp.to_dict()
        assert d["role"] == "assistant"
        assert d["content"][0]["text"] == "Hi"
        assert d["stop_reason"] == "end_turn"


class TestStreamingEvents:
    """流式事件。"""

    def test_message_start(self):
        event = MessageStartEvent(id="msg_123", model="claude-3")
        d = event.to_dict()
        assert d["type"] == "message_start"
        assert d["message"]["id"] == "msg_123"

    def test_content_block_start(self):
        event = ContentBlockStartEvent(index=0)
        d = event.to_dict()
        assert d["type"] == "content_block_start"
        assert d["content_block"]["type"] == "text"

    def test_content_block_delta(self):
        event = ContentBlockDeltaEvent(
            index=0, delta=ContentBlockDelta(type="text_delta", text="Hi")
        )
        d = event.to_dict()
        assert d["type"] == "content_block_delta"
        assert d["delta"]["text"] == "Hi"

    def test_content_block_stop(self):
        event = ContentBlockStopEvent(index=0)
        d = event.to_dict()
        assert d["type"] == "content_block_stop"

    def test_message_delta(self):
        event = MessageDeltaEvent(stop_reason="end_turn", output_tokens=10)
        d = event.to_dict()
        assert d["type"] == "message_delta"
        assert d["delta"]["stop_reason"] == "end_turn"

    def test_message_stop(self):
        event = MessageStopEvent()
        d = event.to_dict()
        assert d["type"] == "message_stop"


class TestAnthropicAdapter:
    """适配器主类。"""

    def test_format_prompt(self):
        adapter = AnthropicAdapter()
        req = AnthropicRequest(
            model="claude-3",
            messages=[{"role": "user", "content": "Hello"}],
            system="Be helpful.",
        )
        prompt = adapter.format_prompt(req)
        assert "Hello" in prompt
        assert "Be helpful." in prompt

    def test_create_response(self):
        adapter = AnthropicAdapter()
        req = AnthropicRequest(
            model="claude-3",
            messages=[{"role": "user", "content": "Hi"}],
        )
        resp = adapter.create_response(req, output="Hello!")
        assert resp.content[0].text == "Hello!"
        assert resp.model == "claude-3"

    def test_create_streaming_events(self):
        adapter = AnthropicAdapter()
        req = AnthropicRequest(
            model="claude-3",
            messages=[{"role": "user", "content": "Hi"}],
        )
        events = list(adapter.create_stream_events(req, ["Hello", " there", "!"]))
        # message_start + content_block_start + 3 deltas + content_block_stop + message_delta + message_stop
        assert len(events) == 8
        assert isinstance(events[0], MessageStartEvent)
        assert isinstance(events[1], ContentBlockStartEvent)
        assert isinstance(events[2], ContentBlockDeltaEvent)
        assert isinstance(events[5], ContentBlockStopEvent)
        assert isinstance(events[6], MessageDeltaEvent)
        assert isinstance(events[7], MessageStopEvent)
