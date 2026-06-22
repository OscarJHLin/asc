"""Asc Anthropic 兼容 API 适配器。

将 Anthropic Messages API 格式转换为 Asc 内部推理请求。
关键差异：
- system 是顶层字段（不在 messages 中）
- content 是数组格式
- 流式事件序列严格遵循 Anthropic 规范
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from asc.api.prompt_converter import convert_messages_to_prompt


@dataclass(frozen=True)
class AnthropicRequest:
    """Anthropic Messages 请求。"""

    model: str
    messages: list[dict[str, Any]]
    max_tokens: int = 128
    temperature: float = 0.7
    system: str | None = None


@dataclass(frozen=True)
class ContentBlock:
    """内容块。"""

    type: str = "text"
    text: str = ""


@dataclass(frozen=True)
class ContentBlockDelta:
    """内容块增量。"""

    type: str = "text_delta"
    text: str = ""


@dataclass
class AnthropicResponse:
    """Anthropic Messages 响应。"""

    id: str
    model: str
    content: list[ContentBlock]
    stop_reason: str
    usage: dict
    type: str = "message"
    role: str = "assistant"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "role": self.role,
            "model": self.model,
            "content": [{"type": b.type, "text": b.text} for b in self.content],
            "stop_reason": self.stop_reason,
            "stop_sequence": None,
            "usage": self.usage,
        }


# --- 流式事件 ---


@dataclass
class MessageStartEvent:
    id: str
    model: str
    type: str = "message_start"

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "message": {
                "id": self.id,
                "type": "message",
                "role": "assistant",
                "model": self.model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        }


@dataclass
class ContentBlockStartEvent:
    index: int
    type: str = "content_block_start"

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "index": self.index,
            "content_block": {"type": "text", "text": ""},
        }


@dataclass
class ContentBlockDeltaEvent:
    index: int
    delta: ContentBlockDelta
    type: str = "content_block_delta"

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "index": self.index,
            "delta": {"type": self.delta.type, "text": self.delta.text},
        }


@dataclass
class ContentBlockStopEvent:
    index: int
    type: str = "content_block_stop"

    def to_dict(self) -> dict:
        return {"type": self.type, "index": self.index}


@dataclass
class MessageDeltaEvent:
    stop_reason: str
    output_tokens: int
    type: str = "message_delta"

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "delta": {"stop_reason": self.stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": self.output_tokens},
        }


@dataclass
class MessageStopEvent:
    type: str = "message_stop"

    def to_dict(self) -> dict:
        return {"type": self.type}


# 别名：供 AnthropicAdapter.format_prompt 和外部测试使用
anthropic_messages_to_prompt = convert_messages_to_prompt


class AnthropicAdapter:
    """Anthropic 兼容 API 适配器。"""

    def format_prompt(self, request: AnthropicRequest) -> str:
        return anthropic_messages_to_prompt(request.messages, request.system)

    def create_response(self, request: AnthropicRequest, output: str) -> AnthropicResponse:
        return AnthropicResponse(
            id=f"msg_{uuid.uuid4().hex[:12]}",
            model=request.model,
            content=[ContentBlock(type="text", text=output)],
            stop_reason="end_turn",
            usage={"input_tokens": 0, "output_tokens": len(output.split())},
        )

    def create_stream_events(
        self,
        request: AnthropicRequest,
        tokens: list[str],
    ) -> list:
        """创建完整的流式事件序列。"""
        msg_id = f"msg_{uuid.uuid4().hex[:12]}"

        events = [
            MessageStartEvent(id=msg_id, model=request.model),
            ContentBlockStartEvent(index=0),
        ]

        for _i, token in enumerate(tokens):
            events.append(
                ContentBlockDeltaEvent(
                    index=0,
                    delta=ContentBlockDelta(type="text_delta", text=token),
                )
            )

        events.extend(
            [
                ContentBlockStopEvent(index=0),
                MessageDeltaEvent(stop_reason="end_turn", output_tokens=len(tokens)),
                MessageStopEvent(),
            ]
        )

        return events
