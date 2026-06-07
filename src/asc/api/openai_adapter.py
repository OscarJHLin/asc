"""Asc OpenAI 兼容 API 适配器。

将 OpenAI Chat Completions 格式转换为 Asc 内部推理请求。
适配器只负责格式转换，不涉及业务逻辑。

支持：
- Chat Completions（流式 + 非流式）
- Models 列表
- Embeddings（三级回退）
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from asc.api.prompt_converter import convert_messages_to_prompt


@dataclass(frozen=True)
class ChatCompletionRequest:
    """OpenAI Chat Completion 请求。"""

    model: str
    messages: list[dict]
    max_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 1.0
    stream: bool = False


@dataclass(frozen=True)
class TokenUsage:
    """Token 使用量。"""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class ChatCompletionResponse:
    """OpenAI Chat Completion 响应。"""

    id: str
    model: str
    content: str
    usage: TokenUsage
    finish_reason: str = "stop"
    created: int = field(default_factory=lambda: int(time.time()))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "object": "chat.completion",
            "created": self.created,
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": self.content},
                    "finish_reason": self.finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens,
            },
        }


@dataclass
class ChatCompletionChunk:
    """OpenAI Chat Completion 流式 chunk。"""

    id: str
    model: str
    delta_content: str
    finish_reason: str | None = None
    created: int = field(default_factory=lambda: int(time.time()))
    is_first: bool = False

    def to_dict(self) -> dict:
        delta: dict = {}
        if self.is_first:
            delta["role"] = "assistant"
        if self.delta_content:
            delta["content"] = self.delta_content
        if self.finish_reason is None and not self.delta_content and not self.is_first:
            delta["role"] = "assistant"

        choice: dict = {
            "index": 0,
            "delta": delta,
        }
        if self.finish_reason is not None:
            choice["finish_reason"] = self.finish_reason

        return {
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [choice],
        }


@dataclass(frozen=True)
class ModelInfo:
    """模型信息。"""

    id: str
    owned_by: str = "asc"
    created: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "object": "model",
            "created": self.created,
            "owned_by": self.owned_by,
        }


def messages_to_prompt(messages: list[dict]) -> str:
    """将 OpenAI messages 数组转换为 llama.cpp prompt。"""
    return convert_messages_to_prompt(messages)


class OpenAIAdapter:
    """OpenAI 兼容 API 适配器。"""

    def __init__(self, model_mappings: dict[str, str] | None = None) -> None:
        self.model_mappings = model_mappings or {}

    def list_models(self) -> list[ModelInfo]:
        """列出可用模型。"""
        return [
            ModelInfo(id=model_id, created=int(time.time())) for model_id in self.model_mappings
        ]

    def format_prompt(self, request: ChatCompletionRequest) -> str:
        """将请求格式化为 prompt。"""
        return messages_to_prompt(request.messages)

    def create_response(
        self,
        request: ChatCompletionRequest,
        output: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> ChatCompletionResponse:
        """创建非流式响应。"""
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            model=request.model,
            content=output,
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    def create_chunk(
        self,
        request: ChatCompletionRequest,
        delta: str,
        finish_reason: str | None = None,
        is_first: bool = False,
    ) -> ChatCompletionChunk:
        """创建流式 chunk。"""
        return ChatCompletionChunk(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            model=request.model,
            delta_content=delta,
            finish_reason=finish_reason,
            is_first=is_first,
        )
