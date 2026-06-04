"""Asc Ollama 兼容 API 适配器。

将 Ollama API 格式转换为 Asc 内部推理请求。
适配器只负责格式转换，不涉及业务逻辑。

支持：
- /api/generate（非流式 + 流式）
- /api/chat（非流式 + 流式）
- /api/tags（模型列表）
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class OllamaGenerateRequest:
    """Ollama Generate 请求。"""

    model: str
    prompt: str
    stream: bool = False
    options: dict = field(default_factory=dict)


@dataclass
class OllamaGenerateResponse:
    """Ollama Generate 响应。"""

    model: str
    response: str
    done: bool = True
    prompt_eval_count: int = 0
    eval_count: int = 0
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S.000000Z"))

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "created_at": self.created_at,
            "response": self.response,
            "done": self.done,
            "prompt_eval_count": self.prompt_eval_count,
            "eval_count": self.eval_count,
        }


@dataclass(frozen=True)
class OllamaChatRequest:
    """Ollama Chat 请求。"""

    model: str
    messages: list[dict]
    stream: bool = False
    options: dict = field(default_factory=dict)


@dataclass
class OllamaChatResponse:
    """Ollama Chat 响应。"""

    model: str
    message: dict
    done: bool = True
    prompt_eval_count: int = 0
    eval_count: int = 0
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S.000000Z"))

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "created_at": self.created_at,
            "message": self.message,
            "done": self.done,
            "prompt_eval_count": self.prompt_eval_count,
            "eval_count": self.eval_count,
        }


@dataclass(frozen=True)
class OllamaModelInfo:
    """Ollama 模型信息。"""

    name: str
    size: int = 0
    modified_at: str = ""
    digest: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "size": self.size,
            "modified_at": self.modified_at,
            "digest": self.digest,
        }


def _messages_to_prompt(messages: list[dict]) -> str:
    """将 Ollama chat messages 转换为 llama.cpp prompt。"""
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        tag = {
            "system": "System",
            "user": "User",
            "assistant": "Assistant",
        }.get(role, "User")
        parts.append(f"[{tag}]: {content}")
    parts.append("[Assistant]:")
    return "\n".join(parts)


class OllamaAdapter:
    """Ollama 兼容 API 适配器。"""

    def __init__(self, model_mappings: dict[str, str] | None = None) -> None:
        self.model_mappings = model_mappings or {}

    def list_models(self) -> list[OllamaModelInfo]:
        """列出可用模型（/api/tags 格式）。"""
        return [OllamaModelInfo(name=model_id) for model_id in self.model_mappings]

    def format_prompt_from_generate(self, request: OllamaGenerateRequest) -> str:
        """将 Generate 请求格式化为 prompt。"""
        return request.prompt

    def format_prompt_from_chat(self, request: OllamaChatRequest) -> str:
        """将 Chat 请求格式化为 prompt。"""
        return _messages_to_prompt(request.messages)

    def create_generate_response(
        self,
        request: OllamaGenerateRequest,
        output: str,
        prompt_eval_count: int = 0,
        eval_count: int = 0,
    ) -> OllamaGenerateResponse:
        """创建 Generate 响应。"""
        return OllamaGenerateResponse(
            model=request.model,
            response=output,
            done=True,
            prompt_eval_count=prompt_eval_count,
            eval_count=eval_count,
        )

    def create_chat_response(
        self,
        request: OllamaChatRequest,
        output: str,
        prompt_eval_count: int = 0,
        eval_count: int = 0,
    ) -> OllamaChatResponse:
        """创建 Chat 响应。"""
        return OllamaChatResponse(
            model=request.model,
            message={"role": "assistant", "content": output},
            done=True,
            prompt_eval_count=prompt_eval_count,
            eval_count=eval_count,
        )
