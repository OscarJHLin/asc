"""消息格式转换器。

将不同 API 的消息格式统一转换为 llama.cpp prompt。
提取自 openai_adapter / anthropic_adapter / ollama_adapter 的重复逻辑。
"""

from __future__ import annotations

from typing import Any


def convert_messages_to_prompt(
    messages: list[dict[str, Any]],
    system: str | None = None,
    format: str = "chat",
) -> str:
    """将消息数组转换为 llama.cpp prompt。

    使用简单的角色标签格式：
    [System]: ...
    [User]: ...
    [Assistant]: ...

    Args:
        messages: 消息列表，每条消息包含 role 和 content。
        system: 顶层 system 指令（Anthropic 风格），优先于 messages 中的 system。
        format: 输出格式，当前仅支持 "chat"。

    Returns:
        转换后的 prompt 字符串。
    """
    parts: list[str] = []

    if system:
        parts.append(f"[System]: {system}")

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                item.get("text", "") for item in content if item.get("type") == "text"
            )
        tag = {
            "system": "System",
            "user": "User",
            "assistant": "Assistant",
        }.get(role, "User")
        parts.append(f"[{tag}]: {content}")

    parts.append("[Assistant]:")
    return "\n".join(parts)
