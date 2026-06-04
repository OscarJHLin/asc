"""Asc FastAPI 服务器。

提供 OpenAI / Anthropic 兼容的 API 端点。
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from asc.api.auth import require_api_key
from asc.api.openai_adapter import (
    ChatCompletionRequest,
    OpenAIAdapter,
)


def create_app(
    model_mappings: dict[str, str] | None = None,
    engine: Any = None,
) -> FastAPI:
    """创建 FastAPI 应用。"""
    app = FastAPI(title="Asc API", version="0.1.0")
    adapter = OpenAIAdapter(model_mappings=model_mappings)
    app.state.adapter = adapter
    app.state.engine = engine

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models():
        models = adapter.list_models()
        return {
            "object": "list",
            "data": [m.to_dict() for m in models],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        # 验证 API Key
        if not require_api_key():
            raise HTTPException(status_code=401, detail="Invalid API key")

        if app.state.engine is None:
            raise HTTPException(status_code=503, detail="No inference engine available")

        prompt = adapter.format_prompt(request)

        if request.stream:
            return StreamingResponse(
                _stream_chat(adapter, request, prompt),
                media_type="text/event-stream",
            )

        # 非流式推理
        try:
            output = app.state.engine.submit(
                __import__("asc.engine.base", fromlist=["InferenceRequest"]).InferenceRequest(
                    prompt=prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                )
            )
            resp = adapter.create_response(request, output)
            return JSONResponse(resp.to_dict())
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/admin/nodes")
    async def admin_nodes():
        return {"nodes": []}

    @app.get("/admin/config")
    async def admin_config():
        return {"config": {}}

    return app


async def _stream_chat(adapter, request, prompt):
    """流式聊天生成器。"""
    # 占位：实际由引擎提供流式输出
    chunk = adapter.create_chunk(request, delta="Not implemented yet", finish_reason="stop")
    yield f"data: {__import__('json').dumps(chunk.to_dict())}\n\n"
    yield "data: [DONE]\n\n"
