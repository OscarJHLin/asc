"""Asc FastAPI 服务器。

提供 OpenAI 兼容的 REST API 端点，使 Asc 集群可被标准客户端（如 OpenAI Python SDK、
LangChain、LlamaIndex）直接调用。

支持的端点：
- GET  /health              : 健康检查
- GET  /v1/models           : 列出可用模型
- POST /v1/chat/completions : 聊天补全（支持流式 SSE 输出）
- GET  /admin/nodes         : 管理端点 - 查看集群节点状态
- GET  /admin/config        : 管理端点 - 查看集群配置摘要

安全机制：
- API Key 认证（X-API-Key 请求头）
- 请求限流（基于 IP 的滑动窗口，60 请求/分钟）
- 输入验证（prompt 长度、max_tokens、temperature 范围）

流式输出说明：
    当前实现为"伪流式"：先通过 engine.submit_async() 获取完整输出，
    再按 token 拆分发送 SSE chunk。这是因为 llama-server 的流式 API
    需要额外适配。未来版本将改为真正的逐 token 流式传输。

性能注意：
    submit_async() 使用 asyncio.to_thread() 避免阻塞事件循环，
    但流式输出的 token 拆分在主协程执行，对于超长输出（>10KB）
    可能需要优化拆分算法。
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware

from asc.api.auth import require_admin, require_api_key
from asc.api.openai_adapter import (
    ChatCompletionRequest,
    OpenAIAdapter,
)
from asc.api.security import InputValidator, RateLimiter
from asc.engine.base import InferenceRequest
from asc.types.state import ClusterState


class RateLimitMiddleware(BaseHTTPMiddleware):
    """基于客户端 IP 的限流中间件。"""

    def __init__(self, app: Any, limiter: RateLimiter) -> None:
        super().__init__(app)
        self._limiter = limiter

    async def dispatch(self, request: Request, call_next: Any) -> Any:
        client_key = request.client.host if request.client else "unknown"
        if not await self._limiter.is_allowed(client_key):
            return JSONResponse(
                status_code=429,
                content={"detail": "请求过于频繁，请稍后再试"},
            )
        response = await call_next(request)
        remaining = await self._limiter.remaining(client_key)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response


def create_app(
    model_mappings: dict[str, str] | None = None,
    engine: Any = None,
    rate_limiter: RateLimiter | None = None,
    cluster_state: ClusterState | None = None,
) -> FastAPI:
    """创建 FastAPI 应用。"""
    app = FastAPI(title="Asc API", version="0.1.0")
    adapter = OpenAIAdapter(model_mappings=model_mappings)
    app.state.adapter = adapter
    app.state.engine = engine
    app.state.cluster_state = cluster_state

    # 挂载限流中间件
    limiter = rate_limiter or RateLimiter(max_requests=60, window_seconds=60.0)
    app.add_middleware(RateLimitMiddleware, limiter=limiter)

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
    async def chat_completions(
        request: ChatCompletionRequest,
        x_api_key: str | None = Header(None, alias="X-API-Key"),
    ):
        # 验证 API Key — 从请求头提取 Key
        if not require_api_key(x_api_key):
            raise HTTPException(status_code=401, detail="Invalid API key")

        # 输入验证 — 提取用户原始内容进行校验
        user_contents = [
            m.get("content", "") for m in request.messages if m.get("role") == "user"
        ]
        raw_prompt = "\n".join(user_contents)
        ok, msg = InputValidator.validate_request(
            prompt=raw_prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )
        if not ok:
            raise HTTPException(status_code=422, detail=msg)

        # model_id 格式验证
        if not request.model or not request.model.strip():
            raise HTTPException(status_code=422, detail="model 不能为空")

        prompt = adapter.format_prompt(request)

        if app.state.engine is None:
            raise HTTPException(status_code=503, detail="No inference engine available")

        if request.stream:
            return StreamingResponse(
                _stream_chat(adapter, request, prompt, app.state.engine),
                media_type="text/event-stream",
            )

        # 非流式推理
        try:
            output = await app.state.engine.submit_async(
                InferenceRequest(
                    prompt=prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                )
            )
            resp = adapter.create_response(request, output)
            return JSONResponse(resp.to_dict())
        except Exception as e:
            raise HTTPException(status_code=500, detail="Internal server error") from e

    @app.get("/admin/nodes")
    async def admin_nodes(
        x_api_key: str | None = Header(None, alias="X-API-Key"),
    ):
        if not require_admin(x_api_key):
            raise HTTPException(status_code=403, detail="Admin access required")
        if app.state.cluster_state is not None:
            nodes = [
                {
                    "node_id": str(nid),
                    "ip": info.ip,
                    "port": info.port,
                    "runner_status": info.runner_status,
                }
                for nid, info in app.state.cluster_state.nodes.items()
            ]
            return {"nodes": nodes}
        return {"nodes": []}

    @app.get("/admin/config")
    async def admin_config(
        x_api_key: str | None = Header(None, alias="X-API-Key"),
    ):
        if not require_admin(x_api_key):
            raise HTTPException(status_code=403, detail="Admin access required")
        if app.state.cluster_state is not None:
            config = {
                "nodes_count": len(app.state.cluster_state.nodes),
                "instances_count": len(app.state.cluster_state.instances),
                "tasks_count": len(app.state.cluster_state.tasks),
                "event_index": app.state.cluster_state.event_index,
            }
            return {"config": config}
        return {"config": {}}

    return app


async def _stream_chat(adapter, request, prompt, engine):
    """流式聊天生成器 — SSE 格式。

    使用引擎的流式推理能力，实时发送生成的 token。
    引擎异常会被捕获并转换为 SSE error 事件发送给客户端。
    """
    try:
        # 使用异步流式推理，实时获取 token
        async for token in engine.submit_async_stream(
            InferenceRequest(
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
        ):
            chunk = adapter.create_chunk(request, delta=token, finish_reason=None)
            yield f"data: {json.dumps(chunk.to_dict())}\n\n"

        # 发送结束 chunk
        final_chunk = adapter.create_chunk(request, delta="", finish_reason="stop")
        yield f"data: {json.dumps(final_chunk.to_dict())}\n\n"
    except Exception as e:
        # 引擎异常时发送错误事件给客户端
        error_chunk = {
            "error": {
                "message": f"流式推理失败: {e}",
                "type": "server_error",
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"

    # 发送 [DONE]
    yield "data: [DONE]\n\n"
