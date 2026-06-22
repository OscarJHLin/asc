"""Asc FastAPI 服务器。

提供 OpenAI 兼容的 REST API 端点，使 Asc 集群可被标准客户端（如 OpenAI Python SDK、
LangChain、LlamaIndex）直接调用。

支持的端点：
- GET  /health              : 健康检查
- GET  /v1/models           : 列出可用模型
- POST /v1/chat/completions : 聊天补全（支持流式 SSE 输出）
- GET  /metrics             : Prometheus 指标导出
- GET  /admin/nodes         : 管理端点 - 查看集群节点状态
- GET  /admin/config        : 管理端点 - 查看集群配置摘要

安全机制：
- API Key 认证（X-API-Key 请求头）
- 请求限流（基于 IP 的滑动窗口，60 请求/分钟）
- 输入验证（prompt 长度、max_tokens、temperature 范围）

流式输出说明：
    通过 engine.submit_async_stream() 实现真正的逐 token 流式传输。
    引擎底层使用 httpx 流式 HTTP 连接消费 llama-server 的 SSE 响应，
    每个 token 生成后立即通过 SSE chunk 推送给客户端，无需等待完整输出。
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from asc.api.auth import require_admin, require_api_key, validate_model_id
from asc.api.openai_adapter import (
    ChatCompletionRequest,
    OpenAIAdapter,
)
from asc.api.security import InputValidator, RateLimiter
from asc.core.cluster_config import ClusterConfig
from asc.core.monitoring import MetricsCollector
from asc.engine.base import InferenceRequest
from asc.network.protocol import Channel, Envelope, Message, MessageType
from asc.types.state import ClusterState

logger = logging.getLogger(__name__)

# 文件上传允许的扩展名白名单
UPLOAD_ALLOWED_EXTENSIONS = {".gguf", ".bin", ".safetensors"}
# 文件上传最大大小 (50GB)
UPLOAD_MAX_SIZE_BYTES = 50 * 1024 * 1024 * 1024


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
    metrics_collector: MetricsCollector | None = None,
    cluster_config: ClusterConfig | None = None,
    config_path: str | None = None,
    tcp_server: Any = None,
    conn_node_map: dict[str, str] | None = None,
    model_distributor: Any = None,
) -> FastAPI:
    """创建 FastAPI 应用。"""
    app = FastAPI(title="Asc API", version="0.1.0")
    adapter = OpenAIAdapter(model_mappings=model_mappings)
    app.state.adapter = adapter
    app.state.engine = engine
    app.state.cluster_state = cluster_state
    app.state.metrics_collector = metrics_collector or MetricsCollector()
    app.state.cluster_config = cluster_config or ClusterConfig()
    app.state.tcp_server = tcp_server
    app.state.conn_node_map = conn_node_map or {}
    app.state.model_distributor = model_distributor
    app.state.node_resources = {}

    # 配置文件路径（用于持久化）
    if config_path:
        app.state.config_path = Path(config_path)
    else:
        app.state.config_path = Path.home() / ".asc" / "cluster_config.json"

    # 启动时从配置文件加载（覆盖传入的空默认值）
    if not cluster_config:
        loaded = ClusterConfig.load(app.state.config_path)
        app.state.cluster_config = loaded
        logger.info("从配置文件加载集群配置: %s", app.state.config_path)

    def _save_config() -> None:
        """持久化集群配置到文件。"""
        try:
            app.state.cluster_config.save(app.state.config_path)
        except Exception as e:
            logger.warning("保存配置文件失败: %s", e)

    # WebSocket 连接管理
    app.state.ws_clients: set[WebSocket] = set()

    # 服务器设置
    app.state.server_settings = {
        "api_port": 8080,
        "require_auth": True,
        "enable_cors": True,
        "serve_on_lan": True,
        "jit_loading": False,
        "auto_unload": False,
        "max_idle_ttl_minutes": 60,
        "only_keep_last_model": False,
    }

    # 挂载 CORS 中间件
    from starlette.middleware.cors import CORSMiddleware
    import os
    cors_origins_str = os.getenv("ASC_CORS_ORIGINS", "")
    if cors_origins_str:
        cors_origins = [o.strip() for o in cors_origins_str.split(",") if o.strip()]
    else:
        # 默认仅允许 localhost 访问；生产环境通过 ASC_CORS_ORIGINS 显式配置
        cors_origins = []
    cors_allow_credentials = len(cors_origins) > 0 and cors_origins != ["*"]
    if cors_origins == ["*"]:
        cors_allow_credentials = False  # 通配符不允许 credentials
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=cors_allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 挂载限流中间件
    limiter = rate_limiter or RateLimiter(max_requests=60, window_seconds=60.0)
    app.add_middleware(RateLimitMiddleware, limiter=limiter)

    # 挂载静态文件（Web UI）
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/")
    async def dashboard():
        """Web UI Dashboard 首页。"""
        index_file = static_dir / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return {"message": "Asc API Server", "docs": "/docs"}

    @app.get("/health")
    async def health():
        """公开健康检查端点，无需认证。

        返回基本的集群健康状态，供负载均衡器或 Kubernetes 探针使用。
        详细的集群健康信息请使用 /admin/health（需认证）。
        """
        state = app.state.cluster_state
        if state is None:
            return {"status": "ok"}

        total_nodes = len(state.nodes)
        online_nodes = sum(
            1 for info in state.nodes.values()
            if info.runner_status == "online"
        )

        # 服务本身运行即为 ok；节点信息仅作附加参考
        status = "ok"
        if total_nodes > 0 and online_nodes == 0:
            status = "degraded"

        return {"status": status, "nodes": total_nodes, "online": online_nodes}

    @app.get("/admin/health")
    async def admin_health(
        x_api_key: str | None = Header(None, alias="X-API-Key"),
    ):
        """集群级健康检查，返回节点在线率、服务可用性等。"""
        if not require_admin(x_api_key):
            raise HTTPException(status_code=403, detail="Admin access required")

        state = app.state.cluster_state
        if state is None:
            return {"status": "degraded", "detail": "cluster state unavailable"}

        total_nodes = len(state.nodes)
        online_nodes = sum(
            1 for info in state.nodes.values()
            if info.runner_status == "online"
        )
        pending_tasks = sum(
            1 for t in state.tasks.values()
            if hasattr(t, "status") and t.status.value == "pending"
        )
        running_tasks = sum(
            1 for t in state.tasks.values()
            if hasattr(t, "status") and t.status.value == "running"
        )
        active_instances = len(state.instances)

        # 健康状态判定
        if total_nodes == 0:
            status = "unhealthy"
        elif online_nodes == 0:
            status = "unhealthy"
        elif online_nodes < total_nodes:
            status = "degraded"
        else:
            status = "healthy"

        return {
            "status": status,
            "nodes": {
                "total": total_nodes,
                "online": online_nodes,
                "offline": total_nodes - online_nodes,
            },
            "tasks": {
                "pending": pending_tasks,
                "running": running_tasks,
            },
            "instances": active_instances,
        }

    @app.get("/metrics")
    async def metrics(x_api_key: str | None = Header(None, alias="X-API-Key")):
        """Prometheus 指标导出端点。"""
        if not require_api_key(x_api_key):
            raise HTTPException(status_code=403, detail="API key required")
        from fastapi.responses import PlainTextResponse

        collector = app.state.metrics_collector
        return PlainTextResponse(
            collector.to_prometheus(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    @app.get("/v1/models")
    async def list_models(x_api_key: str | None = Header(None, alias="X-API-Key")):
        if not require_api_key(x_api_key):
            raise HTTPException(status_code=403, detail="API key required")
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

        # 优先使用本地引擎，否则通过 TCP 分派到 Worker
        if app.state.engine is not None:
            if request.stream:
                return StreamingResponse(
                    _stream_chat(adapter, request, prompt, app.state.engine),
                    media_type="text/event-stream",
                )

            # 非流式推理（本地引擎）
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

        # 通过 TCP 任务分派到 Worker 执行推理
        tcp_server = getattr(app.state, "tcp_server", None)
        cluster_state = app.state.cluster_state
        conn_node_map = getattr(app.state, "conn_node_map", {})

        if tcp_server is None or cluster_state is None or not cluster_state.nodes:
            raise HTTPException(status_code=503, detail="No inference engine available")

        # 选择推理节点：优先选择已加载目标模型的在线节点
        online_node = None
        node_resources = getattr(app.state, "node_resources", {})
        for nid, info in cluster_state.nodes.items():
            if info.runner_status != "online":
                continue
            # 检查该节点是否已加载目标模型
            res = node_resources.get(str(nid), {})
            loaded_models = res.get("loaded_model_ids", [])
            if request.model in loaded_models:
                online_node = (nid, info)
                break
        # 没有节点加载了目标模型，回退到第一个在线节点
        if online_node is None:
            for nid, info in cluster_state.nodes.items():
                if info.runner_status == "online":
                    online_node = (nid, info)
                    break

        if online_node is None:
            raise HTTPException(status_code=503, detail="No online worker nodes")

        node_id, node_info = online_node

        # 查找连接 ID
        target_conn_id = None
        for cid, mapped_nid in conn_node_map.items():
            if mapped_nid == str(node_id):
                target_conn_id = cid
                break

        if target_conn_id is None:
            raise HTTPException(
                status_code=503,
                detail=f"Worker {node_id} not connected",
            )

        # 发送任务分派到 Worker
        import asyncio
        import uuid

        task_id = str(uuid.uuid4())
        result_future: asyncio.Future = asyncio.get_running_loop().create_future()

        # 注册结果回调
        if not hasattr(app.state, "pending_results"):
            app.state.pending_results = {}
        app.state.pending_results[task_id] = result_future

        dispatch_envelope = Envelope(
            channel=Channel.TASK_DISPATCH,
            message=Message(
                type=MessageType.TASK_DISPATCH,
                sender_id="master-api",
                payload={
                    "task_id": task_id,
                    "model_id": request.model,
                    "prompt": prompt,
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                },
            ),
            target=str(node_id),
        )

        try:
            await tcp_server.send(target_conn_id, dispatch_envelope)
        except Exception as e:
            app.state.pending_results.pop(task_id, None)
            raise HTTPException(status_code=503, detail=f"Failed to dispatch task: {e}") from e

        # 等待推理结果（超时 120 秒）
        try:
            result_payload = await asyncio.wait_for(result_future, timeout=120.0)
        except asyncio.TimeoutError:
            app.state.pending_results.pop(task_id, None)
            raise HTTPException(status_code=504, detail="Inference timeout")
        finally:
            app.state.pending_results.pop(task_id, None)

        if result_payload.get("status") == "cancelled":
            raise HTTPException(status_code=499, detail="Task cancelled")

        output_text = result_payload.get("output_text", "")
        resp = adapter.create_response(request, output_text)
        return JSONResponse(resp.to_dict())

    @app.get("/admin/nodes")
    async def admin_nodes(
        x_api_key: str | None = Header(None, alias="X-API-Key"),
    ):
        from asc.api.auth import _default_store
        if not require_admin(x_api_key):
            raise HTTPException(status_code=403, detail="Admin access required")
        node_resources = getattr(app.state, "node_resources", {})
        if app.state.cluster_state is not None:
            nodes = []
            for nid, info in app.state.cluster_state.nodes.items():
                node = {
                    "node_id": str(nid),
                    "ip": info.ip,
                    "port": info.port,
                    "runner_status": info.runner_status,
                }
                res = node_resources.get(str(nid), {})
                if res:
                    node["cpu_count"] = res.get("cpu_count", 0)
                    node["cpu_brand"] = res.get("cpu_brand", "")
                    node["cpu_instruction_set_extensions"] = res.get("cpu_instruction_set_extensions", [])
                    node["memory_total_mb"] = res.get("memory_total_mb", 0)
                    node["memory_free_mb"] = res.get("memory_free_mb", 0)
                    node["gpus"] = res.get("gpus", [])
                    node["compute_score"] = res.get("compute_score", 0.0)
                    node["os_type"] = res.get("os_type", "")
                    node["os_arch"] = res.get("os_arch", "")
                nodes.append(node)
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

    # ---- 集群配置管理端点 ----

    @app.get("/admin/cluster-config")
    async def get_cluster_config(
        x_api_key: str | None = Header(None, alias="X-API-Key"),
    ):
        """获取完整集群配置。"""
        if not require_admin(x_api_key):
            raise HTTPException(status_code=403, detail="Admin access required")
        return app.state.cluster_config.to_dict()

    @app.put("/admin/cluster-config")
    async def update_cluster_config(
        request: Request,
        x_api_key: str | None = Header(None, alias="X-API-Key"),
    ):
        """更新集群全局参数（心跳间隔、容量上报间隔、节点超时等）。"""
        if not require_admin(x_api_key):
            raise HTTPException(status_code=403, detail="Admin access required")
        body = await request.json()
        try:
            app.state.cluster_config.update(body)
            _save_config()
            return {"status": "ok", "config": app.state.cluster_config.to_dict()}
        except (ValueError, TypeError) as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    @app.get("/admin/resource-policy/default")
    async def get_default_resource_policy(
        x_api_key: str | None = Header(None, alias="X-API-Key"),
    ):
        """获取默认资源策略。"""
        if not require_admin(x_api_key):
            raise HTTPException(status_code=403, detail="Admin access required")
        return app.state.cluster_config.default_resource_policy.to_dict()

    @app.put("/admin/resource-policy/default")
    async def update_default_resource_policy(
        request: Request,
        x_api_key: str | None = Header(None, alias="X-API-Key"),
    ):
        """更新默认资源策略。"""
        if not require_admin(x_api_key):
            raise HTTPException(status_code=403, detail="Admin access required")
        body = await request.json()
        try:
            from asc.core.cluster_config import NodeResourcePolicy
            app.state.cluster_config.default_resource_policy = NodeResourcePolicy.from_dict(body)
            _save_config()
            return {"status": "ok", "policy": app.state.cluster_config.default_resource_policy.to_dict()}
        except (ValueError, TypeError) as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    @app.get("/admin/resource-policy/nodes")
    async def list_node_policies(
        x_api_key: str | None = Header(None, alias="X-API-Key"),
    ):
        """列出所有节点资源策略。"""
        if not require_admin(x_api_key):
            raise HTTPException(status_code=403, detail="Admin access required")
        policies = {
            node_id: policy.to_dict()
            for node_id, policy in app.state.cluster_config.node_policies.items()
        }
        return {"node_policies": policies}

    @app.get("/admin/resource-policy/nodes/{node_id}")
    async def get_node_policy(
        node_id: str,
        x_api_key: str | None = Header(None, alias="X-API-Key"),
    ):
        """获取特定节点资源策略。"""
        if not require_admin(x_api_key):
            raise HTTPException(status_code=403, detail="Admin access required")
        policy = app.state.cluster_config.get_node_policy(node_id)
        return {"node_id": node_id, "policy": policy.to_dict()}

    @app.put("/admin/resource-policy/nodes/{node_id}")
    async def update_node_policy(
        node_id: str,
        request: Request,
        x_api_key: str | None = Header(None, alias="X-API-Key"),
    ):
        """更新特定节点资源策略。"""
        if not require_admin(x_api_key):
            raise HTTPException(status_code=403, detail="Admin access required")
        body = await request.json()
        try:
            app.state.cluster_config.update_node_policy(node_id, body)
            _save_config()
            policy = app.state.cluster_config.get_node_policy(node_id)
            return {"status": "ok", "node_id": node_id, "policy": policy.to_dict()}
        except (ValueError, TypeError) as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    @app.delete("/admin/resource-policy/nodes/{node_id}")
    async def delete_node_policy(
        node_id: str,
        x_api_key: str | None = Header(None, alias="X-API-Key"),
    ):
        """删除节点资源策略，回退到默认。"""
        if not require_admin(x_api_key):
            raise HTTPException(status_code=403, detail="Admin access required")
        app.state.cluster_config.remove_node_policy(node_id)
        _save_config()
        return {"status": "ok", "message": f"Node {node_id} policy removed, using default"}

    # ---- Server Settings ----

    @app.get("/admin/server-settings")
    async def get_server_settings(
        x_api_key: str | None = Header(None, alias="X-API-Key"),
    ):
        """获取服务器设置。"""
        if not require_admin(x_api_key):
            raise HTTPException(status_code=403, detail="Admin access required")
        return app.state.server_settings

    @app.put("/admin/server-settings")
    async def update_server_settings(
        request: Request,
        x_api_key: str | None = Header(None, alias="X-API-Key"),
    ):
        """更新服务器设置。"""
        if not require_admin(x_api_key):
            raise HTTPException(status_code=403, detail="Admin access required")
        body = await request.json()
        settings = app.state.server_settings
        for key in ("require_auth", "enable_cors", "serve_on_lan",
                     "jit_loading", "auto_unload", "only_keep_last_model"):
            if key in body:
                settings[key] = bool(body[key])
        for key in ("api_port", "max_idle_ttl_minutes"):
            if key in body:
                settings[key] = int(body[key])
        app.state.server_settings = settings

        # 同步设置到所有在线 Worker
        tcp_server = app.state.tcp_server
        if tcp_server is not None:
            from asc.network.protocol import Envelope, Message, Channel, MessageType
            settings_envelope = Envelope(
                channel=Channel.TASK_DISPATCH,
                message=Message(
                    type=MessageType.SERVER_SETTINGS,
                    sender_id="master",
                    payload=settings,
                ),
            )
            sent_count = 0
            try:
                await tcp_server.broadcast(settings_envelope)
                sent_count = len(tcp_server.connections)
            except Exception as e:
                logger.warning("广播设置到 Worker 失败: %s", e)
            return {"status": "ok", "settings": settings, "workers_notified": sent_count}

        return {"status": "ok", "settings": settings, "workers_notified": 0}

    # ---- Autostart Management ----

    @app.get("/admin/autostart")
    async def get_autostart_status(
        x_api_key: str | None = Header(None, alias="X-API-Key"),
    ):
        """获取 Master/Worker 开机自启动状态。"""
        if not require_admin(x_api_key):
            raise HTTPException(status_code=403, detail="Admin access required")
        from asc.utils.autostart import get_autostart_status as _get_status
        return await asyncio.to_thread(_get_status)

    @app.post("/admin/autostart/{role}/enable")
    async def enable_autostart(
        role: str,
        request: Request,
        x_api_key: str | None = Header(None, alias="X-API-Key"),
    ):
        """启用指定角色的开机自启动。"""
        if not require_admin(x_api_key):
            raise HTTPException(status_code=403, detail="Admin access required")
        if role not in ("master", "worker"):
            raise HTTPException(status_code=422, detail="role must be 'master' or 'worker'")
        body = {}
        try:
            body = await request.json()
        except (json.JSONDecodeError, ValueError):
            pass

        # Worker 角色自动填充 Master 地址（如果未指定）
        if role == "worker" and "master_host" not in body:
            # 从当前 Master 配置获取地址
            body.setdefault("master_host", "127.0.0.1")
            body.setdefault("master_port", 52414)
            # 如果 Master 有 TCP server，使用实际监听地址
            if tcp_server:
                body["master_port"] = tcp_server.port

        from asc.utils.autostart import enable_autostart as _enable
        result = await asyncio.to_thread(_enable, role, body)
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["message"])
        return result

    @app.post("/admin/autostart/{role}/disable")
    async def disable_autostart(
        role: str,
        x_api_key: str | None = Header(None, alias="X-API-Key"),
    ):
        """禁用指定角色的开机自启动。"""
        if not require_admin(x_api_key):
            raise HTTPException(status_code=403, detail="Admin access required")
        if role not in ("master", "worker"):
            raise HTTPException(status_code=422, detail="role must be 'master' or 'worker'")
        from asc.utils.autostart import disable_autostart as _disable
        result = await asyncio.to_thread(_disable, role)
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["message"])
        return result

    # ---- Model Configs ----

    @app.get("/admin/model-configs")
    async def list_model_configs(
        x_api_key: str | None = Header(None, alias="X-API-Key"),
    ):
        """列出所有模型配置。"""
        if not require_admin(x_api_key):
            raise HTTPException(status_code=403, detail="Admin access required")
        models = {
            model_id: config.to_dict()
            for model_id, config in app.state.cluster_config.model_configs.items()
        }
        return {"model_configs": models}

    @app.get("/admin/model-configs/{model_id}")
    async def get_model_config(
        model_id: str,
        x_api_key: str | None = Header(None, alias="X-API-Key"),
    ):
        """获取特定模型配置。"""
        if not require_admin(x_api_key):
            raise HTTPException(status_code=403, detail="Admin access required")
        config = app.state.cluster_config.get_model_config(model_id)
        if config is None:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not configured")
        return {"model_id": model_id, "config": config.to_dict()}

    @app.put("/admin/model-configs/{model_id}")
    async def update_model_config(
        model_id: str,
        request: Request,
        x_api_key: str | None = Header(None, alias="X-API-Key"),
    ):
        """更新模型配置（上下文长度、最大token数、GPU层数等）。"""
        if not require_admin(x_api_key):
            raise HTTPException(status_code=403, detail="Admin access required")
        body = await request.json()
        try:
            app.state.cluster_config.update_model(model_id, body)
            config = app.state.cluster_config.get_model_config(model_id)
            return {"status": "ok", "model_id": model_id, "config": config.to_dict()}
        except (ValueError, TypeError) as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    @app.delete("/admin/model-configs/{model_id}")
    async def delete_model_config(
        model_id: str,
        x_api_key: str | None = Header(None, alias="X-API-Key"),
    ):
        """删除模型配置。"""
        if not require_admin(x_api_key):
            raise HTTPException(status_code=403, detail="Admin access required")
        app.state.cluster_config.remove_model_config(model_id)
        return {"status": "ok", "message": f"Model {model_id} config removed"}

    @app.post("/admin/estimate-resources")
    async def estimate_resources(
        request: Request,
        x_api_key: str | None = Header(None, alias="X-API-Key"),
    ):
        """估算模型资源使用量。"""
        if not require_admin(x_api_key):
            raise HTTPException(status_code=403, detail="Admin access required")
        body = await request.json()
        model_path = body.get("model_path", "")
        config = body.get("config", {})
        node_id = body.get("node_id", "")

        # 获取节点可用资源
        node_resources = getattr(app.state, "node_resources", {})
        res = node_resources.get(node_id, {})
        available_vram_mb = sum(g.get("vram_free_mb", 0) for g in res.get("gpus", []))
        available_ram_mb = res.get("memory_free_mb", 0)

        from asc.worker.resource_estimator import estimate_model_resources
        estimate = estimate_model_resources(
            model_path=model_path,
            config=config,
            available_vram_mb=available_vram_mb,
            available_ram_mb=available_ram_mb,
        )

        return {
            "model_vram_mb": estimate.model_vram_bytes // (1024 * 1024),
            "context_vram_mb": estimate.context_vram_bytes // (1024 * 1024),
            "total_vram_mb": estimate.total_vram_bytes // (1024 * 1024),
            "model_ram_mb": estimate.model_ram_bytes // (1024 * 1024),
            "context_ram_mb": estimate.context_ram_bytes // (1024 * 1024),
            "total_ram_mb": estimate.total_ram_bytes // (1024 * 1024),
            "confidence": estimate.confidence,
            "passes_guardrails": estimate.passes_guardrails,
            "warnings": estimate.warnings,
            "available_vram_mb": available_vram_mb,
            "available_ram_mb": available_ram_mb,
        }

    # ---- WebSocket 实时推送 ----

    @app.websocket("/ws/cluster")
    async def websocket_cluster(websocket: WebSocket):
        """WebSocket 实时集群状态推送。"""
        token = websocket.query_params.get("token", "")
        if not require_admin(token):
            # 必须先 accept 再 close，否则客户端收到 404 而非认证失败
            await websocket.accept()
            await websocket.close(code=1008, reason="Unauthorized")
            return

        await websocket.accept()
        app.state.ws_clients.add(websocket)
        try:
            # 发送初始状态
            await _send_cluster_state(websocket, app)
            # 保持连接，等待前端心跳
            while True:
                data = await websocket.receive_text()
                # 可处理前端请求，如刷新状态
                if data == "refresh":
                    await _send_cluster_state(websocket, app)
        except WebSocketDisconnect:
            pass
        finally:
            app.state.ws_clients.discard(websocket)

    # ---- 集群广播配置 ----

    @app.post("/admin/broadcast-config")
    async def broadcast_config(
        request: Request,
        x_api_key: str | None = Header(None, alias="X-API-Key"),
    ):
        """向所有 Worker 节点广播配置更新。"""
        if not require_admin(x_api_key):
            raise HTTPException(status_code=403, detail="Admin access required")

        body = await request.json()
        action = body.get("action", "")
        valid_actions = {"reload_config", "update_policy", "shutdown"}
        if action not in valid_actions:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid action. Must be one of: {valid_actions}",
            )

        # 构建广播消息
        payload = {"action": action, "data": body.get("policy") or body.get("data", {})}
        envelope = Envelope(
            channel=Channel.COMMANDS,
            message=Message(
                type=MessageType.CONFIG_UPDATE,
                sender_id="master",
                payload=payload,
            ),
        )

        broadcasted = 0
        if app.state.tcp_server is not None:
            try:
                await app.state.tcp_server.broadcast(envelope)
                broadcasted = len(app.state.tcp_server.connections)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Broadcast failed: {e}") from e

        # 同时通知 WebSocket 客户端
        await _notify_ws_clients(app, {"type": "config_update", "data": payload})

        return {
            "status": "ok",
            "action": action,
            "broadcasted_to": broadcasted,
            "message": f"Config broadcast sent to {broadcasted} nodes",
        }

    # ---- 模型部署端点 ----

    @app.post("/admin/deploy-model")
    async def deploy_model(
        request: Request,
        x_api_key: str | None = Header(None, alias="X-API-Key"),
    ):
        """部署模型到集群。"""
        if not require_admin(x_api_key):
            raise HTTPException(status_code=403, detail="Admin access required")

        body = await request.json()
        model_id = body.get("model_id", "")
        source = body.get("source", "")
        uri = body.get("uri", "")
        filename = body.get("filename")
        config = body.get("config", {})

        if not model_id or not uri:
            raise HTTPException(status_code=422, detail="model_id and uri are required")

        # SSRF 防护：限制 URI scheme
        parsed_uri = urlparse(uri)
        if parsed_uri.scheme not in ("http", "https"):
            raise HTTPException(status_code=422, detail="uri must use http or https scheme")

        # 路径遍历防护：校验文件名
        if filename and not validate_model_id(filename):
            raise HTTPException(status_code=422, detail="invalid filename")

        # 下载/配置模型
        from asc.core.model_downloader import DownloadRequest, DownloadSource, ModelDownloader

        models_dir = Path("./models")
        models_dir.mkdir(parents=True, exist_ok=True)
        downloader = ModelDownloader(models_dir)

        source_map = {
            "huggingface": DownloadSource.HUGGINGFACE,
            "modelscope": DownloadSource.MODELSCOPE,
            "manual": DownloadSource.MANUAL,
        }
        download_source = source_map.get(source, DownloadSource.MANUAL)

        download_request = DownloadRequest(
            source=download_source,
            model_uri=uri,
            filename=filename,
        )

        # 异步执行下载
        import asyncio

        def do_download():
            gen = downloader.download(download_request)
            result = None
            while True:
                try:
                    next(gen)
                except StopIteration as e:
                    result = e.value
                    break
            return result

        try:
            result = await asyncio.to_thread(do_download)
            if result and result.success:
                # 保存模型配置
                from asc.core.cluster_config import ModelConfig
                model_config = ModelConfig(
                    model_id=model_id,
                    context_length=config.get("context_length", 4096),
                    gpu_layers=config.get("gpu_layers", -1),
                )
                app.state.cluster_config.add_model_config(model_config)

                # 分发模型到 Worker 节点
                distribute_info = {"distributed": False, "assignments": []}
                tcp_server = getattr(app.state, "tcp_server", None)
                if tcp_server is not None:
                    try:
                        from asc.master.model_distributor import ModelDistributor

                        distributor = ModelDistributor(tcp_server=tcp_server)

                        # 构建目标节点列表（使用 conn_id）
                        target_nodes = {}
                        node_resources_map = {}
                        state = app.state.cluster_state

                        # 需要从 tcp_server 获取 conn_id -> node_id 映射
                        # 这里使用 _conn_node_map 如果可用，否则跳过分发
                        conn_node_map = getattr(app.state, "conn_node_map", {})

                        for nid, ninfo in state.nodes.items():
                            # 查找该节点的 conn_id
                            conn_id = None
                            for cid, mapped_nid in conn_node_map.items():
                                if mapped_nid == str(nid):
                                    conn_id = cid
                                    break
                            if conn_id is None:
                                continue

                            ip = ninfo.ip
                            port = ninfo.port
                            target_nodes[str(nid)] = {"conn_id": conn_id}
                            node_resources_map[str(nid)] = {
                                "ip": ip, "port": port,
                                "vram_free_mb": 0,  # 从注册信息获取
                            }

                        if target_nodes:
                            # 分发模型文件
                            distribute_results = await distributor.distribute_to_cluster(
                                model_id=model_id,
                                model_path=result.file_path,
                                nodes=target_nodes,
                            )

                            # 规划层分配
                            total_layers = config.get("total_layers", 32)
                            assignments = distributor.plan_layer_assignment(
                                model_id=model_id,
                                model_path=result.file_path,
                                total_layers=total_layers,
                                node_resources=node_resources_map,
                                distribute_results=distribute_results,
                            )

                            distribute_info = {
                                "distributed": True,
                                "assignments": [
                                    {
                                        "node_id": a.node_id,
                                        "layers": f"L{a.start_layer}-L{a.end_layer}",
                                        "gpu_layers": a.gpu_layers,
                                    }
                                    for a in assignments
                                ],
                            }

                            # 通知 WebSocket 客户端
                            await _notify_ws_clients(app, {
                                "type": "model_distributed",
                                "data": distribute_info,
                            })
                    except Exception as e:
                        logger.warning("模型分发失败（模型已下载到 Master）: %s", e)

                return {
                    "status": "ok",
                    "model_id": model_id,
                    "file_path": result.file_path,
                    "file_size_mb": result.file_size_mb,
                    "distribute": distribute_info,
                    "message": f"Model {model_id} deployed successfully",
                }
            else:
                error = result.error if result else "Unknown error"
                raise HTTPException(status_code=500, detail=f"Download failed: {error}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Deploy failed: {e}") from e

    # ---- API 配置端点 ----

    @app.put("/admin/api-config")
    async def update_api_config(
        request: Request,
        x_api_key: str | None = Header(None, alias="X-API-Key"),
    ):
        """更新 API 服务配置。"""
        if not require_admin(x_api_key):
            raise HTTPException(status_code=403, detail="Admin access required")

        body = await request.json()
        enabled = body.get("enabled", False)
        api_key = body.get("api_key")
        port = body.get("port", 8000)

        # 更新 API Key
        if api_key:
            from asc.api.auth import ApiKeyStore, set_key_store
            store = ApiKeyStore(admin_keys=[api_key])
            set_key_store(store)

        return {
            "status": "ok",
            "api_enabled": enabled,
            "api_port": port,
        }

    # ---- 模型文件管理端点 ----

    def _get_models_dir() -> Path:
        """获取模型存储目录。"""
        models_dir = Path.home() / ".asc" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        return models_dir

    @app.get("/admin/models/local")
    async def list_local_models(
        x_api_key: str | None = Header(None, alias="X-API-Key"),
    ):
        """扫描 Master 节点模型目录，列出所有模型文件。"""
        if not require_admin(x_api_key):
            raise HTTPException(status_code=403, detail="Admin access required")

        models_dir = _get_models_dir()
        models = []
        for f in sorted(models_dir.rglob("*")):
            if f.is_file():
                rel_path = f.relative_to(models_dir)
                size_mb = round(f.stat().st_size / (1024 * 1024), 2)
                models.append({
                    "name": str(rel_path),
                    "size_mb": size_mb,
                    "filename": f.name,
                })
        return {"models": models, "directory": str(models_dir)}

    @app.post("/admin/models/upload")
    async def upload_model(
        request: Request,
        x_api_key: str | None = Header(None, alias="X-API-Key"),
    ):
        """上传模型文件到 Master 节点。支持 multipart/form-data，流式写入大文件。

        对于大文件（>100MB），建议使用 application/octet-stream + query param 方式，
        避免 multipart 解析的内存开销。
        """
        if not require_admin(x_api_key):
            raise HTTPException(status_code=403, detail="Admin access required")

        models_dir = _get_models_dir()
        content_type = request.headers.get("content-type", "")

        if "application/octet-stream" in content_type:
            # 流式上传：前端直接发送文件二进制流，文件名通过 query param 传递
            filename = request.query_params.get("filename", "upload.bin")
            subfolder = request.query_params.get("subfolder", "").strip("/")

            # 安全校验：文件名路径遍历检查
            if ".." in filename or ".." in subfolder:
                raise HTTPException(status_code=400, detail="Filename or subfolder contains path traversal")
            # 安全校验：扩展名白名单
            file_ext = Path(filename).suffix.lower()
            if file_ext and file_ext not in UPLOAD_ALLOWED_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=f"File extension '{file_ext}' not allowed. Allowed: {sorted(UPLOAD_ALLOWED_EXTENSIONS)}",
                )

            target_dir = models_dir / subfolder if subfolder else models_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / filename

            # 安全校验：最终路径必须在 models_dir 内
            try:
                target_path.resolve().relative_to(models_dir.resolve())
            except ValueError:
                raise HTTPException(status_code=400, detail="Target path escapes models directory")

            chunk_size = 4 * 1024 * 1024  # 4MB
            total_size = 0
            try:
                with open(target_path, "wb") as f:
                    async for chunk in request.stream():
                        if chunk:
                            f.write(chunk)
                            total_size += len(chunk)
                            # 安全校验：文件大小限制
                            if total_size > UPLOAD_MAX_SIZE_BYTES:
                                target_path.unlink(missing_ok=True)
                                raise HTTPException(status_code=413, detail="File exceeds maximum upload size")
            except HTTPException:
                raise
            except Exception as e:
                if target_path.exists():
                    target_path.unlink()
                raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

            return {
                "status": "ok",
                "filename": filename,
                "size_mb": round(total_size / (1024 * 1024), 2),
                "path": str(target_path.relative_to(models_dir)),
            }
        else:
            # multipart/form-data 上传（兼容小文件）
            from fastapi import UploadFile

            try:
                form = await request.form()
            except Exception as e:
                raise HTTPException(status_code=413, detail=f"File too large or upload error: {e}")

            upload_file = None
            subfolder = ""
            for key, value in form.items():
                if key == "subfolder":
                    subfolder = str(value).strip("/")
                elif isinstance(value, UploadFile):
                    upload_file = value

            if upload_file is None:
                raise HTTPException(status_code=400, detail="No file uploaded")

            # 安全校验：文件名路径遍历检查
            upload_filename = upload_file.filename or "upload.bin"
            if ".." in upload_filename or ".." in subfolder:
                raise HTTPException(status_code=400, detail="Filename or subfolder contains path traversal")
            # 安全校验：扩展名白名单
            file_ext = Path(upload_filename).suffix.lower()
            if file_ext and file_ext not in UPLOAD_ALLOWED_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=f"File extension '{file_ext}' not allowed. Allowed: {sorted(UPLOAD_ALLOWED_EXTENSIONS)}",
                )

            target_dir = models_dir / subfolder if subfolder else models_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / upload_filename

            # 安全校验：最终路径必须在 models_dir 内
            try:
                target_path.resolve().relative_to(models_dir.resolve())
            except ValueError:
                raise HTTPException(status_code=400, detail="Target path escapes models directory")

            chunk_size = 4 * 1024 * 1024  # 4MB
            total_size = 0
            try:
                with open(target_path, "wb") as f:
                    while True:
                        chunk = await upload_file.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        total_size += len(chunk)
                        # 安全校验：文件大小限制
                        if total_size > UPLOAD_MAX_SIZE_BYTES:
                            target_path.unlink(missing_ok=True)
                            raise HTTPException(status_code=413, detail="File exceeds maximum upload size")
            except HTTPException:
                raise
            except Exception as e:
                if target_path.exists():
                    target_path.unlink()
                raise HTTPException(status_code=500, detail=f"Upload failed: {e}")
            finally:
                await upload_file.close()

            return {
                "status": "ok",
                "filename": upload_file.filename,
                "size_mb": round(total_size / (1024 * 1024), 2),
                "path": str(target_path.relative_to(models_dir)),
            }

    @app.delete("/admin/models/local/{model_path:path}")
    async def delete_local_model(
        model_path: str,
        x_api_key: str | None = Header(None, alias="X-API-Key"),
    ):
        """删除 Master 节点上的模型文件。"""
        if not require_admin(x_api_key):
            raise HTTPException(status_code=403, detail="Admin access required")

        models_dir = _get_models_dir()
        target = models_dir / model_path

        # 安全检查：防止路径遍历
        try:
            target.resolve().relative_to(models_dir.resolve())
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid model path")

        if not target.exists():
            raise HTTPException(status_code=404, detail="Model file not found")

        if target.is_file():
            target.unlink()
        elif target.is_dir():
            import shutil
            shutil.rmtree(target)

        return {"status": "ok", "deleted": model_path}

    async def _distribute_and_load_model(
        app: FastAPI,
        model_id: str,
        model_path: str,
        full_path: str,
        load_params: dict[str, Any],
    ) -> None:
        """后台任务：将模型文件分发给各 Worker，然后发送 LOAD_MODEL。"""
        tcp_server = getattr(app.state, "tcp_server", None)
        conn_node_map = getattr(app.state, "conn_node_map", {})
        cluster_state = getattr(app.state, "cluster_state", None)

        if tcp_server is None or cluster_state is None:
            logger.error("后台分发模型失败: tcp_server 或 cluster_state 不可用")
            return

        from asc.master.model_distributor import ModelDistributor

        # 构建在线节点字典 {node_id: {"conn_id": str}}
        nodes: dict[str, dict[str, Any]] = {}
        for cid, nid in conn_node_map.items():
            if nid in cluster_state.nodes and cluster_state.nodes[nid].runner_status == "online":
                nodes[nid] = {"conn_id": cid}

        if not nodes:
            logger.warning("没有在线节点可用于分发模型 %s", model_id)
            await _notify_ws_clients(app, {
                "type": "model_load_progress",
                "data": {"model_id": model_id, "status": "failed", "error": "没有在线节点"},
            })
            return

        # 通知前端开始分发
        await _notify_ws_clients(app, {
            "type": "model_load_progress",
            "data": {
                "model_id": model_id,
                "status": "distributing",
                "nodes_total": len(nodes),
                "nodes_done": 0,
                "nodes_failed": 0,
            },
        })

        # 使用 MasterNode 共享的 ModelDistributor 实例，确保 MODEL_CHUNK_ACK
        # 能被 MasterNode._on_frame 正确路由到同一个实例的 _pending_acks
        distributor = getattr(app.state, "model_distributor", None)
        if distributor is None:
            logger.warning("app.state.model_distributor 未设置，创建临时实例（ACK 可能无法正确匹配）")
            distributor = ModelDistributor(tcp_server=tcp_server)

        try:
            distribute_results = await distributor.distribute_to_cluster(
                model_id=model_id,
                model_path=full_path,
                nodes=nodes,
            )
        except Exception as e:
            logger.error("模型 %s 分发异常: %s", model_id, e, exc_info=True)
            await _notify_ws_clients(app, {
                "type": "model_load_progress",
                "data": {"model_id": model_id, "status": "failed", "error": f"分发失败: {e}"},
            })
            return

        # 统计分发结果
        success_nodes = [r.node_id for r in distribute_results if r.success]
        failed_nodes = [r.node_id for r in distribute_results if not r.success]

        for r in distribute_results:
            if not r.success:
                logger.error("模型 %s 分发到 %s 失败: %s", model_id, r.node_id, r.error)

        logger.info(
            "模型 %s 分发完成: 成功 %d, 失败 %d",
            model_id, len(success_nodes), len(failed_nodes),
        )

        await _notify_ws_clients(app, {
            "type": "model_load_progress",
            "data": {
                "model_id": model_id,
                "status": "distributing_done",
                "nodes_total": len(nodes),
                "nodes_done": len(success_nodes),
                "nodes_failed": len(failed_nodes),
            },
        })

        # 向分发成功的节点发送 LOAD_MODEL
        loaded_nodes: list[str] = []
        for nid in success_nodes:
            conn_id = None
            for cid, n_id in conn_node_map.items():
                if n_id == nid:
                    conn_id = cid
                    break
            if conn_id is None:
                continue

            await _notify_ws_clients(app, {
                "type": "model_load_progress",
                "data": {"model_id": model_id, "status": "loading", "node_id": str(nid), "node_status": "dispatched"},
            })

            envelope = Envelope(
                channel=Channel.TASK_DISPATCH,
                message=Message(
                    type=MessageType.LOAD_MODEL,
                    sender_id="master",
                    payload=load_params,
                ),
                target=str(nid),
            )
            try:
                await tcp_server.send(conn_id, envelope)
                loaded_nodes.append(str(nid))
            except (ConnectionError, OSError, RuntimeError) as e:
                logger.warning("发送 LOAD_MODEL 到 %s 失败: %s [类型: %s]", nid, e, type(e).__name__)
                failed_nodes.append(str(nid))
                await _notify_ws_clients(app, {
                    "type": "model_load_progress",
                    "data": {"model_id": model_id, "status": "loading", "node_id": str(nid), "node_status": "failed", "error": str(e)},
                })

        await _notify_ws_clients(app, {
            "type": "config_update",
            "data": {"action": "model_loaded", "model_id": model_id, "nodes": loaded_nodes},
        })

    @app.post("/admin/models/load-to-cluster")
    async def load_model_to_cluster(
        request: Request,
        background_tasks: BackgroundTasks,
        x_api_key: str | None = Header(None, alias="X-API-Key"),
    ):
        """将 Master 节点上的模型载入集群（部署到 Worker）。"""
        if not require_admin(x_api_key):
            raise HTTPException(status_code=403, detail="Admin access required")

        body = await request.json()
        model_path = body.get("model_path", "")
        model_id = body.get("model_id", "")
        model_domain_type = body.get("model_domain_type", "llm")
        gpu_layers = body.get("gpu_layers", -1)
        context_length = body.get("context_length", 4096)
        cpu_threads = body.get("cpu_threads", 0)
        eval_batch_size = body.get("eval_batch_size", 512)
        physical_batch_size = body.get("physical_batch_size", 0)
        max_concurrent = body.get("max_concurrent", 1)
        flash_attention = body.get("flash_attention", True)
        keep_in_memory = body.get("keep_in_memory", True)
        use_mmap = body.get("use_mmap", True)
        offload_kv_to_gpu = body.get("offload_kv_to_gpu", True)
        unified_kv_cache = body.get("unified_kv_cache", False)
        kv_quantization = body.get("kv_quantization", "")
        rope_freq_base = body.get("rope_freq_base", 0.0)
        rope_freq_scale = body.get("rope_freq_scale", 0.0)
        seed = body.get("seed", -1)
        # GPU management
        gpu_offload_ratio = body.get("gpu_offload_ratio", "max")
        gpu_split_strategy = body.get("gpu_split_strategy", "evenly")
        gpu_custom_ratio = body.get("gpu_custom_ratio", [])
        disabled_gpus = body.get("disabled_gpus", [])
        # New loading params
        use_fp16_for_kv_cache = body.get("use_fp16_for_kv_cache", True)
        try_direct_io = body.get("try_direct_io", False)
        gpu_strict_vram_cap = body.get("gpu_strict_vram_cap", False)
        # Inference params
        min_p_sampling = body.get("min_p_sampling", 0.0)
        repeat_penalty = body.get("repeat_penalty", 0.0)
        presence_penalty = body.get("presence_penalty", 0.0)
        frequency_penalty = body.get("frequency_penalty", 0.0)
        context_overflow_policy = body.get("context_overflow_policy", "truncateMiddle")
        # Reasoning parsing
        reasoning_parsing_enabled = body.get("reasoning_parsing_enabled", True)
        reasoning_start_string = body.get("reasoning_start_string", "<think>")
        reasoning_end_string = body.get("reasoning_end_string", "</think>")
        # Speculative decoding
        speculative_draft_model = body.get("speculative_draft_model", "")
        speculative_draft_max_tokens = body.get("speculative_draft_max_tokens", 0)

        if not model_path:
            raise HTTPException(status_code=400, detail="model_path is required")

        models_dir = _get_models_dir()
        full_path = models_dir / model_path

        # 安全检查：禁止路径遍历（不 resolve 符号链接，允许符号链接指向合法路径）
        if ".." in Path(model_path).parts:
            raise HTTPException(status_code=400, detail="Invalid model path")

        if not full_path.exists():
            raise HTTPException(status_code=404, detail="Model file not found")

        # 使用模型文件名作为 model_id（如果未指定）
        if not model_id:
            model_id = full_path.stem

        # 保存模型配置到 ClusterConfig
        cluster_config = app.state.cluster_config
        if cluster_config is not None:
            from asc.core.cluster_config import ModelConfig
            model_config = ModelConfig(
                model_id=model_id,
                model_domain_type=model_domain_type,
                context_length=context_length,
                gpu_layers=gpu_layers,
                gpu_offload_ratio=gpu_offload_ratio,
                gpu_split_strategy=gpu_split_strategy,
                gpu_custom_ratio=gpu_custom_ratio,
                disabled_gpus=disabled_gpus,
                flash_attention=flash_attention,
                offload_kv_cache_to_gpu=offload_kv_to_gpu,
                use_fp16_for_kv_cache=use_fp16_for_kv_cache,
                try_mmap=use_mmap,
                try_direct_io=try_direct_io,
                eval_batch_size=eval_batch_size,
                physical_batch_size=physical_batch_size,
                keep_model_in_memory=keep_in_memory,
                gpu_strict_vram_cap=gpu_strict_vram_cap,
                min_p_sampling=min_p_sampling,
                repeat_penalty=repeat_penalty,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                context_overflow_policy=context_overflow_policy,
                reasoning_parsing_enabled=reasoning_parsing_enabled,
                reasoning_start_string=reasoning_start_string,
                reasoning_end_string=reasoning_end_string,
                speculative_draft_model=speculative_draft_model,
                speculative_draft_max_tokens=speculative_draft_max_tokens,
            )
            cluster_config.add_model_config(model_config)

        # 构建完整的模型加载参数
        # 使用文件名而非 Master 绝对路径，让 Worker 在本地 models_dir 中解析
        load_params = {
            "model_id": model_id,
            "model_path": full_path.name,
            "model_domain_type": model_domain_type,
            "gpu_layers": gpu_layers,
            "context_length": context_length,
            "cpu_threads": cpu_threads,
            "batch_size": eval_batch_size,
            "physical_batch_size": physical_batch_size,
            "max_concurrent": max_concurrent,
            "flash_attention": flash_attention,
            "keep_in_memory": keep_in_memory,
            "use_mmap": use_mmap,
            "offload_kv_to_gpu": offload_kv_to_gpu,
            "use_fp16_for_kv_cache": use_fp16_for_kv_cache,
            "unified_kv_cache": unified_kv_cache,
            "kv_quantization": kv_quantization if kv_quantization and kv_quantization != "none" else "",
            "rope_freq_base": rope_freq_base,
            "rope_freq_scale": rope_freq_scale,
            "seed": seed,
            # GPU management
            "gpu_offload_ratio": gpu_offload_ratio,
            "gpu_split_strategy": gpu_split_strategy,
            "gpu_custom_ratio": gpu_custom_ratio,
            "disabled_gpus": disabled_gpus,
            # New loading params
            "try_direct_io": try_direct_io,
            "gpu_strict_vram_cap": gpu_strict_vram_cap,
            # Inference params
            "min_p_sampling": min_p_sampling,
            "repeat_penalty": repeat_penalty,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "context_overflow_policy": context_overflow_policy,
            # Reasoning parsing
            "reasoning_parsing_enabled": reasoning_parsing_enabled,
            "reasoning_start_string": reasoning_start_string,
            "reasoning_end_string": reasoning_end_string,
            # Speculative decoding
            "speculative_draft_model": speculative_draft_model,
            "speculative_draft_max_tokens": speculative_draft_max_tokens,
        }

        # 启动后台任务：先分发模型文件到各 Worker，再发送 LOAD_MODEL
        background_tasks.add_task(
            _distribute_and_load_model,
            app=app,
            model_id=model_id,
            model_path=model_path,
            full_path=str(full_path),
            load_params=load_params,
        )

        return {
            "status": "distributing",
            "model_id": model_id,
            "model_path": model_path,
            "full_path": str(full_path),
            "message": "模型分发已启动，完成后将自动加载到各节点",
        }

    @app.post("/admin/models/unload-from-cluster")
    async def unload_model_from_cluster(
        request: Request,
        x_api_key: str | None = Header(None, alias="X-API-Key"),
    ):
        """从集群卸载模型。"""
        if not require_admin(x_api_key):
            raise HTTPException(status_code=403, detail="Admin access required")
        body = await request.json()
        model_id = body.get("model_id", "")
        if not model_id:
            raise HTTPException(status_code=400, detail="model_id is required")

        tcp_server = getattr(app.state, "tcp_server", None)
        conn_node_map = getattr(app.state, "conn_node_map", {})
        cluster_state = getattr(app.state, "cluster_state", None)
        unloaded_nodes = []

        if tcp_server is not None and cluster_state is not None:
            from asc.network.protocol import Envelope, Message, MessageType, Channel

            # 通知前端：开始卸载
            await _notify_ws_clients(app, {
                "type": "model_unload_progress",
                "data": {"model_id": model_id, "status": "unloading", "nodes_total": sum(1 for n in cluster_state.nodes.values() if n.runner_status == "online"), "nodes_done": 0},
            })

            for nid, info in cluster_state.nodes.items():
                if info.runner_status != "online":
                    continue
                # 只向加载了该模型的节点发送卸载请求
                node_resources = getattr(app.state, "node_resources", {})
                res = node_resources.get(str(nid), {})
                loaded_models = res.get("loaded_model_ids", [])
                if model_id not in loaded_models:
                    continue

                conn_id = None
                for cid, n_id in conn_node_map.items():
                    if n_id == nid:
                        conn_id = cid
                        break
                if conn_id is None:
                    continue

                envelope = Envelope(
                    channel=Channel.TASK_DISPATCH,
                    message=Message(
                        type=MessageType.UNLOAD_MODEL,
                        sender_id="master",
                        payload={"model_id": model_id},
                    ),
                    target=str(nid),
                )
                try:
                    await tcp_server.send(conn_id, envelope)
                    unloaded_nodes.append(str(nid))
                except (ConnectionError, OSError, RuntimeError) as e:
                    logger.warning("发送 UNLOAD_MODEL 到 %s 失败: %s", nid, e)

        return {
            "status": "ok",
            "model_id": model_id,
            "unloaded_nodes": unloaded_nodes,
        }

    return app


async def _send_cluster_state(websocket: WebSocket, app: FastAPI) -> None:
    """发送当前集群状态到 WebSocket 客户端。"""
    state = app.state.cluster_state
    config = app.state.cluster_config
    node_resources = getattr(app.state, "node_resources", {})

    nodes = []
    total_cpu = 0
    total_memory = 0
    total_vram = 0
    total_score = 0.0
    online_count = 0

    if state is not None:
        for nid, info in state.nodes.items():
            node = {
                "node_id": str(nid),
                "ip": info.ip,
                "port": info.port,
                "online": info.runner_status == "online",
                "runner_status": info.runner_status,
            }
            # 从 node_resources 获取资源数据
            res = node_resources.get(str(nid), {})
            if res:
                node["cpu_count"] = res.get("cpu_count", 0)
                node["cpu_brand"] = res.get("cpu_brand", "")
                node["cpu_physical_count"] = res.get("cpu_physical_count", 0)
                node["cpu_freq_mhz"] = res.get("cpu_freq_mhz", 0)
                node["cpu_instruction_set_extensions"] = res.get("cpu_instruction_set_extensions", [])
                node["memory_total_mb"] = res.get("memory_total_mb", 0)
                node["memory_free_mb"] = res.get("memory_free_mb", 0)
                node["gpus"] = res.get("gpus", [])
                node["compute_score"] = res.get("compute_score", 0.0)
                node["disk_free_mb"] = res.get("disk_free_mb", 0)
                node["network_mbps"] = res.get("network_mbps", 0)
                node["loaded_model_ids"] = res.get("loaded_model_ids", [])
                total_cpu += node["cpu_count"]
                total_memory += node["memory_free_mb"]
                gpu_vram = sum(g.get("vram_total_mb", 0) for g in node["gpus"])
                total_vram += gpu_vram
                total_score += node["compute_score"]
            if node["online"]:
                online_count += 1
            nodes.append(node)

    avg_score = total_score / len(nodes) if nodes else 0.0

    await websocket.send_json({
        "type": "cluster_state",
        "data": {
            "nodes_count": len(nodes),
            "online_nodes": online_count,
            "instances_count": len(state.instances) if state else 0,
            "tasks_count": len(state.tasks) if state else 0,
            "total_cpu": total_cpu,
            "total_memory": total_memory,
            "total_vram": total_vram,
            "avg_score": avg_score,
        },
    })

    await websocket.send_json({
        "type": "node_update",
        "data": {
            "nodes": nodes,
            "node_policies": {nid: p.to_dict() for nid, p in config.node_policies.items()} if config else {},
        },
    })


async def _notify_ws_clients(app: FastAPI, message: dict[str, Any]) -> None:
    """通知所有 WebSocket 客户端。"""
    disconnected = set()
    for ws in app.state.ws_clients:
        try:
            await ws.send_json(message)
        except (WebSocketDisconnect, ConnectionResetError, RuntimeError) as e:
            logger.debug("WebSocket 客户端断开连接: %s [类型: %s]", e, type(e).__name__)
            disconnected.add(ws)
        except Exception as e:
            logger.warning(
                "WebSocket 发送消息失败: %s [类型: %s]",
                e, type(e).__name__,
            )
            disconnected.add(ws)
    for ws in disconnected:
        app.state.ws_clients.discard(ws)
    if disconnected:
        logger.info("已清理 %d 个断开的 WebSocket 连接", len(disconnected))


async def _stream_chat(adapter, request, prompt, engine):
    """流式聊天生成器 — SSE 格式。

    使用引擎的流式推理能力，实时发送生成的 token。
    引擎异常会被捕获并转换为 SSE error 事件发送给客户端。
    """
    try:
        # 使用异步流式推理，实时获取 token
        is_first = True
        async for token in engine.submit_async_stream(
            InferenceRequest(
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
        ):
            chunk = adapter.create_chunk(request, delta=token, finish_reason=None, is_first=is_first)
            is_first = False
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


def run_api_server(
    app: FastAPI | None = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    ssl_certfile: str | None = None,
    ssl_keyfile: str | None = None,
    **kwargs: Any,
) -> None:
    """启动 API 服务器（uvicorn）。

    Args:
        app: FastAPI 应用实例，为 None 时自动创建
        host: 监听地址
        port: 监听端口
        ssl_certfile: SSL 证书文件路径（启用 HTTPS）
        ssl_keyfile: SSL 私钥文件路径（启用 HTTPS）
        **kwargs: 其他 uvicorn 配置参数
    """
    import uvicorn

    if app is None:
        app = create_app()

    uvicorn_config: dict[str, Any] = {
        "app": app,
        "host": host,
        "port": port,
        "timeout-keep-alive": 300,
        "limit-max-header-size": 65536,
    }

    if ssl_certfile and ssl_keyfile:
        uvicorn_config["ssl_certfile"] = ssl_certfile
        uvicorn_config["ssl_keyfile"] = ssl_keyfile

    uvicorn_config.update(kwargs)
    uvicorn.run(**uvicorn_config)
