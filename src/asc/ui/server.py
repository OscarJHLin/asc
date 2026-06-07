"""Asc Web UI 服务器。

提供可视化控制台界面，支持实时监控和管理。
集成 InputValidator 和 RateLimiter 安全中间件。
"""

from __future__ import annotations

import asyncio
import os
import secrets
from functools import wraps
from pathlib import Path
from typing import Any, Callable

try:
    from flask import Flask, jsonify, request
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

from asc.api.security import InputValidator, RateLimiter

_rate_limiter = RateLimiter(max_requests=60, window_seconds=60.0)


def _get_client_key() -> str:
    """获取客户端标识（用于限流）。"""
    if FLASK_AVAILABLE:
        return request.remote_addr or "unknown"
    return "unknown"


def rate_limit(f: Callable) -> Callable:
    """Flask 限流装饰器。"""

    @wraps(f)
    def decorated(*args: Any, **kwargs: Any) -> Any:
        client_key = _get_client_key()
        if not asyncio.run(_rate_limiter.is_allowed(client_key)):
            return jsonify({"success": False, "error": "请求过于频繁，请稍后再试"}), 429
        return f(*args, **kwargs)

    return decorated


def create_ui_app(
    engine: Any = None,
    limiter: RateLimiter | None = None,
) -> Flask:
    """创建 Flask UI 应用。"""
    if not FLASK_AVAILABLE:
        raise RuntimeError("Flask 未安装，无法创建 UI 应用。安装: pip install flask")

    global _rate_limiter
    if limiter is not None:
        _rate_limiter = limiter

    template_dir = Path(__file__).parent / "templates"
    static_dir = Path(__file__).parent / "static"

    app = Flask(
        __name__,
        template_folder=str(template_dir),
        static_folder=str(static_dir) if static_dir.exists() else None,
    )
    app.config["SECRET_KEY"] = os.environ.get("ASC_SECRET_KEY") or secrets.token_hex(32)

    app.state = type("State", (), {"engine": engine})()

    @app.route("/health")
    def health():
        return jsonify({"status": "ok", "service": "asc"})

    @app.route("/api/infer", methods=["POST"])
    @rate_limit
    def api_infer():
        """推理 API — 使用 InputValidator 进行输入验证。"""
        data = request.get_json(silent=True) or {}
        prompt = data.get("prompt", "")
        max_tokens = data.get("max_tokens", 128)
        temperature = data.get("temperature", 0.7)
        top_p = data.get("top_p", 1.0)

        # 使用 InputValidator 统一验证
        ok, msg = InputValidator.validate_request(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        if not ok:
            return jsonify({"success": False, "error": msg}), 422

        model = data.get("model", "")
        if model and not model.strip():
            return jsonify({"success": False, "error": "model 格式无效"}), 422

        engine = app.state.engine
        if engine is None:
            return jsonify({"success": False, "error": "推理引擎不可用"}), 503

        try:
            result = engine.infer(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return jsonify(result)
        except Exception as e:
            return jsonify({"success": False, "error": f"推理失败: {e}"}), 500

    return app
