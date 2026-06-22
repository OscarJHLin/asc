"""Asc TLS/SSL 辅助工具。

提供 SSL 上下文创建函数，用于节点间加密通信。
支持：
- 服务端 SSL 上下文（需要证书和私钥）
- 客户端 SSL 上下文（可选 CA 证书验证）
- mTLS 双向认证（客户端也提供证书）
"""

from __future__ import annotations

import logging
import ssl
from pathlib import Path

logger = logging.getLogger(__name__)


def create_server_ssl_context(
    cert_path: str | Path,
    key_path: str | Path,
    ca_cert_path: str | Path | None = None,
) -> ssl.SSLContext:
    """创建服务端 SSL 上下文。

    Args:
        cert_path: 服务端证书文件路径（PEM 格式）
        key_path: 服务端私钥文件路径（PEM 格式）
        ca_cert_path: CA 证书路径（启用 mTLS 双向认证时需要）

    Returns:
        配置好的 ssl.SSLContext

    Raises:
        FileNotFoundError: 证书或密钥文件不存在
    """
    cert_path = Path(cert_path)
    key_path = Path(key_path)

    if not cert_path.exists():
        raise FileNotFoundError(f"证书文件不存在: {cert_path}")
    if not key_path.exists():
        raise FileNotFoundError(f"私钥文件不存在: {key_path}")

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile=str(cert_path), keyfile=str(key_path))

    # mTLS: 要求客户端提供证书
    if ca_cert_path is not None:
        ca_path = Path(ca_cert_path)
        if not ca_path.exists():
            raise FileNotFoundError(f"CA 证书文件不存在: {ca_path}")
        ctx.load_verify_locations(cafile=str(ca_path))
        ctx.verify_mode = ssl.CERT_REQUIRED

    # 安全配置
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2

    logger.info("服务端 SSL 上下文已创建 (cert=%s, mTLS=%s)", cert_path, ca_cert_path is not None)
    return ctx


def create_client_ssl_context(
    ca_cert_path: str | Path | None = None,
    cert_path: str | Path | None = None,
    key_path: str | Path | None = None,
) -> ssl.SSLContext:
    """创建客户端 SSL 上下文。

    Args:
        ca_cert_path: CA 证书路径（验证服务端证书）。为 None 时不验证服务端证书（不安全）。
        cert_path: 客户端证书路径（mTLS 双向认证时需要）
        key_path: 客户端私钥路径（mTLS 双向认证时需要）

    Returns:
        配置好的 ssl.SSLContext
    """
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)

    if ca_cert_path is not None:
        ca_path = Path(ca_cert_path)
        if ca_path.exists():
            ctx.load_verify_locations(cafile=str(ca_path))
        else:
            raise FileNotFoundError(f"CA 证书文件不存在: {ca_path}")
    else:
        # 无 CA 证书时不验证（仅适用于开发环境）
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        logger.warning("未配置 CA 证书，客户端将不验证服务端证书。生产环境请配置 ca_cert_path。")

    # mTLS: 提供客户端证书
    if cert_path is not None and key_path is not None:
        ctx.load_cert_chain(certfile=str(cert_path), keyfile=str(key_path))

    # 安全配置
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2

    return ctx
