"""补充 TLS 模块测试，提升覆盖率至 90%+。

原测试仅覆盖基础场景，本文件补充：
- 服务端 SSL 上下文完整创建
- 客户端 SSL 上下文各种配置组合
- mTLS 双向认证
- 证书不存在时的错误处理
- 安全配置验证
"""

from __future__ import annotations

import ssl
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from asc.network.tls import create_client_ssl_context, create_server_ssl_context


def _generate_self_signed_cert():
    """生成自签名测试证书（使用 Python 标准库，无需 openssl 命令）。"""
    import datetime

    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
    except ImportError:
        pytest.skip("cryptography not installed")

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([
        x509.NameAttribute(x509.NameOID.COUNTRY_NAME, "CN"),
        x509.NameAttribute(x509.NameOID.ORGANIZATION_NAME, "Test"),
        x509.NameAttribute(x509.NameOID.COMMON_NAME, "localhost"),
    ])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=1))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName("localhost")]), critical=False)
        .sign(key, hashes.SHA256())
    )

    cert_pem = cert.public_bytes(serialization.Encoding.PEM)
    key_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return cert_pem, key_pem


@pytest.fixture
def cert_files():
    """创建临时证书文件。"""
    cert_pem, key_pem = _generate_self_signed_cert()
    with tempfile.TemporaryDirectory() as tmpdir:
        cert_path = Path(tmpdir) / "cert.pem"
        key_path = Path(tmpdir) / "key.pem"
        cert_path.write_bytes(cert_pem)
        key_path.write_bytes(key_pem)
        yield cert_path, key_path


class TestServerSSLContext:
    """测试服务端 SSL 上下文创建。"""

    def test_create_basic(self, cert_files):
        """基础服务端 SSL 上下文。"""
        cert_path, key_path = cert_files
        ctx = create_server_ssl_context(cert_path, key_path)
        assert isinstance(ctx, ssl.SSLContext)
        assert ctx.minimum_version == ssl.TLSVersion.TLSv1_2

    def test_create_with_mtls(self, cert_files):
        """启用 mTLS 的服务端 SSL 上下文。"""
        cert_path, key_path = cert_files
        ctx = create_server_ssl_context(cert_path, key_path, ca_cert_path=cert_path)
        assert isinstance(ctx, ssl.SSLContext)
        assert ctx.verify_mode == ssl.CERT_REQUIRED

    def test_cert_not_found(self):
        """证书不存在时应抛出 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError, match="证书文件不存在"):
            create_server_ssl_context("/nonexistent/cert.pem", "/nonexistent/key.pem")

    def test_key_not_found(self, cert_files):
        """私钥不存在时应抛出 FileNotFoundError。"""
        cert_path, _ = cert_files
        with pytest.raises(FileNotFoundError, match="私钥文件不存在"):
            create_server_ssl_context(cert_path, "/nonexistent/key.pem")

    def test_ca_not_found(self, cert_files):
        """CA 证书不存在时应抛出 FileNotFoundError。"""
        cert_path, key_path = cert_files
        with pytest.raises(FileNotFoundError, match="CA 证书文件不存在"):
            create_server_ssl_context(cert_path, key_path, ca_cert_path="/nonexistent/ca.pem")


class TestClientSSLContext:
    """测试客户端 SSL 上下文创建。"""

    def test_no_ca_insecure(self):
        """无 CA 证书时应创建不验证的上下文（开发环境）。"""
        ctx = create_client_ssl_context()
        assert isinstance(ctx, ssl.SSLContext)
        assert ctx.check_hostname is False
        assert ctx.verify_mode == ssl.CERT_NONE

    def test_with_ca(self, cert_files):
        """使用 CA 证书验证服务端。"""
        cert_path, _ = cert_files
        ctx = create_client_ssl_context(ca_cert_path=cert_path)
        assert isinstance(ctx, ssl.SSLContext)
        # 有 CA 时默认会验证
        assert ctx.verify_mode == ssl.CERT_REQUIRED

    def test_ca_not_exists_raises(self, cert_files):
        """CA 文件不存在时应抛出 FileNotFoundError（严格安全策略）。"""
        with pytest.raises(FileNotFoundError, match="CA 证书文件不存在"):
            create_client_ssl_context(ca_cert_path="/nonexistent/ca.pem")

    def test_mtls(self, cert_files):
        """mTLS 双向认证。"""
        cert_path, key_path = cert_files
        ctx = create_client_ssl_context(
            ca_cert_path=cert_path,
            cert_path=cert_path,
            key_path=key_path,
        )
        assert isinstance(ctx, ssl.SSLContext)

    def test_tls_version(self):
        """最低 TLS 版本应为 1.2。"""
        ctx = create_client_ssl_context()
        assert ctx.minimum_version == ssl.TLSVersion.TLSv1_2

    def test_only_cert_no_key(self, cert_files):
        """只提供证书不提供私钥。"""
        cert_path, _ = cert_files
        # 不提供 key 时不会加载证书链
        ctx = create_client_ssl_context(cert_path=cert_path)
        assert isinstance(ctx, ssl.SSLContext)
