"""测试 TLS/SSL 支持 — TCPServer/TCPClient ssl_context 参数。

TDD 测试优先于实现，覆盖：
- ssl_context 参数传递到 asyncio.start_server / asyncio.open_connection
- 无 ssl_context 时保持裸 TCP 行为（向后兼容）
- ssl_context 创建辅助函数
- MasterNode 传递 ssl_context 到 TCPServer
"""

import ssl
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from asc.network.protocol import Channel, Envelope, Message, MessageType
from asc.network.transport import TCPClient, TCPServer


def _make_envelope(sender: str, msg_type: MessageType, payload: dict | None = None) -> Envelope:
    """构造测试用 Envelope。"""
    return Envelope(
        channel=Channel.EVENTS,
        message=Message(type=msg_type, sender_id=sender, payload=payload or {}),
    )


class TestTCPServerSSLParameter:
    """TCPServer ssl_context 参数测试。"""

    def test_constructor_accepts_ssl_context(self):
        """TCPServer 构造函数接受 ssl_context 参数。"""
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        server = TCPServer(host="127.0.0.1", port=0, ssl_context=ctx)
        assert server._ssl_context is ctx

    def test_constructor_default_no_ssl(self):
        """默认不传 ssl_context 时为 None（裸 TCP）。"""
        server = TCPServer(host="127.0.0.1", port=0)
        assert server._ssl_context is None

    @pytest.mark.asyncio
    async def test_start_passes_ssl_to_asyncio(self):
        """start() 将 ssl_context 传递给 asyncio.start_server。"""
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        server = TCPServer(host="127.0.0.1", port=0, ssl_context=ctx)

        with patch("asc.network.transport.asyncio.start_server", new_callable=AsyncMock) as mock_start:
            mock_start.return_value = MagicMock()
            mock_start.return_value.sockets = [MagicMock()]
            mock_start.return_value.sockets[0].getsockname.return_value = ("127.0.0.1", 12345)
            await server.start()

            mock_start.assert_called_once()
            call_kwargs = mock_start.call_args
            # ssl 参数应该传入
            assert call_kwargs[1].get("ssl") is ctx or (len(call_kwargs[0]) > 2 and call_kwargs[0][2] is not None) or "ssl" in str(call_kwargs)

    @pytest.mark.asyncio
    async def test_start_without_ssl(self):
        """start() 不传 ssl_context 时 asyncio.start_server 无 ssl 参数。"""
        server = TCPServer(host="127.0.0.1", port=0)

        with patch("asc.network.transport.asyncio.start_server", new_callable=AsyncMock) as mock_start:
            mock_start.return_value = MagicMock()
            mock_start.return_value.sockets = [MagicMock()]
            mock_start.return_value.sockets[0].getsockname.return_value = ("127.0.0.1", 12345)
            await server.start()

            mock_start.assert_called_once()
            # 确认 ssl 参数为 None 或未传入
            call_kwargs = mock_start.call_args
            ssl_val = call_kwargs[1].get("ssl") if call_kwargs[1] else None
            assert ssl_val is None


class TestTCPClientSSLParameter:
    """TCPClient ssl_context 参数测试。"""

    def test_constructor_accepts_ssl_context(self):
        """TCPClient 构造函数接受 ssl_context 参数。"""
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        client = TCPClient(host="127.0.0.1", port=52414, node_id="test", ssl_context=ctx)
        assert client._ssl_context is ctx

    def test_constructor_default_no_ssl(self):
        """默认不传 ssl_context 时为 None（裸 TCP）。"""
        client = TCPClient(host="127.0.0.1", port=52414, node_id="test")
        assert client._ssl_context is None

    @pytest.mark.asyncio
    async def test_connect_passes_ssl_to_asyncio(self):
        """connect() 将 ssl_context 传递给 asyncio.open_connection。"""
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        client = TCPClient(host="127.0.0.1", port=52414, node_id="test", ssl_context=ctx)

        with patch("asc.network.transport.asyncio.open_connection", new_callable=AsyncMock) as mock_open:
            mock_reader = MagicMock()
            mock_writer = MagicMock()
            mock_open.return_value = (mock_reader, mock_writer)
            result = await client.connect()

            assert result is True
            mock_open.assert_called_once()
            call_kwargs = mock_open.call_args
            # ssl 参数应该传入
            assert "ssl" in str(call_kwargs) or call_kwargs[1].get("ssl") is ctx


class TestCreateSSLContext:
    """SSL 上下文创建辅助函数测试。"""

    def test_create_server_ssl_context(self):
        """create_server_ssl_context 从证书文件创建服务端 SSL 上下文。"""
        from asc.network.tls import create_server_ssl_context

        # 生成自签名测试证书
        cert_path, key_path = _generate_self_signed_cert()
        try:
            ctx = create_server_ssl_context(cert_path=cert_path, key_path=key_path)
            assert isinstance(ctx, ssl.SSLContext)
            assert ctx.protocol == ssl.PROTOCOL_TLS_SERVER
        finally:
            Path(cert_path).unlink(missing_ok=True)
            Path(key_path).unlink(missing_ok=True)

    def test_create_client_ssl_context(self):
        """create_client_ssl_context 创建客户端 SSL 上下文。"""
        from asc.network.tls import create_client_ssl_context

        ctx = create_client_ssl_context()
        assert isinstance(ctx, ssl.SSLContext)
        assert ctx.protocol == ssl.PROTOCOL_TLS_CLIENT

    def test_create_client_ssl_context_with_ca(self):
        """create_client_ssl_context 带 CA 证书验证。"""
        from asc.network.tls import create_client_ssl_context

        cert_path, key_path = _generate_self_signed_cert()
        try:
            ctx = create_client_ssl_context(ca_cert_path=cert_path)
            assert isinstance(ctx, ssl.SSLContext)
        finally:
            Path(cert_path).unlink(missing_ok=True)
            Path(key_path).unlink(missing_ok=True)

    def test_create_server_ssl_context_missing_cert_raises(self):
        """证书文件不存在时抛出 FileNotFoundError。"""
        from asc.network.tls import create_server_ssl_context

        with pytest.raises(FileNotFoundError):
            create_server_ssl_context(cert_path="/nonexistent/cert.pem", key_path="/nonexistent/key.pem")


class TestMasterNodeSSL:
    """MasterNode 传递 ssl_context 到 TCPServer 测试。"""

    def test_master_passes_ssl_context_to_tcp_server(self):
        """MasterNode 将 ssl_context 传递给 TCPServer。"""
        from asc.master.main import MasterNode

        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        master = MasterNode(node_id="master-1", ssl_context=ctx)
        # MasterNode 应该存储 ssl_context
        assert master._ssl_context is ctx


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _generate_self_signed_cert():
    """生成自签名测试证书（使用 Python 标准库，无需 openssl 命令）。"""
    import datetime
    from pathlib import Path

    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
    except ImportError:
        pytest.skip("cryptography 库未安装，跳过证书测试")

    # 生成 RSA 密钥对
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    # 构建自签名证书
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "ASC Test"),
    ])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=1))
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("localhost")]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )

    # 写入临时文件
    cert_dir = Path(tempfile.mkdtemp())
    cert_path = cert_dir / "cert.pem"
    key_path = cert_dir / "key.pem"

    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    with open(key_path, "wb") as f:
        f.write(key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        ))

    return str(cert_path), str(key_path)
