"""测试 LlamaServerBuilder 异步启动（asyncio.create_subprocess_exec）。

验证 builder 的异步路径：
- aload() 使用 asyncio.create_subprocess_exec 替代 subprocess.Popen
- aload() 使用 asyncio.sleep + httpx.AsyncClient 替代 time.sleep + httpx.Client
- abuild() 返回 LlamaServerEngine（兼容 asyncio.subprocess.Process）
- LlamaServerEngine.close() 和 status() 兼容异步进程
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from asc.engine.base import EngineStatus
from asc.engine.llama_server import LlamaServerBuilder, LlamaServerEngine


def _make_async_mock_process(returncode=None):
    """创建模拟 asyncio.subprocess.Process 的 mock。

    terminate/kill 是同步方法，不能返回协程，因此用 MagicMock 而非 AsyncMock。
    """
    mock_process = MagicMock()
    mock_process.returncode = returncode
    return mock_process


# --- aload 异步启动测试 ---


class TestLlamaServerBuilderAsyncLoad:
    """aload() 异步加载测试。"""

    async def test_aload_uses_create_subprocess_exec(self):
        """aload 应使用 asyncio.create_subprocess_exec 而非 subprocess.Popen。"""
        builder = LlamaServerBuilder(model_path="/m.gguf", port=8081)

        mock_process = _make_async_mock_process(returncode=None)

        with (
            patch.object(builder, "_find_executable", return_value="/bin/llama-server"),
            patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_create,
            patch.object(builder, "_wait_for_ready_async"),
        ):
            async for _ in builder.aload():
                pass
            mock_create.assert_called_once()
            # 验证传入了正确的命令参数
            call_args = mock_create.call_args
            cmd = call_args[0]
            assert cmd[0] == "/bin/llama-server"
            assert "-m" in cmd
            assert "/m.gguf" in cmd

    async def test_aload_yields_progress(self):
        """aload 应 yield 加载进度。"""
        builder = LlamaServerBuilder(model_path="/m.gguf")

        mock_process = _make_async_mock_process(returncode=None)

        with (
            patch.object(builder, "_find_executable", return_value="/bin/llama-server"),
            patch("asyncio.create_subprocess_exec", return_value=mock_process),
            patch.object(builder, "_wait_for_ready_async"),
        ):
            progress = []
            async for p in builder.aload():
                progress.append(p)
            assert len(progress) == 3
            assert progress[0].current == 0
            assert progress[1].current == 1
            assert progress[2].current == 3

    async def test_aload_executable_not_found(self):
        """找不到可执行文件时应抛出 FileNotFoundError。"""
        builder = LlamaServerBuilder(model_path="/m.gguf")

        with (
            patch.object(builder, "_find_executable", return_value=None),
            pytest.raises(FileNotFoundError),
        ):
            async for _ in builder.aload():
                pass

    async def test_aload_with_rpc_and_tensor_split(self):
        """aload 应正确传递 rpc 和 tensor-split 参数。"""
        builder = LlamaServerBuilder(
            model_path="/m.gguf",
            rpc_servers=["192.168.1.2:50052"],
            tensor_split=[0.6, 0.4],
        )

        mock_process = _make_async_mock_process(returncode=None)

        with (
            patch.object(builder, "_find_executable", return_value="/bin/llama-server"),
            patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_create,
            patch.object(builder, "_wait_for_ready_async"),
        ):
            async for _ in builder.aload():
                pass
            cmd = mock_create.call_args[0]
            assert "--rpc" in cmd
            assert "192.168.1.2:50052" in cmd
            assert "--tensor-split" in cmd
            assert "0.6,0.4" in cmd


# --- _wait_for_ready_async 测试 ---


class TestLlamaServerBuilderWaitForReadyAsync:
    """_wait_for_ready_async 异步健康检查测试。"""

    async def test_wait_ready_async_success(self):
        """异步健康检查成功。"""
        builder = LlamaServerBuilder(model_path="/m.gguf", port=8081)
        builder._async_process = _make_async_mock_process(returncode=None)

        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch("httpx.AsyncClient.get", return_value=mock_resp):
            await builder._wait_for_ready_async(timeout=1.0)

    async def test_wait_ready_async_timeout(self):
        """异步健康检查超时。"""
        builder = LlamaServerBuilder(model_path="/m.gguf", port=8081)
        builder._async_process = _make_async_mock_process(returncode=None)

        with (
            patch("httpx.AsyncClient.get", side_effect=httpx.ConnectError("refused")),
            pytest.raises(TimeoutError),
        ):
            await builder._wait_for_ready_async(timeout=0.1)

    async def test_wait_ready_async_process_failed(self):
        """异步健康检查时进程已退出。"""
        builder = LlamaServerBuilder(model_path="/m.gguf", port=8081)
        builder._async_process = _make_async_mock_process(returncode=1)
        builder._async_process.stderr = AsyncMock()
        builder._async_process.stderr.read.return_value = b"error msg"

        with (
            patch("httpx.AsyncClient.get", side_effect=httpx.ConnectError("refused")),
            pytest.raises(RuntimeError, match="启动失败"),
        ):
            await builder._wait_for_ready_async(timeout=0.1)


# --- abuild 测试 ---


class TestLlamaServerBuilderAsyncBuild:
    """abuild() 异步构建测试。"""

    async def test_abuild_returns_engine(self):
        """abuild 应返回 LlamaServerEngine。"""
        builder = LlamaServerBuilder(model_path="/m.gguf", port=8081)
        mock_process = _make_async_mock_process(returncode=None)
        builder._async_process = mock_process

        engine = await builder.abuild()
        assert isinstance(engine, LlamaServerEngine)
        assert engine.model_path == "/m.gguf"
        assert engine.base_url == "http://127.0.0.1:8081"
        assert engine._is_async_process is True

    async def test_abuild_without_aload_raises(self):
        """未调用 aload 时 abuild 应抛出异常。"""
        builder = LlamaServerBuilder(model_path="/m.gguf")
        with pytest.raises(RuntimeError, match="必须先调用 aload"):
            await builder.abuild()


# --- LlamaServerEngine 兼容异步进程测试 ---


class TestLlamaServerEngineAsyncProcess:
    """LlamaServerEngine 兼容 asyncio.subprocess.Process 测试。"""

    def test_status_detects_crash_async_process(self):
        """异步进程崩溃时 status 应返回 ERROR。"""
        mock_process = _make_async_mock_process(returncode=1)

        engine = LlamaServerEngine(
            model_path="/m.gguf",
            base_url="http://127.0.0.1:8081",
            process=mock_process,
            http_client=MagicMock(),
        )
        engine._is_async_process = True
        engine._set_status(EngineStatus.READY)
        assert engine.status() == EngineStatus.ERROR

    def test_status_ready_async_process(self):
        """异步进程正常运行时 status 应返回 READY。"""
        mock_process = _make_async_mock_process(returncode=None)

        engine = LlamaServerEngine(
            model_path="/m.gguf",
            base_url="http://127.0.0.1:8081",
            process=mock_process,
            http_client=MagicMock(),
        )
        engine._is_async_process = True
        engine._set_status(EngineStatus.READY)
        assert engine.status() == EngineStatus.READY

    def test_close_terminates_async_process(self):
        """close 应终止异步进程。"""
        mock_process = _make_async_mock_process(returncode=None)
        mock_client = MagicMock()

        engine = LlamaServerEngine(
            model_path="/m.gguf",
            base_url="http://127.0.0.1:8081",
            process=mock_process,
            http_client=mock_client,
        )
        engine._is_async_process = True
        engine._set_status(EngineStatus.READY)
        engine.close()
        mock_client.close.assert_called_once()
        mock_process.terminate.assert_called_once()
        assert engine.status() == EngineStatus.SHUTDOWN

    def test_close_already_dead_async_process(self):
        """异步进程已退出时 close 不应调用 terminate。"""
        mock_process = _make_async_mock_process(returncode=0)
        mock_client = MagicMock()

        engine = LlamaServerEngine(
            model_path="/m.gguf",
            base_url="http://127.0.0.1:8081",
            process=mock_process,
            http_client=mock_client,
        )
        engine._is_async_process = True
        engine.close()
        mock_process.terminate.assert_not_called()
        assert engine.status() == EngineStatus.SHUTDOWN

    def test_close_kills_async_process_on_timeout(self):
        """异步进程 terminate 后不退出时应 kill。"""
        mock_process = _make_async_mock_process(returncode=None)
        mock_client = MagicMock()

        engine = LlamaServerEngine(
            model_path="/m.gguf",
            base_url="http://127.0.0.1:8081",
            process=mock_process,
            http_client=mock_client,
        )
        engine._is_async_process = True
        engine._set_status(EngineStatus.READY)
        engine.close()
        # 异步进程：terminate 后直接 kill（无法同步 wait）
        mock_process.terminate.assert_called_once()
        mock_process.kill.assert_called_once()

    def test_sync_process_backward_compatible(self):
        """同步进程（subprocess.Popen）仍正常工作。"""
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_client = MagicMock()

        engine = LlamaServerEngine(
            model_path="/m.gguf",
            base_url="http://127.0.0.1:8081",
            process=mock_process,
            http_client=mock_client,
        )
        engine._set_status(EngineStatus.READY)
        assert engine.status() == EngineStatus.READY

        engine.close()
        mock_process.terminate.assert_called_once()
        mock_process.wait.assert_called_once_with(timeout=5)
