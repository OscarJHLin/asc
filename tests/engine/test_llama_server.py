"""测试 llama-server 引擎实现。"""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from asc.engine.base import EngineStatus, InferenceRequest
from asc.engine.llama_server import LlamaServerBuilder, LlamaServerEngine


class TestLlamaServerBuilderExecutable:
    """_find_executable 测试。"""

    def test_find_executable_from_env(self):
        builder = LlamaServerBuilder(model_path="/m.gguf")
        expected = str(Path("/custom/llama-server"))
        with (
            patch("os.getenv", return_value="/custom"),
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "resolve", return_value=Path("/custom/llama-server")),
        ):
            result = builder._find_executable()
            assert result == expected

    def test_find_executable_not_found(self):
        builder = LlamaServerBuilder(model_path="/m.gguf")
        with (
            patch("os.getenv", return_value=None),
            patch("shutil.which", return_value=None),
            patch.object(Path, "exists", return_value=False),
        ):
            result = builder._find_executable()
            assert result is None


class TestLlamaServerBuilderStartServer:
    """_start_server 测试。"""

    def test_start_server_basic_cmd(self):
        builder = LlamaServerBuilder(
            model_path="/models/m.gguf",
            host="127.0.0.1",
            port=8081,
            n_gpu_layers=33,
        )
        mock_process = MagicMock()
        with patch("subprocess.Popen", return_value=mock_process) as popen:
            builder._start_server("/bin/llama-server")
            cmd = popen.call_args[0][0]
            assert cmd[0] == "/bin/llama-server"
            assert "-m" in cmd
            assert "/models/m.gguf" in cmd
            assert "--host" in cmd
            assert "127.0.0.1" in cmd
            assert "--port" in cmd
            assert "8081" in cmd
            assert "-ngl" in cmd
            # 默认 gpu_offload_ratio="max" → -ngl 999
            assert "999" in cmd

    def test_start_server_with_rpc_and_tensor_split(self):
        builder = LlamaServerBuilder(
            model_path="/m.gguf",
            rpc_servers=["192.168.1.2:50052", "192.168.1.3:50052"],
            tensor_split=[0.6, 0.4],
        )
        mock_process = MagicMock()
        with patch("subprocess.Popen", return_value=mock_process) as popen:
            builder._start_server("/bin/llama-server")
            cmd = popen.call_args[0][0]
            assert "--rpc" in cmd
            assert "192.168.1.2:50052,192.168.1.3:50052" in cmd
            assert "--tensor-split" in cmd
            assert "0.6,0.4" in cmd


class TestLlamaServerBuilderWaitForReady:
    """_wait_for_ready 测试。"""

    def test_wait_ready_success(self):
        builder = LlamaServerBuilder(model_path="/m.gguf", port=8081)
        builder._process = MagicMock()
        builder._process.poll.return_value = None

        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch("httpx.Client.get", return_value=mock_resp):
            builder._wait_for_ready(timeout=1.0)

    def test_wait_ready_timeout(self):
        builder = LlamaServerBuilder(model_path="/m.gguf", port=8081)
        builder._process = MagicMock()
        builder._process.poll.return_value = None
        builder._process.stderr = None

        with (
            patch("httpx.Client.get", side_effect=httpx.ConnectError("refused")),
            pytest.raises(TimeoutError),
        ):
            builder._wait_for_ready(timeout=0.1)

    def test_wait_ready_process_failed(self):
        builder = LlamaServerBuilder(model_path="/m.gguf", port=8081)
        builder._process = MagicMock()
        builder._process.poll.return_value = 1
        builder._process.stderr.read.return_value = "error msg"

        with (
            patch("httpx.Client.get", side_effect=httpx.ConnectError("refused")),
            pytest.raises(RuntimeError, match="启动失败"),
        ):
            builder._wait_for_ready(timeout=0.1)


class TestLlamaServerBuilderLoad:
    """load() 生成器测试。"""

    def test_load_yields_progress(self):
        builder = LlamaServerBuilder(model_path="/m.gguf")
        with (
            patch.object(builder, "_find_executable", return_value="/bin/llama-server"),
            patch.object(builder, "_start_server"),
            patch.object(builder, "_wait_for_ready"),
        ):
            progress = list(builder.load())
            assert len(progress) == 3
            assert progress[0].current == 0
            assert progress[1].current == 1
            assert progress[2].current == 3

    def test_load_executable_not_found(self):
        builder = LlamaServerBuilder(model_path="/m.gguf")
        with (
            patch.object(builder, "_find_executable", return_value=None),
            pytest.raises(FileNotFoundError),
        ):
            list(builder.load())


class TestLlamaServerBuilderBuild:
    """build() 测试。"""

    def test_build_returns_engine(self):
        builder = LlamaServerBuilder(model_path="/m.gguf", port=8081)
        builder._process = MagicMock()
        engine = builder.build()
        assert isinstance(engine, LlamaServerEngine)
        assert engine.model_path == "/m.gguf"
        assert engine.base_url == "http://127.0.0.1:8081"

    def test_build_without_load_raises(self):
        builder = LlamaServerBuilder(model_path="/m.gguf")
        with pytest.raises(RuntimeError, match="必须先调用 load"):
            builder.build()


class TestLlamaServerEngineStatus:
    """引擎状态测试。"""

    @staticmethod
    def _make_mock_process(running: bool = True):
        p = MagicMock()
        p.poll.return_value = None if running else 1
        return p

    def test_status_ready(self):
        engine = LlamaServerEngine(
            model_path="/m.gguf",
            base_url="http://127.0.0.1:8081",
            process=self._make_mock_process(),
            http_client=MagicMock(),
        )
        engine._set_status(EngineStatus.READY)
        assert engine.status() == EngineStatus.READY

    def test_status_detects_crash(self):
        engine = LlamaServerEngine(
            model_path="/m.gguf",
            base_url="http://127.0.0.1:8081",
            process=self._make_mock_process(running=False),
            http_client=MagicMock(),
        )
        engine._set_status(EngineStatus.READY)
        assert engine.status() == EngineStatus.ERROR


class TestLlamaServerEngineSubmit:
    """submit / _sync_infer 测试。"""

    @staticmethod
    def _make_mock_process():
        p = MagicMock()
        p.poll.return_value = None
        return p

    def test_submit_success(self):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Hello world"}}]
        }
        mock_client.post.return_value = mock_resp
        engine = LlamaServerEngine(
            model_path="/m.gguf",
            base_url="http://127.0.0.1:8081",
            process=self._make_mock_process(),
            http_client=mock_client,
        )
        engine._set_status(EngineStatus.READY)

        result = engine.submit(
            InferenceRequest(prompt="Hi", max_tokens=32, temperature=0.5)
        )
        assert result == "Hello world"
        assert engine.status() == EngineStatus.READY

    def test_submit_http_error(self):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500",
            request=MagicMock(),
            response=MagicMock(),
        )
        mock_client.post.return_value = mock_resp
        engine = LlamaServerEngine(
            model_path="/m.gguf",
            base_url="http://127.0.0.1:8081",
            process=self._make_mock_process(),
            http_client=mock_client,
        )
        engine._set_status(EngineStatus.READY)

        with pytest.raises(RuntimeError, match="推理请求失败"):
            engine.submit(InferenceRequest(prompt="Hi"))
        assert engine.status() == EngineStatus.ERROR

    def test_submit_not_ready_raises(self):
        engine = LlamaServerEngine(
            model_path="/m.gguf",
            base_url="http://127.0.0.1:8081",
            process=self._make_mock_process(),
            http_client=MagicMock(),
        )
        with pytest.raises(RuntimeError, match="无法提交请求"):
            engine.submit(InferenceRequest(prompt="Hi"))


class TestLlamaServerEngineClose:
    """close 测试。"""

    @staticmethod
    def _make_mock_process():
        p = MagicMock()
        p.poll.return_value = None
        return p

    def test_close_terminates_process(self):
        proc = self._make_mock_process()
        mock_client = MagicMock()
        engine = LlamaServerEngine(
            model_path="/m.gguf",
            base_url="http://127.0.0.1:8081",
            process=proc,
            http_client=mock_client,
        )
        engine._set_status(EngineStatus.READY)
        engine.close()
        mock_client.close.assert_called_once()
        proc.terminate.assert_called_once()
        proc.wait.assert_called_once_with(timeout=5)
        assert engine.status() == EngineStatus.SHUTDOWN

    def test_close_kills_on_timeout(self):
        proc = self._make_mock_process()
        proc.wait.side_effect = subprocess.TimeoutExpired(cmd="", timeout=5)
        engine = LlamaServerEngine(
            model_path="/m.gguf",
            base_url="http://127.0.0.1:8081",
            process=proc,
            http_client=MagicMock(),
        )
        engine.close()
        proc.kill.assert_called_once()

    def test_close_already_dead(self):
        proc = MagicMock()
        proc.poll.return_value = 0
        engine = LlamaServerEngine(
            model_path="/m.gguf",
            base_url="http://127.0.0.1:8081",
            process=proc,
            http_client=MagicMock(),
        )
        engine.close()
        proc.terminate.assert_not_called()
        assert engine.status() == EngineStatus.SHUTDOWN


class TestLlamaServerEngineStep:
    """step 测试。"""

    def test_step_returns_empty(self):
        proc = MagicMock()
        engine = LlamaServerEngine(
            model_path="/m.gguf",
            base_url="http://127.0.0.1:8081",
            process=proc,
            http_client=MagicMock(),
        )
        assert engine.step() == []
