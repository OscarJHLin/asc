"""测试 MODEL_CHUNK / MODEL_CHUNK_ACK 二进制帧协议。

覆盖：
- MODEL_CHUNK 帧负载编码/解码
- MODEL_CHUNK_ACK 帧负载编码/解码
- 分片逻辑（4MB 分块）
- 端到端模型分发（Mock TCP）
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from asc.network.frame import (
    Frame,
    FrameType,
    decode_frame,
    encode_frame,
)

# ------------------------------------------------------------------
# 负载编解码辅助函数（与 model_distributor.py 中的实现一致）
# ------------------------------------------------------------------

def encode_model_chunk_payload(
    model_id: str,
    chunk_index: int,
    total_chunks: int,
    chunk_data: bytes,
) -> bytes:
    """编码 MODEL_CHUNK 帧负载。

    格式: [model_id_len:2B][model_id][chunk_index:4B][total_chunks:4B][chunk_data]
    """
    model_id_bytes = model_id.encode("utf-8")
    return (
        struct.pack("!H", len(model_id_bytes))
        + model_id_bytes
        + struct.pack("!II", chunk_index, total_chunks)
        + chunk_data
    )


def decode_model_chunk_payload(payload: bytes) -> tuple[str, int, int, bytes]:
    """解码 MODEL_CHUNK 帧负载。

    Returns:
        (model_id, chunk_index, total_chunks, chunk_data)
    """
    offset = 0
    model_id_len = struct.unpack_from("!H", payload, offset)[0]
    offset += 2
    model_id = payload[offset : offset + model_id_len].decode("utf-8")
    offset += model_id_len
    chunk_index, total_chunks = struct.unpack_from("!II", payload, offset)
    offset += 8
    chunk_data = payload[offset:]
    return model_id, chunk_index, total_chunks, chunk_data


def encode_model_chunk_ack_payload(
    model_id: str,
    chunk_index: int,
    success: bool,
) -> bytes:
    """编码 MODEL_CHUNK_ACK 帧负载。

    格式: [model_id_len:2B][model_id][chunk_index:4B][success:1B]
    """
    model_id_bytes = model_id.encode("utf-8")
    return (
        struct.pack("!H", len(model_id_bytes))
        + model_id_bytes
        + struct.pack("!I", chunk_index)
        + (b"\x01" if success else b"\x00")
    )


def decode_model_chunk_ack_payload(payload: bytes) -> tuple[str, int, bool]:
    """解码 MODEL_CHUNK_ACK 帧负载。

    Returns:
        (model_id, chunk_index, success)
    """
    offset = 0
    model_id_len = struct.unpack_from("!H", payload, offset)[0]
    offset += 2
    model_id = payload[offset : offset + model_id_len].decode("utf-8")
    offset += model_id_len
    chunk_index = struct.unpack_from("!I", payload, offset)[0]
    offset += 4
    success = payload[offset] == 0x01
    return model_id, chunk_index, success


# ------------------------------------------------------------------
# 测试：MODEL_CHUNK 帧负载编码/解码
# ------------------------------------------------------------------


class TestModelChunkPayload:
    """MODEL_CHUNK 帧负载编解码。"""

    def test_encode_decode_roundtrip(self):
        """编码后解码应还原原始数据。"""
        payload = encode_model_chunk_payload(
            model_id="qwen-7b",
            chunk_index=0,
            total_chunks=3,
            chunk_data=b"\x00\x01\x02\x03",
        )
        model_id, chunk_index, total_chunks, chunk_data = decode_model_chunk_payload(payload)
        assert model_id == "qwen-7b"
        assert chunk_index == 0
        assert total_chunks == 3
        assert chunk_data == b"\x00\x01\x02\x03"

    def test_encode_decode_large_model_id(self):
        """较长的 model_id 编解码。"""
        long_id = "a" * 500
        payload = encode_model_chunk_payload(
            model_id=long_id,
            chunk_index=99,
            total_chunks=100,
            chunk_data=b"data",
        )
        model_id, chunk_index, total_chunks, chunk_data = decode_model_chunk_payload(payload)
        assert model_id == long_id
        assert chunk_index == 99
        assert total_chunks == 100
        assert chunk_data == b"data"

    def test_encode_decode_empty_chunk_data(self):
        """空 chunk_data（最后一个分片可能为空）。"""
        payload = encode_model_chunk_payload(
            model_id="model",
            chunk_index=5,
            total_chunks=6,
            chunk_data=b"",
        )
        model_id, chunk_index, total_chunks, chunk_data = decode_model_chunk_payload(payload)
        assert chunk_data == b""

    def test_payload_format_structure(self):
        """验证负载格式：[model_id_len:2B][model_id][chunk_index:4B][total_chunks:4B][chunk_data]。"""
        model_id = "test"
        chunk_index = 2
        total_chunks = 5
        chunk_data = b"ABC"
        payload = encode_model_chunk_payload(model_id, chunk_index, total_chunks, chunk_data)

        # 手动解析验证格式
        offset = 0
        model_id_len = struct.unpack_from("!H", payload, offset)[0]
        assert model_id_len == 4  # "test" 长度
        offset += 2
        assert payload[offset : offset + 4] == b"test"
        offset += 4
        ci, tc = struct.unpack_from("!II", payload, offset)
        assert ci == 2
        assert tc == 5
        offset += 8
        assert payload[offset:] == b"ABC"

    def test_unicode_model_id(self):
        """Unicode model_id 编解码。"""
        model_id = "模型-7b-v2"
        payload = encode_model_chunk_payload(
            model_id=model_id,
            chunk_index=0,
            total_chunks=1,
            chunk_data=b"x",
        )
        decoded_id, _, _, _ = decode_model_chunk_payload(payload)
        assert decoded_id == model_id


# ------------------------------------------------------------------
# 测试：MODEL_CHUNK_ACK 帧负载编码/解码
# ------------------------------------------------------------------


class TestModelChunkAckPayload:
    """MODEL_CHUNK_ACK 帧负载编解码。"""

    def test_encode_decode_success(self):
        """成功 ACK 编解码。"""
        payload = encode_model_chunk_ack_payload(
            model_id="qwen-7b",
            chunk_index=0,
            success=True,
        )
        model_id, chunk_index, success = decode_model_chunk_ack_payload(payload)
        assert model_id == "qwen-7b"
        assert chunk_index == 0
        assert success is True

    def test_encode_decode_failure(self):
        """失败 ACK 编解码。"""
        payload = encode_model_chunk_ack_payload(
            model_id="qwen-7b",
            chunk_index=3,
            success=False,
        )
        model_id, chunk_index, success = decode_model_chunk_ack_payload(payload)
        assert model_id == "qwen-7b"
        assert chunk_index == 3
        assert success is False

    def test_ack_payload_format(self):
        """验证 ACK 负载格式：[model_id_len:2B][model_id][chunk_index:4B][success:1B]。"""
        payload = encode_model_chunk_ack_payload("ab", 10, True)
        offset = 0
        model_id_len = struct.unpack_from("!H", payload, offset)[0]
        assert model_id_len == 2
        offset += 2
        assert payload[offset : offset + 2] == b"ab"
        offset += 2
        ci = struct.unpack_from("!I", payload, offset)[0]
        assert ci == 10
        offset += 4
        assert payload[offset] == 0x01


# ------------------------------------------------------------------
# 测试：MODEL_CHUNK 帧在 Frame 协议中的集成
# ------------------------------------------------------------------


class TestModelChunkFrame:
    """MODEL_CHUNK 帧与 Frame 协议集成。"""

    def test_model_chunk_frame_encode_decode(self):
        """MODEL_CHUNK 帧通过 Frame 编解码。"""
        chunk_payload = encode_model_chunk_payload(
            model_id="qwen-7b",
            chunk_index=0,
            total_chunks=2,
            chunk_data=b"hello",
        )
        frame = Frame(frame_type=FrameType.MODEL_CHUNK, payload=chunk_payload)
        data = encode_frame(frame)
        decoded = decode_frame(data)

        assert decoded.frame_type == FrameType.MODEL_CHUNK
        model_id, chunk_index, total_chunks, chunk_data = decode_model_chunk_payload(decoded.payload)
        assert model_id == "qwen-7b"
        assert chunk_index == 0
        assert total_chunks == 2
        assert chunk_data == b"hello"

    def test_model_chunk_ack_frame_encode_decode(self):
        """MODEL_CHUNK_ACK 帧通过 Frame 编解码。"""
        ack_payload = encode_model_chunk_ack_payload(
            model_id="qwen-7b",
            chunk_index=0,
            success=True,
        )
        frame = Frame(frame_type=FrameType.MODEL_CHUNK_ACK, payload=ack_payload)
        data = encode_frame(frame)
        decoded = decode_frame(data)

        assert decoded.frame_type == FrameType.MODEL_CHUNK_ACK
        model_id, chunk_index, success = decode_model_chunk_ack_payload(decoded.payload)
        assert model_id == "qwen-7b"
        assert chunk_index == 0
        assert success is True

    def test_model_chunk_frame_type_value(self):
        """MODEL_CHUNK 帧类型值为 0x40。"""
        assert FrameType.MODEL_CHUNK.value == 0x40

    def test_model_chunk_ack_frame_type_value(self):
        """MODEL_CHUNK_ACK 帧类型值为 0x41。"""
        assert FrameType.MODEL_CHUNK_ACK.value == 0x41

    def test_large_chunk_data(self):
        """4MB 分片数据通过 Frame 传输。"""
        chunk_data = b"\xAB" * (4 * 1024 * 1024)
        chunk_payload = encode_model_chunk_payload(
            model_id="big-model",
            chunk_index=0,
            total_chunks=1,
            chunk_data=chunk_data,
        )
        frame = Frame(frame_type=FrameType.MODEL_CHUNK, payload=chunk_payload)
        data = encode_frame(frame)
        decoded = decode_frame(data)

        _, _, _, decoded_data = decode_model_chunk_payload(decoded.payload)
        assert decoded_data == chunk_data
        assert len(decoded_data) == 4 * 1024 * 1024


# ------------------------------------------------------------------
# 测试：分片逻辑
# ------------------------------------------------------------------


class TestModelChunkSplitting:
    """模型文件分片逻辑。"""

    def test_small_file_single_chunk(self, tmp_path: Path):
        """小于 4MB 的文件只有一个分片。"""
        from asc.master.model_distributor import split_model_into_chunks

        file_path = tmp_path / "small.gguf"
        file_path.write_bytes(b"x" * 1024)

        chunks = split_model_into_chunks(file_path)
        assert len(chunks) == 1
        assert chunks[0]["index"] == 0
        assert chunks[0]["total"] == 1
        assert chunks[0]["data"] == b"x" * 1024

    def test_exact_chunk_size(self, tmp_path: Path):
        """恰好 4MB 的文件只有一个分片。"""
        from asc.master.model_distributor import CHUNK_SIZE, split_model_into_chunks

        file_path = tmp_path / "exact.gguf"
        file_path.write_bytes(b"\x00" * CHUNK_SIZE)

        chunks = split_model_into_chunks(file_path)
        assert len(chunks) == 1
        assert chunks[0]["total"] == 1

    def test_multi_chunk_file(self, tmp_path: Path):
        """大于 4MB 的文件分多个分片。"""
        from asc.master.model_distributor import CHUNK_SIZE, split_model_into_chunks

        file_path = tmp_path / "big.gguf"
        # 4MB + 1 字节 = 2 个分片
        data = b"\x00" * (CHUNK_SIZE + 1)
        file_path.write_bytes(data)

        chunks = split_model_into_chunks(file_path)
        assert len(chunks) == 2
        assert chunks[0]["index"] == 0
        assert chunks[0]["total"] == 2
        assert len(chunks[0]["data"]) == CHUNK_SIZE
        assert chunks[1]["index"] == 1
        assert chunks[1]["total"] == 2
        assert len(chunks[1]["data"]) == 1

    def test_chunk_data_integrity(self, tmp_path: Path):
        """分片数据拼接后等于原始文件。"""
        from asc.master.model_distributor import CHUNK_SIZE, split_model_into_chunks

        file_path = tmp_path / "integrity.gguf"
        original = b"\xAB" * (CHUNK_SIZE * 2 + 1234)
        file_path.write_bytes(original)

        chunks = split_model_into_chunks(file_path)
        reassembled = b"".join(c["data"] for c in chunks)
        assert reassembled == original


# ------------------------------------------------------------------
# 测试：端到端模型分发（Mock TCP）
# ------------------------------------------------------------------


class TestModelDistributionE2E:
    """端到端模型分发测试（使用 Mock TCP）。"""

    @pytest.mark.asyncio
    async def test_distribute_to_worker_sends_chunks(self, tmp_path: Path):
        """distribute_to_worker 应通过 Binary Frame 发送所有分片。"""
        from asc.master.model_distributor import (
            CHUNK_SIZE,
            DistributeTask,
            ModelDistributor,
            decode_model_chunk_payload,
            encode_model_chunk_ack_payload,
        )

        # 创建测试模型文件
        model_path = tmp_path / "test.gguf"
        model_data = b"\xAA" * (CHUNK_SIZE + 100)
        model_path.write_bytes(model_data)

        # Mock TCPServer
        mock_tcp_server = MagicMock()
        sent_frames: list[tuple[str, Frame]] = []

        async def mock_send_frame(conn_id: str, frame: Frame) -> bool:
            sent_frames.append((conn_id, frame))
            # 模拟 Worker 立即回复 ACK
            if frame.frame_type == FrameType.MODEL_CHUNK:
                model_id, chunk_index, total_chunks, _ = decode_model_chunk_payload(frame.payload)
                ack_payload = encode_model_chunk_ack_payload(
                    model_id=model_id,
                    chunk_index=chunk_index,
                    success=True,
                )
                ack_frame = Frame(frame_type=FrameType.MODEL_CHUNK_ACK, payload=ack_payload)
                distributor.handle_chunk_ack(ack_frame)
            return True

        mock_tcp_server.send_frame = mock_send_frame

        distributor = ModelDistributor(tcp_server=mock_tcp_server)
        task = DistributeTask(
            model_id="test-model",
            model_path=str(model_path),
            target_node_id="worker-1",
            target_conn_id="conn-1",
        )

        result = await distributor.distribute_to_worker(task)

        # 验证发送了 MODEL_CHUNK 帧
        chunk_frames = [
            (cid, f) for cid, f in sent_frames
            if f.frame_type == FrameType.MODEL_CHUNK
        ]
        assert len(chunk_frames) == 2  # 4MB + 100B = 2 个分片

        # 验证帧内容
        for cid, frame in chunk_frames:
            assert cid == "conn-1"
            model_id, chunk_index, total_chunks, chunk_data = decode_model_chunk_payload(
                frame.payload
            )
            assert model_id == "test-model"
            assert total_chunks == 2

    @pytest.mark.asyncio
    async def test_worker_handles_model_chunk(self, tmp_path: Path):
        """Worker 应正确处理 MODEL_CHUNK 帧，写入文件并回复 ACK。"""
        from asc.worker.agent import WorkerAgent

        models_dir = tmp_path / "models"
        models_dir.mkdir()

        agent = WorkerAgent(node_id="worker-1")
        agent._models_dir = models_dir

        # Mock client
        sent_frames: list[Frame] = []
        mock_client = MagicMock()

        async def mock_send_frame(frame: Frame) -> bool:
            sent_frames.append(frame)
            return True

        mock_client.send_frame = mock_send_frame
        agent._client = mock_client

        # 构造 MODEL_CHUNK 帧
        chunk_data = b"hello world model data"
        chunk_payload = encode_model_chunk_payload(
            model_id="test-model",
            chunk_index=0,
            total_chunks=1,
            chunk_data=chunk_data,
        )
        frame = Frame(frame_type=FrameType.MODEL_CHUNK, payload=chunk_payload)

        # 处理帧
        await agent._handle_model_chunk_frame(frame)

        # 验证 ACK 已发送
        assert len(sent_frames) == 1
        ack_frame = sent_frames[0]
        assert ack_frame.frame_type == FrameType.MODEL_CHUNK_ACK
        ack_model_id, ack_chunk_index, ack_success = decode_model_chunk_ack_payload(
            ack_frame.payload
        )
        assert ack_model_id == "test-model"
        assert ack_chunk_index == 0
        assert ack_success is True

    @pytest.mark.asyncio
    async def test_worker_reassembles_chunks(self, tmp_path: Path):
        """Worker 收到所有分片后应能正确重组文件。"""
        from asc.master.model_distributor import CHUNK_SIZE
        from asc.worker.agent import WorkerAgent

        models_dir = tmp_path / "models"
        models_dir.mkdir()

        agent = WorkerAgent(node_id="worker-1")
        agent._models_dir = models_dir

        # Mock client
        sent_frames: list[Frame] = []
        mock_client = MagicMock()

        async def mock_send_frame(frame: Frame) -> bool:
            sent_frames.append(frame)
            return True

        mock_client.send_frame = mock_send_frame
        agent._client = mock_client

        # 模拟 3 个分片
        original_data = b"\xBB" * (CHUNK_SIZE * 2 + 500)
        total_chunks = 3
        chunk_sizes = [CHUNK_SIZE, CHUNK_SIZE, 500]
        offset = 0

        for i in range(total_chunks):
            chunk_data = original_data[offset : offset + chunk_sizes[i]]
            chunk_payload = encode_model_chunk_payload(
                model_id="big-model",
                chunk_index=i,
                total_chunks=total_chunks,
                chunk_data=chunk_data,
            )
            frame = Frame(frame_type=FrameType.MODEL_CHUNK, payload=chunk_payload)
            await agent._handle_model_chunk_frame(frame)
            offset += chunk_sizes[i]

        # 验证文件已正确写入
        saved_file = models_dir / "big-model.gguf"
        assert saved_file.exists()
        assert saved_file.read_bytes() == original_data

    @pytest.mark.asyncio
    async def test_distribute_model_in_master(self, tmp_path: Path):
        """MasterNode.distribute_model 应使用 TCPServer 发送 MODEL_CHUNK 帧。"""
        from dataclasses import replace

        from asc.master.main import MasterNode
        from asc.master.model_distributor import (
            decode_model_chunk_payload,
            encode_model_chunk_ack_payload,
        )
        from asc.types.state import NodeId, NodeInfo

        # 创建测试模型文件
        model_path = tmp_path / "test.gguf"
        model_path.write_bytes(b"\xCC" * 1024)

        master = MasterNode(node_id="master-1")

        # Mock TCP server
        sent_frames: list[tuple[str, Frame]] = []

        async def mock_send_frame(conn_id: str, frame: Frame) -> bool:
            sent_frames.append((conn_id, frame))
            # 模拟 Worker 回复 ACK
            if frame.frame_type == FrameType.MODEL_CHUNK:
                model_id, chunk_index, total_chunks, _ = decode_model_chunk_payload(frame.payload)
                ack_payload = encode_model_chunk_ack_payload(
                    model_id=model_id,
                    chunk_index=chunk_index,
                    success=True,
                )
                ack_frame = Frame(frame_type=FrameType.MODEL_CHUNK_ACK, payload=ack_payload)
                if master._model_distributor is not None:
                    master._model_distributor.handle_chunk_ack(ack_frame)
            return True

        async def mock_send(conn_id: str, envelope: Any) -> bool:
            return True

        mock_tcp = MagicMock()
        mock_tcp.send_frame = mock_send_frame
        mock_tcp.send = mock_send
        master._tcp_server = mock_tcp

        # 添加一个模拟节点
        try:
            import immutables
            nodes_map = immutables.Map({
                NodeId("worker-1"): NodeInfo(node_id=NodeId("worker-1"), ip="10.0.0.2", port=52414, runner_status="online"),
            })
        except ImportError:
            nodes_map = {NodeId("worker-1"): NodeInfo(node_id=NodeId("worker-1"), ip="10.0.0.2", port=52414, runner_status="online")}

        master._state = replace(master._state, nodes=nodes_map)
        master._conn_node_map["conn-abc"] = "worker-1"
        master._node_resources["worker-1"] = {
            "gpus": [{"vram_free_mb": 8192}],
        }

        # 执行分发
        assignments = await master.distribute_model(
            model_id="test-model",
            model_path=str(model_path),
            total_layers=32,
        )

        # 验证发送了 MODEL_CHUNK 帧（不再使用 HTTP）
        chunk_frames = [
            (cid, f) for cid, f in sent_frames
            if f.frame_type == FrameType.MODEL_CHUNK
        ]
        assert len(chunk_frames) >= 1  # 至少一个分片

    @pytest.mark.asyncio
    async def test_no_http_used_in_distribution(self, tmp_path: Path):
        """验证模型分发不再使用 httpx。"""
        from asc.master.model_distributor import (
            DistributeTask,
            ModelDistributor,
            decode_model_chunk_payload,
            encode_model_chunk_ack_payload,
        )

        model_path = tmp_path / "test.gguf"
        model_path.write_bytes(b"test data")

        mock_tcp = MagicMock()

        async def mock_send_frame(conn_id: str, frame: Frame) -> bool:
            # 模拟 ACK
            if frame.frame_type == FrameType.MODEL_CHUNK:
                model_id, chunk_index, total_chunks, _ = decode_model_chunk_payload(frame.payload)
                ack_payload = encode_model_chunk_ack_payload(
                    model_id=model_id,
                    chunk_index=chunk_index,
                    success=True,
                )
                ack_frame = Frame(frame_type=FrameType.MODEL_CHUNK_ACK, payload=ack_payload)
                distributor.handle_chunk_ack(ack_frame)
            return True

        mock_tcp.send_frame = mock_send_frame

        distributor = ModelDistributor(tcp_server=mock_tcp)
        task = DistributeTask(
            model_id="test",
            model_path=str(model_path),
            target_node_id="w1",
            target_conn_id="c1",
        )

        # 确保没有 httpx 调用
        with patch("httpx.AsyncClient", side_effect=AssertionError("httpx should not be used")):
            result = await distributor.distribute_to_worker(task)
            assert result.success
