"""ASC 项目性能基准测试。

测试核心组件在高负载下的表现，包括：
- 帧编解码吞吐
- 信封序列化/反序列化 (MessagePack vs JSON)
- 模型分片编解码
- 负载均衡器节点选择
- 事件日志序列化
- 批处理吞吐
- TCP 传输吞吐
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest

from asc.core.event_log import _deserialize_event, _serialize_event
from asc.engine.base import InferenceResult
from asc.engine.batch import BatchConfig, BatchProcessor, InferenceRequest
from asc.master.model_distributor import (
    decode_model_chunk_ack_payload,
    decode_model_chunk_payload,
    encode_model_chunk_ack_payload,
    encode_model_chunk_payload,
    split_model_into_chunks,
)
from asc.network.frame import Frame, FrameType, decode_frame, encode_frame
from asc.network.protocol import (
    Channel,
    Envelope,
    Message,
    MessageType,
    decode_envelope,
    encode_envelope,
    encode_envelope_json,
)
from asc.network.transport import TCPClient, TCPServer
from asc.scheduler.load_balancer import LoadBalancer
from asc.types.common import NodeId
from asc.types.events import NodeJoined

# ------------------------------------------------------------------
# 帧编解码性能
# ------------------------------------------------------------------

class TestFrameCodec:
    """测试二进制帧编解码性能。"""

    def test_encode_frame_throughput(self):
        """测量帧编码吞吐量。"""
        payload = b"x" * 1024  # 1KB payload
        frame = Frame(frame_type=FrameType.ENVELOPE, payload=payload)

        iterations = 100_000
        start = time.perf_counter()
        for _ in range(iterations):
            encode_frame(frame)
        elapsed = time.perf_counter() - start

        throughput = iterations / elapsed
        print(f"\n[Frame] encode throughput: {throughput:,.0f} ops/sec ({elapsed:.3f}s for {iterations})")
        assert throughput > 100_000, f"帧编码吞吐量过低: {throughput:,.0f} ops/sec"

    def test_decode_frame_throughput(self):
        """测量帧解码吞吐量。"""
        payload = b"x" * 1024
        frame = Frame(frame_type=FrameType.ENVELOPE, payload=payload)
        data = encode_frame(frame)

        iterations = 100_000
        start = time.perf_counter()
        for _ in range(iterations):
            decode_frame(data)
        elapsed = time.perf_counter() - start

        throughput = iterations / elapsed
        print(f"[Frame] decode throughput: {throughput:,.0f} ops/sec ({elapsed:.3f}s for {iterations})")
        assert throughput > 100_000, f"帧解码吞吐量过低: {throughput:,.0f} ops/sec"

    def test_large_frame_encode_decode(self):
        """测试大帧 (4MB) 编解码性能。"""
        payload = b"x" * (4 * 1024 * 1024)  # 4MB
        frame = Frame(frame_type=FrameType.MODEL_CHUNK, payload=payload)

        iterations = 100
        start = time.perf_counter()
        for _ in range(iterations):
            data = encode_frame(frame)
            decode_frame(data)
        elapsed = time.perf_counter() - start

        throughput = iterations / elapsed
        mb_per_sec = (payload_size := len(payload)) * iterations / elapsed / (1024 * 1024)
        print(f"[Frame] 4MB encode+decode: {throughput:,.0f} ops/sec, {mb_per_sec:.0f} MB/s")
        assert throughput > 50, f"大帧编解码吞吐量过低: {throughput:.0f} ops/sec"


# ------------------------------------------------------------------
# 信封序列化性能
# ------------------------------------------------------------------

class TestEnvelopeSerialization:
    """测试信封序列化性能。"""

    def _make_envelope(self) -> Envelope:
        return Envelope(
            channel=Channel.COMMANDS,
            message=Message(
                type=MessageType.TASK_DISPATCH,
                sender_id="master-1",
                payload={"task_id": "t123", "model_id": "llama-7b", "prompt": "hello world" * 100},
            ),
        )

    def test_msgpack_encode_throughput(self):
        """测量 MessagePack 编码吞吐量。"""
        envelope = self._make_envelope()
        iterations = 50_000
        start = time.perf_counter()
        for _ in range(iterations):
            encode_envelope(envelope)
        elapsed = time.perf_counter() - start

        throughput = iterations / elapsed
        print(f"\n[Envelope] MessagePack encode: {throughput:,.0f} ops/sec")
        assert throughput > 20_000, f"MessagePack 编码吞吐量过低: {throughput:,.0f}"

    def test_msgpack_decode_throughput(self):
        """测量 MessagePack 解码吞吐量。"""
        envelope = self._make_envelope()
        data = encode_envelope(envelope)
        iterations = 50_000
        start = time.perf_counter()
        for _ in range(iterations):
            decode_envelope(data)
        elapsed = time.perf_counter() - start

        throughput = iterations / elapsed
        print(f"[Envelope] MessagePack decode: {throughput:,.0f} ops/sec")
        assert throughput > 20_000, f"MessagePack 解码吞吐量过低: {throughput:,.0f}"

    def test_json_encode_throughput(self):
        """测量 JSON 编码吞吐量。"""
        envelope = self._make_envelope()
        iterations = 10_000
        start = time.perf_counter()
        for _ in range(iterations):
            encode_envelope_json(envelope)
        elapsed = time.perf_counter() - start

        throughput = iterations / elapsed
        print(f"[Envelope] JSON encode: {throughput:,.0f} ops/sec")
        assert throughput > 5_000, f"JSON 编码吞吐量过低: {throughput:,.0f}"

    def test_msgpack_vs_json_size(self):
        """比较 MessagePack 和 JSON 的序列化后体积。"""
        envelope = self._make_envelope()
        mp_size = len(encode_envelope(envelope))
        json_size = len(encode_envelope_json(envelope))
        ratio = json_size / mp_size
        print(f"\n[Envelope] MessagePack: {mp_size}B, JSON: {json_size}B, ratio: {ratio:.2f}x")
        assert ratio > 1.05, f"MessagePack 未显著减小体积: ratio={ratio:.2f}"


# ------------------------------------------------------------------
# 模型分片编解码性能
# ------------------------------------------------------------------

class TestModelChunkCodec:
    """测试模型分片编解码性能。"""

    def test_encode_model_chunk_throughput(self):
        """测量 MODEL_CHUNK 负载编码吞吐量。"""
        chunk_data = b"x" * (4 * 1024 * 1024)  # 4MB
        iterations = 1_000
        start = time.perf_counter()
        for _ in range(iterations):
            encode_model_chunk_payload("model-123", 0, 10, chunk_data)
        elapsed = time.perf_counter() - start

        throughput = iterations / elapsed
        mb_per_sec = len(chunk_data) * iterations / elapsed / (1024 * 1024)
        print(f"\n[ModelChunk] encode: {throughput:,.0f} ops/sec, {mb_per_sec:.0f} MB/s")
        assert throughput > 500, f"MODEL_CHUNK 编码吞吐量过低: {throughput:,.0f}"

    def test_decode_model_chunk_throughput(self):
        """测量 MODEL_CHUNK 负载解码吞吐量。"""
        chunk_data = b"x" * (4 * 1024 * 1024)
        payload = encode_model_chunk_payload("model-123", 0, 10, chunk_data)
        iterations = 1_000
        start = time.perf_counter()
        for _ in range(iterations):
            decode_model_chunk_payload(payload)
        elapsed = time.perf_counter() - start

        throughput = iterations / elapsed
        print(f"[ModelChunk] decode: {throughput:,.0f} ops/sec")
        assert throughput > 500, f"MODEL_CHUNK 解码吞吐量过低: {throughput:,.0f}"

    def test_model_chunk_ack_roundtrip(self):
        """测量 MODEL_CHUNK_ACK 往返性能。"""
        iterations = 100_000
        start = time.perf_counter()
        for _ in range(iterations):
            payload = encode_model_chunk_ack_payload("model-123", 42, True)
            decode_model_chunk_ack_payload(payload)
        elapsed = time.perf_counter() - start

        throughput = iterations / elapsed
        print(f"[ModelChunk] ACK roundtrip: {throughput:,.0f} ops/sec")
        assert throughput > 100_000, f"ACK 往返吞吐量过低: {throughput:,.0f}"


# ------------------------------------------------------------------
# 负载均衡器性能
# ------------------------------------------------------------------

class TestLoadBalancer:
    """测试负载均衡器性能。"""

    def _make_lb(self, node_count: int = 100) -> LoadBalancer:
        lb = LoadBalancer(strategy="weighted")
        for i in range(node_count):
            lb.register_node(f"node-{i}", compute_score=float(i + 1))
        return lb

    def test_select_throughput_round_robin(self):
        """测量轮询策略选择吞吐量。"""
        lb = LoadBalancer(strategy="round_robin")
        for i in range(100):
            lb.register_node(f"node-{i}")

        iterations = 100_000
        start = time.perf_counter()
        for _ in range(iterations):
            lb.select([f"node-{i}" for i in range(100)])
        elapsed = time.perf_counter() - start

        throughput = iterations / elapsed
        print(f"\n[LoadBalancer] round_robin select: {throughput:,.0f} ops/sec")
        assert throughput > 30_000, f"轮询选择吞吐量过低: {throughput:,.0f}"

    def test_select_throughput_weighted(self):
        """测量加权策略选择吞吐量。"""
        lb = self._make_lb(node_count=100)
        node_ids = [f"node-{i}" for i in range(100)]

        iterations = 100_000
        start = time.perf_counter()
        for _ in range(iterations):
            lb.select(node_ids)
        elapsed = time.perf_counter() - start

        throughput = iterations / elapsed
        print(f"[LoadBalancer] weighted select: {throughput:,.0f} ops/sec")
        assert throughput > 30_000, f"加权选择吞吐量过低: {throughput:,.0f}"

    def test_tracked_request_cleanup(self):
        """测量过期请求清理性能。"""
        lb = LoadBalancer(strategy="round_robin")
        for i in range(100):
            lb.register_node(f"node-{i}")

        # 模拟大量请求
        for i in range(10_000):
            lb.start_request(f"node-{i % 100}", request_id=f"req-{i}")

        # 等待所有请求过期
        time.sleep(0.1)

        start = time.perf_counter()
        lb.select([f"node-{i}" for i in range(100)])
        elapsed = time.perf_counter() - start

        print(f"[LoadBalancer] cleanup+select: {elapsed*1000:.2f} ms")
        assert elapsed < 0.1, f"过期请求清理过慢: {elapsed*1000:.2f} ms"


# ------------------------------------------------------------------
# 事件日志性能
# ------------------------------------------------------------------

class TestEventLog:
    """测试事件日志序列化性能。"""

    def test_serialize_event_throughput(self):
        """测量事件序列化吞吐量。"""
        event = NodeJoined(node_id=NodeId("node-1"), ip="10.0.0.1", port=52415)
        iterations = 50_000
        start = time.perf_counter()
        for _ in range(iterations):
            _serialize_event(event)
        elapsed = time.perf_counter() - start

        throughput = iterations / elapsed
        print(f"\n[EventLog] serialize: {throughput:,.0f} ops/sec")
        assert throughput > 20_000, f"事件序列化吞吐量过低: {throughput:,.0f}"

    def test_deserialize_event_throughput(self):
        """测量事件反序列化吞吐量。"""
        event = NodeJoined(node_id=NodeId("node-1"), ip="10.0.0.1", port=52415)
        data = _serialize_event(event)
        iterations = 50_000
        start = time.perf_counter()
        for _ in range(iterations):
            _deserialize_event(data)
        elapsed = time.perf_counter() - start

        throughput = iterations / elapsed
        print(f"[EventLog] deserialize: {throughput:,.0f} ops/sec")
        assert throughput > 20_000, f"事件反序列化吞吐量过低: {throughput:,.0f}"


# ------------------------------------------------------------------
# 批处理性能
# ------------------------------------------------------------------

class TestBatchProcessor:
    """测试请求批处理性能。"""

    def test_submit_and_flush_throughput(self):
        """测量批处理提交和刷新吞吐量。"""
        processor = BatchProcessor(BatchConfig(max_batch_size=8, max_wait_ms=1000.0))

        def dummy_infer(requests: list[Any]) -> list[Any]:
            return [InferenceResult(text="ok", tokens_generated=1, tokens_per_second=100.0) for _ in requests]

        iterations = 10_000
        start = time.perf_counter()
        for i in range(iterations):
            processor.submit(InferenceRequest(prompt=f"prompt-{i}"), model_id="m1")
            if processor.should_flush():
                processor.flush(dummy_infer)
        elapsed = time.perf_counter() - start

        throughput = iterations / elapsed
        print(f"\n[Batch] submit+flush: {throughput:,.0f} ops/sec")
        assert throughput > 5_000, f"批处理吞吐过低: {throughput:,.0f}"


# ------------------------------------------------------------------
# TCP 传输吞吐 (async)
# ------------------------------------------------------------------

class TestTCPTransport:
    """测试 TCP 传输层吞吐量。"""

    @pytest.mark.asyncio
    async def test_tcp_echo_throughput(self):
        """测量 TCP 回显吞吐量。"""
        received = []

        async def on_message(conn_id: str, envelope: Envelope) -> None:
            received.append(envelope)

        server = TCPServer(host="127.0.0.1", port=0, on_message=on_message)
        await server.start()

        client = TCPClient(
            host="127.0.0.1",
            port=server.port,
            node_id="test-client",
        )
        connected = await client.connect()
        assert connected

        envelope = Envelope(
            channel=Channel.HEARTBEATS,
            message=Message(
                type=MessageType.HEARTBEAT,
                sender_id="test-client",
                payload={"status": "ok"},
            ),
        )

        iterations = 1_000
        start = time.perf_counter()
        for _ in range(iterations):
            await client.send(envelope)
        elapsed = time.perf_counter() - start

        # 等待接收完成
        for _ in range(50):
            if len(received) >= iterations:
                break
            await asyncio.sleep(0.01)

        throughput = iterations / elapsed
        print(f"\n[TCP] echo throughput: {throughput:,.0f} msgs/sec")

        await client.disconnect()
        await server.stop()

        assert len(received) >= iterations * 0.95, f"消息丢失严重: {len(received)}/{iterations}"
        assert throughput > 500, f"TCP 吞吐量过低: {throughput:,.0f} msgs/sec"

    @pytest.mark.asyncio
    async def test_tcp_broadcast_throughput(self):
        """测量 TCP 广播吞吐量。"""
        client_count = 10
        clients_received = [0] * client_count

        async def on_message(conn_id: str, envelope: Envelope) -> None:
            # 简单计数，实际测试不需要精确映射
            pass

        server = TCPServer(host="127.0.0.1", port=0, on_message=on_message)
        await server.start()

        clients = []
        for i in range(client_count):
            c = TCPClient(host="127.0.0.1", port=server.port, node_id=f"client-{i}")
            await c.connect()
            clients.append(c)

        envelope = Envelope(
            channel=Channel.COMMANDS,
            message=Message(
                type=MessageType.CONFIG_UPDATE,
                sender_id="master",
                payload={"config": {"key": "value"}},
            ),
        )

        iterations = 100
        start = time.perf_counter()
        for _ in range(iterations):
            await server.broadcast(envelope)
        elapsed = time.perf_counter() - start

        throughput = (iterations * client_count) / elapsed
        print(f"[TCP] broadcast throughput: {throughput:,.0f} msgs/sec ({iterations} rounds x {client_count} clients)")

        for c in clients:
            await c.disconnect()
        await server.stop()

        assert throughput > 1_000, f"广播吞吐量过低: {throughput:,.0f} msgs/sec"


# ------------------------------------------------------------------
# 文件分片性能
# ------------------------------------------------------------------

class TestFileChunking:
    """测试模型文件分片性能。"""

    def test_split_small_file(self, tmp_path: Path):
        """测量小文件 (64MB) 分片性能。"""
        file_path = tmp_path / "model.gguf"
        size = 64 * 1024 * 1024
        file_path.write_bytes(b"x" * size)

        start = time.perf_counter()
        chunks = split_model_into_chunks(file_path)
        elapsed = time.perf_counter() - start

        throughput = size / elapsed / (1024 * 1024)
        print(f"\n[Chunking] 64MB split: {throughput:.0f} MB/s, {len(chunks)} chunks")
        assert throughput > 500, f"文件分片吞吐过低: {throughput:.0f} MB/s"

    def test_split_large_file(self, tmp_path: Path):
        """测量大文件 (1GB) 分片性能。"""
        file_path = tmp_path / "model.gguf"
        size = 1024 * 1024 * 1024
        file_path.write_bytes(b"x" * size)

        start = time.perf_counter()
        chunks = split_model_into_chunks(file_path)
        elapsed = time.perf_counter() - start

        throughput = size / elapsed / (1024 * 1024)
        print(f"[Chunking] 1GB split: {throughput:.0f} MB/s, {len(chunks)} chunks")
        assert throughput > 500, f"大文件分片吞吐过低: {throughput:.0f} MB/s"
