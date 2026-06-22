"""跨协议层集成测试。

测试完整消息流生命周期，覆盖所有协议层的端到端集成：
- 消息定义 -> Envelope -> Frame -> 字节流 -> Frame -> Envelope -> 消息还原
- TCP 传输层多消息序列
- MessageRouter + Transport 全链路集成
"""

from __future__ import annotations

import asyncio
import json

import pytest
import pytest_asyncio

from asc.network.capacity import (
    CapacityMetrics,
    CapacityQuery,
    CapacityReport,
    CapacityResponse,
    GpuMetrics,
)
from asc.network.discovery import DiscoveryMessage
from asc.network.frame import (
    Frame,
    FrameType,
    decode_envelope_frame,
    decode_frame,
    encode_envelope_frame,
    encode_frame,
    write_frame_to_bytes,
)
from asc.network.protocol import (
    Channel,
    Envelope,
    Message,
    MessageType,
    decode_envelope,
    encode_envelope,
)
from asc.network.rebalance import (
    RebalanceAckMessage,
    RebalanceAction,
    RebalanceCompleteMessage,
    RebalancePlan,
    RebalanceRequestMessage,
    RebalanceType,
)
from asc.network.router import MessageRouter
from asc.network.task_dispatch import (
    InferenceTask,
    TaskAcceptMessage,
    TaskCancelMessage,
    TaskDispatchMessage,
    TaskProgressMessage,
    TaskResultMessage,
    TaskStatus,
)
from asc.network.transport import TCPClient, TCPServer

# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _full_stack_roundtrip(
    payload: dict,
    channel: Channel,
    msg_type: MessageType,
    sender_id: str = "node-A",
    target: str | None = None,
) -> Envelope:
    """执行完整的 消息 -> Envelope -> Frame -> bytes -> Frame -> Envelope 往返。"""
    msg = Message(type=msg_type, sender_id=sender_id, payload=payload)
    envelope = Envelope(channel=channel, message=msg, target=target)

    # Envelope -> Frame
    frame = encode_envelope_frame(envelope)
    assert frame.frame_type == FrameType.ENVELOPE

    # Frame -> bytes
    raw = write_frame_to_bytes(frame)
    assert isinstance(raw, bytes) and len(raw) > 0

    # bytes -> Frame
    restored_frame = decode_frame(raw)
    assert restored_frame.frame_type == FrameType.ENVELOPE
    assert restored_frame.payload == frame.payload

    # Frame -> Envelope
    restored_envelope = decode_envelope_frame(restored_frame)
    return restored_envelope


def _assert_envelope_equal(orig: Envelope, restored: Envelope) -> None:
    """断言两个 Envelope 的业务字段一致（忽略 timestamp 浮点差异）。"""
    assert restored.channel == orig.channel
    assert restored.message.type == orig.message.type
    assert restored.message.sender_id == orig.message.sender_id
    assert restored.message.payload == orig.message.payload
    assert restored.target == orig.target


# ===========================================================================
# 1. 任务分派生命周期
# ===========================================================================


class TestTaskDispatchLifecycle:
    """任务分派完整生命周期：InferenceTask -> 各种消息 -> 全栈往返。"""

    def _make_task(self) -> InferenceTask:
        return InferenceTask(
            task_id="task-001",
            model_id="llama-7b",
            prompt="你好世界",
            max_tokens=512,
            temperature=0.7,
            priority=3,
            stream=True,
        )

    def test_task_dispatch_full_stack(self) -> None:
        """TaskDispatchMessage 全栈往返：创建 -> 序列化 -> Envelope -> Frame -> bytes -> 还原。"""
        task = self._make_task()
        dispatch = TaskDispatchMessage(
            task=task,
            target_node_id="worker-1",
            instance_id="inst-001",
        )
        payload = dispatch.to_dict()

        # 验证序列化字段
        assert payload["target_node_id"] == "worker-1"
        assert payload["task"]["task_id"] == "task-001"
        assert payload["task"]["stream"] is True

        # 全栈往返
        restored = _full_stack_roundtrip(
            payload=payload,
            channel=Channel.TASK_DISPATCH,
            msg_type=MessageType.TASK_DISPATCH,
            target="worker-1",
        )

        # 从还原的 payload 反序列化
        dispatch_back = TaskDispatchMessage.from_dict(restored.message.payload)
        assert dispatch_back.task.task_id == "task-001"
        assert dispatch_back.task.model_id == "llama-7b"
        assert dispatch_back.task.prompt == "你好世界"
        assert dispatch_back.task.max_tokens == 512
        assert dispatch_back.task.temperature == 0.7
        assert dispatch_back.task.priority == 3
        assert dispatch_back.task.stream is True
        assert dispatch_back.target_node_id == "worker-1"
        assert dispatch_back.instance_id == "inst-001"
        assert restored.target == "worker-1"

    def test_task_accept_full_stack(self) -> None:
        """TaskAcceptMessage 全栈往返（接受和拒绝两种场景）。"""
        # 接受
        accept = TaskAcceptMessage(
            task_id="task-001",
            node_id="worker-1",
            estimated_latency_ms=120.5,
            accepted=True,
        )
        restored = _full_stack_roundtrip(
            payload=accept.to_dict(),
            channel=Channel.TASK_DISPATCH,
            msg_type=MessageType.TASK_ACCEPT,
        )
        accept_back = TaskAcceptMessage.from_dict(restored.message.payload)
        assert accept_back.task_id == "task-001"
        assert accept_back.node_id == "worker-1"
        assert accept_back.estimated_latency_ms == 120.5
        assert accept_back.accepted is True

        # 拒绝
        reject = TaskAcceptMessage(
            task_id="task-002",
            node_id="worker-2",
            estimated_latency_ms=0.0,
            accepted=False,
        )
        restored2 = _full_stack_roundtrip(
            payload=reject.to_dict(),
            channel=Channel.TASK_DISPATCH,
            msg_type=MessageType.TASK_ACCEPT,
        )
        reject_back = TaskAcceptMessage.from_dict(restored2.message.payload)
        assert reject_back.accepted is False

    def test_task_progress_full_stack(self) -> None:
        """TaskProgressMessage 全栈往返。"""
        progress = TaskProgressMessage(
            task_id="task-001",
            node_id="worker-1",
            tokens_generated=128,
            tokens_per_second=45.3,
            progress_fraction=0.25,
        )
        restored = _full_stack_roundtrip(
            payload=progress.to_dict(),
            channel=Channel.TASK_DISPATCH,
            msg_type=MessageType.TASK_PROGRESS,
        )
        progress_back = TaskProgressMessage.from_dict(restored.message.payload)
        assert progress_back.task_id == "task-001"
        assert progress_back.tokens_generated == 128
        assert progress_back.tokens_per_second == 45.3
        assert progress_back.progress_fraction == 0.25

    def test_task_result_full_stack(self) -> None:
        """TaskResultMessage 全栈往返（成功和失败两种场景）。"""
        # 成功
        result = TaskResultMessage(
            task_id="task-001",
            node_id="worker-1",
            status=TaskStatus.COMPLETED,
            output_text="生成的文本内容",
            tokens_generated=512,
            tokens_per_second=42.0,
            latency_ms=12190.5,
        )
        restored = _full_stack_roundtrip(
            payload=result.to_dict(),
            channel=Channel.TASK_DISPATCH,
            msg_type=MessageType.TASK_RESULT,
        )
        result_back = TaskResultMessage.from_dict(restored.message.payload)
        assert result_back.task_id == "task-001"
        assert result_back.status == TaskStatus.COMPLETED
        assert result_back.output_text == "生成的文本内容"
        assert result_back.error == ""

        # 失败
        failed = TaskResultMessage(
            task_id="task-002",
            node_id="worker-1",
            status=TaskStatus.FAILED,
            output_text="",
            tokens_generated=0,
            tokens_per_second=0.0,
            latency_ms=500.0,
            error="OOM: 显存不足",
        )
        restored2 = _full_stack_roundtrip(
            payload=failed.to_dict(),
            channel=Channel.TASK_DISPATCH,
            msg_type=MessageType.TASK_RESULT,
        )
        failed_back = TaskResultMessage.from_dict(restored2.message.payload)
        assert failed_back.status == TaskStatus.FAILED
        assert failed_back.error == "OOM: 显存不足"

    def test_task_cancel_full_stack(self) -> None:
        """TaskCancelMessage 全栈往返。"""
        cancel = TaskCancelMessage(
            task_id="task-001",
            reason="用户取消请求",
        )
        restored = _full_stack_roundtrip(
            payload=cancel.to_dict(),
            channel=Channel.TASK_DISPATCH,
            msg_type=MessageType.CANCEL_TASK,
        )
        cancel_back = TaskCancelMessage.from_dict(restored.message.payload)
        assert cancel_back.task_id == "task-001"
        assert cancel_back.reason == "用户取消请求"


# ===========================================================================
# 2. 容量报告生命周期
# ===========================================================================


class TestCapacityLifecycle:
    """容量协议完整生命周期：CapacityMetrics -> 各种消息 -> 全栈往返。"""

    def _make_gpu(self) -> GpuMetrics:
        return GpuMetrics(
            index=0,
            name="NVIDIA RTX 4090",
            vram_total_mb=24576,
            vram_free_mb=16384,
            utilization_percent=33.5,
            temperature_c=62,
        )

    def _make_metrics(self) -> CapacityMetrics:
        return CapacityMetrics(
            cpu_count=16,
            cpu_percent=45.2,
            memory_total_mb=65536,
            memory_free_mb=32768,
            gpus=[self._make_gpu()],
            active_tasks=3,
            max_concurrent_tasks=8,
            compute_score=92.5,
            network_latency_ms=5.3,
        )

    def test_capacity_report_full_stack(self) -> None:
        """CapacityReport 全栈往返：含 GpuMetrics 的嵌套结构。"""
        metrics = self._make_metrics()
        report = CapacityReport(node_id="worker-1", metrics=metrics)
        payload = report.to_dict()

        # 验证嵌套序列化
        assert len(payload["metrics"]["gpus"]) == 1
        assert payload["metrics"]["gpus"][0]["name"] == "NVIDIA RTX 4090"

        restored = _full_stack_roundtrip(
            payload=payload,
            channel=Channel.CAPACITY,
            msg_type=MessageType.CAPACITY_REPORT,
        )

        report_back = CapacityReport.from_dict(restored.message.payload)
        assert report_back.node_id == "worker-1"
        assert report_back.metrics.cpu_count == 16
        assert report_back.metrics.cpu_percent == 45.2
        assert report_back.metrics.available_slots == 5
        assert report_back.metrics.total_vram_free_mb == 16384
        assert len(report_back.metrics.gpus) == 1
        assert report_back.metrics.gpus[0].name == "NVIDIA RTX 4090"
        assert report_back.metrics.gpus[0].vram_free_mb == 16384
        assert report_back.metrics.gpus[0].temperature_c == 62

    def test_capacity_query_full_stack(self) -> None:
        """CapacityQuery 全栈往返。"""
        query = CapacityQuery(requester_id="master-1")
        restored = _full_stack_roundtrip(
            payload=query.to_dict(),
            channel=Channel.CAPACITY,
            msg_type=MessageType.CAPACITY_QUERY,
        )
        query_back = CapacityQuery.from_dict(restored.message.payload)
        assert query_back.requester_id == "master-1"

    def test_capacity_response_full_stack(self) -> None:
        """CapacityResponse 全栈往返。"""
        metrics = self._make_metrics()
        response = CapacityResponse(node_id="worker-1", metrics=metrics)
        restored = _full_stack_roundtrip(
            payload=response.to_dict(),
            channel=Channel.CAPACITY,
            msg_type=MessageType.CAPACITY_RESPONSE,
        )
        response_back = CapacityResponse.from_dict(restored.message.payload)
        assert response_back.node_id == "worker-1"
        assert response_back.metrics.compute_score == 92.5
        assert len(response_back.metrics.gpus) == 1

    def test_capacity_report_multiple_gpus(self) -> None:
        """多 GPU 场景的容量报告全栈往返。"""
        gpus = [
            GpuMetrics(index=0, name="RTX 4090", vram_total_mb=24576,
                       vram_free_mb=16384, utilization_percent=33.5, temperature_c=62),
            GpuMetrics(index=1, name="RTX 4090", vram_total_mb=24576,
                       vram_free_mb=8192, utilization_percent=66.7, temperature_c=71),
        ]
        metrics = CapacityMetrics(
            cpu_count=32, cpu_percent=55.0, memory_total_mb=131072,
            memory_free_mb=65536, gpus=gpus, active_tasks=5,
            max_concurrent_tasks=16, compute_score=180.0, network_latency_ms=3.2,
        )
        report = CapacityReport(node_id="worker-gpu", metrics=metrics)
        restored = _full_stack_roundtrip(
            payload=report.to_dict(),
            channel=Channel.CAPACITY,
            msg_type=MessageType.CAPACITY_REPORT,
        )
        report_back = CapacityReport.from_dict(restored.message.payload)
        assert len(report_back.metrics.gpus) == 2
        assert report_back.metrics.total_vram_free_mb == 16384 + 8192
        assert report_back.metrics.gpus[1].utilization_percent == 66.7


# ===========================================================================
# 3. 重平衡生命周期
# ===========================================================================


class TestRebalanceLifecycle:
    """重平衡协议完整生命周期。"""

    def _make_plan(self) -> RebalancePlan:
        actions = [
            RebalanceAction(
                action_type=RebalanceType.TENSOR_RESPLIT,
                source_node_id="worker-1",
                target_node_id="worker-2",
                task_ids=["task-001", "task-002"],
                params={"split_ratio": [0.4, 0.6]},
            ),
            RebalanceAction(
                action_type=RebalanceType.TASK_MIGRATION,
                source_node_id="worker-3",
                target_node_id="worker-2",
                task_ids=["task-003"],
                params={"reason": "负载均衡"},
            ),
        ]
        return RebalancePlan(
            plan_id="plan-001",
            actions=actions,
            reason="worker-3 负载过高",
        )

    def test_rebalance_request_full_stack(self) -> None:
        """RebalanceRequestMessage 全栈往返：含多个 RebalanceAction 的嵌套结构。"""
        plan = self._make_plan()
        request = RebalanceRequestMessage(plan=plan, initiator_id="master-1")
        payload = request.to_dict()

        # 验证嵌套序列化
        assert len(payload["plan"]["actions"]) == 2
        assert payload["plan"]["actions"][0]["action_type"] == "tensor_resplit"

        restored = _full_stack_roundtrip(
            payload=payload,
            channel=Channel.REBALANCE,
            msg_type=MessageType.REBALANCE_REQUEST,
        )

        request_back = RebalanceRequestMessage.from_dict(restored.message.payload)
        assert request_back.initiator_id == "master-1"
        assert request_back.plan.plan_id == "plan-001"
        assert request_back.plan.reason == "worker-3 负载过高"
        assert len(request_back.plan.actions) == 2

        a0 = request_back.plan.actions[0]
        assert a0.action_type == RebalanceType.TENSOR_RESPLIT
        assert a0.source_node_id == "worker-1"
        assert a0.target_node_id == "worker-2"
        assert a0.task_ids == ["task-001", "task-002"]
        assert a0.params == {"split_ratio": [0.4, 0.6]}

        a1 = request_back.plan.actions[1]
        assert a1.action_type == RebalanceType.TASK_MIGRATION
        assert a1.params == {"reason": "负载均衡"}

    def test_rebalance_ack_accept_full_stack(self) -> None:
        """RebalanceAckMessage 接受场景全栈往返。"""
        ack = RebalanceAckMessage(
            plan_id="plan-001",
            node_id="worker-1",
            accepted=True,
        )
        restored = _full_stack_roundtrip(
            payload=ack.to_dict(),
            channel=Channel.REBALANCE,
            msg_type=MessageType.REBALANCE_ACK,
        )
        ack_back = RebalanceAckMessage.from_dict(restored.message.payload)
        assert ack_back.plan_id == "plan-001"
        assert ack_back.accepted is True
        assert ack_back.reason == ""

    def test_rebalance_ack_reject_full_stack(self) -> None:
        """RebalanceAckMessage 拒绝场景全栈往返。"""
        ack = RebalanceAckMessage(
            plan_id="plan-001",
            node_id="worker-2",
            accepted=False,
            reason="当前任务不可迁移",
        )
        restored = _full_stack_roundtrip(
            payload=ack.to_dict(),
            channel=Channel.REBALANCE,
            msg_type=MessageType.REBALANCE_ACK,
        )
        ack_back = RebalanceAckMessage.from_dict(restored.message.payload)
        assert ack_back.accepted is False
        assert ack_back.reason == "当前任务不可迁移"

    def test_rebalance_complete_success_full_stack(self) -> None:
        """RebalanceCompleteMessage 成功场景全栈往返。"""
        complete = RebalanceCompleteMessage(
            plan_id="plan-001",
            node_id="worker-1",
            success=True,
        )
        restored = _full_stack_roundtrip(
            payload=complete.to_dict(),
            channel=Channel.REBALANCE,
            msg_type=MessageType.REBALANCE_COMPLETE,
        )
        complete_back = RebalanceCompleteMessage.from_dict(restored.message.payload)
        assert complete_back.success is True
        assert complete_back.error == ""

    def test_rebalance_complete_failure_full_stack(self) -> None:
        """RebalanceCompleteMessage 失败场景全栈往返。"""
        complete = RebalanceCompleteMessage(
            plan_id="plan-001",
            node_id="worker-2",
            success=False,
            error="tensor 重分片失败: 显存不足",
        )
        restored = _full_stack_roundtrip(
            payload=complete.to_dict(),
            channel=Channel.REBALANCE,
            msg_type=MessageType.REBALANCE_COMPLETE,
        )
        complete_back = RebalanceCompleteMessage.from_dict(restored.message.payload)
        assert complete_back.success is False
        assert complete_back.error == "tensor 重分片失败: 显存不足"


# ===========================================================================
# 4. Discovery 消息通过 Frame
# ===========================================================================


class TestDiscoveryThroughFrame:
    """DiscoveryMessage -> JSON -> Envelope -> Frame -> bytes -> 还原。"""

    def test_discovery_message_through_frame(self) -> None:
        """DiscoveryMessage 的 JSON 载荷在完整帧栈中保持不变。"""
        discovery = DiscoveryMessage(
            node_id="node-A",
            ip="192.168.1.100",
            port=9000,
        )

        # DiscoveryMessage -> JSON -> payload
        json_str = discovery.to_json()
        payload = {"discovery_json": json_str}

        # 全栈往返
        restored = _full_stack_roundtrip(
            payload=payload,
            channel=Channel.DISCOVERY,
            msg_type=MessageType.DISCOVER,
        )

        # 从还原的 payload 中提取 JSON 并反序列化
        restored_json = restored.message.payload["discovery_json"]
        discovery_back = DiscoveryMessage.from_json(restored_json)
        assert discovery_back is not None
        assert discovery_back.node_id == "node-A"
        assert discovery_back.ip == "192.168.1.100"
        assert discovery_back.port == 9000

    def test_discovery_message_unicode_preserved(self) -> None:
        """DiscoveryMessage 中含 Unicode 字符时 JSON 载荷完整保留。"""
        discovery = DiscoveryMessage(
            node_id="节点-Alpha",
            ip="10.0.0.1",
            port=8080,
        )
        json_str = discovery.to_json()
        payload = {"discovery_json": json_str}

        restored = _full_stack_roundtrip(
            payload=payload,
            channel=Channel.DISCOVERY,
            msg_type=MessageType.DISCOVER,
        )

        restored_json = restored.message.payload["discovery_json"]
        discovery_back = DiscoveryMessage.from_json(restored_json)
        assert discovery_back is not None
        assert discovery_back.node_id == "节点-Alpha"

    def test_discovery_message_invalid_magic(self) -> None:
        """无效 magic 的 DiscoveryMessage 反序列化返回 None。"""
        bad_json = json.dumps({"magic": "INVALID", "node_id": "x", "ip": "1.2.3.4", "port": 9})
        result = DiscoveryMessage.from_json(bad_json)
        assert result is None


# ===========================================================================
# 5. Model Sync Chunk 通过 Frame
# ===========================================================================


class TestModelSyncChunkThroughFrame:
    """MODEL_CHUNK 和 MODEL_CHUNK_ACK 帧的二进制载荷往返。"""

    def test_model_chunk_binary_roundtrip(self) -> None:
        """MODEL_CHUNK 帧的二进制载荷编码/解码后完整保留。"""
        # 模拟模型分片的二进制数据
        binary_payload = bytes(range(256)) * 4  # 1024 字节
        chunk_metadata = json.dumps({
            "model_id": "llama-7b",
            "chunk_index": 0,
            "offset": 0,
            "size": len(binary_payload),
            "sha256": "abc123",
        }).encode("utf-8")

        # 创建 MODEL_CHUNK 帧
        frame = Frame(frame_type=FrameType.MODEL_CHUNK, payload=chunk_metadata + binary_payload)
        assert frame.frame_type == FrameType.MODEL_CHUNK

        # Frame -> bytes -> Frame
        raw = write_frame_to_bytes(frame)
        restored_frame = decode_frame(raw)

        assert restored_frame.frame_type == FrameType.MODEL_CHUNK
        assert restored_frame.payload == frame.payload

        # 分离元数据和二进制载荷
        meta_len = len(chunk_metadata)
        restored_meta = json.loads(restored_frame.payload[:meta_len].decode("utf-8"))
        restored_binary = restored_frame.payload[meta_len:]

        assert restored_meta["model_id"] == "llama-7b"
        assert restored_meta["chunk_index"] == 0
        assert restored_binary == binary_payload

    def test_model_chunk_ack_roundtrip(self) -> None:
        """MODEL_CHUNK_ACK 帧的往返。"""
        ack_payload = json.dumps({
            "model_id": "llama-7b",
            "chunk_index": 0,
            "accepted": True,
        }).encode("utf-8")

        frame = Frame(frame_type=FrameType.MODEL_CHUNK_ACK, payload=ack_payload)
        raw = write_frame_to_bytes(frame)
        restored = decode_frame(raw)

        assert restored.frame_type == FrameType.MODEL_CHUNK_ACK
        ack_data = json.loads(restored.payload.decode("utf-8"))
        assert ack_data["model_id"] == "llama-7b"
        assert ack_data["accepted"] is True

    def test_large_binary_payload_roundtrip(self) -> None:
        """较大的二进制载荷（1MB）往返完整保留。"""
        # 1MB 二进制数据
        large_payload = b"\xAB" * (1024 * 1024)
        frame = Frame(frame_type=FrameType.MODEL_CHUNK, payload=large_payload)

        raw = write_frame_to_bytes(frame)
        assert len(raw) == 10 + len(large_payload)  # 头部 10 字节 + 载荷

        restored = decode_frame(raw)
        assert restored.payload == large_payload


# ===========================================================================
# 6. 多消息序列通过 TCP
# ===========================================================================


class TestMultiMessageSequenceOverTCP:
    """通过 TCP 传输发送多种消息类型的序列，验证有序接收。"""

    @pytest_asyncio.fixture
    async def tcp_pair(self):
        """创建 TCP 服务器和客户端对。"""
        received_frames: list[Frame] = []
        received_envelopes: list[Envelope] = []
        event = asyncio.Event()

        async def on_frame(conn_id: str, frame: Frame) -> None:
            received_frames.append(frame)
            if frame.frame_type == FrameType.ENVELOPE:
                try:
                    env = decode_envelope_frame(frame)
                    received_envelopes.append(env)
                except Exception:
                    pass
            if len(received_frames) >= 3:
                event.set()

        server = TCPServer(host="127.0.0.1", port=0, on_frame=on_frame)
        await server.start()
        client = TCPClient(host="127.0.0.1", port=server.port, node_id="client-1")
        connected = await client.connect()
        assert connected is True

        # 等待连接建立
        await asyncio.sleep(0.1)

        yield server, client, received_frames, received_envelopes, event

        await client.disconnect()
        await server.stop()

    @pytest.mark.asyncio
    async def test_heartbeat_dispatch_capacity_sequence(self, tcp_pair) -> None:
        """发送心跳、任务分派、容量报告序列，验证全部有序接收。"""
        server, client, received_frames, received_envelopes, event = tcp_pair

        # 消息 1: 心跳
        heartbeat_env = Envelope(
            channel=Channel.HEARTBEATS,
            message=Message(type=MessageType.HEARTBEAT, sender_id="client-1", payload={"ts": 1}),
        )
        # 消息 2: 任务分派
        task = InferenceTask(task_id="t-1", model_id="m-1", prompt="hi",
                             max_tokens=100, temperature=0.5)
        dispatch = TaskDispatchMessage(task=task, target_node_id="w-1", instance_id="i-1")
        dispatch_env = Envelope(
            channel=Channel.TASK_DISPATCH,
            message=Message(type=MessageType.TASK_DISPATCH, sender_id="client-1",
                            payload=dispatch.to_dict()),
        )
        # 消息 3: 容量报告
        metrics = CapacityMetrics(
            cpu_count=8, cpu_percent=30.0, memory_total_mb=32768,
            memory_free_mb=16384, gpus=[], active_tasks=1,
            max_concurrent_tasks=4, compute_score=50.0, network_latency_ms=2.0,
        )
        report = CapacityReport(node_id="w-1", metrics=metrics)
        capacity_env = Envelope(
            channel=Channel.CAPACITY,
            message=Message(type=MessageType.CAPACITY_REPORT, sender_id="client-1",
                            payload=report.to_dict()),
        )

        # 依次发送
        sent = await client.send(heartbeat_env)
        assert sent is True
        sent = await client.send(dispatch_env)
        assert sent is True
        sent = await client.send(capacity_env)
        assert sent is True

        # 等待接收完成
        await asyncio.wait_for(event.wait(), timeout=5.0)

        # 验证接收数量
        assert len(received_envelopes) == 3

        # 验证顺序和类型
        assert received_envelopes[0].message.type == MessageType.HEARTBEAT
        assert received_envelopes[1].message.type == MessageType.TASK_DISPATCH
        assert received_envelopes[2].message.type == MessageType.CAPACITY_REPORT

        # 验证载荷内容
        dispatch_back = TaskDispatchMessage.from_dict(received_envelopes[1].message.payload)
        assert dispatch_back.task.task_id == "t-1"

        report_back = CapacityReport.from_dict(received_envelopes[2].message.payload)
        assert report_back.node_id == "w-1"
        assert report_back.metrics.cpu_count == 8

    @pytest.mark.asyncio
    async def test_mixed_frame_types_over_tcp(self, tcp_pair) -> None:
        """混合发送 Envelope 帧和原始帧（HEARTBEAT、MODEL_CHUNK），验证帧类型保留。"""
        server, client, received_frames, received_envelopes, event = tcp_pair

        # 发送原始 HEARTBEAT 帧
        hb_frame = Frame(frame_type=FrameType.HEARTBEAT, payload=b"\x00")
        sent = await client.send_frame(hb_frame)
        assert sent is True

        # 发送 Envelope 帧
        env = Envelope(
            channel=Channel.HEARTBEATS,
            message=Message(type=MessageType.HEARTBEAT, sender_id="c-1", payload={"seq": 1}),
        )
        sent = await client.send(env)
        assert sent is True

        # 发送 MODEL_CHUNK 帧
        chunk_data = b"\x01\x02\x03\x04"
        chunk_frame = Frame(frame_type=FrameType.MODEL_CHUNK, payload=chunk_data)
        sent = await client.send_frame(chunk_frame)
        assert sent is True

        # 等待接收
        await asyncio.wait_for(event.wait(), timeout=5.0)

        # 验证帧类型
        assert len(received_frames) >= 3
        frame_types = [f.frame_type for f in received_frames[:3]]
        assert frame_types[0] == FrameType.HEARTBEAT
        assert frame_types[1] == FrameType.ENVELOPE
        assert frame_types[2] == FrameType.MODEL_CHUNK

        # 验证二进制载荷
        assert received_frames[0].payload == b"\x00"
        assert received_frames[2].payload == chunk_data


# ===========================================================================
# 7. 完整 Router + Transport 集成
# ===========================================================================


class TestRouterTransportIntegration:
    """MessageRouter + TCP Transport 全链路集成测试。"""

    @pytest_asyncio.fixture
    async def router_pair(self):
        """创建两个通过 TCP 连接的 MessageRouter。"""
        # Router B 的服务端
        received_by_b: list[Envelope] = []
        b_event = asyncio.Event()

        async def on_message_b(conn_id: str, envelope: Envelope) -> None:
            await router_b.handle_incoming(envelope)
            received_by_b.append(envelope)
            b_event.set()

        server_b = TCPServer(host="127.0.0.1", port=0, on_message=on_message_b)
        await server_b.start()

        # Router A 的客户端连接到 Router B
        client_a = TCPClient(
            host="127.0.0.1",
            port=server_b.port,
            node_id="node-A",
        )
        connected = await client_a.connect()
        assert connected is True
        await asyncio.sleep(0.1)

        # 创建路由器
        router_a = MessageRouter(node_id="node-A")
        router_b = MessageRouter(node_id="node-B")

        # 设置 Router A 的远程发送器
        async def remote_send_a(envelope: Envelope, target: str) -> bool:
            return await client_a.send(envelope)

        router_a.set_remote_sender(remote_send_a)

        yield router_a, router_b, client_a, server_b, received_by_b, b_event

        await client_a.disconnect()
        await server_b.stop()

    @pytest.mark.asyncio
    async def test_router_a_publishes_to_router_b(self, router_pair) -> None:
        """Router A 远程发布消息，Router B 通过 TCP 接收并分发到本地订阅者。"""
        router_a, router_b, client_a, server_b, received_by_b, b_event = router_pair

        # Router B 订阅 TASK_DISPATCH 通道
        local_received: list[Envelope] = []
        router_b.subscribe(Channel.TASK_DISPATCH, lambda env: local_received.append(env))

        # Router A 发布任务分派消息
        task = InferenceTask(
            task_id="task-router-1",
            model_id="llama-13b",
            prompt="路由测试",
            max_tokens=256,
            temperature=0.8,
        )
        dispatch = TaskDispatchMessage(task=task, target_node_id="node-B", instance_id="inst-r1")
        msg = Message(
            type=MessageType.TASK_DISPATCH,
            sender_id="node-A",
            payload=dispatch.to_dict(),
        )
        envelope = Envelope(channel=Channel.TASK_DISPATCH, message=msg, target="node-B")

        result = await router_a.publish_remote(envelope, "node-B")
        assert result is True

        # 等待 Router B 接收
        await asyncio.wait_for(b_event.wait(), timeout=5.0)

        # 验证 Router B 收到消息
        assert len(received_by_b) == 1
        assert received_by_b[0].message.type == MessageType.TASK_DISPATCH

        # 验证本地订阅者收到消息
        assert len(local_received) == 1
        dispatch_back = TaskDispatchMessage.from_dict(local_received[0].message.payload)
        assert dispatch_back.task.task_id == "task-router-1"
        assert dispatch_back.task.model_id == "llama-13b"

    @pytest.mark.asyncio
    async def test_loop_prevention(self, router_pair) -> None:
        """Router B 忽略自己发出的消息，防止回环。"""
        router_a, router_b, client_a, server_b, received_by_b, b_event = router_pair

        # Router B 订阅 HEARTBEATS 通道
        local_received: list[Envelope] = []
        router_b.subscribe(Channel.HEARTBEATS, lambda env: local_received.append(env))

        # 模拟 Router B 发出的消息被回传（sender_id == node-B）
        msg = Message(
            type=MessageType.HEARTBEAT,
            sender_id="node-B",  # 与 Router B 的 node_id 相同
            payload={"seq": 1},
        )
        envelope = Envelope(channel=Channel.HEARTBEATS, message=msg)

        # 通过 TCP 发送到 Router B 的服务器（模拟回环）
        await client_a.send(envelope)

        # 等待一下确保消息处理
        await asyncio.sleep(0.3)

        # Router B 的 handle_incoming 应忽略此消息
        # 但 received_by_b 会记录 TCP 层收到的原始消息
        # 关键是 local_received 应为空（回环被阻止）
        assert len(local_received) == 0

    @pytest.mark.asyncio
    async def test_capacity_report_through_routers(self, router_pair) -> None:
        """容量报告通过 Router 全链路传递。"""
        router_a, router_b, client_a, server_b, received_by_b, b_event = router_pair

        # Router B 订阅容量通道
        local_received: list[Envelope] = []
        router_b.subscribe(Channel.CAPACITY, lambda env: local_received.append(env))

        # Router A 发布容量报告
        gpu = GpuMetrics(
            index=0, name="A100", vram_total_mb=81920,
            vram_free_mb=65536, utilization_percent=20.0, temperature_c=45,
        )
        metrics = CapacityMetrics(
            cpu_count=64, cpu_percent=15.0, memory_total_mb=262144,
            memory_free_mb=200000, gpus=[gpu], active_tasks=2,
            max_concurrent_tasks=32, compute_score=95.0, network_latency_ms=1.5,
        )
        report = CapacityReport(node_id="node-A", metrics=metrics)
        msg = Message(
            type=MessageType.CAPACITY_REPORT,
            sender_id="node-A",
            payload=report.to_dict(),
        )
        envelope = Envelope(channel=Channel.CAPACITY, message=msg)

        result = await router_a.publish_remote(envelope, "node-B")
        assert result is True

        await asyncio.wait_for(b_event.wait(), timeout=5.0)

        assert len(local_received) == 1
        report_back = CapacityReport.from_dict(local_received[0].message.payload)
        assert report_back.node_id == "node-A"
        assert report_back.metrics.gpus[0].name == "A100"
        assert report_back.metrics.gpus[0].vram_free_mb == 65536

    @pytest.mark.asyncio
    async def test_rebalance_request_through_routers(self, router_pair) -> None:
        """重平衡请求通过 Router 全链路传递。"""
        router_a, router_b, client_a, server_b, received_by_b, b_event = router_pair

        # Router B 订阅重平衡通道
        local_received: list[Envelope] = []
        router_b.subscribe(Channel.REBALANCE, lambda env: local_received.append(env))

        # Router A 发布重平衡请求
        action = RebalanceAction(
            action_type=RebalanceType.PIPELINE_RESHIFT,
            source_node_id="node-A",
            target_node_id="node-B",
            task_ids=["task-1"],
            params={"layer_range": [0, 16]},
        )
        plan = RebalancePlan(plan_id="plan-r1", actions=[action], reason="测试重平衡")
        request = RebalanceRequestMessage(plan=plan, initiator_id="node-A")
        msg = Message(
            type=MessageType.REBALANCE_REQUEST,
            sender_id="node-A",
            payload=request.to_dict(),
        )
        envelope = Envelope(channel=Channel.REBALANCE, message=msg)

        result = await router_a.publish_remote(envelope, "node-B")
        assert result is True

        await asyncio.wait_for(b_event.wait(), timeout=5.0)

        assert len(local_received) == 1
        request_back = RebalanceRequestMessage.from_dict(local_received[0].message.payload)
        assert request_back.plan.plan_id == "plan-r1"
        assert request_back.plan.actions[0].action_type == RebalanceType.PIPELINE_RESHIFT


# ===========================================================================
# 跨层边界验证
# ===========================================================================


class TestCrossLayerBoundary:
    """跨协议层边界条件验证。"""

    def test_envelope_direct_roundtrip(self) -> None:
        """Envelope 直接 JSON 序列化/反序列化往返。"""
        msg = Message(
            type=MessageType.TASK_DISPATCH,
            sender_id="node-X",
            payload={"key": "值", "number": 42, "nested": {"a": [1, 2, 3]}},
        )
        envelope = Envelope(channel=Channel.TASK_DISPATCH, message=msg, target="node-Y")

        encoded = encode_envelope(envelope)
        decoded = decode_envelope(encoded)

        _assert_envelope_equal(envelope, decoded)

    def test_frame_direct_roundtrip(self) -> None:
        """Frame 直接二进制编码/解码往返。"""
        payload = b"binary\x00data\xff"
        frame = Frame(frame_type=FrameType.HEARTBEAT, payload=payload)

        raw = encode_frame(frame)
        restored = decode_frame(raw)

        assert restored.frame_type == FrameType.HEARTBEAT
        assert restored.payload == payload

    def test_frame_header_structure(self) -> None:
        """验证帧头部结构：magic + version + type + length。"""
        import struct

        from asc.network.frame import FRAME_MAGIC, FRAME_VERSION

        payload = b"test"
        frame = Frame(frame_type=FrameType.ACK, payload=payload)
        raw = encode_frame(frame)

        # 解析头部
        magic, version, ftype, length = struct.unpack("!IBBI", raw[:10])
        assert magic == FRAME_MAGIC
        assert version == FRAME_VERSION
        assert ftype == FrameType.ACK.value
        assert length == len(payload)
        assert raw[10:] == payload

    def test_empty_payload_frame(self) -> None:
        """空载荷帧的往返。"""
        frame = Frame(frame_type=FrameType.NACK, payload=b"")
        raw = encode_frame(frame)
        restored = decode_frame(raw)
        assert restored.frame_type == FrameType.NACK
        assert restored.payload == b""

    def test_all_frame_types_roundtrip(self) -> None:
        """所有帧类型都能正确编码/解码。"""
        for ft in FrameType:
            frame = Frame(frame_type=ft, payload=b"\x01\x02\x03")
            raw = encode_frame(frame)
            restored = decode_frame(raw)
            assert restored.frame_type == ft
            assert restored.payload == b"\x01\x02\x03"

    def test_all_channels_envelope_roundtrip(self) -> None:
        """所有通道的 Envelope 都能正确序列化/反序列化。"""
        for ch in Channel:
            msg = Message(type=MessageType.HEARTBEAT, sender_id="s", payload={"ch": ch.value})
            env = Envelope(channel=ch, message=msg)
            encoded = encode_envelope(env)
            decoded = decode_envelope(encoded)
            assert decoded.channel == ch

    def test_invalid_frame_magic(self) -> None:
        """无效 magic 的帧解码抛出 ValueError。"""
        import struct
        bad_header = struct.pack("!IBBI", 0xDEADBEEF, 1, 0x00, 0)
        with pytest.raises(ValueError, match="无效 magic"):
            decode_frame(bad_header)

    def test_invalid_frame_version(self) -> None:
        """不支持的版本号帧解码抛出 ValueError。"""
        import struct

        from asc.network.frame import FRAME_MAGIC
        bad_header = struct.pack("!IBBI", FRAME_MAGIC, 99, 0x00, 0)
        with pytest.raises(ValueError, match="不支持的版本"):
            decode_frame(bad_header)

    def test_truncated_frame_data(self) -> None:
        """截断的帧数据解码抛出 ValueError。"""
        import struct

        from asc.network.frame import FRAME_MAGIC
        # 声明长度 100 但实际只有 0 字节载荷
        bad_header = struct.pack("!IBBI", FRAME_MAGIC, 1, 0x00, 100)
        with pytest.raises(ValueError, match="数据长度不匹配"):
            decode_frame(bad_header)

    def test_envelope_frame_type_mismatch(self) -> None:
        """非 ENVELOPE 帧类型调用 decode_envelope_frame 抛出 ValueError。"""
        frame = Frame(frame_type=FrameType.HEARTBEAT, payload=b"{}")
        with pytest.raises(ValueError, match="帧类型不是 ENVELOPE"):
            decode_envelope_frame(frame)
