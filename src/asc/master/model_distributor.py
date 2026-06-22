"""模型分发器：从 Master 节点向 Worker 节点分发模型文件。

核心流程：
1. Master 下载模型到本地
2. Master 根据集群资源规划分片策略（哪些节点加载哪些层）
3. Master 通过 Binary Frame 协议将模型文件分片分发到各 Worker 节点
4. Worker 通过 Binary Frame 接收模型分片并写入本地文件
5. Master 通知 Worker 加载对应的模型层

设计原则：
- 使用 Binary Frame (MODEL_CHUNK) 分发大文件，统一协议栈
- 4MB 分片大小，确保不超过 MAX_FRAME_SIZE (10MB)
- 每个 MODEL_CHUNK 等待 MODEL_CHUNK_ACK 确认后再发下一个（可靠传输）
- Worker 端通过 on_frame 回调接收 MODEL_CHUNK 帧
"""

from __future__ import annotations

import asyncio
import logging
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from asc.network.frame import Frame, FrameType

logger = logging.getLogger(__name__)

# 分块大小：4MB（安全值，远小于 MAX_FRAME_SIZE 10MB）
CHUNK_SIZE = 4 * 1024 * 1024


@dataclass(frozen=True)
class DistributeTask:
    """模型分发任务。"""

    model_id: str
    model_path: str  # Master 上的模型文件路径
    target_node_id: str
    target_conn_id: str  # TCP 连接 ID（替代原来的 IP:Port）
    target_dir: str = ""  # Worker 上的目标目录，空则使用默认


@dataclass(frozen=True)
class DistributeResult:
    """分发结果。"""

    success: bool
    model_id: str
    node_id: str
    remote_path: str = ""
    file_size_mb: int = 0
    error: str = ""


@dataclass(frozen=True)
class LayerAssignment:
    """模型层分配。"""

    node_id: str
    node_ip: str
    node_port: int
    model_path: str  # Worker 上的模型文件路径
    start_layer: int
    end_layer: int
    total_layers: int
    gpu_layers: int  # 该节点应加载到 GPU 的层数


# ------------------------------------------------------------------
# 负载编解码函数
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
# 分片辅助函数
# ------------------------------------------------------------------


def split_model_into_chunks(file_path: Path) -> list[dict[str, Any]]:
    """将模型文件分割为 4MB 分片。

    Returns:
        [{"index": int, "total": int, "data": bytes}, ...]
    """
    file_size = file_path.stat().st_size
    total_chunks = max(1, (file_size + CHUNK_SIZE - 1) // CHUNK_SIZE)
    chunks: list[dict[str, Any]] = []

    with open(file_path, "rb") as f:
        for i in range(total_chunks):
            chunk_data = f.read(CHUNK_SIZE)
            chunks.append({
                "index": i,
                "total": total_chunks,
                "data": chunk_data,
            })

    return chunks


# ------------------------------------------------------------------
# 模型分发器
# ------------------------------------------------------------------


class ModelDistributor:
    """模型分发器。

    从 Master 向 Worker 节点分发模型文件，
    并根据集群资源规划层分配方案。

    使用 Binary Frame 协议 (MODEL_CHUNK) 替代 HTTP 传输。
    """

    def __init__(
        self,
        models_dir: Path | None = None,
        tcp_server: Any = None,
    ) -> None:
        self._models_dir = models_dir or Path("./models")
        self._models_dir.mkdir(parents=True, exist_ok=True)
        self._tcp_server = tcp_server
        # 等待 ACK 的 Future 字典: (model_id, chunk_index) -> asyncio.Future
        self._pending_acks: dict[tuple[str, int], asyncio.Future] = {}

    def set_tcp_server(self, tcp_server: Any) -> None:
        """设置 TCP Server 引用。"""
        self._tcp_server = tcp_server

    def handle_chunk_ack(self, frame: Frame) -> None:
        """处理收到的 MODEL_CHUNK_ACK 帧。

        由 TCPServer 的 on_frame 回调调用。
        """
        model_id, chunk_index, success = decode_model_chunk_ack_payload(frame.payload)
        key = (model_id, chunk_index)
        future = self._pending_acks.get(key)
        if future is not None and not future.done():
            future.set_result(success)
        else:
            logger.warning("收到未预期的 ACK: model_id=%s, chunk_index=%d", model_id, chunk_index)

    async def distribute_to_worker(
        self,
        task: DistributeTask,
        timeout: float = 300.0,
    ) -> DistributeResult:
        """将模型文件分发到单个 Worker 节点。

        通过 Binary Frame (MODEL_CHUNK) 将模型文件分片发送到 Worker。
        每个分片等待 MODEL_CHUNK_ACK 确认后再发送下一个。

        Args:
            task: 分发任务
            timeout: 超时时间（秒）

        Returns:
            DistributeResult
        """
        if self._tcp_server is None:
            return DistributeResult(
                success=False,
                model_id=task.model_id,
                node_id=task.target_node_id,
                error="TCPServer 未设置",
            )

        model_path = Path(task.model_path)
        if not model_path.exists():
            return DistributeResult(
                success=False,
                model_id=task.model_id,
                node_id=task.target_node_id,
                error=f"模型文件不存在: {model_path}",
            )

        file_size = model_path.stat().st_size
        file_size_mb = int(file_size // (1024 * 1024))

        logger.info(
            "开始分发模型 %s 到 %s (conn_id=%s), 文件大小: %d MB",
            task.model_id, task.target_node_id,
            task.target_conn_id, file_size_mb,
        )

        try:
            # 分片
            chunks = split_model_into_chunks(model_path)
            total_chunks = len(chunks)

            for chunk_info in chunks:
                chunk_index = chunk_info["index"]
                chunk_data = chunk_info["data"]

                # 编码 MODEL_CHUNK 帧负载
                payload = encode_model_chunk_payload(
                    model_id=task.model_id,
                    chunk_index=chunk_index,
                    total_chunks=total_chunks,
                    chunk_data=chunk_data,
                )

                frame = Frame(frame_type=FrameType.MODEL_CHUNK, payload=payload)

                # 注册 ACK Future
                ack_key = (task.model_id, chunk_index)
                loop = asyncio.get_running_loop()
                ack_future: asyncio.Future[bool] = loop.create_future()
                self._pending_acks[ack_key] = ack_future

                try:
                    # 发送帧
                    sent = await self._tcp_server.send_frame(task.target_conn_id, frame)
                    if not sent:
                        return DistributeResult(
                            success=False,
                            model_id=task.model_id,
                            node_id=task.target_node_id,
                            error=f"发送 MODEL_CHUNK 帧失败 (chunk {chunk_index}/{total_chunks})",
                        )

                    # 等待 ACK（每片至少 10 秒超时，大模型分片数多时有足够时间）
                    try:
                        per_chunk_timeout = max(timeout / total_chunks, 10.0)
                        ack_success = await asyncio.wait_for(ack_future, timeout=per_chunk_timeout)
                        if not ack_success:
                            return DistributeResult(
                                success=False,
                                model_id=task.model_id,
                                node_id=task.target_node_id,
                                error=f"Worker 拒绝分片 {chunk_index}/{total_chunks}",
                            )
                    except asyncio.TimeoutError:
                        return DistributeResult(
                            success=False,
                            model_id=task.model_id,
                            node_id=task.target_node_id,
                            error=f"等待 ACK 超时 (chunk {chunk_index}/{total_chunks})",
                        )
                finally:
                    self._pending_acks.pop(ack_key, None)

            # 所有分片发送成功
            remote_path = str(Path(task.target_dir) / f"{task.model_id}.gguf") if task.target_dir else ""
            logger.info(
                "模型 %s 分发到 %s 成功 (%d 个分片)",
                task.model_id, task.target_node_id, total_chunks,
            )
            return DistributeResult(
                success=True,
                model_id=task.model_id,
                node_id=task.target_node_id,
                remote_path=remote_path,
                file_size_mb=file_size_mb,
            )

        except Exception as e:
            return DistributeResult(
                success=False,
                model_id=task.model_id,
                node_id=task.target_node_id,
                error=f"分发异常: {e}",
            )

    async def distribute_to_cluster(
        self,
        model_id: str,
        model_path: str,
        nodes: dict[str, dict[str, Any]],
    ) -> list[DistributeResult]:
        """将模型文件并行分发到集群中的多个 Worker 节点。

        Args:
            model_id: 模型 ID
            model_path: Master 上的模型文件路径
            nodes: 目标节点 {node_id: {"conn_id": str}}

        Returns:
            分发结果列表
        """
        tasks = []
        for node_id, info in nodes.items():
            task = DistributeTask(
                model_id=model_id,
                model_path=model_path,
                target_node_id=node_id,
                target_conn_id=info["conn_id"],
            )
            tasks.append(task)

        # 并行分发
        results = await asyncio.gather(
            *[self.distribute_to_worker(t) for t in tasks],
            return_exceptions=True,
        )

        # 处理异常
        final_results = []
        for result in results:
            if isinstance(result, Exception):
                final_results.append(DistributeResult(
                    success=False,
                    model_id=model_id,
                    node_id="unknown",
                    error=str(result),
                ))
            else:
                final_results.append(result)

        success_count = sum(1 for r in final_results if r.success)
        logger.info(
            "模型 %s 分发完成: %d/%d 成功",
            model_id, success_count, len(final_results),
        )

        return final_results

    def plan_layer_assignment(
        self,
        model_id: str,
        model_path: str,
        total_layers: int,
        node_resources: dict[str, dict[str, Any]],
        distribute_results: list[DistributeResult],
    ) -> list[LayerAssignment]:
        """根据节点资源规划模型层分配方案。

        按各节点可用 VRAM 比例分配层数，VRAM 越大分配越多层。
        GPU 层数也按 VRAM 比例计算，剩余层由 CPU 卸载。

        Args:
            model_id: 模型 ID
            model_path: 模型文件路径（Worker 上的路径）
            total_layers: 模型总层数
            node_resources: {node_id: {"vram_free_mb": int, "ip": str, "port": int}}
            distribute_results: 分发结果（用于获取远程路径）

        Returns:
            LayerAssignment 列表
        """
        # 构建节点 VRAM 映射（仅包含分发成功的节点）
        node_vram: dict[str, int] = {}
        node_info: dict[str, dict] = {}

        for result in distribute_results:
            if not result.success:
                continue
            nid = result.node_id
            res = node_resources.get(nid, {})
            vram = 0
            gpus = res.get("gpus", [])
            if gpus:
                vram = gpus[0].get("vram_free_mb", 0) if isinstance(gpus[0], dict) else 0
            if vram > 0:
                node_vram[nid] = vram
                node_info[nid] = res

        if not node_vram:
            logger.warning("无可用节点进行层分配")
            return []

        # 按 VRAM 比例分配层
        total_vram = sum(node_vram.values())
        assignments: list[LayerAssignment] = []
        assigned_so_far = 0
        node_list = list(node_vram.keys())

        for i, nid in enumerate(node_list):
            vram = node_vram[nid]
            info = node_info[nid]

            if i == len(node_list) - 1:
                # 最后一个节点分配剩余所有层
                assigned_layers = total_layers - assigned_so_far
            else:
                # 按 VRAM 比例分配
                ratio = vram / total_vram
                assigned_layers = max(1, int(total_layers * ratio))

            # GPU 层数 = 分配层数 * (VRAM可用 / VRAM总量)
            # 简化：假设所有分配的层都加载到 GPU
            gpu_layers = assigned_layers

            # 查找分发结果中的远程路径
            remote_path = ""
            for r in distribute_results:
                if r.node_id == nid and r.success:
                    remote_path = r.remote_path
                    break

            start_layer = assigned_so_far
            end_layer = assigned_so_far + assigned_layers - 1

            assigned_so_far += assigned_layers

            assignments.append(LayerAssignment(
                node_id=nid,
                node_ip=info.get("ip", ""),
                node_port=info.get("port", 0),
                model_path=remote_path,
                start_layer=start_layer,
                end_layer=end_layer,
                total_layers=total_layers,
                gpu_layers=gpu_layers,
            ))

        logger.info(
            "模型 %s 层分配方案: %s",
            model_id,
            [(a.node_id, f"L{a.start_layer}-L{a.end_layer}") for a in assignments],
        )

        return assignments
