"""Asc Master 节点主循环。

Master 负责：
- 处理 Command，产生 Event
- 维护 ClusterState
- 调度任务
- 写入事件日志
"""

from __future__ import annotations

from asc.core.event_log import EventLog, MemoryEventLog
from asc.types import (
    ClusterState,
    IndexedEvent,
    InstanceCreated,
    InstanceDeleted,
    NodeId,
    NodeJoined,
    NodeLeft,
    RunnerStatusUpdated,
    TaskCancelled,
    TaskCreated,
    apply,
    empty_state,
    generate_instance_id,
    generate_task_id,
)
from asc.types.commands import (
    CancelTask,
    CreateInstance,
    DeleteInstance,
    ShutdownRunner,
    StartInference,
)
from asc.types.events import Event


class MasterNode:
    """Master 节点。

    处理 Command -> 产生 Event -> 更新 State -> 写入 EventLog。
    """

    def __init__(
        self,
        node_id: str,
        event_log: EventLog | None = None,
    ) -> None:
        self.node_id = node_id
        self._state = empty_state()
        self._next_index = 1
        self.event_log = event_log or MemoryEventLog()

    @property
    def state(self) -> ClusterState:
        return self._state

    def _emit(self, event: Event) -> list[Event]:
        """发出事件，更新状态，写入日志。"""
        indexed = IndexedEvent(event=event, index=self._next_index)
        self._state = apply(self._state, indexed)
        self._next_index += 1
        self.event_log.append(indexed)
        return [event]

    def process_node_joined(self, node_id: NodeId, ip: str, port: int) -> list[Event]:
        """处理节点加入。"""
        return self._emit(NodeJoined(node_id=node_id, ip=ip, port=port))

    def process_node_left(self, node_id: NodeId) -> list[Event]:
        """处理节点离开。"""
        return self._emit(NodeLeft(node_id=node_id))

    def process_create_instance(self, cmd: CreateInstance) -> list[Event]:
        """处理创建实例命令。"""
        # 简化：选择所有在线节点
        node_ids = list(self._state.nodes.keys())
        if not node_ids:
            raise ValueError("无可用节点")

        inst_id = generate_instance_id()
        return self._emit(
            InstanceCreated(
                instance_id=inst_id,
                model_id=cmd.model_id,
                node_ids=node_ids,
                sharding=cmd.sharding,
            )
        )

    def process_delete_instance(self, cmd: DeleteInstance) -> list[Event]:
        """处理删除实例命令。"""
        if cmd.instance_id not in self._state.instances:
            raise ValueError(f"实例 {cmd.instance_id} 不存在")
        return self._emit(InstanceDeleted(instance_id=cmd.instance_id))

    def process_start_inference(self, cmd: StartInference) -> list[Event]:
        """处理推理命令。"""
        if cmd.instance_id not in self._state.instances:
            raise ValueError(f"实例 {cmd.instance_id} 不存在")

        task_id = generate_task_id()
        return self._emit(
            TaskCreated(
                task_id=task_id,
                instance_id=cmd.instance_id,
                prompt=cmd.prompt,
            )
        )

    def process_cancel_task(self, cmd: CancelTask) -> list[Event]:
        """处理取消任务命令。"""
        return self._emit(TaskCancelled(task_id=cmd.task_id))

    def process_shutdown_runner(self, cmd: ShutdownRunner) -> list[Event]:
        """处理关闭 Runner 命令。"""
        return self._emit(RunnerStatusUpdated(node_id=cmd.node_id, status="shutdown"))
