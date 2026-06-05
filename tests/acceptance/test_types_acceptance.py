"""ASC 验收测试 - 类型系统模块

测试范围：types/common.py, types/events.py, types/commands.py, types/state.py
测试维度：功能测试、边界条件测试、异常场景测试
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from asc.types.commands import (
    CancelTask,
    CreateInstance,
    DeleteInstance,
    ShutdownRunner,
    StartInference,
    command_type,
)
from asc.types.common import (
    InstanceId,
    NodeId,
    SessionId,
    TaskId,
    generate_event_id,
    generate_instance_id,
    generate_node_id,
    generate_task_id,
)
from asc.types.events import (
    IndexedEvent,
    InstanceCreated,
    InstanceDeleted,
    NodeJoined,
    NodeLeft,
    RunnerStatusUpdated,
    TaskCancelled,
    TaskCompleted,
    TaskCreated,
    TaskFailed,
    event_type,
)
from asc.types.state import (
    InstanceInfo,
    InstanceState,
    NodeInfo,
    TaskInfo,
    TaskStatus,
    apply,
    empty_state,
)

# ======================================================================
# 1. types/common.py 测试
# ======================================================================


class TestIDGeneration:
    """ID 生成器功能测试。"""

    def test_generate_node_id_format(self):
        """验证 NodeId 格式为 'node-' + 12位hex。"""
        nid = generate_node_id()
        assert nid.startswith("node-")
        assert len(nid) == 17  # "node-" (5) + 12 hex chars

    def test_generate_instance_id_format(self):
        """验证 InstanceId 格式为 'inst-' + 12位hex。"""
        iid = generate_instance_id()
        assert iid.startswith("inst-")
        assert len(iid) == 17

    def test_generate_task_id_format(self):
        """验证 TaskId 格式为 'task-' + 12位hex。"""
        tid = generate_task_id()
        assert tid.startswith("task-")
        assert len(tid) == 17

    def test_generate_event_id_format(self):
        """验证 EventId 格式为 'evt-' + 16位hex。"""
        eid = generate_event_id()
        assert eid.startswith("evt-")
        assert len(eid) == 20  # "evt-" (4) + 16 hex chars

    def test_generate_node_id_uniqueness(self):
        """验证连续生成的 NodeId 不重复。"""
        ids = {generate_node_id() for _ in range(100)}
        assert len(ids) == 100

    def test_generate_instance_id_uniqueness(self):
        """验证连续生成的 InstanceId 不重复。"""
        ids = {generate_instance_id() for _ in range(100)}
        assert len(ids) == 100

    def test_generate_task_id_uniqueness(self):
        """验证连续生成的 TaskId 不重复。"""
        ids = {generate_task_id() for _ in range(100)}
        assert len(ids) == 100

    def test_generate_event_id_uniqueness(self):
        """验证连续生成的 EventId 不重复。"""
        ids = {generate_event_id() for _ in range(100)}
        assert len(ids) == 100


class TestIDBoundary:
    """ID 边界条件测试。"""

    def test_node_id_is_str_subtype(self):
        """NodeId 是 str 的 NewType，应可当字符串使用。"""
        nid = NodeId("custom-id")
        assert isinstance(nid, str)
        assert nid == "custom-id"
        assert len(nid) == 9

    def test_empty_string_id(self):
        """空字符串也可作为 ID（NewType 不做验证）。"""
        nid = NodeId("")
        assert nid == ""

    def test_special_chars_in_id(self):
        """特殊字符可作为 ID。"""
        nid = NodeId("node-中文-🎉")
        assert "中文" in nid


class TestSessionId:
    """SessionId 值对象测试。"""

    def test_session_id_creation(self):
        """正常创建 SessionId。"""
        sid = SessionId(master_node_id=NodeId("node-abc"), election_clock=1)
        assert sid.master_node_id == "node-abc"
        assert sid.election_clock == 1

    def test_session_id_frozen(self):
        """SessionId 是不可变的。"""
        sid = SessionId(master_node_id=NodeId("node-abc"), election_clock=1)
        with pytest.raises(AttributeError):
            sid.election_clock = 2

    def test_session_id_zero_clock(self):
        """election_clock 为 0 的边界情况。"""
        sid = SessionId(master_node_id=NodeId("node-abc"), election_clock=0)
        assert sid.election_clock == 0


# ======================================================================
# 2. types/events.py 测试
# ======================================================================


class TestEventCreation:
    """事件创建功能测试。"""

    def test_node_joined_creation(self):
        """NodeJoined 事件正常创建。"""
        evt = NodeJoined(node_id=NodeId("n1"), ip="192.168.1.1", port=52415)
        assert evt.node_id == "n1"
        assert evt.ip == "192.168.1.1"
        assert evt.port == 52415

    def test_node_left_creation(self):
        """NodeLeft 事件正常创建。"""
        evt = NodeLeft(node_id=NodeId("n1"))
        assert evt.node_id == "n1"

    def test_instance_created_creation(self):
        """InstanceCreated 事件正常创建，含默认 rpc_endpoints。"""
        evt = InstanceCreated(
            instance_id=InstanceId("i1"),
            model_id="llama-7b",
            node_ids=[NodeId("n1"), NodeId("n2")],
            sharding="tensor",
        )
        assert evt.instance_id == "i1"
        assert evt.model_id == "llama-7b"
        assert len(evt.node_ids) == 2
        assert evt.sharding == "tensor"
        assert evt.rpc_endpoints == []

    def test_instance_created_with_rpc(self):
        """InstanceCreated 事件带 rpc_endpoints。"""
        evt = InstanceCreated(
            instance_id=InstanceId("i1"),
            model_id="llama-7b",
            node_ids=[NodeId("n1")],
            sharding="tensor",
            rpc_endpoints=["192.168.1.2:50052"],
        )
        assert len(evt.rpc_endpoints) == 1

    def test_task_created_creation(self):
        """TaskCreated 事件正常创建。"""
        evt = TaskCreated(
            task_id=TaskId("t1"),
            instance_id=InstanceId("i1"),
            prompt="Hello",
        )
        assert evt.task_id == "t1"
        assert evt.prompt == "Hello"

    def test_task_completed_creation(self):
        """TaskCompleted 事件正常创建。"""
        evt = TaskCompleted(task_id=TaskId("t1"), output="Hi there")
        assert evt.output == "Hi there"

    def test_task_failed_creation(self):
        """TaskFailed 事件正常创建。"""
        evt = TaskFailed(task_id=TaskId("t1"), error="OOM")
        assert evt.error == "OOM"

    def test_task_cancelled_creation(self):
        """TaskCancelled 事件正常创建。"""
        evt = TaskCancelled(task_id=TaskId("t1"))
        assert evt.task_id == "t1"

    def test_runner_status_updated_creation(self):
        """RunnerStatusUpdated 事件正常创建。"""
        evt = RunnerStatusUpdated(node_id=NodeId("n1"), status="ready")
        assert evt.status == "ready"


class TestEventImmutability:
    """事件不可变性测试。"""

    def test_node_joined_frozen(self):
        """NodeJoined 事件不可修改。"""
        evt = NodeJoined(node_id=NodeId("n1"), ip="1.1.1.1", port=80)
        with pytest.raises(AttributeError):
            evt.ip = "2.2.2.2"

    def test_instance_created_frozen(self):
        """InstanceCreated 事件不可修改。"""
        evt = InstanceCreated(
            instance_id=InstanceId("i1"),
            model_id="m1",
            node_ids=[NodeId("n1")],
            sharding="tensor",
        )
        with pytest.raises(AttributeError):
            evt.model_id = "m2"

    def test_indexed_event_frozen(self):
        """IndexedEvent 不可修改。"""
        evt = IndexedEvent(
            event=NodeJoined(node_id=NodeId("n1"), ip="1.1.1.1", port=80),
            index=1,
        )
        with pytest.raises(AttributeError):
            evt.index = 2


class TestEventTypeFunction:
    """event_type() 函数测试。"""

    def test_event_type_node_joined(self):
        assert event_type(NodeJoined(node_id=NodeId("n1"), ip="1.1.1.1", port=80)) == "node_joined"

    def test_event_type_node_left(self):
        assert event_type(NodeLeft(node_id=NodeId("n1"))) == "node_left"

    def test_event_type_instance_created(self):
        assert event_type(
            InstanceCreated(instance_id=InstanceId("i1"), model_id="m", node_ids=[], sharding="t")
        ) == "instance_created"

    def test_event_type_instance_deleted(self):
        assert event_type(InstanceDeleted(instance_id=InstanceId("i1"))) == "instance_deleted"

    def test_event_type_task_created(self):
        assert event_type(
            TaskCreated(task_id=TaskId("t1"), instance_id=InstanceId("i1"), prompt="p")
        ) == "task_created"

    def test_event_type_task_completed(self):
        assert event_type(TaskCompleted(task_id=TaskId("t1"), output="o")) == "task_completed"

    def test_event_type_task_failed(self):
        assert event_type(TaskFailed(task_id=TaskId("t1"), error="e")) == "task_failed"

    def test_event_type_task_cancelled(self):
        assert event_type(TaskCancelled(task_id=TaskId("t1"))) == "task_cancelled"

    def test_event_type_runner_status_updated(self):
        assert event_type(
            RunnerStatusUpdated(node_id=NodeId("n1"), status="idle")
        ) == "runner_status_updated"


class TestEventBoundary:
    """事件边界条件测试。"""

    def test_empty_prompt_task_created(self):
        """空 prompt 创建 TaskCreated。"""
        evt = TaskCreated(task_id=TaskId("t1"), instance_id=InstanceId("i1"), prompt="")
        assert evt.prompt == ""

    def test_empty_error_task_failed(self):
        """空 error 创建 TaskFailed。"""
        evt = TaskFailed(task_id=TaskId("t1"), error="")
        assert evt.error == ""

    def test_empty_node_ids_instance_created(self):
        """空 node_ids 创建 InstanceCreated。"""
        evt = InstanceCreated(
            instance_id=InstanceId("i1"), model_id="m", node_ids=[], sharding="tensor"
        )
        assert evt.node_ids == []

    def test_port_zero(self):
        """端口为 0 的边界情况。"""
        evt = NodeJoined(node_id=NodeId("n1"), ip="0.0.0.0", port=0)
        assert evt.port == 0

    def test_port_max(self):
        """端口为 65535 的边界情况。"""
        evt = NodeJoined(node_id=NodeId("n1"), ip="1.1.1.1", port=65535)
        assert evt.port == 65535

    def test_indexed_event_zero_index(self):
        """IndexedEvent index 为 0。"""
        ie = IndexedEvent(
            event=NodeJoined(node_id=NodeId("n1"), ip="1.1.1.1", port=80),
            index=0,
        )
        assert ie.index == 0

    def test_indexed_event_large_index(self):
        """IndexedEvent 大索引值。"""
        ie = IndexedEvent(
            event=NodeJoined(node_id=NodeId("n1"), ip="1.1.1.1", port=80),
            index=999999999,
        )
        assert ie.index == 999999999


# ======================================================================
# 3. types/commands.py 测试
# ======================================================================


class TestCommandCreation:
    """命令创建功能测试。"""

    def test_create_instance_creation(self):
        cmd = CreateInstance(model_id="llama-7b", sharding="tensor")
        assert cmd.model_id == "llama-7b"
        assert cmd.sharding == "tensor"

    def test_delete_instance_creation(self):
        cmd = DeleteInstance(instance_id=InstanceId("i1"))
        assert cmd.instance_id == "i1"

    def test_start_inference_defaults(self):
        cmd = StartInference(instance_id=InstanceId("i1"), prompt="Hello")
        assert cmd.max_tokens == 128
        assert cmd.temperature == 0.7

    def test_start_inference_custom(self):
        cmd = StartInference(
            instance_id=InstanceId("i1"), prompt="Hello", max_tokens=512, temperature=1.5
        )
        assert cmd.max_tokens == 512
        assert cmd.temperature == 1.5

    def test_cancel_task_creation(self):
        cmd = CancelTask(task_id=TaskId("t1"))
        assert cmd.task_id == "t1"

    def test_shutdown_runner_creation(self):
        cmd = ShutdownRunner(node_id=NodeId("n1"))
        assert cmd.node_id == "n1"


class TestCommandImmutability:
    """命令不可变性测试。"""

    def test_create_instance_frozen(self):
        cmd = CreateInstance(model_id="m", sharding="t")
        with pytest.raises(AttributeError):
            cmd.model_id = "m2"

    def test_start_inference_frozen(self):
        cmd = StartInference(instance_id=InstanceId("i1"), prompt="p")
        with pytest.raises(AttributeError):
            cmd.prompt = "new"


class TestCommandTypeFunction:
    """command_type() 函数测试。"""

    def test_command_type_create_instance(self):
        assert command_type(CreateInstance(model_id="m", sharding="t")) == "create_instance"

    def test_command_type_delete_instance(self):
        assert command_type(DeleteInstance(instance_id=InstanceId("i1"))) == "delete_instance"

    def test_command_type_start_inference(self):
        assert command_type(
            StartInference(instance_id=InstanceId("i1"), prompt="p")
        ) == "start_inference"

    def test_command_type_cancel_task(self):
        assert command_type(CancelTask(task_id=TaskId("t1"))) == "cancel_task"

    def test_command_type_shutdown_runner(self):
        assert command_type(ShutdownRunner(node_id=NodeId("n1"))) == "shutdown_runner"


class TestCommandBoundary:
    """命令边界条件测试。"""

    def test_start_inference_max_tokens_zero_raises(self):
        """max_tokens 为 0 应抛出 ValueError。"""
        with pytest.raises(ValueError, match="max_tokens"):
            StartInference(instance_id=InstanceId("i1"), prompt="p", max_tokens=0)

    def test_start_inference_temperature_zero(self):
        """temperature 为 0 的边界情况。"""
        cmd = StartInference(instance_id=InstanceId("i1"), prompt="p", temperature=0.0)
        assert cmd.temperature == 0.0

    def test_start_inference_negative_temperature_raises(self):
        """temperature 为负数应抛出 ValueError。"""
        with pytest.raises(ValueError, match="temperature"):
            StartInference(instance_id=InstanceId("i1"), prompt="p", temperature=-1.0)

    def test_start_inference_very_large_max_tokens(self):
        """max_tokens 为极大值的边界情况。"""
        cmd = StartInference(instance_id=InstanceId("i1"), prompt="p", max_tokens=2**31)
        assert cmd.max_tokens == 2**31

    def test_empty_prompt(self):
        """空 prompt。"""
        cmd = StartInference(instance_id=InstanceId("i1"), prompt="")
        assert cmd.prompt == ""


# ======================================================================
# 4. types/state.py 测试
# ======================================================================


class TestEmptyState:
    """空状态测试。"""

    def test_empty_state_has_no_nodes(self):
        state = empty_state()
        assert len(state.nodes) == 0

    def test_empty_state_has_no_instances(self):
        state = empty_state()
        assert len(state.instances) == 0

    def test_empty_state_has_no_tasks(self):
        state = empty_state()
        assert len(state.tasks) == 0

    def test_empty_state_event_index_zero(self):
        state = empty_state()
        assert state.event_index == 0


class TestStateImmutability:
    """状态不可变性测试。"""

    def test_cluster_state_frozen(self):
        state = empty_state()
        with pytest.raises(AttributeError):
            state.event_index = 99

    def test_node_info_frozen(self):
        info = NodeInfo(node_id=NodeId("n1"), ip="1.1.1.1", port=80)
        with pytest.raises(AttributeError):
            info.ip = "2.2.2.2"

    def test_instance_info_frozen(self):
        info = InstanceInfo(
            instance_id=InstanceId("i1"), model_id="m", node_ids=[], sharding="t"
        )
        with pytest.raises(AttributeError):
            info.model_id = "m2"

    def test_task_info_frozen(self):
        info = TaskInfo(task_id=TaskId("t1"), instance_id=InstanceId("i1"), prompt="p")
        with pytest.raises(AttributeError):
            info.prompt = "new"


class TestApplyNodeEvents:
    """apply() 节点事件测试。"""

    def test_apply_node_joined(self):
        state = empty_state()
        ie = IndexedEvent(
            event=NodeJoined(node_id=NodeId("n1"), ip="192.168.1.1", port=52415),
            index=1,
        )
        new_state = apply(state, ie)
        assert NodeId("n1") in new_state.nodes
        assert new_state.nodes[NodeId("n1")].ip == "192.168.1.1"
        assert new_state.event_index == 1

    def test_apply_node_left(self):
        state = empty_state()
        state = apply(
            state,
            IndexedEvent(
                event=NodeJoined(node_id=NodeId("n1"), ip="1.1.1.1", port=80), index=1
            ),
        )
        state = apply(
            state,
            IndexedEvent(event=NodeLeft(node_id=NodeId("n1")), index=2),
        )
        assert NodeId("n1") not in state.nodes

    def test_apply_node_joined_twice_overwrites(self):
        """同一节点重复加入，后一次覆盖。"""
        state = empty_state()
        state = apply(
            state,
            IndexedEvent(
                event=NodeJoined(node_id=NodeId("n1"), ip="1.1.1.1", port=80), index=1
            ),
        )
        state = apply(
            state,
            IndexedEvent(
                event=NodeJoined(node_id=NodeId("n1"), ip="2.2.2.2", port=90), index=2
            ),
        )
        assert state.nodes[NodeId("n1")].ip == "2.2.2.2"

    def test_apply_node_left_nonexistent(self):
        """删除不存在的节点，不报错，状态不变。"""
        state = empty_state()
        new_state = apply(
            state,
            IndexedEvent(event=NodeLeft(node_id=NodeId("ghost")), index=1),
        )
        assert len(new_state.nodes) == 0

    def test_apply_multiple_nodes(self):
        """多个节点加入。"""
        state = empty_state()
        for i in range(5):
            state = apply(
                state,
                IndexedEvent(
                    event=NodeJoined(
                        node_id=NodeId(f"n{i}"), ip=f"10.0.0.{i}", port=52415
                    ),
                    index=i + 1,
                ),
            )
        assert len(state.nodes) == 5


class TestApplyInstanceEvents:
    """apply() 实例事件测试。"""

    def _state_with_node(self):
        state = empty_state()
        return apply(
            state,
            IndexedEvent(
                event=NodeJoined(node_id=NodeId("n1"), ip="1.1.1.1", port=80), index=1
            ),
        )

    def test_apply_instance_created(self):
        state = self._state_with_node()
        state = apply(
            state,
            IndexedEvent(
                event=InstanceCreated(
                    instance_id=InstanceId("i1"),
                    model_id="llama-7b",
                    node_ids=[NodeId("n1")],
                    sharding="tensor",
                    rpc_endpoints=["1.1.1.1:50052"],
                ),
                index=2,
            ),
        )
        assert InstanceId("i1") in state.instances
        inst = state.instances[InstanceId("i1")]
        assert inst.model_id == "llama-7b"
        assert inst.state == InstanceState.CREATING

    def test_apply_instance_deleted(self):
        state = self._state_with_node()
        state = apply(
            state,
            IndexedEvent(
                event=InstanceCreated(
                    instance_id=InstanceId("i1"),
                    model_id="m",
                    node_ids=[NodeId("n1")],
                    sharding="tensor",
                ),
                index=2,
            ),
        )
        state = apply(
            state,
            IndexedEvent(event=InstanceDeleted(instance_id=InstanceId("i1")), index=3),
        )
        assert InstanceId("i1") not in state.instances

    def test_apply_instance_deleted_nonexistent(self):
        """删除不存在的实例，不报错。"""
        state = self._state_with_node()
        new_state = apply(
            state,
            IndexedEvent(event=InstanceDeleted(instance_id=InstanceId("ghost")), index=2),
        )
        assert len(new_state.instances) == 0


class TestApplyTaskEvents:
    """apply() 任务事件测试。"""

    def test_apply_task_created(self):
        state = empty_state()
        state = apply(
            state,
            IndexedEvent(
                event=TaskCreated(
                    task_id=TaskId("t1"), instance_id=InstanceId("i1"), prompt="Hello"
                ),
                index=1,
            ),
        )
        assert TaskId("t1") in state.tasks
        assert state.tasks[TaskId("t1")].status == TaskStatus.PENDING

    def test_apply_task_completed(self):
        state = empty_state()
        state = apply(
            state,
            IndexedEvent(
                event=TaskCreated(
                    task_id=TaskId("t1"), instance_id=InstanceId("i1"), prompt="Hello"
                ),
                index=1,
            ),
        )
        state = apply(
            state,
            IndexedEvent(
                event=TaskCompleted(task_id=TaskId("t1"), output="Hi"), index=2
            ),
        )
        assert state.tasks[TaskId("t1")].status == TaskStatus.COMPLETED
        assert state.tasks[TaskId("t1")].output == "Hi"

    def test_apply_task_failed(self):
        state = empty_state()
        state = apply(
            state,
            IndexedEvent(
                event=TaskCreated(
                    task_id=TaskId("t1"), instance_id=InstanceId("i1"), prompt="Hello"
                ),
                index=1,
            ),
        )
        state = apply(
            state,
            IndexedEvent(
                event=TaskFailed(task_id=TaskId("t1"), error="OOM"), index=2
            ),
        )
        assert state.tasks[TaskId("t1")].status == TaskStatus.FAILED
        assert state.tasks[TaskId("t1")].error == "OOM"

    def test_apply_task_cancelled(self):
        state = empty_state()
        state = apply(
            state,
            IndexedEvent(
                event=TaskCreated(
                    task_id=TaskId("t1"), instance_id=InstanceId("i1"), prompt="Hello"
                ),
                index=1,
            ),
        )
        state = apply(
            state,
            IndexedEvent(event=TaskCancelled(task_id=TaskId("t1")), index=2),
        )
        assert state.tasks[TaskId("t1")].status == TaskStatus.CANCELLED

    def test_apply_task_completed_nonexistent(self):
        """完成不存在的任务，状态不变。"""
        state = empty_state()
        new_state = apply(
            state,
            IndexedEvent(
                event=TaskCompleted(task_id=TaskId("ghost"), output="x"), index=1
            ),
        )
        assert len(new_state.tasks) == 0

    def test_apply_task_failed_nonexistent(self):
        """失败不存在的任务，状态不变。"""
        state = empty_state()
        new_state = apply(
            state,
            IndexedEvent(event=TaskFailed(task_id=TaskId("ghost"), error="e"), index=1),
        )
        assert len(new_state.tasks) == 0

    def test_apply_task_cancelled_nonexistent(self):
        """取消不存在的任务，状态不变。"""
        state = empty_state()
        new_state = apply(
            state,
            IndexedEvent(event=TaskCancelled(task_id=TaskId("ghost")), index=1),
        )
        assert len(new_state.tasks) == 0


class TestApplyRunnerStatus:
    """apply() Runner 状态更新测试。"""

    def test_apply_runner_status_updated(self):
        state = empty_state()
        state = apply(
            state,
            IndexedEvent(
                event=NodeJoined(node_id=NodeId("n1"), ip="1.1.1.1", port=80), index=1
            ),
        )
        state = apply(
            state,
            IndexedEvent(
                event=RunnerStatusUpdated(node_id=NodeId("n1"), status="ready"), index=2
            ),
        )
        assert state.nodes[NodeId("n1")].runner_status == "ready"

    def test_apply_runner_status_nonexistent_node(self):
        """更新不存在节点的 Runner 状态，状态不变。"""
        state = empty_state()
        new_state = apply(
            state,
            IndexedEvent(
                event=RunnerStatusUpdated(node_id=NodeId("ghost"), status="ready"), index=1
            ),
        )
        assert len(new_state.nodes) == 0


class TestApplyUnknownEvent:
    """未知事件类型测试。"""

    def test_apply_unknown_event_ignored(self):
        """未知事件类型应被忽略，仅更新 event_index。"""

        @dataclass(frozen=True)
        class UnknownEvent:
            data: str

        state = empty_state()
        ie = IndexedEvent(event=UnknownEvent(data="test"), index=1)
        new_state = apply(state, ie)
        assert new_state.event_index == 1
        assert len(new_state.nodes) == 0


class TestStateEventIndex:
    """event_index 追踪测试。"""

    def test_event_index_increments(self):
        """event_index 随事件递增。"""
        state = empty_state()
        for i in range(1, 6):
            state = apply(
                state,
                IndexedEvent(
                    event=NodeJoined(node_id=NodeId(f"n{i}"), ip=f"1.1.1.{i}", port=80),
                    index=i,
                ),
            )
        assert state.event_index == 5

    def test_event_index_not_sequential(self):
        """event_index 不一定连续（可能跳过）。"""
        state = empty_state()
        state = apply(
            state,
            IndexedEvent(
                event=NodeJoined(node_id=NodeId("n1"), ip="1.1.1.1", port=80), index=100
            ),
        )
        assert state.event_index == 100


class TestInstanceStateEnum:
    """InstanceState 枚举测试。"""

    def test_all_instance_states(self):
        assert InstanceState.CREATING.value == "creating"
        assert InstanceState.RUNNING.value == "running"
        assert InstanceState.DEGRADED.value == "degraded"
        assert InstanceState.FAILED.value == "failed"
        assert InstanceState.STOPPED.value == "stopped"


class TestTaskStatusEnum:
    """TaskStatus 枚举测试。"""

    def test_all_task_statuses(self):
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.CANCELLED.value == "cancelled"
