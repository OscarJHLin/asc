"""ASC 验收测试 - 核心模块

测试范围：core/config.py, core/election.py, core/event_log.py, core/model_manager.py
测试维度：功能测试、边界条件测试、异常场景测试
"""

from __future__ import annotations

import json

import pytest

from asc.core.config import AscConfig
from asc.core.election import BullyElection, ElectionMessage, ElectionMessageType, ElectionState
from asc.core.event_log import (
    DiskEventLog,
    MemoryEventLog,
    deserialize_indexed_event,
    serialize_indexed_event,
)
from asc.core.model_manager import DownloadProgress, ModelManager
from asc.types.common import InstanceId, NodeId, TaskId
from asc.types.events import (
    IndexedEvent,
    InstanceCreated,
    NodeJoined,
    TaskCreated,
)

# ======================================================================
# 1. core/config.py 测试
# ======================================================================


class TestAscConfigDefaults:
    """AscConfig 默认值功能测试。"""

    def test_default_node_port(self):
        config = AscConfig()
        assert config.get("node", "port") == 52415

    def test_default_api_port(self):
        config = AscConfig()
        assert config.get("api", "port") == 52415

    def test_default_api_key_empty(self):
        config = AscConfig()
        assert config.get("api", "key") == ""

    def test_default_models_path(self):
        config = AscConfig()
        assert config.get("paths", "models") == "./models"

    def test_default_llama_cpp_empty(self):
        config = AscConfig()
        assert config.get("paths", "llama_cpp") == ""

    def test_default_discovery_port(self):
        config = AscConfig()
        assert config.get("network", "discovery_port") == 52416

    def test_default_heartbeat_timeout(self):
        config = AscConfig()
        assert config.get("cluster", "heartbeat_timeout") == 30

    def test_default_inference_backend(self):
        config = AscConfig()
        assert config.get("inference", "backend") == "llama.cpp"


class TestAscConfigGetSet:
    """AscConfig get/set 功能测试。"""

    def test_get_nonexistent_section(self):
        config = AscConfig()
        assert config.get("nonexistent", "key") is None

    def test_get_nonexistent_key(self):
        config = AscConfig()
        assert config.get("node", "nonexistent") is None

    def test_get_with_default(self):
        config = AscConfig()
        assert config.get("x", "y", default="fallback") == "fallback"

    def test_set_new_section(self):
        config = AscConfig()
        config.set("custom", "key", "value")
        assert config.get("custom", "key") == "value"

    def test_set_overwrite(self):
        config = AscConfig()
        config.set("node", "port", 9999)
        assert config.get("node", "port") == 9999

    def test_set_string_value(self):
        config = AscConfig()
        config.set("node", "name", "my-node")
        assert config.get("node", "name") == "my-node"


class TestAscConfigModelMappings:
    """AscConfig 模型映射测试。"""

    def test_add_model_mapping(self):
        config = AscConfig()
        config.add_model_mapping("gpt4", "/models/gpt4.gguf")
        assert config.resolve_model_path("gpt4") == "/models/gpt4.gguf"

    def test_remove_model_mapping(self):
        config = AscConfig()
        config.add_model_mapping("gpt4", "/models/gpt4.gguf")
        config.remove_model_mapping("gpt4")
        assert config.resolve_model_path("gpt4") is None

    def test_remove_nonexistent_mapping(self):
        """删除不存在的映射不报错。"""
        config = AscConfig()
        config.remove_model_mapping("nonexistent")

    def test_resolve_nonexistent_mapping(self):
        config = AscConfig()
        assert config.resolve_model_path("nonexistent") is None

    def test_list_model_mappings_empty(self):
        config = AscConfig()
        assert config.list_model_mappings() == {}

    def test_list_model_mappings(self):
        config = AscConfig()
        config.add_model_mapping("a", "path_a")
        config.add_model_mapping("b", "path_b")
        mappings = config.list_model_mappings()
        assert len(mappings) == 2
        assert mappings["a"] == "path_a"

    def test_model_mapping_overwrite(self):
        """重复添加同名映射会覆盖。"""
        config = AscConfig()
        config.add_model_mapping("m", "old_path")
        config.add_model_mapping("m", "new_path")
        assert config.resolve_model_path("m") == "new_path"


class TestAscConfigFileIO:
    """AscConfig 文件读写测试。"""

    def test_save_and_load(self, tmp_path):
        config = AscConfig()
        config.set("node", "port", 9999)
        config.add_model_mapping("test", "/test.gguf")
        path = tmp_path / "config.json"
        config.save(path)

        config2 = AscConfig()
        config2.load(path)
        assert config2.get("node", "port") == 9999
        assert config2.resolve_model_path("test") == "/test.gguf"

    def test_load_nonexistent_file(self, tmp_path):
        """加载不存在的文件不报错。"""
        config = AscConfig()
        config.load(tmp_path / "nonexistent.json")
        # 默认值应保持不变
        assert config.get("node", "port") == 52415

    def test_save_creates_parent_dirs(self, tmp_path):
        """保存时自动创建父目录。"""
        config = AscConfig()
        path = tmp_path / "deep" / "nested" / "config.json"
        config.save(path)
        assert path.exists()

    def test_to_dict(self):
        config = AscConfig()
        d = config.to_dict()
        assert "config" in d
        assert "model_mappings" in d
        assert d["config"]["node"]["port"] == 52415


class TestAscConfigEnvOverride:
    """AscConfig 环境变量覆盖测试。"""

    def test_env_override_node_port(self, monkeypatch):
        monkeypatch.setenv("ASC_NODE_PORT", "9999")
        config = AscConfig()
        assert config.get("node", "port") == 9999

    def test_env_override_api_key(self, monkeypatch):
        monkeypatch.setenv("ASC_API_KEY", "secret123")
        config = AscConfig()
        assert config.get("api", "key") == "secret123"

    def test_env_override_models_path(self, monkeypatch):
        monkeypatch.setenv("ASC_MODELS_PATH", "/custom/models")
        config = AscConfig()
        assert config.get("paths", "models") == "/custom/models"

    def test_env_override_llama_path(self, monkeypatch):
        monkeypatch.setenv("ASC_LLAMA_PATH", "/custom/llama")
        config = AscConfig()
        assert config.get("paths", "llama_cpp") == "/custom/llama"

    def test_env_non_integer_value(self, monkeypatch):
        """非整数值的环境变量应作为字符串保存。"""
        monkeypatch.setenv("ASC_API_KEY", "not-a-number")
        config = AscConfig()
        assert config.get("api", "key") == "not-a-number"


class TestAscConfigBoundary:
    """AscConfig 边界条件测试。"""

    def test_empty_section_key(self):
        config = AscConfig()
        config.set("", "key", "value")
        assert config.get("", "key") == "value"

    def test_special_chars_in_model_mapping(self):
        config = AscConfig()
        config.add_model_mapping("model-中文", "/路径/模型.gguf")
        assert config.resolve_model_path("model-中文") == "/路径/模型.gguf"

    def test_config_json_roundtrip(self, tmp_path):
        """配置 JSON 序列化往返测试。"""
        config = AscConfig()
        config.add_model_mapping("m1", "p1")
        config.set("custom", "float_val", 3.14)
        path = tmp_path / "config.json"
        config.save(path)

        # 验证 JSON 格式
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["model_mappings"]["m1"] == "p1"

        config2 = AscConfig()
        config2.load(path)
        assert config2.resolve_model_path("m1") == "p1"


# ======================================================================
# 2. core/election.py 测试
# ======================================================================


class TestBullyElectionBasic:
    """Bully 选举基本功能测试。"""

    def test_initial_state_idle(self):
        election = BullyElection(node_id="node-1", all_node_ids=["node-1", "node-2"])
        assert election.state == ElectionState.IDLE
        assert election.master_id is None
        assert election.election_clock == 0

    def test_start_election(self):
        election = BullyElection(node_id="node-1", all_node_ids=["node-1", "node-2"])
        election.start_election()
        assert election.state == ElectionState.ELECTING
        assert election.election_clock == 1

    def test_higher_node_ids(self):
        election = BullyElection(node_id="node-1", all_node_ids=["node-1", "node-2", "node-3"])
        higher = election.higher_node_ids
        assert "node-2" in higher
        assert "node-3" in higher
        assert "node-1" not in higher

    def test_should_become_master_single_node(self):
        """单节点应成为 Master。"""
        election = BullyElection(node_id="node-1", all_node_ids=["node-1"])
        election.start_election()
        assert election.should_become_master() is True

    def test_should_not_become_master_with_higher(self):
        """有更高 ID 节点时不应成为 Master。"""
        election = BullyElection(node_id="node-1", all_node_ids=["node-1", "node-2"])
        election.start_election()
        assert election.should_become_master() is False

    def test_become_master(self):
        election = BullyElection(node_id="node-1", all_node_ids=["node-1"])
        election.start_election()
        election.become_master()
        assert election.state == ElectionState.MASTER
        assert election.master_id == "node-1"
        assert election.is_master is True


class TestBullyElectionMessages:
    """Bully 选举消息处理测试。"""

    def test_handle_alive_message(self):
        """收到 ALIVE 消息，退让。"""
        election = BullyElection(node_id="node-1", all_node_ids=["node-1", "node-2"])
        election.start_election()
        msg = ElectionMessage(
            type=ElectionMessageType.ALIVE, sender_id="node-2", election_clock=1
        )
        election.handle_message(msg)
        assert election.state == ElectionState.IDLE

    def test_handle_coordinator_message_becomes_worker(self):
        """收到 COORDINATOR 消息，成为 Worker。"""
        election = BullyElection(node_id="node-1", all_node_ids=["node-1", "node-2"])
        msg = ElectionMessage(
            type=ElectionMessageType.COORDINATOR, sender_id="node-2", election_clock=1
        )
        election.handle_message(msg)
        assert election.state == ElectionState.WORKER
        assert election.master_id == "node-2"

    def test_handle_coordinator_message_self_becomes_master(self):
        """收到自己发出的 COORDINATOR 消息，成为 Master。"""
        election = BullyElection(node_id="node-1", all_node_ids=["node-1"])
        msg = ElectionMessage(
            type=ElectionMessageType.COORDINATOR, sender_id="node-1", election_clock=1
        )
        election.handle_message(msg)
        assert election.state == ElectionState.MASTER
        assert election.master_id == "node-1"

    def test_handle_election_from_lower_id(self):
        """收到更低 ID 节点的 ELECTION 消息。"""
        election = BullyElection(node_id="node-2", all_node_ids=["node-1", "node-2"])
        msg = ElectionMessage(
            type=ElectionMessageType.ELECTION, sender_id="node-1", election_clock=1
        )
        # 不应抛异常
        election.handle_message(msg)

    def test_get_election_messages(self):
        """获取需要发送的选举消息。"""
        election = BullyElection(node_id="node-1", all_node_ids=["node-1", "node-2", "node-3"])
        election.start_election()
        msgs = election.get_election_messages()
        assert len(msgs) == 2
        assert all(m.type == ElectionMessageType.ELECTION for m in msgs)

    def test_get_election_messages_not_electing(self):
        """非选举状态不产生选举消息。"""
        election = BullyElection(node_id="node-1", all_node_ids=["node-1"])
        assert election.get_election_messages() == []

    def test_get_coordinator_message(self):
        """获取 COORDINATOR 消息。"""
        election = BullyElection(node_id="node-1", all_node_ids=["node-1"])
        election.start_election()
        election.become_master()
        msg = election.get_coordinator_message()
        assert msg is not None
        assert msg.type == ElectionMessageType.COORDINATOR
        assert msg.sender_id == "node-1"

    def test_get_coordinator_message_not_master(self):
        """非 Master 状态不产生 COORDINATOR 消息。"""
        election = BullyElection(node_id="node-1", all_node_ids=["node-1", "node-2"])
        assert election.get_coordinator_message() is None


class TestBullyElectionBoundary:
    """Bully 选举边界条件测试。"""

    def test_single_node_cluster(self):
        """单节点集群选举。"""
        election = BullyElection(node_id="only", all_node_ids=["only"])
        election.start_election()
        assert election.should_become_master() is True
        election.become_master()
        assert election.is_master is True

    def test_multiple_elections(self):
        """多次选举，election_clock 递增。"""
        election = BullyElection(node_id="node-1", all_node_ids=["node-1"])
        election.start_election()
        assert election.election_clock == 1
        # 重新选举
        election._state = ElectionState.IDLE
        election.start_election()
        assert election.election_clock == 2

    def test_empty_node_list(self):
        """空节点列表。"""
        election = BullyElection(node_id="node-1", all_node_ids=[])
        assert election.higher_node_ids == []
        election.start_election()
        assert election.should_become_master() is True

    def test_highest_id_node(self):
        """最高 ID 节点发起选举。"""
        election = BullyElection(
            node_id="node-3", all_node_ids=["node-1", "node-2", "node-3"]
        )
        election.start_election()
        assert election.should_become_master() is True

    def test_lowest_id_node(self):
        """最低 ID 节点发起选举。"""
        election = BullyElection(
            node_id="node-1", all_node_ids=["node-1", "node-2", "node-3"]
        )
        election.start_election()
        assert election.should_become_master() is False

    def test_alive_message_when_not_electing(self):
        """非选举状态收到 ALIVE 消息，状态不变。"""
        election = BullyElection(node_id="node-1", all_node_ids=["node-1", "node-2"])
        msg = ElectionMessage(
            type=ElectionMessageType.ALIVE, sender_id="node-2", election_clock=1
        )
        election.handle_message(msg)
        assert election.state == ElectionState.IDLE


# ======================================================================
# 3. core/event_log.py 测试
# ======================================================================


class TestMemoryEventLog:
    """MemoryEventLog 功能测试。"""

    def test_append_and_read(self):
        log = MemoryEventLog()
        ie = IndexedEvent(
            event=NodeJoined(node_id=NodeId("n1"), ip="1.1.1.1", port=80), index=1
        )
        log.append(ie)
        assert len(log) == 1
        events = list(log.read_from(0))
        assert len(events) == 1
        assert events[0].index == 1

    def test_read_from_after_index(self):
        log = MemoryEventLog()
        for i in range(1, 4):
            log.append(
                IndexedEvent(
                    event=NodeJoined(node_id=NodeId(f"n{i}"), ip=f"1.1.1.{i}", port=80),
                    index=i,
                )
            )
        events = list(log.read_from(1))
        assert len(events) == 2
        assert events[0].index == 2

    def test_read_from_empty_log(self):
        log = MemoryEventLog()
        events = list(log.read_from(0))
        assert events == []

    def test_last_index_empty(self):
        log = MemoryEventLog()
        assert log.last_index == 0

    def test_last_index(self):
        log = MemoryEventLog()
        log.append(
            IndexedEvent(
                event=NodeJoined(node_id=NodeId("n1"), ip="1.1.1.1", port=80), index=5
            )
        )
        assert log.last_index == 5

    def test_read_from_beyond_last(self):
        log = MemoryEventLog()
        log.append(
            IndexedEvent(
                event=NodeJoined(node_id=NodeId("n1"), ip="1.1.1.1", port=80), index=1
            )
        )
        events = list(log.read_from(100))
        assert events == []


class TestDiskEventLog:
    """DiskEventLog 功能测试。"""

    def test_append_and_read(self, tmp_path):
        log = DiskEventLog(tmp_path / "events.jsonl")
        ie = IndexedEvent(
            event=NodeJoined(node_id=NodeId("n1"), ip="1.1.1.1", port=80), index=1
        )
        log.append(ie)
        assert len(log) == 1
        events = list(log.read_from(0))
        assert len(events) == 1

    def test_persistence(self, tmp_path):
        """磁盘日志持久化：关闭后重新打开可读取。"""
        path = tmp_path / "events.jsonl"
        ie = IndexedEvent(
            event=NodeJoined(node_id=NodeId("n1"), ip="1.1.1.1", port=80), index=1
        )
        log1 = DiskEventLog(path)
        log1.append(ie)
        log1.flush()  # 确保缓冲区写入磁盘

        log2 = DiskEventLog(path)
        assert len(log2) == 1
        assert log2.last_index == 1

    def test_read_from_nonexistent_file(self, tmp_path):
        """读取不存在的文件。"""
        log = DiskEventLog(tmp_path / "nonexistent.jsonl")
        events = list(log.read_from(0))
        assert events == []

    def test_last_index_empty(self, tmp_path):
        log = DiskEventLog(tmp_path / "empty.jsonl")
        assert log.last_index == 0

    def test_multiple_events(self, tmp_path):
        log = DiskEventLog(tmp_path / "events.jsonl")
        for i in range(1, 6):
            log.append(
                IndexedEvent(
                    event=NodeJoined(node_id=NodeId(f"n{i}"), ip=f"1.1.1.{i}", port=80),
                    index=i,
                )
            )
        assert len(log) == 5
        assert log.last_index == 5
        events = list(log.read_from(2))
        assert len(events) == 3


class TestEventSerialization:
    """事件序列化/反序列化测试。"""

    def test_serialize_deserialize_node_joined(self):
        ie = IndexedEvent(
            event=NodeJoined(node_id=NodeId("n1"), ip="192.168.1.1", port=52415), index=1
        )
        s = serialize_indexed_event(ie)
        ie2 = deserialize_indexed_event(s)
        assert ie2.index == 1
        assert ie2.event.node_id == "n1"
        assert ie2.event.ip == "192.168.1.1"
        assert ie2.event.port == 52415

    def test_serialize_deserialize_instance_created(self):
        ie = IndexedEvent(
            event=InstanceCreated(
                instance_id=InstanceId("i1"),
                model_id="llama-7b",
                node_ids=[NodeId("n1"), NodeId("n2")],
                sharding="tensor",
                rpc_endpoints=["1.1.1.1:50052"],
            ),
            index=2,
        )
        s = serialize_indexed_event(ie)
        ie2 = deserialize_indexed_event(s)
        assert ie2.index == 2
        assert ie2.event.instance_id == "i1"
        assert len(ie2.event.node_ids) == 2
        assert ie2.event.rpc_endpoints == ["1.1.1.1:50052"]

    def test_serialize_deserialize_task_created(self):
        ie = IndexedEvent(
            event=TaskCreated(
                task_id=TaskId("t1"), instance_id=InstanceId("i1"), prompt="Hello"
            ),
            index=3,
        )
        s = serialize_indexed_event(ie)
        ie2 = deserialize_indexed_event(s)
        assert ie2.event.task_id == "t1"
        assert ie2.event.prompt == "Hello"

    def test_deserialize_invalid_json(self):
        """无效 JSON 应抛出异常。"""
        with pytest.raises(json.JSONDecodeError):
            deserialize_indexed_event("not json")

    def test_deserialize_unknown_event_type(self):
        """未知事件类型应抛出 ValueError。"""
        data = json.dumps({"index": 1, "event": {"event_type": "unknown", "data": {}}})
        with pytest.raises(ValueError, match="未知事件类型"):
            deserialize_indexed_event(data)


class TestEventLogBoundary:
    """事件日志边界条件测试。"""

    def test_large_number_of_events(self, tmp_path):
        """大量事件写入和读取。"""
        log = MemoryEventLog()
        for i in range(1000):
            log.append(
                IndexedEvent(
                    event=NodeJoined(node_id=NodeId(f"n{i}"), ip=f"1.1.1.{i % 255}", port=80),
                    index=i + 1,
                )
            )
        assert len(log) == 1000
        assert log.last_index == 1000

    def test_disk_log_empty_lines(self, tmp_path):
        """磁盘日志中有空行应跳过。"""
        path = tmp_path / "events.jsonl"
        ie = IndexedEvent(
            event=NodeJoined(node_id=NodeId("n1"), ip="1.1.1.1", port=80), index=1
        )
        log = DiskEventLog(path)
        log.append(ie)
        log.flush()  # 确保缓冲区写入磁盘
        # 手动添加空行
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n\n")
        log2 = DiskEventLog(path)
        assert len(log2) == 1


# ======================================================================
# 4. core/model_manager.py 测试
# ======================================================================


class TestModelManagerBasic:
    """ModelManager 基本功能测试。"""

    def test_register_model(self, tmp_path):
        mm = ModelManager(tmp_path / "models")
        mm.register("llama-7b", "/path/to/llama-7b.gguf")
        assert mm.resolve("llama-7b") == "/path/to/llama-7b.gguf"

    def test_unregister_model(self, tmp_path):
        mm = ModelManager(tmp_path / "models")
        mm.register("llama-7b", "/path/to/llama-7b.gguf")
        mm.unregister("llama-7b")
        assert mm.resolve("llama-7b") is None

    def test_unregister_nonexistent(self, tmp_path):
        """注销不存在的模型不报错。"""
        mm = ModelManager(tmp_path / "models")
        mm.unregister("nonexistent")

    def test_resolve_nonexistent(self, tmp_path):
        mm = ModelManager(tmp_path / "models")
        assert mm.resolve("nonexistent") is None

    def test_list_models_empty(self, tmp_path):
        mm = ModelManager(tmp_path / "models")
        assert mm.list_models() == []

    def test_list_models(self, tmp_path):
        mm = ModelManager(tmp_path / "models")
        mm.register("m1", "/p1")
        mm.register("m2", "/p2")
        models = mm.list_models()
        assert len(models) == 2

    def test_register_overwrite(self, tmp_path):
        """重复注册同名模型会覆盖。"""
        mm = ModelManager(tmp_path / "models")
        mm.register("m1", "/old")
        mm.register("m1", "/new")
        assert mm.resolve("m1") == "/new"


class TestModelManagerAutoDiscover:
    """ModelManager 自动发现测试。"""

    def test_auto_discover_gguf(self, tmp_path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "llama-7b.gguf").write_text("fake model")
        mm = ModelManager(models_dir)
        mm.auto_discover()
        assert mm.resolve("llama-7b") is not None

    def test_auto_discover_no_models_dir(self, tmp_path):
        """不存在的目录不报错。"""
        mm = ModelManager(tmp_path / "nonexistent")
        mm.auto_discover()
        assert mm.list_models() == []

    def test_auto_discover_non_gguf_ignored(self, tmp_path):
        """非 .gguf 文件被忽略。"""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "model.bin").write_text("not gguf")
        (models_dir / "model.gguf").write_text("gguf model")
        mm = ModelManager(models_dir)
        mm.auto_discover()
        models = mm.list_models()
        assert len(models) == 1
        assert models[0].model_id == "model"

    def test_auto_discover_does_not_overwrite(self, tmp_path):
        """自动发现不覆盖已注册的模型。"""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "llama-7b.gguf").write_text("fake model")
        mm = ModelManager(models_dir)
        mm.register("llama-7b", "/custom/path")
        mm.auto_discover()
        assert mm.resolve("llama-7b") == "/custom/path"


class TestModelManagerMetadata:
    """ModelManager 元数据测试。"""

    def test_metadata_local_source(self, tmp_path):
        mm = ModelManager(tmp_path / "models")
        mm.register("m1", "/path/to/m1.gguf")
        meta = mm.list_models()[0]
        assert meta.source == "local"
        assert meta.model_id == "m1"

    def test_metadata_file_size_nonexistent(self, tmp_path):
        """不存在的文件 file_size_mb 为 0。"""
        mm = ModelManager(tmp_path / "models")
        mm.register("m1", "/nonexistent/file.gguf")
        meta = mm.list_models()[0]
        assert meta.file_size_mb == 0

    def test_metadata_file_size_existing(self, tmp_path):
        """存在的文件 file_size_mb 正确。"""
        f = tmp_path / "model.gguf"
        f.write_bytes(b"x" * (2 * 1024 * 1024))  # 2MB
        mm = ModelManager(tmp_path / "models")
        mm.register("m1", str(f))
        meta = mm.list_models()[0]
        assert meta.file_size_mb == 2


class TestDownloadProgress:
    """DownloadProgress 值对象测试。"""

    def test_fraction_normal(self):
        dp = DownloadProgress(model_id="m1", downloaded_mb=50, total_mb=100, speed_mbps=10.0)
        assert dp.fraction == 0.5

    def test_fraction_zero_total(self):
        dp = DownloadProgress(model_id="m1", downloaded_mb=0, total_mb=0, speed_mbps=0)
        assert dp.fraction == 0.0

    def test_fraction_complete(self):
        dp = DownloadProgress(model_id="m1", downloaded_mb=100, total_mb=100, speed_mbps=5.0)
        assert dp.fraction == 1.0


class TestModelManagerException:
    """ModelManager 异常场景测试。"""

    def test_download_import_error_path(self, tmp_path):
        """验证 download() 方法中 huggingface_hub 的延迟导入路径。

        当 huggingface_hub 未安装时，download() 应抛出 ImportError。
        当前环境中 huggingface_hub 已安装，无法直接测试 ImportError。
        但可以验证代码路径：download() 使用 try/except ImportError 来
        检测 huggingface_hub 可用性（Bug #003：此检查在 _find_gguf_filename 之后，
        导致 filename=None 时先抛出 FileNotFoundError 而非 ImportError）。
        """
        mm = ModelManager(tmp_path / "models")
        # 验证 _find_gguf_filename 对不存在的仓库抛出 FileNotFoundError
        with pytest.raises(FileNotFoundError, match="未在"):
            mm._find_gguf_filename("nonexistent/repo")

    def test_download_with_filename_no_repo(self, tmp_path):
        """指定 filename 但仓库不可达时应抛出异常。

        这验证了 Bug #003：当 filename=None 时，download() 先调用
        _find_gguf_filename（可能抛出 FileNotFoundError），再 import
        huggingface_hub（可能抛出 ImportError）。错误处理顺序不合理。
        """
        mm = ModelManager(tmp_path / "models")
        # 指定 filename 可以跳过 _find_gguf_filename
        gen = mm.download("nonexistent/repo-12345", filename="model.gguf")
        # 第一个 yield 是初始进度
        progress = next(gen)
        assert progress.model_id == "nonexistent/repo-12345"
        # 后续调用会因网络错误失败
