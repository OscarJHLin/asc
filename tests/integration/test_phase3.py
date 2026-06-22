"""Phase 3 集成测试：验证生产化模块的完整协作。

测试 API 适配器 + Master 主循环 + Discovery + Pipeline 的端到端流程。
"""


from asc.api.anthropic_adapter import AnthropicAdapter, AnthropicRequest
from asc.api.ollama_adapter import OllamaAdapter, OllamaChatRequest
from asc.api.openai_adapter import ChatCompletionRequest, OpenAIAdapter
from asc.master.main import MasterNode
from asc.network.discovery import DiscoveryMessage, NodeDiscovery
from asc.scheduler.pipeline import PipelinePlanner
from asc.scheduler.placement import PlacementEngine, PlacementStrategy
from asc.scheduler.topology import build_topology
from asc.types import (
    RunnerStatusUpdated,
    TaskCreated,
)
from asc.types.commands import CreateInstance, StartInference
from asc.worker.agent import NodeResources
from asc.worker.gpu_info import GPUInfo


class TestAPIAdaptersWithMaster:
    """API 适配器与 Master 协作：请求转换 -> 事件溯源 -> 响应生成。"""

    def test_openai_request_to_inference(self):
        """OpenAI 请求 -> Master 处理 -> 响应。"""
        # 1. Master 处理节点加入
        master = MasterNode(node_id="master")
        master.process_node_joined("n1", "10.0.0.1", 52415)
        master.process_node_joined("n2", "10.0.0.2", 52415)
        master._emit(RunnerStatusUpdated(node_id="n1", status="ready"))
        master._emit(RunnerStatusUpdated(node_id="n2", status="ready"))

        # 2. 创建实例
        cmd = CreateInstance(model_id="llama-3.1-8b", sharding="tensor")
        master.process_create_instance(cmd)

        # 3. OpenAI 适配器格式化请求
        adapter = OpenAIAdapter(model_mappings={"llama-3.1-8b": "/models/llama.gguf"})
        req = ChatCompletionRequest(
            model="llama-3.1-8b",
            messages=[{"role": "user", "content": "Hello"}],
        )
        prompt = adapter.format_prompt(req)
        assert "Hello" in prompt

        # 4. Master 处理推理
        inst_id = list(master.state.instances.keys())[0]
        inf_cmd = StartInference(instance_id=inst_id, prompt=prompt)
        events = master.process_start_inference(inf_cmd)
        task_id = None
        for e in events:
            if isinstance(e, TaskCreated):
                task_id = e.task_id
        assert task_id is not None

        # 5. 适配器生成响应
        resp = adapter.create_response(
            req, output="Hi there!", prompt_tokens=5, completion_tokens=3
        )
        d = resp.to_dict()
        assert d["choices"][0]["message"]["content"] == "Hi there!"

    def test_ollama_request_to_inference(self):
        """Ollama 请求 -> Master 处理 -> 响应。"""
        master = MasterNode(node_id="master")
        master.process_node_joined("n1", "10.0.0.1", 52415)
        master._emit(RunnerStatusUpdated(node_id="n1", status="ready"))

        cmd = CreateInstance(model_id="llama3", sharding="tensor")
        master.process_create_instance(cmd)

        adapter = OllamaAdapter(model_mappings={"llama3": "/models/llama.gguf"})
        req = OllamaChatRequest(
            model="llama3",
            messages=[{"role": "user", "content": "Hello"}],
        )
        prompt = adapter.format_prompt_from_chat(req)
        assert "Hello" in prompt

        resp = adapter.create_chat_response(req, output="Hi!", prompt_eval_count=3, eval_count=5)
        assert resp.message["content"] == "Hi!"


class TestModelRegistrationWithMaster:
    """模型注册与 Master 协作。"""

    def test_model_instance_creation(self):
        """Master 创建实例。"""
        master = MasterNode(node_id="master")
        master.process_node_joined("n1", "10.0.0.1", 52415)
        master._emit(RunnerStatusUpdated(node_id="n1", status="ready"))

        cmd = CreateInstance(model_id="llama-3.1-8b", sharding="tensor")
        master.process_create_instance(cmd)

        # 验证实例创建
        assert len(master.state.instances) == 1


class TestDiscoveryWithTopology:
    """节点发现与拓扑构建协作。"""

    def test_discovered_nodes_build_topology(self):
        """发现节点 -> 构建拓扑 -> 放置。"""
        disc = NodeDiscovery(node_id="master", port=52415)

        # 模拟收到发现消息
        msg = DiscoveryMessage(node_id="worker-1", ip="10.0.0.2", port=52415)
        raw = msg.to_json().encode()
        parsed = disc.parse_message(raw)
        assert parsed is not None
        assert parsed.node_id == "worker-1"

        # 构建拓扑
        resources = {
            "master": NodeResources(
                cpu_count=8,
                cpu_percent=25.0,
                memory_total_mb=32768,
                memory_free_mb=24000,
                gpus=[GPUInfo(index=0, name="RTX 4090", vram_total_mb=24564, vram_free_mb=8000)],
            ),
            parsed.node_id: NodeResources(
                cpu_count=4,
                cpu_percent=10.0,
                memory_total_mb=16384,
                memory_free_mb=12000,
                gpus=[GPUInfo(index=0, name="RTX 3060", vram_total_mb=12288, vram_free_mb=6000)],
            ),
        }
        topo = build_topology("master", resources, {parsed.node_id: f"{parsed.ip}:{parsed.port}"})

        # 放置
        engine = PlacementEngine()
        result = engine.place(model_vram_required_mb=12000, topology=topo)
        assert result.success


class TestPipelineWithPlacement:
    """Pipeline 分片与放置算法协作。"""

    def test_pipeline_placement_workflow(self):
        """放置 -> Pipeline 分片 -> 创建实例。"""
        # 1. 构建拓扑
        resources = {
            "n1": NodeResources(
                cpu_count=8,
                cpu_percent=20.0,
                memory_total_mb=32768,
                memory_free_mb=24000,
                gpus=[GPUInfo(index=0, name="RTX 4090", vram_total_mb=24564, vram_free_mb=12000)],
            ),
            "n2": NodeResources(
                cpu_count=4,
                cpu_percent=10.0,
                memory_total_mb=16384,
                memory_free_mb=12000,
                gpus=[GPUInfo(index=0, name="RTX 3060", vram_total_mb=12288, vram_free_mb=8000)],
            ),
        }
        topo = build_topology("n1", resources, {"n2": "10.0.0.2:52415"})

        # 2. 放置（Pipeline 策略）
        engine = PlacementEngine()
        result = engine.place(
            model_vram_required_mb=16000,
            topology=topo,
            strategy=PlacementStrategy.PIPELINE,
        )
        assert result.success
        assert len(result.selected_nodes) == 2

        # 3. Pipeline 分片
        planner = PipelinePlanner()
        plan = planner.plan_by_vram(
            total_layers=32,
            node_vram_mb={"n1": 12000, "n2": 8000},
        )
        assert plan.is_valid
        assert sum(s.num_layers for s in plan.stages) == 32

        # 4. 创建实例（事件溯源）
        master = MasterNode(node_id="n1")
        for nid in result.selected_nodes:
            master.process_node_joined(nid, f"10.0.0.{nid[-1]}", 52415)
            master._emit(RunnerStatusUpdated(node_id=nid, status="ready"))

        cmd = CreateInstance(model_id="llama-3.1-8b", sharding="pipeline")
        master.process_create_instance(cmd)

        # 验证实例
        inst = list(master.state.instances.values())[0]
        assert inst.sharding == "pipeline"


class TestEndToEndMultiAPI:
    """多 API 适配器端到端测试。"""

    def test_same_inference_different_api_formats(self):
        """同一推理结果通过不同 API 格式返回。"""
        output = "The answer is 42."

        # OpenAI 格式
        openai = OpenAIAdapter(model_mappings={"m": "/m.gguf"})
        openai_req = ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "Q"}])
        openai_resp = openai.create_response(
            openai_req, output=output, prompt_tokens=10, completion_tokens=5
        )
        assert openai_resp.to_dict()["choices"][0]["message"]["content"] == output

        # Ollama 格式
        ollama = OllamaAdapter(model_mappings={"m": "/m.gguf"})
        ollama_req = OllamaChatRequest(model="m", messages=[{"role": "user", "content": "Q"}])
        ollama_resp = ollama.create_chat_response(
            ollama_req, output=output, prompt_eval_count=10, eval_count=5
        )
        assert ollama_resp.message["content"] == output

        # Anthropic 格式
        anthropic = AnthropicAdapter()
        anthropic_req = AnthropicRequest(model="m", messages=[{"role": "user", "content": "Q"}])
        anthropic_resp = anthropic.create_response(anthropic_req, output=output)
        assert anthropic_resp.content[0].text == output
