"""测试 Pipeline Parallel 分片。

Pipeline Parallel 将模型按层切分到不同节点，
每个节点负责一部分层的计算，形成流水线。
"""

from asc.scheduler.pipeline import (
    PipelinePlan,
    PipelinePlanner,
    PipelineStage,
)


class TestPipelineStage:
    """Pipeline 阶段。"""

    def test_create(self):
        stage = PipelineStage(
            node_id="node-1",
            start_layer=0,
            end_layer=15,
            num_layers=16,
        )
        assert stage.node_id == "node-1"
        assert stage.start_layer == 0
        assert stage.end_layer == 15
        assert stage.num_layers == 16

    def test_layer_count(self):
        stage = PipelineStage(node_id="n1", start_layer=0, end_layer=9, num_layers=10)
        assert stage.num_layers == 10


class TestPipelinePlan:
    """Pipeline 计划。"""

    def test_create(self):
        stages = [
            PipelineStage(node_id="n1", start_layer=0, end_layer=9, num_layers=10),
            PipelineStage(node_id="n2", start_layer=10, end_layer=19, num_layers=10),
        ]
        plan = PipelinePlan(stages=stages, total_layers=20)
        assert len(plan.stages) == 2
        assert plan.total_layers == 20

    def test_total_layers_coverage(self):
        """所有阶段应覆盖完整层范围。"""
        stages = [
            PipelineStage(node_id="n1", start_layer=0, end_layer=9, num_layers=10),
            PipelineStage(node_id="n2", start_layer=10, end_layer=19, num_layers=10),
        ]
        plan = PipelinePlan(stages=stages, total_layers=20)
        assert plan.stages[0].start_layer == 0
        assert plan.stages[-1].end_layer == 19

    def test_is_valid(self):
        stages = [
            PipelineStage(node_id="n1", start_layer=0, end_layer=9, num_layers=10),
            PipelineStage(node_id="n2", start_layer=10, end_layer=19, num_layers=10),
        ]
        plan = PipelinePlan(stages=stages, total_layers=20)
        assert plan.is_valid

    def test_is_invalid_gap(self):
        """阶段间有间隙应无效。"""
        stages = [
            PipelineStage(node_id="n1", start_layer=0, end_layer=8, num_layers=9),
            PipelineStage(node_id="n2", start_layer=10, end_layer=19, num_layers=10),
        ]
        plan = PipelinePlan(stages=stages, total_layers=20)
        assert not plan.is_valid


class TestPipelinePlanner:
    """Pipeline 分片规划器。"""

    def test_single_node(self):
        """单节点应将所有层分配到该节点。"""
        planner = PipelinePlanner()
        plan = planner.plan_by_vram(
            total_layers=32,
            node_vram_mb={"node-1": 8000},
        )
        assert len(plan.stages) == 1
        assert plan.stages[0].num_layers == 32
        assert plan.stages[0].node_id == "node-1"

    def test_two_nodes_equal_vram(self):
        """两节点 VRAM 相等应均分层。"""
        planner = PipelinePlanner()
        plan = planner.plan_by_vram(
            total_layers=32,
            node_vram_mb={"node-1": 8000, "node-2": 8000},
        )
        assert len(plan.stages) == 2
        assert plan.stages[0].num_layers == 16
        assert plan.stages[1].num_layers == 16

    def test_two_nodes_unequal_vram(self):
        """两节点 VRAM 不等应按比例分。"""
        planner = PipelinePlanner()
        plan = planner.plan_by_vram(
            total_layers=32,
            node_vram_mb={"node-1": 12000, "node-2": 4000},
        )
        assert len(plan.stages) == 2
        # 12000:4000 = 3:1, node-1 应得 24 层, node-2 应得 8 层
        assert plan.stages[0].num_layers == 24
        assert plan.stages[1].num_layers == 8

    def test_three_nodes(self):
        """三节点按比例分配。"""
        planner = PipelinePlanner()
        plan = planner.plan_by_vram(
            total_layers=30,
            node_vram_mb={"n1": 5000, "n2": 3000, "n3": 2000},
        )
        assert len(plan.stages) == 3
        # 5000:3000:2000 = 5:3:2, 总30层 -> 15:9:6
        assert plan.stages[0].num_layers == 15
        assert plan.stages[1].num_layers == 9
        assert plan.stages[2].num_layers == 6

    def test_layers_sum_to_total(self):
        """所有阶段层数之和应等于总层数。"""
        planner = PipelinePlanner()
        plan = planner.plan_by_vram(
            total_layers=33,
            node_vram_mb={"n1": 10000, "n2": 6000, "n3": 4000},
        )
        total = sum(s.num_layers for s in plan.stages)
        assert total == 33

    def test_stages_contiguous(self):
        """阶段应连续覆盖所有层。"""
        planner = PipelinePlanner()
        plan = planner.plan_by_vram(
            total_layers=32,
            node_vram_mb={"n1": 8000, "n2": 8000},
        )
        # 第一阶段从 0 开始
        assert plan.stages[0].start_layer == 0
        # 后续阶段的 start = 前一阶段的 end + 1
        for i in range(1, len(plan.stages)):
            assert plan.stages[i].start_layer == plan.stages[i - 1].end_layer + 1

    def test_empty_nodes(self):
        """无节点应返回空计划。"""
        planner = PipelinePlanner()
        plan = planner.plan_by_vram(total_layers=32, node_vram_mb={})
        assert len(plan.stages) == 0
        assert not plan.is_valid

    def test_zero_vram_node_excluded(self):
        """VRAM 为 0 的节点应被排除。"""
        planner = PipelinePlanner()
        plan = planner.plan_by_vram(
            total_layers=32,
            node_vram_mb={"n1": 8000, "n2": 0},
        )
        assert len(plan.stages) == 1
        assert plan.stages[0].node_id == "n1"

    def test_plan_by_compute(self):
        """按算力比例分配层数。"""
        planner = PipelinePlanner()
        plan = planner.plan_by_compute(
            total_layers=30,
            node_compute_scores={"n1": 100.0, "n2": 50.0},
        )
        assert len(plan.stages) == 2
        assert plan.stages[0].num_layers == 20
        assert plan.stages[1].num_layers == 10
