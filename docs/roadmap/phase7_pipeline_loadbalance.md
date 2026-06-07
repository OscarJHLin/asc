# Phase 7: Pipeline 并行与动态负载均衡

## 目标

实现 Pipeline 并行推理和动态负载均衡，最大化集群算力利用率。

## 现状

当前仅支持张量分割（Tensor Split）：
```python
class TensorSplitCalculator:
    def calculate(self, local_vram, workers_vram) -> TensorSplitResult:
        # 仅按 VRAM 比例分配
```

Pipeline 并行在 `pipeline.py` 中仅为占位：
```python
class PipelineSplitCalculator:
    def calculate(self, num_layers, nodes) -> PipelineSplitResult:
        """TODO: 实现层分配算法"""
```

## 核心原理

### Pipeline 并行 vs Tensor 并行

| 维度 | Tensor 并行 | Pipeline 并行 |
|------|------------|---------------|
| 分割单位 | 每层内的张量 | 模型的层 |
| 通信量 | 大（每层的激活值） | 小（层间的隐藏状态） |
| 适用场景 | 单节点多 GPU | 多节点网络通信 |
| 效率 | 高（低延迟） | 中（需填充气泡） |

### Pipeline 执行流程

```
Node A (Layers 0-10) -> Node B (Layers 11-20) -> Node C (Layers 21-30)
   Input -> Forward -> Hidden -> Forward -> Hidden -> Forward -> Output
```

## 需求

### 7.1 层分配算法

```python
class PipelineSplitCalculator:
    def calculate(
        self,
        num_layers: int,
        nodes: list[NodeInfo],
    ) -> PipelineSplitResult:
        """
        分配策略：
        1. 按各节点算力比例分配层数
        2. 优先将连续层放在同一节点（减少通信）
        3. 考虑网络拓扑（相邻层放在网络近的节点）
        """
```

### 7.2 动态负载均衡

```python
class LoadBalancer:
    """动态负载均衡器。"""
    
    def rebalance(self, cluster_state: ClusterState) -> RebalancePlan:
        """
        触发条件：
        - 新节点加入
        - 节点负载不均（某些节点排队过长）
        - 节点性能变化
        
        策略：
        - 迁移部分层到空闲节点
        - 调整请求路由权重
        """
```

### 7.3 请求调度

```python
class RequestScheduler:
    """请求调度器。"""
    
    def schedule(self, request: InferenceRequest) -> NodeSelection:
        """
        调度策略：
        1. 轮询（Round Robin）
        2. 最少连接（Least Connections）
        3. 算力加权（Compute Score Weighted）
        4. 局部性优先（模型已在节点上）
        """
```

### 7.4 流水线气泡填充

```python
class PipelineExecutor:
    """Pipeline 执行器。"""
    
    async def execute_batch(self, requests: list[InferenceRequest]) -> None:
        """
        使用 micro-batch 填充流水线气泡：
        
        Time ->
        Node A: [Req1 L0-10] [Req2 L0-10] [Req3 L0-10]
        Node B:          [Req1 L11-20] [Req2 L11-20] [Req3 L11-20]
        Node C:                   [Req1 L21-30] [Req2 L21-30] [Req3 L21-30]
        """
```

## 实现计划

1. **Day 1-2**: 实现 PipelineSplitCalculator 层分配算法
2. **Day 3-4**: 实现 PipelineExecutor 流水线执行
3. **Day 5-6**: 实现 LoadBalancer 动态负载均衡
4. **Day 7**: 集成测试（三节点 Pipeline 并行）

## 文件变更

- `src/asc/scheduler/pipeline.py` - 实现 Pipeline 分割和执行
- `src/asc/scheduler/load_balancer.py` - 新增负载均衡器
- `src/asc/scheduler/request_scheduler.py` - 新增请求调度器
- `src/asc/master/main.py` - 集成调度逻辑
- `tests/scheduler/test_pipeline_execution.py` - 新增测试

## 关键技术点

1. **层分配公平性**：算力强的节点分配更多层
2. **通信优化**：相邻层尽量放在同一节点或网络近的节点
3. **气泡最小化**：使用 micro-batch 提高流水线效率
4. **动态调整**：运行时根据负载重新分配
