# Phase 8: 安全加固与生产就绪

## 目标

将系统从开发框架升级为生产可用，包括安全、监控、容错、性能优化。

## 现状

当前安全机制：
```python
def require_api_key(api_key: str | None = None) -> bool:
    expected = os.getenv("ASC_API_KEY")
    if not expected:
        return True  # 内网模式，跳过认证
    return api_key == expected
```

问题：
- 无 HTTPS
- 无 API 限流
- 无请求日志
- 无监控指标

## 需求

### 8.1 API 安全

#### HTTPS 支持
```python
class SecureServer:
    """HTTPS 服务器。"""
    
    def __init__(self, ssl_cert: str, ssl_key: str) -> None:
        self._ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        self._ssl_context.load_cert_chain(ssl_cert, ssl_key)
```

#### API 限流
```python
from slowapi import Limiter

limiter = Limiter(key_func=lambda: request.client.host)

@app.post("/v1/chat/completions")
@limiter.limit("10/minute")  # 每 IP 每分钟 10 次
def chat_completions(request):
    ...
```

#### 请求验证
- 输入长度限制（防止超长 prompt）
- 参数范围校验（temperature 0-2）
- 内容安全过滤（可选）

### 8.2 监控与日志

#### 结构化日志
```python
import structlog

logger = structlog.get_logger()

logger.info(
    "inference_complete",
    model="llama-3-8b",
    tokens_in=100,
    tokens_out=50,
    latency_ms=1200,
    node="worker-1",
)
```

#### 指标收集
```python
from prometheus_client import Counter, Histogram

inference_counter = Counter("asc_inference_total", "Total inferences", ["model"])
latency_histogram = Histogram("asc_inference_latency", "Inference latency")
```

#### 健康检查
```python
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "nodes": len(cluster.nodes),
        "models_loaded": len(model_manager.list_models()),
        "uptime_seconds": time.time() - start_time,
    }
```

### 8.3 容错与恢复

#### 节点故障检测
```python
class FailureDetector:
    """故障检测器。"""
    
    async def check_node(self, node_id: str) -> NodeHealth:
        """
        检测方式：
        1. 心跳超时（30秒无响应）
        2. TCP 连接探测
        3. 推理健康检查（发送测试请求）
        """
```

#### 自动故障转移
```python
class FailoverManager:
    """故障转移管理器。"""
    
    async def handle_node_failure(self, node_id: str) -> None:
        """
        处理流程：
        1. 标记节点为离线
        2. 迁移该节点上的推理任务
        3. 重新计算 tensor-split / pipeline
        4. 通知其他节点更新拓扑
        """
```

#### 优雅关闭
```python
async def graceful_shutdown():
    """
    1. 停止接收新请求
    2. 等待进行中的推理完成
    3. 保存状态
    4. 关闭连接
    5. 退出进程
    """
```

### 8.4 性能优化

#### 模型缓存
- 热点模型常驻内存
- LRU 缓存策略
- 预加载预测

#### 批处理
```python
class BatchProcessor:
    """请求批处理器。"""
    
    async def process_batch(self, requests: list[InferenceRequest]) -> None:
        """
        将多个请求合并为一批处理：
        - 相同模型的请求合并
        - 动态批处理窗口（最大等待 10ms）
        - 使用 llama-server 的并发槽位
        """
```

#### 连接池
- 复用与 llama-server 的 HTTP 连接
- 节点间 TCP 连接池

## 实现计划

1. **Day 1**: HTTPS + API 限流
2. **Day 2**: 结构化日志 + Prometheus 指标
3. **Day 3**: 故障检测 + 自动转移
4. **Day 4**: 性能优化（缓存、批处理）
5. **Day 5**: 压力测试 + 调优

## 文件变更

- `src/asc/api/security.py` - 新增安全模块
- `src/asc/core/monitoring.py` - 新增监控模块
- `src/asc/core/failover.py` - 新增故障转移
- `src/asc/engine/batch.py` - 新增批处理
- `tests/integration/test_production.py` - 生产环境测试

## 依赖

```toml
[project.optional-dependencies]
production = [
    "slowapi>=0.1.9",          # API 限流
    "structlog>=24.1.0",       # 结构化日志
    "prometheus-client>=0.19", # 指标
    "uvloop>=0.19.0",          # 高性能事件循环 (Linux)
]
```

## 部署检查清单

- [ ] SSL 证书配置
- [ ] API Key 设置
- [ ] 日志目录权限
- [ ] 防火墙端口开放
- [ ] 监控告警配置
- [ ] 备份策略
- [ ] 灾难恢复预案
