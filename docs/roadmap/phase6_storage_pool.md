# Phase 6: 统一存储池与模型同步

## 目标

构建分布式模型存储池，实现模型文件在节点间的自动同步，使任意节点都能获取所需模型。

## 现状

当前 `ModelManager` 仅支持本地模型：
```python
class ModelManager:
    def auto_discover(self) -> None:
        """仅扫描本地 models 目录。"""
        for gguf_file in self._models_dir.glob("*.gguf"):
            ...
```

## 需求

### 6.1 分布式模型注册表

```python
class ModelRegistry:
    """集群级模型注册表。"""
    
    def register(self, node_id: str, model_id: str, path: str) -> None:
        """注册节点上的模型。"""
        
    def find_model(self, model_id: str) -> list[ModelLocation]:
        """查找模型在哪些节点上可用。"""
        
    def get_best_source(self, model_id: str, requester: str) -> ModelLocation:
        """选择最佳下载源（最近/最快）。"""

@dataclass
class ModelLocation:
    node_id: str
    path: str
    size_mb: int
    network_latency_ms: float
```

### 6.2 模型同步协议

```python
class ModelSyncProtocol:
    """模型同步协议。"""
    
    async def request_model(
        self,
        model_id: str,
        source_node: str,
    ) -> None:
        """从源节点请求同步模型。"""
        
    async def send_model_chunk(
        self,
        model_id: str,
        chunk_index: int,
        chunk_data: bytes,
    ) -> None:
        """发送模型分片。"""
```

### 6.3 分片下载与校验

- 模型文件分片（每片 10MB）
- 断点续传支持
- SHA256 校验完整性
- 下载进度实时上报

### 6.4 存储策略

| 策略 | 说明 |
|------|------|
| 全量复制 | 每个节点存储所有模型（小模型） |
| 按需加载 | 需要时从其他节点下载 |
| 智能缓存 | LRU 缓存，自动清理不常用模型 |

### 6.5 模型预热

- Master 预先将模型分发到需要的 Worker
- 推理开始前确保模型已在各节点就绪
- 支持后台预加载

## 实现计划

1. **Day 1-2**: 实现 ModelRegistry 和模型定位
2. **Day 3-4**: 实现分片下载协议
3. **Day 5**: 实现断点续传和校验
4. **Day 6**: 集成到 Master 启动流程

## 文件变更

- `src/asc/core/model_registry.py` - 新增分布式注册表
- `src/asc/network/sync.py` - 新增同步协议
- `src/asc/core/model_manager.py` - 扩展支持远程模型
- `tests/core/test_model_registry.py` - 新增测试

## 关键技术点

1. **带宽控制**：避免同步占用全部网络带宽
2. **并发下载**：多节点同时传输不同分片
3. **存储配额**：每个节点设置最大存储空间
4. **模型版本管理**：支持多版本共存
