# Phase 5: 分布式推理引擎（RPC 模式）

## 目标

实现真正的分布式推理：多节点协同运行单个大模型，通过 llama-server 的 RPC 模式将模型层分布到不同节点。

## 现状

当前 `LlamaServerEngine` 仅支持单节点：
```python
class LlamaServerEngine:
    def _build_cmd(self) -> list[str]:
        return [
            str(self._exe_path),
            "--model", self._model_path,
            "--port", str(self._port),
            # 缺少 --rpc 参数
        ]
```

## 核心原理

llama.cpp 的 RPC 模式：
```bash
# Worker 节点启动 RPC 服务端
llama-server --rpc-server-bind 0.0.0.0:50052

# Master 节点启动推理，指定 RPC 端点
llama-server --model model.gguf \
  --rpc 192.168.1.10:50052,192.168.1.11:50052 \
  --tensor-split 0.5,0.3,0.2
```

## 需求

### 5.1 RPC 服务端（Worker）

```python
class RpcServer:
    """Worker 节点上的 RPC 服务端。"""
    
    async def start(self, port: int = 50052) -> None:
        """启动 llama-server --rpc-server-bind。"""
        
    async def stop(self) -> None:
        """停止 RPC 服务。"""
        
    @property
    def is_ready(self) -> bool:
        """RPC 服务是否就绪。"""
```

### 5.2 RPC 客户端（Master）

```python
class DistributedEngine(Engine):
    """分布式推理引擎。"""
    
    def __init__(
        self,
        model_path: str,
        rpc_endpoints: list[str],      # ["192.168.1.10:50052", ...]
        tensor_split: list[float],      # [0.5, 0.3, 0.2]
    ) -> None:
        
    def submit(self, request: InferenceRequest) -> str:
        """提交推理请求到分布式集群。"""
```

### 5.3 节点协商流程

```
1. Master 发现 Worker 节点
2. Master 查询各 Worker 的 VRAM
3. Master 计算 tensor-split 方案
4. Master 通知 Workers 启动 RPC Server
5. Workers 确认 RPC Server 就绪
6. Master 启动 llama-server --rpc <endpoints>
7. 开始分布式推理
```

### 5.4 故障恢复

- Worker 掉线时，自动重新计算 split 方案
- 支持优雅降级（减少节点继续运行）
- 模型状态持久化，快速恢复

## 实现计划

1. **Day 1-2**: 实现 RpcServer 和 RpcClient 类
2. **Day 3-4**: 修改 LlamaServerEngine 支持 --rpc 参数
3. **Day 5**: 实现节点协商和启动流程
4. **Day 6**: 实现故障检测和恢复
5. **Day 7**: 集成测试（双节点推理）

## 文件变更

- `src/asc/engine/rpc_server.py` - 新增 RPC 服务端
- `src/asc/engine/distributed.py` - 新增分布式引擎
- `src/asc/engine/llama_server.py` - 修改支持 RPC
- `src/asc/master/main.py` - 集成分布式启动流程
- `tests/engine/test_distributed.py` - 新增测试

## 关键技术点

1. **RPC 端口管理**：每个 Worker 需要独立 RPC 端口
2. **Tensor Split 动态计算**：根据实时 VRAM 调整
3. **网络延迟处理**：高延迟节点减少分配权重
4. **进程生命周期管理**：确保 llama-server 进程正确启停
