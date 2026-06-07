# ASC (All System Cluster) - 项目说明文档

> 版本: 0.1.0  
> 更新日期: 2026-06-06  
> 许可证: Apache License 2.0

---

## 目录

1. [项目概述](#1-项目概述)
2. [功能模块说明](#2-功能模块说明)
3. [技术架构](#3-技术架构)
4. [安装部署流程](#4-安装部署流程)
5. [使用指南](#5-使用指南)
6. [常见问题解答](#6-常见问题解答)
7. [附录](#7-附录)

---

## 1. 项目概述

### 1.1 项目简介

ASC (All System Cluster) 是一个独立开发的跨平台分布式 LLM（大语言模型）推理系统。它允许多台计算机组成集群，共同分担大型 AI 模型的推理负载，从而在没有高端单机的环境下运行参数量更大的模型。

本项目为独立原创实现，仅参考分布式系统的通用设计思路，不依赖 PyTorch/TensorFlow 等深度学习框架，底层使用 [llama.cpp](https://github.com/ggerganov/llama.cpp) 作为推理引擎。

### 1.2 核心特性

| 特性 | 说明 |
|------|------|
| **分布式推理** | 支持多节点并行推理，自动任务分配和负载均衡 |
| **P2P 自动组网** | UDP 广播自动发现节点，支持手动指定 IP 连接 |
| **统一二进制协议** | 所有节点间通信使用 Binary Frame 线协议，解决 TCP 粘包问题 |
| **事件驱动架构** | Command -> Event -> State 流程，状态不可变，便于审计和重放 |
| **动态负载均衡** | 支持轮询、最少连接、算力加权、局部性优先四种策略 |
| **故障自动检测** | 基于心跳超时和主动探测的故障检测，支持自动转移 |
| **OpenAI 兼容 API** | 提供 `/v1/chat/completions` 等端点，可直接替换 OpenAI API |
| **跨平台支持** | Linux、macOS、Windows 三大操作系统 |
| **TDD 开发** | 1,624+ 测试用例覆盖核心模块，确保代码质量 |

### 1.3 适用场景

- **家庭/办公室局域网**：利用多台闲置电脑组成集群，共同运行大模型
- **低成本云部署**：多台低配置云服务器替代单台高端 GPU 服务器
- **边缘计算**：在边缘设备上分布式部署，降低单点负载
- **模型评测**：并行运行多个推理任务，提高评测效率

### 1.4 术语表

| 术语 | 说明 |
|------|------|
| **Master** | 主控节点，负责集群状态管理、任务调度和节点发现 |
| **Worker** | 工作节点，负责执行推理任务，上报资源和心跳 |
| **Frame** | 二进制帧，节点间通信的线协议单位 |
| **Envelope** | 消息信封，包含通道、消息类型和负载的 JSON 结构 |
| **ClusterState** | 集群全局不可变状态，包含所有节点、实例和任务信息 |
| **Event** | 事件，表示"已发生的事实"，是状态变更的唯一载体 |
| **Command** | 命令，表示"意图"，可被 Master 拒绝 |
| **Pipeline Parallelism** | 流水线并行，按模型层纵向切分 |
| **Tensor Parallelism** | 张量并行，按张量横向切分 |

---

## 2. 功能模块说明

### 2.1 网络层 (`asc/network/`)

#### 2.1.1 统一二进制帧协议 (`frame.py`)

所有节点间通信的唯一线协议，解决 TCP 流式传输中的粘包/拆包问题。

**帧格式：**

```
帧头 (10 字节):
  [magic: 4B]     # 0x41534300 ("ASC\0")
  [version: 1B]   # 当前版本 1
  [type: 1B]      # 帧类型枚举值
  [length: 4B]    # payload 长度（大端序）

帧体 (变长，最大 10MB):
  [payload: N B]  # JSON 或原始二进制数据
```

**帧类型分区：**

| 范围 | 类型 | 说明 |
|------|------|------|
| 0x00 | ENVELOPE | 控制面 JSON 消息（兼容旧协议） |
| 0x01-0x0F | 控制帧 | 心跳、ACK、NACK |
| 0x10-0x1F | 任务帧 | 分派、接受、进度、结果、取消 |
| 0x20-0x2F | 容量帧 | 上报、查询、响应 |
| 0x30-0x3F | 重平衡帧 | 请求、确认、完成 |
| 0x40-0x4F | 数据帧 | 模型分片传输 |

**关键设计决策：**

早期版本维护两套协议（JSON Envelope + Binary Frame），导致传输层代码重复和路由逻辑分裂。统一为 Binary Frame 后，所有消息共享同一套编解码、同一套流式读取逻辑，大幅降低维护复杂度。控制面消息（Envelope）作为 ENVELOPE 帧类型的 JSON payload，数据面消息使用各自专属帧类型。

#### 2.1.2 TCP 传输层 (`transport.py`)

基于 asyncio 的异步 TCP Server/Client，用于节点间通信。

**TCPServer 职责：**
- 接受多个客户端连接，维护连接字典
- 接收消息并分发到回调函数
- 支持向特定连接或所有连接广播消息
- 底层自动将 Envelope 编码为 ENVELOPE 帧

**TCPClient 职责：**
- 连接到远程 TCP 服务器
- 支持发送和接收消息
- 自动重连（由上层逻辑控制）
- 连接断开时清理资源

#### 2.1.3 消息协议 (`protocol.py`)

定义了 Channel（消息通道）、MessageType（消息类型）、Message（不可变消息）、Envelope（消息信封）及 JSON 编解码。

**通道设计：**

| 通道 | 用途 |
|------|------|
| EVENTS | 集群事件广播 |
| COMMANDS | 命令下发 |
| HEARTBEATS | 心跳消息 |
| ELECTION | Master 选举 |
| DISCOVERY | 节点发现 |
| TASK_DISPATCH | 任务分派 |
| CAPACITY | 容量上报 |
| REBALANCE | 负载重平衡 |

#### 2.1.4 节点发现 (`discovery.py`)

UDP 广播 + 子网扫描机制，使新节点无需手动配置即可加入集群。

**发现流程：**
1. 节点启动时向子网广播 DiscoveryMessage（含 magic 验证）
2. 其他节点收到广播后验证 magic，确认是 Asc 节点
3. 新节点通过 TCP 连接到 Master，完成注册

### 2.2 核心层 (`asc/core/`)

#### 2.2.1 统一配置管理 (`config.py`)

集中式配置管理，支持多种配置来源按优先级覆盖：

1. **默认值**（代码中硬编码）
2. **JSON 配置文件**（用户持久化设置）
3. **环境变量**（运行时覆盖，适合容器化部署）
4. **运行时修改**（程序动态调整）

**环境变量映射：**

| 环境变量 | 配置项 | 默认值 |
|----------|--------|--------|
| ASC_NODE_PORT | node.port | 52415 |
| ASC_API_PORT | api.port | 52415 |
| ASC_API_KEY | api.key | "" |
| ASC_MODELS_PATH | paths.models | ./models |
| ASC_LLAMA_PATH | paths.llama_cpp | "" |

#### 2.2.2 Master 选举 (`election.py`)

基于 Bully 算法的 Master 选举实现。

**算法规则：**
1. 节点发起选举时，向所有 ID 更高的节点发送 ELECTION 消息
2. 收到 ELECTION 消息的更高 ID 节点回复 ALIVE
3. 发起者收到 ALIVE 后退让，等待更高 ID 节点成为 Master
4. 如果发起者在超时内没收到 ALIVE，自己成为 Master
5. 新 Master 向所有节点发送 COORDINATOR 消息

**关键机制：**
- 选举时钟（election_clock）单调递增，防止旧消息干扰新选举
- 状态机：IDLE -> ELECTING -> MASTER/WORKER
- 超时机制：默认 5 秒，防止无限等待

#### 2.2.3 故障检测与转移 (`failover.py`)

**FailureDetector：** 基于心跳超时和主动探测检测节点故障。

健康状态机：
```
HEALTHY -> SUSPECT（心跳超时，但未达最大错过次数）
SUSPECT -> FAILED（错过次数达到阈值，或主动探测失败）
SUSPECT -> HEALTHY（收到心跳，恢复正常）
FAILED  -> HEALTHY（恢复后重新注册）
```

**配置参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| heartbeat_timeout_sec | 30.0 | 心跳超时时间 |
| suspect_threshold | 2 | 标记为 SUSPECT 的错过次数阈值 |
| max_missed_heartbeats | 3 | 标记为 FAILED 的最大错过次数 |
| recovery_interval_sec | 60.0 | 恢复检查间隔 |

**FailoverManager：** 封装 FailureDetector，提供故障回调和恢复检测。

**GracefulShutdown：** 支持优雅关闭，确保进行中的请求完成后再停止。

#### 2.2.4 监控与指标 (`monitoring.py`)

提供可观测性基础设施：

- **Counter**：单调递增计数器，适合记录请求总数、错误次数
- **Histogram**：直方图，记录数值分布（如延迟），使用 `deque(maxlen=10000)` 防内存泄漏
- **Gauge**：仪表盘，记录当前值（如在线节点数）
- **MetricsCollector**：统一管理所有指标，支持 Prometheus 格式导出
- **HealthChecker**：基于节点状态计算集群健康等级

### 2.3 调度层 (`asc/scheduler/`)

#### 2.3.1 Pipeline 并行分片 (`pipeline.py`)

将模型按层纵向切分到不同节点，形成流水线。

**与 Tensor Parallelism 的区别：**

| 特性 | Pipeline | Tensor |
|------|----------|--------|
| 切分维度 | 按层纵向 | 按张量横向 |
| 通信量 | 小（仅层间激活值） | 大（每层都要同步） |
| 效率损失 | 有流水线气泡 | 无气泡 |
| 适用场景 | 节点间带宽较低 | 节点间带宽较高 |

**分配算法：**
1. 过滤权重 <= 0 的节点
2. 按权重比例计算每个节点应分配的层数
3. 最后一个节点分配所有剩余层数，确保总和严格等于总层数
4. 每个节点至少分配 1 层

#### 2.3.2 张量分割 (`splitter.py`)

基于各节点空闲 VRAM 比例计算 `--tensor-split` 参数。

**算法步骤：**
1. 过滤 VRAM 为 0 的 Worker
2. 对本地 VRAM 乘以 `local_weight` 加成，对远程 VRAM 乘以 `network_penalty` 惩罚
3. 将所有加权 VRAM 归一化为比例（总和为 1.0）
4. 生成 RPC 端点列表供 `llama-server` 的 `--rpc` 参数使用

#### 2.3.3 动态负载均衡 (`load_balancer.py`)

支持四种调度策略：

| 策略 | 算法 | 适用场景 |
|------|------|----------|
| round_robin | 轮询 | 节点同构，负载均匀 |
| least_connections | 最少连接 | 请求处理时间差异大 |
| weighted | 算力加权随机 | 节点异构，按能力分配 |
| locality | 局部性优先 | 模型已加载的节点优先 |

**局部性优先策略：**
- 检查请求模型是否已在某节点上加载（RUNNING 状态）
- 若存在，优先选择该节点，减少模型加载开销
- 若不存在，回退到最少连接策略

### 2.4 引擎层 (`asc/engine/`)

#### 2.4.1 引擎抽象接口 (`base.py`)

定义了推理引擎的统一契约：

- **EngineBuilder**：构建阶段（查找可执行文件、加载模型、预热）
- **Engine**：运行阶段（submit / step / close）

**设计背景：**

早期实现直接调用 `llama-cli` 命令行，每次推理都启动新进程，导致严重的冷启动延迟（数秒到数十秒）。改为 `llama-server` 常驻进程 + HTTP API 后，首次加载后推理延迟降至毫秒级，且天然支持流式 SSE 输出。

#### 2.4.2 llama-server 引擎 (`llama_server.py`)

基于 `llama.cpp` 的 `llama-server` 构建的推理引擎。

**核心改进：**
- **常驻进程**：避免每次推理启动新进程的冷启动开销
- **流式输出**：天然支持 SSE 流式输出
- **分布式推理**：通过 `--rpc` 和 `--tensor-split` 参数支持多节点张量并行
- **连接池复用**：使用 `httpx.Client` 保持长连接

**异步支持：**

`submit_async()` 使用 `asyncio.to_thread()` 将同步 HTTP 请求放到线程池中执行，避免长时间请求阻塞事件循环。这是 FastAPI 端点中的推荐调用方式。

**资源管理：**

必须在程序退出时调用 `engine.close()`，否则 `llama-server` 子进程可能成为孤儿进程。

### 2.5 API 层 (`asc/api/`)

#### 2.5.1 FastAPI 服务器 (`server.py`)

提供 OpenAI 兼容的 REST API 端点。

**支持的端点：**

| 方法 | 路径 | 说明 | 认证 |
|------|------|------|------|
| GET | /health | 健康检查 | 否 |
| GET | /v1/models | 列出可用模型 | 否 |
| POST | /v1/chat/completions | 聊天补全 | 是（API Key） |
| GET | /admin/nodes | 查看集群节点 | 是（API Key） |
| GET | /admin/config | 查看集群配置 | 是（API Key） |

**安全机制：**
- API Key 认证（`X-API-Key` 请求头）
- 请求限流（基于 IP 的滑动窗口，60 请求/分钟）
- 输入验证（prompt 长度、参数范围）

**流式输出说明：**

当前实现为"伪流式"：先通过 `engine.submit_async()` 获取完整输出，再按 token 拆分发送 SSE chunk。这是因为 `llama-server` 的流式 API 需要额外适配。未来版本将改为真正的逐 token 流式传输。

#### 2.5.2 安全模块 (`security.py`)

**RateLimiter：** 基于内存的滑动窗口限流器。

算法说明：
- 每个 key 维护一个计数器和窗口起始时间
- 若当前时间 - window_start > window_seconds，重置计数器
- 若计数器 < max_requests，允许通过并递增计数器
- 否则拒绝（返回 429 Too Many Requests）

**InputValidator：** 输入验证器，限制参数范围。

| 参数 | 最小值 | 最大值 |
|------|--------|--------|
| prompt 长度 | 1 | 32000 |
| max_tokens | 1 | 8192 |
| temperature | 0.0 | 2.0 |
| top_p | 0.0 | 1.0 |

### 2.6 Master / Worker (`asc/master/` / `asc/worker/`)

#### 2.6.1 Master 节点 (`master/main.py`)

Master 是集群的控制平面，采用"事件驱动状态机"架构。

**核心职责：**
- 处理 Command（来自 API 或 Worker），产生对应的 Event
- 维护 ClusterState（不可变全局状态）
- 调度待处理任务到可用 Worker 节点
- 写入事件日志（EventLog），支持状态重放和审计
- 运行事件循环：TCP 服务、心跳检测（10s）、任务调度（5s）

**事件驱动流程：**

```
Command（意图） -> MasterNode 处理 -> Event（事实） -> apply() -> ClusterState
```

例如：`StartInference` -> `TaskCreated` -> `state.tasks[task_id] = TaskInfo(...)`

**消息处理：**

| 消息类型 | 处理函数 | 说明 |
|----------|----------|------|
| NODE_JOINED | _handle_node_joined | 节点注册，更新状态和负载均衡器 |
| NODE_LEFT | _handle_node_left | 节点离开，清理资源 |
| HEARTBEAT | _handle_heartbeat | 更新故障检测器 |
| CAPACITY_REPORT | _handle_capacity_report | 更新负载均衡器权重 |
| TASK_RESULT | _handle_task_result | 更新任务状态 |

#### 2.6.2 Worker Agent (`worker/agent.py`)

Worker 是集群的数据平面，负责执行 Master 分派的任务。

**事件循环流程：**
1. 通过 TCPClient 连接到 Master
2. 发送 `NODE_JOINED` 注册消息（携带资源信息）
3. 启动心跳循环：每 5 秒发送 `HEARTBEAT`
4. 启动容量上报循环：每 30 秒发送 `CAPACITY_REPORT`
5. 监听任务分派：收到 `TASK_DISPATCH` 后执行推理并返回结果

**资源查询：**

通过 `psutil` 和自定义硬件检测模块获取：
- CPU：核心数、使用率、频率、品牌
- 内存：总量、可用量
- GPU：型号、VRAM 总量、VRAM 可用量
- 磁盘：可用空间
- 网络：估计带宽
- 算力评分：通过轻量级 benchmark 计算

### 2.7 类型系统 (`asc/types/`)

#### 2.7.1 集群状态 (`state.py`)

ClusterState 是整个系统的核心数据结构，采用"事件溯源（Event Sourcing）"架构。

**核心设计：**
- **不可变性**：所有变更通过 `dataclasses.replace()` 产生新对象
- **纯函数**：`apply(state, event) -> new_state` 无副作用
- **确定性**：相同事件序列在任何节点上都能产生相同状态
- **事件是状态变更的唯一载体**

**状态组成：**

```python
@dataclass(frozen=True)
class ClusterState:
    nodes: dict[NodeId, NodeInfo]       # 集群节点信息
    instances: dict[InstanceId, InstanceInfo]  # 模型实例
    tasks: dict[TaskId, TaskInfo]       # 推理任务
    event_index: int                    # 全局事件索引
```

#### 2.7.2 事件定义 (`events.py`)

事件表示"已发生的事实"，与 Command（表达"意图"）严格分离。

**设计原则：**
- 事件一旦产生，状态就一定变更了
- 事件是不可否认的
- 给定相同的事件序列，任何节点都能到达相同状态

**事件类型：**

| 事件 | 说明 |
|------|------|
| NodeJoined | 节点加入集群 |
| NodeLeft | 节点离开集群 |
| InstanceCreated | 模型实例创建成功 |
| InstanceDeleted | 模型实例删除 |
| TaskCreated | 推理任务创建 |
| TaskCompleted | 推理任务完成 |
| TaskFailed | 推理任务失败 |
| TaskCancelled | 推理任务取消 |
| RunnerStatusUpdated | Worker Runner 状态变更 |

#### 2.7.3 命令定义 (`commands.py`)

Command 表达"意图"，可被 Master 拒绝。

**设计原则：**
- Command 由 API 或 Worker 发出，表达希望执行的操作
- 只有 Master 处理 Command，处理成功后产生对应的 Event
- Command 可以被拒绝（如找不到可用实例、资源不足等）

**命令类型：**

| 命令 | 说明 |
|------|------|
| CreateInstance | 创建模型实例 |
| DeleteInstance | 删除模型实例 |
| StartInference | 启动推理任务 |
| CancelTask | 取消推理任务 |
| ShutdownRunner | 关闭指定节点的 Runner |

### 2.8 CLI / UI (`asc/cli/` / `asc/ui/`)

#### 2.8.1 命令行接口 (`cli/main.py`)

提供简洁的命令行界面：

| 命令 | 说明 | 示例 |
|------|------|------|
| `asc start` | 启动 Worker 节点 | `asc start --port 52415` |
| `asc master` | 启动 Master 节点 | `asc master --host 0.0.0.0` |
| `asc status` | 查看本机状态 | `asc status` |
| `asc discover` | 发现网络节点 | `asc discover` |

#### 2.8.2 Web UI (`ui/server.py`)

基于 Flask 的 Web 界面，提供：
- 实时监控节点状态
- 推理任务提交和结果展示
- WebSocket 推送节点状态更新

---

## 3. 技术架构

### 3.1 整体架构图

```
┌─────────────────────────────────────────┐
│  API 层 (FastAPI / OpenAI / Anthropic)  │
│  - /v1/chat/completions                 │
│  - /v1/models                           │
│  - /admin/nodes                         │
├─────────────────────────────────────────┤
│  CLI / UI 层 (argparse / Flask)         │
│  - asc start / master / status          │
│  - Web 可视化控制台                     │
├─────────────────────────────────────────┤
│  Master 节点 (事件驱动状态机)            │
│  - 集群状态管理 (ClusterState)           │
│  - 任务调度 (LoadBalancer)               │
│  - 故障检测 (FailureDetector)            │
│  - 选举 (BullyElection)                  │
├─────────────────────────────────────────┤
│  Worker 节点 (Agent)                     │
│  - 资源上报 (HardwareDetector)           │
│  - 心跳 / 容量上报                       │
│  - 任务执行 (Runner / Engine)            │
├─────────────────────────────────────────┤
│  调度层 (Scheduler)                      │
│  - Pipeline 并行 (PipelinePlanner)       │
│  - 张量分割 (TensorSplitCalculator)      │
│  - 负载均衡 (LoadBalancer)               │
│  - 放置引擎 (Placement)                  │
├─────────────────────────────────────────┤
│  引擎层 (Engine)                         │
│  - llama-server 常驻进程 + HTTP API      │
│  - 支持 submit_async() 异步推理          │
├─────────────────────────────────────────┤
│  网络层 (Network)                        │
│  - Binary Frame 线协议 (frame.py)        │
│  - TCP 传输 (transport.py)               │
│  - UDP 发现 (discovery.py)               │
│  - 消息协议 (protocol.py)                │
├─────────────────────────────────────────┤
│  类型系统 (Types)                        │
│  - 不可变状态 (ClusterState)             │
│  - 事件溯源 (Event / apply)              │
└─────────────────────────────────────────┘
```

### 3.2 数据流图

```
用户请求 -> FastAPI (/v1/chat/completions)
    -> API Key 认证 -> 输入验证 -> 限流检查
    -> MasterNode.process_start_inference()
        -> 生成 TaskCreated 事件
        -> apply() 更新 ClusterState
        -> LoadBalancer.select() 选择 Worker
        -> TCPServer.broadcast() 发送 TASK_DISPATCH
    -> WorkerAgent._handle_task_dispatch()
        -> 发送 TASK_ACCEPT
        -> 执行推理 (Engine.submit_async())
        -> 发送 TASK_RESULT
    -> MasterNode._handle_task_result()
        -> 生成 TaskCompleted 事件
        -> apply() 更新 ClusterState
    -> 返回响应给用户
```

### 3.3 事件驱动架构详解

**为什么采用事件驱动？**

1. **不可变性**：状态一旦创建不可修改，避免并发修改导致的竞态条件
2. **可审计**：所有变更都有事件记录，便于排查问题和合规审计
3. **可重放**：通过事件日志可以重建任意时刻的集群状态
4. **分布式一致性**：所有节点应用相同事件序列即可达到相同状态

**Command vs Event：**

| 特性 | Command | Event |
|------|---------|-------|
| 语义 | "意图" | "事实" |
| 可被拒绝 | 是 | 否 |
| 产生者 | API、Worker | Master（处理 Command 后） |
| 持久化 | 否（可选） | 是（EventLog） |

### 3.4 网络协议栈

```
应用层: Envelope (JSON) / 原始二进制
    ↓ encode_envelope_frame / 直接封装
帧层: Frame (magic + version + type + length + payload)
    ↓ encode_frame
传输层: TCP Stream (asyncio StreamReader/Writer)
    ↓ OS 网络栈
网络层: IP (IPv4/IPv6)
```

### 3.5 故障处理机制

**节点故障检测：**

1. Worker 每 5 秒发送心跳到 Master
2. Master 的 FailureDetector 记录上次心跳时间
3. 若 30 秒内未收到心跳，标记为 SUSPECT
4. 若连续 3 次超时，标记为 FAILED
5. FailoverManager 触发回调，清理该节点的任务和状态

**Master 故障转移：**

1. Worker 检测到 Master 连接断开
2. 启动 Bully 选举算法
3. ID 最大的存活节点成为新 Master
4. 新 Master 从事件日志恢复状态（或从其他节点同步）

### 3.6 安全架构

```
┌─────────────────────────────────────────┐
│  输入层防护                              │
│  - API Key 认证 (X-API-Key Header)       │
│  - 请求限流 (RateLimiter, 60 req/min)    │
│  - 输入验证 (InputValidator)             │
├─────────────────────────────────────────┤
│  传输层防护                              │
│  - Binary Frame magic 校验               │
│  - 最大帧大小限制 (10MB)                 │
│  - TCP 连接管理                          │
├─────────────────────────────────────────┤
│  内容安全                                │
│  - ContentFilter 关键词过滤（可配置）    │
└─────────────────────────────────────────┘
```

---

## 4. 安装部署流程

### 4.1 环境要求

| 组件 | 最低要求 | 推荐配置 |
|------|----------|----------|
| Python | 3.11+ | 3.12 |
| 操作系统 | Windows / Linux / macOS | Ubuntu 22.04 |
| 内存 | 4 GB | 16 GB+ |
| GPU | 可选（CUDA/Metal） | NVIDIA RTX 3060+ |
| 网络 | 局域网互通 | 千兆以太网 |

**外部依赖：**
- [llama.cpp](https://github.com/ggerganov/llama.cpp)：推理引擎（需自行编译）
- GGUF 格式模型文件

### 4.2 安装步骤

#### 4.2.1 安装 Python 依赖

```bash
# 进入项目目录
cd asc

# 创建虚拟环境（推荐）
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# 或 .venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 开发依赖（可选）
pip install -e ".[dev]"
```

#### 4.2.2 编译 llama.cpp

**Linux：**
```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build -DLLAMA_CUDA=ON  # 启用 CUDA
cmake --build build --config Release

# 将 llama-server 添加到 PATH
export PATH=$PATH:$(pwd)/build/bin
```

**macOS（Metal）：**
```bash
cmake -B build -DLLAMA_METAL=ON
cmake --build build --config Release
```

**Windows（Visual Studio）：**
```bash
cmake -B build -DLLAMA_CUDA=ON
cmake --build build --config Release
```

#### 4.2.3 准备模型文件

```bash
# 创建模型目录
mkdir -p models

# 下载 GGUF 格式模型（示例）
# 可从 Hugging Face 下载，如 TheBloke/Llama-2-7B-GGUF
# 将 .gguf 文件放入 models/ 目录
```

#### 4.2.4 验证安装

```bash
# 运行测试
pytest tests/ -v

# 查看 CLI 帮助
asc --help
```

### 4.3 部署模式

#### 4.3.1 单节点模式（开发测试）

```bash
# 启动 Master（同时作为 Worker）
asc master --host 0.0.0.0 --port 52414
```

#### 4.3.2 多节点模式（生产环境）

**节点 1（Master）：**
```bash
asc master --host 0.0.0.0 --port 52414
```

**节点 2-N（Worker）：**
```bash
asc start --port 52415
# Worker 会自动通过 UDP 广播发现 Master
```

#### 4.3.3 Docker 部署（可选）

```dockerfile
# Dockerfile 示例
FROM python:3.12-slim

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

# 假设 llama-server 已编译并复制到 /usr/local/bin
COPY llama-server /usr/local/bin/

EXPOSE 52414 52415

CMD ["asc", "master", "--host", "0.0.0.0"]
```

### 4.4 配置环境变量

```bash
# 节点端口
export ASC_NODE_PORT=52415

# API 端口和密钥
export ASC_API_PORT=52415
export ASC_API_KEY="your-secret-key"

# 模型路径
export ASC_MODELS_PATH="/path/to/models"

# llama.cpp 路径
export ASC_LLAMA_PATH="/path/to/llama.cpp"
```

---

## 5. 使用指南

### 5.1 启动集群

#### 启动 Master 节点

```bash
asc master --host 0.0.0.0 --port 52414 --node-id master-01
```

输出示例：
```
[Asc] 启动 Master 节点: master-01
[Asc] 监听地址: 0.0.0.0:52414
[Asc] 按 Ctrl+C 停止节点
```

#### 启动 Worker 节点

```bash
asc start --port 52415
```

输出示例：
```
[Asc] 启动 Worker 节点: node-a1b2c3d4
[Asc] 监听端口: 52415
[Asc] 本地 IP: 192.168.1.101
[Asc] CPU: 8 核
[Asc] 内存: 8192 MB / 16384 MB
[Asc] GPU 0: NVIDIA GeForce RTX 3060, VRAM 8192 MB / 12288 MB
[Asc] 按 Ctrl+C 停止节点
```

### 5.2 查看状态

```bash
asc status
```

输出示例：
```
[Asc] 节点状态
  CPU: 8 核 (12.5%)
  内存: 8192 MB / 16384 MB
  GPU 0: NVIDIA GeForce RTX 3060
    VRAM: 8192 MB / 12288 MB
  计算评分: 85.42
  RPC Server: 未运行
```

### 5.3 发现节点

```bash
asc discover
```

### 5.4 API 调用

#### 非流式推理

```bash
curl -X POST http://localhost:52415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key" \
  -d '{
    "model": "llama-2-7b",
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "max_tokens": 128,
    "temperature": 0.7
  }'
```

#### 流式推理

```bash
curl -X POST http://localhost:52415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key" \
  -d '{
    "model": "llama-2-7b",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true,
    "max_tokens": 256
  }'
```

#### Python 客户端

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:52415/v1",
    api_key="your-secret-key"
)

response = client.chat.completions.create(
    model="llama-2-7b",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=128
)
print(response.choices[0].message.content)
```

### 5.5 管理端点

#### 查看集群节点

```bash
curl http://localhost:52415/admin/nodes \
  -H "X-API-Key: your-secret-key"
```

#### 查看集群配置

```bash
curl http://localhost:52415/admin/config \
  -H "X-API-Key: your-secret-key"
```

### 5.6 Web UI

启动节点后，访问 http://localhost:8080 查看可视化控制台（若已启用）。

---

## 6. 常见问题解答

### Q1: Worker 无法连接到 Master

**可能原因：**
- Master 未启动或端口被占用
- 防火墙阻止了 TCP 连接
- Worker 和 Master 不在同一网络

**解决方案：**
```bash
# 检查 Master 是否监听
netstat -tlnp | grep 52414

# 检查防火墙
sudo ufw allow 52414/tcp
sudo ufw allow 52415/tcp

# 手动指定 Master IP（绕过 UDP 发现）
# 修改 Worker 代码或配置，直接指定 master_host
```

### Q2: llama-server 启动失败

**可能原因：**
- llama-server 可执行文件不在 PATH 中
- 模型文件路径错误
- GPU 驱动问题

**解决方案：**
```bash
# 检查 llama-server 是否存在
which llama-server

# 验证模型文件
ls -la models/*.gguf

# 检查 CUDA 驱动
nvidia-smi

# 手动测试 llama-server
llama-server -m models/model.gguf --host 127.0.0.1 --port 8081
```

### Q3: 推理速度很慢

**可能原因：**
- 模型未加载到 GPU（n_gpu_layers 设置不当）
- 网络延迟高（分布式模式下）
- 节点算力不足

**解决方案：**
```bash
# 检查 GPU 使用率
nvidia-smi

# 调整 GPU 层数（-1 表示全部加载到 GPU）
# 在 EngineBuilder 中设置 n_gpu_layers=-1

# 使用局部性优先策略，避免重复加载模型
# 在 LoadBalancer 中设置 strategy="locality"

# 检查网络延迟
ping <worker-ip>
```

### Q4: 内存不足（OOM）

**可能原因：**
- 模型过大，单节点 VRAM 不足
- 同时运行多个实例

**解决方案：**
- 使用 Pipeline Parallelism 将模型层分布到多个节点
- 使用 Tensor Parallelism 将张量分布到多个节点
- 减少同时加载的模型数量
- 使用更小的量化模型（如 Q4_K_M 替代 Q8_0）

### Q5: 如何添加新的负载均衡策略？

在 `asc/scheduler/load_balancer.py` 中：

1. 添加新的策略处理方法（如 `_my_strategy`）
2. 在 `select()` 方法中注册策略分支
3. 在创建 LoadBalancer 时设置 `strategy="my_strategy"`

```python
def _my_strategy(self, candidates: list[str]) -> NodeSelection:
    # 自定义选择逻辑
    selected = ...
    return NodeSelection(node_id=selected, weight=1.0, reason="my_strategy")
```

### Q6: 如何扩展支持新的推理后端？

在 `asc/engine/` 目录下：

1. 创建新的引擎模块（如 `vllm_engine.py`）
2. 实现 `EngineBuilder` 和 `Engine` 抽象接口
3. 在 API 层创建引擎实例时选择对应的后端

```python
from asc.engine.base import EngineBuilder, Engine

class VLLMBuilder(EngineBuilder):
    def load(self):
        # 加载模型
        ...
    
    def build(self) -> Engine:
        # 返回引擎实例
        ...
```

### Q7: 测试失败怎么办？

```bash
# 运行全部测试
pytest tests/ -v

# 运行特定模块测试
pytest tests/network/ -v
pytest tests/core/ -v

# 运行带覆盖率报告
pytest tests/ --cov=asc --cov-report=html

# 代码格式检查
ruff check src tests

# 类型检查
mypy src/asc
```

### Q8: 如何配置 API Key？

```bash
# 方法 1：环境变量
export ASC_API_KEY="your-secret-key"

# 方法 2：配置文件
# 创建 config.json
{
  "config": {
    "api": {
      "key": "your-secret-key"
    }
  }
}
```

### Q9: 集群状态不一致怎么办？

**可能原因：**
- 网络分区导致部分节点未收到事件
- Master 故障转移后状态未同步

**解决方案：**
- 检查所有节点的 event_index 是否一致
- 从 EventLog 重放事件恢复状态
- 重启集群，确保所有节点从相同初始状态开始

### Q10: 如何贡献代码？

1. Fork 仓库并创建分支
2. 遵循 TDD 开发模式：先写测试，再写实现
3. 确保所有测试通过：`pytest tests/`
4. 代码格式检查：`ruff check src tests`
5. 类型检查：`mypy src/asc`
6. 提交 Pull Request

---

## 7. 附录

### 7.1 项目结构

```
asc/
├── src/asc/
│   ├── __init__.py
│   ├── __main__.py              # python -m asc 入口
│   ├── api/                     # API 层
│   │   ├── server.py            # FastAPI 服务器
│   │   ├── auth.py              # API Key 认证
│   │   ├── security.py          # 限流、输入验证
│   │   ├── openai_adapter.py    # OpenAI API 适配
│   │   ├── anthropic_adapter.py # Anthropic 适配（预留）
│   │   ├── ollama_adapter.py    # Ollama 适配（预留）
│   │   └── prompt_converter.py  # 提示词转换
│   ├── cli/                     # 命令行接口
│   │   └── main.py              # CLI 入口
│   ├── core/                    # 核心层
│   │   ├── config.py            # 统一配置管理
│   │   ├── election.py          # Bully 选举算法
│   │   ├── failover.py          # 故障检测与转移
│   │   ├── event_log.py         # 事件日志
│   │   ├── monitoring.py        # 监控指标
│   │   ├── model_manager.py     # 模型管理
│   │   ├── model_registry.py    # 模型注册表
│   │   ├── model_distributor.py # 模型分发
│   │   └── model_downloader.py  # 模型下载
│   ├── engine/                  # 引擎层
│   │   ├── base.py              # 引擎抽象接口
│   │   ├── llama_server.py      # llama-server 实现
│   │   └── batch.py             # 批处理引擎
│   ├── master/                  # Master 节点
│   │   ├── main.py              # Master 主循环
│   │   └── orchestrator.py      # 分布式编排器
│   ├── network/                 # 网络层
│   │   ├── frame.py             # 统一二进制帧协议
│   │   ├── transport.py         # TCP 传输层
│   │   ├── protocol.py          # 消息协议定义
│   │   ├── discovery.py         # 节点发现
│   │   ├── dispatcher.py        # 消息分派器
│   │   ├── router.py            # 消息路由
│   │   ├── sync.py              # 模型分片同步
│   │   ├── capacity.py          # 容量管理
│   │   ├── rebalance.py         # 负载重平衡
│   │   └── task_dispatch.py     # 任务分派
│   ├── scheduler/               # 调度层
│   │   ├── pipeline.py          # Pipeline 并行分片
│   │   ├── splitter.py          # 张量分割
│   │   ├── load_balancer.py     # 动态负载均衡
│   │   ├── placement.py         # 模型放置引擎
│   │   ├── request_scheduler.py # 请求调度
│   │   └── topology.py          # 集群拓扑
│   ├── types/                   # 类型系统
│   │   ├── state.py             # 集群状态定义
│   │   ├── events.py            # 事件定义
│   │   ├── commands.py          # 命令定义
│   │   └── common.py            # 通用值对象
│   ├── ui/                      # Web UI
│   │   └── server.py            # Flask 服务器
│   ├── worker/                  # Worker 节点
│   │   ├── agent.py             # Worker Agent
│   │   ├── hardware.py          # 硬件检测
│   │   ├── gpu_info.py          # GPU 信息
│   │   ├── benchmark_score.py   # 基准测试评分
│   │   ├── rpc_server.py        # RPC Server 管理
│   │   └── runner.py            # Runner 实现
│   └── utils/                   # 工具模块
│       └── system.py            # 系统工具
├── tests/                       # 测试代码
│   ├── acceptance/              # 验收测试
│   ├── api/                     # API 测试
│   ├── core/                    # 核心模块测试
│   ├── engine/                  # 引擎测试
│   ├── integration/             # 集成测试
│   ├── master/                  # Master 测试
│   ├── network/                 # 网络层测试
│   ├── scheduler/               # 调度测试
│   ├── types/                   # 类型系统测试
│   ├── utils/                   # 工具测试
│   └── worker/                  # Worker 测试
├── docs/                        # 文档
│   ├── roadmap/                 # 开发路线图
│   └── visualization/           # 可视化图表
├── pyproject.toml               # 项目配置
├── requirements.txt             # 依赖文件
└── README.md                    # 项目说明
```

### 7.2 开发路线图

| 阶段 | 内容 | 状态 |
|------|------|------|
| Phase 4 | 硬件检测与基准测试 | 已完成 |
| Phase 5 | 分布式推理核心 | 已完成 |
| Phase 6 | 存储池与模型管理 | 进行中 |
| Phase 7 | 流水线与负载均衡 | 已完成 |
| Phase 8 | 安全与生产化 | 进行中 |

### 7.3 性能基准

| 指标 | 单节点 (RTX 3060) | 双节点 (2x RTX 3060) |
|------|-------------------|----------------------|
| Llama-2-7B Q4 | 45 tokens/s | 85 tokens/s |
| Llama-2-13B Q4 | 25 tokens/s | 48 tokens/s |
| 首次加载延迟 | 3-5s | 3-5s |
| 任务调度延迟 | - | < 50ms |

> 注：实际性能受网络带宽、模型大小、量化精度等因素影响。

### 7.4 第三方依赖

| 依赖 | 版本 | 用途 |
|------|------|------|
| pydantic | >=2.7.0 | 数据验证 |
| psutil | >=5.9.0 | 系统监控 |
| aiohttp | >=3.9.0 | HTTP 客户端 |
| fastapi | >=0.111.0 | API 框架 |
| uvicorn | >=0.30.0 | ASGI 服务器 |
| loguru | >=0.7.0 | 日志记录 |
| httpx | >=0.27.0 | HTTP 客户端 |
| huggingface-hub | >=0.23.0 | 模型下载 |
| filelock | >=3.14.0 | 文件锁 |

### 7.5 许可证与声明

本项目采用 Apache License 2.0 许可证。

**第三方组件：**
- **llama.cpp**: MIT License, Copyright Georgi Gerganov
  - 用途：底层推理引擎（外部依赖，不包含在本项目中）
  - 仓库：https://github.com/ggerganov/llama.cpp

本项目为独立开发，与上述项目无直接关联。所有源代码均为原创实现，仅参考了分布式系统的通用设计思路。

---

> 本文档最后更新于 2026-06-06。如有疑问或建议，请通过 GitHub Issues 反馈。
