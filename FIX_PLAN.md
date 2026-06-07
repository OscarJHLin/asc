# ASC 分布式 LLM 推理系统 — 修复报告与计划 v3.0

**基于审计报告**: `AUDIT_REPORT_v3.md` (2026-06-06)
**上版计划**: `FIX_PLAN.md` v2.0 (2026-06-06)
**更新日期**: 2026-06-06

---

## 零、三次审计趋势总览

### 问题数量趋势

| 严重级别 | v1 | v2 | v3 | 趋势 |
|---------|-----|-----|-----|------|
| Critical | 8 | 3 | **1** | ↓↓ |
| High | 25 | 18 | **12** | ↓↓ |
| Medium | 48 | 35 | **26** | ↓↓ |
| Low | 24 | 22 | **16** | ↓ |
| **合计** | **105** | **78** | **55** | **↓↓ (-50, -48%)** |

### 综合评分趋势

| 维度 | v1 | v2 | v3 | 趋势 |
|------|-----|-----|-----|------|
| 安全性 | 3 | 7 | **8** | ↑↑ |
| 功能完整性 | 4 | 6 | **7** | ↑ |
| 代码质量 | 6 | 7 | **8** | ↑ |
| 测试覆盖 | 8 | 9 | **9** | → |
| 可维护性 | 5 | 6 | **7** | ↑ |
| 性能 | 5 | 4 | **5** | → |
| 文档 | 5 | 5 | **4** | ↓ |
| **综合** | **5.1** | **6.3** | **6.9** | **↑ (+1.8)** |

### 修复效率

| 审计轮次 | 发现问题 | 修复 | 修复率 |
|---------|---------|------|--------|
| v1 → v2 | 105 → 78 | 27 | 26% |
| v2 → v3 | 78 → 55 | 23 | 29% |
| **累计** | **105 → 55** | **50** | **48%** |

---

## 一、v3 修复概览

### 1.1 v2 → v3 新增修复（8 项）

| ID | 类别 | 问题 | 修复方式 |
|----|------|------|---------|
| **S-14** | 安全 | RateLimiter 无线程安全 | 添加 `asyncio.Lock`，`is_allowed()`/`remaining()`/`cleanup()` 均加锁 |
| **F-17** | 功能 | admin 端点返回假数据 | `/admin/nodes` 从 `cluster_state.nodes` 读取真实数据 |
| **F-20** | 功能 | 选举未检查 election_clock | 拒绝过时 COORDINATOR 消息 |
| **P-02** | 性能 | Histogram 无界增长 | `deque(maxlen=10000)` 替代无界列表 |
| **P-09** | 性能 | DiskEventLog 每次打开/关闭文件 | 添加 `_buffer` 批量写入（默认 100） |
| **Q-11/Q-19** | 代码质量 | RPCServerManager 死代码 | 移除类定义和实例化 |
| **Q-12** | 代码质量 | GracefulShutdown 无锁 | 添加 `asyncio.Lock` |
| **Q-15** | 代码质量 | create_instance 109 行 | 拆分为 8 个子方法 |

### 1.2 问题总览

| 严重级别 | 数量 | 需修复 | 备注 |
|---------|------|--------|------|
| **Critical** | 1 | 1 | Q-01 (_archived/ 旧代码) |
| **High** | 12 | 10 | 核心：流式推理、同步阻塞、无连接池、CLI、NODE_LEFT |
| **Medium** | 26 | 21 | 消息协议、I/O、配置、文档 |
| **Low** | 16 | 8 | 关键词、类型、变量命名等 |

---

## 二、P0 — 立即修复（5 项，阻塞生产）

> **v2 对比**: 减少了 GracefulShutdown 锁和 RateLimiter 锁（已修复），新增了 httpx 连接池和 CLI master 命令。

### 2.1 P-03/F-04: server.py 改用 submit_async() — [v2 未修复]

**问题**: `api/server.py:114` 非流式 + `api/server.py:173` 流式端点均调用同步 `engine.submit()`，阻塞事件循环

**修复**:
1. [ ] `api/server.py:114`: `engine.submit()` → `await engine.submit_async()`
2. [ ] `api/server.py:173`: 流式端点也改为异步调用

---

### 2.2 N-01: NODE_LEFT 无 handler — [v2 未修复]

**问题**: `master/main.py:375-380` 只注册 4 种消息类型，无 `NODE_LEFT`，无 `_handle_node_left` 方法

**修复**:
1. [ ] 实现 `_handle_node_left()` — 移出节点列表、取消相关任务
2. [ ] 在 `_MESSAGE_HANDLERS` 中注册 `MessageType.NODE_LEFT`

---

### 2.3 F-03: CLI 缺少 `asc master` 命令 — [v2 未修复]

**问题**: `cli/main.py:114-126` 只有 `start`/`status`/`discover`，`_cmd_start` 启动的是 WorkerAgent

**修复**:
1. [ ] 添加 `master` 子命令，接受 `--host`/`--port`/`--config`
2. [ ] 对接 `master/main.py` 的 `MasterNode.run()`

---

### 2.4 P-05: httpx 使用 Client 连接池 — [v2 未修复]

**问题**: `engine/llama_server.py:110,179` 每次 `httpx.get()`/`httpx.post()` 新建 TCP 连接

**修复**:
1. [ ] 创建模块级 `httpx.Client()` 实例，配置连接池
2. [ ] 所有 HTTP 调用复用该 client

---

### 2.5 P-01: compute_file_sha256 无用计算 — [v2 未修复]

**问题**: `model_distributor.py:107` 计算结果只用于 `f"SHA256: {file_sha256[:16]}..."` 日志

**修复**:
1. [ ] 若需完整性校验：在目标节点传递 SHA256 并校验
2. [ ] 若仅日志：移除计算，改用文件大小和时间戳

---

## 三、P1 — 1 周内修复（核心功能完善，7 项）

| # | ID | 修复项 | 文件 | 关键步骤 |
|---|----|--------|------|---------|
| 1 | **F-04** | 真正的流式推理 | `api/server.py`, 引擎层 | 引擎层实现 yield token 接口；重写 `_stream_chat()` 使用异步流式 |
| 2 | **N-02** | Master/Worker 统一 MessageDispatcher | `master/main.py`, `worker/agent.py` | 替换 `_MESSAGE_HANDLERS` 字典和 if/elif 为 `MessageDispatcher.dispatch()` |
| 3 | **N-05** | task_id 提取路径修复 | `worker/agent.py:248` | `payload.get("task", {}).get(...)` → `payload.get("task_id", "unknown")` |
| 4 | **P-08** | split_file_into_chunks 预读 | `network/sync.py:118-149` | 合并 chunk 切分和 SHA256 为一次读取 |
| 5 | **P-10** | ReceiveState 重复求和 | `network/sync.py`, `model_manager.py:305` | 维护 `received_bytes` 累加器替代 O(n) 遍历 |
| 6 | **Q-01** | 清理 _archived/ 旧代码 | `_archived/` | 删除目录或 `.gitignore` 排除 |
| 7 | **Q-31** | router.py 协程 fire-and-forget | `network/router.py:55-57` | `asyncio.create_task(result)` 异常时记录日志 |

---

## 四、P2 — 计划修复（6 项，性能 + 可维护性）

| # | ID | 修复项 | 文件 | 步骤 |
|---|----|--------|------|------|
| 1 | **F-18/F-21** | 统一配置管理 | `core/config.py` 等 | 将 `FailoverConfig`/`BatchConfig`/`RpcServer` 默认值统一到 `AscConfig`；清理未使用配置项 |
| 2 | **N-03** | `import asyncio` 移到模块顶 | `api/server.py:170` | 从 `_stream_chat` 内移到文件顶部 |
| 3 | **S-17** | 固定依赖版本 | `pyproject.toml:7-16` | 添加上限约束，运行 `pip-audit` |
| 4 | **S-15** | ContentFilter 默认关键词 | `api/security.py:159` | 添加基本安全关键词 |
| 5 | **P-11** | Worker RPC 串行→并行 | `orchestrator.py` | `asyncio.gather()` 并行发送 |
| 6 | **P-12** | 模型分片串行→并行 | `model_distributor.py` | `asyncio.gather()` 并行传输 |

---

## 五、P3 — 文档与体验改进（5 项）

| # | ID | 修复项 | 说明 |
|---|----|--------|------|
| 1 | **D-01** | 更新 README 架构描述 | 仍描述旧架构，需同步 `master/`/`worker/`/`scheduler/` |
| 2 | **D-02** | FastAPI OpenAPI 元数据 | 所有端点添加 `summary`/`description`/`tags`/`response_model` |
| 3 | **D-03** | Quick Start 指南 | 添加"启动 Master → Worker → 提交任务"端到端示例 |
| 4 | **D-05** | Anthropic/Ollama 路由注册 | `anthropic_adapter.py`/`ollama_adapter.py` 存在但无路由入口 |
| 5 | **D-09** | 创建 configuration.md | 完整列出 AscConfig 所有配置项 |

---

## 六、P0 快速执行清单

```
P0 — 本周必须完成（5 项）
├── [ ] 1. server.py 改用 submit_async()                    (P-03, F-04)
├── [ ] 2. 添加 NODE_LEFT handler                           (N-01)
├── [ ] 3. 添加 CLI asc master 命令                         (F-03)
├── [ ] 4. httpx 使用 Client 连接池                          (P-05)
└── [ ] 5. 移除 compute_file_sha256 无用计算                  (P-01)
```

---

## 七、执行路线图

```
Week 1 (P0)
├── Day 1: server.py 异步化 (P-03/F-04)
├── Day 2: NODE_LEFT handler (N-01) + CLI master 命令 (F-03)
├── Day 3: httpx 连接池 (P-05) + SHA256 移除 (P-01)
└── Day 4-5: 回归测试 + 验证

Week 2 (P1)
├── Day 1-3: 真正的流式推理 (F-04)
├── Day 4: MessageDispatcher 统一 (N-02) + task_id 修复 (N-05)
└── Day 5: I/O 优化 (P-08, P-10) + 清理 _archived_ (Q-01) + router 修复 (Q-31)

Week 3 (P2)
├── Day 1-2: 统一配置管理 (F-18, F-21)
├── Day 3: import asyncio + 依赖固定 + ContentFilter (N-03, S-17, S-15)
└── Day 4-5: 并行化 (P-11, P-12)

Week 4+ (P3)
├── 文档更新 (D-01, D-02, D-03, D-05, D-09)
└── 其他 low 项
```

---

## 八、验证策略

- **P0 完成**: 1,624 测试通过 + 异步调用不阻塞事件循环 + CLI 可启动 Master
- **P1 完成**: 流式推理 SSE 输出为真正 token-by-token + 统一分派逻辑
- **P2 完成**: 配置统一 + 性能基准无退化
- **P3 完成**: README 与代码一致 + `/docs` 页面完整

### 仍未覆盖的关键集成测试

| 场景 | 严重度 | 说明 |
|------|--------|------|
| 多节点集群端到端通信 | **Critical** | 无集成测试 |
| 选举竞争与网络分区 | **High** | 无测试 |
| 模型下载断点续传 | **High** | 无测试 |
| 并发推理压力测试 | **Medium** | 无压力测试 |
| 故障转移端到端 | **Medium** | 无测试 |

---

## 九、核心结论

三次审计确认项目持续改善：**105 → 78 → 55**，综合评分 **5.1 → 6.3 → 6.9**。

**已解决**: 并发安全（RateLimiter/GracefulShutdown 加锁）、代码结构（create_instance 拆分、死代码移除）、性能局部改善（Histogram 有界、DiskEventLog 缓冲）、功能真实化（admin 端点、选举鲁棒性）

**核心差距**: 性能问题（同步阻塞、无连接池、I/O 翻倍）、流式推理伪实现、CLI 缺失、文档滞后

**下一步最关键**: P0 的 5 项生产阻塞问题 — 特别是 server.py 异步化消除同步阻塞。

---

*修复计划基于 AUDIT_REPORT_v3.md (2026-06-06) 生成*
*对比基准: FIX_PLAN.md v2.0 (2026-06-06)*