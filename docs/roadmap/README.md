# ASC 开发路线图

> 版本：v0.1.0 -> v1.0.0
> 目标：构建生产可用的跨平台分布式 LLM 推理系统

## 当前状态（v0.1.0）

**已完成**：
- 类型系统、配置管理、事件驱动架构
- 网络层（UDP 发现、TCP 传输、消息协议）
- API 适配层（OpenAI / Ollama / Anthropic 兼容）
- 调度算法框架（张量分割、放置引擎）
- 346 个单元测试全部通过

**关键缺失**：
- 完整硬件检测（CPU/内存/GPU 类型）
- 真正的分布式推理（llama-server RPC 模式）
- 统一存储池与模型同步
- Pipeline 并行与动态负载均衡
- 安全加固（HTTPS、API 限流）

---

## 开发阶段

| 阶段 | 主题 | 优先级 | 预计工作量 |
|------|------|--------|-----------|
| [Phase 4](phase4_hardware_detection.md) | 完整硬件检测与资源监控 | P0 | 3-5 天 |
| [Phase 5](phase5_distributed_inference.md) | 分布式推理引擎（RPC 模式） | P0 | 5-7 天 |
| [Phase 6](phase6_storage_pool.md) | 统一存储池与模型同步 | P1 | 4-6 天 |
| [Phase 7](phase7_pipeline_loadbalance.md) | Pipeline 并行与动态负载均衡 | P1 | 5-7 天 |
| [Phase 8](phase8_security_production.md) | 安全加固与生产就绪 | P1 | 3-5 天 |

---

## 技术选型

| 功能 | 选型 | 理由 |
|------|------|------|
| 硬件检测 | `psutil` + `pynvml` / `GPUtil` | 跨平台、成熟稳定 |
| 分布式推理 | `llama-server --rpc` | llama.cpp 原生支持 |
| 模型同步 | HTTP 分片下载 + 校验 | 简单可靠 |
| Pipeline 并行 | 自定义层分配算法 | 灵活可控 |
| API 安全 | `fastapi` + `slowapi` + SSL | 生态成熟 |

---

## 验收标准

- [ ] 单节点可运行 7B 模型
- [ ] 双节点可协同运行 13B+ 模型
- [ ] Hermes Agent 可正常调用 API
- [ ] 跨平台（macOS/Windows/Linux）测试通过
- [ ] 压力测试：100 并发请求稳定运行
