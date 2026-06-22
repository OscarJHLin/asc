# ASC (All System Cluster) - 分布式 LLM 推理框架

## 项目简介

ASC (All System Cluster) 是一个独立开发的分布式 LLM 推理框架，支持多节点并行推理、自动组网、模型分发、可视化管理和跨平台部署。

本项目为独立原创实现，仅参考分布式系统的通用设计思路。

## 核心特性

- **分布式推理**: 多节点并行推理，按 VRAM 比例自动分配模型层，支持 Tensor/Pipeline 并行
- **模型分发**: Master 节点下载模型后通过内网 HTTP 分发到各 Worker，自动规划层分配方案
- **自动组网**: UDP 广播自动发现 Master 节点，Worker 零配置加入集群
- **可视化管理**: Master GUI 管理集群、部署模型、配置 API；Worker GUI 查看硬件状态
- **事件驱动架构**: 不可变状态 + 事件溯源，支持 Master 故障恢复和状态重建
- **跨平台**: Linux、macOS、Windows，基于 llama.cpp 后端
- **OpenAI 兼容 API**: FastAPI 提供 OpenAI 格式的推理接口，可配置 API Key

## 组网部署要求

ASC 的组网**完全由项目自身完成**，不需要借助 SSH、Ansible、Docker 等外部部署工具。唯一需要提前准备的基础配置是**防火墙端口开放**和**llama.cpp 编译**。

### 各节点独立安装

每个节点只需独立克隆项目、安装依赖、运行 `asc start` 即可。节点间通过 UDP 广播自动发现，通过 Binary Frame 协议自动注册和通信。

### 防火墙配置

所有节点必须在同一局域网（或路由可达），并开放以下端口：

| 端口 | 协议 | 方向 | 说明 |
|------|------|------|------|
| 52414 | TCP | 入站 | Master 的 Binary Frame 监听端口（Worker 连接） |
| 8080 | TCP | 入站 | Master 的 HTTP API / 管理面板 |
| 52416 | UDP | 入站 | 节点发现广播端口（Master/Worker 互相发现） |

#### Linux (ufw) 示例

```bash
sudo ufw allow 52414/tcp
sudo ufw allow 8080/tcp
sudo ufw allow 52416/udp
```

#### Linux (iptables) 示例

```bash
sudo iptables -A INPUT -p tcp --dport 52414 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 8080 -j ACCEPT
sudo iptables -A INPUT -p udp --dport 52416 -j ACCEPT
```

> **注意**：SSH（端口 22 或其他）仅在开发、调试或手动运维时需要。正常的 ASC 组网部署**不依赖 SSH**。

### llama.cpp 编译

ASC 使用 llama.cpp 作为推理后端，需要在每个 Worker 节点上编译安装 `llama-server`。

#### 自动编译脚本（推荐）

项目提供一键编译脚本，自动检测 GPU 类型（NVIDIA CUDA / AMD ROCm / CPU）：

```bash
# 交互模式（会提示确认）
bash setup_llama_cpp.sh

# 自动模式（使用默认选项，无需交互）
bash setup_llama_cpp.sh --accept-defaults
```

脚本会自动：
1. 检测 GPU 类型（nvidia-smi / rocm-smi / lspci）
2. 下载并安装 CUDA Toolkit（NVIDIA）或提示安装 ROCm（AMD）
3. 克隆并编译 llama.cpp（启用对应 GPU 后端）
4. 安装 `llama-server` 到 `~/.local/bin/`

#### 手动编译

**NVIDIA GPU（CUDA）：**

```bash
# 1. 安装 CUDA Toolkit（用户级，无需 sudo）
wget https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda_12.6.3_560.35.05_linux.run
sh cuda_12.6.3_560.35.05_linux.run --toolkit --toolkitpath=$HOME/.local/cuda-toolkit --silent

# 2. 设置环境变量（加入 ~/.bashrc）
export CUDA_PATH=$HOME/.local/cuda-toolkit
export LD_LIBRARY_PATH=$HOME/.local/cuda-toolkit/lib64:$HOME/.local/cuda-toolkit/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export PATH=$HOME/.local/bin:$PATH

# 3. 编译 llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build -j$(nproc)
cmake --install build --prefix $HOME/.local

# 4. 安装共享库（动态编译时需要）
cp build/src/libllama-server-impl.so $HOME/.local/lib/ 2>/dev/null || true
cp build/ggml/src/libggml.so $HOME/.local/lib/ 2>/dev/null || true
export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH
```

**AMD GPU（ROCm）：**

```bash
# 安装 ROCm 后
cmake -B build -DGGML_HIPBLAS=ON
cmake --build build -j$(nproc)
cmake --install build --prefix $HOME/.local
```

**仅 CPU：**

```bash
cmake -B build
cmake --build build -j$(nproc)
cmake --install build --prefix $HOME/.local
```

> **多 GPU 注意**：如果服务器有多张 GPU，ASC 会自动使用单设备模式（`-sm none --device CUDA0`），避免 llama.cpp 的 `GGML_SCHED_MAX_SPLIT_INPUTS` 崩溃。

### 环境变量

| 环境变量 | 说明 | 默认值 |
|---------|------|--------|
| ASC_ADMIN_API_KEY | Admin API 密钥 | 无（启动时生成） |
| ASC_ALLOW_NO_AUTH | 允许无认证访问 | 0（不推荐） |
| ASC_ROLE | 节点角色（master/worker） | 无（交互选择） |
| ASC_MASTER_HOST | Master 节点地址 | 自动发现 |
| ASC_MASTER_PORT | Master TCP 端口 | 52414 |
| CUDA_PATH | CUDA Toolkit 路径 | /usr/local/cuda |
| LD_LIBRARY_PATH | 共享库搜索路径 | 系统默认 |

### 模型管理

模型文件统一存放在 Master 节点的 `~/.asc/models/` 目录，仅支持 `.gguf` 格式。

**方式一：WebUI 上传**
- 在管理面板的"模型管理"区域，拖拽 `.gguf` 文件到上传区域
- 文件会流式上传到 Master 的 `~/.asc/models/` 目录

**方式二：手动放置**
- 直接将 `.gguf` 文件复制到 Master 节点的 `~/.asc/models/` 目录
- 在管理面板点击"扫描服务器模型目录"按钮刷新列表

**载入集群**：上传或放置模型后，点击模型旁的"载入集群"按钮，将模型注册到集群供 Worker 使用。

### 非交互模式部署

适用于服务器无终端交互的场景：

```bash
# Master 节点
ASC_ADMIN_API_KEY=your-api-key python -m asc master \
    --host 0.0.0.0 --port 52414 --api-port 8080 \
    --config /path/to/node_config_master.json \
    --skip-benchmark --non-interactive

# Worker 节点
ASC_ADMIN_API_KEY=your-api-key ASC_ROLE=worker \
ASC_MASTER_HOST=192.168.1.10 ASC_MASTER_PORT=52414 \
python -m asc start \
    --config /path/to/node_config.json \
    --skip-benchmark --non-interactive
```

---

## 快速开始

### 安装

**在每个节点上执行：**

```bash
# 克隆项目
git clone https://github.com/OscarJHLin/asc.git
cd asc

# 安装依赖
pip install -e .

# 编译 llama.cpp（参考 llama.cpp 官方文档）
# Linux:   cmake -B build && cmake --build build
# macOS:   cmake -B build -DLLAMA_METAL=ON && cmake --build build
# Windows: cmake -B build && cmake --build build --config Release
```

### 首次启动

**在每个节点上执行：**

```bash
# 启动节点（首次运行自动引导配置）
asc start
```

首次运行会自动进行：
1. 环境检测和性能基准测试
2. 选择节点角色（Master / Worker）
3. 配置节点参数

#### 选择 Master 节点

```
==================================================
  欢迎使用 ASC 分布式 LLM 推理集群
==================================================

请选择节点角色：
  1. Master 节点 - 集群控制平面，提供管理界面
  2. Worker 节点 - 计算节点，执行推理任务

请输入选项 (1/2): 1

--- Master 节点配置 ---

默认管理员密码: a1b2c3d4e5f6g7h8
(请妥善保存，首次登录需要使用)

是否启用对外 API 服务? (Y/n): y
API Key 已自动生成: x1y2z3...

==================================================
  Master 节点已启动
==================================================

  管理界面地址:
    http://192.168.1.10:8080

  默认密码: a1b2c3d4e5f6g7h8

  TCP 监听: 0.0.0.0:52414
  API 监听: 0.0.0.0:8080
==================================================
```

#### 选择 Worker 节点

```
请输入选项 (1/2): 2

--- Worker 节点配置 ---

Worker 节点需要连接到 Master 节点。
是否启用自动发现 Master? (Y/n): y

==================================================
  Worker 节点已启动
==================================================

  节点控制界面:
    http://192.168.1.20:62415

  节点 ID: gpu-server-abcd
  监听端口: 52415
  Master 发现: 自动发现模式
==================================================
```

> Worker 启动后会通过 UDP 广播自动发现同一网络中的 Master，并通过 Binary Frame 协议完成注册、心跳和任务通信。无需手动配置 IP 或借助 SSH 同步。

### 其他命令

```bash
# 直接启动 Master
asc master --host 0.0.0.0 --port 52414

# 查看本机状态
asc status

# 发现网络节点
asc discover

# 重新配置节点
asc start --reconfigure
```

## 项目架构

```
asc/
├── src/asc/
│   ├── cli/                   # 命令行接口
│   │   └── main.py            # CLI 入口（asc start/master/status/discover）
│   ├── master/                # Master 节点
│   │   ├── main.py            # MasterNode（事件驱动状态机）
│   │   ├── orchestrator.py    # DistributedOrchestrator（RPC + llama-server 编排）
│   │   └── model_distributor.py  # 模型分发器（HTTP 分发 + 层分配）
│   ├── worker/                # Worker 节点
│   │   ├── agent.py           # WorkerAgent（注册/心跳/任务执行）
│   │   ├── node_gui.py        # Worker 节点控制 GUI（Flask）
│   │   ├── hardware.py        # 硬件检测（CPU/GPU/内存）
│   │   ├── gpu_info.py        # GPU 信息采集
│   │   ├── rpc_server.py      # RPC Server 管理
│   │   └── benchmark_score.py # 性能评分
│   ├── api/                   # API 服务
│   │   ├── server.py          # FastAPI 服务器（OpenAI 兼容 + Admin API）
│   │   ├── auth.py            # API Key 认证
│   │   └── static/index.html  # Master 管理面板 GUI
│   ├── network/               # 网络通信
│   │   ├── transport.py       # TCP Server/Client（Binary Frame 协议）
│   │   ├── protocol.py        # 消息协议（Envelope/Message/Channel）
│   │   ├── discovery.py       # UDP 广播节点发现
│   │   └── frame.py           # 二进制帧编解码
│   ├── core/                  # 核心模块
│   │   ├── event_log.py       # 事件日志（内存/磁盘持久化）
│   │   ├── cluster_config.py  # 集群配置（资源策略/模型配置）
│   │   ├── model_downloader.py # 模型下载器（HuggingFace/ModelScope）
│   │   └── model_distributor.py # 分片传输模型分发器
│   ├── engine/                # 推理引擎
│   │   └── llama_server.py    # llama.cpp 封装（同步/异步启动）
│   ├── scheduler/             # 调度器
│   │   ├── load_balancer.py   # 负载均衡
│   │   └── placement.py       # 放置策略
│   ├── types/                 # 类型系统
│   │   ├── commands.py        # 命令类型
│   │   ├── events.py          # 事件类型
│   │   ├── state.py           # 状态类型
│   │   └── common.py          # 公共类型
│   └── ui/                    # 旧版 UI（兼容）
│       └── server.py
├── tests/                     # 测试（2000+ 测试用例）
│   ├── integration/           # 集成测试
│   ├── api/                   # API 测试
│   ├── network/               # 网络测试
│   ├── worker/                # Worker 测试
│   ├── master/                # Master 测试
│   ├── core/                  # 核心模块测试
│   ├── engine/                # 引擎测试
│   └── cli/                   # CLI 测试
├── pyproject.toml
└── LICENSE
```

## 技术架构

### 整体流程

```
用户运行 asc start
    │
    ├── 首次运行 → 环境检测 → 选择角色 → 生成配置
    │
    ├── Master 节点
    │   ├── 启动 TCP Server（52414）监听 Worker 连接
    │   ├── 启动 API Server（8080）提供管理界面和推理 API
    │   ├── 接收 Worker 注册（携带主机名、CPU/GPU/内存信息）
    │   ├── 下载模型 → HTTP 分发到 Worker → 规划层分配
    │   └── 通知 Worker 加载对应层 → 编排分布式推理
    │
    └── Worker 节点
        ├── UDP 广播发现 Master → TCP 连接注册
        ├── 启动节点控制 GUI（62415）
        ├── 接收模型文件（Binary Frame 分片） → 加载分配的层
        └── 启动 LlamaServerEngine → 执行推理任务 → 返回结果
```

### 网络层

| 组件 | 协议 | 端口 | 说明 |
|------|------|------|------|
| Master TCP | Binary Frame | 52414 | Worker 连接、心跳、任务分发 |
| Master API | HTTP/WebSocket | 8080 | 管理面板、OpenAI 兼容 API |
| Worker GUI | HTTP | 62415 | 节点控制界面（无需登录） |
| 节点发现 | UDP 广播 | 52416 | Worker 自动发现 Master |

- **Binary Frame 协议**: `[magic:4B][ver:1B][type:1B][len:4B][payload]`
- **消息类型**: NODE_JOINED, HEARTBEAT, TASK_DISPATCH, MODEL_DISTRIBUTE, CONFIG_UPDATE 等
- **安全**: API Key 认证，Admin 端点需认证

### 事件驱动架构

```
Command → MasterNode.process_xxx() → Event → State 更新 → EventLog 持久化
```

- **不可变状态**: 所有状态变更通过事件触发，状态不可直接修改
- **事件溯源**: 从事件日志可完整重建状态，支持 Master 故障恢复
- **确定性**: 相同事件序列产生相同状态

### 模型分发流程

```
1. Master 下载模型（HuggingFace / ModelScope / 手动放置）
2. Master 通过 HTTP 将模型文件分发到各 Worker
3. Master 根据各节点 VRAM 比例规划层分配方案
4. Master 通过 TCP 通知各 Worker 加载对应层
5. Worker 确认模型文件存在，回复 ACK
6. Master 编排分布式推理（Tensor/Pipeline 并行）
```

## Master 管理面板

访问 Master 节点的 `http://<master-ip>:8080`，使用密码登录。

### 功能

- **集群总览**: 节点数、在线节点、总 CPU/内存/显存、平均算力评分
- **子节点管理**: 右上角查看所有已连接节点，点击节点可配置资源限制
- **模型部署**: 支持 HuggingFace、ModelScope、手动下载三种来源
- **API 配置**: 开启/关闭对外 API，配置 API Key 和端口
- **资源策略**: 为每个节点配置显存/内存限制、CPU 卸载比例
- **操作日志**: 实时查看集群操作记录

## Worker 节点控制界面

访问 Worker 节点的 `http://<worker-ip>:62415`，无需登录。

### 显示信息

- **CPU**: 型号、物理/逻辑核心数、频率、占用率
- **内存**: 总量、可用量、使用率
- **GPU**: 每张 GPU 独立显示型号、厂商、总显存、可用显存、使用率
- **连接状态**: 与 Master 的连接状态

## 配置说明

节点配置保存在 `~/.asc/node_config.json`，可通过 `asc start --reconfigure` 重新配置。

### 资源策略

通过 Master GUI 或 Admin API 配置：

| 配置项 | 说明 |
|--------|------|
| vram_limit_mb | 显存使用限制（MB） |
| memory_limit_mb | 内存使用限制（MB） |
| vram_reserve_mb | 显存预留（MB） |
| memory_offload_ratio | CPU 卸载比例（0.0-1.0） |

### API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/v1/chat/completions` | POST | OpenAI 兼容推理接口 |
| `/admin/health` | GET | 集群健康检查 |
| `/admin/nodes` | GET | 节点列表 |
| `/admin/resource-policy/nodes/{id}` | PUT | 更新节点资源策略 |
| `/admin/deploy-model` | POST | 部署模型 |
| `/admin/api-config` | PUT | 更新 API 配置 |
| `/admin/cluster-config` | GET/PUT | 集群配置 |
| `/admin/broadcast-config` | POST | 广播配置到所有节点 |
| `/ws/cluster` | WebSocket | 实时集群状态推送 |

## 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest tests/ -v

# 运行集成测试
pytest tests/integration/ -v

# 代码检查
ruff check src/
mypy src/
```

## 许可证

本项目采用 Apache License 2.0 许可证。

Copyright 2024-2025 Oscar Lin

## 第三方组件声明

- **llama.cpp**: MIT License, Copyright Georgi Gerganov
  - 用途：底层推理引擎（外部依赖，不包含在本项目中）
  - 仓库：https://github.com/ggerganov/llama.cpp

- **exo**: MIT License, Copyright exo-explore
  - 用途：分布式推理概念参考（无代码共享）
  - 仓库：https://github.com/exo-explore/exo

本项目为独立开发，与上述项目无直接关联。所有源代码均为原创实现，
仅参考了分布式系统的通用设计思路。

## 免责声明

本项目按"原样"提供，不提供任何明示或暗示的担保。使用者需自行承担
使用风险，并确保遵守所在地区的相关法律法规。
