# ASC (All System Cluster) - 分布式AI推理框架

## 项目简介

ASC (All System Cluster) 是一个独立开发的分布式AI推理框架，支持多节点并行推理、P2P自动组网、可视化监控和跨平台部署。

本项目为独立原创实现，仅参考分布式系统的通用设计思路。

## 核心特性

- **分布式推理**: 支持多节点并行推理，自动任务分配
- **P2P组网**: UDP广播自动发现节点，支持手动指定IP连接
- **可视化控制台**: Web界面实时监控节点状态和推理任务
- **跨平台支持**: Linux、macOS、Windows 三大操作系统
- **简洁CLI**: 一键启动，降低使用门槛
- **独立架构**: 不依赖 PyTorch/TensorFlow，使用 llama.cpp 作为后端引擎

## 项目架构

```
asc/
├── src/
│   ├── core/              # 核心模块
│   │   ├── node.py        # 节点管理（状态、资源、心跳）
│   │   ├── cluster.py     # 集群管理（节点发现、调度）
│   │   ├── config.py      # 配置管理
│   │   └── logging.py     # 日志管理
│   ├── network/           # 网络通信
│   │   ├── discovery.py   # 节点发现（UDP广播 + HTTP健康检查）
│   │   └── protocol.py    # 通信协议（消息格式、验证）
│   ├── inference/         # 推理引擎
│   │   └── engine.py      # llama.cpp 封装（本地/分布式推理）
│   ├── ui/                # 可视化界面
│   │   ├── server.py      # Web服务器（Flask + SocketIO）
│   │   └── templates/     # HTML模板
│   └── cli/               # 命令行接口
│       └── main.py        # CLI入口
├── models/                # 模型目录（用户自行放置）
├── logs/                  # 日志目录
├── LICENSE                # Apache 2.0 许可证
├── README.md              # 项目说明
└── requirements.txt       # Python依赖
```

## 环境要求

- Python 3.10+
- llama.cpp（需自行编译）
- 支持平台：Windows / Linux / macOS

## 安装步骤

```bash
# 克隆项目
git clone https://github.com/OscarJHLin/asc.git
cd asc

# 安装Python依赖
pip install -r requirements.txt

# 编译 llama.cpp（请参考 llama.cpp 官方文档）
# Windows: 使用 CMake + Visual Studio
# Linux:   cmake -B build && cmake --build build
# macOS:   cmake -B build -DLLAMA_METAL=ON && cmake --build build

# 放置模型文件到 models/ 目录
# 模型格式：GGUF
```

## 使用说明

### 启动节点

```bash
# 启动节点并自动发现（默认模式）
python -m asc

# 启动节点，禁用Web UI
python -m asc start --no-ui

# 启动节点，禁用自动发现
python -m asc start --no-discover

# 工作节点连接指定主节点
python -m asc start --mode worker --master-ip 192.168.1.100
```

### 运行推理

```bash
# 单节点推理
python -m asc infer --model models/model.gguf --prompt "Hello"

# 指定参数
python -m asc infer \
  --model models/model.gguf \
  --prompt "Explain quantum computing" \
  --max-tokens 256 \
  --temperature 0.8
```

### 查看状态

```bash
# 查看本地节点状态
python -m asc status

# 发现网络中的节点
python -m asc discover
```

### Web UI

启动节点后，访问 http://localhost:8080 查看可视化控制台。

## 技术架构

### 网络层
- **发现机制**: UDP广播 + HTTP健康检查
- **连接管理**: TCP长连接，自动重连
- **协议设计**: JSON消息协议，支持心跳、任务分配、状态同步

### 推理层
- **后端引擎**: llama.cpp（外部依赖，通过命令行调用）
- **任务执行**: 本地子进程方式，支持流式输出
- **模型管理**: 自动查找和加载 GGUF 模型

### UI层
- **实时监控**: WebSocket推送节点状态
- **任务管理**: 可视化推理任务提交和结果展示
- **REST API**: HTTP接口供外部调用

## 配置说明

配置文件为 JSON 格式，可通过环境变量覆盖：

| 环境变量 | 说明 | 默认值 |
|---------|------|--------|
| ASC_NODE_PORT | 节点HTTP端口 | 52415 |
| ASC_RPC_PORT | RPC端口 | 50052 |
| ASC_NODE_NAME | 节点名称 | 主机名 |
| ASC_DISCOVERY_PORT | 发现端口 | 52416 |
| ASC_UI_PORT | Web UI端口 | 8080 |
| ASC_MODELS_PATH | 模型目录 | ./models |

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
