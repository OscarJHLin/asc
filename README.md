# ExoLlama - 分布式AI推理框架

## 项目简介

ExoLlama 是一个基于 llama.cpp 的分布式AI推理框架，支持多节点并行推理、P2P组网、可视化监控和跨平台部署。

## 核心特性

- **分布式推理**: 支持多节点并行推理，自动任务分配与结果聚合
- **P2P组网**: 自动节点发现与连接，无需中心服务器
- **可视化控制台**: Web界面实时监控节点状态和推理任务
- **跨平台支持**: Linux、macOS、Windows 三大操作系统
- **简洁CLI**: 一键启动，降低使用门槛

## 项目架构

```
exollama/
├── src/
│   ├── core/           # 核心模块
│   │   ├── node.py     # 节点管理
│   │   ├── cluster.py  # 集群管理
│   │   └── config.py   # 配置管理
│   ├── network/        # 网络通信
│   │   ├── discovery.py # 节点发现
│   │   ├── p2p.py      # P2P连接
│   │   └── protocol.py # 通信协议
│   ├── inference/      # 推理引擎
│   │   ├── engine.py   # 推理引擎
│   │   ├── scheduler.py # 任务调度
│   │   └── aggregator.py # 结果聚合
│   ├── ui/             # 可视化界面
│   │   ├── server.py   # Web服务器
│   │   ├── static/     # 静态资源
│   │   └── templates/  # HTML模板
│   └── cli/            # 命令行接口
│       └── main.py     # CLI入口
├── tests/              # 测试用例
├── docs/               # 文档
├── models/             # 模型目录
├── scripts/            # 辅助脚本
├── LICENSE             # 许可证
└── README.md           # 项目说明
```

## 安装指南

### 环境要求
- Python 3.10+
- CMake 3.20+
- C++ 编译器

### 安装步骤

```bash
# 克隆项目
git clone https://github.com/OscarJHLin/exollama.git
cd exollama

# 安装依赖
pip install -r requirements.txt

# 编译 llama.cpp
python scripts/setup.py

# 启动节点
python -m exollama
```

## 使用说明

### 启动节点
```bash
# 启动主节点
python -m exollama --mode master

# 启动工作节点
python -m exollama --mode worker --master-ip <主节点IP>
```

### 运行推理
```bash
# 单节点推理
python -m exollama infer --model models/model.gguf --prompt "Hello"

# 集群推理
python -m exollama infer --model models/model.gguf --prompt "Hello" --cluster
```

### 查看状态
```bash
# 查看节点状态
python -m exollama status

# 查看集群状态
python -m exollama cluster status
```

## 技术架构

### 网络层
- **发现机制**: UDP广播 + HTTP健康检查
- **连接管理**: TCP长连接，自动重连
- **协议设计**: JSON-RPC over WebSocket

### 推理层
- **任务分割**: 按层分割模型，分配到不同节点
- **负载均衡**: 动态评估节点算力，优化任务分配
- **容错机制**: 节点故障自动迁移任务

### UI层
- **实时监控**: WebSocket推送节点状态
- **任务管理**: 可视化推理任务队列
- **模型管理**: 支持模型上传、转换、删除

## 贡献规范

1. Fork 项目
2. 创建特性分支
3. 提交代码
4. 创建 Pull Request

## 许可证

本项目采用 Apache License 2.0 许可证。

## 致谢

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - 提供底层推理引擎
- [exo](https://github.com/exo-explore/exo) - 提供分布式推理思路参考

## 免责声明

本项目为独立开发，与 exo 和 llama.cpp 项目无直接关联。代码实现均为原创，仅参考了分布式系统的通用设计思路。
