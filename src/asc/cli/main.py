"""Asc CLI 入口。

提供命令行界面，用于启动 Worker、Master 节点，查看状态和发现网络节点。

支持的命令：
- asc start   : 启动 Worker 节点（默认监听 52415 端口）
- asc master  : 启动 Master 节点（默认监听 52414 端口）
- asc status  : 查看本机硬件资源和 RPC Server 状态
- asc discover: 扫描子网，发现潜在节点

设计原则：
- 简洁：每个命令对应一个明确操作，降低使用门槛
- 信息丰富：启动时自动打印节点 ID、IP、硬件资源等信息
- 信号处理：支持 Ctrl+C 优雅停止，清理资源

使用示例：
    # 启动 Worker
    asc start --port 52415

    # 启动 Master
    asc master --host 0.0.0.0 --port 52414

    # 查看状态
    asc status
"""

from __future__ import annotations

import argparse
import socket
import sys
import uuid


def _generate_node_id() -> str:
    """生成唯一节点 ID。"""
    return f"node-{uuid.uuid4().hex[:8]}"


def _get_local_ip() -> str:
    """获取本机 IP 地址。"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def _cmd_start(args: argparse.Namespace) -> None:
    """启动 Worker 节点。"""
    from asc.worker.agent import WorkerAgent

    node_id = _generate_node_id()
    agent = WorkerAgent(node_id=node_id, port=args.port)

    print(f"[Asc] 启动 Worker 节点: {node_id}")
    print(f"[Asc] 监听端口: {args.port}")
    print(f"[Asc] 本地 IP: {_get_local_ip()}")

    if not args.no_worker:
        resources = agent.get_resources()
        print(f"[Asc] CPU: {resources.cpu_count} 核")
        print(f"[Asc] 内存: {resources.memory_free_mb} MB / {resources.memory_total_mb} MB")
        if resources.gpus:
            for i, gpu in enumerate(resources.gpus):
                print(
                    f"[Asc] GPU {i}: {gpu.name},"
                    f" VRAM {gpu.vram_free_mb} MB / {gpu.vram_total_mb} MB"
                )
        else:
            print("[Asc] GPU: 未检测到")

    if not args.no_discover:
        print("[Asc] 节点发现已启用")

    print("[Asc] 按 Ctrl+C 停止节点")
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[Asc] 正在停止节点...")
        agent.stop_rpc()
        print("[Asc] 节点已停止")


def _cmd_status(_args: argparse.Namespace) -> None:
    """查看本机节点状态。"""
    from asc.worker.agent import WorkerAgent

    agent = WorkerAgent(node_id="cli-status", port=0)
    resources = agent.get_resources()
    rpc = agent.rpc_status()

    print("[Asc] 节点状态")
    print(f"  CPU: {resources.cpu_count} 核 ({resources.cpu_percent}%)")
    print(f"  内存: {resources.memory_free_mb} MB / {resources.memory_total_mb} MB")
    if resources.gpus:
        for i, gpu in enumerate(resources.gpus):
            print(f"  GPU {i}: {gpu.name}")
            print(f"    VRAM: {gpu.vram_free_mb} MB / {gpu.vram_total_mb} MB")
    else:
        print("  GPU: 未检测到")
    print(f"  计算评分: {resources.compute_score:.2f}")
    print(f"  RPC Server: {'运行中' if rpc.get('running') else '未运行'}")
    if rpc.get("running"):
        print(f"    端点: {rpc.get('host')}:{rpc.get('port')}")


def _cmd_discover(_args: argparse.Namespace) -> None:
    """发现网络节点。"""
    from asc.network.discovery import NodeDiscovery

    node_id = _generate_node_id()
    discovery = NodeDiscovery(node_id=node_id, port=52415)
    addr = discovery.broadcast_address

    print("[Asc] 节点发现服务")
    print(f"[Asc] 广播地址: {addr[0]}:{addr[1]}")
    print("[Asc] 扫描子网节点...")

    local_ip = _get_local_ip()
    subnet = ".".join(local_ip.split(".")[:3])
    addresses = discovery.scan_addresses(subnet)
    print(f"[Asc] 扫描范围: {subnet}.1 - {subnet}.254 ({len(addresses)} 个地址)")
    print("[Asc] 发现完成（发现功能需配合运行中的节点使用）")


def _cmd_master(args: argparse.Namespace) -> None:
    """启动 Master 节点。"""
    import asyncio

    from asc.master.main import MasterNode

    node_id = args.node_id or _generate_node_id()
    master = MasterNode(node_id=node_id)

    print(f"[Asc] 启动 Master 节点: {node_id}")
    print(f"[Asc] 监听地址: {args.host}:{args.port}")
    print("[Asc] 按 Ctrl+C 停止节点")

    try:
        asyncio.run(master.run(host=args.host, port=args.port))
    except KeyboardInterrupt:
        print("\n[Asc] 正在停止 Master...")
        asyncio.run(master.stop())
        print("[Asc] Master 已停止")


def main() -> None:
    """Asc 命令行入口。"""
    parser = argparse.ArgumentParser(
        prog="asc",
        description="Asc - 跨平台分布式 LLM 推理系统",
    )
    parser.add_argument("--version", action="version", version="asc 0.1.0")

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # start 命令
    start_parser = subparsers.add_parser("start", help="启动 Asc Worker 节点")
    start_parser.add_argument("--port", type=int, default=52415, help="服务端口")
    start_parser.add_argument("--no-worker", action="store_true", help="不启动 Worker")
    start_parser.add_argument("--no-discover", action="store_true", help="禁用节点发现")

    # master 命令
    master_parser = subparsers.add_parser("master", help="启动 Asc Master 节点")
    master_parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    master_parser.add_argument("--port", type=int, default=52414, help="监听端口")
    master_parser.add_argument("--node-id", type=str, default=None, help="节点 ID")

    # status 命令
    subparsers.add_parser("status", help="查看集群状态")

    # discover 命令
    subparsers.add_parser("discover", help="发现网络节点")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    command_map = {
        "start": _cmd_start,
        "master": _cmd_master,
        "status": _cmd_status,
        "discover": _cmd_discover,
    }

    handler = command_map.get(args.command)
    if handler is not None:
        handler(args)
    else:
        print(f"[Asc] 未知命令: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
