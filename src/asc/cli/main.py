"""Asc CLI 入口。"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    """Asc 命令行入口。"""
    parser = argparse.ArgumentParser(
        prog="asc",
        description="Asc - 跨平台分布式 LLM 推理系统",
    )
    parser.add_argument("--version", action="version", version="asc 0.1.0")

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # start 命令
    start_parser = subparsers.add_parser("start", help="启动 Asc 节点")
    start_parser.add_argument("--port", type=int, default=52415, help="服务端口")
    start_parser.add_argument("--no-worker", action="store_true", help="不启动 Worker")
    start_parser.add_argument("--no-discover", action="store_true", help="禁用节点发现")

    # status 命令
    subparsers.add_parser("status", help="查看集群状态")

    # discover 命令
    subparsers.add_parser("discover", help="发现网络节点")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Phase 2 实现：start / status / discover 的实际逻辑
    print(f"[Asc] 命令 '{args.command}' 尚未实现，Phase 2 开发中")


if __name__ == "__main__":
    main()
