"""Asc CLI 入口。

提供命令行界面，用于启动 Worker、Master 节点，查看状态和发现网络节点。

支持的命令：
- asc start   : 启动节点（首次运行引导选择 master/worker）
- asc master  : 启动 Master 节点（默认监听 52414 端口）
- asc status  : 查看本机硬件资源和 RPC Server 状态
- asc discover: 扫描子网，发现潜在节点

设计原则：
- 简洁：每个命令对应一个明确操作，降低使用门槛
- 信息丰富：启动时自动打印节点 ID、IP、硬件资源等信息
- 信号处理：支持 Ctrl+C 优雅停止，清理资源

使用示例：
    # 启动节点（首次运行会引导选择角色）
    asc start

    # 直接启动 Master
    asc master --host 0.0.0.0 --port 52414

    # 查看状态
    asc status
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import logging
import os
import secrets
import socket
import sys
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _get_hostname() -> str:
    """获取主机名。"""
    return socket.gethostname()


def _generate_node_id() -> str:
    """生成唯一节点 ID，基于主机名。"""
    hostname = _get_hostname()
    return f"{hostname}-{uuid.uuid4().hex[:4]}"


def _get_local_ip() -> str:
    """获取本机 IP 地址。"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def _get_config_dir() -> Path:
    """获取配置目录。

    优先级：
    1. ASC_CONFIG_DIR 环境变量
    2. 默认 ~/.asc
    """
    env_dir = os.getenv("ASC_CONFIG_DIR")
    if env_dir:
        config_dir = Path(env_dir)
    else:
        config_dir = Path.home() / ".asc"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def _get_first_run_flag() -> Path:
    """获取首次运行标志文件路径。

    标志文件与配置文件同目录，文件名包含配置文件名以区分不同角色。
    例如：node_config_master.json -> .node_config_master_first_run
    """
    config_path = _get_node_config_path()
    config_name = config_path.stem  # e.g. "node_config_master"
    return config_path.parent / f".{config_name}_first_run"


def _is_first_run() -> bool:
    """检查是否为首次运行。"""
    return not _get_first_run_flag().exists()


def _mark_first_run_complete() -> None:
    """标记首次运行已完成。"""
    _get_first_run_flag().touch()


def _get_node_config_path() -> Path:
    """获取节点配置文件路径。

    优先级：
    1. ASC_CONFIG_PATH 环境变量（完整文件路径）
    2. --config 指定的路径
    3. ASC_CONFIG_DIR 环境变量下的角色专属配置文件
    4. 默认 ~/.asc/node_config.json（无角色时）或 ~/.asc/node_config_{role}.json

    当 --role 或角色子命令指定了角色时，使用角色专属配置文件，
    避免 Master 和 Worker 共享同一配置文件导致角色冲突。
    """
    # 环境变量指定的完整路径优先
    env_path = os.getenv("ASC_CONFIG_PATH")
    if env_path:
        return Path(env_path)

    # --config 指定的路径
    if _custom_config_path is not None:
        return _custom_config_path

    # 角色专属配置文件
    config_dir = _get_config_dir()
    if _active_role is not None:
        return config_dir / f"node_config_{_active_role}.json"

    # 兜底：无角色时使用通用配置
    return config_dir / "node_config.json"


# 可通过 --config 参数覆盖的配置文件路径
_custom_config_path: Path | None = None

# 当前激活的角色（由 --role 或子命令 master/worker 设定）
_active_role: str | None = None


def _set_config_path(path: str) -> None:
    """设置自定义配置文件路径（由 --config 参数触发）。"""
    global _custom_config_path
    _custom_config_path = Path(path).resolve()


def _set_active_role(role: str | None) -> None:
    """设置当前激活的角色，影响配置文件路径。"""
    global _active_role
    _active_role = role


def _load_node_config() -> dict[str, Any]:
    """加载节点配置。

    优先级：
    1. --config 指定的路径
    2. 默认 _get_node_config_path()
    """
    if _custom_config_path is not None:
        path = _custom_config_path
    else:
        path = _get_node_config_path()
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _save_node_config(config: dict[str, Any]) -> None:
    """保存节点配置。"""
    if _custom_config_path is not None:
        path = _custom_config_path
    else:
        path = _get_node_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")


def _prompt_role() -> str:
    """交互式提示用户选择节点角色。"""
    print("\n" + "=" * 50)
    print("  欢迎使用 ASC 分布式 LLM 推理集群")
    print("=" * 50)
    print("\n请选择节点角色：")
    print("  1. Master 节点 - 集群控制平面，提供管理界面")
    print("  2. Worker 节点 - 计算节点，执行推理任务")
    print()

    while True:
        choice = input("请输入选项 (1/2): ").strip()
        if choice == "1":
            return "master"
        elif choice == "2":
            return "worker"
        print("无效选项，请重新输入。")


def _prompt_master_config() -> dict[str, Any]:
    """交互式配置 Master 节点。"""
    print("\n--- Master 节点配置 ---")

    # 生成默认密码
    default_password = secrets.token_hex(8)
    print("\n默认管理员密码已生成（请妥善保存，首次登录需要使用）")
    print("密码: ********")

    custom_pwd = input("是否自定义密码? (y/N): ").strip().lower()
    if custom_pwd == "y":
        password = getpass.getpass("请输入密码: ").strip()
        if not password:
            password = default_password
            print("密码为空，使用默认密码: ********")
    else:
        password = default_password

    # API 配置
    enable_api = input("\n是否启用对外 API 服务? (Y/n): ").strip().lower()
    api_config = {"enabled": enable_api != "n"}

    if api_config["enabled"]:
        api_key = input("请输入 API Key (留空则自动生成): ").strip()
        if not api_key:
            api_key = secrets.token_hex(16)
            print(f"API Key 已自动生成: {api_key}")
        api_config["api_key"] = api_key

    return {
        "password": password,
        "api_config": api_config,
    }


def _prompt_worker_config() -> dict[str, Any]:
    """交互式配置 Worker 节点。"""
    print("\n--- Worker 节点配置 ---")

    # 询问 Master 地址
    print("\nWorker 节点需要连接到 Master 节点。")
    print("如果 Master 和 Worker 在同一网络，可以使用自动发现。")

    auto_discover = input("是否启用自动发现 Master? (Y/n): ").strip().lower()

    master_host = None
    master_port = 52414

    if auto_discover == "n":
        master_host = input("请输入 Master 节点 IP: ").strip()
        port_input = input("请输入 Master 节点端口 (默认 52414): ").strip()
        if port_input:
            try:
                master_port = int(port_input)
            except ValueError:
                print("端口无效，使用默认 52414")
                master_port = 52414

    return {
        "auto_discover": auto_discover != "n",
        "master_host": master_host,
        "master_port": master_port,
    }


def _run_first_time_setup(*, non_interactive: bool = False, skip_benchmark: bool = False) -> dict[str, Any]:
    """执行首次运行设置流程。

    Args:
        non_interactive: 非交互模式，从环境变量读取配置，不等待 stdin 输入。
        skip_benchmark: 跳过基准测试步骤。
    """
    # 运行环境检测
    from asc.worker.node_setup import NodeSetup

    node_id = os.getenv("ASC_NODE_ID") or _generate_node_id()
    setup = NodeSetup(node_id=node_id, skip_benchmark=skip_benchmark, non_interactive=non_interactive)

    if not non_interactive:
        print("\n[Asc] 首次运行，开始初始化配置...")
        print("\n[Asc] 正在检测环境...")

    result = setup.run()

    if not non_interactive:
        for step in result.steps:
            status_icon = {
                "completed": "✓",
                "failed": "✗",
                "skipped": "○",
            }.get(step.status, "·")
            print(f"  {status_icon} {step.name}: {step.message}")

        if result.success:
            print(f"\n[Asc] 环境配置完成，性能评分: {result.benchmark_score:.2f}")
        else:
            print("\n[Asc] 环境配置部分失败，节点将以受限模式启动")

    # 选择角色
    if non_interactive:
        role = os.getenv("ASC_ROLE", "worker").lower()
        if role not in ("master", "worker"):
            logger.warning("ASC_ROLE 环境变量无效 (%s)，使用默认值 worker", role)
            role = "worker"
    else:
        role = _prompt_role()

    config: dict[str, Any] = {
        "node_id": node_id,
        "role": role,
        "first_run": result.to_dict(),
    }

    if role == "master":
        if non_interactive:
            master_config = _non_interactive_master_config()
        else:
            master_config = _prompt_master_config()
        config.update(master_config)
    else:
        if non_interactive:
            worker_config = _non_interactive_worker_config()
        else:
            worker_config = _prompt_worker_config()
        config.update(worker_config)

    # 保存配置
    _save_node_config(config)
    _mark_first_run_complete()

    if not non_interactive:
        print(f"\n[Asc] 配置已保存到 {_get_node_config_path()}")
    else:
        logger.info("非交互模式：配置已保存到 %s", _get_node_config_path())

    return config


def _non_interactive_master_config() -> dict[str, Any]:
    """非交互模式下从环境变量获取 Master 配置。"""
    password = os.getenv("ASC_PASSWORD") or secrets.token_hex(8)
    api_key = os.getenv("ASC_ADMIN_API_KEY") or os.getenv("ASC_API_KEY") or secrets.token_hex(16)

    api_config: dict[str, Any] = {"enabled": True, "api_key": api_key}

    return {
        "password": password,
        "api_config": api_config,
    }


def _non_interactive_worker_config() -> dict[str, Any]:
    """非交互模式下从环境变量获取 Worker 配置。"""
    master_host = os.getenv("ASC_MASTER_HOST")
    master_port = int(os.getenv("ASC_MASTER_PORT", "52414"))
    auto_discover = master_host is None

    return {
        "auto_discover": auto_discover,
        "master_host": master_host,
        "master_port": master_port,
    }


def _print_master_info(config: dict[str, Any], host: str, port: int, api_port: int | None) -> None:
    """打印 Master 节点启动信息。"""
    local_ip = _get_local_ip()
    password = config.get("password", "unknown")

    print("\n" + "=" * 50)
    print("  Master 节点已启动")
    print("=" * 50)
    print("\n  管理界面地址:")
    print(f"    http://{local_ip}:{api_port or port}")
    print(f"    http://127.0.0.1:{api_port or port}")
    print("\n  默认密码: ********")
    print(f"\n  TCP 监听: {host}:{port}")
    if api_port:
        print(f"  API 监听: {host}:{api_port}")
    print("\n" + "=" * 50)


def _print_worker_info(config: dict[str, Any], port: int) -> None:
    """打印 Worker 节点启动信息。"""
    local_ip = _get_local_ip()

    print("\n" + "=" * 50)
    print("  Worker 节点已启动")
    print("=" * 50)
    print("\n  节点控制界面:")
    print(f"    http://{local_ip}:{port + 1000}")
    print(f"    http://127.0.0.1:{port + 1000}")
    print(f"\n  节点 ID: {config.get('node_id', 'unknown')}")
    print(f"  监听端口: {port}")

    if config.get("auto_discover"):
        print("  Master 发现: 自动发现模式")
    elif config.get("master_host"):
        print(f"  Master 地址: {config['master_host']}:{config.get('master_port', 52414)}")
    print("\n" + "=" * 50)


def _cmd_start(args: argparse.Namespace) -> None:
    """启动节点（支持首次运行引导）。"""
    # --config 参数覆盖配置文件路径
    if getattr(args, 'config', None):
        _set_config_path(args.config)

    # --role 参数设定角色，影响配置文件路径
    role_arg = getattr(args, 'role', None)
    if role_arg:
        _set_active_role(role_arg)

    config = _load_node_config()

    non_interactive = getattr(args, 'non_interactive', False)
    skip_benchmark = getattr(args, 'skip_benchmark', False)

    # 首次运行或强制重新配置
    if _is_first_run() or args.reconfigure:
        config = _run_first_time_setup(non_interactive=non_interactive, skip_benchmark=skip_benchmark)

    # --role 参数优先于配置文件中的角色
    if role_arg:
        role = role_arg
        config["role"] = role
    else:
        role = config.get("role", "worker")

    # --master-host / --master-port 参数（Worker 角色）
    master_host_arg = getattr(args, 'master_host', None)
    master_port_arg = getattr(args, 'master_port', None)
    if master_host_arg:
        config["master_host"] = master_host_arg
        config["auto_discover"] = False
    if master_port_arg:
        config["master_port"] = master_port_arg

    if role == "master":
        _start_master(args, config)
    else:
        _start_worker(args, config)


def _start_master(args: argparse.Namespace, config: dict[str, Any]) -> None:
    """启动 Master 节点。"""

    from asc.api.auth import ApiKeyStore, set_key_store
    from asc.master.main import MasterNode

    node_id = getattr(args, 'node_id', None) or config.get("node_id", _generate_node_id())

    # 确定端口
    tcp_port = args.port if args.port != 52414 else config.get("tcp_port", 52414)
    api_port = args.api_port if args.api_port is not None else config.get("api_port", 8080)

    # 保存实际使用的端口
    config["tcp_port"] = tcp_port
    config["api_port"] = api_port
    _save_node_config(config)

    # 从配置文件中注入 api_key 到 ApiKeyStore
    api_key = config.get("api_config", {}).get("api_key")
    if api_key:
        # 合并环境变量中的密钥和配置文件中的密钥
        env_store = ApiKeyStore.from_env()
        admin_keys = list(env_store._admin_keys) + [api_key]
        user_keys = list(env_store._user_keys)
        set_key_store(ApiKeyStore(user_keys=user_keys, admin_keys=admin_keys))
        logger.info("已从配置文件注入 API Key 到 ApiKeyStore")
    else:
        # 仅使用环境变量中的密钥
        set_key_store(ApiKeyStore.from_env())

    master = MasterNode(
        node_id=node_id,
        api_host=args.host,
        api_port=api_port,
    )

    _print_master_info(config, args.host, tcp_port, api_port)

    try:
        asyncio.run(master.run(host=args.host, port=tcp_port))
    except KeyboardInterrupt:
        print("\n[Asc] 正在停止 Master...")
        asyncio.run(master.stop())
        print("[Asc] Master 已停止")


def _start_worker(args: argparse.Namespace, config: dict[str, Any]) -> None:
    """启动 Worker 节点。"""

    from asc.worker.agent import WorkerAgent

    node_id = config.get("node_id", _generate_node_id())
    port = args.port

    agent = WorkerAgent(node_id=node_id, port=port, skip_benchmark=getattr(args, 'skip_benchmark', False))

    # 启动节点控制 GUI
    from asc.worker.node_gui import run_node_gui

    gui_port = port + 1000
    gui_task = None

    _print_worker_info(config, port)

    # 获取 Master 地址
    master_host = config.get("master_host")
    master_port = config.get("master_port", 52414)

    if config.get("auto_discover", True) and not master_host:
        # 自动发现 Master
        from asc.network.discovery import NodeDiscovery

        discovery = NodeDiscovery(node_id=node_id, port=port, role="worker")
        print("[Asc] 正在扫描 Master 节点...")

        # 使用异步发现
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            discovered_ip = loop.run_until_complete(
                discovery.discover_master(local_ip=_get_local_ip(), timeout=10.0)
            )
            if discovered_ip:
                master_host = discovered_ip
                print(f"[Asc] 发现 Master 节点: {master_host}:{master_port}")
            else:
                print("[Asc] 未在局域网内发现 Master 节点")
                print("[Asc] 请手动指定 Master 地址，或使用 asc start --reconfigure 重新配置")
                return
        except Exception as e:
            print(f"[Asc] 自动发现失败: {e}")
            print("[Asc] 请手动指定 Master 地址")
            return

    # 启动 GUI 服务器（在后台）
    gui_thread = None
    try:
        import threading

        gui_thread = threading.Thread(
            target=run_node_gui,
            args=(agent, gui_port),
            daemon=True,
        )
        gui_thread.start()
        print(f"[Asc] 节点控制界面已启动: http://{_get_local_ip()}:{gui_port}")
    except Exception as e:
        print(f"[Asc] 节点控制界面启动失败: {e}")

    # 连接 Master 并运行
    if master_host:
        print(f"[Asc] 正在连接 Master {master_host}:{master_port}...")
        try:
            asyncio.run(agent.run(master_host=master_host, master_port=master_port))
        except KeyboardInterrupt:
            print("\n[Asc] 正在停止 Worker...")
            asyncio.run(agent.stop())
            print("[Asc] Worker 已停止")
    else:
        # 不连接 Master，仅作为独立节点运行
        print("[Asc] 以独立模式运行，未连接 Master")
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
    """直接启动 Master 节点（保留向后兼容）。"""
    # --config 参数覆盖配置文件路径
    if getattr(args, 'config', None):
        _set_config_path(args.config)

    # 设定角色为 master，使用角色专属配置文件
    _set_active_role("master")

    config = _load_node_config()

    non_interactive = getattr(args, 'non_interactive', False)
    skip_benchmark = getattr(args, 'skip_benchmark', False)

    # 确保有配置
    if not config or config.get("role") != "master":
        config = _run_first_time_setup(non_interactive=non_interactive, skip_benchmark=skip_benchmark)
        config["role"] = "master"

    _start_master(args, config)


def _cmd_worker(args: argparse.Namespace) -> None:
    """直接启动 Worker 节点。"""
    # --config 参数覆盖配置文件路径
    if getattr(args, 'config', None):
        _set_config_path(args.config)

    # 设定角色为 worker，使用角色专属配置文件
    _set_active_role("worker")

    config = _load_node_config()

    non_interactive = getattr(args, 'non_interactive', False)
    skip_benchmark = getattr(args, 'skip_benchmark', False)

    # 首次运行或强制重新配置
    if _is_first_run() or getattr(args, 'reconfigure', False):
        config = _run_first_time_setup(non_interactive=non_interactive, skip_benchmark=skip_benchmark)

    # --role 参数优先
    role_arg = getattr(args, 'role', None)
    if role_arg:
        config["role"] = role_arg
    else:
        config["role"] = "worker"

    # Worker 子命令可以指定 Master 地址
    master_host = getattr(args, 'master_host', None)
    master_port = getattr(args, 'master_port', None)
    if master_host:
        config["master_host"] = master_host
        config["auto_discover"] = False
    if master_port:
        config["master_port"] = master_port

    _start_worker(args, config)


def _setup_logging() -> None:
    """配置 asc 包的日志输出。

    默认 Python root logger 只输出 WARNING 及以上级别到 stderr。
    此函数为 asc 包配置 INFO 级别输出到 stderr，并可选输出到文件。
    """
    asc_logger = logging.getLogger("asc")
    asc_logger.setLevel(logging.INFO)

    # 避免重复添加 handler（main() 可能被多次调用）
    if asc_logger.handlers:
        return

    # stderr handler（容忍 daemon 线程中流已关闭的情况）
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.INFO)
    stderr_handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    )
    # 防止 daemon 线程日志写入已关闭的流时抛出异常
    original_emit = stderr_handler.emit
    def _safe_emit(record):
        try:
            original_emit(record)
        except (OSError, ValueError):
            pass
    stderr_handler.emit = _safe_emit
    asc_logger.addHandler(stderr_handler)

    # 文件 handler（写入配置目录下的 asc.log）
    try:
        log_path = _get_config_dir() / "asc.log"
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        asc_logger.addHandler(file_handler)
    except Exception:
        # 文件 handler 创建失败不影响主流程
        pass


def main() -> None:
    """Asc 命令行入口。"""
    parser = argparse.ArgumentParser(
        prog="asc",
        description="Asc - 跨平台分布式 LLM 推理系统",
    )
    parser.add_argument("--version", action="version", version="asc 0.1.0")

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # start 命令（支持引导）
    start_parser = subparsers.add_parser("start", help="启动 Asc 节点（首次运行自动引导）")
    start_parser.add_argument("--port", type=int, default=52415, help="服务端口")
    start_parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    start_parser.add_argument("--api-port", type=int, default=None, help="API/GUI 端口")
    start_parser.add_argument("--role", type=str, choices=["master", "worker"], default=None,
                              help="节点角色（master/worker），指定后使用角色专属配置文件")
    start_parser.add_argument("--reconfigure", action="store_true", help="重新配置节点")
    start_parser.add_argument("--skip-benchmark", action="store_true", help="跳过基准测试")
    start_parser.add_argument("--config", type=str, default=None, help="指定配置文件路径")
    start_parser.add_argument("--non-interactive", action="store_true", help="非交互模式，从环境变量读取配置")
    start_parser.add_argument("--master-host", type=str, default=None, help="Master 节点地址（Worker 角色）")
    start_parser.add_argument("--master-port", type=int, default=None, help="Master 节点端口（Worker 角色）")

    # master 命令（直接启动，保留兼容）
    master_parser = subparsers.add_parser("master", help="启动 Asc Master 节点")
    master_parser.add_argument("--host", type=str, default="0.0.0.0", help="TCP 监听地址")
    master_parser.add_argument("--port", type=int, default=52414, help="TCP 监听端口")
    master_parser.add_argument("--api-host", type=str, default="0.0.0.0", help="API Server 监听地址")
    master_parser.add_argument("--api-port", type=int, default=8080, help="API Server 监听端口")
    master_parser.add_argument("--node-id", type=str, default=None, help="节点 ID")
    master_parser.add_argument("--config", type=str, default=None, help="指定配置文件路径")
    master_parser.add_argument("--non-interactive", action="store_true", help="非交互模式，从环境变量读取配置")
    master_parser.add_argument("--skip-benchmark", action="store_true", help="跳过基准测试")

    # worker 命令（直接启动 Worker 节点）
    worker_parser = subparsers.add_parser("worker", help="启动 Asc Worker 节点")
    worker_parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    worker_parser.add_argument("--port", type=int, default=52415, help="服务端口")
    worker_parser.add_argument("--master-host", type=str, default=None, help="Master 节点地址")
    worker_parser.add_argument("--master-port", type=int, default=None, help="Master 节点端口")
    worker_parser.add_argument("--config", type=str, default=None, help="指定配置文件路径")
    worker_parser.add_argument("--non-interactive", action="store_true", help="非交互模式，从环境变量读取配置")
    worker_parser.add_argument("--skip-benchmark", action="store_true", help="跳过基准测试")
    worker_parser.add_argument("--reconfigure", action="store_true", help="重新配置节点")

    # status 命令
    subparsers.add_parser("status", help="查看集群状态")

    # discover 命令
    subparsers.add_parser("discover", help="发现网络节点")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    _setup_logging()

    command_map = {
        "start": _cmd_start,
        "master": _cmd_master,
        "worker": _cmd_worker,
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
