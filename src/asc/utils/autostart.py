"""跨平台自启动管理模块。

在 Linux 上通过 systemd 服务文件管理 ASC Master / Worker 的开机自启；
在 Windows 上通过计划任务（schtasks）管理。
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_IS_LINUX = platform.system() == "Linux"
_IS_WINDOWS = platform.system() == "Windows"

_SYSTEMD_SERVICE_TEMPLATE = """\
[Unit]
Description=ASC {role} Node
After=network.target

[Service]
Type=simple
User={user}
WorkingDirectory={work_dir}
ExecStart={python_path} -m asc.cli.main start --role {role} --host {host} --port {port} --non-interactive --skip-benchmark{master_args}
Restart=on-failure
RestartSec=5
Environment=PATH={python_path_dir}:/usr/local/bin:/usr/bin:/bin
Environment=LD_LIBRARY_PATH={ld_library_path}
Environment=ASC_ROLE={role}
Environment=ASC_MASTER_HOST={master_host}
Environment=ASC_MASTER_PORT={master_port}

[Install]
WantedBy=multi-user.target
"""

_DEFAULTS = {
    "master": {
        "host": "0.0.0.0",
        "port": 52414,
    },
    "worker": {
        "host": "127.0.0.1",
        "port": 52415,
        "master_host": "127.0.0.1",
        "master_port": 52414,
    },
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_role(role: str) -> None:
    if role not in ("master", "worker"):
        raise ValueError(f"Invalid role: {role!r}, must be 'master' or 'worker'")


def _get_user() -> str:
    """获取当前 Linux 用户名。"""
    try:
        import pwd
        return pwd.getpwuid(os.getuid()).pw_name
    except (ImportError, KeyError):
        return os.getlogin()


def _systemd_service_path(role: str) -> Path:
    return Path(f"/etc/systemd/system/asc-{role}.service")


def _windows_task_name(role: str) -> str:
    return f"ASC_{role.capitalize()}"


def _build_service_content(role: str, config: dict) -> str:
    defaults = _DEFAULTS[role]
    host = config.get("host", defaults["host"])
    port = config.get("port", defaults["port"])
    python_path = sys.executable
    python_path_dir = str(Path(python_path).parent)
    work_dir = os.getcwd()
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    user = _get_user()

    # Worker 角色需要 Master 地址参数
    if role == "worker":
        master_host = config.get("master_host", defaults.get("master_host", "127.0.0.1"))
        master_port = config.get("master_port", defaults.get("master_port", 52414))
        master_args = f" --master-host {master_host} --master-port {master_port}"
    else:
        master_host = ""
        master_port = ""
        master_args = ""

    return _SYSTEMD_SERVICE_TEMPLATE.format(
        role=role,
        user=user,
        work_dir=work_dir,
        python_path=python_path,
        host=host,
        port=port,
        python_path_dir=python_path_dir,
        ld_library_path=ld_library_path,
        master_args=master_args,
        master_host=master_host,
        master_port=master_port,
    )


def _build_windows_command(role: str, config: dict) -> list[str]:
    """构建 Windows 计划任务的启动命令。"""
    defaults = _DEFAULTS[role]
    host = config.get("host", defaults["host"])
    port = config.get("port", defaults["port"])
    python_path = sys.executable

    cmd_parts = [
        f'"{python_path}"',
        "-m",
        "asc.cli.main",
        "start",
        "--role",
        role,
        "--host",
        host,
        "--port",
        str(port),
        "--non-interactive",
        "--skip-benchmark",
    ]

    if role == "worker":
        master_host = config.get("master_host", defaults.get("master_host", "127.0.0.1"))
        master_port = config.get("master_port", defaults.get("master_port", 52414))
        cmd_parts.extend(["--master-host", master_host, "--master-port", str(master_port)])

    return cmd_parts


def _run_sudo(args: list[str]) -> subprocess.CompletedProcess:
    """以 sudo 执行命令（Linux）。"""
    cmd = ["sudo"] + args
    return subprocess.run(cmd, capture_output=True, text=True)


# ---------------------------------------------------------------------------
# Linux (systemd) implementation
# ---------------------------------------------------------------------------

def _linux_is_enabled(role: str) -> bool:
    service_path = _systemd_service_path(role)
    if not service_path.exists():
        return False
    result = subprocess.run(
        ["systemctl", "is-enabled", f"asc-{role}.service"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() == "enabled"


def _linux_enable(role: str, config: dict) -> dict:
    content = _build_service_content(role, config)
    service_path = _systemd_service_path(role)

    # 写入服务文件（需要 sudo）
    try:
        process = subprocess.run(
            ["sudo", "tee", str(service_path)],
            input=content,
            capture_output=True,
            text=True,
        )
        if process.returncode != 0:
            return {"success": False, "message": f"Failed to write service file: {process.stderr.strip()}"}
    except FileNotFoundError:
        return {"success": False, "message": "sudo is required to create systemd service files"}

    # daemon-reload
    result = _run_sudo(["systemctl", "daemon-reload"])
    if result.returncode != 0:
        return {"success": False, "message": f"daemon-reload failed: {result.stderr.strip()}"}

    # enable
    result = _run_sudo(["systemctl", "enable", f"asc-{role}.service"])
    if result.returncode != 0:
        return {"success": False, "message": f"Failed to enable service: {result.stderr.strip()}"}

    logger.info("ASC %s autostart enabled (systemd)", role)
    return {"success": True, "message": f"ASC {role} autostart enabled via systemd"}


def _linux_disable(role: str) -> dict:
    service_path = _systemd_service_path(role)

    if not service_path.exists():
        return {"success": True, "message": f"ASC {role} autostart is not configured"}

    # disable
    result = _run_sudo(["systemctl", "disable", f"asc-{role}.service"])
    if result.returncode != 0:
        return {"success": False, "message": f"Failed to disable service: {result.stderr.strip()}"}

    # remove service file
    result = _run_sudo(["rm", "-f", str(service_path)])
    if result.returncode != 0:
        return {"success": False, "message": f"Failed to remove service file: {result.stderr.strip()}"}

    # daemon-reload
    result = _run_sudo(["systemctl", "daemon-reload"])
    if result.returncode != 0:
        return {"success": False, "message": f"daemon-reload failed: {result.stderr.strip()}"}

    logger.info("ASC %s autostart disabled (systemd)", role)
    return {"success": True, "message": f"ASC {role} autostart disabled"}


# ---------------------------------------------------------------------------
# Windows (schtasks) implementation
# ---------------------------------------------------------------------------

def _windows_is_enabled(role: str) -> bool:
    task_name = _windows_task_name(role)
    result = subprocess.run(
        ["schtasks", "/query", "/tn", task_name],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _windows_enable(role: str, config: dict) -> dict:
    task_name = _windows_task_name(role)
    python_path = sys.executable
    work_dir = os.getcwd()
    cmd_parts = _build_windows_command(role, config)
    command_str = " ".join(cmd_parts)

    args = [
        "schtasks",
        "/create",
        "/tn",
        task_name,
        "/tr",
        f'cmd /c "cd /d {work_dir} && {command_str}"',
        "/sc",
        "onlogon",
        "/rl",
        "highest",
        "/f",
    ]

    try:
        result = subprocess.run(args, capture_output=True, text=True)
        if result.returncode != 0:
            return {"success": False, "message": f"schtasks create failed: {result.stderr.strip()}"}
    except FileNotFoundError:
        return {"success": False, "message": "schtasks command not found"}

    logger.info("ASC %s autostart enabled (schtasks)", role)
    return {"success": True, "message": f"ASC {role} autostart enabled via scheduled task"}


def _windows_disable(role: str) -> dict:
    task_name = _windows_task_name(role)

    if not _windows_is_enabled(role):
        return {"success": True, "message": f"ASC {role} autostart is not configured"}

    try:
        result = subprocess.run(
            ["schtasks", "/delete", "/tn", task_name, "/f"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return {"success": False, "message": f"schtasks delete failed: {result.stderr.strip()}"}
    except FileNotFoundError:
        return {"success": False, "message": "schtasks command not found"}

    logger.info("ASC %s autostart disabled (schtasks)", role)
    return {"success": True, "message": f"ASC {role} autostart disabled"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_autostart_enabled(role: str) -> bool:
    """检查指定角色的自启动是否已启用。

    Args:
        role: "master" 或 "worker"

    Returns:
        自启动是否已启用。
    """
    _validate_role(role)

    if _IS_LINUX:
        return _linux_is_enabled(role)
    elif _IS_WINDOWS:
        return _windows_is_enabled(role)
    else:
        logger.warning("Unsupported platform for autostart: %s", platform.system())
        return False


def enable_autostart(role: str, config: dict) -> dict:
    """启用指定角色的自启动。

    Args:
        role: "master" 或 "worker"
        config: 配置字典，可包含 host、port 等字段。

    Returns:
        {"success": bool, "message": str}
    """
    _validate_role(role)

    if _IS_LINUX:
        return _linux_enable(role, config)
    elif _IS_WINDOWS:
        return _windows_enable(role, config)
    else:
        return {"success": False, "message": f"Unsupported platform: {platform.system()}"}


def disable_autostart(role: str) -> dict:
    """禁用指定角色的自启动。

    Args:
        role: "master" 或 "worker"

    Returns:
        {"success": bool, "message": str}
    """
    _validate_role(role)

    if _IS_LINUX:
        return _linux_disable(role)
    elif _IS_WINDOWS:
        return _windows_disable(role)
    else:
        return {"success": False, "message": f"Unsupported platform: {platform.system()}"}


def get_autostart_status() -> dict:
    """获取 Master 和 Worker 的自启动状态。

    Returns:
        {"master": {"enabled": bool, "supported": bool}, "worker": {"enabled": bool, "supported": bool}}
    """
    supported = _IS_LINUX or _IS_WINDOWS
    status = {}
    for role in ("master", "worker"):
        enabled = is_autostart_enabled(role) if supported else False
        status[role] = {"enabled": enabled, "supported": supported}
    return status
