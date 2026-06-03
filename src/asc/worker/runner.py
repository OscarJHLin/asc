"""Asc Runner 状态机。

Runner 是 Worker 节点上的推理引擎管理器，实现严格的状态机。
借鉴 exo 的 Runner 状态机思想，但用更简洁的 Python 实现。

状态机：
    Idle -> Loading -> Ready -> Running -> Ready (循环)
    任意状态 -> Error -> Idle (恢复)
    Ready -> Shutdown

设计原则：
- 每个状态转换都是显式的，通过 transition() 方法
- 非法转换抛出 RunnerTransitionError
- 状态转换历史可追溯
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


class RunnerState(enum.Enum):
    """Runner 状态。"""

    IDLE = "idle"
    LOADING = "loading"
    READY = "ready"
    RUNNING = "running"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class RunnerCommand(enum.Enum):
    """Runner 状态转换命令。"""

    LOAD = "load"
    LOAD_COMPLETE = "load_complete"
    START_INFERENCE = "start_inference"
    INFERENCE_COMPLETE = "inference_complete"
    ERROR = "error"
    RESET = "reset"
    SHUTDOWN = "shutdown"


class RunnerTransitionError(Exception):
    """非法状态转换异常。"""

    def __init__(self, from_state: RunnerState, command: RunnerCommand):
        self.from_state = from_state
        self.command = command
        super().__init__(f"非法转换: {from_state.value} + {command.value}")


# 合法转换表：{(当前状态, 命令) -> 目标状态}
_VALID_TRANSITIONS: dict[tuple[RunnerState, RunnerCommand], RunnerState] = {
    # Idle
    (RunnerState.IDLE, RunnerCommand.LOAD): RunnerState.LOADING,
    # Loading
    (RunnerState.LOADING, RunnerCommand.LOAD_COMPLETE): RunnerState.READY,
    (RunnerState.LOADING, RunnerCommand.ERROR): RunnerState.ERROR,
    # Ready
    (RunnerState.READY, RunnerCommand.START_INFERENCE): RunnerState.RUNNING,
    (RunnerState.READY, RunnerCommand.SHUTDOWN): RunnerState.SHUTDOWN,
    # Running
    (RunnerState.RUNNING, RunnerCommand.INFERENCE_COMPLETE): RunnerState.READY,
    (RunnerState.RUNNING, RunnerCommand.ERROR): RunnerState.ERROR,
    # Error
    (RunnerState.ERROR, RunnerCommand.RESET): RunnerState.IDLE,
}


@dataclass
class Runner:
    """Worker Runner 状态机。"""

    node_id: str
    _state: RunnerState = field(default=RunnerState.IDLE, init=False)
    _last_error: str | None = field(default=None, init=False)
    _state_history: list[tuple[RunnerState, RunnerState]] = field(default_factory=list, init=False)

    @property
    def state(self) -> RunnerState:
        return self._state

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def state_history(self) -> list[tuple[RunnerState, RunnerState]]:
        return list(self._state_history)

    def transition(self, command: RunnerCommand, error_msg: str | None = None) -> None:
        """执行状态转换。

        Args:
            command: 转换命令
            error_msg: 仅在 command=ERROR 时使用

        Raises:
            RunnerTransitionError: 非法转换
        """
        key = (self._state, command)
        target = _VALID_TRANSITIONS.get(key)

        if target is None:
            raise RunnerTransitionError(self._state, command)

        old_state = self._state
        self._state = target
        self._state_history.append((old_state, target))

        if command == RunnerCommand.ERROR:
            self._last_error = error_msg
        elif command == RunnerCommand.RESET:
            self._last_error = None

    def can_accept_inference(self) -> bool:
        """判断是否可接受推理请求。"""
        return self._state == RunnerState.READY
