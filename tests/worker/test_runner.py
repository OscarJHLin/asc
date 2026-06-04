"""测试 Runner 状态机。

Runner 是 Worker 节点上的推理引擎管理器，实现严格的状态机：
Idle -> Loading -> Ready -> Running -> Ready (循环)
任意状态 -> Error -> Idle (恢复)
"""

from asc.worker.runner import (
    Runner,
    RunnerCommand,
    RunnerState,
    RunnerTransitionError,
)


class TestRunnerState:
    """Runner 状态枚举。"""

    def test_all_states(self):
        assert RunnerState.IDLE.value == "idle"
        assert RunnerState.LOADING.value == "loading"
        assert RunnerState.READY.value == "ready"
        assert RunnerState.RUNNING.value == "running"
        assert RunnerState.ERROR.value == "error"
        assert RunnerState.SHUTDOWN.value == "shutdown"


class TestRunnerInitialState:
    """Runner 初始状态。"""

    def test_initial_state_is_idle(self):
        runner = Runner(node_id="node-1")
        assert runner.state == RunnerState.IDLE


class TestRunnerValidTransitions:
    """合法状态转换。"""

    def test_idle_to_loading(self):
        runner = Runner(node_id="node-1")
        runner.transition(RunnerCommand.LOAD)
        assert runner.state == RunnerState.LOADING

    def test_loading_to_ready(self):
        runner = Runner(node_id="node-1")
        runner.transition(RunnerCommand.LOAD)  # idle -> loading
        runner.transition(RunnerCommand.LOAD_COMPLETE)
        assert runner.state == RunnerState.READY

    def test_ready_to_running(self):
        runner = Runner(node_id="node-1")
        runner.transition(RunnerCommand.LOAD)
        runner.transition(RunnerCommand.LOAD_COMPLETE)
        runner.transition(RunnerCommand.START_INFERENCE)
        assert runner.state == RunnerState.RUNNING

    def test_running_to_ready(self):
        runner = Runner(node_id="node-1")
        runner.transition(RunnerCommand.LOAD)
        runner.transition(RunnerCommand.LOAD_COMPLETE)
        runner.transition(RunnerCommand.START_INFERENCE)
        runner.transition(RunnerCommand.INFERENCE_COMPLETE)
        assert runner.state == RunnerState.READY

    def test_ready_to_shutdown(self):
        runner = Runner(node_id="node-1")
        runner.transition(RunnerCommand.LOAD)
        runner.transition(RunnerCommand.LOAD_COMPLETE)
        runner.transition(RunnerCommand.SHUTDOWN)
        assert runner.state == RunnerState.SHUTDOWN

    def test_loading_to_error(self):
        runner = Runner(node_id="node-1")
        runner.transition(RunnerCommand.LOAD)
        runner.transition(RunnerCommand.ERROR, error_msg="Load failed")
        assert runner.state == RunnerState.ERROR

    def test_running_to_error(self):
        runner = Runner(node_id="node-1")
        runner.transition(RunnerCommand.LOAD)
        runner.transition(RunnerCommand.LOAD_COMPLETE)
        runner.transition(RunnerCommand.START_INFERENCE)
        runner.transition(RunnerCommand.ERROR, error_msg="OOM")
        assert runner.state == RunnerState.ERROR

    def test_error_to_idle(self):
        runner = Runner(node_id="node-1")
        runner.transition(RunnerCommand.LOAD)
        runner.transition(RunnerCommand.ERROR, error_msg="fail")
        runner.transition(RunnerCommand.RESET)
        assert runner.state == RunnerState.IDLE

    def test_full_lifecycle(self):
        """完整生命周期：idle -> loading -> ready -> running -> ready -> shutdown"""
        runner = Runner(node_id="node-1")
        runner.transition(RunnerCommand.LOAD)
        runner.transition(RunnerCommand.LOAD_COMPLETE)
        runner.transition(RunnerCommand.START_INFERENCE)
        runner.transition(RunnerCommand.INFERENCE_COMPLETE)
        runner.transition(RunnerCommand.SHUTDOWN)
        assert runner.state == RunnerState.SHUTDOWN


class TestRunnerInvalidTransitions:
    """非法状态转换应抛出异常。"""

    def test_idle_cannot_start_inference(self):
        runner = Runner(node_id="node-1")
        try:
            runner.transition(RunnerCommand.START_INFERENCE)
            raise AssertionError("Should raise RunnerTransitionError")
        except RunnerTransitionError:
            pass

    def test_idle_cannot_complete_inference(self):
        runner = Runner(node_id="node-1")
        try:
            runner.transition(RunnerCommand.INFERENCE_COMPLETE)
            raise AssertionError("Should raise RunnerTransitionError")
        except RunnerTransitionError:
            pass

    def test_loading_cannot_start_inference(self):
        runner = Runner(node_id="node-1")
        runner.transition(RunnerCommand.LOAD)
        try:
            runner.transition(RunnerCommand.START_INFERENCE)
            raise AssertionError("Should raise RunnerTransitionError")
        except RunnerTransitionError:
            pass

    def test_running_cannot_load(self):
        runner = Runner(node_id="node-1")
        runner.transition(RunnerCommand.LOAD)
        runner.transition(RunnerCommand.LOAD_COMPLETE)
        runner.transition(RunnerCommand.START_INFERENCE)
        try:
            runner.transition(RunnerCommand.LOAD)
            raise AssertionError("Should raise RunnerTransitionError")
        except RunnerTransitionError:
            pass

    def test_shutdown_cannot_transition(self):
        runner = Runner(node_id="node-1")
        runner.transition(RunnerCommand.LOAD)
        runner.transition(RunnerCommand.LOAD_COMPLETE)
        runner.transition(RunnerCommand.SHUTDOWN)
        try:
            runner.transition(RunnerCommand.LOAD)
            raise AssertionError("Should raise RunnerTransitionError")
        except RunnerTransitionError:
            pass


class TestRunnerErrorTracking:
    """错误信息跟踪。"""

    def test_error_message_stored(self):
        runner = Runner(node_id="node-1")
        runner.transition(RunnerCommand.LOAD)
        runner.transition(RunnerCommand.ERROR, error_msg="Model not found")
        assert runner.last_error == "Model not found"

    def test_error_cleared_on_reset(self):
        runner = Runner(node_id="node-1")
        runner.transition(RunnerCommand.LOAD)
        runner.transition(RunnerCommand.ERROR, error_msg="fail")
        runner.transition(RunnerCommand.RESET)
        assert runner.last_error is None


class TestRunnerStateHistory:
    """状态转换历史记录。"""

    def test_records_transitions(self):
        runner = Runner(node_id="node-1")
        runner.transition(RunnerCommand.LOAD)
        runner.transition(RunnerCommand.LOAD_COMPLETE)
        assert len(runner.state_history) == 2
        assert runner.state_history[0] == (RunnerState.IDLE, RunnerState.LOADING)
        assert runner.state_history[1] == (RunnerState.LOADING, RunnerState.READY)

    def test_error_transition_recorded(self):
        runner = Runner(node_id="node-1")
        runner.transition(RunnerCommand.LOAD)
        runner.transition(RunnerCommand.ERROR, error_msg="fail")
        assert runner.state_history[-1] == (RunnerState.LOADING, RunnerState.ERROR)


class TestRunnerCanAcceptInference:
    """便捷方法：判断是否可接受推理请求。"""

    def test_idle_cannot_accept(self):
        runner = Runner(node_id="node-1")
        assert not runner.can_accept_inference()

    def test_ready_can_accept(self):
        runner = Runner(node_id="node-1")
        runner.transition(RunnerCommand.LOAD)
        runner.transition(RunnerCommand.LOAD_COMPLETE)
        assert runner.can_accept_inference()

    def test_running_cannot_accept(self):
        runner = Runner(node_id="node-1")
        runner.transition(RunnerCommand.LOAD)
        runner.transition(RunnerCommand.LOAD_COMPLETE)
        runner.transition(RunnerCommand.START_INFERENCE)
        assert not runner.can_accept_inference()
