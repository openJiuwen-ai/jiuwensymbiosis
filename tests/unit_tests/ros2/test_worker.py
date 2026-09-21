# coding: utf-8
"""``ros2.worker`` —— 主进程侧 worker 协议（用真子进程，不需要 ROS）。"""

from __future__ import annotations

import subprocess
import sys

import pytest

from jiuwensymbiosis.agent.lifecycle import HardwareCleanupError
from jiuwensymbiosis.ros2 import worker as W


def _script(body: str) -> list[str]:
    """一个用当前解释器跑的迷你 worker（stdlib only，无 rclpy）。"""
    return [sys.executable, "-c", body]


class TestWorkerPath:
    def test_resolves_an_existing_module(self):
        path = W.worker_path("jiuwensymbiosis.ros2.image_decode")
        assert path.name == "image_decode.py" and path.is_file()

    def test_unknown_module_raises(self):
        with pytest.raises(ModuleNotFoundError):
            W.worker_path("jiuwensymbiosis.ros2.definitely_not_a_worker")


class TestRunOnce:
    def test_returns_last_json_line(self):
        # 前面几行是日志，最后一行才是结果——worker 允许边跑边打日志。
        out = W.run_once(
            _script('print("booting"); print(\'{"ok": true, "yaw_turned": 1.5}\')'), timeout_s=10.0, label="[t]"
        )
        assert out == {"ok": True, "yaw_turned": 1.5}

    @pytest.mark.parametrize(
        "body",
        ["import sys; sys.exit(3)", "pass", 'print("not json")', 'print("[1, 2]")', "import time; time.sleep(30)"],
    )
    def test_missing_completion_evidence_is_cleanup_failure(self, body):
        with pytest.raises(HardwareCleanupError):
            W.run_once(_script(body), timeout_s=0.5, label="[t]")

    def test_unlaunchable_command_is_worker_error(self):
        out = W.run_once(["/nonexistent/interpreter"], timeout_s=5.0, label="[t]")
        assert out == {"ok": False, "reason": "worker_error"}


class TestStopAndCollect:
    def test_collects_result_of_a_finished_worker(self):
        proc = subprocess.Popen(
            _script('print(\'{"ok": true, "dist_traveled": 2.0}\')'),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        proc.wait(timeout=10.0)
        out = W.stop_and_collect(proc, label="[t]", kind="drive")
        assert out == {"ok": True, "dist_traveled": 2.0}

    def test_stop_sentinel_halts_a_running_worker(self):
        # 一直等 stdin 的 worker：收到 'stop' 才打印结果并退出。
        body = (
            "import sys\n"
            "for line in sys.stdin:\n"
            "    if line.strip() == 'stop':\n"
            '        print(\'{"ok": true, "yaw_turned": 0.25}\'); break\n'
        )
        proc = subprocess.Popen(_script(body), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        out = W.stop_and_collect(proc, label="[t]", kind="spin")
        assert out == {"ok": True, "yaw_turned": 0.25}
        assert proc.poll() is not None

    def test_silent_worker_has_no_stop_confirmation(self):
        proc = subprocess.Popen(_script("pass"), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        proc.wait(timeout=10.0)
        with pytest.raises(HardwareCleanupError):
            W.stop_and_collect(proc, label="[t]", kind="spin")

    @pytest.mark.parametrize("ignore_sigterm", [False, True])
    def test_wedged_worker_is_reaped_but_stop_remains_unconfirmed(self, ignore_sigterm):
        # SIGTERM/SIGKILL can reap a process without confirming a physical stop.
        body = "import signal, time\n"
        if ignore_sigterm:
            body += "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        body += "print('ready', flush=True)\ntime.sleep(60)"
        proc = subprocess.Popen(_script(body), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        assert proc.stdout.readline().strip() == "ready"
        with pytest.raises(HardwareCleanupError):
            W.stop_and_collect(proc, label="[t]", kind="drive", timeout_s=0.05, kill_timeout_s=0.05)
        assert proc.poll() is not None


class TestResidentWorker:
    @staticmethod
    def _echo_worker() -> W.ResidentWorker:
        """每收一行请求就回一行 JSON，收到 'stop' 退出。"""
        body = (
            "import sys\n"
            "for line in sys.stdin:\n"
            "    s = line.strip()\n"
            "    if s == 'stop':\n"
            "        break\n"
            '    print(\'{"ok": true, "echo": "%s"}\' % s, flush=True)\n'
        )
        return W.ResidentWorker(lambda: _script(body), label="[t]")

    def test_serves_multiple_requests_from_one_process(self):
        worker = self._echo_worker()
        try:
            assert worker.request_json("a", 10.0, bad_output_reason="bad") == {"ok": True, "echo": "a"}
            pid = worker.proc.pid
            assert worker.request_json("b", 10.0, bad_output_reason="bad") == {"ok": True, "echo": "b"}
            assert worker.proc.pid == pid  # 同一个进程，rclpy/DDS 发现只付一次
        finally:
            worker.stop()

    def test_stop_is_idempotent_and_clears_the_handle(self):
        worker = self._echo_worker()
        worker.request_json("a", 10.0, bad_output_reason="bad")
        worker.stop()
        assert worker.proc is None
        worker.stop()  # 第二次不得抛

    def test_unlaunchable_worker_reports_none(self):
        worker = W.ResidentWorker(lambda: ["/nonexistent/interpreter"], label="[t]")
        assert worker.request_json("a", 5.0, bad_output_reason="bad") is None
        assert worker.proc is None

    def test_dead_worker_cleanup_uncertainty_propagates_and_blocks_restart(self):
        # worker 立刻退出：句柄已死，但它没有确认 stop；不能把此情况当成
        # 普通无回复并回退到另一路硬件命令。
        worker = W.ResidentWorker(lambda: _script("pass"), label="[t]")
        with pytest.raises(RuntimeError, match="exited before a stop was confirmed"):
            worker.request_json("a", 2.0, bad_output_reason="bad")
        assert worker.proc is None
        with pytest.raises(RuntimeError, match="shutdown remains unconfirmed"):
            worker.request_json("b", 2.0, bad_output_reason="bad")

    def test_empty_reply_with_confirmed_stop_remains_a_soft_failure(self):
        body = "import sys\nfor line in sys.stdin:\n    if line.strip() == 'stop':\n        break\n"
        worker = W.ResidentWorker(lambda: _script(body), label="[t]")
        assert worker.request_json("a", 0.1, bad_output_reason="bad") is None
        assert worker.proc is None
        worker.stop()

    def test_worker_that_dies_between_requests_is_not_silently_replaced(self):
        from unittest.mock import Mock

        launch = Mock()
        worker = W.ResidentWorker(launch, label="[t]")
        dead_proc = Mock()
        dead_proc.poll.return_value = -9
        worker._proc = dead_proc
        with pytest.raises(RuntimeError, match="exited before a stop was confirmed"):
            worker.request("next", 0.01)
        launch.assert_not_called()
        with pytest.raises(RuntimeError, match="shutdown remains unconfirmed"):
            worker.stop()

    def test_live_process_handle_is_kept_retryable_when_forced_stop_does_not_finish(self):
        from unittest.mock import Mock

        class _StubbornProc:
            stdin = Mock()

            def __init__(self):
                self.alive = True
                self.returncode = None
                self.allow_stop = False

            def poll(self):
                return None if self.alive else self.returncode

            def communicate(self, input=None, timeout=None):
                if input and self.allow_stop:
                    self.alive = False
                    self.returncode = 0
                    return "", ""
                raise subprocess.TimeoutExpired("fake-worker", timeout)

            def terminate(self):
                pass

            def kill(self):
                pass

        worker = W.ResidentWorker(lambda: _script("pass"), label="[t]")
        proc = _StubbornProc()
        worker._proc = proc

        with pytest.raises(RuntimeError, match="still alive after terminate/kill"):
            worker.stop()
        assert worker._proc is proc
        assert worker.proc is proc

        with pytest.raises(RuntimeError, match="shutdown remains unconfirmed"):
            worker.request("another motion", 0.01)
        proc.stdin.write.assert_not_called()

        proc.allow_stop = True
        worker.stop()
        assert worker._proc is None
        assert worker.proc is None

    def test_forced_process_termination_is_latched_as_uncertain(self):
        class _TerminatedProc:
            stdin = object()

            def __init__(self):
                self.alive = True
                self.returncode = None

            def poll(self):
                return None if self.alive else self.returncode

            def communicate(self, input=None, timeout=None):
                if input:
                    raise subprocess.TimeoutExpired("fake-worker", timeout)
                return "", ""

            def terminate(self):
                self.alive = False
                self.returncode = -15

            def kill(self):
                self.alive = False
                self.returncode = -9

        launches = []
        worker = W.ResidentWorker(lambda: launches.append(True) or _script("pass"), label="[t]")
        worker._proc = _TerminatedProc()

        with pytest.raises(RuntimeError, match="required forced termination"):
            worker.stop()
        assert worker._proc is None
        with pytest.raises(RuntimeError, match="shutdown remains unconfirmed"):
            worker._ensure()
        assert launches == []

    def test_live_worker_talking_nonsense_is_not_restarted(self):
        # 活着但输出不是 JSON → 报 bad_output，但【不】杀进程：乱说话不是重启的理由。
        body = "import sys\nfor line in sys.stdin:\n    print('not json', flush=True)\n"
        worker = W.ResidentWorker(lambda: _script(body), label="[t]")
        try:
            assert worker.request_json("a", 10.0, bad_output_reason="bad") == {"ok": False, "reason": "bad"}
            assert worker.proc is not None
        finally:
            worker.stop()

    def test_restarts_after_the_previous_worker_died(self):
        worker = self._echo_worker()
        try:
            worker.request_json("a", 10.0, bad_output_reason="bad")
            first = worker.proc.pid
            worker.stop()
            assert worker.request_json("b", 10.0, bad_output_reason="bad") == {"ok": True, "echo": "b"}
            assert worker.proc.pid != first
        finally:
            worker.stop()
