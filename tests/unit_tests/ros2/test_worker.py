# coding: utf-8
"""``ros2.worker`` —— 主进程侧 worker 协议（用真子进程，不需要 ROS）。"""

from __future__ import annotations

import subprocess
import sys

import pytest

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
            _script('print("booting"); print(\'{"ok": true, "yaw_turned": 1.5}\')'),
            timeout_s=10.0, label="[t]")
        assert out == {"ok": True, "yaw_turned": 1.5}

    def test_nonzero_returncode_is_worker_failed(self):
        out = W.run_once(_script("import sys; sys.exit(3)"), timeout_s=10.0, label="[t]", reason_prefix="wheel_")
        assert out == {"ok": False, "reason": "wheel_worker_failed"}

    def test_empty_stdout_is_no_output(self):
        out = W.run_once(_script("pass"), timeout_s=10.0, label="[t]", reason_prefix="wheel_")
        assert out == {"ok": False, "reason": "wheel_no_output"}

    def test_non_json_stdout_is_bad_output(self):
        out = W.run_once(_script('print("not json")'), timeout_s=10.0, label="[t]", reason_prefix="wheel_")
        assert out == {"ok": False, "reason": "wheel_bad_output"}

    def test_timeout_is_worker_error(self):
        out = W.run_once(_script("import time; time.sleep(30)"), timeout_s=0.5,
                         label="[t]", reason_prefix="wheel_")
        assert out == {"ok": False, "reason": "wheel_worker_error"}

    def test_unlaunchable_command_is_worker_error(self):
        out = W.run_once(["/nonexistent/interpreter"], timeout_s=5.0, label="[t]")
        assert out == {"ok": False, "reason": "worker_error"}

    def test_json_array_is_bad_output(self):
        # 结果契约是一个 JSON 对象；数组解析得出来但不是契约，必须当坏输出而不是原样透传。
        out = W.run_once(_script('print("[1, 2]")'), timeout_s=10.0, label="[t]")
        assert out == {"ok": False, "reason": "bad_output"}


class TestStopAndCollect:
    def test_collects_result_of_a_finished_worker(self):
        proc = subprocess.Popen(_script('print(\'{"ok": true, "dist_traveled": 2.0}\')'),
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        proc.wait(timeout=10.0)
        out = W.stop_and_collect(proc, label="[t]", kind="drive", empty_result={"ok": True, "dist_traveled": 0.0})
        assert out == {"ok": True, "dist_traveled": 2.0}

    def test_stop_sentinel_halts_a_running_worker(self):
        # 一直等 stdin 的 worker：收到 'stop' 才打印结果并退出。
        body = (
            "import sys\n"
            "for line in sys.stdin:\n"
            "    if line.strip() == 'stop':\n"
            "        print('{\"ok\": true, \"yaw_turned\": 0.25}'); break\n"
        )
        proc = subprocess.Popen(_script(body), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        out = W.stop_and_collect(proc, label="[t]", kind="spin", empty_result={"ok": True, "yaw_turned": 0.0})
        assert out == {"ok": True, "yaw_turned": 0.25}
        assert proc.poll() is not None

    def test_silent_worker_yields_the_empty_result(self):
        proc = subprocess.Popen(_script("pass"), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        proc.wait(timeout=10.0)
        out = W.stop_and_collect(proc, label="[t]", kind="spin", empty_result={"ok": True, "yaw_turned": 0.0})
        assert out == {"ok": True, "yaw_turned": 0.0}

    def test_wedged_worker_is_terminated_not_left_running(self):
        # 忽略 stdin 且不退出 → communicate 超时 → SIGTERM 兜底。底盘绝不能被留在运动状态。
        proc = subprocess.Popen(_script("import time; time.sleep(60)"),
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        out = W.stop_and_collect(proc, label="[t]", kind="drive",
                                 empty_result={"ok": True, "dist_traveled": 0.0}, timeout_s=0.5)
        assert out["ok"] is True and out["dist_traveled"] == 0.0
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
            "    print('{\"ok\": true, \"echo\": \"%s\"}' % s, flush=True)\n"
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

    def test_dead_worker_reports_none_and_drops_the_handle(self):
        # worker 立刻退出：请求读不到回复 → None（调用方据此回退到一次性路径），句柄被丢弃。
        worker = W.ResidentWorker(lambda: _script("pass"), label="[t]")
        assert worker.request_json("a", 2.0, bad_output_reason="bad") is None
        assert worker.proc is None

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
