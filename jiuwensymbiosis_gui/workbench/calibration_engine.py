# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""CalibrationEngine —— 手眼标定的后台执行引擎(示教 / 采集 / 求解)。

与 ``PerceptionEngine`` 同构:后台线程干活,事件塞进队列,界面用 ``ui.timer`` 轮询
``drain()``。业务逻辑一行不写在这里 —— 三个阶段分别直接调
``jiuwensymbiosis.calibration`` 的 ``collect_waypoints`` / ``execute_calibration``,
本模块只负责三件界面才需要的事:

1. **把阻塞式提问变成按钮**。``collect_waypoints`` 用 ``prompt_fn`` 索取下一个动作,
   CLI 的实现是 ``input()``。这里换成一个「轮询命令队列、顺便推预览帧」的实现:
   等待用户按钮的间隙里取帧 + 跑 ChArUco 检测,于是示教时能实时看到板认没认出来
   ——CLI 只能盲按回车,废姿态要等采集阶段才暴露。预览与示教在**同一个线程**里
   交替进行,不额外碰硬件,避免相机/串口的并发访问。
2. **把日志变成进度**。标定子系统全程走 ``logging``,复用 ``run_engine`` 的
   ``QueueLogHandler`` 挂到 ``calibration`` logger 树上即可,无需改业务代码。
3. **把力矩恢复失败单独拎出来**。``ManualGuidanceRecoveryError`` 意味着机械臂可能
   仍处于失力矩状态,是唯一需要用户立刻用手接住的错误,因此单独标 ``fatal``。
4. **在动力矩之前设一道确认门**。示教方式取决于设备实现了哪个端口
   (``resolve_teaching_mode``:实现 ``ManualGuidance`` 才松力矩),而这只有连接后
   才知道。所以连接后先把模式报给界面 (``teach_mode``) 并阻塞等确认,再进
   ``collect_waypoints`` —— 保证"力矩会掉"的警告严格早于力矩真的掉,且判据是端口
   而非机型名。

事件: ``preview`` / ``detect`` / ``waypoint`` / ``teach_mode`` / ``phase`` / ``log`` /
``done`` / ``error``。
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import queue
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from typing import Any

import yaml

from jiuwensymbiosis.agent.lifecycle import CleanupReport
from jiuwensymbiosis.utils.logging import get_logger
from jiuwensymbiosis_gui.workbench import imaging
from jiuwensymbiosis_gui.workbench.board_print import BoardParams
from jiuwensymbiosis_gui.workbench.maintenance import (
    BoundedEventQueue,
    MaintenanceOwner,
    append_cleanup_errors,
    cleanup_report_for,
    cleanup_report_message,
    finish_admission,
)
from jiuwensymbiosis_gui.workbench.run_engine import QueueLogHandler

logger = get_logger(__name__)

__all__ = ["CalibrationEngine", "CalibrationSetup", "TeachingAborted"]

# 示教预览节流:ChArUco 检测是 CPU 密集的,~5fps 已足够让操作者判断板在不在视野里。
_PREVIEW_PERIOD_S = 0.2
# 命令队列轮询周期。远小于预览周期,保证按钮响应感觉是即时的。
_COMMAND_POLL_S = 0.05
# 命令令牌。空串/``q`` 是 ``collect_waypoints`` 的 ``prompt_fn`` 契约(记录 / 结束);
# 其余几个只在本模块的取指令循环里消费,不会作为答复交回工作流,取值与前两者不冲突。
_CMD_RECORD = ""
_CMD_FINISH = "q"
_CMD_START = "go"
_CMD_PAUSE = "pause"
_CMD_RESUME = "resume"
_CMD_ABORT = "abort"


class TeachingAborted(Exception):
    """示教被用户中止。

    从 ``prompt_fn`` 里抛出去,让它穿过 ``manual_guidance`` 的 ``__exit__``(力矩照常恢复)
    并让 ``collect_waypoints`` 半途退出 —— 中止就是"不要这一轮的数据",不能像正常结束那样
    写出 waypoint 归档。
    """


# 标定子系统的 logger 树根;挂一个 handler 就能收全 collect/execute/publication/preflight。
_CALIBRATION_LOGGER = "calibration"


@dataclass(frozen=True)
class CalibrationSetup:
    """一次标定运行的全部输入(界面在第 0 步组装好,后续阶段共用)。"""

    adapter_module: str
    config_data: dict[str, Any]
    board: BoardParams
    out_path: Path
    waypoint_path: Path
    calibration_profile: dict[str, Any]
    min_corners: int = 16
    n_stations: int = 20


class CalibrationEngine:
    """标定后台引擎。一个实例服务一次完整标定(示教 → 采集),阶段间可重复调用。"""

    def __init__(self, setup: CalibrationSetup, *, admission: MaintenanceOwner | None = None) -> None:
        """记下输入;线程与队列在 ``start_teaching`` / ``start_capture`` 时才建立。"""
        self._setup = setup
        self._admission = admission
        self._events = BoundedEventQueue()
        self._commands: queue.Queue[str] = queue.Queue()
        self._thread: Thread | None = None
        self._stop = False
        self._binding: Any = None
        self._lease: Any = None
        self._active_session: Any = None
        self._fatal_recovery_error: str | None = None
        self._hardware_cleanup_error: str | None = None
        self._waypoint_count = 0
        self._station_count = 0
        self._profile_path: Path | None = None
        self._profile_dir: tempfile.TemporaryDirectory | None = None

    # ------------------------------------------------------------------ 控制
    def start_teaching(self) -> None:
        """起线程:连接 → 松力矩示教 → 写出 waypoint 归档。幂等。"""
        self._spawn(self._run_teaching, "jiuwen-gui-calib-teach", "calibration_teaching")

    def start_capture(self, *, dry_run: bool) -> None:
        """起线程:沿 waypoint 归档自动采集 → 求解 → 发布。``dry_run`` 只校验不动。"""
        operation = "calibration_dry_run" if dry_run else "calibration_capture"
        self._spawn(lambda: self._run_capture(dry_run=dry_run), "jiuwen-gui-calib-capture", operation)

    def confirm_teaching(self) -> None:
        """界面线程调用:用户已读过本机型的示教指引,放行到 ``collect_waypoints``。"""
        self._commands.put(_CMD_START)

    def record_waypoint(self) -> None:
        """界面线程调用:记录当前姿态(只入队,真正的快照在工作线程里做)。"""
        self._commands.put(_CMD_RECORD)

    def finish_teaching(self) -> None:
        """界面线程调用:结束示教并写归档。"""
        self._commands.put(_CMD_FINISH)

    def pause_teaching(self) -> None:
        """界面线程调用:在示教途中把机械臂托住(恢复力矩),让操作者松手歇一下。"""
        self._commands.put(_CMD_PAUSE)

    def resume_teaching(self) -> None:
        """界面线程调用:重新松力矩,继续手动示教。"""
        self._commands.put(_CMD_RESUME)

    def abort_teaching(self) -> None:
        """界面线程调用:中止这一轮示教,已记录的姿态一并作废。"""
        self._commands.put(_CMD_ABORT)

    def stop(self) -> None:
        """请求停止(工作线程在下一次轮询时退出)。

        走的是中止而非结束:离开页面 / 重新开始不该顺手写出一份半截的 waypoint 归档。
        """
        self._stop = True
        self._commands.put(_CMD_ABORT)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def can_dispose(self) -> bool:
        """Host ownership ends only after worker exit and confirmed release."""
        return not self.is_running() and (self._lease is None or self._lease.released)

    def join(self, timeout: float | None = None) -> None:
        thread = self._thread
        if thread is not None and getattr(thread, "ident", None) is not None:
            thread.join(timeout)

    def drain(self) -> list[tuple[str, Any]]:
        """非阻塞取出全部排队事件,供界面 ``ui.timer`` 周期消费。"""
        events: list[tuple[str, Any]] = []
        while True:
            try:
                events.append(self._events.get_nowait())
            except queue.Empty:
                return events

    # ------------------------------------------------------------------ 内部:线程
    def _spawn(self, target: Callable[[], None], name: str, operation: str) -> None:
        if self.is_running():
            return
        if self._lease is not None:
            self._emit("error", {"reason": "上次标定阶段的硬件释放未确认，资源仍被保留。", "fatal": False})
            return
        self._stop = False
        self._active_session = None
        self._fatal_recovery_error = None
        self._hardware_cleanup_error = None
        _drain_queue(self._commands)
        if self._admission is not None:
            try:
                # Profile overrides are part of the exact config this phase connects with.
                self._binding, self._lease = self._admission.acquire(self._session_config(), operation, stop=self.stop)
            except Exception as exc:
                self._binding = None
                self._lease = None
                self._emit("error", {"reason": f"{type(exc).__name__}: {exc}", "fatal": False})
                return

        def _run_stage() -> None:
            try:
                target()
            finally:
                report = cleanup_report_for(self._active_session)
                if self._fatal_recovery_error is not None:
                    report = append_cleanup_errors(
                        report, [f"manual guidance torque recovery state is unknown: {self._fatal_recovery_error}"]
                    )
                if self._hardware_cleanup_error is not None:
                    report = append_cleanup_errors(
                        report, [f"hardware cleanup unconfirmed: {self._hardware_cleanup_error}"]
                    )
                released, report = finish_admission(self._admission, self._lease, report)
                if released:
                    self._lease = None
                    self._binding = None
                elif self._fatal_recovery_error is None:
                    self._emit("error", {"reason": cleanup_report_message(report), "fatal": False})

        self._thread = Thread(target=_run_stage, name=name, daemon=True)
        try:
            self._thread.start()
        except BaseException as exc:
            released, report = finish_admission(self._admission, self._lease, CleanupReport())
            if released:
                self._lease = None
                self._binding = None
            elif report.errors or report.pending_work or report.connected:
                self._emit("error", {"reason": cleanup_report_message(report), "fatal": False})
            self._emit("error", {"reason": f"后台线程启动失败: {type(exc).__name__}: {exc}", "fatal": False})
            if not isinstance(exc, Exception):
                raise
            return

    def _emit(self, tag: str, payload: Any) -> None:
        self._events.put((tag, payload))

    # ------------------------------------------------------------------ 阶段 1:示教
    def _run_teaching(self) -> None:
        self._guarded(self._teach)

    def _teach(self) -> None:
        from jiuwensymbiosis.calibration import collect_waypoints
        from jiuwensymbiosis.calibration.domain.ports import GuidanceHold
        from jiuwensymbiosis.calibration.workflows.preflight import resolve_teaching_mode

        self._waypoint_count = 0
        with self._log_capture(), self._connected_device() as (device, spec, options):
            # 判据是端口而非机型:只有实现了 ManualGuidance 的设备才会松力矩。等界面
            # 确认后再往下走,否则力矩可能在用户读到警告之前就掉了。
            mode = resolve_teaching_mode(device)
            # 能不能暂停取决于设备实现了哪个端口,与机型名无关 —— 力矩本来就没松的
            # external_snapshot 也谈不上暂停。
            can_pause = mode == "manual_guidance" and isinstance(device, GuidanceHold)
            self._emit("teach_mode", {"mode": mode, "can_pause": can_pause})
            if not self._await_start(device):
                return
            self._emit("phase", {"phase": "teaching", "msg": "示教进行中:摆好一个姿态就点「记录当前姿态」。"})
            path = collect_waypoints(
                device,
                self._setup.waypoint_path,
                options,
                adapter_package=spec.package,
                mount=device.camera_mount,
                prompt_fn=lambda _prompt: self._await_command(device),
            )
            self._emit(
                "done",
                {"phase": "teaching", "count": self._waypoint_count, "path": str(path), "mode": mode},
            )

    def _await_start(self, device) -> bool:
        """阻塞等待示教确认门放行;结束令牌返回 ``False``(不进入示教)。"""
        return self._pump_until_command(device) == _CMD_START

    def _await_command(self, device) -> str | None:
        """阻塞等待界面按钮,期间持续推预览帧 + 板检测结果。

        返回 ``""`` 记录一个 waypoint、``"q"`` 结束示教。这是 ``collect_waypoints``
        的 ``prompt_fn`` 契约。
        """
        if self._pump_until_command(device) == _CMD_FINISH:
            return _CMD_FINISH
        self._waypoint_count += 1
        self._emit("waypoint", {"count": self._waypoint_count})
        return _CMD_RECORD

    def _pump_until_command(self, device) -> str:
        """取一条界面命令,期间按预览周期持续推帧;中止请求抛 ``TeachingAborted``。

        暂停/继续在这里就地消费掉:它们改的是机械臂的力矩状态,不是给
        ``collect_waypoints`` 的答复。
        """
        next_preview = 0.0
        while True:
            if self._stop:
                raise TeachingAborted("界面请求停止")
            now = time.monotonic()
            if now >= next_preview:
                self._pump_preview(device)
                next_preview = now + _PREVIEW_PERIOD_S
            try:
                command = self._commands.get(timeout=_COMMAND_POLL_S)
            except queue.Empty:
                continue
            if command == _CMD_ABORT:
                raise TeachingAborted("用户中止示教")
            if command in (_CMD_PAUSE, _CMD_RESUME):
                self._set_paused(device, paused=command == _CMD_PAUSE)
                continue
            return command

    def _set_paused(self, device, *, paused: bool) -> None:
        """托住 / 松开机械臂。设备不支持就只提示,不假装暂停成功。"""
        from jiuwensymbiosis.calibration.domain.ports import GuidanceHold

        if not isinstance(device, GuidanceHold):
            self._emit("detect", {"ok": False, "reason": "本机型不支持示教中途暂停。"})
            return
        if paused:
            device.hold_arm()
        else:
            device.release_arm()
        self._emit("paused", {"paused": paused})

    def _pump_preview(self, device) -> None:
        """取一帧、跑板检测,把画面与角点数推给界面(失败只提示,不中断示教)。"""
        try:
            frame = device.capture_calibration_frame()
        except Exception as exc:
            self._emit("detect", {"ok": False, "reason": f"取帧失败:{exc}"})
            return
        try:
            self._emit("preview", imaging.to_data_uri(frame.rgb))
        except Exception as exc:
            logger.debug("标定预览帧编码失败: %s", exc)
        detection = self._detect(frame)
        corners = detection.image_points
        self._emit(
            "detect",
            {
                "ok": bool(detection.ok),
                "n_corners": 0 if corners is None else int(len(corners)),
                "reason": detection.reason,
            },
        )

    # ------------------------------------------------------------------ 阶段 2:采集 + 求解
    def _run_capture(self, *, dry_run: bool) -> None:
        self._guarded(lambda: self._capture(dry_run=dry_run))

    def _capture(self, *, dry_run: bool) -> None:
        from jiuwensymbiosis.calibration import execute_calibration

        self._station_count = 0
        phase = "dry_run" if dry_run else "capture"
        with self._log_capture(), self._connected_device(dry_run=dry_run) as (device, spec, _options):
            self._emit(
                "phase",
                {"phase": phase, "msg": "正在校验轨迹与数据契约…" if dry_run else "正在沿示教轨迹采集…"},
            )
            dependencies = self._dependencies(spec, dry_run=dry_run)
            outcome = execute_calibration(
                device,
                self._setup.waypoint_path,
                mount=device.camera_mount,
                dependencies=dependencies,
            )
            limits = _limits_payload(device.camera_mount, dependencies)
            self._emit("done", {"phase": phase, **_outcome_payload(outcome), "limits": limits})

    # ------------------------------------------------------------------ 内部:装配
    def _connected_device(self, *, dry_run: bool = False):
        """上下文管理器:建会话(不启 detector)→ 连接 → 产出标定设备。

        只负责建立与清理,异常一律向上抛给 :meth:`_guarded` 处理 —— 在这里 ``except``
        会让 generator 走完却没 yield 过,调用方的 ``with`` 只会收到一个
        ``RuntimeError: generator didn't yield``,把真正的原因(比如串口打不开)盖掉。
        """
        from contextlib import contextmanager

        from jiuwensymbiosis.calibration.integration.integration import load_adapter_spec

        @contextmanager
        def _ctx():
            spec = load_adapter_spec(self._setup.adapter_module)
            session = (
                self._binding.build_session()
                if self._binding is not None
                else spec.session_factory.from_dict(self._session_config(), include_sidecars=False)
            )
            self._active_session = session
            self._emit("phase", {"phase": "connect", "msg": "正在连接机械臂与相机…"})
            with session:
                device = spec.make_calibration_device(session.env)
                yield device, spec, self._options(dry_run=dry_run)

        return _ctx()

    def _guarded(self, run: Callable[[], None]) -> None:
        """跑一个阶段,把异常翻译成界面事件。

        ``ManualGuidanceRecoveryError`` 单独标 ``fatal``:它意味着机械臂可能仍处于失
        力矩状态,用户必须先用手接住,不能只当作一条普通错误滚进日志。
        """
        from jiuwensymbiosis.agent.lifecycle import HardwareCleanupError
        from jiuwensymbiosis.calibration.domain.ports import ManualGuidanceRecoveryError
        from jiuwensymbiosis.calibration.workflows.preflight import PreflightError

        try:
            run()
        except TeachingAborted as exc:
            # 中止是用户主动按的,不是故障:力矩已在 manual_guidance 退出时恢复。
            logger.info("标定示教已中止: %s", exc)
            self._emit("aborted", {"reason": str(exc)})
        except ManualGuidanceRecoveryError as exc:
            logger.exception("标定示教后力矩恢复失败")
            self._fatal_recovery_error = str(exc)
            self._emit("error", {"reason": str(exc), "fatal": True})
        except HardwareCleanupError as exc:
            logger.exception("标定会话硬件清理未确认")
            self._hardware_cleanup_error = str(exc)
            self._emit("error", {"reason": str(exc), "fatal": False})
        except PreflightError as exc:
            logger.warning("标定预检未通过: %s", exc)
            self._emit("error", {"reason": str(exc), "fatal": False})
        except Exception as exc:
            logger.exception("标定运行失败")
            self._emit("error", {"reason": str(exc), "fatal": False})

    def _session_config(self) -> dict[str, Any]:
        """把标定档案的 ``env_overrides`` 合进本体配置(深拷贝,不动界面在用的那份)。"""
        data = copy.deepcopy(self._setup.config_data)
        overrides = self._setup.calibration_profile.get("env_overrides") or {}
        if not overrides:
            return data
        low_level = data.setdefault("env", {}).setdefault("cfg", {}).setdefault("low_level", {})
        low_level.update(copy.deepcopy(overrides))
        return data

    def _profile_yaml(self) -> Path:
        """标定档案的 ``profile`` 段写成的临时 YAML —— 工作流从文件读取标定参数。

        轨迹空间、可观测性阈值与采集门限属于标定,不属于本体运行配置,所以不去污染
        用户的 YAML。每个引擎实例只生成一份(``_options`` 与 ``_dependencies`` 必须
        指向同一个文件,否则两条路径读到的阈值可能不一致),由 ``close`` 收尾。
        """
        if self._profile_path is not None:
            return self._profile_path
        block = self._setup.calibration_profile.get("profile") or {}
        payload = {
            "calibration": {
                **copy.deepcopy(block),
                "adapter_module": self._setup.adapter_module,
                "output": str(self._setup.out_path),
            }
        }
        # 引擎持有目录而非裸文件描述符:目录对象自带 finalizer,引擎被丢弃或进程退出时
        # 同样会清掉,不必指望 close 一定被调到。
        self._profile_dir = tempfile.TemporaryDirectory(prefix="jiuwen-calib-")
        path = Path(self._profile_dir.name) / "profile.calib.yaml"
        path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
        self._profile_path = path
        return self._profile_path

    def close(self) -> None:
        """删除本次运行生成的临时标定 profile(界面离开标定工具时调用)。"""
        if self._profile_dir is not None:
            self._profile_dir.cleanup()
            self._profile_dir = None
        self._profile_path = None

    def _options(self, *, dry_run: bool):
        from jiuwensymbiosis.calibration import CalibrationRunOptions

        return CalibrationRunOptions(
            config=self._profile_yaml(),
            out=self._setup.out_path,
            dry_run=dry_run,
            n_stations=self._setup.n_stations,
        )

    def _dependencies(self, spec, *, dry_run: bool):
        """复用 CLI 的依赖装配(阈值/profile/reload 校验),只换掉板检测函数。

        走 ``_cli_common.workflow_dependencies`` 是为了让界面与命令行共享同一套阈值
        解析与发布门禁;它内部的 ``detect_fn`` 用裸模块名导入 ``handeye_board``(依赖
        CLI 的 sys.path 布置),界面这边改用包路径导入的版本替换掉。
        """
        from types import SimpleNamespace

        from scripts.calibrate._cli_common import workflow_dependencies

        board = self._setup.board
        args = SimpleNamespace(
            config=self._profile_yaml(),
            out=self._setup.out_path,
            dry_run=dry_run,
            n_stations=self._setup.n_stations,
            board=board.kind,
            squares_x=board.squares_x,
            squares_y=board.squares_y,
            square_size_mm=board.square_size_mm,
            marker_size_mm=board.marker_size_mm,
            min_corners=self._setup.min_corners,
        )
        dependencies = workflow_dependencies(spec, args, live=False)
        return dataclasses.replace(dependencies, detect_fn=self._detect_station)

    def _detect_station(self, frame):
        """采集途中的板检测,兼作进度上报点:``capture_stations`` 每个采集点调用它一次。"""
        detection = self._detect(frame)
        self._station_count += 1
        self._emit(
            "station",
            {"index": self._station_count, "total": self._setup.n_stations, "ok": bool(detection.ok)},
        )
        return detection

    def _detect(self, frame):
        """板检测(包路径导入,不依赖 CLI 的 sys.path 布置)。"""
        from scripts.calibrate.handeye_board import detect_board

        return detect_board(
            frame.rgb,
            self._setup.board.to_board_spec(),
            frame.intrinsics,
            frame.distortion,
            min_corners=self._setup.min_corners,
        )

    def _log_capture(self):
        """把标定子系统的日志接进事件队列(界面的进度条就是这些日志)。"""
        from contextlib import contextmanager

        @contextmanager
        def _ctx():
            handler = QueueLogHandler(self._events, level=logging.INFO)
            target = logging.getLogger(_CALIBRATION_LOGGER)
            previous = target.level
            target.setLevel(logging.INFO)
            target.addHandler(handler)
            try:
                yield
            finally:
                target.removeHandler(handler)
                target.setLevel(previous)

        return _ctx()


def _drain_queue(q: queue.Queue) -> None:
    """清空队列(每阶段开始前丢掉上一阶段的残留命令)。"""
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            return


def _outcome_payload(outcome) -> dict[str, Any]:
    """把 ``RunOutcome`` 摊平成界面可直接渲染的字典。"""
    decision = outcome.decision
    return {
        "candidate": bool(outcome.candidate),
        "artifact_path": None if outcome.artifact_path is None else str(outcome.artifact_path),
        "accept": None if decision is None else bool(decision.accept),
        "failed_checks": [] if decision is None else list(decision.failed_checks),
        "reasons": [] if decision is None else list(decision.reasons),
        "n_stations": 0 if outcome.result is None else int(getattr(outcome.result, "n_stations", 0)),
        "method": "" if outcome.result is None else str(getattr(outcome.result, "method", "")),
        "quality": _quality_payload(outcome.result),
    }


def _limits_payload(mount: str, dependencies) -> dict[str, float]:
    """这一轮验收实际用的门限。

    取自 ``acceptance_policy`` 而不是界面自备一份:界面显示的"合格线"必须就是判 accept /
    reject 的那条线,否则会出现指标标着"良好"、结论却是不通过。策略按 mount 不同,
    eye-in-hand 没有刚性门,那两项就不给。
    """
    from jiuwensymbiosis.calibration.workflows.publication import acceptance_policy

    policy = acceptance_policy(mount, dependencies)
    limits = {
        "reproj_good_px": float(getattr(policy, "reproj_good_px", 1.0)),
        "reproj_warn_px": float(getattr(policy, "reproj_warn_px", 2.0)),
    }
    thresholds = getattr(policy, "target_consistency_thresholds", None)
    if thresholds is not None:
        limits["translation_std_mm"] = float(thresholds.max_translation_std_mm)
        limits["rotation_spread_deg"] = float(thresholds.max_rotation_spread_deg)
    return limits


def _quality_payload(result) -> dict[str, Any]:
    """抽出界面要显示的质量指标(缺项一律留空,不猜)。"""
    quality = getattr(result, "quality", None)
    if quality is None:
        return {}
    payload: dict[str, Any] = {}
    reprojection = getattr(quality, "reprojection", None)
    summary = getattr(reprojection, "summary", None)
    if summary is not None:
        payload["reprojection_px"] = {"mean": float(summary.mean), "max": float(summary.max)}
    axxb = getattr(quality, "axxb_residual", None)
    if axxb is not None:
        payload["axxb"] = {
            "rotation_deg": float(axxb.rotation_deg.mean),
            "translation_mm": float(axxb.translation_mm.mean),
        }
    consistency = getattr(quality, "target_consistency", None)
    if consistency is not None:
        payload["rigidity"] = {
            # 刚性不变量按 mount 不同:eye-in-hand 看 base_target,eye-to-hand 看
            # flange_target。透传出去,界面才不会把两者都标成同一个坐标系。
            "invariant_frame": str(consistency.invariant_frame),
            "translation_mm": float(consistency.translation_residual_mm.max),
            # 门禁判的是平移的 std、旋转的 max(见 EyeToHandAcceptancePolicy.decide),
            # 所以这两个统计量都要给,界面才能显示被判的那一个。
            "translation_std_mm": float(consistency.translation_residual_mm.std),
            "rotation_deg": float(consistency.rotation_residual_deg.max),
        }
    return payload
