#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""jiuwensymbiosis 适配器运行时冒烟测试.

验证 ``validate_adapter.py`` 静态结构检查不到的运行时行为：用适配器**自己的
Env** 配一个桩驱动（见 ``_StubDriver``）驱动它的 Api，逐个调用
``list_tool_meta`` 列出的动作，断言不抛异常、返回值可 JSON 序列化。这能把
"字段名拼写错""get_observation 在 mock 下崩"之类的运行时错误前移到接入期。

用法::

    python scripts/smoke_test_adapter.py --module jiuwensymbiosis.adapters.piper
    python scripts/smoke_test_adapter.py --path adapters/my_robot/
    python scripts/smoke_test_adapter.py --module ... --config configs/so101/so101.yaml

诚实边界
--------
泛型冒烟无法构造每个工具的合法参数（例如 ``goto_xyzr`` 的可达坐标依赖具体
机器人），只保证：
  * 能枚举出所有有效工具；
  * 能用启发式默认值调用的工具不崩、返回可序列化；
  * 无法构造参数的工具被明确 SKIP（而不是假装通过）。

桩驱动只兑现 ``env/protocol.py`` 里的**契约**。一个动作若要用契约之外的驱动
成员，这里会以 ``AttributeError`` 报出来——那是真实发现，不是本脚本的缺陷。

退出码：有 FAIL 时非零。
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

# 仓根必须先进 sys.path：以 ``python scripts/smoke_test_adapter.py`` 运行时，
# sys.path[0] 是 scripts/ 而不是仓根，``import jiuwensymbiosis`` 会安静地落到
# 已安装的那份副本上——即检了另一个 checkout，却报告本仓通过。
# 用替换而不是追加：sys.path[0] 上的 scripts/ 本身就是重名隐患（一个
# scripts/json.py 会遮蔽标准库），而本脚本只 import 标准库，不需要它。
_REPO_ROOT = Path(__file__).resolve().parent.parent
if sys.path and Path(sys.path[0]).resolve() == Path(__file__).resolve().parent:
    sys.path[0] = str(_REPO_ROOT)
elif str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

logger = logging.getLogger("smoke_test_adapter")

# Default values for common positional/keyword params by name. Tools whose
# required params aren't in this table are SKIPPED (we can't guess a safe value).
_DEFAULTS_BY_NAME = {
    "object_name": "box",
    "object": "box",
    "target": "box",
    "text_prompt": "box",
    "x": 200.0,
    "y": 0.0,
    "z": 250.0,
    "r": 0.0,
    "rz": 0.0,
    "rx": 180.0,
    "ry": 0.0,
    "u": 320.0,
    "v": 240.0,
    "depth_m": 0.5,
    "width_mm": 70.0,
    "force_n": None,
    "q": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    # Filled from the body's own joint names below when it states any; an empty mapping
    # would only be refused, and a guessed name would only be refused differently.
    "targets": {},
}


class _StubPose:
    """A Cartesian pose every framework field read resolves against."""

    __slots__ = ("x", "y", "z", "rx", "ry", "rz")

    def __init__(self) -> None:
        self.x, self.y, self.z = 200.0, 0.0, 250.0
        self.rx, self.ry, self.rz = 180.0, 0.0, 0.0

    def __repr__(self) -> str:
        return f"_StubPose(x={self.x}, y={self.y}, z={self.z}, rx={self.rx}, ry={self.ry}, rz={self.rz})"


class _StubDriver:
    """Answer every driver Protocol slice in ``env/protocol.py`` with a plausible typed value.

    Deliberately NOT a ``MagicMock``: a member the Protocols never promised must raise
    ``AttributeError``, because "the Api reads a driver field outside the contract" is exactly
    the class of bug this smoke test exists to catch — a MagicMock would absorb it and report
    a vacuous pass.

    Where a plausible value is body-specific (camera frames, hand-eye calibration) it answers
    "unavailable" rather than inventing one: every adapter has a defined degrade path for a
    missing camera, and running that path is worth more than a fake image.
    """

    def __init__(self, joint_dim: int = 6, joint_names: list[str] | None = None) -> None:
        self._joint_dim = joint_dim
        # The body's own joint names when it states them (env.joint_limits is name-keyed on
        # every adapter), so a named read answers with keys the Api will recognise.
        names = joint_names or [f"joint{i + 1}" for i in range(joint_dim)]
        self._joint_names = dict.fromkeys(names, 0.0)

    # -- RobotDriver / lifecycle
    @staticmethod
    def close() -> None:
        return None

    @staticmethod
    def connect() -> None:
        return None

    @staticmethod
    def disconnect() -> None:
        return None

    # -- CartesianDriver
    @property
    def home_pose(self) -> _StubPose:
        return _StubPose()

    @property
    def z_min_safe(self) -> float:
        return 50.0

    @property
    def flange_z_min_safe(self) -> float:
        return 40.0

    @property
    def tool_offset_mm(self) -> float:
        return 0.0

    @staticmethod
    def home() -> None:
        return None

    @staticmethod
    def recovery_home() -> None:
        return None

    @staticmethod
    def retreat_home() -> None:
        return None

    @staticmethod
    def get_pose() -> _StubPose:
        return _StubPose()

    @staticmethod
    def move_to_pose_blocking(*args: Any, **kwargs: Any) -> None:
        return None

    # -- JointDriver (both encodings) / ServoDriver
    def get_angles(self) -> list[float]:
        return [0.0] * self._joint_dim

    @staticmethod
    def move_joint_blocking(*args: Any, **kwargs: Any) -> None:
        return None

    def get_joint_positions(self) -> dict[str, float]:
        return dict(self._joint_names)

    @staticmethod
    def move_joints_blocking(targets: dict[str, float], **kwargs: Any) -> dict:
        return {"ok": True, "reached": dict(targets)}

    @staticmethod
    def servo_to_pose(*args: Any, **kwargs: Any) -> bool:
        return True

    # -- BaseDriver / ContinuousBaseDriver
    @staticmethod
    def navigate_relative(*args: Any, **kwargs: Any) -> dict:
        return {"ok": True}

    @staticmethod
    def navigate_arc(*args: Any, **kwargs: Any) -> dict:
        return {"ok": True}

    @staticmethod
    def start_base_drive(**kwargs: Any) -> str:
        return "stub-drive"

    @staticmethod
    def base_drive_running(handle: Any) -> bool:
        return False

    @staticmethod
    def steer_base_drive(handle: Any, bearing_rad: float) -> None:
        return None

    @staticmethod
    def hold_base_drive(handle: Any) -> None:
        return None

    @staticmethod
    def stop_base_drive(handle: Any) -> dict:
        return {"ok": True}

    # -- LifterDriver / WaistDriver
    @staticmethod
    def set_lifter(q_lifter: dict[str, float]) -> None:
        return None

    @staticmethod
    def turn_waist(delta_rad: float) -> None:
        return None

    # -- CameraDriver / VisionDriver.
    # The calibration is answered for real (a generic pinhole K, identity transforms): those are
    # the SHAPE the contract promises, not a body-specific value, and supplying them is what lets
    # the projection math actually run — where a field-name typo would show up. Frames stay None:
    # a fake image would be inventing content, and every adapter has a defined no-camera path.
    @property
    def intrinsics(self) -> Any:
        import numpy as np

        return np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])

    @property
    def tf_flange_cam(self) -> Any:
        import numpy as np

        return np.eye(4)

    @property
    def tf_base_cam(self) -> Any:
        import numpy as np

        return np.eye(4)

    @property
    def calibration(self) -> dict:
        return {"intrinsics": self.intrinsics, "tf_base_cam": self.tf_base_cam}

    @property
    def camera_available(self) -> bool:
        return False

    @staticmethod
    def grab_frames(*args: Any, **kwargs: Any) -> None:
        return None

    # -- SuctionDriver / GripperDriver
    @property
    def suction_state(self) -> bool:
        return False

    @property
    def suction_di_last(self) -> None:
        return None

    @staticmethod
    def set_suction(on: bool) -> None:
        return None

    @staticmethod
    def set_gripper(on: bool) -> None:
        return None

    @property
    def gripper_state(self) -> bool:
        return False


def _attach_stub_driver(env: Any) -> bool:
    """Bind a ``_StubDriver`` to ``env`` so its tools run without hardware.

    Goes through the public ``low_level`` seam — the binding an env accepts while nothing is
    bound yet. Returns False when the env refuses (already bound, or no setter at all), so the
    caller reports that rather than silently smoke-testing a disconnected env.
    """
    limits = getattr(env, "joint_limits", None)
    try:
        env.low_level = _StubDriver(
            joint_dim=_env_joint_dim(env) or 6,
            joint_names=list(limits) if limits else None,
        )
    except (AttributeError, TypeError) as exc:
        logger.debug("%s refused a stub driver: %s", type(env).__name__, exc)
        return False
    return True


def _build_args(func: Any) -> tuple[dict[str, Any], Optional[str]]:
    """Heuristically build call kwargs for a bound tool function.

    Returns ``(kwargs, skip_reason)``. ``skip_reason`` is non-None when a
    required parameter has no default and no known safe value.
    """
    sig = inspect.signature(func)
    kwargs: dict[str, Any] = {}
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if param.default is not inspect.Parameter.empty:
            continue  # leave it to the function's default
        if name in _DEFAULTS_BY_NAME:
            kwargs[name] = _DEFAULTS_BY_NAME[name]
        else:
            return kwargs, f"required param {name!r} has no safe default — skipping"
    return kwargs, None


def _env_joint_dim(env: Any) -> int | None:
    """Arm-joint count the env exposes, for sizing a generic ``move_joint`` ``q``.

    The fixed ``q = [0]*6`` default is wrong for arms whose joint count differs
    (e.g. a 5-joint arm with a real length check in its driver). Prefer
    ``joint_limits`` (SafetyRail keys its length off it); fall back to a connected
    observation's ``joints``. ``None`` when neither is available (leave the default).
    """
    if env is None:
        return None
    jl = getattr(env, "joint_limits", None)
    if jl:
        return len(jl)
    try:
        joints = getattr(env.get_observation(), "joints", None)
    except Exception:
        return None
    return len(joints) if joints else None


def smoke_test_api(api: Any, *, env: Any = None) -> list[dict[str, Any]]:
    """Call every emitted tool on ``api`` with heuristic defaults.

    Args:
      api: a ``BaseRobotApi`` instance (already constructed against an env).
      env: optional env; when given, tools are gated by ``api ∩ env`` capabilities
        exactly as ``build_robot_tools`` would gate them.

    Returns a list of ``{name, status, ...}`` dicts (``status`` ∈ pass/fail/skip).
    Every result (including return values) is JSON-serializable so the report
    can be dumped to a file.
    """
    from jiuwensymbiosis.tools.builder import list_tool_meta

    joint_dim = _env_joint_dim(env)
    joint_names = list(getattr(env, "joint_names", None) or ())
    results: list[dict[str, Any]] = []
    for meta in list_tool_meta(api, env=env):
        name = meta["name"]
        func = getattr(api, name, None)
        if func is None:
            results.append({"name": name, "status": "skip", "reason": "method not bound on api"})
            continue
        kwargs, skip_reason = _build_args(func)
        if skip_reason is not None:
            results.append({"name": name, "status": "skip", "reason": skip_reason})
            continue
        if joint_dim is not None and "q" in kwargs:
            kwargs["q"] = [0.0] * joint_dim
        if "targets" in kwargs and joint_names:
            # One real joint at 0.0: enough to drive the name→vector conversion and the
            # hold-the-rest path without commanding a body-specific angle.
            kwargs["targets"] = {joint_names[0]: 0.0}
        try:
            ret = func(**kwargs)
        except Exception as exc:
            results.append(
                {"name": name, "status": "fail", "error": f"{type(exc).__name__}: {exc}"}
            )
            continue
        entry: dict[str, Any] = {"name": name, "status": "pass"}
        if ret is None:
            entry["returns_none"] = True
        else:
            entry["return"] = _jsonable(ret)
        results.append(entry)
    return results


def _jsonable(obj: Any) -> Any:
    """Coerce a tool return value into something json.dumps accepts.

    numpy arrays/scalars → lists/floats; everything else is best-effort.
    Falls back to ``repr`` so a non-serializable return never crashes the report.
    """
    try:
        import numpy as np

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
    except ImportError:
        pass
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return repr(obj)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _resolve_module(module_str: Optional[str], path_str: Optional[str]) -> str:
    if module_str:
        return module_str
    if path_str:
        p = Path(path_str).resolve()
        # Inside this repo the answer is exact. Don't search the absolute path for the first
        # "jiuwensymbiosis" — the repo directory carries that name too, so a checkout at
        # <...>/jiuwensymbiosis/jiuwensymbiosis/adapters/piper would yield the package twice.
        try:
            return ".".join(p.relative_to(_REPO_ROOT).parts).replace(".py", "")
        except ValueError:
            pass
        parts = list(p.parts)
        try:
            idx = parts.index("jiuwensymbiosis")
        except ValueError:
            idx = -1
            for i, part in enumerate(parts):
                if part == "adapters":
                    idx = i - 1 if i > 0 else -1
                    break
            if idx < 0:
                idx = len(parts) - 1
        return ".".join(parts[idx:]).replace(".py", "")
    return ""


def _load_builder(module_str: str):
    """Import the adapter package and return its ``build_xxx_session`` callable."""
    module = importlib.import_module(module_str)
    for attr in ("build_session",):
        candidate = getattr(module, attr, None)
        if callable(candidate):
            return candidate
    # Fallback: the first module-level attribute whose name starts with build_.
    for attr_name in dir(module):
        if attr_name.startswith("build_") and attr_name.endswith("_session"):
            candidate = getattr(module, attr_name)
            if callable(candidate):
                return candidate
    raise AttributeError(f"no build_xxx_session builder found in {module_str}")


def _default_config_for(module_str: str) -> Path | None:
    """The shipped YAML for this adapter, by the ``configs/<name>/<name>.yaml`` convention."""
    name = module_str.rstrip(".").rsplit(".", 1)[-1]
    candidate = _REPO_ROOT / "configs" / name / f"{name}.yaml"
    return candidate if candidate.is_file() else None


def _build_session(builder: Any, module_str: str, config: str | None) -> Any:
    """Build the adapter's session from its shipped YAML, falling back to an empty config dict.

    The shipped config comes first because it is how the adapter actually runs: an empty dict
    drops ``calib_path`` and the like, so a vision action then fails for a reason that has
    nothing to do with the adapter's code. An empty dict is only a fallback for an adapter that
    ships no config — and it only works when the config dataclass defaults every field
    (So101Config requires ``port``). There is no bare ``builder()`` path: ``make_builder``
    always returns ``build(cfg)``, so calling it with no argument could only ever raise — and
    used to, masking the real config error with a confusing TypeError.
    """
    import yaml

    from jiuwensymbiosis.adapters._common.config import parse_config
    from jiuwensymbiosis.adapters._common.resources import complete_resource_keys

    source = Path(config).expanduser().resolve() if config else _default_config_for(module_str)
    data = yaml.safe_load(source.read_text(encoding="utf-8")) if source is not None else {}
    load_config = getattr(getattr(builder, "config_factory", None), "from_dict", None)
    resource_keys = getattr(builder, "resource_keys", None)
    if not callable(load_config) or not callable(resource_keys):
        raise TypeError("session builder must expose config_factory.from_dict and resource_keys(cfg)")
    data, cfg = parse_config(builder.config_factory, data, source_dir=source.parent if source is not None else None)
    # Validate admission before installing a stub. No lock or hardware is opened.
    complete_resource_keys(resource_keys(cfg), cfg, physical_device_id=data.get("physical_device_id"))
    return builder(cfg)


def _configure_logging() -> None:
    logging.getLogger().setLevel(logging.WARNING)
    for noisy in ("common", "openjiuwen"):
        logging.getLogger(noisy).addFilter(lambda _record: False)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False


def main() -> int:
    _configure_logging()
    parser = argparse.ArgumentParser(
        description="jiuwensymbiosis 适配器运行时冒烟测试 (用 MockEnv 驱动每个动作)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--module", "-m", type=str, default=None, help="适配器模块路径")
    parser.add_argument(
        "--path", "-p", type=str, default=None, help="适配器目录路径 (自动推导模块)"
    )
    parser.add_argument(
        "--config", "-c", type=str, default=None,
        help="配置 YAML；缺省先试空配置，再回退到 configs/<name>/<name>.yaml",
    )
    parser.add_argument("--json", action="store_true", help="输出 JSON 而非格式化报告")
    args = parser.parse_args()

    module_str = _resolve_module(args.module, args.path)
    if not module_str:
        parser.error("需要 --module 或 --path")

    # In --json mode the only thing on stdout is the JSON payload (so it can be
    # piped to jq / redirected to a file); the human-readable banner is skipped.
    if not args.json:
        import jiuwensymbiosis

        logger.info("=" * 65)
        logger.info(" jiuwensymbiosis 适配器冒烟测试")
        logger.info(f" 目标: {module_str}")
        # Name the checkout under test: an installed copy shadowing this repo is a silent
        # false pass, and this line is what makes it visible.
        logger.info(f" 包路径: {jiuwensymbiosis.__file__}")
        logger.info("=" * 65)
        logger.info("")

    try:
        builder = _load_builder(module_str)
        session = _build_session(builder, module_str, args.config)
        api = session.api
        env = session.env
        # A stub driver instead of session.connect(): connecting a real env would open a CAN
        # bus / serial port / ROS graph that no CI machine has. Binding the stub is what makes
        # the tools actually RUN — without it every hardware-touching action fails identically
        # with "env not connected" and the report says nothing about the adapter.
        if not _attach_stub_driver(env):
            logger.error(
                f"{type(env).__name__} 不接受通过 `low_level` 绑定驱动，无法注入桩驱动。\n"
                "请在 Python 里绑定自己的假驱动后调用 "
                "smoke_test_api(api, env=env)。"
            )
            return 2
    except Exception as exc:
        logger.error(f"无法构造适配器 session: {type(exc).__name__}: {exc}")
        logger.error(
            "提示：用 --config <yaml> 指定一份配置；若 builder 还需要更多必需字段，"
            "请在 Python 里手动构造 session 后调用 smoke_test_api(api, env=env)。"
        )
        return 2

    results = smoke_test_api(api, env=env)

    if args.json:
        logger.info(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        _print_report(results)

    failures = sum(1 for r in results if r["status"] == "fail")
    return 1 if failures else 0


def _print_report(results: list[dict[str, Any]]) -> None:
    passed = [r for r in results if r["status"] == "pass"]
    failed = [r for r in results if r["status"] == "fail"]
    skipped = [r for r in results if r["status"] == "skip"]

    logger.info(f" PASS ({len(passed)}):")
    for r in passed:
        logger.info(f"  [OK] {r['name']}")
    logger.info("")
    if skipped:
        logger.info(f" SKIP ({len(skipped)}):")
        for r in skipped:
            logger.info(f"  [--] {r['name']} — {r.get('reason', '')}")
        logger.info("")
    if failed:
        logger.error(f" FAIL ({len(failed)}):")
        for r in failed:
            logger.error(f"  [XX] {r['name']} — {r.get('error', '')}")
        logger.info("")

    # An action that reached a driver member the stub does not have is a different finding from
    # a logic crash: the Api went outside the contract in env/protocol.py. Name it as such so
    # the fix is aimed at the Protocol (or the Api), not at this script.
    contract_gaps = sorted(
        {
            r["error"].rsplit("attribute ", 1)[-1].strip("'\"")
            for r in failed
            if "_StubDriver" in r.get("error", "") and "has no attribute" in r.get("error", "")
        }
    )
    if contract_gaps:
        logger.error(
            f"契约缺口：Api 用到了 env/protocol.py 未声明的驱动成员 {contract_gaps}。\n"
            "桩驱动只兑现已声明的 Protocol，所以这类失败是真实发现——要么把成员补进 Protocol，\n"
            "要么让 Api 改走已声明的动词。"
        )
        logger.info("")

    logger.info("=" * 65)
    if failed:
        logger.error(f" 结果: {len(failed)} FAIL — 请修复运行时崩溃（先排除上方契约缺口）")
    else:
        logger.info(f" 结果: {len(passed)} pass, {len(skipped)} skip — 无崩溃")
    logger.info("=" * 65)


if __name__ == "__main__":
    raise SystemExit(main())
