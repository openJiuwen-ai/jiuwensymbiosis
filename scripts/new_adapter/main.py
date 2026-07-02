# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Adapter onboarding wizard — generate a consistent skeleton, then guide the
engineer through filling the driver until the adapter actually drives.

Run from the repository root::

    python -m scripts.new_adapter.main                       # interactive
    python -m scripts.new_adapter.main --name my_robot \\
        --dof 6 --joint --end-effector parallel --non-interactive
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import checks
from .render import render_all
from .spec import CONNECTIONS, END_EFFECTORS, TOOL_GEOMETRIES, Spec, ask_interactive, validate_name

REPO_ROOT = checks.REPO_ROOT

# Named logger shared with ``spec.py`` (same object). Configured in ``main()``
# with a raw ``%(message)s`` handler so wizard output looks exactly like print.
logger = logging.getLogger("new_adapter")


def _configure_logging() -> None:
    """Route wizard output through a raw-message handler on stdout.

    Mirrors ``scripts/validate_adapter.py`` / ``scripts/smoke_test_adapter.py``:
    the message is emitted verbatim (no level/timestamp prefix), so the CLI
    reads like ``print`` while satisfying the "use logging" coding rule. The
    root level stays at WARNING to mute third-party INFO chatter.
    """
    if logger.handlers:
        return
    logging.getLogger().setLevel(logging.WARNING)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

# One-line contract hints shown in Phase B for each still-mock method.
METHOD_HINTS = {
    "connect": "打开硬件连接（串口/CAN/socket），必须幂等。",
    "disconnect": "释放硬件资源，幂等、任意状态可调。",
    "get_pose": "读当前末端位姿，返回带 x,y,z(+旋转) 字段的对象（FLANGE 系）。",
    "home": "阻塞式回零/回原点。",
    "move_to_pose_blocking": "阻塞式笛卡尔运动到 pose（FLANGE 系，mm/deg）。",
    "move_joint_blocking": "阻塞式关节运动到 q（度）。",
    "set_gripper": "on=True 闭合 / False 张开。",
    "set_suction": "on=True 吸附 / False 释放。",
    "grab_frames": "返回 (rgb HxWx3 uint8, depth HxW float32 米) 或 None。",
    "get_grasp_info_simple": "检测 object_name 并反投影到基座；需检测服务+手眼标定（见 docs §6.4 / piper）。",
    "pixel_to_base_xyz": "像素(u,v)+depth_m → 基座 XYZ（需手眼标定）。",
    "analyze_scene": "基于 object_name 的高层场景分析。",
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="new_adapter",
        description="生成并带填一个 jiuwensymbiosis 适配器",
    )
    parser.add_argument("--name", help="适配器/机器人名字 (小写, 如 my_robot)")
    parser.add_argument("--dof", type=int, choices=(4, 6), default=6, help="自由度 (默认 6)")
    parser.add_argument("--joint", action="store_true", help="支持关节空间运动")
    parser.add_argument(
        "--end-effector", choices=END_EFFECTORS, default="none", help="末端执行器 (默认 none)"
    )
    parser.add_argument("--camera", action="store_true", help="有相机 (RGB)")
    parser.add_argument("--detection", action="store_true", help="自然语言目标检测 (蕴含 --camera)")
    parser.add_argument(
        "--tool",
        choices=TOOL_GEOMETRIES,
        default="straight_down",
        help="工具几何 (默认 straight_down)",
    )
    parser.add_argument(
        "--connection",
        choices=CONNECTIONS,
        default="can",
        help="硬件连接方式；目前 can 会生成较完整模板，其它方式为空模板 (默认 can)",
    )
    parser.add_argument("--non-interactive", action="store_true", help="不交互，全用 flag")
    parser.add_argument("--force", action="store_true", help="目标已存在时覆盖重写")
    return parser.parse_args(argv)


def _spec_from_args(args: argparse.Namespace) -> Spec:
    err = validate_name(args.name or "")
    if err is not None:
        raise ValueError(f"--name 无效: {err}")
    spec = Spec(
        name=args.name,
        dof=args.dof,
        joint=args.joint,
        end_effector=args.end_effector,
        camera=args.camera,
        detection=args.detection,
        tool_geometry=args.tool,
        connection=args.connection,
    )
    return spec.normalized()


# ---------------------------------------------------------------------------
# Phase A — generate
# ---------------------------------------------------------------------------


def _adapter_dir(spec: Spec) -> Path:
    return REPO_ROOT / "jiuwensymbiosis" / "adapters" / spec.name


def _generate(spec: Spec) -> list[str]:
    files = render_all(spec)
    written: list[str] = []
    abs_paths: list[Path] = []
    for rel, text in files.items():
        path = REPO_ROOT / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        written.append(rel)
        if path.suffix == ".py":
            abs_paths.append(path)
    checks.format_with_black(abs_paths)
    return written


def _phase_a(spec: Spec, with_smoke: bool) -> None:
    logger.info("\n[阶段 A] 生成骨架")
    logger.info(f"  能力: {', '.join(spec.capabilities)}")
    logger.info(f"  连接方式: {spec.connection}")
    if spec.connection != "can":
        if spec.connection == "custom":
            logger.info("  提示: custom 会生成最空连接模板，需要你按硬件 SDK 完全填充。")
        else:
            logger.info(f"  提示: {spec.connection} 当前先生成空连接模板，后续会实现更完整模板。")
    written = _generate(spec)
    for rel in written:
        logger.info(f"  + {rel}")
    module = f"jiuwensymbiosis.adapters.{spec.name}"
    logger.info("  自检（此刻全部还是示例 mock，可离线验证）：")
    _print_result(checks.run_validate(module))
    if with_smoke:
        _print_result(checks.run_smoke(module))


# ---------------------------------------------------------------------------
# Phase B — guided completion
# ---------------------------------------------------------------------------


def _print_result(res: checks.Result) -> None:
    mark = "✓" if res.ok else "✗"
    logger.info(f"  [{mark}] {res.title}")
    if res.detail:
        for line in res.detail.splitlines():
            logger.info(f"        {line}")


_FILE_ORDER = ["lowlevel.py", "env.py", "api.py", "config.py", "session.py"]


def _ordered_pending(pending: dict[str, list[str]]) -> list[tuple[str, list[str]]]:
    """Driver first, then env/api — the order an engineer fills them in."""
    ordered = [(f, pending[f]) for f in _FILE_ORDER if f in pending]
    ordered += [(f, m) for f, m in pending.items() if f not in _FILE_ORDER]
    return ordered


def _print_pending(spec: Spec, pending: dict[str, list[str]]) -> None:
    ordered = _ordered_pending(pending)
    total = sum(len(methods) for _, methods in ordered)
    pkg = f"jiuwensymbiosis/adapters/{spec.name}"
    logger.info(f"\n  还有 {total} 处需要你用机器人的真实 SDK / 标定来补充。")
    logger.info(f"  配置文件: configs/{spec.name}/default.yaml")
    if spec.connection == "can":
        logger.info(
            "  CAN 模板已把 can_port / can_bitrate / move_speed / tool_offset_mm "
            "从 config 传到 lowlevel driver。"
        )
    elif spec.connection == "custom":
        logger.info("  custom 连接方式已生成最空模板：请自行在 config.py/yaml 和 lowlevel.py 中补硬件 SDK 字段。")
    else:
        logger.info(f"  {spec.connection} 连接方式当前是空模板，后续会实现更完整模板；请先按硬件 SDK 手动补齐。")
    logger.info("  请在编辑器里打开下面的文件，找到对应函数，把示例实现换成你的代码：\n")
    for fname, methods in ordered:
        logger.info(f"    {pkg}/{fname}")
        for m in methods:
            logger.info(f"        - {m}()   {METHOD_HINTS.get(m, '')}")
    first_file, first_methods = ordered[0]
    logger.info(
        f"\n  不知道从哪开始？先打开 {pkg}/{first_file}，找到 {first_methods[0]}() 这个函数，"
        "把里面的示例实现换成你的 SDK 调用。"
    )
    logger.info(
        "  每补好一个函数，删掉该函数体里标着 `GENERATED-MOCK` 的那一行注释——它就是「这里待填」的记号。"
    )


def _phase_b_interactive(spec: Spec) -> None:
    module = f"jiuwensymbiosis.adapters.{spec.name}"
    adir = _adapter_dir(spec)
    logger.info("\n[阶段 B] 补全适配器（按清单逐个填，我每次帮你复检）")
    while True:
        pending = checks.scan_pending(adir)
        if not pending:
            break
        _print_pending(spec, pending)
        ans = (
            input("\n  补完后回车：我重新扫描记号并做一次静态校验(结构)；输入 q 暂停: ")
            .strip()
            .lower()
        )
        if ans == "q":
            logger.info("  已暂停。稍后重跑本命令即可从这里继续。")
            return
        _print_result(checks.run_validate(module))
    logger.info("\n  清单已全部补完。")
    res = checks.run_validate(module)
    _print_result(res)
    if res.ok:
        logger.info(f"\n✅ 结构校验通过。下一步：填好 configs/{spec.name}/default.yaml 的真实连接参数，")
        logger.info("   再用 `with session:` 接真机调试（mock 阶段的冒烟已在阶段 A 通过）。")
    else:
        logger.info("\n⚠️ 结构校验未通过，请按上面的 [E-/C-/A-/D-] 提示修复后回车重试。")


def _phase_b_summary(spec: Spec) -> None:
    """Non-interactive: print what's left + the resume command, no blocking."""
    adir = _adapter_dir(spec)
    pending = checks.scan_pending(adir)
    if pending:
        _print_pending(spec, pending)
    logger.info("\n  继续补全（逐项填，每次自动复检）：")
    logger.info(f"    python -m scripts.new_adapter.main --name {spec.name}  # 重跑即从未完成处续跑")


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    args = _parse_args(argv)

    try:
        if args.non_interactive:
            if not args.name:
                raise SystemExit("--non-interactive 需要 --name")
            spec = _spec_from_args(args)
        elif args.name:
            spec = _spec_from_args(args)
        else:
            spec = ask_interactive()
    except ValueError as exc:
        raise SystemExit(str(exc)) from None

    adir = _adapter_dir(spec)
    module = f"jiuwensymbiosis.adapters.{spec.name}"
    resume = adir.exists() and not args.force

    if resume:
        logger.info(f"\n适配器 {spec.name} 已存在，进入续跑（阶段 B）。")
        res = checks.run_validate(module)
        _print_result(res)
        if not res.ok:
            logger.info("  生成物结构有误，可用 --force 重新生成。")
    else:
        _phase_a(spec, with_smoke=not args.non_interactive)

    if args.non_interactive:
        _phase_b_summary(spec)
        # Exit code reflects structural validity only (pending mocks are expected).
        return 0 if checks.run_validate(module).ok else 1

    _phase_b_interactive(spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
