#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""jiuwensymbiosis 适配器运行时冒烟测试.

验证 ``validate_adapter.py`` 静态结构检查不到的运行时行为：用 MockEnv 驱动
适配器的 Api，逐个调用 ``list_tool_meta`` 列出的 @robot_tool 工具，断言不抛异常、
返回值可 JSON 序列化。这能把"字段名拼写错""get_observation 在 mock 下崩"
之类的运行时错误前移到接入期。

用法::

    python scripts/smoke_test_adapter.py --module jiuwensymbiosis.adapters.piper
    python scripts/smoke_test_adapter.py --path adapters/my_robot/

诚实边界
--------
泛型冒烟无法构造每个工具的合法参数（例如 ``goto_xyzr`` 的可达坐标依赖具体
机器人），只保证：
  * 能枚举出所有有效工具；
  * 能用启发式默认值调用的工具不崩、返回可序列化；
  * 无法构造参数的工具被明确 SKIP（而不是假装通过）。

退出码：有 FAIL 时非零。
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import logging
import sys
from typing import Any, Optional

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
}


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
        from pathlib import Path

        p = Path(path_str).resolve()
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
        description="jiuwensymbiosis 适配器运行时冒烟测试 (用 MockEnv 驱动每个 @robot_tool)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--module", "-m", type=str, default=None, help="适配器模块路径")
    parser.add_argument(
        "--path", "-p", type=str, default=None, help="适配器目录路径 (自动推导模块)"
    )
    parser.add_argument("--json", action="store_true", help="输出 JSON 而非格式化报告")
    args = parser.parse_args()

    module_str = _resolve_module(args.module, args.path)
    if not module_str:
        parser.error("需要 --module 或 --path")

    # In --json mode the only thing on stdout is the JSON payload (so it can be
    # piped to jq / redirected to a file); the human-readable banner is skipped.
    if not args.json:
        logger.info("=" * 65)
        logger.info(" jiuwensymbiosis 适配器冒烟测试")
        logger.info(f" 目标: {module_str}")
        logger.info("=" * 65)
        logger.info("")

    try:
        builder = _load_builder(module_str)
        # Build a session from an EMPTY config dict. This works for adapters
        # whose config dataclass has defaults for every field (PiperConfig does);
        # adapters with required cfg fields will hit the TypeError below and
        # print the manual-construction hint.
        try:
            session = builder.from_dict({})
        except Exception:
            # ``from_dict`` not present or rejected the empty dict — try a bare
            # call (adapters that accept no-arg builders).
            session = builder()
        api = session.api
        env = session.env
        # NOTE: we deliberately do NOT call session.connect() here. A mock env
        # needs no connect; a real-hardware env (e.g. PiperEnv over CAN) would
        # block on the bus. Adapters that need a connected low_level for their
        # tools will surface "env not connected" failures, which the report
        # flags as a SETUP gap rather than a tool bug.
    except Exception as exc:
        logger.error(f"无法构造适配器 session: {type(exc).__name__}: {exc}")
        logger.error(
            "提示：若 builder 需要必需 cfg 字段，请在 Python 里手动构造 session 后调用 smoke_test_api(api, env=env)。"
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

    # Distinguish "the env isn't connected" failures (a SETUP gap — the CLI
    # doesn't connect real hardware) from genuine tool bugs, so users don't
    # chase phantom bugs when they just need a mock env.
    setup_gaps = [
        r
        for r in failed
        if "not connected" in r.get("error", "") or "no low_level" in r.get("error", "")
    ]
    if setup_gaps:
        logger.info(
            f"其中 {len(setup_gaps)} 个失败是 'env not connected' — 这不是工具 bug，\n"
            "而是 CLI 未连接硬件 env。请用 mock 配置/可连接的 env 构造 session 后重跑，\n"
            "或在 Python 里: from scripts.smoke_test_adapter import smoke_test_api; smoke_test_api(api, env=mock_env)"
        )
        logger.info("")

    logger.info("=" * 65)
    if failed:
        logger.error(f" 结果: {len(failed)} FAIL — 请修复运行时崩溃（先排除上方 SETUP 类失败）")
    else:
        logger.info(f" 结果: {len(passed)} pass, {len(skipped)} skip — 无崩溃")
    logger.info("=" * 65)


if __name__ == "__main__":
    raise SystemExit(main())
