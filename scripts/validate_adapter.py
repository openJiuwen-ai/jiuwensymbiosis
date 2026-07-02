#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""jiuwensymbiosis 适配器兼容性验证工具.

用法::

    # 通过模块路径
    python scripts/validate_adapter.py --module jiuwensymbiosis.adapters.my_robot

    # 通过文件路径 (自动推导模块路径)
    python scripts/validate_adapter.py --path adapters/my_robot/

    # 详细输出
    python scripts/validate_adapter.py --module ... --verbose

    # 仅输出错误
    python scripts/validate_adapter.py --module ... --errors-only

检查项:
    ERROR (必须修复)  — Config/Env/Api 结构不满足框架约定
    WARN  (建议修复)  — Mixin 未覆写 / 能力不一致
    INFO  (仅供参考)  — 标记能力 / 可选特性
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import inspect
import logging
import sys
from pathlib import Path
from typing import Any, Optional

# Single source of truth — import from the package so the validator and the
# ``__init_subclass__`` gate in env/base.py can never drift out of sync.
from jiuwensymbiosis.env.base import KNOWN_CAPABILITIES

# Capability → contract maps, single-sourced in the package so the validator
# (checker) and scripts/new_adapter (generator) share one definition.
from jiuwensymbiosis.adapters._common.capability_spec import (
    CAPABILITY_DRIVER_MEMBERS,
    MIXIN_ABSTRACT_METHODS,
)

# Dedicated logger — configured in main() with a raw-message handler so the
# report keeps its visual layout, while third-party INFO logs (e.g. openjiuwen)
# are suppressed via the root logger level.
logger = logging.getLogger("validate_adapter")

# ``KNOWN_CAPABILITIES`` (E-04), ``MIXIN_ABSTRACT_METHODS`` (A-10) and
# ``CAPABILITY_DRIVER_MEMBERS`` (D-14) are all imported above from the package
# so this checker and the generator stay single-sourced.

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_Color = {"reset": 0, "red": 31, "green": 32, "yellow": 33, "cyan": 36, "bold": 1}

# Use ASCII-safe markers to avoid GBK/Windows console encoding issues
_CHECK_MARK = "[OK]"
_CROSS_MARK = "[XX]"


def _c(text: str, color: str) -> str:
    code = _Color.get(color, 0)
    return f"\033[{code}m{text}\033[0m"


def _safe(text: str) -> str:
    """Replace Unicode chars unsafe for GBK console."""
    return text.encode("gbk", errors="replace").decode("gbk")


def _resolve_module(module_str: str | None, path_str: str | None) -> str:
    if module_str:
        return module_str
    if path_str:
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


def _import_or_none(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def _find_class(module: Any, base_class: type) -> Optional[type]:
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if obj is base_class:
            continue
        try:
            if issubclass(obj, base_class) and obj.__module__ == module.__name__:
                return obj
        except TypeError:
            pass
    return None


def _resolve_module_path(module_str: str) -> str:
    """Convert module path to file system path or return the module_str itself."""
    try:
        spec = importlib.util.find_spec(module_str)
    except (ImportError, ValueError, AttributeError):
        return module_str
    if spec is not None and spec.origin is not None:
        return spec.origin
    return module_str


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

_SEVERITY_ERROR = "ERROR"
_SEVERITY_WARN = "WARN"
_SEVERITY_INFO = "INFO"

CheckResult = tuple[str, str, str]  # (code, severity, message)


def run_checks(module_str: str) -> list[CheckResult]:
    results: list[CheckResult] = []

    # --- Try to import the adapter module ---
    mod = _import_or_none(module_str)
    if mod is None:
        # Try importing submodules individually (main module may fail due to hardware deps)
        sub_results: list[CheckResult] = []
        for suffix in (".config", ".env", ".api", ".session"):
            sub = _import_or_none(module_str + suffix)
            if sub is not None:
                sub_results.append(("IMPORT", _SEVERITY_INFO, f"Sub-module {module_str}{suffix} imported successfully"))
        if sub_results:
            results.extend(sub_results)
            # Continue with partial findings
            mod = type("_DummyModule", (), {"__name__": module_str})()
        else:
            results.append(
                (
                    "IMPORT",
                    _SEVERITY_ERROR,
                    f"Unable to import module '{module_str}'. "
                    f"Check the path and ensure hardware dependencies (if any) are installed.",
                )
            )
            return results

    mod_path = _resolve_module_path(module_str)

    # --- Collect all classes from the module tree ---
    # Adapters are typically organized as packages with config.py, env.py, api.py, session.py
    # Each submodule is imported independently so that one failure doesn't block others
    all_modules = [mod] if mod is not None and not isinstance(mod, type) else []
    failed_subs: list[str] = []
    for suffix in (".config", ".env", ".api", ".session"):
        sub = _import_or_none(module_str + suffix)
        if sub is not None:
            all_modules.append(sub)
        else:
            failed_subs.append(suffix)

    if failed_subs and not all_modules:
        results.append(
            (
                "IMPORT",
                _SEVERITY_ERROR,
                f"Unable to import any submodule. Missing hardware dependencies? " f"Failed: {', '.join(failed_subs)}",
            )
        )
        return results

    if failed_subs:
        results.append(
            (
                "IMPORT",
                _SEVERITY_WARN,
                f"Some submodules failed to import (likely missing hardware deps): "
                f"{', '.join(failed_subs)}. Partial validation only.",
            )
        )

    def _all_classes(from_modules: list) -> list[tuple[Any, str]]:
        """Return (cls, module_name) for all classes defined in these modules.

        Only collects a class from its *defining* module (``obj.__module__ ==
        m.__name__``). ``seen`` is checked/marked *after* that filter so that a
        class re-exported by a package ``__init__`` is not "consumed" by the
        top module (where ``__module__`` does not match) and skipped at its
        real definition site — which would silently drop it from the scan.
        """
        found: list[tuple[Any, str]] = []
        seen: set[int] = set()
        for m in from_modules:
            for _, obj in inspect.getmembers(m, inspect.isclass):
                if obj.__module__ != m.__name__:
                    continue
                if id(obj) in seen:
                    continue
                seen.add(id(obj))
                found.append((obj, m.__name__))
        return found

    all_cls = _all_classes(all_modules)

    # --- Locate Config ---
    # Priority: classes with BOTH from_yaml AND from_dict > dataclasses
    cfg_cls = None
    for cls, _ in all_cls:
        has_yaml = hasattr(cls, "from_yaml") and callable(getattr(cls, "from_yaml"))
        has_dict = hasattr(cls, "from_dict") and callable(getattr(cls, "from_dict"))
        if has_yaml and has_dict:
            cfg_cls = cls
            break
    if cfg_cls is None:
        for cls, _ in all_cls:
            if dataclasses.is_dataclass(cls):
                cfg_cls = cls
                break

    # --- Locate Env ---
    import jiuwensymbiosis.env.base as _base_module

    BaseRobotEnv = _base_module.BaseRobotEnv
    env_cls = None
    for cls, _ in all_cls:
        try:
            if issubclass(cls, BaseRobotEnv) and cls is not BaseRobotEnv:
                env_cls = cls
                break
        except TypeError:
            pass

    # --- Locate Api ---
    import jiuwensymbiosis.api.base as _api_base_module

    BaseRobotApi = _api_base_module.BaseRobotApi
    api_cls = None
    for cls, _ in all_cls:
        try:
            if issubclass(cls, BaseRobotApi) and cls is not BaseRobotApi:
                api_cls = cls
                break
        except TypeError:
            pass

    # --- Locate session builder ---
    builder = None
    for name in ("build_",):
        for m in all_modules:
            obj = getattr(m, name, None)
            if obj is not None and callable(obj):
                builder = obj
                break
        if builder is not None:
            break
    if builder is None:
        for m in all_modules:
            for attr_name in dir(m):
                if attr_name.startswith("build_") and not attr_name.startswith("_"):
                    obj = getattr(m, attr_name)
                    if callable(obj):
                        builder = obj
                        break
            if builder is not None:
                break

    # ====================================================================
    # [C-01] Config class is a dataclass .......... ERROR
    # ====================================================================
    if cfg_cls is None:
        results.append(("C-01", _SEVERITY_ERROR, "未找到 Config 类 (需要 @dataclass 且有 from_yaml/from_dict)"))
    elif not dataclasses.is_dataclass(cfg_cls):
        results.append(
            ("C-01", _SEVERITY_ERROR, f"{cfg_cls.__name__} 不是 @dataclass，请添加 @dataclasses.dataclass 装饰器")
        )
    else:
        results.append(("C-01", _SEVERITY_INFO, f"[OK] Config 类 {cfg_cls.__name__} 已找到 (dataclass)"))

    # ====================================================================
    # [C-02] Config has from_yaml / from_dict ........ ERROR
    # ====================================================================
    if cfg_cls is not None:
        has_from_yaml = hasattr(cfg_cls, "from_yaml") and callable(getattr(cfg_cls, "from_yaml"))
        has_from_dict = hasattr(cfg_cls, "from_dict") and callable(getattr(cfg_cls, "from_dict"))
        if not has_from_yaml or not has_from_dict:
            missing = []
            if not has_from_yaml:
                missing.append("from_yaml")
            if not has_from_dict:
                missing.append("from_dict")
            results.append(
                ("C-02", _SEVERITY_ERROR, f"Config 缺少 classmethod: {', '.join(missing)} (make_builder 契约要求)")
            )
        else:
            results.append(("C-02", _SEVERITY_INFO, "[OK] Config 具有 from_yaml / from_dict classmethod"))

    # ====================================================================
    # [E-03] Env inherits BaseRobotEnv .............. ERROR
    # ====================================================================
    if env_cls is None:
        results.append(("E-03", _SEVERITY_ERROR, f"未找到继承 BaseRobotEnv 的 Env 类 (搜索模块: {module_str})"))
    else:
        results.append(("E-03", _SEVERITY_INFO, f"[OK] Env 类 {env_cls.__name__} 继承自 BaseRobotEnv"))

    # ====================================================================
    # [E-04] Env.capabilities ⊆ KNOWN_CAPABILITIES .. ERROR
    # ====================================================================
    if env_cls is not None:
        env_caps = getattr(env_cls, "capabilities", frozenset())
        unknown = set(env_caps) - KNOWN_CAPABILITIES
        if unknown:
            results.append(
                (
                    "E-04",
                    _SEVERITY_ERROR,
                    f"Env.capabilities 包含未知能力: {sorted(unknown)}。"
                    f"请先添加到 KNOWN_CAPABILITIES (env/base.py:35)",
                )
            )
        else:
            results.append(("E-04", _SEVERITY_INFO, f"[OK] Env.capabilities 全部合法: {sorted(env_caps)}"))

    # ====================================================================
    # [E-05] Env 覆写了 connect / disconnect / get_observation .... ERROR
    # ====================================================================
    if env_cls is not None:
        missing_abstract = []
        for method_name in ("connect", "disconnect", "get_observation"):
            if method_name not in env_cls.__dict__:
                # Check if it's in any base class that's still abstract
                m = getattr(env_cls, method_name, None)
                if m is None or getattr(m, "__isabstractmethod__", False):
                    missing_abstract.append(method_name)
        if missing_abstract:
            results.append(("E-05", _SEVERITY_ERROR, f"Env 未覆写抽象方法: {', '.join(missing_abstract)}"))
        else:
            results.append(("E-05", _SEVERITY_INFO, "[OK] Env 已覆写 connect / disconnect / get_observation"))

    # ====================================================================
    # [A-06] Api inherits BaseRobotApi .............. ERROR
    # ====================================================================
    if api_cls is None:
        results.append(("A-06", _SEVERITY_ERROR, f"未找到继承 BaseRobotApi 的 Api 类 (搜索模块: {module_str})"))
    else:
        results.append(("A-06", _SEVERITY_INFO, f"[OK] Api 类 {api_cls.__name__} 继承自 BaseRobotApi"))

    # ====================================================================
    # [A-07] Api 至少继承了一个 Mixin ................ WARN
    # ====================================================================
    if api_cls is not None:
        api_caps = _compute_api_capabilities(api_cls)
        if not api_caps:
            results.append(("A-07", _SEVERITY_WARN, "Api 未继承任何 Capability Mixin — 不会生成工具"))
        else:
            results.append(("A-07", _SEVERITY_INFO, f"[OK] Api Mixin capabilities: {sorted(api_caps)}"))

    # ====================================================================
    # [A-08] Api Mixin capabilities ⊆ Env.capabilities .... ERROR
    # ====================================================================
    if api_cls is not None and env_cls is not None:
        env_caps = set(getattr(env_cls, "capabilities", frozenset()))
        api_caps = _compute_api_capabilities(api_cls)
        missing_in_env = api_caps - env_caps
        if missing_in_env:
            results.append(
                (
                    "A-08",
                    _SEVERITY_ERROR,
                    f"Api Mixin 能力不在 Env.capabilities 中: {sorted(missing_in_env)}。"
                    f"运行时按 api∩env 门控，这些能力的工具不会暴露给 LLM",
                )
            )
        else:
            results.append(("A-08", _SEVERITY_INFO, "[OK] Api Mixin 能力 ⊆ Env.capabilities"))

    # ====================================================================
    # [A-09] Env 有无 Mixin 的标记能力 .............. INFO
    # ====================================================================
    if env_cls is not None and api_cls is not None:
        env_caps = set(getattr(env_cls, "capabilities", frozenset()))
        api_caps = _compute_api_capabilities(api_cls)
        marker_caps = env_caps - api_caps
        if marker_caps:
            results.append(
                (
                    "A-09",
                    _SEVERITY_INFO,
                    f"Env 声明了无 Mixin 的标记能力: {sorted(marker_caps)} " f"(正常现象，不影响运行)",
                )
            )
        else:
            results.append(("A-09", _SEVERITY_INFO, "[OK] 无标记能力"))

    # ====================================================================
    # [A-10] Mixin 抽象方法已在 Api 覆写 ............ WARN
    # ====================================================================
    if api_cls is not None:
        unmeth_overridden = _check_mixin_overrides(api_cls)
        if unmeth_overridden:
            items = [f"{m}.{meth}" for m, meth, _ in unmeth_overridden]
            results.append(
                (
                    "A-10",
                    _SEVERITY_WARN,
                    f"以下 Mixin 抽象方法未覆写: {', '.join(items)}。" f"调用时将抛 NotImplementedError",
                )
            )
        else:
            results.append(("A-10", _SEVERITY_INFO, "[OK] 所有 Mixin 抽象方法已覆写"))

    # ====================================================================
    # [S-11] Session builder 存在且可调用 ........... ERROR
    # ====================================================================
    if builder is None:
        results.append(
            ("S-11", _SEVERITY_ERROR, f"未找到 build_xxx_session 构建器 (搜索 {module_str} 和 {module_str}.session)")
        )
    elif not callable(builder):
        results.append(("S-11", _SEVERITY_ERROR, f"Session builder 存在但不可调用 "))
    else:
        results.append(("S-11", _SEVERITY_INFO, "[OK] Session builder 存在且可调用"))

    # ====================================================================
    # [S-12] Session builder 有 from_yaml ........... INFO
    # ====================================================================
    if builder is not None:
        has_yaml = hasattr(builder, "from_yaml")
        has_dict = hasattr(builder, "from_dict")
        if has_yaml and has_dict:
            results.append(("S-12", _SEVERITY_INFO, "[OK] Session builder 具有 from_yaml / from_dict 接口"))
        else:
            results.append(
                ("S-12", _SEVERITY_INFO, "Session builder 无 from_yaml / from_dict (make_builder 产物应自动具备)")
            )

    # ====================================================================
    # [A-13] @robot_tool capability 标签有效 ........ WARN
    # ====================================================================
    if api_cls is not None and env_cls is not None:
        env_caps = set(getattr(env_cls, "capabilities", frozenset()))
        tagged_warnings = _check_tool_tags(api_cls, env_caps)
        if tagged_warnings:
            for tw in tagged_warnings:
                results.append(("A-13", _SEVERITY_WARN, tw))
        else:
            results.append(("A-13", _SEVERITY_INFO, "[OK] 所有 @robot_tool capability 标签有效"))

    # ====================================================================
    # [D-14] 驱动类满足 capability 对应的 Protocol 子集 ...... ERROR
    # ====================================================================
    if env_cls is not None:
        env_caps = set(getattr(env_cls, "capabilities", frozenset()))
        driver_cls = _find_driver_class(module_str)
        if driver_cls is None:
            results.append(
                (
                    "D-14",
                    _SEVERITY_INFO,
                    "未找到 lowlevel 驱动类 (含 get_pose/move_to_pose_blocking)，" "跳过驱动接口校验",
                )
            )
        else:
            missing: list[str] = []
            for cap in sorted(env_caps):
                for member in CAPABILITY_DRIVER_MEMBERS.get(cap, []):
                    if not hasattr(driver_cls, member) and member not in missing:
                        missing.append(member)
            if missing:
                results.append(
                    (
                        "D-14",
                        _SEVERITY_ERROR,
                        f"驱动类 {driver_cls.__name__} 缺少 capability 所需方法: {missing}。"
                        f"Env/Api 委托调用时会 AttributeError (见 _common/protocol.py)",
                    )
                )
            else:
                results.append(
                    ("D-14", _SEVERITY_INFO, f"[OK] 驱动类 {driver_cls.__name__} 满足已声明 capability 的驱动接口")
                )

    # ====================================================================
    # [E-15] 声明 motion.cartesian 但未暴露 z_min_safe ...... INFO
    # ====================================================================
    if env_cls is not None:
        env_caps = set(getattr(env_cls, "capabilities", frozenset()))
        if "motion.cartesian" in env_caps:
            overrides_z = any("z_min_safe" in c.__dict__ for c in env_cls.__mro__ if c.__name__ != "BaseRobotEnv")
            if not overrides_z:
                results.append(
                    (
                        "E-15",
                        _SEVERITY_INFO,
                        "Env 声明 motion.cartesian 但未覆写 z_min_safe property — "
                        "SafetyRail 不会拦截 Z 下限 (建议从 config 暴露 z_min_safe)",
                    )
                )
            else:
                results.append(("E-15", _SEVERITY_INFO, "[OK] Env 暴露了 z_min_safe 安全下限"))

    return results


def _find_driver_class(module_str: str) -> Optional[type]:
    """Best-effort locate the low-level driver class in ``<module>.lowlevel``.

    Anchors on the structural signature ``get_pose`` + ``move_to_pose_blocking``
    (the RobotDriver core) to avoid picking helper classes. Returns None when
    no lowlevel module / candidate is found (D-14 then degrades to INFO).
    """
    ll_mod = _import_or_none(module_str + ".lowlevel")
    if ll_mod is None:
        return None
    for _, obj in inspect.getmembers(ll_mod, inspect.isclass):
        if getattr(obj, "__module__", None) != ll_mod.__name__:
            continue
        if hasattr(obj, "get_pose") and hasattr(obj, "move_to_pose_blocking"):
            return obj
    return None


# ---------------------------------------------------------------------------
# Check implementations
# ---------------------------------------------------------------------------


def _compute_api_capabilities(api_cls: type) -> set[str]:
    caps: set[str] = set()
    for cls in api_cls.__mro__:
        cap = getattr(cls, "capability", None)
        if isinstance(cap, str):
            caps.add(cap)
        elif isinstance(cap, (set, frozenset, list, tuple)):
            caps.update(cap)
    return caps


def _check_mixin_overrides(api_cls: type) -> list[tuple[str, str, str]]:
    """Check that Api overrides all abstract methods from its Mixins.
    Returns list of (mixin_name, method_name, reason).
    """
    results = []
    api_mro_names = {cls.__name__ for cls in api_cls.__mro__}

    for mixin_name, methods in MIXIN_ABSTRACT_METHODS.items():
        if mixin_name not in api_mro_names:
            continue
        # Find the Mixin class from the MRO
        mixin_cls = None
        for cls in api_cls.__mro__:
            if cls.__name__ == mixin_name:
                mixin_cls = cls
                break
        if mixin_cls is None:
            continue

        for meth_name in methods:
            api_meth = getattr(api_cls, meth_name, None)
            mixin_meth = getattr(mixin_cls, meth_name, None)
            if api_meth is None:
                results.append((mixin_name, meth_name, "方法缺失"))
                continue
            # If the method bodies are the same object, it's not overridden
            if api_meth is mixin_meth:
                results.append((mixin_name, meth_name, "未覆写（将抛 NotImplementedError）"))
    return results


def _check_tool_tags(api_cls: type, env_caps: set[str]) -> list[str]:
    """Check @robot_tool methods' capability tags against env capabilities."""
    warnings = []
    for attr_name in dir(api_cls):
        obj = getattr(api_cls, attr_name, None)
        if obj is None:
            continue
        meta = getattr(obj, "__robot_tool__", None)
        if meta is None:
            continue
        cap = getattr(meta, "capability", None)
        if cap is None:
            continue
        if isinstance(cap, str) and cap not in env_caps:
            warnings.append(f"Tool '{attr_name}' 的 capability '{cap}' 不在 Env.capabilities 中")
    return warnings


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def print_report(results: list[CheckResult], module_str: str, verbose: bool, errors_only: bool):
    error_count = sum(1 for _, s, _ in results if s == _SEVERITY_ERROR)
    warn_count = sum(1 for _, s, _ in results if s == _SEVERITY_WARN)
    info_count = sum(1 for _, s, _ in results if s == _SEVERITY_INFO)
    total = len(results)

    # Header
    logger.info(_c("=" * 65, "bold"))
    logger.info(_c(" jiuwensymbiosis 适配器兼容性验证", "bold"))
    logger.info(_c(f" 目标: {module_str}", "cyan"))
    logger.info(_c("=" * 65, "bold"))
    logger.info("")

    if results and results[0][0] == "IMPORT" and results[0][1] == _SEVERITY_ERROR:
        logger.error(_c(f"  [FAIL] {results[0][2]}", "red"))
        logger.info("")
        return

    # Print by severity
    printed_something = False

    if error_count > 0:
        logger.error(_c(f" ERROR ({error_count}) — 必须修复:", "red"))
        logger.info("")
        for code, sev, msg in results:
            if sev == _SEVERITY_ERROR:
                logger.error(f"  [{code}] {msg}")
                printed_something = True
        logger.info("")

    if warn_count > 0 and not errors_only:
        logger.warning(_c(f" WARN ({warn_count}) — 建议修复:", "yellow"))
        logger.info("")
        for code, sev, msg in results:
            if sev == _SEVERITY_WARN:
                logger.warning(f"  [{code}] {msg}")
                printed_something = True
        logger.info("")

    if info_count > 0 and verbose and not errors_only:
        logger.info(_c(f" INFO ({info_count}):", "cyan"))
        logger.info("")
        for code, sev, msg in results:
            if sev == _SEVERITY_INFO:
                logger.info(f"  [{code}] {msg}")
                printed_something = True
        logger.info("")

    if not printed_something:
        logger.info("  (无匹配的输出)")

    # Summary
    score = total - error_count
    pct = (score / total * 100) if total > 0 else 0
    logger.info(_c("=" * 65, "bold"))
    if error_count > 0:
        logger.error(_c(f" 结果: {error_count} ERROR, {warn_count} WARN — 兼容度 {pct:.0f}% ({score}/{total})", "red"))
        logger.error(_c(" 请修复所有 ERROR 后重试", "red"))
    elif warn_count > 0:
        logger.warning(_c(f" 结果: {warn_count} WARN — 兼容度 {pct:.0f}% ({score}/{total})", "yellow"))
        logger.warning(_c(" 基本兼容，建议修复 WARN 以完整体验所有功能", "yellow"))
    else:
        logger.info(_c(f" 结果: 全部通过 — 兼容度 100% ({score}/{total})", "green"))
    logger.info(_c("=" * 65, "bold"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _configure_logging() -> None:
    """Configure a dedicated handler for ``validate_adapter`` so the report
    keeps its raw-message layout, while third-party INFO logs (e.g. openjiuwen)
    are suppressed via the root logger level.

    The handler is bound to ``sys.stdout`` (not the StreamHandler default
    ``stderr``) so the report remains the tool's primary stdout output and
    shell redirection (``> report.txt``) keeps working.

    ``common`` / ``openjiuwen`` get a reject-all filter rather than just a
    level raise: those loggers install their own stdout handler and reset
    their level to INFO during import, so ``setLevel(WARNING)`` would be
    overridden. A filter persists on the logger object and rejects records
    at emit time regardless of the framework's own configuration.
    """
    logging.getLogger().setLevel(logging.WARNING)
    for noisy in ("common", "openjiuwen"):
        logging.getLogger(noisy).addFilter(lambda _record: False)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False


def main():
    _configure_logging()
    parser = argparse.ArgumentParser(
        description="jiuwensymbiosis 适配器兼容性验证工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--module",
        "-m",
        type=str,
        default=None,
        help="适配器 Python 模块路径 (如 jiuwensymbiosis.adapters.my_robot)",
    )
    parser.add_argument(
        "--path",
        "-p",
        type=str,
        default=None,
        help="适配器文件系统路径 (如 adapters/my_robot/)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="显示所有 INFO 级别检查结果",
    )
    parser.add_argument(
        "--errors-only",
        "-e",
        action="store_true",
        help="仅显示 ERROR 级别检查结果",
    )
    args = parser.parse_args()

    if not args.module and not args.path:
        parser.print_help()
        sys.exit(1)

    module_str = _resolve_module(args.module, args.path)
    if not module_str:
        logger.error(_c("错误: 无法解析模块路径", "red"))
        sys.exit(1)

    results = run_checks(module_str)
    print_report(results, module_str, args.verbose, args.errors_only)

    # Exit code by error count
    error_count = sum(1 for _, s, _ in results if s == _SEVERITY_ERROR)
    sys.exit(1 if error_count > 0 else 0)


if __name__ == "__main__":
    main()
