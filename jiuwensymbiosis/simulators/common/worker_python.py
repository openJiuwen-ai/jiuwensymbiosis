"""Resolve the renderer's interpreter without importing a physics SDK in the host."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .._paths import source_checkout
from .preview_backends import MUJOCO_VERSION, installation_extra, required_modules

MUJOCO_PYTHON_ENV = "JIUWEN_MUJOCO_PYTHON"

_PROBE = """
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
import sys

expected, *modules = sys.argv[1:]
try:
    import mujoco
except Exception as exc:
    print(f"无法导入 mujoco（{type(exc).__name__}）")
    sys.exit(1)

try:
    core = version("mujoco-core")
except PackageNotFoundError:
    core = "未安装"
if mujoco.__version__ != expected or core != expected:
    print(f"加载的 mujoco 为 {mujoco.__version__}，mujoco-core 为 {core}")
    sys.exit(1)

missing = [name for name in modules if find_spec(name) is None]
if missing:
    print(f"缺少 {', '.join(missing)}")
    sys.exit(1)
"""


def _probe_mujoco(python: str, modules: tuple[str, ...], environment: dict[str, str]) -> str | None:
    try:
        check = subprocess.run(
            [python, "-P", "-c", _PROBE, MUJOCO_VERSION, *modules],
            env=environment,
            capture_output=True,
            text=True,
            timeout=4,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "依赖检查超时"
    except OSError:
        return "无法运行此 Python"
    if check.returncode == 0:
        return None
    messages = check.stdout.strip().splitlines()
    return messages[-1] if messages else "无法检查 MuJoCo 依赖"


def mujoco_worker_python(mode: str, environment: dict[str, str]) -> str:
    override = environment.get(MUJOCO_PYTHON_ENV)
    if override is not None:
        configured = Path(override).expanduser()
        if not configured.is_absolute() or not configured.is_file():
            raise RuntimeError(f"{MUJOCO_PYTHON_ENV} 必须指向已安装 MuJoCo 的 Python 绝对路径。")
        candidates = [str(configured)]
    else:
        candidates = [sys.executable]
        checkout = source_checkout()
        if checkout is not None:
            fallback = checkout / ".venv-sim4/bin/python"
            # Resolving the venv symlink would run the base interpreter instead.
            if fallback.is_file() and str(fallback) != sys.executable:
                candidates.append(str(fallback))

    modules = required_modules("mujoco", mode)
    failures = []
    for python in candidates:
        error = _probe_mujoco(python, modules, environment)
        if error is None:
            return python
        failures.append(f"{python}（{error}）")
    extra = installation_extra("mujoco", mode)
    raise RuntimeError(
        f"MuJoCo 需要 mujoco-core {MUJOCO_VERSION}。已检查 {'、'.join(failures)}。"
        f"请在独立环境安装 .[{extra}]，或用 {MUJOCO_PYTHON_ENV} 指向该环境的 Python。"
    )
