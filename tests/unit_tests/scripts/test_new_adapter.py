# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the ``scripts/new_adapter`` generator.

Each preset is generated into the real ``jiuwensymbiosis/adapters/`` tree (the
import path is hard-coded there), checked, then removed. Generation, validate and
smoke all run as subprocesses with ``PYTHONPATH``/cwd pinned to this repo, so the
test never imports the generated module in-process and cleanup is a plain delete.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.new_adapter import checks
from scripts.new_adapter.spec import Spec

REPO_ROOT = Path(__file__).resolve().parents[3]

PRESETS = [
    Spec(name="gentest_scara", dof=4, end_effector="suction").normalized(),
    Spec(name="gentest_arm", dof=6, joint=True, end_effector="parallel").normalized(),
    Spec(name="gentest_vis", dof=6, end_effector="parallel", detection=True).normalized(),
]


def _flags(spec: Spec) -> list[str]:
    flags = [
        "--name",
        spec.name,
        "--dof",
        str(spec.dof),
        "--end-effector",
        spec.end_effector,
        "--tool",
        spec.tool_geometry,
        "--connection",
        spec.connection,
    ]
    if spec.joint:
        flags.append("--joint")
    if spec.detection:
        flags.append("--detection")
    elif spec.camera:
        flags.append("--camera")
    return flags


def _run_generator(spec: Spec) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.new_adapter.main",
            *_flags(spec),
            "--non-interactive",
            "--force",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def cleanup():
    """Remove any adapter/config dirs the test created, even on failure."""
    names: list[str] = []
    yield names
    for name in names:
        shutil.rmtree(REPO_ROOT / "jiuwensymbiosis" / "adapters" / name, ignore_errors=True)
        shutil.rmtree(REPO_ROOT / "configs" / name, ignore_errors=True)


@pytest.mark.parametrize("spec", PRESETS, ids=lambda s: s.name)
def test_generated_adapter_passes_checks(spec, cleanup):
    cleanup.append(spec.name)

    proc = _run_generator(spec)
    assert proc.returncode == 0, f"generator failed:\n{proc.stdout}\n{proc.stderr}"

    adapter_dir = REPO_ROOT / "jiuwensymbiosis" / "adapters" / spec.name
    for fname in ("__init__.py", "config.py", "lowlevel.py", "env.py", "api.py", "session.py"):
        assert (adapter_dir / fname).is_file(), f"missing {fname}"
    assert (REPO_ROOT / "configs" / spec.name / "default.yaml").is_file()

    module = f"jiuwensymbiosis.adapters.{spec.name}"

    # Static structural validation: zero ERROR.
    v = checks.run_validate(module)
    assert v.ok, f"validate failed:\n{v.detail}"

    # Runtime smoke (mock env connected): zero FAIL.
    s = checks.run_smoke(module)
    assert s.ok, f"smoke failed:\n{s.detail}"


@pytest.mark.parametrize("spec", PRESETS, ids=lambda s: s.name)
def test_capabilities_aligned_in_env(spec, cleanup):
    cleanup.append(spec.name)
    assert _run_generator(spec).returncode == 0

    env_text = (REPO_ROOT / "jiuwensymbiosis" / "adapters" / spec.name / "env.py").read_text(encoding="utf-8")
    for cap in spec.capabilities:
        assert f'"{cap}"' in env_text, f"capability {cap} missing from env.py"


@pytest.mark.parametrize("spec", PRESETS, ids=lambda s: s.name)
def test_driver_methods_marked_pending(spec, cleanup):
    cleanup.append(spec.name)
    assert _run_generator(spec).returncode == 0

    adapter_dir = REPO_ROOT / "jiuwensymbiosis" / "adapters" / spec.name
    pending = checks.scan_pending(adapter_dir)
    # The driver's lifecycle + motion methods are always generated as mocks.
    assert "lowlevel.py" in pending
    for method in ("connect", "disconnect", "get_pose", "home", "move_to_pose_blocking"):
        assert method in pending["lowlevel.py"], f"{method} not flagged pending"


def test_generated_adapter_is_black_clean(cleanup):
    """The generator auto-formats its output, so black --check is a no-op.

    black is optional / best-effort (see ``checks.format_with_black``): skip
    cleanly when it is not installed instead of failing. ``python -m black``
    exits non-zero (not FileNotFoundError) when the module is absent, so detect
    it up front via ``importlib`` rather than trying to catch the subprocess.
    """
    if importlib.util.find_spec("black") is None:
        pytest.skip("black not installed")

    spec = PRESETS[1]
    cleanup.append(spec.name)
    assert _run_generator(spec).returncode == 0

    adapter_dir = REPO_ROOT / "jiuwensymbiosis" / "adapters" / spec.name
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "black",
            "--line-length",
            "100",
            "--check",
            "--fast",
            str(adapter_dir),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"generated code not black-clean:\n{proc.stderr}"


def test_can_connection_config_flows_to_driver(cleanup):
    spec = Spec(name="gentest_can", dof=6, joint=True, end_effector="parallel").normalized()
    cleanup.append(spec.name)
    assert _run_generator(spec).returncode == 0

    adapter_dir = REPO_ROOT / "jiuwensymbiosis" / "adapters" / spec.name
    config_text = (adapter_dir / "config.py").read_text(encoding="utf-8")
    env_text = (adapter_dir / "env.py").read_text(encoding="utf-8")
    lowlevel_text = (adapter_dir / "lowlevel.py").read_text(encoding="utf-8")
    yaml_text = (REPO_ROOT / "configs" / spec.name / "default.yaml").read_text(encoding="utf-8")

    assert 'connection: str = "can"' in config_text
    assert "can_port" in config_text
    assert "can_bitrate" in config_text
    assert "can_port=cfg.can_port" in env_text
    assert "can_bitrate=cfg.can_bitrate" in env_text
    assert "tool_offset_mm=cfg.tool_offset_mm" in env_text
    assert "home_pose_xyzrxryrz_mm_deg=cfg.home_pose_xyzrxryrz_mm_deg" in env_text
    assert "offline/mock fallbacks only" in lowlevel_text
    assert "CAN reference shape" in lowlevel_text
    assert "from robot_sdk import RobotClient" in lowlevel_text
    assert "self._client = RobotClient" in lowlevel_text
    assert "def _open_can_client" not in lowlevel_text
    assert "from your_robot_sdk" not in lowlevel_text
    assert "CanRobotClient" not in lowlevel_text
    assert 'connection: "can"' in yaml_text


def test_non_can_connection_is_placeholder_but_valid(cleanup):
    spec = Spec(name="gentest_tcp", dof=6, end_effector="none", connection="tcp").normalized()
    cleanup.append(spec.name)

    proc = _run_generator(spec)
    assert proc.returncode == 0, f"generator failed:\n{proc.stdout}\n{proc.stderr}"
    assert "tcp 当前先生成空连接模板，后续会实现更完整模板" in proc.stdout

    adapter_dir = REPO_ROOT / "jiuwensymbiosis" / "adapters" / spec.name
    config_text = (adapter_dir / "config.py").read_text(encoding="utf-8")
    env_text = (adapter_dir / "env.py").read_text(encoding="utf-8")
    lowlevel_text = (adapter_dir / "lowlevel.py").read_text(encoding="utf-8")

    assert 'connection: str = "tcp"' in config_text
    assert "host" in config_text
    assert "port" in config_text
    assert "connection_note" in config_text
    assert "host=cfg.host" in env_text
    assert "port=cfg.port" in env_text
    assert "tcp 模板当前是占位版本" in lowlevel_text

    module = f"jiuwensymbiosis.adapters.{spec.name}"
    assert checks.run_validate(module).ok
    assert checks.run_smoke(module).ok
