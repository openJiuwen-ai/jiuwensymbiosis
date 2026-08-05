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

# Heavyweight: each case generates a real adapter into the repo tree, validates
# and smoke-tests it as a subprocess (4-15s each, ~100s total). Marked as
# integration so the fast unit suite can select it out with -m "not integration".
pytestmark = pytest.mark.integration

# tests/integration/ is two levels below the repo root (``tests`` + this file's
# parent), so parents[2] reaches the repository root.
REPO_ROOT = Path(__file__).resolve().parents[2]

PRESETS = [
    Spec(name="gentest_scara", dof=4, end_effector="suction").normalized(),
    Spec(name="gentest_arm", dof=6, joint=True, end_effector="parallel").normalized(),
    Spec(name="gentest_vis", dof=6, end_effector="parallel", detection=True).normalized(),
]


def _flags(spec: Spec) -> list[str]:
    if spec.is_joint_ik:
        flags = [
            "--name",
            spec.name,
            "--motion-backend",
            "joint_ik",
            "--joint-count",
            str(spec.joint_count),
            "--end-effector",
            spec.end_effector,
            "--connection",
            spec.connection,
        ]
        if spec.detection:
            flags.append("--detection")
        elif spec.camera:
            flags.append("--camera")
        if spec.servo:
            flags.append("--servo")
        return flags
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


@pytest.fixture(scope="module")
def generated_presets():
    """Generate shared presets once for all read-only structural checks."""
    generated: dict[str, subprocess.CompletedProcess] = {}
    try:
        for spec in PRESETS:
            proc = _run_generator(spec)
            generated[spec.name] = proc
            if proc.returncode != 0:
                pytest.fail(f"generator failed for {spec.name}:\n{proc.stdout}\n{proc.stderr}")
        yield generated
    finally:
        for spec in PRESETS:
            shutil.rmtree(REPO_ROOT / "jiuwensymbiosis" / "adapters" / spec.name, ignore_errors=True)
            shutil.rmtree(REPO_ROOT / "configs" / spec.name, ignore_errors=True)


@pytest.mark.parametrize("spec", PRESETS, ids=lambda s: s.name)
def test_generated_adapter_passes_checks(spec, generated_presets):
    proc = generated_presets[spec.name]
    assert proc.returncode == 0

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
def test_capabilities_aligned_in_env(spec, generated_presets):
    assert generated_presets[spec.name].returncode == 0
    env_text = (REPO_ROOT / "jiuwensymbiosis" / "adapters" / spec.name / "env.py").read_text(encoding="utf-8")
    for cap in spec.capabilities:
        assert f'"{cap}"' in env_text, f"capability {cap} missing from env.py"


@pytest.mark.parametrize("spec", PRESETS, ids=lambda s: s.name)
def test_driver_methods_marked_pending(spec, generated_presets):
    assert generated_presets[spec.name].returncode == 0
    adapter_dir = REPO_ROOT / "jiuwensymbiosis" / "adapters" / spec.name
    pending = checks.scan_pending(adapter_dir)
    # The driver's lifecycle + motion methods are always generated as mocks.
    assert "lowlevel.py" in pending
    for method in ("connect", "disconnect", "get_pose", "home", "move_to_pose_blocking"):
        assert method in pending["lowlevel.py"], f"{method} not flagged pending"


def test_generated_adapter_is_black_clean(generated_presets):
    """The generator auto-formats its output, so black --check is a no-op.

    black is optional / best-effort (see ``checks.format_with_black``): skip
    cleanly when it is not installed instead of failing. ``python -m black``
    exits non-zero (not FileNotFoundError) when the module is absent, so detect
    it up front via ``importlib`` rather than trying to catch the subprocess.
    """
    if importlib.util.find_spec("black") is None:
        pytest.skip("black not installed")

    spec = PRESETS[1]
    assert generated_presets[spec.name].returncode == 0

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


def test_joint_limits_emitted_when_joint_enabled(cleanup):
    spec = Spec(name="gentest_jlim", dof=6, joint=True, end_effector="parallel").normalized()
    cleanup.append(spec.name)
    assert _run_generator(spec).returncode == 0

    adapter_dir = REPO_ROOT / "jiuwensymbiosis" / "adapters" / spec.name
    config_text = (adapter_dir / "config.py").read_text(encoding="utf-8")
    env_text = (adapter_dir / "env.py").read_text(encoding="utf-8")

    assert "joint_limits:" in config_text
    assert "isinstance(_raw, dict)" in config_text
    assert "len(_v) != 2" in config_text
    assert "def joint_limits(self)" in env_text
    assert 'getattr(self._cfg, "joint_limits", None)' in env_text
    assert "raise AttributeError" in env_text
    yaml_text = (REPO_ROOT / "configs" / spec.name / "default.yaml").read_text(encoding="utf-8")
    assert "# joint_limits:" in yaml_text
    assert "#   J1: [-360.0, 360.0]" in yaml_text

    module = f"jiuwensymbiosis.adapters.{spec.name}"
    assert checks.run_validate(module).ok, "validate failed"
    assert checks.run_smoke(module).ok, "smoke failed"


def test_no_joint_limits_when_joint_disabled(cleanup):
    """A non-joint adapter must NOT emit joint_limits (no move_joint tool)."""
    spec = Spec(name="gentest_nojlim", dof=4, end_effector="suction").normalized()
    cleanup.append(spec.name)
    assert _run_generator(spec).returncode == 0

    adapter_dir = REPO_ROOT / "jiuwensymbiosis" / "adapters" / spec.name
    config_text = (adapter_dir / "config.py").read_text(encoding="utf-8")
    env_text = (adapter_dir / "env.py").read_text(encoding="utf-8")
    yaml_text = (REPO_ROOT / "configs" / spec.name / "default.yaml").read_text(encoding="utf-8")
    assert "joint_limits" not in config_text
    assert "joint_limits" not in env_text
    assert "joint_limits" not in yaml_text


def test_generated_from_dict_handles_malformed_joint_limits(cleanup):
    spec = Spec(name="gentest_malformed", dof=6, joint=True, end_effector="parallel").normalized()
    cleanup.append(spec.name)
    assert _run_generator(spec).returncode == 0

    import importlib

    mod = importlib.import_module(f"jiuwensymbiosis.adapters.{spec.name}.config")
    cfg_cls = getattr(mod, f"{spec.prefix}Config")
    # Each of these must not raise:
    assert cfg_cls.from_dict({"joint_limits": [-360.0, 360.0]}).joint_limits is None  # list, not dict
    assert cfg_cls.from_dict({"joint_limits": {"J1": [1, 2, 3]}}).joint_limits is None  # 3 elements
    assert cfg_cls.from_dict({"joint_limits": {"J1": ["a", "b"]}}).joint_limits is None  # non-float
    # Mixed: keep good, drop bad.
    cfg = cfg_cls.from_dict({"joint_limits": {"J1": [-360.0, 360.0], "J2": "bad"}})
    assert cfg.joint_limits == {"J1": (-360.0, 360.0)}


JOINT_IK_PRESETS = [
    Spec(name="gjik_par", motion_backend="joint_ik", joint_count=5, end_effector="parallel").normalized(),
    Spec(name="gjik_suc", motion_backend="joint_ik", joint_count=6, end_effector="suction").normalized(),
    Spec(
        name="gjik_vis",
        motion_backend="joint_ik",
        joint_count=6,
        end_effector="parallel",
        detection=True,
        servo=True,
    ).normalized(),
]


@pytest.mark.parametrize("spec", JOINT_IK_PRESETS, ids=lambda s: s.name)
def test_joint_ik_adapter_passes_checks(spec, cleanup):
    """Every joint_ik combo (gripper / suction / vision+servo) validates and smokes."""
    cleanup.append(spec.name)

    proc = _run_generator(spec)
    assert proc.returncode == 0, f"generator failed:\n{proc.stdout}\n{proc.stderr}"

    adapter_dir = REPO_ROOT / "jiuwensymbiosis" / "adapters" / spec.name
    for fname in ("__init__.py", "config.py", "lowlevel.py", "env.py", "api.py", "session.py"):
        assert (adapter_dir / fname).is_file(), f"missing {fname}"

    module = f"jiuwensymbiosis.adapters.{spec.name}"
    assert checks.run_validate(module).ok, "validate failed"
    assert checks.run_smoke(module).ok, "smoke failed"


def test_joint_ik_seams_are_the_only_pending(cleanup):
    """The joint_ik skeleton flags exactly the transport + FK/IK seams as mocks."""
    spec = JOINT_IK_PRESETS[0]
    cleanup.append(spec.name)
    assert _run_generator(spec).returncode == 0

    adapter_dir = REPO_ROOT / "jiuwensymbiosis" / "adapters" / spec.name
    pending = checks.scan_pending(adapter_dir)
    assert set(pending) == {"lowlevel.py"}, f"only lowlevel seams should be pending, got {sorted(pending)}"
    for method in (
        "open",
        "close",
        "precheck",
        "read_arm_joints",
        "send_arm_joints",
        "read_effector",
        "send_effector",
        "forward_kinematics",
        "inverse_kinematics",
    ):
        assert method in pending["lowlevel.py"], f"{method} not flagged pending"


def test_joint_ik_output_is_generic(cleanup):
    """Generated joint_ik code must name no concrete SDK/body and reuse the core."""
    spec = JOINT_IK_PRESETS[2]  # the richest combo (vision + servo + gripper)
    cleanup.append(spec.name)
    assert _run_generator(spec).returncode == 0

    adapter_dir = REPO_ROOT / "jiuwensymbiosis" / "adapters" / spec.name
    blob = "\n".join(
        (adapter_dir / f).read_text(encoding="utf-8")
        for f in ("config.py", "lowlevel.py", "env.py", "api.py", "session.py")
    ).lower()
    for trace in ("lerobot", "sofollower", "robotkinematics", "so101", "so-101"):
        assert trace not in blob, f"generated code leaks a concrete SDK/body name: {trace}"

    lowlevel = (adapter_dir / "lowlevel.py").read_text(encoding="utf-8")
    assert "from jiuwensymbiosis.adapters._common.kinematic_driver import" in lowlevel
    assert "KinematicArmDriver" in lowlevel  # reuses the shared motion core


def test_joint_ik_composes_orthogonal_capabilities(cleanup):
    """Suction / vision / servo compose by reusing the existing capability machinery."""
    suction = JOINT_IK_PRESETS[1]
    vision = JOINT_IK_PRESETS[2]
    cleanup.extend([suction.name, vision.name])
    assert _run_generator(suction).returncode == 0
    assert _run_generator(vision).returncode == 0

    suc_dir = REPO_ROOT / "jiuwensymbiosis" / "adapters" / suction.name
    suc_api = (suc_dir / "api.py").read_text(encoding="utf-8")
    suc_env = (suc_dir / "env.py").read_text(encoding="utf-8")
    # Suction reuses SuctionMixin (no bespoke gripper override) + the effector seam.
    assert "SuctionMixin" in suc_api
    assert "open_gripper" not in suc_api
    assert '"grasp.suction"' in suc_env
    assert "send_effector" in (suc_dir / "lowlevel.py").read_text(encoding="utf-8")

    vis_dir = REPO_ROOT / "jiuwensymbiosis" / "adapters" / vision.name
    vis_api = (vis_dir / "api.py").read_text(encoding="utf-8")
    vis_env = (vis_dir / "env.py").read_text(encoding="utf-8")
    # Vision reuses VisionMixin + honest stubs; servo just declares the capability.
    assert "VisionMixin" in vis_api
    assert "not_implemented" in vis_api
    for cap in ('"vision.camera"', '"vision.detection"', '"motion.servo"'):
        assert cap in vis_env


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
