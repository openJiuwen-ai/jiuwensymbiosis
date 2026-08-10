# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Render adapter source files from a :class:`Spec`.

The generator writes runnable mock adapters first. Each mock body carries
``SENTINEL`` so the guided script can tell the user exactly which SDK-specific
methods are still pending.
"""

from __future__ import annotations

from textwrap import dedent

from jiuwensymbiosis.adapters._common.capability_spec import CAPABILITY_MIXIN

from .spec import Spec

HEADER = "# coding: utf-8\n# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.\n"

# Tags a generated mock body. ``checks.py`` greps for it to know which driver
# methods are still mocks; the user deletes the line once a method is real.
SENTINEL = "# >>> GENERATED-MOCK: replace with real hardware <<<"

_TOOL_DOWN_RX = 180.0
_TOOL_DOWN_RY = 30.0


# ---------------------------------------------------------------------------
# Source helpers
# ---------------------------------------------------------------------------


def _clean(text: str) -> str:
    return dedent(text).strip("\n")


def _render(template: str, **parts: str) -> str:
    """Dedent a template, replace explicit markers, and add the project header."""
    text = _clean(template)
    for name, value in parts.items():
        text = text.replace(f"__{name}__", value.rstrip("\n"))
    return HEADER + "\n" + text.rstrip() + "\n"


def _block(lines: list[str], spaces: int = 0) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + line if line else "" for line in lines)


def _indent(text: str, spaces: int) -> str:
    return _block(_clean(text).splitlines(), spaces)


def _by_connection(mapping: dict[str, list[str]], connection: str) -> list[str]:
    """Look up a per-connection block, failing loudly on an unknown value."""
    try:
        return mapping[connection]
    except KeyError:
        raise ValueError(f"unsupported connection: {connection}") from None


def _default_pose_pairs(spec: Spec) -> list[tuple[str, float]]:
    xyz = [("x", 200.0), ("y", 0.0), ("z", 250.0)]
    rotation = [("r", 0.0)] if spec.dof == 4 else [("rx", 0.0), ("ry", 90.0), ("rz", 0.0)]
    return xyz + rotation


def _dict_literal(pairs: list[tuple[str, str]]) -> str:
    return "{" + ", ".join(f'"{key}": {value}' for key, value in pairs) + "}"


def _mixin_names(spec: Spec) -> list[str]:
    names = [CAPABILITY_MIXIN[cap] for cap in spec.capabilities if CAPABILITY_MIXIN.get(cap)]
    return list(dict.fromkeys(names))


def _effective_tilted(spec: Spec) -> bool:
    # SCARA (4-DoF) has no tilt axis; only a 6-DoF tool can be mounted tilted.
    return spec.tool_geometry == "tilted" and spec.dof == 6


def _connection_future_note(spec: Spec) -> str:
    if spec.connection == "custom":
        return "custom 连接方式会生成最空模板，请按硬件 SDK 完全填充。"
    if spec.connection == "can":
        return "CAN 连接会生成较完整模板。"
    return f"{spec.connection} 当前先生成空连接模板，后续会实现更完整模板。"


def _pose_keys_literal(spec: Spec) -> str:
    return "[" + ", ".join(repr(k) for k in spec.pose_fields) + "]"


# ---------------------------------------------------------------------------
# Connection-specific fragments
# ---------------------------------------------------------------------------


def _connection_config_fields(spec: Spec) -> str:
    fields = {
        "can": [
            'can_port: str = "can0"              # CAN 网卡名，如 can0 / can_left',
            "can_bitrate: int = 1_000_000       # CAN 波特率，仅记录/提示用",
            "# TODO: 把 can_port/can_bitrate 换成真实硬件参数",
        ],
        "serial": [
            'serial_port: str = "/dev/ttyUSB0"  # 串口设备名（空模板，后续会实现完整模板）',
            "baudrate: int = 115200",
            'connection_note: str = "serial template placeholder"',
        ],
        "tcp": [
            'host: str = "192.168.1.10"         # 控制器 IP（空模板，后续会实现完整模板）',
            "port: int = 3000",
            'connection_note: str = "tcp template placeholder"',
        ],
        "usb": [
            "device_serial: Optional[str] = None # USB 设备序列号（空模板，后续会实现完整模板）",
            'connection_note: str = "usb template placeholder"',
        ],
        "ros": [
            'ros_namespace: str = ""            # ROS/ROS2 命名空间（空模板，后续会实现完整模板）',
            'command_topic: str = "/robot/command"',
            'connection_note: str = "ros template placeholder"',
        ],
        "custom": ['connection_note: str = "custom connection: fill hardware SDK fields here"'],
    }
    return _block(_by_connection(fields, spec.connection), 4)


def _driver_params(spec: Spec) -> str:
    params = {
        "can": [
            "# Defaults below are offline/mock fallbacks only.",
            "# Change real hardware values in configs/<adapter>/default.yaml.",
            'can_port: str = "can0",',
            "can_bitrate: int = 1_000_000,",
        ],
        "serial": [
            'serial_port: str = "/dev/ttyUSB0",',
            "baudrate: int = 115200,",
            'connection_note: str = "serial template placeholder",',
        ],
        "tcp": [
            'host: str = "192.168.1.10",',
            "port: int = 3000,",
            'connection_note: str = "tcp template placeholder",',
        ],
        "usb": [
            "device_serial: Optional[str] = None,",
            'connection_note: str = "usb template placeholder",',
        ],
        "ros": [
            'ros_namespace: str = "",',
            'command_topic: str = "/robot/command",',
            'connection_note: str = "ros template placeholder",',
        ],
        "custom": ['connection_note: str = "custom connection: fill hardware SDK fields here",'],
    }
    common = [
        "move_speed: int = 50,",
        "tool_offset_mm: float = 0.0,",
        "home_pose_xyzrxryrz_mm_deg: Optional[list[float]] = None,",
    ]
    gripper = []
    if spec.end_effector == "parallel":
        gripper = [
            "gripper_open_mm: float = 70.0,",
            "gripper_effort: int = 1000,",
        ]
    return _block(_by_connection(params, spec.connection) + common + gripper, 8)


def _driver_assignments(spec: Spec) -> str:
    assignments = {
        "can": [
            "self.can_port = can_port",
            "self.can_bitrate = int(can_bitrate)",
        ],
        "serial": [
            "self.serial_port = serial_port",
            "self.baudrate = int(baudrate)",
            "self.connection_note = connection_note",
        ],
        "tcp": [
            "self.host = host",
            "self.port = int(port)",
            "self.connection_note = connection_note",
        ],
        "usb": [
            "self.device_serial = device_serial",
            "self.connection_note = connection_note",
        ],
        "ros": [
            "self.ros_namespace = ros_namespace",
            "self.command_topic = command_topic",
            "self.connection_note = connection_note",
        ],
        "custom": ["self.connection_note = connection_note"],
    }
    return _block(_by_connection(assignments, spec.connection), 8)


def _driver_kwargs(spec: Spec) -> str:
    kwargs = {
        "can": ["can_port=cfg.can_port,", "can_bitrate=cfg.can_bitrate,"],
        "serial": [
            "serial_port=cfg.serial_port,",
            "baudrate=cfg.baudrate,",
            "connection_note=cfg.connection_note,",
        ],
        "tcp": [
            "host=cfg.host,",
            "port=cfg.port,",
            "connection_note=cfg.connection_note,",
        ],
        "usb": [
            "device_serial=cfg.device_serial,",
            "connection_note=cfg.connection_note,",
        ],
        "ros": [
            "ros_namespace=cfg.ros_namespace,",
            "command_topic=cfg.command_topic,",
            "connection_note=cfg.connection_note,",
        ],
        "custom": ["connection_note=cfg.connection_note,"],
    }
    common = [
        "move_speed=cfg.move_speed,",
        "tool_offset_mm=cfg.tool_offset_mm,",
        "home_pose_xyzrxryrz_mm_deg=cfg.home_pose_xyzrxryrz_mm_deg,",
    ]
    gripper = []
    if spec.end_effector == "parallel":
        gripper = [
            "gripper_open_mm=cfg.gripper_open_mm,",
            "gripper_effort=cfg.gripper_effort,",
        ]
    return _block(_by_connection(kwargs, spec.connection) + common + gripper, 12)


def _connect_docstring(spec: Spec) -> str:
    if spec.connection != "can":
        return '        """Open hardware connection. Must be idempotent."""'
    return _indent(
        """
        \"\"\"Open hardware connection. Must be idempotent.

        CAN reference shape (replace this method body with your SDK calls)::

            from robot_sdk import RobotClient
            self._client = RobotClient(channel=self.can_port, bitrate=self.can_bitrate)
            self._client.connect()
            self._client.enable()
            self._connected = True

        Real values come from configs/<adapter>/default.yaml.
        \"\"\"
        """,
        8,
    )


def _connect_note(spec: Spec) -> str:
    if spec.connection == "can":
        lines = ["# Keep this mock line until the body above has been replaced."]
    elif spec.connection == "custom":
        lines = ["# custom 模板：在这里创建硬件 SDK client、打开连接、使能机械臂。"]
    else:
        lines = [
            f"# {spec.connection} 模板当前是占位版本，后续会提供更完整的生成模板。",
            "# 现在请在这里创建硬件 SDK client、打开连接、使能机械臂。",
        ]
    return _block(lines, 8)


# ---------------------------------------------------------------------------
# config.py
# ---------------------------------------------------------------------------


# Camera / detection config blocks + from_dict/from_yaml fixes are shared with
# the joint_ik backend (vision composes with either motion backend), so they live
# in one place. The strings are unchanged, so Family-1 output stays identical.
def _camera_config_fields() -> str:
    return _indent(
        """
        # ==================== 相机 [选填-仅 vision.*] ====================
        camera_serial: Optional[str] = None # 相机序列号 (None=禁用)
        camera_resolution: tuple[int, int] = (640, 480)
        camera_fps: int = 30
        """,
        4,
    )


def _detection_config_fields() -> str:
    return _indent(
        """
        # ============== 检测校正 [选填-仅 vision.detection] ==============
        z_correction_mm: float = 0.0        # Z 向常值校正
        grasp_z_offset_mm: float = -25.0    # 抓取点相对物体顶面偏移
        place_z_offset_mm: float = 75.0     # 堆叠放置偏移
        detector_url: str = "http://127.0.0.1:8114"  # 检测服务地址
        calib_path: Optional[str] = None    # 手眼标定文件 (JSON)
        """,
        4,
    )


def _camera_resolution_fix() -> str:
    return _indent(
        """
        if "camera_resolution" in clean and isinstance(clean["camera_resolution"], list):
            clean["camera_resolution"] = tuple(clean["camera_resolution"])
        """,
        8,
    )


def _calib_path_fix() -> str:
    return _indent(
        """
        if cfg.calib_path and not Path(cfg.calib_path).is_absolute():
            candidate = (path.parent / cfg.calib_path).resolve()
            if candidate.exists():
                cfg.calib_path = str(candidate)
        """,
        8,
    )


def _config_optional_fields(spec: Spec) -> str:
    blocks: list[str] = []
    if spec.joint:
        blocks.append(
            _indent(
                """
                # ============== 关节软限位 [选填-仅 motion.joint] ==============
                # 单位须与 move_joint(q) 一致。
                joint_limits: Optional[dict[str, tuple[float, float]]] = None
                """,
                4,
            )
        )
    if spec.end_effector == "parallel":
        blocks.append(
            _indent(
                """
                # ==================== 夹爪 [选填-仅 grasp.parallel] ====================
                gripper_open_mm: float = 70.0       # 打开宽度 (mm)
                gripper_effort: int = 1000          # 夹持力 (驱动单位)
                """,
                4,
            )
        )
    if spec.has_camera:
        blocks.append(_camera_config_fields())
    if spec.detection:
        blocks.append(_detection_config_fields())
    return "\n\n".join(blocks)


def render_config(spec: Spec) -> str:
    home_default = ", ".join(str(value) for _, value in _default_pose_pairs(spec))
    camera_resolution_fix = _camera_resolution_fix() if spec.has_camera else ""
    joint_limits_fix = ""
    if spec.joint:
        joint_limits_fix = _indent(
            """
            if "joint_limits" in clean:
                _raw = clean["joint_limits"]
                if not isinstance(_raw, dict):
                    clean["joint_limits"] = None
                else:
                    _norm: dict[str, tuple[float, float]] = {}
                    for _k, _v in _raw.items():
                        if not isinstance(_v, (list, tuple)) or len(_v) != 2:
                            continue
                        try:
                            _norm[str(_k)] = (float(_v[0]), float(_v[1]))
                        except (TypeError, ValueError):
                            continue
                    clean["joint_limits"] = _norm if _norm else None
            """,
            8,
        )
    calib_path_fix = _calib_path_fix() if spec.detection else ""

    return _render(
        f'''
        """{spec.config_cls} — hardware configuration dataclass.

        Fields are annotated [必填]/[选填]/[选填-仅 <capability>].
        Load with ``from_yaml(path)`` or construct with keyword arguments.
        """

        from __future__ import annotations

        import dataclasses
        from dataclasses import dataclass, field
        from pathlib import Path
        from typing import Any, Optional

        import yaml


        @dataclass
        class {spec.config_cls}:
            """Hardware configuration for the {spec.name} robot."""

            # ==================== 基本信息 [必填] ====================
            name: str = "{spec.name}"

            # ==================== 硬件连接 [必填] ====================
            connection: str = "{spec.connection}"
        __CONNECTION_FIELDS__
            move_speed: int = 50                # [选填] 运动速度百分比 (0-100)

            # ==================== 运动学 [选填] ====================
            tool_offset_mm: float = 0.0         # 法兰 → 工具末端 Z 向偏移 (mm)
            home_pose_xyzrxryrz_mm_deg: list[float] = field(
                default_factory=lambda: [{home_default}]
            )
            home_use_init_pose: bool = False    # [选填] 用当前位置作 home

            # ==================== 安全边界 [选填] ====================
            z_min_safe_mm: float = 50.0         # Z 向安全下限 (SafetyRail 读取)
            x_min_mm: Optional[float] = 0.0     # X 工作空间下界 (None=不限制)
            x_max_mm: Optional[float] = 700.0
            y_min_mm: Optional[float] = -500.0
            y_max_mm: Optional[float] = 500.0
            z_max_mm: Optional[float] = 800.0

        __OPTIONAL_FIELDS__

            # ==================== Loaders — 勿改 (框架契约) ====================
            @classmethod
            def from_dict(cls, data: dict[str, Any]) -> "{spec.config_cls}":
                """Construct from a flat dict; unknown keys are ignored."""
                valid = {{f.name for f in dataclasses.fields(cls)}}
                clean: dict[str, Any] = {{k: v for k, v in data.items() if k in valid}}
        __CAMERA_RESOLUTION_FIX__
        __JOINT_LIMITS_FIX__
                return cls(**clean)

            @classmethod
            def from_yaml(cls, path: str | Path) -> "{spec.config_cls}":
                """Load config from a YAML file."""
                path = Path(path).resolve()
                with path.open("r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {{}}
                cfg = cls.from_dict(data)
        __CALIB_PATH_FIX__
                return cfg
        ''',
        CONNECTION_FIELDS=_connection_config_fields(spec),
        OPTIONAL_FIELDS=_config_optional_fields(spec),
        CAMERA_RESOLUTION_FIX=camera_resolution_fix,
        JOINT_LIMITS_FIX=joint_limits_fix,
        CALIB_PATH_FIX=calib_path_fix,
    )


# ---------------------------------------------------------------------------
# lowlevel.py
# ---------------------------------------------------------------------------


def _lowlevel_joint_block(spec: Spec) -> str:
    if not spec.joint:
        return ""
    return _indent(
        f"""
        # ----------------------- 关节 [选填-仅 motion.joint] -----------------------
        def move_joint_blocking(self, q: list[float]) -> None:
            \"\"\"Blocking joint-space move.

            Reference shape::

                self._client.move_joints(q, speed=self.move_speed)
                self._client.wait_until_idle()

            Use the angle convention your adapter documents, commonly degrees.
            \"\"\"
            {SENTINEL}
            return None
        """,
        4,
    )


def _lowlevel_effector_block(spec: Spec) -> str:
    if spec.end_effector == "suction":
        return _indent(
            f"""
            # --------------------- 末端 [选填-仅 grasp.suction] ---------------------
            def set_suction(self, on: bool) -> None:
                \"\"\"on=True activates suction; on=False releases.

                Reference shape::

                    self._client.set_suction(bool(on))
                \"\"\"
                {SENTINEL}
                return None
            """,
            4,
        )
    if spec.end_effector == "parallel":
        return _indent(
            f"""
            # --------------------- 末端 [选填-仅 grasp.parallel] ---------------------
            def set_gripper(self, on: bool) -> None:
                \"\"\"on=True closes/grips; on=False opens/releases.

                Reference shape::

                    width = 0.0 if on else self._gripper_open_mm
                    self._client.set_gripper(width_mm=width)
                \"\"\"
                {SENTINEL}
                return None
            """,
            4,
        )
    return ""


def _lowlevel_camera_block(spec: Spec) -> str:
    if not spec.has_camera:
        return ""
    return _indent(
        f"""
        # ----------------------- 传感器 [选填-仅 vision.*] -----------------------
        def grab_frames(self) -> Optional[tuple]:
            \"\"\"Grab one (rgb HxWx3 uint8, depth HxW float32 m) pair, or None.

            Reference shape::

                rgb, depth_m = self._camera.read_rgbd()
                return rgb, depth_m

            Depth must be meters. Return None when the camera is unavailable.
            \"\"\"
            {SENTINEL}
            return None
        """,
        4,
    )


def _lowlevel_detection_block(spec: Spec) -> str:
    if not spec.detection:
        return ""
    return _indent(
        """
        # ---------------- 视觉标定 [选填-仅 vision.detection] ----------------
        @property
        def tf_flange_cam(self) -> Optional[Any]:
            \"\"\"4x4 hand-eye transform (camera → flange).\"\"\"
            return self._tf_flange_cam

        @property
        def calibration(self) -> Optional[dict]:
            \"\"\"Raw calibration dict loaded from calib_path.\"\"\"
            return self._calibration

        @property
        def intrinsics(self) -> Optional[Any]:
            \"\"\"3x3 camera intrinsics matrix.\"\"\"
            return self._intrinsics
        """,
        4,
    )


def _join_optional_blocks(*blocks: str) -> str:
    return "\n\n".join(block for block in blocks if block)


def render_lowlevel(spec: Spec) -> str:
    pairs = _default_pose_pairs(spec)
    init_pose = _dict_literal([(key, repr(value)) for key, value in pairs])
    get_pose_ref = (
        "return SimpleNamespace(x=raw.x, y=raw.y, z=raw.z, r=raw.r)"
        if spec.dof == 4
        else "return SimpleNamespace(x=raw.x, y=raw.y, z=raw.z, rx=raw.rx, ry=raw.ry, rz=raw.rz)"
    )
    move_ref = (
        "self._client.move_linear(x=pose.x, y=pose.y, z=pose.z, r=pose.r, speed=self.move_speed)"
        if spec.dof == 4
        else (
            "self._client.move_linear(x=pose.x, y=pose.y, z=pose.z, "
            "rx=pose.rx, ry=pose.ry, rz=pose.rz, speed=self.move_speed)"
        )
    )
    home_update = _block(
        [f'self._pose["{key}"] = float(getattr(hp, "{key}", {default!r}))' for key, default in pairs],
        8,
    )
    pose_update = _block(
        [f'self._pose["{key}"] = float(getattr(pose, "{key}", {default!r}))' for key, default in pairs],
        8,
    )
    gripper_attrs = ""
    if spec.end_effector == "parallel":
        gripper_attrs = _block(
            [
                "self._gripper_open_mm = float(gripper_open_mm)",
                "self._gripper_effort = int(gripper_effort)",
            ],
            8,
        )
    calibration_attrs = ""
    if spec.detection:
        calibration_attrs = _block(
            [
                "self._tf_flange_cam: Optional[Any] = None  # 4x4 相机→法兰 (手眼标定)",
                "self._calibration: Optional[dict] = None",
                "self._intrinsics: Optional[Any] = None     # 3x3 相机内参",
            ],
            8,
        )

    return _render(
        f'''
        """{spec.driver_cls} — low-level hardware communication.

        Replace each mock method body (marked with the sentinel comment) with
        your real SDK calls (serial / CAN / socket). A plain class satisfying the
        RobotDriver Protocol (adapters/_common/protocol.py) — Env verbs delegate here.
        """

        from __future__ import annotations

        from types import SimpleNamespace
        from typing import Any, Optional


        class {spec.driver_cls}:
            """Hardware driver — mock tracks pose in memory until you wire real I/O."""

            def __init__(
                self,
        __DRIVER_PARAMS__
            ) -> None:
                self.connection: str = "{spec.connection}"
        __DRIVER_ASSIGNMENTS__
                self.move_speed = int(move_speed)
                self._pose: dict[str, float] = {init_pose}
                pose_keys = {_pose_keys_literal(spec)}
                if home_pose_xyzrxryrz_mm_deg:
                    for key, value in zip(pose_keys, home_pose_xyzrxryrz_mm_deg):
                        self._pose[key] = float(value)
                self.home_pose = SimpleNamespace(**self._pose)
                self.tool_offset_mm: float = float(tool_offset_mm)
        __GRIPPER_ATTRS__
                self._client: Optional[Any] = None
                self._connected: bool = False
        __CALIBRATION_ATTRS__

            # ----------------------------- 生命周期 [必填] -----------------------------
            def connect(self) -> None:
        __CONNECT_DOCSTRING__
                if self._connected:
                    return
                {SENTINEL}
        __CONNECT_NOTE__
                self._connected = True

            def disconnect(self) -> None:
                """Release hardware. Idempotent and safe at any state.

                Reference shape (replace SDK-specific lines as needed)::

                    if self._client is not None:
                        self._client.disconnect()  # or close() / shutdown()
                    self._client = None
                    self._connected = False
                """
                if not self._connected:
                    return
                {SENTINEL}
                if self._client is not None:
                    close = getattr(self._client, "close", None) or getattr(self._client, "disconnect", None)
                    if callable(close):
                        close()
                self._client = None
                self._connected = False

            # ----------------------------- 运动 [必填] -----------------------------
            def get_pose(self) -> Any:
                """Return current FLANGE pose (mm/deg).

                Reference shape::

                    raw = self._client.get_pose()
                    {get_pose_ref}

                If the SDK returns meters/radians or joint-frame values, convert here.
                """
                {SENTINEL}
                return SimpleNamespace(**self._pose)

            def home(self) -> None:
                """Execute homing sequence (blocking).

                Reference shape::

                    self._client.home()
                    self._client.wait_until_idle()

                If there is no dedicated home command, call move_to_pose_blocking(self.home_pose).
                """
                {SENTINEL}
                hp = self.home_pose
        __HOME_UPDATE__

            def move_to_pose_blocking(self, pose: Any) -> None:
                """Blocking Cartesian move to pose in FLANGE frame (mm/deg).

                Reference shape::

                    {move_ref}
                    self._client.wait_until_idle()

                Keep units at the framework boundary as mm/deg.
                """
                {SENTINEL}
        __POSE_UPDATE__

        __OPTIONAL_METHODS__
        ''',
        DRIVER_PARAMS=_driver_params(spec),
        DRIVER_ASSIGNMENTS=_driver_assignments(spec),
        GRIPPER_ATTRS=gripper_attrs,
        CALIBRATION_ATTRS=calibration_attrs,
        CONNECT_DOCSTRING=_connect_docstring(spec),
        CONNECT_NOTE=_connect_note(spec),
        HOME_UPDATE=home_update,
        POSE_UPDATE=pose_update,
        OPTIONAL_METHODS=_join_optional_blocks(
            _lowlevel_joint_block(spec),
            _lowlevel_effector_block(spec),
            _lowlevel_camera_block(spec),
            _lowlevel_detection_block(spec),
        ),
    )


# ---------------------------------------------------------------------------
# env.py
# ---------------------------------------------------------------------------


def _env_joint_limits_prop(spec: Spec) -> str:
    if not spec.joint:
        return ""
    return _indent(
        """
        @property
        def joint_limits(self) -> Optional[dict[str, tuple[float, float]]]:
            \"\"\"Joint soft limits from config, or None when unconfigured.\"\"\"
            return getattr(self._cfg, "joint_limits", None)

        @joint_limits.setter
        def joint_limits(self, _: Optional[dict[str, tuple[float, float]]]) -> None:
            raise AttributeError(f"{type(self).__name__}.joint_limits is read-only (read from config)")
        """,
        8,
    )


def render_env(spec: Spec) -> str:
    caps = _block([f'"{cap}",' for cap in spec.capabilities], 12)
    pose_items = ", ".join(f'"{field}": getattr(p, "{field}", 0.0)' for field in spec.pose_fields)
    camera_observation = ""
    if spec.has_camera:
        camera_observation = _indent(
            """
            if "vision.camera" in self.capabilities:
                try:
                    frames = ll.grab_frames()
                    if frames is not None:
                        rgb, depth = frames
                except Exception:
                    pass
            """,
            8,
        )

    return _render(
        f'''
        """{spec.env_cls} — hardware abstraction wrapping {spec.driver_cls}.

        connect() creates self.low_level; Env verbs (home / move_to_flange / ...)
        delegate to it. See docs/hardware-porting-guide.md Step 3.
        """

        from __future__ import annotations

        from typing import Optional

        from jiuwensymbiosis.env.base import BaseRobotEnv, RobotObservation
        from jiuwensymbiosis.adapters.{spec.name}.lowlevel import {spec.driver_cls}


        class {spec.env_cls}(BaseRobotEnv):
            """Hardware environment for the {spec.name} robot."""

            capabilities = frozenset(
                {{
        __CAPABILITIES__
                }}
            )
            name: str = "{spec.name}"

            def __init__(self, cfg) -> None:
                self._cfg = cfg
                self.low_level: Optional[{spec.driver_cls}] = None

            # ----------------------------------------------------- lifecycle
            def connect(self) -> None:
                """Open hardware connection. Must be idempotent."""
                if self.low_level is not None:
                    return
                cfg = self._cfg
                self.low_level = {spec.driver_cls}(
        __DRIVER_KWARGS__
                )
                self.low_level.connect()

            def disconnect(self) -> None:
                """Release hardware. Idempotent and safe at any state."""
                if self.low_level is None:
                    return
                try:
                    self.low_level.disconnect()
                finally:
                    self.low_level = None

            # ---------------------------------------------------- observation
            def get_observation(self) -> RobotObservation:
                """Best-effort snapshot. Should not raise on transient gaps."""
                ll = self.low_level
                if ll is None:
                    return RobotObservation()
                try:
                    p = ll.get_pose()
                    pose = {{{pose_items}}}
                except Exception:
                    pose = None
                rgb = None
                depth = None
        __CAMERA_OBSERVATION__
                return RobotObservation(pose=pose, rgb=rgb, depth=depth)

            # ----------------------------------------------- safety boundaries
            @property
            def z_min_safe(self) -> float:
                """Z floor (mm) — SafetyRail reads this automatically."""
                return float(self._cfg.z_min_safe_mm)

            @property
            def workspace_bounds(self) -> Optional[tuple]:
                """XY workspace bounds (xmin,ymin,xmax,ymax) or None."""
                cfg = self._cfg
                if cfg.x_min_mm is not None:
                    return (cfg.x_min_mm, cfg.y_min_mm, cfg.x_max_mm, cfg.y_max_mm)
                return None
        __JOINT_LIMITS_PROP__

            # -------------------------------------------- robot body constants
            @property
            def home_pose(self):
                """Home pose object from the driver, or None before connect."""
                if self.low_level is not None:
                    return self.low_level.home_pose
                return None

            @property
            def tool_offset_mm(self) -> float:
                """Flange→tip offset (mm) from the driver, or 0 before connect."""
                if self.low_level is not None:
                    return float(self.low_level.tool_offset_mm)
                return float(self._cfg.tool_offset_mm)
        ''',
        CAPABILITIES=caps,
        DRIVER_KWARGS=_driver_kwargs(spec),
        CAMERA_OBSERVATION=camera_observation,
        JOINT_LIMITS_PROP=_env_joint_limits_prop(spec),
    )


# ---------------------------------------------------------------------------
# api.py
# ---------------------------------------------------------------------------


def _api_imports(mixins: list[str], tilted: bool) -> str:
    lines = ["from __future__ import annotations", ""]
    if tilted:
        lines += ["import math", ""]
    lines += [
        "from types import SimpleNamespace",
        "from typing import Any, Optional",
        "",
        "from jiuwensymbiosis.api.base import BaseRobotApi",
        "from jiuwensymbiosis.api.decorators import robot_tool",
        "from jiuwensymbiosis.api.mixins import (",
        *(f"    {mixin}," for mixin in mixins),
        ")",
    ]
    return "\n".join(lines)


def _api_detection_init() -> str:
    return _indent(
        """
        def __init__(
            self,
            env,
            *,
            detector_service_url: str = "http://127.0.0.1:8114",
            z_correction_mm: float = 0.0,
            grasp_z_offset_mm: float = -25.0,
            place_z_offset_mm: float = 75.0,
        ) -> None:
            super().__init__(env)
            self._detector_service_url = detector_service_url
            self._z_correction_mm = float(z_correction_mm)
            self._grasp_z_offset_mm = float(grasp_z_offset_mm)
            self._place_z_offset_mm = float(place_z_offset_mm)
            self._seg_fn = None
        """,
        4,
    )


def _api_vision_block() -> str:
    return _indent(
        f"""
        # ----------------------------------------------------------- Vision
        # Stubs return a serializable placeholder so the adapter passes
        # smoke; replace each body (see docs §6.4 and piper/api.py).
        @robot_tool(desc="Detect object_name, project to base XYZ. Returns {{ok, position, grasp_z, ...}}.")
        def get_grasp_info_simple(self, object_name: str) -> dict:
            \"\"\"Detect object_name and return grasp geometry.

            Recommended: implement ``_project_pixel_to_base_raw`` (the projection
            seam) and inherit this method from VisionMixin instead of overriding
            it — the shared pipeline runs detect -> project -> correct ->
            grasp/place geometry for you. This stub keeps the adapter smoke-clean
            until you wire the detector + calibration; see piper/api.py for a
            worked eye-in-hand example.
            \"\"\"
            {SENTINEL}
            return {{"ok": False, "object": object_name, "reason": "not_implemented"}}

        @robot_tool(desc="Convert pixel (u,v) at depth_m to base XYZ mm.")
        def pixel_to_base_xyz(self, u: float, v: float, depth_m: float) -> dict:
            \"\"\"Convert image pixel + depth to base-frame XYZ in mm.

            Reference shape::

                ll = self.env.low_level
                intrinsics = ll.intrinsics
                tf_flange_cam = ll.tf_flange_cam
                x_mm, y_mm, z_mm = project_pixel_to_base(u, v, depth_m, intrinsics, tf_flange_cam)
                return {{"ok": True, "position": [x_mm, y_mm, z_mm]}}

            This is calibration-dependent; use piper/geometry.py as the concrete example.
            \"\"\"
            {SENTINEL}
            return {{"ok": False, "reason": "not_implemented"}}

        @robot_tool(desc="Higher-level scene analysis grounded on object_name.")
        def analyze_scene(self, object_name: Optional[str] = None) -> dict:
            \"\"\"Return a lightweight scene summary.

            Reference shape::

                rgb = self.get_image()
                if rgb is None:
                    return {{"ok": False, "reason": "no_camera"}}
                detections = self._seg_fn(rgb, text_prompt=object_name or "object")
                return {{"ok": True, "count": len(detections)}}

            Keep this method side-effect free; it should observe, not move.
            \"\"\"
            {SENTINEL}
            return {{"ok": False, "reason": "not_implemented"}}
        """,
        4,
    )


def render_api(spec: Spec) -> str:
    mixins = _mixin_names(spec)
    tilted = _effective_tilted(spec)
    constants = ""
    if tilted:
        constants = (
            f"_TOOL_DOWN_RX = {_TOOL_DOWN_RX}\n_TOOL_DOWN_RY = {_TOOL_DOWN_RY}  # 略倾以改善抓取可达性 (参考 piper)\n\n"
        )
    get_items = ['"x": p.x', '"y": p.y', '"z": p.z - tool_off']
    get_items += [f'"{field}": getattr(p, "{field}", 0.0)' for field in spec.rot_fields]
    home_items = ", ".join(f'"{field}": getattr(hp, "{field}", 0.0)' for field in spec.pose_fields)
    r_default = (
        'r = getattr(self.env.get_flange_pose(), "r", 0.0)'
        if spec.dof == 4
        else 'r = getattr(self.env.get_flange_pose(), "rz", 0.0)'
    )
    if tilted:
        pose_build = _indent(
            """
            ry_rad = math.radians(_TOOL_DOWN_RY)
            flange_x = x + tool_off * math.sin(ry_rad)
            flange_z = z + tool_off * math.cos(ry_rad)
            pose = SimpleNamespace(
                x=flange_x,
                y=y,
                z=flange_z,
                rx=_TOOL_DOWN_RX,
                ry=_TOOL_DOWN_RY,
                rz=float(r),
            )
            """,
            8,
        )
    elif spec.dof == 4:
        pose_build = _block(
            ["pose = SimpleNamespace(x=float(x), y=float(y), z=float(z) + tool_off, r=float(r))"],
            8,
        )
    else:
        pose_build = _block(
            ["pose = SimpleNamespace(x=float(x), y=float(y), z=float(z) + tool_off, rx=180.0, ry=0.0, rz=float(r))"],
            8,
        )

    return _render(
        f'''
        """{spec.api_cls} — capability-mixin API for the {spec.name} robot.

        Motion / grasp / get_image inherit working defaults that delegate to the
        Env verbs; only the offset/tilt geometry and (if any) the vision methods
        are overridden here. See docs/hardware-porting-guide.md Step 4.
        """

        __IMPORTS__


        __CONSTANTS__class {spec.api_cls}(
        __MIXIN_BASES__
            BaseRobotApi,
        ):
            """Robot API for {spec.name}."""

        __INIT_BLOCK__

            # ----------------------------------------------------------- Motion
            @robot_tool(desc="Return {spec.name} to its home pose.", tags=["motion"])
            def home(self) -> None:
                """Return to the home pose (motion command → Env verb)."""
                self.env.home()

            @robot_tool(desc="Get current TIP pose (mm/deg, base frame).")
            def get_pose(self) -> dict:
                """Current tip pose (flange pose minus the tool offset)."""
                p = self.env.get_flange_pose()
                tool_off = self.env.tool_offset_mm
                return {{{", ".join(get_items)}}}

            @robot_tool(desc="Get the home pose constants (read-only).")
            def get_home_pose(self) -> dict:
                """Home pose constants read from the env."""
                hp = self.env.home_pose
                return {{{home_items}}}

            @robot_tool(desc="Move the TIP to absolute (x, y, z[, r]) in mm/deg, base frame.", tags=["motion"])
            def goto_xyzr(self, x: float, y: float, z: float, r: Optional[float] = None) -> None:
                """Move tip to target. tip↔flange geometry stays in the api layer."""
                tool_off = self.env.tool_offset_mm
                if r is None:
        __R_DEFAULT__
        __POSE_BUILD__
                self.env.move_to_flange(pose)

        __VISION_BLOCK__
        ''',
        IMPORTS=_api_imports(mixins, tilted),
        CONSTANTS=constants,
        MIXIN_BASES=_block([f"{mixin}," for mixin in mixins], 4),
        INIT_BLOCK=_api_detection_init() if spec.detection else "",
        R_DEFAULT=_block([r_default], 12),
        POSE_BUILD=pose_build,
        VISION_BLOCK=_api_vision_block() if spec.detection else "",
    )


# ---------------------------------------------------------------------------
# session.py
# ---------------------------------------------------------------------------


def render_session(spec: Spec) -> str:
    if spec.detection:
        body = f'''
        """{spec.builder_name} — one call from YAML to a ready-to-connect session.

            session = {spec.builder_name}.from_yaml('configs/{spec.name}/default.yaml')
        """

        from __future__ import annotations

        from jiuwensymbiosis.adapters._common.builder import make_builder
        from jiuwensymbiosis.adapters.{spec.name}.config import {spec.config_cls}
        from jiuwensymbiosis.adapters.{spec.name}.env import {spec.env_cls}
        from jiuwensymbiosis.adapters.{spec.name}.api import {spec.api_cls}


        {spec.builder_name} = make_builder(
            {spec.config_cls},
            {spec.env_cls},
            {spec.api_cls},
            api_kwargs_from_cfg=[
                "detector_url:detector_service_url",
                "z_correction_mm",
                "grasp_z_offset_mm",
                "place_z_offset_mm",
            ],
        )

        # To auto-spawn the GroundingDINO+SAM2 detector as a sidecar, give the
        # config a nested `detector` sub-config and add
        #   sidecar_builders=[make_detector_sidecar()]
        # above (see jiuwensymbiosis/adapters/piper/session.py).
        '''
    else:
        body = f'''
        """{spec.builder_name} — one call from YAML to a ready-to-connect session.

            session = {spec.builder_name}.from_yaml('configs/{spec.name}/default.yaml')
        """

        from __future__ import annotations

        from jiuwensymbiosis.adapters._common.builder import make_builder
        from jiuwensymbiosis.adapters.{spec.name}.config import {spec.config_cls}
        from jiuwensymbiosis.adapters.{spec.name}.env import {spec.env_cls}
        from jiuwensymbiosis.adapters.{spec.name}.api import {spec.api_cls}


        {spec.builder_name} = make_builder({spec.config_cls}, {spec.env_cls}, {spec.api_cls})
        '''
    return _render(body)


# ---------------------------------------------------------------------------
# __init__.py
# ---------------------------------------------------------------------------


def render_init(spec: Spec) -> str:
    return _render(
        f'''
        """{spec.name} adapter package."""

        from jiuwensymbiosis.adapters.{spec.name}.config import {spec.config_cls}
        from jiuwensymbiosis.adapters.{spec.name}.env import {spec.env_cls}
        from jiuwensymbiosis.adapters.{spec.name}.api import {spec.api_cls}
        from jiuwensymbiosis.adapters.{spec.name}.session import {spec.builder_name}

        __all__ = [
            "{spec.config_cls}",
            "{spec.env_cls}",
            "{spec.api_cls}",
            "{spec.builder_name}",
        ]
        '''
    )


# ---------------------------------------------------------------------------
# YAML
# ---------------------------------------------------------------------------


def render_yaml(spec: Spec) -> str:
    home = ", ".join(str(v) for _, v in _default_pose_pairs(spec))
    lines = [
        f"# {spec.name} 机械臂配置 (由 new_adapter 生成)",
        f"# 连接方式: {spec.connection}。{_connection_future_note(spec)}",
        "",
        f'name: "{spec.name}"',
        "",
        "# ---- 硬件连接 [必填] ----",
        f'connection: "{spec.connection}"',
    ]
    if spec.connection == "can":
        lines += ['can_port: "can0"', "can_bitrate: 1000000"]
    elif spec.connection == "serial":
        lines += [
            'serial_port: "/dev/ttyUSB0"',
            "baudrate: 115200",
            'connection_note: "serial template placeholder"',
        ]
    elif spec.connection == "tcp":
        lines += [
            'host: "192.168.1.10"',
            "port: 3000",
            'connection_note: "tcp template placeholder"',
        ]
    elif spec.connection == "usb":
        lines += [
            "device_serial: null",
            'connection_note: "usb template placeholder"',
        ]
    elif spec.connection == "ros":
        lines += [
            'ros_namespace: ""',
            'command_topic: "/robot/command"',
            'connection_note: "ros template placeholder"',
        ]
    else:
        lines += ['connection_note: "custom connection: fill hardware SDK fields here"']
    lines += [
        "move_speed: 50",
        "",
        "# ---- 运动学 ----",
        "tool_offset_mm: 0.0",
        f"home_pose_xyzrxryrz_mm_deg: [{home}]",
        "",
        "# ---- 安全边界 ----",
        "z_min_safe_mm: 50.0",
        "x_min_mm: 0.0",
        "x_max_mm: 700.0",
        "y_min_mm: -500.0",
        "y_max_mm: 500.0",
    ]
    if spec.joint:
        lines += [
            "",
            "# ---- 关节软限位 [选填-仅 motion.joint] ----",
            "# 单位须与 move_joint(q) 一致；键顺序 = q 索引顺序 (按 J1/J2/... 写)。",
            "# 限位值以官方手册为准，示例仅为占位。未配置则 SafetyRail 跳过越限检查。",
            "# joint_limits:",
            "#   J1: [-360.0, 360.0]",
            "#   J2: [-135.0, 135.0]",
            "#   J3: [-135.0, 135.0]",
            "#   J4: [-360.0, 360.0]",
            "#   J5: [-135.0, 135.0]",
            "#   J6: [-360.0, 360.0]",
        ]
    if spec.end_effector == "parallel":
        lines += ["", "# ---- 夹爪 ----", "gripper_open_mm: 70.0", "gripper_effort: 1000"]
    if spec.has_camera:
        lines += [
            "",
            "# ---- 相机 ----",
            "camera_serial: null",
            "camera_resolution: [640, 480]",
            "camera_fps: 30",
        ]
    if spec.detection:
        lines += [
            "",
            "# ---- 检测 ----",
            "z_correction_mm: 0.0",
            "grasp_z_offset_mm: -25.0",
            "place_z_offset_mm: 75.0",
            'detector_url: "http://127.0.0.1:8114"',
            "# calib_path: calib.json",
        ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# joint_ik backend — arms whose SDK takes only joint targets. The reusable
# KinematicArmDriver (adapters/_common) supplies the whole Cartesian/IK layer;
# the generated skeleton only holds the two body-specific seams (a JointTransport
# and an FkIkBackend) plus config. The backend composes with every orthogonal
# axis (gripper/suction/servo/vision) by reusing the shared render blocks and the
# driver's Gripper/Suction/Servo/Vision surface — the same capability machinery
# Family-1 uses. Motion.servo is declared only when asked (needs real-hw timing).
# ---------------------------------------------------------------------------


def _joint_names_literal(spec: Spec) -> str:
    return "[" + ", ".join(f'"{name}"' for name in spec.arm_joint_names) + "]"


def _home_zeros_literal(spec: Spec) -> str:
    return "[" + ", ".join("0.0" for _ in range(spec.joint_count)) + "]"


def _orientation_weight(spec: Spec) -> str:
    return "0.01" if spec.orientation_mode == "soft" else "1.0"


def _orientation_tolerance(spec: Spec) -> str:
    return "None" if spec.orientation_mode == "soft" else "5.0"


def _yaml_connection_lines(spec: Spec) -> list[str]:
    table = {
        "can": ['can_port: "can0"', "can_bitrate: 1000000"],
        "serial": ['serial_port: "/dev/ttyUSB0"', "baudrate: 115200", 'connection_note: "serial template placeholder"'],
        "tcp": ['host: "192.168.1.10"', "port: 3000", 'connection_note: "tcp template placeholder"'],
        "usb": ["device_serial: null", 'connection_note: "usb template placeholder"'],
        "ros": ['ros_namespace: ""', 'command_topic: "/robot/command"', 'connection_note: "ros template placeholder"'],
        "custom": ['connection_note: "custom connection: fill hardware SDK fields here"'],
    }
    return _by_connection(table, spec.connection)


def _joint_ik_effector_config_fields(spec: Spec) -> str:
    if not spec.has_grasp:
        return ""
    return _indent(
        """
        # ==================== 末端执行器 (两态) [选填-仅 grasp.*] ====================
        # engaged=夹紧/吸附, released=松开/关；原生单位 (parallel: 开/闭位置; suction: 开/关值)。
        effector_engaged_pos: float = 0.0
        effector_released_pos: float = 100.0
        effector_settle_s: float = 0.4
        """,
        4,
    )


def _joint_ik_optional_config_fields(spec: Spec) -> str:
    blocks = [_joint_ik_effector_config_fields(spec)]
    if spec.has_camera:
        blocks.append(_camera_config_fields())
    if spec.detection:
        blocks.append(_detection_config_fields())
    return "\n\n".join(b for b in blocks if b)


def render_config_joint_ik(spec: Spec) -> str:
    camera_resolution_fix = _camera_resolution_fix() if spec.has_camera else ""
    calib_path_fix = _calib_path_fix() if spec.detection else ""
    return _render(
        f'''
        """{spec.config_cls} — joint-level (URDF FK/IK) hardware configuration.

        Consumed by the reusable KinematicArmDriver; only the joint names, home,
        limits, URDF and tolerances are body-specific.
        """

        from __future__ import annotations

        import dataclasses
        from dataclasses import dataclass, field
        from pathlib import Path
        from typing import Any, Optional

        import yaml


        @dataclass
        class {spec.config_cls}:
            """Configuration for the {spec.name} joint-level arm."""

            # ==================== 基本信息 / 连接 [必填] ====================
            name: str = "{spec.name}"
            connection: str = "{spec.connection}"
        __CONNECTION_FIELDS__

            # ==================== 运动学关节 [必填] ====================
            # 不含夹爪；顺序即 move_joint(q) / FK / IK 的顺序。
            arm_joint_names: list[str] = field(default_factory=lambda: {_joint_names_literal(spec)})
            home_joints_deg: list[float] = field(default_factory=lambda: {_home_zeros_literal(spec)})
            joint_limits: Optional[dict[str, tuple[float, float]]] = None

            # ==================== URDF / IK ====================
            urdf_path: Optional[str] = None            # 相对 YAML 解析；指向本体 URDF
            ik_target_frame: str = "tool_frame"        # 控制帧 link 名 (须在 URDF 中)
            ik_orientation_weight: float = {_orientation_weight(spec)}
            ik_position_tolerance_mm: float = 3.0
            ik_orientation_tolerance_deg: Optional[float] = {_orientation_tolerance(spec)}

            # ==================== 轨迹 / 到位 [选填] ====================
            max_joint_step_deg: float = 2.0
            max_cartesian_step_mm: float = 5.0
            max_ik_jump_deg: float = 30.0
            trajectory_hz: float = 30.0
            joint_tolerance_deg: float = 1.5
            settle_samples: int = 3
            move_timeout_s: float = 30.0

            # ==================== 安全边界 [选填] ====================
            z_min_safe_mm: float = 50.0
            x_min_mm: Optional[float] = None
            x_max_mm: Optional[float] = None
            y_min_mm: Optional[float] = None
            y_max_mm: Optional[float] = None
        __OPTIONAL_FIELDS__

            # ==================== Loaders — 勿改 (框架契约) ====================
            @classmethod
            def from_dict(cls, data: dict[str, Any]) -> "{spec.config_cls}":
                """Construct from a flat dict; unknown keys ignored."""
                valid = {{f.name for f in dataclasses.fields(cls)}}
                clean: dict[str, Any] = {{k: v for k, v in data.items() if k in valid}}
                if "joint_limits" in clean:
                    _raw = clean["joint_limits"]
                    if not isinstance(_raw, dict):
                        clean["joint_limits"] = None
                    else:
                        _norm: dict[str, tuple[float, float]] = {{}}
                        for _k, _v in _raw.items():
                            if not isinstance(_v, (list, tuple)) or len(_v) != 2:
                                continue
                            try:
                                _norm[str(_k)] = (float(_v[0]), float(_v[1]))
                            except (TypeError, ValueError):
                                continue
                        clean["joint_limits"] = _norm if _norm else None
        __CAMERA_RESOLUTION_FIX__
                return cls(**clean)

            @classmethod
            def from_yaml(cls, path: str | Path) -> "{spec.config_cls}":
                """Load config from a YAML file; resolve a relative urdf_path."""
                path = Path(path).resolve()
                with path.open("r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {{}}
                cfg = cls.from_dict(data)
                if cfg.urdf_path and not Path(cfg.urdf_path).is_absolute():
                    cfg.urdf_path = str((path.parent / cfg.urdf_path).resolve())
        __CALIB_PATH_FIX__
                return cfg
        ''',
        CONNECTION_FIELDS=_connection_config_fields(spec),
        OPTIONAL_FIELDS=_joint_ik_optional_config_fields(spec),
        CAMERA_RESOLUTION_FIX=camera_resolution_fix,
        CALIB_PATH_FIX=calib_path_fix,
    )


def render_lowlevel_joint_ik(spec: Spec) -> str:
    prefix = spec.prefix
    typing_import = "from typing import Any, Optional" if spec.has_camera else "from typing import Any"
    effector_init = "float(getattr(cfg, 'effector_released_pos', 0.0))" if spec.has_grasp else "0.0"
    effector_spec_kwargs = ""
    if spec.has_grasp:
        effector_spec_kwargs = _block(
            [
                "effector_engaged_pos=cfg.effector_engaged_pos,",
                "effector_released_pos=cfg.effector_released_pos,",
                "effector_settle_s=cfg.effector_settle_s,",
            ],
            8,
        )
    calibration_attrs = ""
    if spec.detection:
        calibration_attrs = _block(
            [
                "# Loaded in open() from cfg.calib_path; the properties below just expose them.",
                "self._tf_flange_cam: Optional[Any] = None  # 4x4 camera->flange (hand-eye extrinsic)",
                "self._calibration: Optional[dict] = None    # raw calibration payload",
                "self._intrinsics: Optional[Any] = None      # 3x3 camera intrinsics matrix",
            ],
            8,
        )
    open_vision_hint = ""
    if spec.detection:
        open_vision_hint = _block(
            [
                "# Vision: start the camera and load cfg.calib_path here, then set",
                "# self._intrinsics / self._tf_flange_cam / self._calibration.",
            ],
            8,
        )
    # Camera / hand-eye methods live on the Transport (reused verbatim from the
    # shared blocks); KinematicArmDriver forwards its VisionDriver surface here.
    transport_vision = _join_optional_blocks(_lowlevel_camera_block(spec), _lowlevel_detection_block(spec))
    return _render(
        f'''
        """{prefix} joint-level seams — fill the two GENERATED-MOCK classes.

        Only the vendor SDK glue ({prefix}Transport) and the URDF FK/IK wrapper
        ({prefix}Kinematics) are body-specific. KinematicArmDriver
        (adapters/_common) adds the Cartesian/IK/waypoint/arrival logic plus the
        gripper/suction/servo/vision surface, so env.py binds it as low_level.
        """

        from __future__ import annotations

        {typing_import}

        import numpy as np

        from jiuwensymbiosis.adapters._common.kinematic_driver import (
            KinematicArmDriver,
            KinematicSpec,
        )


        class {prefix}Transport:
            """Vendor SDK seam (JointTransport). Mock keeps joints in memory."""

            def __init__(self, cfg: Any) -> None:
                self._cfg = cfg
                self._joints = [float(v) for v in cfg.home_joints_deg]
                self._effector = {effector_init}
                self._open = False
        __CALIBRATION_ATTRS__

            def open(self) -> None:
                """Open the SDK connection (serial / CAN / socket)."""
                {SENTINEL}
        __OPEN_VISION_HINT__
                self._open = True

            def close(self) -> None:
                """Release the SDK connection. Must be idempotent."""
                {SENTINEL}
                self._open = False

            def precheck(self) -> None:
                """Validate SDK preconditions (calibration file / feature names)."""
                {SENTINEL}
                return None

            def read_arm_joints(self) -> list[float]:
                """Current arm joints (native unit, ARM order, no effector)."""
                {SENTINEL}
                return list(self._joints)

            def send_arm_joints(self, q: list[float]) -> list[float]:
                """Command arm-joint targets; return the accepted targets."""
                {SENTINEL}
                self._joints = [float(v) for v in q]
                return list(self._joints)

            def read_effector(self) -> float:
                """Current end-effector value (native unit; gripper or suction)."""
                {SENTINEL}
                return float(self._effector)

            def send_effector(self, pos: float) -> float:
                """Command an end-effector value; return the accepted value."""
                {SENTINEL}
                self._effector = float(pos)
                return self._effector

        __TRANSPORT_VISION__

        class {prefix}Kinematics:
            """FK/IK backend (FkIkBackend). Mock = 3-axis gantry; wrap a real
            URDF solver (ikpy / pinocchio / vendor kinematics) here, built with
            cfg.urdf_path + cfg.arm_joint_names + cfg.ik_target_frame.
            """

            def __init__(self, cfg: Any) -> None:
                self._cfg = cfg

            def forward_kinematics(self, q: list[float]) -> np.ndarray:
                """Joint vector -> 4x4 SE(3) control-frame pose (translation in metres)."""
                {SENTINEL}
                matrix = np.eye(4)
                for i in range(min(3, len(q))):
                    matrix[i, 3] = float(q[i]) / 1000.0
                return matrix

            def inverse_kinematics(
                self, current: list[float], target: np.ndarray, *, orientation_weight: float
            ) -> list[float]:
                """Desired 4x4 SE(3) pose -> joint vector, seeded by current."""
                {SENTINEL}
                out = [float(v) for v in current]
                translation_mm = np.asarray(target)[:3, 3] * 1000.0
                for i in range(min(3, len(out))):
                    out[i] = float(translation_mm[i])
                return out


        def build_driver(cfg: Any) -> KinematicArmDriver:
            """Assemble the reusable driver from cfg + the two seams above."""
            spec = KinematicSpec(
                arm_joint_names=tuple(cfg.arm_joint_names),
                home_joints=tuple(cfg.home_joints_deg),
                joint_limits=dict(cfg.joint_limits or {{}}),
                ik_orientation_weight=cfg.ik_orientation_weight,
                ik_position_tolerance_mm=cfg.ik_position_tolerance_mm,
                ik_orientation_tolerance_deg=cfg.ik_orientation_tolerance_deg,
                max_joint_step_deg=cfg.max_joint_step_deg,
                max_cartesian_step_mm=cfg.max_cartesian_step_mm,
                max_ik_jump_deg=cfg.max_ik_jump_deg,
                trajectory_hz=cfg.trajectory_hz,
                joint_tolerance_deg=cfg.joint_tolerance_deg,
                settle_samples=cfg.settle_samples,
                move_timeout_s=cfg.move_timeout_s,
                z_min_safe_mm=cfg.z_min_safe_mm,
        __EFFECTOR_SPEC_KWARGS__
            )
            return KinematicArmDriver({prefix}Transport(cfg), {prefix}Kinematics(cfg), spec, name=cfg.name)
        ''',
        CALIBRATION_ATTRS=calibration_attrs,
        OPEN_VISION_HINT=open_vision_hint,
        TRANSPORT_VISION=transport_vision,
        EFFECTOR_SPEC_KWARGS=effector_spec_kwargs,
    )


def _joint_ik_effector_observation(spec: Spec) -> str:
    if not spec.has_grasp:
        return ""
    return _indent(
        """
        try:
            extra["effector"] = ll.get_effector_position()
            extra["effector_engaged"] = ll.gripper_state
        except Exception:
            pass
        """,
        8,
    )


def _joint_ik_camera_observation(spec: Spec) -> str:
    if not spec.has_camera:
        return ""
    return _indent(
        """
        if "vision.camera" in self.capabilities:
            try:
                frames = ll.grab_frames()
                if frames is not None:
                    rgb, depth = frames
            except Exception:
                pass
        """,
        8,
    )


def render_env_joint_ik(spec: Spec) -> str:
    caps = _block([f'"{cap}",' for cap in spec.capabilities], 12)
    return _render(
        f'''
        """{spec.env_cls} — joint-level hardware abstraction for {spec.name}.

        connect() builds the reusable KinematicArmDriver (adapters/_common) via
        lowlevel.build_driver and binds it as low_level; Env verbs delegate there.
        """

        from __future__ import annotations

        from typing import Any, Optional

        from jiuwensymbiosis.env.base import BaseRobotEnv, RobotObservation
        from jiuwensymbiosis.adapters.{spec.name}.lowlevel import build_driver


        class {spec.env_cls}(BaseRobotEnv):
            """Joint-level environment for the {spec.name} robot."""

            capabilities = frozenset(
                {{
        __CAPABILITIES__
                }}
            )
            name: str = "{spec.name}"

            def __init__(self, cfg: Any) -> None:
                self._cfg = cfg
                self._inner = None
                self._connected = False

            @property
            def low_level(self):
                """The bound KinematicArmDriver, or None before connect()."""
                return self._inner

            @low_level.setter
            def low_level(self, _: Any) -> None:
                raise AttributeError(f"{{type(self).__name__}}.low_level is read-only (bound in connect)")

            # ----------------------------------------------------- lifecycle
            def connect(self) -> None:
                """Build + connect the driver atomically; bind only on success."""
                if self._connected:
                    return
                driver = build_driver(self._cfg)
                driver.connect()
                self._inner = driver
                self._connected = True

            def disconnect(self) -> None:
                """Release the driver. Idempotent and safe at any state."""
                if not self._connected:
                    return
                try:
                    self._inner.disconnect()
                finally:
                    self._inner = None
                    self._connected = False

            # ---------------------------------------------------- observation
            def get_observation(self) -> RobotObservation:
                """Best-effort snapshot: pose (mm/deg), arm joints, effector/camera extra."""
                ll = self._inner
                if ll is None:
                    return RobotObservation()
                pose = None
                try:
                    p = ll.get_pose()
                    pose = {{"x": p.x, "y": p.y, "z": p.z, "rx": p.rx, "ry": p.ry, "rz": p.rz}}
                except Exception:
                    pose = None
                joints = None
                try:
                    joints = list(ll.get_angles())
                except Exception:
                    joints = None
                rgb = None
                depth = None
        __CAMERA_OBS__
                extra: dict = {{}}
        __EFFECTOR_EXTRA__
                return RobotObservation(pose=pose, joints=joints, rgb=rgb, depth=depth, extra=extra)

            # ----------------------------------------------- safety boundaries
            @property
            def z_min_safe(self) -> float:
                """Z floor (mm) — SafetyRail reads this automatically."""
                return float(self._cfg.z_min_safe_mm)

            @z_min_safe.setter
            def z_min_safe(self, _: float) -> None:
                raise AttributeError(f"{{type(self).__name__}}.z_min_safe is read-only (from config)")

            @property
            def workspace_bounds(self) -> Optional[tuple]:
                """XY workspace bounds (xmin, ymin, xmax, ymax) in mm, or None."""
                cfg = self._cfg
                if cfg.x_min_mm is not None and cfg.x_max_mm is not None:
                    return (cfg.x_min_mm, cfg.y_min_mm, cfg.x_max_mm, cfg.y_max_mm)
                return None

            @workspace_bounds.setter
            def workspace_bounds(self, _: Any) -> None:
                raise AttributeError(f"{{type(self).__name__}}.workspace_bounds is read-only (from config)")

            @property
            def joint_limits(self) -> Optional[dict]:
                """Arm-joint soft limits, ordered by arm_joint_names, or None."""
                jl = getattr(self._cfg, "joint_limits", None)
                if not jl:
                    return None
                return {{name: jl[name] for name in self._cfg.arm_joint_names if name in jl}}

            @joint_limits.setter
            def joint_limits(self, _: Any) -> None:
                raise AttributeError(f"{{type(self).__name__}}.joint_limits is read-only (from config)")

            # -------------------------------------------- robot body constants
            @property
            def home_pose(self):
                """Control-frame home pose (FK of home joints) from the driver, or None."""
                if self._inner is not None:
                    return self._inner.home_pose
                return None

            @home_pose.setter
            def home_pose(self, _: Any) -> None:
                raise AttributeError(f"{{type(self).__name__}}.home_pose is read-only (from driver)")

            @property
            def tool_offset_mm(self) -> float:
                """Flange->tip offset (mm); the control-frame convention keeps it 0."""
                if self._inner is not None:
                    return float(self._inner.tool_offset_mm)
                return 0.0

            @tool_offset_mm.setter
            def tool_offset_mm(self, _: float) -> None:
                raise AttributeError(f"{{type(self).__name__}}.tool_offset_mm is read-only (from driver)")
        ''',
        CAPABILITIES=caps,
        CAMERA_OBS=_joint_ik_camera_observation(spec),
        EFFECTOR_EXTRA=_joint_ik_effector_observation(spec),
    )


def _api_imports_joint_ik(mixins: list[str]) -> str:
    mixin_import = "\n".join(f"    {mixin}," for mixin in mixins)
    return (
        "from __future__ import annotations\n\n"
        "from types import SimpleNamespace\n"
        "from typing import Optional\n\n"
        "from jiuwensymbiosis.api.base import BaseRobotApi\n"
        "from jiuwensymbiosis.api.decorators import robot_tool\n"
        "from jiuwensymbiosis.api.mixins import (\n"
        f"{mixin_import}\n"
        ")"
    )


def _api_goto_joint_ik(spec: Spec) -> str:
    if spec.orientation_mode == "soft":
        desc = (
            "Move the TIP to absolute (x, y, z[, r]) in mm/deg, base frame. "
            "Underactuated arm: position is enforced, orientation is best-effort."
        )
    else:
        desc = "Move the TIP to absolute (x, y, z[, r]) in mm/deg, base frame."
    return _indent(
        f'''
        @robot_tool(desc="{desc}", tags=["motion"])
        def goto_xyzr(self, x: float, y: float, z: float, r: Optional[float] = None) -> None:
            """Position-first Cartesian move; the local IK handles posture."""
            if r is None:
                r = getattr(self.env.get_flange_pose(), "rz", 0.0)
            pose = SimpleNamespace(x=float(x), y=float(y), z=float(z), rx=180.0, ry=0.0, rz=float(r))
            self.env.move_to_flange(pose)
        ''',
        4,
    )


def _api_gripper_joint_ik(spec: Spec) -> str:
    if spec.end_effector != "parallel":
        return ""
    return _indent(
        """
        # Two-state gripper: the ToolCard exposes NO width/force (honest for a
        # position-only effector); the mixin-parity args are accepted but ignored.
        @robot_tool(
            desc="Open the gripper to its configured open position.",
            capability="grasp.parallel",
            input_params={"type": "object", "properties": {}},
            tags=["grasp"],
        )
        def open_gripper(self, width_mm: float = 80.0) -> dict:
            \"\"\"Open to the configured open position (width_mm ignored).\"\"\"
            self.env.set_end_effector(False)
            return {"ok": True, "state": "open"}

        @robot_tool(
            desc="Close the gripper to its configured close position.",
            capability="grasp.parallel",
            input_params={"type": "object", "properties": {}},
            tags=["grasp"],
        )
        def close_gripper(self, force_n: Optional[float] = None) -> dict:
            \"\"\"Close to the configured close position (force_n ignored).\"\"\"
            self.env.set_end_effector(True)
            return {"ok": True, "state": "closed"}
        """,
        4,
    )


def render_api_joint_ik(spec: Spec) -> str:
    mixins = _mixin_names(spec)
    return _render(
        f'''
        """{spec.api_cls} — capability-mixin API for the joint-level {spec.name} arm.

        Motion / joint / gripper / suction / vision inherit the mixin defaults that
        delegate to the Env verbs (the KinematicArmDriver does the IK and forwards
        the camera). Only the honest soft-posture goto and, for a parallel gripper,
        the two-state card are specialized; the vision stubs are the same the rest
        of the framework generates (fill them with your detector, or implement the
        VisionMixin projection seam _project_pixel_to_base_raw — see piper/api.py).
        """

        __IMPORTS__


        class {spec.api_cls}(
        __MIXIN_BASES__
            BaseRobotApi,
        ):
            """Robot API for the {spec.name} joint-level arm."""

        __INIT_BLOCK__
        __GOTO_OVERRIDE__
        __GRIPPER_OVERRIDE__
        __VISION_BLOCK__
        ''',
        IMPORTS=_api_imports_joint_ik(mixins),
        MIXIN_BASES=_block([f"{mixin}," for mixin in mixins], 4),
        INIT_BLOCK=_api_detection_init() if spec.detection else "",
        GOTO_OVERRIDE=_api_goto_joint_ik(spec),
        GRIPPER_OVERRIDE=_api_gripper_joint_ik(spec),
        VISION_BLOCK=_api_vision_block() if spec.detection else "",
    )


def render_yaml_joint_ik(spec: Spec) -> str:
    lines = [
        f"# {spec.name} 机械臂配置 (motion_backend=joint_ik, 由 new_adapter 生成)",
        "# 关节级 SDK + 本地 URDF FK/IK；填 urdf_path/关节名/home/限位即可。",
        "",
        f'name: "{spec.name}"',
        "",
        "# ---- 硬件连接 [必填] ----",
        f'connection: "{spec.connection}"',
        *_yaml_connection_lines(spec),
        "",
        "# ---- 运动学关节 (不含夹爪; 顺序=move_joint/FK/IK 顺序) [必填] ----",
        f"arm_joint_names: [{', '.join(repr(n) for n in spec.arm_joint_names)}]",
        f"home_joints_deg: [{', '.join('0.0' for _ in range(spec.joint_count))}]",
        "# joint_limits: (单位=度; 键须与 arm_joint_names 一致; 不填则跳过越限检查)",
        *[f"#   {name}: [-180.0, 180.0]" for name in spec.arm_joint_names],
        "",
        "# ---- URDF / IK ----",
        "# urdf_path: description/robot.urdf   # 相对本 YAML 解析",
        'ik_target_frame: "tool_frame"',
        f"ik_orientation_weight: {_orientation_weight(spec)}",
        "ik_position_tolerance_mm: 3.0",
        f"ik_orientation_tolerance_deg: {'null' if spec.orientation_mode == 'soft' else '5.0'}",
        "",
        "# ---- 轨迹 / 到位 ----",
        "max_joint_step_deg: 2.0",
        "max_cartesian_step_mm: 5.0",
        "max_ik_jump_deg: 30.0",
        "trajectory_hz: 30.0",
        "joint_tolerance_deg: 1.5",
        "settle_samples: 3",
        "move_timeout_s: 30.0",
        "",
        "# ---- 安全边界 ----",
        "z_min_safe_mm: 50.0",
        "# x_min_mm: -400.0",
        "# x_max_mm: 400.0",
        "# y_min_mm: -400.0",
        "# y_max_mm: 400.0",
    ]
    if spec.servo:
        lines += ["", "# motion.servo 已声明；真机验证持续 read+IK+send 的稳定频率/抖动后再启用伺服循环。"]
    if spec.has_grasp:
        kind = "parallel: 开/闭位置" if spec.end_effector == "parallel" else "suction: 开/关值"
        lines += [
            "",
            f"# ---- 末端执行器 (两态; {kind}) ----",
            "effector_engaged_pos: 0.0",
            "effector_released_pos: 100.0",
            "effector_settle_s: 0.4",
        ]
    if spec.has_camera:
        lines += [
            "",
            "# ---- 相机 ----",
            "camera_serial: null",
            "camera_resolution: [640, 480]",
            "camera_fps: 30",
        ]
    if spec.detection:
        lines += [
            "",
            "# ---- 检测 (手眼标定+检测服务; 未标定前视觉工具是诚实占位) ----",
            "z_correction_mm: 0.0",
            "grasp_z_offset_mm: -25.0",
            "place_z_offset_mm: 75.0",
            'detector_url: "http://127.0.0.1:8114"',
            "# calib_path: calib.json",
        ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def render_all(spec: Spec) -> dict[str, str]:
    """Return {repo-relative path: file text} for the whole adapter + config."""
    pkg = f"jiuwensymbiosis/adapters/{spec.name}"
    if spec.is_joint_ik:
        return {
            f"{pkg}/__init__.py": render_init(spec),
            f"{pkg}/config.py": render_config_joint_ik(spec),
            f"{pkg}/lowlevel.py": render_lowlevel_joint_ik(spec),
            f"{pkg}/env.py": render_env_joint_ik(spec),
            f"{pkg}/api.py": render_api_joint_ik(spec),
            f"{pkg}/session.py": render_session(spec),
            f"configs/{spec.name}/default.yaml": render_yaml_joint_ik(spec),
        }
    return {
        f"{pkg}/__init__.py": render_init(spec),
        f"{pkg}/config.py": render_config(spec),
        f"{pkg}/lowlevel.py": render_lowlevel(spec),
        f"{pkg}/env.py": render_env(spec),
        f"{pkg}/api.py": render_api(spec),
        f"{pkg}/session.py": render_session(spec),
        f"configs/{spec.name}/default.yaml": render_yaml(spec),
    }
