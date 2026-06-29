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

HEADER = (
    "# coding: utf-8\n"
    "# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.\n"
)

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


def _config_optional_fields(spec: Spec) -> str:
    blocks: list[str] = []
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
        blocks.append(
            _indent(
                """
                # ==================== 相机 [选填-仅 vision.*] ====================
                camera_serial: Optional[str] = None # 相机序列号 (None=禁用)
                camera_resolution: tuple[int, int] = (640, 480)
                camera_fps: int = 30
                """,
                4,
            )
        )
    if spec.detection:
        blocks.append(
            _indent(
                """
                # ============== 检测校正 [选填-仅 vision.detection] ==============
                z_correction_mm: float = 0.0        # Z 向常值校正
                grasp_z_offset_mm: float = -25.0    # 抓取点相对物体顶面偏移
                chip_thickness_mm: float = 75.0     # 堆叠放置偏移
                detector_url: str = "http://127.0.0.1:8114"  # 检测服务地址
                calib_path: Optional[str] = None    # 手眼标定文件 (JSON)
                """,
                4,
            )
        )
    return "\n\n".join(blocks)


def render_config(spec: Spec) -> str:
    home_default = ", ".join(str(value) for _, value in _default_pose_pairs(spec))
    camera_resolution_fix = ""
    if spec.has_camera:
        camera_resolution_fix = _indent(
            """
            if "camera_resolution" in clean and isinstance(clean["camera_resolution"], list):
                clean["camera_resolution"] = tuple(clean["camera_resolution"])
            """,
            8,
        )
    calib_path_fix = ""
    if spec.detection:
        calib_path_fix = _indent(
            """
            if cfg.calib_path and not Path(cfg.calib_path).is_absolute():
                candidate = (path.parent / cfg.calib_path).resolve()
                if candidate.exists():
                    cfg.calib_path = str(candidate)
            """,
            8,
        )

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
        [
            f'self._pose["{key}"] = float(getattr(hp, "{key}", {default!r}))'
            for key, default in pairs
        ],
        8,
    )
    pose_update = _block(
        [
            f'self._pose["{key}"] = float(getattr(pose, "{key}", {default!r}))'
            for key, default in pairs
        ],
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
            chip_thickness_mm: float = 75.0,
        ) -> None:
            super().__init__(env)
            self._detector_service_url = detector_service_url
            self._z_correction_mm = float(z_correction_mm)
            self._grasp_z_offset_mm = float(grasp_z_offset_mm)
            self._chip_thickness_mm = float(chip_thickness_mm)
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

            Reference shape::

                frames = self.env.low_level.grab_frames()
                if frames is None:
                    return {{"ok": False, "object": object_name, "reason": "no_camera"}}
                rgb, depth_m = frames
                detector_result = run_detector(rgb, object_name)
                u, v = detector_result.pixel_uv
                xyz = self.pixel_to_base_xyz(u, v, depth_m[v, u])
                return build_grasp_result(object_name, xyz, detector_result)

            For eye-in-hand RGB-D, compare piper/api.py and _common/vision.py.
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
            f"_TOOL_DOWN_RX = {_TOOL_DOWN_RX}\n"
            f"_TOOL_DOWN_RY = {_TOOL_DOWN_RY}  # 略倾以改善抓取可达性 (参考 piper)\n\n"
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
            [
                "pose = SimpleNamespace("
                "x=float(x), y=float(y), z=float(z) + tool_off, rx=180.0, ry=0.0, rz=float(r))"
            ],
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
                "chip_thickness_mm",
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
            "chip_thickness_mm: 75.0",
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
    return {
        f"{pkg}/__init__.py": render_init(spec),
        f"{pkg}/config.py": render_config(spec),
        f"{pkg}/lowlevel.py": render_lowlevel(spec),
        f"{pkg}/env.py": render_env(spec),
        f"{pkg}/api.py": render_api(spec),
        f"{pkg}/session.py": render_session(spec),
        f"configs/{spec.name}/default.yaml": render_yaml(spec),
    }
