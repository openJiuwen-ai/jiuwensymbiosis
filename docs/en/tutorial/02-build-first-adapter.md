# Build Your First Robot Adapter

> Category: Tutorial. The [Chinese source](../../zh/tutorial/02-build-first-adapter.md) is authoritative.

This tutorial builds a runnable SCARA-with-suction adapter without hardware. Its sole goal is to teach the adapter
boundary; vendor SDK integration, calibration, and hardware acceptance belong to the
[production porting guide](../how-to/port-hardware-adapter.md). Use the
[adapter reference](../reference/adapter-reference.md) for exact contracts and parameters.

## 0. Prerequisites

### Layered framework architecture

| Layer | What the framework provides | What an adapter provides |
| --- | --- | --- |
| Agent and Rails | Agent construction, lifecycle, safety, recovery, tracing | Configuration values |
| Tools and Skills | LLM tool generation and reusable workflows | Capability selection |
| API | Capability Mixins and shared default behavior | Body-specific geometry and raw vision projection |
| Env | The single hardware contract | Connection, observation, and safety properties |
| Hardware | Driver Protocols | CAN, serial, socket, camera, and actuator calls |

The Env is the only hardware contract used by Agent, Rails, and Tools. Motion and end-effector commands use public Env
verbs such as `home()`, `move_to_flange()`, and `set_end_effector()`. Safety reads `z_min_safe`,
`workspace_bounds`, and `joint_limits`. Vendor calibration remains available through the typed `env.low_level`
penetration point.

### Core concepts

| Concept | Meaning |
| --- | --- |
| Capability | A known hardware feature such as `motion.cartesian` or `grasp.suction` |
| Mixin | A class that declares one capability and supplies `@robot_tool` methods |
| Driver | The vendor-facing implementation of the relevant Driver Protocols |
| Env | The lifecycle, observation, motion, and safety wrapper around the driver |
| Api | A composition of capability Mixins plus body-specific overrides |
| Config | A dataclass that loads hardware and service settings |
| Session | Env, Api, sidecars, and lifecycle assembled as one unit |

Env capabilities are declared manually. Api capabilities are derived from its Mixin MRO. Tools are exposed only for
the intersection `api.capabilities ∩ env.capabilities`; adding a Mixin without matching Env support does not expose an
unsafe tool.

### Conventions

This tutorial uses a four-axis SCARA arm with suction, millimeters for cartesian positions, degrees for rotation, and an
in-memory driver. Replace only the driver I/O after the adapter passes validation; keep the public Env and Api contracts
unchanged.

## 1. Beginner overview: run a minimal adapter in five minutes

```bash
# 1. Copy the template
cp -r templates/xxx_adapter/ jiuwensymbiosis/adapters/my_robot/

# 2. Edit the adapter files
#  - config.py: define hardware connection settings
#  - lowlevel.py: implement hardware I/O, or begin with a mock
#  - env.py: declare capabilities and implement connect/disconnect/observe
#  - api.py: compose Mixins and override body-specific geometry
#  - session.py: normally remains declarative make_builder wiring

# 3. Validate static structure and runtime behavior
python scripts/validate_adapter.py --module jiuwensymbiosis.adapters.my_robot
python scripts/smoke_test_adapter.py --module jiuwensymbiosis.adapters.my_robot

# 4. Run with the YAML configuration after creating it
```

Once you have a YAML configuration:

```bash
python -c "
from jiuwensymbiosis.adapters.my_robot import build_my_robot_session
session = build_my_robot_session.from_yaml('configs/my_robot/default.yaml')
with session:
    print(session.describe())
"
```

## 2. Implement the Mock Driver

Now implement the complete minimal adapter under `jiuwensymbiosis/adapters/my_scara/`. Most work is in the driver;
the Api only adapts SCARA field names and inherits the remaining Mixin behavior.

Start with an in-memory driver. Replace the method bodies with serial or CAN calls after the framework integration works.

```python
from types import SimpleNamespace


class MockScaraDriver:
    def __init__(self):
        self._pose = {"x": 200.0, "y": 0.0, "z": 250.0, "rz": 0.0}
        self.home_pose = SimpleNamespace(x=200.0, y=0.0, z=250.0, rz=0.0)
        self.tool_offset_mm = 0.0
        self._connected = False

    def connect(self) -> None:
        self._connected = True

    def close(self) -> None:
        self._connected = False

    def get_pose(self):
        return SimpleNamespace(**self._pose)

    def home(self) -> None:
        self._pose = {"x": 200.0, "y": 0.0, "z": 250.0, "rz": 0.0}

    def move_to_pose_blocking(self, pose) -> None:
        self._pose.update(x=pose.x, y=pose.y, z=pose.z, rz=getattr(pose, "rz", 0.0))

    def set_suction(self, on: bool) -> None:
        pass
```

## 3. Define Config

```python
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ScaraConfig:
    name: str = "scara"
    serial_port: str = "/dev/ttyUSB0"
    z_min_safe_mm: float = 30.0
    tool_offset_mm: float = 0.0
    home_xyzr: tuple = (200.0, 0.0, 250.0, 0.0)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScaraConfig":
        valid = {field.name for field in dataclasses.fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in valid})

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ScaraConfig":
        with Path(path).open(encoding="utf-8") as stream:
            return cls.from_dict(yaml.safe_load(stream) or {})
```

## 4. Wrap the Env

```python
from jiuwensymbiosis.adapters.my_scara.lowlevel import MockScaraDriver
from jiuwensymbiosis.env.base import BaseRobotEnv, RobotObservation


class ScaraEnv(BaseRobotEnv):
    capabilities = frozenset({"motion.cartesian", "grasp.suction"})
    name = "scara"

    def __init__(self, cfg):
        self._cfg = cfg
        self.low_level = None

    def connect(self) -> None:
        if self.low_level is not None:
            return
        self.low_level = MockScaraDriver()
        self.low_level.connect()

    def disconnect(self) -> None:
        if self.low_level is not None:
            self.low_level.close()
            self.low_level = None

    def get_observation(self) -> RobotObservation:
        if self.low_level is None:
            return RobotObservation()
        pose = self.low_level.get_pose()
        return RobotObservation(pose={"x": pose.x, "y": pose.y, "z": pose.z, "r": pose.rz})

    @property
    def z_min_safe(self) -> float:
        return self._cfg.z_min_safe_mm

    @property
    def home_pose(self):
        return self.low_level.home_pose if self.low_level is not None else None

    @property
    def tool_offset_mm(self) -> float:
        return self.low_level.tool_offset_mm if self.low_level is not None else 0.0
```

`BaseRobotEnv` supplies the common `home`, `get_flange_pose`, `move_to_flange`, `set_end_effector`, and
`grab_rgb` delegations. The Env only adds lifecycle, observation, and configuration-backed safety properties.

## 5. Compose the Api

```python
from jiuwensymbiosis.api.base import BaseRobotApi
from jiuwensymbiosis.api.decorators import robot_tool
from jiuwensymbiosis.api.mixins import MotionMixin, SuctionMixin


class ScaraApi(MotionMixin, SuctionMixin, BaseRobotApi):
    @robot_tool(desc="Get the current tip pose in mm/deg.")
    def get_pose(self) -> dict:
        pose = self.env.get_flange_pose()
        return {"x": pose.x, "y": pose.y, "z": pose.z, "r": pose.rz}

    @robot_tool(desc="Get the configured home pose.")
    def get_home_pose(self) -> dict:
        pose = self.env.home_pose
        return {"x": pose.x, "y": pose.y, "z": pose.z, "r": pose.rz}
```

The Api inherits `home`, `goto_xyzr`, suction control, and movement helpers. The overrides above only adapt the
driver's `rz` field to the SCARA-facing `r` convention.

## 6. Assemble the Session

```python
from jiuwensymbiosis.adapters._common.builder import make_builder
from jiuwensymbiosis.adapters.my_scara.api import ScaraApi
from jiuwensymbiosis.adapters.my_scara.config import ScaraConfig
from jiuwensymbiosis.adapters.my_scara.env import ScaraEnv

build_scara_session = make_builder(ScaraConfig, ScaraEnv, ScaraApi)
```

## 7. Write the YAML configuration

```yaml
# SCARA arm with suction configuration
name: scara
serial_port: /dev/ttyUSB0
z_min_safe_mm: 30.0
tool_offset_mm: 0.0
home_xyzr: [200.0, 0.0, 250.0, 0.0]
```

## 8. Validate and continue to production porting

```bash
python scripts/validate_adapter.py --module jiuwensymbiosis.adapters.my_scara
python scripts/smoke_test_adapter.py --module jiuwensymbiosis.adapters.my_scara
```

The static validator checks structure, capability alignment, and signatures. The smoke test connects a mock environment,
calls every generated `@robot_tool`, and verifies serializable output. Before replacing the mock with hardware, read the
[porting guide](../how-to/port-hardware-adapter.md) for workspace limits, recovery behavior, sidecars, testing, and
hardware acceptance.
