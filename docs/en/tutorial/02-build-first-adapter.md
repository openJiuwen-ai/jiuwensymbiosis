# Build Your First Robot Adapter

> Category: Tutorial. The [Chinese source](../../zh/tutorial/02-build-first-adapter.md) is authoritative.

This tutorial builds a runnable SCARA-with-suction adapter without hardware. Its sole goal is to teach the adapter
boundary; vendor SDK integration, calibration, and hardware acceptance belong to the
[production porting guide](../how-to/port-hardware-adapter.md). Use the
[adapter reference](../reference/adapter-reference.md) for exact contracts and parameters.

## 0. Prerequisites

### Layered framework architecture

```
┌──────────────────────────────────────────────────┐
│  Agent Layer      │  build_robot_agent()         │  one-call LLM agent construction
│                   │  RobotSession                │  hardware lifecycle + subprocesses
│                   │  RobotAgentConfig            │  model, mode, safety switches
├──────────────────────────────────────────────────┤
│  Safety Rails     │  SafetyRail                  │  pre-motion boundary interception
│                   │  RecoveryRail                │  automatic homing on failure
│                   │  VisualFeedbackRail          │  post-action visual verification
│                   │  DiagnosisRail / TraceRail   │  optional diagnosis and execution tracing
├──────────────────────────────────────────────────┤
│  Tool Layer       │  build_robot_tools(api)      │  each @implements → one LLM tool
│                   │  RobotControlTool(api)       │  single entry, action/params dispatch
│                   │  InProcessCodeTool           │  in-process Python execution
├──────────────────────────────────────────────────┤
│  Skill Layer      │  visual_pick/SKILL.md        │  pre-authored workflow documents
│                   │  visual_place/SKILL.md       │  auto-loaded by SkillUseRail
│                   │  transport/SKILL.md          │
├──────────────────────────────────────────────────┤
│  API Layer        │  ActionSpec vocabulary       │  action contract (name/capability/pre/effects)
│  (Action          │  @implements(SPEC)           │  binds one contract + ToolMeta
│   Contract)       │  api/defaults                │  generic implementations (free functions, not a base class)
│                   │  BaseRobotApi                │  holds the env reference
├──────────────────────────────────────────────────┤
│  Env Layer        │  BaseRobotEnv                │  the (single) hardware contract surface
│  (Hardware        │  connect/disconnect/observe  │  capability declaration (env.capabilities)
│   Abstraction)    │  move_to_flange/set_end_...  │  default driver delegation; home is body-defined
│                   │  home_pose/tool_offset_mm    │  robot constant properties
│                   │  grab_rgb()                  │  single frame (goes through get_observation by default)
│                   │  z_min_safe/workspace_bounds │  safety contract properties
│                   │  joint_limits                │  joint soft limits (motion.joint only)
│                   │  low_level: RobotDriver      │  controlled penetration point (calibration / vendor-specific)
├──────────────────────────────────────────────────┤
│  Hardware Layer   │  XxxDriver (lowlevel.py)     │  implements the Driver Protocols
│  (Your Code)      │  serial / CAN / socket / ... │  the adapter author's main work
└──────────────────────────────────────────────────┘
```

> Env is the **single contract surface** between Agent/Rails/Tools and the hardware:
> - motion / end effector through Env verbs (`home`/`move_to_flange`/`move_joint`/`get_flange_pose`/`set_end_effector`)
> - robot constants through Env properties (`home_pose`/`tool_offset_mm`)
> - safety bounds through Env properties (`z_min_safe`/`workspace_bounds`/`joint_limits`)
> - a single frame through an Env method (`grab_rgb()`, delegating to `get_observation().rgb` by default)
> - calibration data through `env.low_level` (controlled penetration typed by `RobotDriver` + its sub-Protocols)
>
> Upper layers reach hardware through Env's public API and never `getattr` the private driver. `set_end_effector`
> dispatches deterministically on declared capability (`grasp.parallel`/`grasp.suction`).

### Core concepts

| Concept | Definition | Who defines it |
|------|------|--------|
| **Capability** | A named string for a hardware capability, e.g. `"motion.cartesian"` | `env/base.py:KNOWN_CAPABILITIES` |
| **ActionSpec** | One action's contract: name/description/capability gate/params/result shape/pre-conditions and effects/location freshness | Class in `api/decorators.py`; instances in `api/actions.py` |
| **Env** | The hardware-driver wrapper, implementing `connect/disconnect/get_observation` | The adapter author |
| **Api** | Subclasses `BaseRobotApi` and binds each action with `@implements(SPEC)`; forwards to `api/defaults` where there is no difference and implements body-specific geometry/camera hooks explicitly | The adapter author |
| **Config** | A dataclass of hardware parameters, with `from_yaml/from_dict` | The adapter author |
| **Session** | Packages Env + Api + subprocesses as one lifecycle unit | generated by `make_builder()` |
| **Sidecar** | A subprocess started/stopped with the Session (e.g. the vision detection server) | `perception/detector_sidecar.py` (`_common/builder.py` provides assembly helpers) |

### Conventions

- An Env's `capabilities` is a **manually declared** frozenset.
- An Api's `capabilities` is **derived from the actions' specs** (each `@implements` contributes its `ActionSpec`'s
  capability, plus any declared marker-capability class attribute).
- Tools are gated on **`api.capabilities ∩ env.capabilities`**: a capability the Api has but the Env lacks means its
  tool is **never exposed to the LLM at runtime** (`build_robot_tools` enforces the intersection, and
  `validate_adapter` reports an ERROR). `session.describe()`'s `effective_capabilities` is exactly this intersection.
  A marker capability the Env has but the Api lacks (such as `vision.camera`) does not affect the run.

This tutorial uses a four-axis SCARA arm with suction, millimeters for cartesian positions, degrees for rotation, and an
in-memory driver. Replace only the driver I/O after the adapter passes validation; keep the public Env and Api contracts
unchanged.

## 1. Beginner overview: run a minimal adapter in five minutes

```bash
# 1. Copy the template
cp -r templates/xxx_adapter/ jiuwensymbiosis/adapters/my_robot/

# 2. Edit the adapter files and replace every Xxx/xxx/build_xxx_session placeholder
#  - config.py: define hardware connection settings
#  - lowlevel.py: implement hardware I/O, or begin with a mock
#  - env.py: declare capabilities and implement connect/disconnect/observe
#  - api.py: bind each action with @implements(SPEC), forwarding to defaults or override geometry
#  - session.py: keep declarative make_builder wiring, but rename classes and build_xxx_session
#  - __init__.py: rename the exported build_xxx_session as well
#  - config_template.yaml: copy it to configs/my_robot/default.yaml and fill it in

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
the Api only adapts SCARA field names and forwards the rest to `api.defaults`.

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

from jiuwensymbiosis.adapters._common.config import load_yaml_config


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
        return load_yaml_config(cls, path)
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

    def home(self) -> None:
        self._require_cartesian().home()

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

> `get_flange_pose`/`move_to_flange`/`set_end_effector`/`grab_rgb` are delegated to `low_level` by `BaseRobotEnv`
> already, so ScaraEnv need not implement them. `home` is abstract — every body states its own safe posture, so an arm
> delegates to the driver while a mobile body writes its own sequence. `home_pose`/`tool_offset_mm` are exposed as
> properties.

## 5. Compose the Api

Each method binds one entry of the shared action vocabulary with `@implements(SPEC)` — the action name, capability
gate, params and effects all come from `api/actions.py`, not from this class. Anything with no body-specific difference
(`goto_xyzr` / suction on-off) forwards to `api.defaults`; only a real difference gets a method body — here, exposing
the driver's `rz` as the `r` field SCARA users expect:

```python
from typing import Literal

from jiuwensymbiosis.api import defaults
from jiuwensymbiosis.api.actions import (
    ACTIVATE_SUCTION,
    GET_HOME_POSE,
    GET_POSE,
    GOTO_XYZR,
    implements,
)
from jiuwensymbiosis.api.base import BaseRobotApi


class ScaraApi(BaseRobotApi):
    # Nothing body-specific: forward to the shared implementation.
    @implements(GOTO_XYZR)
    def goto_xyzr(
        self,
        x: float,
        y: float,
        z: float,
        r: float | None = None,
        orientation_policy: Literal["top_down", "preserve"] = "top_down",
    ) -> None:
        return defaults.goto_xyzr(
            self, x, y, z, r, orientation_policy=orientation_policy
        )

    @implements(ACTIVATE_SUCTION)
    def activate_suction(self) -> dict:
        return defaults.activate_suction(self)

    # Body-specific: the driver reports `rz`, the SCARA vocabulary says `r`.
    @implements(GET_POSE)
    def get_pose(self) -> dict:
        pose = self.env.get_flange_pose()
        return {"x": pose.x, "y": pose.y, "z": pose.z, "r": pose.rz}

    @implements(GET_HOME_POSE)
    def get_home_pose(self) -> dict:
        pose = self.env.home_pose
        return {"x": pose.x, "y": pose.y, "z": pose.z, "r": pose.rz}
```

Every method binds one entry of the shared action vocabulary, so the name, capability gate, and effects come from
`api/actions.py` rather than from this class. Where the body has nothing of its own to say, the body forwards to
`api.defaults`; where it does — the `rz`/`r` field name above — it writes the body out in full. `ScaraApi` declares no
capabilities directly: `BaseRobotApi.capabilities` derives them from the specs its methods implement.

## 6. Assemble the Session

```python
from pathlib import Path

from jiuwensymbiosis.adapters._common.builder import make_builder
from jiuwensymbiosis.adapters.my_scara.api import ScaraApi
from jiuwensymbiosis.adapters.my_scara.config import ScaraConfig
from jiuwensymbiosis.adapters.my_scara.env import ScaraEnv

build_my_scara_session = make_builder(
    ScaraConfig, ScaraEnv, ScaraApi,
    resource_keys=lambda cfg: (f"serial:{Path(cfg.serial_port).expanduser().resolve()}",),
)
```

Export `build_my_scara_session` from `adapters/my_scara/__init__.py`, matching the package name. `resource_keys(cfg)` declares the actual command endpoint; aliases for one serial device must yield the same key. The mock driver never opens the port, but declaring its identity exercises the complete admission contract. Builders only assemble sessions; Runtime or the official CLI `admitted_session` owns admission.

## 7. Write the YAML configuration

Save this configuration as `configs/my_scara/default.yaml` and copy it to the adapter's `config_template.yaml`.

```yaml
# SCARA arm with suction configuration
adapter: my_scara
name: scara
serial_port: /dev/ttyUSB0
z_min_safe_mm: 30.0
tool_offset_mm: 0.0
home_xyzr: [200.0, 0.0, 250.0, 0.0]
```

## 8. Validate and continue to production porting

```bash
python scripts/validate_adapter.py --module jiuwensymbiosis.adapters.my_scara
python scripts/smoke_test_adapter.py --module jiuwensymbiosis.adapters.my_scara --config configs/my_scara/default.yaml
```

The static validator checks structure, capability alignment, and signatures. The smoke test injects a stub driver into
the Env without connecting real hardware, calls every generated `@implements` action, and verifies serializable output. Before replacing the mock with hardware, read the
[porting guide](../how-to/port-hardware-adapter.md) for workspace limits, recovery behavior, sidecars, testing, and
hardware acceptance.

Resource identity is part of completion: S-17 checks the factory and callback interfaces, then smoke validates device keys from the actual config. When no device key can be derived, declare top-level `physical_device_id`; camera or detector keys do not identify the robot. For calibration or URDF path fields, declare `path_fields: ClassVar[tuple[str, ...]]` on Config and keep `load_yaml_config`; Runtime and smoke share the same source-relative resolution even for missing files.
