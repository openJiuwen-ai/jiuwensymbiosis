# JiuwenSymbiosis

## Introduction

**JiuwenSymbiosis** is an embodied agent framework built on top of openjiuwen — a Symbiosis architecture designed for embodied intelligence. It provides configuration-agnostic tools, safety policies, and multi-agent collaboration capabilities for embodied intelligence scenarios. Through the Capability Mixin architecture, a single codebase adapts to different form factors including SCARA, 6-DoF, suction cup, and gripper configurations. Built-in safety rails and visual feedback closed loops enable LLMs to safely manipulate the physical world.

## Why JiuwenSymbiosis?

- **Hardware Decoupling**: A single framework adapts to various form factors including SCARA, 6-DoF, suction cup, and gripper configurations. New hardware only requires a YAML configuration and a hardware adapter layer — no changes to the core agent logic.
- **Built-in Safety Rails**: SafetyRail (Z-axis lower limit / workspace boundary interception), RecoveryRail (automatic homing on motion failure + end-effector release), and VisualFeedbackRail (camera frame injection after every action for result verification) ensure LLMs operate safely in the physical world.
- **Visual Closed Loop**: The visual perception service runs as a sidecar process, combined with wrist camera back-projection to base coordinates. The LLM only needs to describe the target object in natural language to obtain 3D grasp poses — no need to understand pixels or camera intrinsics.
- **Auditable Skill Workflows**: Pre-built SKILL.md skill documents such as visual_pick / visual_place / slot_pick guide the LLM to execute in standardized steps rather than free-form orchestration, ensuring reproducibility and auditability.

## Quick Start

### Installation

- Operating System: The current version supports Ubuntu (verified on 22.04).
- Python Version: >= 3.11.

Create a new conda environment:
```bash
conda create -n your_env_name python=3.11
conda activate your_env_name
```

Install the package (editable mode recommended):

```bash
# Development installation (core + test deps)
pip install -e ".[dev]"

# Full installation (adds vision/GPU deps: torch, transformers, FastAPI, etc.)
pip install -e ".[full]"

# Piper hardware (installs piper_sdk)
pip install -e ".[piper]"
```

Or install from the pinned requirements file for reproducibility:

```bash
pip install -r requirements.txt
```

### Example

The following example performs a vision-guided pick using a Piper 6-DoF robotic arm. It requires a real hardware environment (CAN bus activated, visual perception service deployed).

```python
import asyncio

from jiuwensymbiosis.utils.proxy import clear_proxy_env

clear_proxy_env()  # Must be called before importing openjiuwen

from jiuwensymbiosis import build_robot_agent
from jiuwensymbiosis.agent import RobotAgentConfig, ModelSpec
from jiuwensymbiosis.adapters.piper import build_piper_session


def main():
    # 1. Build a hardware session from YAML
    session = build_piper_session.from_yaml("configs/piper/pick_box.yaml")

    # 2. Configure the LLM
    model_spec = ModelSpec(
        provider="OpenAI",
        api_base="https://api.siliconflow.cn/v1",
        api_key="your-api-key-here",
        model_name="deepseek-ai/DeepSeek-V3.2",
    )

    # 3. Configure the Agent
    config = RobotAgentConfig(
        mode="hybrid",
        model_spec=model_spec,
        enable_skill=True,             # Enable visual_pick / visual_place skill documents
        enable_visual_feedback=False,  # Disable visual feedback when using text-only models
        max_iterations=30,
    )

    # 4. Run
    with session:
        agent = build_robot_agent(session, config=config)
        result = asyncio.run(
            agent.invoke({
                "query": "Pick up the black box and place it on top of the white box.",
                "conversation_id": "pick-box-001",
            })
        )
        print(result)


if __name__ == "__main__":
    main()
```

Expected output: The agent executes the visual_pick / visual_place skill workflow in sequence — homing → detecting target → approach → grasping → transporting → placing → releasing — and returns a task completion status.

### Running the Demo

```bash
# Via console script (after pip install)
piper-pick-demo --config configs/piper/pick_box.yaml --mock

# Or run the script directly
python examples/piper_pick_demo.py --config configs/piper/pick_box.yaml --mock

# Real hardware mode (CAN bus must be active)
python examples/piper_pick_demo.py \
    --config configs/piper/pick_box.yaml \
    --max-iter 30 \
    --api-key ""
```

## Architecture

```
env/         Hardware abstraction layer (BaseRobotEnv, RobotObservation, MockArmEnv)
api/         Capability mixins + @robot_tool decorator (MotionMixin, SuctionMixin, VisionMixin...)
tools/       Tool builder / InProcessCodeTool / RobotControlTool / slot_pick
agent/       RobotSession + build_robot_agent / build_robot_agent_config + configuration
rails/       Safety policies (SafetyRail, RecoveryRail, VisualFeedbackRail)
skills/      Built-in skills (visual_pick, visual_place, slot_pick)
adapters/    Hardware adapter layer (piper/ + _common/ generic builder)
serving/     Visual perception service subprocess (current version: GroundingDINO + SAM2)
```

* **Hardware Abstraction Layer**: `BaseRobotEnv` defines the minimal hardware contract (connect / disconnect / observe). `MockArmEnv` provides a hardware-free testing environment.
* **Hardware Capability Layer**: `MotionMixin`, `SuctionMixin`, `ParallelGripperMixin`, `VisionMixin`, etc. declare atomic capabilities for various hardware types. Concrete hardware API classes compose required capabilities through multiple inheritance.
* **Tool Layer**: `build_robot_tools` automatically wraps `@robot_tool` methods as openjiuwen tools; `RobotControlTool` provides a single-entry dispatch pattern; `InProcessCodeTool` supports in-process Python script execution.
* **Agent Layer**: `RobotSession` manages the hardware lifecycle and sidecar processes; `build_robot_agent` constructs a callable DeepAgent in one step.
* **Safety Policy Layer**: Subclasses of openjiuwen AgentRail that insert safety checks, exception recovery, and visual feedback before and after tool invocations.
* **Hardware Adapter Layer**: `piper/` demonstrates how to integrate a specific hardware platform; `_common/builder.py` provides a generic polymorphic session factory.

## Features

### Situational Awareness Loop

A "Perceive → Plan → Execute → Observe → Feedback" closed loop for continuous system stability improvement.

**Perception & Understanding:**
Enables the embodied agent to actively perceive the physical world.
**Safe Planning:**
Performs task planning based on task instructions and world state. Dynamically assigns skill parameters, validates physical feasibility, safety, and constraints, and filters out infeasible plans.
**Physical Execution:**
Invokes atomic action tools for execution following the steps suggested in skill.md.
**Observation & Feedback:**
Collects the real-world state after execution via sensors, identifies object poses and environmental changes, and feeds deviation, anomaly, and result signals back to the planning module for real-time action parameter adjustment, dynamic plan optimization, and autonomous anomaly recovery.


### Capability Mixins and Automatic Tool Generation

Each Mixin declares a capability string and a set of `@robot_tool` methods. The framework automatically generates JSON Schema from type annotations and performs capability gating based on the intersection of `api.capabilities` and `env.capabilities` — tools for capabilities unsupported by the hardware are not exposed to the LLM.

```python
class MotionMixin:
    capability = "motion.cartesian"

    @robot_tool(desc="Move to absolute pose (x,y,z[,r]), mm/deg", tags=["motion"])
    def goto_xyzr(self, x: float, y: float, z: float, r: Optional[float] = None) -> None: ...
```

### Dual Tool Strategy

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `build_robot_tools(api)` | Each `@robot_tool` method is independently exposed as an LLM tool | Few tools, ample prompt capacity |
| `RobotControlTool(api)` | Single `robot_control` entry point + `action` field dispatch | Works with SKILL.md workflows, shortens tool list |

Both can coexist. The safety policy layer uniformly unwraps the `action/params` packaging of `robot_control` to perform safety checks on the actual action.

### Three-Layer Safety Rails

- **SafetyRail (Pre-motion Boundary Check)**: Before executing motion tools such as `goto_xyzr`, validates Z limits and XY workspace boundaries. Out-of-bounds commands are rejected immediately with a reason provided to the LLM.
- **RecoveryRail (Automatic Anomaly Recovery)**: On motion/grasp tool failures, automatically attempts homing + end-effector release to bring the hardware back to a known safe state.
- **VisualFeedbackRail (Post-action Visual Verification)**: Captures a camera frame after every motion or grasp, injects it into the agent context for the VLM to verify the action result.

### Visual Perception Service

The visual perception service runs as a subprocess, sharing the GPU with the agent process while maintaining isolated state. `RobotSession` automatically starts the detection subprocess on connect and shuts it down on disconnect; multiple agents can share a single detection instance.

### Multi-Agent Collaboration

`build_robot_agent_config` returns a `SubAgentConfig`, allowing multiple hardware platforms to be orchestrated as sub-agents under a single top-level agent:

```python
left_arm_cfg = build_robot_agent_config(left_session, name="left_arm")
right_arm_cfg = build_robot_agent_config(right_session, name="right_arm")
top_agent = create_deep_agent(..., subagents=[left_arm_cfg, right_arm_cfg])
```

## Contributing

We welcome all forms of contributions, including but not limited to:
- Submitting issues and feature requests
- Improving documentation
- Submitting code
- Sharing usage experiences

## License

This project is licensed under the Apache-2.0 License.
