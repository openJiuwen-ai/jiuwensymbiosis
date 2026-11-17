# Simulator

Genesis provides robot scene previews, physics execution and observations for arm motion, gripping and target picking. cuRobo plans arm trajectories, while GraspGenX proposes grasp poses.

## Features

- **Arm motion**: `move_joint` and `move_tool_poses` execute joint or tool-pose targets after model, joint-limit and collision checks.
- **Gripping and placement**: `open_gripper` and `close_gripper` control grippers. Contact, object motion and support observations determine the outcome.
- **Target picking**: `pick_target` combines approach, closing and lifting. Perception mode uses RGB-D and segmentation; ground-truth mode uses simulated object meshes. Both can use GraspGenX.

## Getting started

Install the `genesis`, `genesis-planning` and optional `genesis-graspgenx` extras in a dedicated environment. Prepare cuRobo, robot assets and the GraspGenX inference service separately. Perception-based picking also requires a GroundingDINO/SAM2 service.

Planning examples are in `configs/simulators/planning/`. Set `planning_profile` through Python's `PreviewSettings` or `GenesisPreview.configure`, then create an Agent session after the backend is ready. Available actions depend on the observation mode, robot capabilities and configuration checks.

## Scope

Stop the base before arm planning. Grasping currently supports parallel or coupled grippers with single rigid objects. Closing alone does not confirm a grasp; unconfirmed stopping or payload state blocks subsequent actions.

The MOZ1 configuration targets simulation; Panda profiles are model and planning references. Other robots and physical hardware require separate adaptation and validation.

See the [Genesis backend reference](../reference/genesis-physical-scene.md) for configuration details.
