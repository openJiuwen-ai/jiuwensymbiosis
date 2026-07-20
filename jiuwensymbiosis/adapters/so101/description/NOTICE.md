# SO-101 Description Assets

## Source

These files (`so101_new_calib.urdf`, `*.xml`, `assets/*.stl`, `assets/*.part`)
were copied verbatim from the SO-101 robot description distributed with
[LeRobot](https://github.com/huggingface/lerobot) 0.6.x, obtained via the
RobiAgent repository's `thirdparty/lerobot/so101/description/` tree.

- **Upstream**: HuggingFace LeRobot (Apache License 2.0)
- **Transit repo**: RobiAgent (`thirdparty/lerobot/so101/`) — Apache License 2.0
- **RobiAgent source commit**: `219ddcc41df0063bcc71c63cfd1ab5baa701a6ba`
  (2026-05-08), path `thirdparty/lerobot/so101/description/`

## Local modifications

None. The URDF and mesh files are byte-for-byte copies of the upstream files.

## License

Apache License 2.0 — see the project root `LICENSE` and the upstream
`huggingface/lerobot` LICENSE. The included `gripper_frame_link` is the control
frame this adapter's `RobotKinematics(urdf_path, target_frame_name=...)` is built
against (see `config.py: ik_target_frame`).
