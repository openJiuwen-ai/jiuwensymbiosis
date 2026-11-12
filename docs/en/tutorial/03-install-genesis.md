# Install local simulation

Apple Silicon uses the `genesis-macos` extra, with CPU/Metal physics, OpenGL rendering and software H264.
Use native arm64 Python 3.12. From the repository root:

```bash
uv venv --python 3.12 .venv-genesis
uv pip install --python .venv-genesis/bin/python -e '.[gui,genesis-macos]'
uv pip check --python .venv-genesis/bin/python
.venv-genesis/bin/python -m jiuwensymbiosis_gui --gui workbench
```

Open the simulation page, select the dynamic demo, start simulation and connect video.
Start with 540p / 30 FPS on an M4 with 16 GB. First startup compiles physics kernels.
Set `JIUWEN_GENESIS_MACOS_BACKEND=metal` before launch to compare the Metal backend;
CPU is the validated default. The FPS selection is an upper bound, not a performance guarantee.

Add `jumper` or `microduck` to the installation extras for native MuJoCo policy worlds:

```bash
uv pip install --python .venv-genesis/bin/python -e '.[gui,genesis-macos,jumper,microduck]'
```

Follow [Jumper registration and Agent control (Chinese)](../../zh/how-to/jumper-simulation.md)
or [Microduck registration and official policy (Chinese)](../../zh/how-to/microduck-assets.md).
Assets and weights are downloaded separately and are not included in source releases.
Microduck's executable reference and policy are verified against pinned SHA256 hashes.

Mac Genesis provides the basic demo and observer camera. USD/URDF scenes and their calibrated
robot cameras require the Linux/NVIDIA path; native Jumper/Microduck head cameras work independently.
Native worlds expose RGB observations and relative motion, without RGB-D perception or truth navigation.
AI control additionally needs a model supporting images and tool calls.

Linux currently requires the existing Genesis/NVIDIA initialization, including for native policy worlds.
Use `.[genesis,gui]`, `.[genesis,jumper,gui]` or `.[genesis,microduck,gui]`; see the
[Linux environment and startup commands (Chinese)](../../zh/tutorial/03-install-genesis.md#linux--nvidia).
Keep simulator and test environments separate: the simulator's Pillow/Torch pins differ from `dev`/`full`.
Software H264 selects libx264; binary distributors must review the exact codec build and licenses in
`Open_Source_Software_Notice.txt`. Noncommercial Microduck STL assets need separate permission for commercial distribution.
