# Installation and Quick Start

> Category: Tutorial. Initialize a JiuwenSymbiosis Agent without a robot, GPU, or real LLM.

## 1. Prepare Python

Ubuntu 22.04 is the verified platform. Use Python `>=3.11,<3.14`; the SO-101 adapter requires Python 3.12.

```bash
git clone https://gitcode.com/openJiuwen/jiuwensymbiosis.git
cd jiuwensymbiosis
conda create -n jiuwensymbiosis python=3.12
conda activate jiuwensymbiosis
python -m pip install --upgrade pip
python -m pip install -e .
```

## 2. Run the Piper Mock Agent

```bash
python examples/piper_pick_demo.py \
  --config configs/piper/piper.yaml \
  --mock \
  --max-iter 1 \
  --no-visual-feedback \
  --workspace /tmp/jiuwensymbiosis-demo \
  --query "Pick up the black box and place it on the white box."
```

The command uses `MockArmEnv` and the offline `MockModelClient`. It initializes a `RobotSession`, builds the Agent,
loads the built-in skills and tools, performs one model turn, and never connects to CAN, a camera, a detector, or a
model endpoint.

## 3. Verify the result

The final result should contain:

```text
"mock: no real model, task skipped"
```

The process exits with status `0`. This is an Agent wiring smoke test: the fixed offline model intentionally returns a
final answer without calling robot tools, so Mock mode does not claim that physical manipulation succeeded.

## 4. Keep proxy imports safe

HTTP proxy environment variables can break local vLLM and detector calls. In a custom Python entry point, clear them
before importing `openjiuwen` or any module that imports it:

```python
from jiuwensymbiosis.utils.proxy import clear_proxy_env

clear_proxy_env()

# Import openjiuwen or the remaining JiuwenSymbiosis Agent modules below.
```

The bundled CLI and examples already perform this step.

## 5. Install optional capabilities

```bash
# Tests and development tools
python -m pip install -e ".[dev]"

# Vision and GPU dependencies
python -m pip install -e ".[full]" \
  --extra-index-url https://download.pytorch.org/whl/cu128

# Piper SDK
python -m pip install -e ".[piper]"

# SO-101 / LeRobot (Python 3.12)
python -m pip install -e ".[so101]"

# ASR and audio capture
python -m pip install -e ".[voice]"

# Browser GUI
python -m pip install -e ".[gui]"

# Hand-eye calibration
python -m pip install -e ".[calib]"
```

Extras can be combined, for example `.[full,piper]` or `.[full,so101]`; commands containing `[full]` still require the
PyTorch CUDA 12.8 extra index shown above.

## 6. Install version-pinned runtime dependencies

[`requirements.txt`](../../../requirements.txt) pins the project's direct full-runtime dependencies, including the
vision/GPU stack. It is not a complete environment lock: transitive dependencies, development tools, and the Piper SDK
are not all pinned there. To install that runtime set and then the package itself without re-resolving dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Before a first real-hardware run, validate the adapter connection, limits, E-stop, calibration, and workspace. Continue
with [Build Your First Robot Adapter](02-build-first-adapter.md) or the [GUI guide](../how-to/configure-gui.md).
