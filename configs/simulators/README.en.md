# Local simulation asset catalog

[中文](README.md)

The catalog scans existing local assets into a rebuildable SQLite index. It does
not download assets, execute asset scripts, launch a simulator, or establish
robot control or physical validity. No Genesis or Torch dependency is required.

## Setup and use

Generate a local configuration, then edit `roots[].path` before building:

```bash
python -m jiuwensymbiosis.simulators.assets init-config
python -m jiuwensymbiosis.simulators.assets build
python -m jiuwensymbiosis.simulators.assets status
python -m jiuwensymbiosis.simulators.assets search --kind environment
python -m jiuwensymbiosis.simulators.assets show library://geniesim/background/kitchen/kitchen_1/background
python -m jiuwensymbiosis.simulators.assets probe library://geniesim/background/kitchen/kitchen_1/background --usd
```

For JSON output without initializing the Agent SDK, run the source directory
entry point, `python jiuwensymbiosis/simulators/assets`, with the same arguments.
This entry point requires only Pydantic and PyYAML. The module entry point may
also emit framework startup logs. Optional `probe --usd` requires `usd-core`.

The data root is selected in this order:

1. `JIUWENSYMBIOSIS_SIMULATOR_HOME`, set before launch to an absolute path (`~` is supported).
2. A source checkout containing both `pyproject.toml` and `configs/simulators/`.
3. `~/.jiuwensymbiosis/simulators/` for an installed wheel.

The default configuration is `configs/simulators/assets.yaml` under that root;
the default index is `outputs/simulator-assets/index.sqlite`. Path resolution
does not create directories. `init-config` copies the packaged YAML example
without overwriting an existing configuration. Installed packages write user
data outside `site-packages`, regardless of the launch directory.

Use `--config /path/to/assets.yaml build` for another build configuration, and
`--index /path/to/index.sqlite search` for queries. Global options precede the
subcommand. Queries read SQLite without reading YAML or scanning directories.
Relative root and index paths are resolved against the YAML directory. Shell
environment variables inside YAML values are not expanded. Rebuild after
changing asset paths, contents, or scan rules; failed builds retain the old index.

## Scan and reference contracts

Supported scanners are `geniesim`, `spiritai`, and configurable `content_tree`.
Each root declares `id`, `path`, `scanner`, and optionally `exclude`. A
`content_tree` root also declares `families`, with `kind`, `min_depth`,
`preview_depth`, and optional first-level `preview_depth_overrides`.

Identifiers use `library://<root-id>/<relative-entry-without-extension>` with
URL-escaped path components. Auxiliary mesh/material directories are excluded;
ambiguous entries are diagnosed rather than guessed. Queries support keyword,
root, kind, category and preview-candidate filtering, with bounded pagination.

Robot USD entries may associate an URDF using a matching `export_manifest.json`
or an unambiguous same-stem URDF. Material bundles use `materials/manifest.json`
to identify the render URDF and SHA-256 hashes. Associated files must remain
inside the configured root. SQLite stores associations, options and checksums;
source asset files remain necessary. `resolve` checks entry locations and file
fingerprints; `probe` additionally validates URDF references and material hashes.

`preview_mode` is candidate metadata (`static_usd` or `urdf`), not a working viewer.
USD probing reports units, axes and unresolved dependencies. Neither indexing
nor probing establishes rendering quality, collision validity or robot control.
USD-to-URDF conversion, workbench integration and continuous simulation workers are outside this delivery.

## USD scenes and optional rendering dependencies

Install CPU scene inspection and surface picking without Genesis/Torch:

```bash
python -m pip install -e ".[simulator-scenes]"
python -m jiuwensymbiosis.simulators.genesis.usd_probe --inventory --output outputs/usd-inventory
```

Build the asset index first. The probe accepts indexed `library://...` IDs and does not
scan or download assets automatically. Inventory mode reads USD metadata without starting
a renderer. With editable source installation, appearance/lighting YAML files are read
from the source package data directory. Generated static scene layers
and reports go to the explicit output directory, preserving source assets.

Use a dedicated Python 3.12 environment for GPU rendering:

```bash
python -m pip install -e ".[genesis]" --extra-index-url https://download.pytorch.org/whl/cu130
python -m jiuwensymbiosis.simulators.genesis.usd_probe --scene <asset-id> --output outputs/usd-preview
```

This pins Torch 2.11.0+cu130, Genesis 1.4.4, Nyx plugin 0.1.9/Nyx 0.1.7 and Pillow 11.3.0.
A supported NVIDIA GPU/driver is required; successful installation is not GPU acceptance.
Do not combine this profile with `full`, `vision-server` or `voice-local` (different Torch),
or `dev`/`remote` (Pillow>=12). CPU tests can use `.[dev,gui,simulator-scenes]`.
A standalone worker and WebRTC backend are available; the browser UI and workbench integration arrive in the next batch.
Source materials and textures are passed directly to the renderer, without material
conversion, texture baking or glass substitution. Exposure and tone mapping affect only
the camera display. Material appearance depends on renderer support and may differ from
the source authoring application.
USD/geometry tests are not physics or robot-task acceptance.

## Standalone runtime and media verification

`GenesisPreview` owns the isolated Genesis/CUDA/Nyx worker. `PreviewStream` owns
one WebRTC viewer lease without binding an HTTP listener. Disconnecting video
keeps physics alive; explicitly call `stop()` and confirm the worker has exited
when the owning host shuts down.

CPU-only media tests can use `pip install -e ".[dev,simulator-scenes,simulator-media]"`
and `python -m pytest tests/unit_tests/simulators/genesis/test_stream.py -q`.
Run `node --test tests/unit_tests/simulators/genesis/metrics.test.mjs` for counters.
The media profile pins aiortc 1.15.0 and av 17.1.0. PyAV's binding license does
not replace the licenses of the selected FFmpeg/codec binaries; inspect those
with `python -m av --version`. This repository does not bundle those binaries.

The worker's `startup` snapshot records phase durations and the actual Quadrants
cache configuration.
GPU acceptance is opt-in through `JIUWEN_GENESIS_INTEGRATION=1`; set
`JIUWEN_GENESIS_EVIDENCE` to an `outputs/` directory to retain images and measurements.
CPU transport, GPU rendering, physical-task success and browser QA are separate checks.
