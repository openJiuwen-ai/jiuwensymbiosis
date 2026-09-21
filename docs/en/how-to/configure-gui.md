# Configure and Use the GUI

> Category: How-to. The [Chinese source](../../zh/how-to/configure-gui.md) is authoritative.

The **JiuwenSymbiosis GUI** is a visual console in the browser: no command line needed — pick a body, choose a task,
edit the configuration, and run it with one click, with an execution view clear enough for non-developers.

![GUI home: six tabs, body selection with capability tags, task cards and run/configure buttons](../../images/gui-home.png)

The tabs across the top switch between **Home / Configuration / Run / Tools / History / Settings**. **Home**, shown above, is
the main entry point: pick the **body** from the dropdown on the left (e.g. *Piper 6-axis arm*), select a **task** card
in the middle, and act on the current task with "▶ Run / ⚙ Configure" below — one click on Run starts the task.

---

## Contents

- [1. Feature overview](#1-feature-overview)
- [2. Install](#2-install)
  - [Install GUI dependencies](#install-gui-dependencies)
  - [Select a GUI plugin](#select-a-gui-plugin)
  - [Install a desktop launcher (recommended)](#install-a-desktop-launcher-recommended)
- [3. Run examples](#3-run-examples)
  - [Run a task](#run-a-task)
- [4. Other features](#4-other-features)
  - [Inspect each step](#inspect-each-step)
  - [Replay history](#replay-history)
  - [Automatic error diagnosis and one-click fixes](#automatic-error-diagnosis-and-one-click-fixes)
- [5. Notes](#5-notes)

---

## 1. Feature overview

The interface has six pages (switched by the top tabs):

| Page | Purpose |
|---|---|
| **Home** | Pick the body (e.g. *Piper 6-axis arm*) and a task card, then act on the current task with "▶ Run / ⚙ Configure". |
| **Configuration** | Common fields grouped by category as forms + a "raw YAML" fallback (two-way synchronized). |
| **Run** | Live monitoring: camera view + a one-line current action + a step-by-step timeline on the right (click a step for the raw tool call/arguments); a bottom drawer holds raw logs, safety events, and error diagnosis. |
| **Tools** | Task-independent operations: perception testing (click-to-reproject), a four-step hand-eye wizard (requires `pip install -e ".[calib]"`), and hardware control for releasing torque, hand-posing the arm, and restoring it (only for drivers implementing `HandGuidingDriver`). |
| **History** | Lists recorded execution traces and **replays** them in the browser with the built-in self-contained HTML (camera frames included). |
| **Settings** | Where run records are stored (the workspace directory), and the UI language. |

> The interface uses [NiceGUI](https://nicegui.io) in **browser mode**, listening only on `127.0.0.1` and never using
> `native=True` — so it needs no system-level display library and has almost no installation barrier.

---

## 2. Install

### Install GUI dependencies

The existing workbench is independent of the heavier GPU stack and installs on its own (it needs `nicegui` + `pillow`).
In your conda environment (default name `jiuwensymbiosis`):

```bash
pip install -e ".[gui]"
```

> If the selected workbench is missing NiceGUI, startup preflight shows a dialog suggesting
> `pip install -e ".[gui]"` (visible even from a desktop icon with no terminal).

### Select a GUI plugin

The unified launcher defaults to the existing `workbench`. List the installed GUI plugins, or select one explicitly:

```bash
jiuwensymbiosis-gui --list-guis
jiuwensymbiosis-gui --gui workbench
python -m jiuwensymbiosis_gui --gui workbench
```

To select another plugin, install its distribution and documented dependencies in the same Python environment, then run
`jiuwensymbiosis-gui --gui <plugin-key>`. Each plugin owns its UI framework and front-end dependencies; installing a
plugin does not make its UI framework a dependency of every other GUI. The launcher starts only the selected plugin.

To build another interface, see [Add a GUI Plugin](add-gui.md) for an installable example and testing rules.

`python -m jiuwensymbiosis.gui` remains as a temporary compatibility shim. New launch scripts should use
`python -m jiuwensymbiosis_gui` or `jiuwensymbiosis-gui`.

### Install a desktop launcher (recommended)

```bash
bash scripts/install_desktop_entry.sh
```

Afterwards, search the system's **application menu / activities list** for **"Jiuwen Symbiosis"** to open it, or right
click to **pin it to the dock**. The icon and launch paths are generated dynamically from the real local paths, so
switching machines or clone directories needs no edit.

Uninstall:

```bash
bash scripts/install_desktop_entry.sh --uninstall
```

> The desktop icon launches through `scripts/launch_gui.sh`, which activates the conda environment `jiuwensymbiosis`
> automatically (set `JIUWEN_CONDA_ENV=<your-env-name>` if yours is named differently) and runs the repository's **live
> source** — editing code or upgrading (`git pull` / switching branches) takes effect the next time you open it, with
> **no reinstall and no uninstall first**. Only moving or renaming the repository directory requires re-running the
> install script so the paths regenerate.

---

## 3. Run examples

Start it any of these ways (all open the default browser at `http://127.0.0.1:8770`):

```bash
# 1) Desktop icon: click "Jiuwen Symbiosis" in the application menu
# 2) Console script
jiuwensymbiosis-gui --gui workbench
# 3) Module entry point
python -m jiuwensymbiosis_gui --gui workbench
# 4) Launcher script (activates the conda environment, runs repository source)
bash scripts/launch_gui.sh
```

### Run a task

1. On startup you land on **Home**, with the body defaulting to *Piper 6-axis arm* and the task to "pick up the box".
   Choose your body and task.
2. Go to **Configuration** and fill in the real endpoints and hardware parameters: the model `api_base` / `api_key`,
   the wrist camera serial, the CAN interface, and so on (or switch to "raw YAML" and edit it wholesale).
3. Confirm the hardware prerequisites: CAN activated, the wrist RealSense connected, and the vision detection models
   (GroundingDINO + SAM2) available locally. For hardware and calibration details see the repository root `README` and
   `configs/piper/`.
4. Under "Configuration → task instruction", enter the action you want the agent to perform, e.g. "place the black box
   on the white box". You do not need to spell out the pick/place steps or height math — the built-in SKILL.md files
   decide those.
5. Select **▶ Run**. On common problems such as a missing vision model, the interface gives you an **error diagnosis +
   one-click fix** (see [4.3](#automatic-error-diagnosis-and-one-click-fixes)) rather than running blind until timeout.

![GUI run page with camera view, step-by-step timeline, and success state](../../images/gui-run.png)

---

## 4. Other features

### Inspect each step

Every step in the **Run** page's right-hand timeline can be expanded: it shows that step's **raw tool name, arguments,
AI explanation, duration, and return value / error**. Clicking a historical step also switches the camera panel back to
**the frame from that moment** ("↩ back to live view" returns to live).

### Replay history

The **History** page scans the workspace's `traces/` directory, lists a summary of each run, and "🌐 open replay in
browser" reviews the whole execution trace using the self-contained HTML (with inlined camera frames).

> Replay depends on execution traces: enable `enable_tracing: true` in the `agent` block under **Configuration**
> (off by default, zero overhead). Once on, trace JSON lands in `<workspace>/traces/`. The workspace location can be
> changed on the **Settings** page.

```yaml
agent:
  enable_tracing: true
  trace_save_frames: true
```

### Automatic error diagnosis and one-click fixes

When a run fails, the bottom "run details" drawer switches automatically to the **error diagnosis** page, translating
the technical error into one plain sentence + the steps to handle it, and offering a **one-click fix** for common
problems:

- **Vision detection model not ready / download timed out**: either "auto-detect" a directory of already-downloaded
  GroundingDINO / SAM2 and other vision models and fill it in directly, or "switch in one click" to a domestic mirror
  and re-download.
- Other recognized cases: detector startup failure / occupied port, model authentication failure (API key), inability to
  reach the model service, insufficient GPU memory, robot connection failure — each with a diagnosis and a suggestion.

Diagnostics do not bypass robot safety or silently rerun a motion. Review the proposed action and hardware state before
trying again.

---

## 5. Notes

- **Port**: `8770` by default, bound to the loopback interface only, never exposed externally.
- **conda environment**: `launch_gui.sh` and the desktop icon default to `jiuwensymbiosis`; override with
  `JIUWEN_CONDA_ENV`.
- **Language**: the interface is Simplified Chinese.
- The workbench is an in-process wrapper over library functions (it does not shell out to the CLI); live feedback comes
  back through a thread-safe event queue. Its Python modules live under `jiuwensymbiosis_gui.workbench`; hardware-free
  workbench tests live in `tests/gui/workbench/`.
