# Example GUI plugin

This package is a minimal third-party GUI plugin example. It registers the
`example` entry point, declares its `GuiSpec` without loading its application,
and implements the `main(LaunchOptions)` lifecycle with the public `Runtime`
facade. Its sample page uses only Python's standard-library HTTP server; it does
not depend on the workbench or NiceGUI.

From the repository root, install the core development package and this plugin:

```bash
python -m pip install -e ".[dev]"
python -m pip install -e examples/gui_plugin
python -m jiuwensymbiosis_gui --list-guis
```

Run the no-hardware plugin and Runtime lifecycle self-check:

```bash
python -m jiuwensymbiosis_gui --gui example \
  --gui-config examples/gui_plugin/jiuwensymbiosis_example_gui/configs/self-check.json \
  --workspace /tmp/jiuwensymbiosis-example-gui --no-browser
```

To serve a task page, replace `--gui-config` with a real robot `--config`:

```bash
python -m jiuwensymbiosis_gui --gui example \
  --config configs/piper/piper.yaml --workspace /tmp/jiuwensymbiosis-example-gui
```

The task page is bound only to `127.0.0.1`. A browser request with a different
`Origin` or `Host` is rejected. The UI-specific JSON file uses `--gui-config`;
it does not supply or emulate robot hardware configuration. Actual tasks require
a valid `--config` and the selected adapter's runtime dependencies.

The browser creates a `request_id` for each task submission and reuses it if the
HTTP response is lost, allowing Runtime to deduplicate a retry. The host checks
the result of `Runtime.close()` before exiting; if cleanup remains unconfirmed,
it reports an error and must not be configured for automatic restart.

The developer guide is [Adding a GUI plugin](../../docs/en/how-to/add-gui.md)
and [添加 GUI 插件](../../docs/zh/how-to/add-gui.md).
