"""Plugin contract and application lifecycle for the installed example GUI."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from jiuwensymbiosis_gui.plugin import GUI_API_VERSION, GuiSpec, LaunchOptions, StartupError

logger = logging.getLogger(__name__)


def describe() -> GuiSpec:
    """Return metadata without importing the server or any UI framework."""
    return GuiSpec(
        key="example",
        display_name="Example HTTP GUI",
        api_version=GUI_API_VERSION,
        launch_target="jiuwensymbiosis_example_gui.plugin:main",
    )


def _read_gui_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StartupError(f"cannot read example GUI JSON config: {path}") from exc
    if not isinstance(value, dict):
        raise StartupError("example GUI config must contain a JSON object")
    return value


def _self_check(options: LaunchOptions) -> int:
    """Exercise plugin selection and Runtime construction without robot config."""
    from jiuwensymbiosis.runtime import Runtime

    runtime = Runtime(workspace=options.workspace)
    try:
        state = runtime.get_runtime_state()
        closing = runtime.close()
    except Exception:
        runtime.close()
        raise
    if not closing.get("closed"):
        logger.critical("Runtime self-check could not confirm clean shutdown: %s", closing)
    result = {
        "gui": "example",
        "mode": "self-check",
        "runtime_closed": bool(closing.get("closed")),
        "busy": bool(state.get("busy")),
        "blocked": bool(state.get("blocked")),
    }
    print(json.dumps(result, sort_keys=True))
    return 0 if result["runtime_closed"] else 1


def main(options: LaunchOptions) -> int:
    """Launch this plugin with the common, normalized launcher options."""
    if options.host != "127.0.0.1":
        raise StartupError("the example GUI only binds to 127.0.0.1")

    gui_config = _read_gui_config(options.gui_config_path)
    mode = gui_config.get("mode", "serve")
    if mode == "self-check":
        return _self_check(options)
    if mode != "serve":
        raise StartupError("example GUI config mode must be 'serve' or 'self-check'")
    if options.config_path is None:
        raise StartupError("serving tasks requires --config with a robot runtime YAML")

    from jiuwensymbiosis.runtime import Runtime
    from jiuwensymbiosis.utils.logging import configure_logging
    from jiuwensymbiosis_example_gui.webapp import serve

    # Reuse the handler that agent startup will configure again for task logging.
    configure_logging()
    runtime = Runtime(workspace=options.workspace)
    try:
        binding = runtime.prepare_binding(options.config_path)
        serve(
            runtime,
            binding,
            host=options.host,
            port=options.port,
            title=str(gui_config.get("title", "JiuwenSymbiosis example GUI")),
            open_browser=options.open_browser,
        )
    finally:
        closing = runtime.close(timeout=5.0)
        while not closing.get("closed") and not closing.get("blocked"):
            logger.error("Runtime is still cleaning up; keeping the plugin process alive")
            closing = runtime.close(timeout=5.0)
        if not closing.get("closed"):
            logger.critical(
                "Runtime cleanup was not confirmed; resources remain blocked. "
                "Do not restart this host automatically. State: %s",
                closing,
            )
            raise RuntimeError("Runtime shutdown is incomplete; inspect cleanup state before restarting")
    return 0
