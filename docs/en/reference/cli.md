# Command-Line Reference

> Category: Reference. The [Chinese source](../../zh/reference/cli.md) is authoritative. Console entry points are defined in `pyproject.toml`.

## `piper-pick-demo`

```bash
piper-pick-demo --config PATH [--query TEXT | --voice ...] [--mock]
```

`--config` is required. Non-voice execution requires `--query`; the hardware configuration does not contain a default
task. `--mock` uses the offline model and Mock environment.

Common temporary overrides include `--model`, `--server-url`, `--api-key`, `--max-iter`, `--workspace`, `--debug`,
`--no-skill`, and `--no-visual-feedback`. Voice mode supports `--voice`, `--voice-text`, `--voice-audio-file`,
`--voice-once`, `--no-wake`, `--tts`, and `--asr-device`.

## `jiuwensymbiosis-replay`

```bash
jiuwensymbiosis-replay TRACE_JSON [--text]
```

The default mode generates a self-contained HTML replay and prints its path. `--text` prints a terminal timeline and
frame paths without generating the visual report.

## `jiuwensymbiosis-gui`

```bash
jiuwensymbiosis-gui
# Equivalent module entry point:
python -m jiuwensymbiosis.gui
```

Starts the local NiceGUI browser service on `127.0.0.1`. Startup preflight reports the `.[gui]` installation command
when optional dependencies are missing.
