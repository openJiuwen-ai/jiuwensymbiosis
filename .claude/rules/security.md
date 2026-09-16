---
paths:
  - "jiuwensymbiosis/**/*.py"
  - "jiuwensymbiosis/skills/**/SKILL.md"
  - "scripts/**/*.py"
  - "examples/**/*.py"
  - "templates/**/*.py"
  - "templates/**/*.yaml"
  - "configs/**/*.yaml"
  - "pyproject.toml"
---

# Security and Physical Safety

## Motion and recovery

- Preserve the policy checks on each changed execution path. Calling
  `api.goto_xyzr()`, an `@implements` method, or a tool object directly does
  not itself run `SafetyRail`. Agent callbacks and explicitly invoked
  synchronous policy checks are separate from action implementations.
- Check the applicable capability, envelope, units, and coordinate frame.
  An Env limit of `None` means no range check, not a validated safe range.
  Never invent or relax a hardware limit to make an operation pass.
- Servo paths must validate each target before dispatch using the existing
  synchronous policy entry; drivers retain their hardware-boundary limits.
  Calibration and manual-control paths need their own explicit validation
  and recovery responsibilities, even when they do not use agent callbacks.
- Preserve payload-aware recovery in `jiuwensymbiosis/rails/recovery.py`.
  Do not replace it with unconditional home/release or recover twice after
  another layer has already handled the failure.
- Preserve hand-guiding target resynchronization before torque restoration
  and the end-effector release choice. Calibration publication must retain
  acceptance gates and adapter reload validation; never promote a candidate
  report into a runtime artifact by hand.

## Executable inputs and subprocesses

- `jiuwensymbiosis/tools/inproc_code.py` executes Python in the agent process
  with live objects and no sandbox. Treat changes to its inputs, globals, and
  reachability as trust-boundary changes. Do not assume outer tool callbacks
  inspect or gate every motion performed inside generated code.
- Distinguish task instructions from external tool, document, and perception
  content. Do not treat prompt wording as an enforced execution restriction.
- Build subprocess commands as argument lists with `shell=False`. Check the
  source of executable paths, arguments, and environment. `shutil.which()`
  resolves a path; it does not establish that an executable is trusted.
- Keep detector sidecar startup, failure cleanup, and shutdown owned by
  `RobotSession`; see `jiuwensymbiosis/perception/detector_sidecar.py`.

## Credentials, networking, and persistence

- Load credentials at runtime; do not commit them in source, configs, or
  `.env` files. Device addresses are configuration, not credentials: keep
  deployment-specific values configurable and real devices out of unit tests.
- Preserve `clear_proxy_env()` before importing `openjiuwen`; follow the
  existing entry-point pattern and `tests/conftest.py`.
- Check actual bind addresses and transport requirements for network changes.
  Local loopback HTTP and remote exposure have different trust assumptions.
  Moving a URL literal to evade a checker does not improve transport security.
- Keep credentials out of logs, exception text, tool output, and traces.
  Review persisted frames and observation `extra` fields when adding data.
  A logger or summary field does not automatically redact its contents.
- Review new dependencies and their transitive impact using `pyproject.toml`.
  Run `pip-audit` against the relevant installed environment before merging
  dependency changes; identify scan scope and report unavailable checks.

For an explicit security/safety review, use
[security-review](../skills/security-review/SKILL.md). Its invocation policy is
manual; these rules apply independently of whether that skill is invoked.
