---
description: Credentials, hardware safety, dependency review, and proxy hygiene rules for jiuwensymbiosis.
language: chinese
paths:
  - "jiuwensymbiosis/**/*.py"
  - "configs/**/*.yaml"
alwaysApply: false
---

# Security Rules

> jiuwensymbiosis is a robotics framework with no sandbox / sys_operation /
> prompt-injection surface like agent-core. These rules cover the security
> concerns that *do* apply here: credentials, **physical safety**, dependency
> review, and proxy hygiene.

## Credential Handling

- Never hard-code model API keys, tokens, or real hardware endpoints in
  source files or YAML configs.
- All credentials must come from environment variables or config loaded at
  runtime.
- In tests and the `--mock` demo path, use `MockModel` / `MockDriver` /
  `MockArmEnv` — never real credentials.

## Physical Safety (robotics-specific)

This is jiuwensymbiosis's most important "security" surface — code that
can move real hardware.

- **Never bypass `SafetyRail`**: the Z-floor (`z_min_safe`) and XY
  workspace bounds checks in `jiuwensymbiosis/rails/safety_rail.py` run
  before every `goto_xyzr` / `goto_pose`. Do not call driver motion methods
  directly, skipping the rail.
- **`z_min_safe` is a hard floor**: adapter envs must expose it as a
  property reflecting the real arm's collision limit. Do not set it to a
  permissive value to "make tests pass" on real hardware.
- **`RecoveryRail` homes + releases on failure**: preserve this fallback;
  do not swallow motion exceptions in a way that skips recovery.
- **Velocity / force limits** belong in the driver (`lowlevel.py`), enforced
  at the hardware boundary — not in Python-level "best effort" checks.

## .env and Proxy Hygiene

- `.env` and `.env.*` must not be committed (already gitignored).
- `clear_proxy_env()` **must** be called before `import openjiuwen` in any
  entry point or test. Proxy env vars break local vLLM / detection calls
  by routing localhost through the proxy. The root `conftest.py` does this
  for tests; entry points must do it themselves.

## Dependency Review

- Do not add dependencies without reviewing `pyproject.toml` and the
  security implications.
- New network-facing dependencies (model clients, detector backends)
  require review.
- Run `pip-audit` on the new dependency before merging:

```bash
pip install pip-audit
pip-audit
```

## Security-Sensitive Areas

- `jiuwensymbiosis/rails/` — safety / recovery / visual-feedback rails
  (motion boundary checks, physical safety)
- `jiuwensymbiosis/adapters/*/lowlevel.py` — direct hardware I/O
  (serial/CAN/socket), velocity and force enforcement
- `jiuwensymbiosis/serving/` — detection subprocess (external model
  invocation)

Changes to these areas require extra review and testing. For the full
checklist see `skills/security-review`.
