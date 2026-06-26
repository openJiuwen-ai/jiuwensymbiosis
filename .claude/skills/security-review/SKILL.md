---
name: security-review
description: Security & safety checklist for jiuwensymbiosis — secrets, physical safety, subprocess, dependencies, logging.
disable-model-invocation: true
---

# Security & Safety Review

Checklist for jiuwensymbiosis. Run through all categories before any
security- or safety-sensitive change or PR.

> jiuwensymbiosis has **no** sandbox / sys_operation / SQL / prompt-injection
> surface. The checklist below covers what actually applies here: secrets,
> **physical safety** (the most important surface for a robotics framework),
> subprocess safety, dependencies, and log hygiene.

See `.claude/rules/python/security.md` for tool-specific guidance
(subprocess safety, dependency review). See `.claude/rules/security.md`
for credential and proxy rules.

## 1. Secrets Management

**Rule:** Credentials never enter source code or YAML configs.

- [ ] No model API keys, tokens, or passwords hardcoded in `.py` / `.yaml`
- [ ] All secrets loaded from environment variables via `os.getenv()`
- [ ] `.env` files not committed (already in `.gitignore` — do not remove)
- [ ] Test / demo paths use `MockModel` / `MockDriver` instead of fake keys
- [ ] No real hardware device paths or endpoints in test fixtures

```python
# Bad
OPENAI_API_KEY = "sk-1234567890abcdef"

# Good
import os
api_key: str = os.getenv("OPENAI_API_KEY")  # set in environment

# Best for tests — no key needed
from jiuwensymbiosis.agent import MockModel
agent = build_robot_agent(..., model=MockModel())
```

## 2. Physical Safety (robotics-specific — highest priority)

**Rule:** Motion commands never bypass `SafetyRail`; safety bounds reflect
real hardware limits.

- [ ] No driver motion method (`goto_xyzr`, `goto_pose`, `move_joint`)
  called directly outside the rail-gated path
- [ ] `SafetyRail` Z-floor (`z_min_safe`) and XY workspace bounds enforced
  before every motion
- [ ] `z_min_safe` on env subclasses reflects the **real** arm collision
  limit, not a permissive value
- [ ] `RecoveryRail` recovery path (home + release) preserved — exceptions
  not swallowed in a way that skips recovery
- [ ] Velocity / force limits enforced in `lowlevel.py` at the hardware
  boundary, not as Python "best effort" checks
- [ ] `VisualFeedbackRail` frame capture not bypassed when enabled in
  config

```python
# Bad — bypasses safety rail, can crash the arm
driver.goto_xyzr(x, y, z, r)

# Good — goes through the rail-gated tool path
api.goto_xyzr(x=x, y=y, z=z, r=r)
```

## 3. Subprocess Safety (detection sidecar)

**Rule:** The GroundingDINO + SAM2 subprocess is constructed safely.

- [ ] `subprocess.run(..., shell=False)` with argument lists — never
  `shell=True`
- [ ] Executable paths from config resolved with `shutil.which()` or
  validated against an allowlist
- [ ] No unsanitized user/config input concatenated into the command
- [ ] Sidecar lifecycle managed by `RobotSession` (started in `__enter__`,
  stopped in `__exit__`)

## 4. Dependency Security

**Rule:** All dependencies scanned before merging PRs.

- [ ] New dependencies reviewed for known CVEs: `pip-audit`
- [ ] New network-facing dependencies (model clients, detector backends)
  reviewed for security implications
- [ ] Review decision documented in the PR

```bash
# Run before merging dependency changes
pip-audit
```

## 5. Sensitive Data in Logs & Traces

**Rule:** Logs and execution traces must not expose credentials or sensitive
data.

- [ ] No API keys, tokens, or passwords in log output
- [ ] Execution traces (`ExecutionTrace` observation snapshots) contain no
  raw arrays / secrets — only pose/joints/extra summaries
- [ ] `TraceLogHandler` captures `WARNING`+ from `trace_capture_loggers`
  only — verify new loggers added there are safe to persist
- [ ] `get_logger(name)` used instead of `print()` — so `TraceLogHandler`
  and file handlers attach correctly

```python
# Bad — token in log
logger.info(f"Authenticated with token {token}")

# Good
logger.info("Authenticated", extra={"user_id": user_id})
```

## Pre-Review Checklist

Before marking a safety- or security-sensitive PR as ready for review, run
through all categories above. Document the review in the PR description:

```
Security & Safety Review
=======================
Secrets:         PASS (no hardcoded credentials)
Physical Safety: PASS (SafetyRail enforced, z_min_safe verified)
Subprocess:      PASS (shell=False, paths validated)
Dependencies:    PASS (pip-audit clean)
Log/Trace:       PASS (no secrets in logs or traces)
```

For changes to `jiuwensymbiosis/rails/`, `jiuwensymbiosis/adapters/*/lowlevel.py`,
or `jiuwensymbiosis/serving/`, request a dedicated review from a second
reviewer — these areas can move real hardware.
