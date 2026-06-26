---
description: Secret management, subprocess safety, and dependency review for jiuwensymbiosis Python code.
language: chinese
paths:
  - "jiuwensymbiosis/**/*.py"
  - "jiuwensymbiosis/adapters/**/*.py"
alwaysApply: false
---

# Python Security (Extended)

Extends `rules/security.md` with Python-specific guidance (subprocess
safety, dependency review). See `skills/security-review` for the full
checklist.

## Secret Management

All secrets must be loaded from environment variables at runtime:

```python
import os
api_key: str = os.getenv("OPENAI_API_KEY")  # Must be set in production
```

**Never** commit `.env` files. They are already in `.gitignore` — do not
override that.

For test/demo credentials, use the mock paths instead of fake key strings:

```python
# Good — no key needed at all
from jiuwensymbiosis.agent import MockModel
agent = build_robot_agent(..., model=MockModel())
```

## Subprocess Safety (detection sidecar)

`jiuwensymbiosis/serving/` and `adapters/_common/detector_sidecar.py` spawn
the GroundingDINO + SAM2 subprocess. When modifying:

- Do not construct the subprocess command from unsanitized config input.
- Prefer `subprocess.run(..., shell=False)` with argument lists — never
  `shell=True`.
- Validate any config-supplied executable paths against an allowlist or
  resolve them with `shutil.which()`.

## Dependency Security

Before adding a new dependency, especially one with network access:

1. Review the package's own dependencies (PyPI page, GitHub security tab).
2. Run `pip-audit` on the new dependency.
3. Check for known CVEs against `pyproject.toml` transitive deps.

New network-facing dependencies require an additional security review.
Document the review decision in the PR.

## Hardware-Sensitive Code

The following areas are safety-critical — changes require extra review:

- `jiuwensymbiosis/rails/` — motion boundary checks, recovery
- `jiuwensymbiosis/adapters/*/lowlevel.py` — direct hardware I/O, velocity
  and force limits
- `jiuwensymbiosis/serving/` — external model subprocess

In these areas, prefer allowlist over denylist for permissions. Never
bypass `SafetyRail` to call driver motion methods directly.
