---
name: python-patterns
description: Python idioms, immutability, async patterns, and anti-patterns for jiuwensymbiosis.
---

# Python Patterns

Reference guide for idiomatic Python in jiuwensymbiosis. Covers patterns
that appear repeatedly in the codebase and establishes conventions for new
code.

## Immutability

Prefer immutable data structures. Mutable state is a source of bugs in
concurrent and async code.

### Frozen Dataclasses

Use `@dataclass(frozen=True)` for data-only objects (cards, event records):

```python
from dataclasses import dataclass, field

@dataclass(frozen=True)
class ToolMeta:
    name: str
    description: str
    capability: str
    tags: tuple[str, ...] = field(default_factory=tuple)
```

Never mutate a frozen dataclass after construction. If you need a modified
copy, use `dataclasses.replace()`:

```python
from dataclasses import replace
updated = replace(original, description="new desc")
```

> jiuwensymbiosis config dataclasses (`<Feature>Config`) are intentionally
> mutable because `from_yaml()` and runtime overrides set fields. Do not
> force `frozen=True` onto existing config classes without auditing call
> sites — the `@robot_tool` decorator and `build_robot_tools` rely on
> reading config fields, not mutating them, so freezing is safe for *new*
> pure-data types but verify per case.

### NamedTuple

Use `NamedTuple` for simple fixed-length records:

```python
from typing import NamedTuple

class GraspResult(NamedTuple):
    success: bool
    centroid_xy: tuple[int, int]
    depth_mm: float
```

### typing.Final

Mark values that should never be reassigned:

```python
from typing import Final

DEFAULT_LOG_FORMAT: Final[str] = "%(asctime)s %(levelname)s %(name)s: %(message)s"
MAX_TRACE_ENTRIES: Final[int] = 1000
```

## Protocol-Based Duck Typing

`jiuwensymbiosis/adapters/_common/protocol.py` already uses `typing.Protocol`
for the `RobotDriver` contract. Follow the same pattern for new structural
interfaces:

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class DetectorClient(Protocol):
    def detect(self, image, prompt: str) -> list: ...
    def segment(self, image, boxes) -> list: ...

# Any class with the right methods satisfies the Protocol — no inheritance
def run_detection(client: DetectorClient, frame) -> list:
    return client.detect(frame, prompt="box")
```

`Protocol` is especially useful for the mixin architecture: capability
mixins (`MotionMixin`, `VisionMixin`, `SuctionMixin`) delegate to
`self.env.<verb>()`, and the env contract is structural — `BaseRobotEnv`
subclasses satisfy it by implementing the right methods, not by inheriting
a shared interface.

## Custom Exception Hierarchy

jiuwensymbiosis currently uses `ValueError` for safety-rail rejections
(intentional — so the LLM can self-correct) and lets hardware errors
propagate from drivers. When adding new exception types, define a
project-wide hierarchy rather than raising bare `Exception`:

```python
class JiuwenSymbiosisError(Exception):
    """Base exception for all framework errors."""
    pass

class SafetyViolationError(JiuwenSymbiosisError):
    """Raised when a motion command violates safety bounds."""
    pass

class AdapterError(JiuwenSymbiosisError):
    """Raised when a hardware adapter operation fails."""
    pass
```

Always catch the most specific exception possible. Never use bare `except:`.
Preserve the `ValueError` convention for rails — that's a deliberate
control-flow signal, not a style violation.

## Context Managers

Use context managers for resource acquisition and release. `RobotSession`
is already a context manager (`__enter__` connects, `__exit__` disconnects,
both idempotent) — follow this pattern for new lifecycle objects:

```python
class DetectorSidecar:
    def __enter__(self) -> "DetectorSidecar":
        self._proc = self._start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._stop(self._proc)
```

For simple cases, use `@contextmanager`:

```python
from contextlib import contextmanager

@contextmanager
def temporary_workspace(base: Path):
    workspace = base / "tmp"
    workspace.mkdir(exist_ok=True)
    try:
        yield workspace
    finally:
        shutil.rmtree(workspace)
```

## Async Patterns

jiuwensymbiosis is mostly synchronous (hardware I/O is blocking), but the
agent loop and some sidecars are async. Follow these patterns consistently
where async appears.

### Running Coroutines Concurrently

```python
import asyncio

results: list = await asyncio.gather(
    *(observe_joint() for _ in range(n))
)
```

### Timeout

```python
import asyncio

try:
    result = await asyncio.wait_for(detect_frame(frame), timeout=5.0)
except asyncio.TimeoutError:
    logger.warning("detection timed out after 5s")
    raise
```

### Async Generators (streaming)

```python
async def stream_steps(agent):
    while True:
        step = await agent.next_step()
        if step is None:
            break
        yield step
```

## Decorators

`@robot_tool` is the project's central decorator — it annotates methods
with `ToolMeta` and lets `build_robot_tools` discover them via MRO. When
writing your own cross-cutting decorators:

```python
import functools

def log_motion(logger):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger.debug("calling %s", func.__name__)
            return func(*args, **kwargs)
        return wrapper
    return decorator
```

Always use `functools.wraps` to preserve the wrapped function's metadata —
`build_robot_tools` inspects `__name__` and type hints to generate the
tool's JSON Schema.

## Package Layout

Follow the existing layout under `jiuwensymbiosis/`:

```
jiuwensymbiosis/
  __init__.py              # Public API exports only
  agent/                   # RobotSession, build_robot_agent, RobotAgentConfig
  api/                     # BaseRobotApi, @robot_tool, capability mixins
  env/                     # BaseRobotEnv, MockArmEnv, KNOWN_CAPABILITIES
  tools/                   # build_robot_tools, RobotControlTool, InProcessCodeTool
  rails/                   # SafetyRail, RecoveryRail, VisualFeedbackRail
  adapters/
    <name>/                # 6-file pattern: config/lowlevel/env/api/session/yaml
    _common/               # Shared adapter utilities
  utils/                   # Proxy hygiene, logging
```

Keep `__init__.py` minimal — re-export only the public API. Internal
implementation details (e.g. `_Formatter`, `_FrameworkFilter` in
`utils/logging.py`) start with `_` and should not be exported.

## Anti-Patterns

### Mutable Default Arguments

```python
# Bad
def add_tool(tools: list = []) -> None:
    tools.append(new_tool)

# Good
def add_tool(tools: list | None = None) -> None:
    if tools is None:
        tools = []
    tools.append(new_tool)
```

### Bare Except

```python
# Bad
try:
    result = await risky_op()
except:
    pass

# Good — preserve ValueError for rail self-correction
try:
    result = await risky_op()
except SafetyViolationError:
    raise
except TimeoutError as e:
    logger.warning("op timed out: %s", e)
```

### Type Checking with type()

```python
# Bad
if type(x) is str:

# Good
if isinstance(x, str):
```

### print() in library code

```python
# Bad
print("connected to arm")

# Good
from jiuwensymbiosis.utils.logging import get_logger
logger = get_logger(__name__)
logger.info("connected to arm")
```

## pyproject.toml Toolchain Configuration

The repo pre-wires formatter/linter/type-checker configs so they work as
soon as the tools are installed. `ruff` is the single tool for both
linting and formatting (Black-compatible drop-in — no separate `black`):

```toml
[tool.ruff]
line-length = 120
target-version = "py311"

[tool.ruff.lint]
select = [
    "E",     # pycodestyle errors
    "W",     # pycodestyle warnings
    "F",     # Pyflakes
    "I",     # isort
    "B",     # flake8-builtins
    "C4",    # flake8-comprehensions
    "UP",    # pyupgrade
    "ASYNC", # flake8-async
]
ignore = [
    "E501",  # line too long (handled by ruff format)
]

[tool.ruff.format]
skip-magic-trailing-comma = false
quote-style = "double"
indent-style = "space"
line-ending = "auto"
docstring-code-format = true
docstring-code-line-length = "dynamic"

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_ignores = true

[tool.pytest.ini_options]
asyncio_mode = "auto"
markers = [
    "unit: no hardware or GPU required",
    "integration: requires real hardware, GPU, or external services",
]
```
