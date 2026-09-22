"""Share one post-action observation between execution-event consumers."""

from typing import Any

_KEY = "jiuwen_post_action_observation"


def begin_action_observation(ctx: Any) -> None:
    """Called before a tool executes; never reuse a preceding action's frame."""
    ctx.extra.pop(_KEY, None)


def action_observation(ctx: Any, env: Any) -> Any:
    """Capture lazily once; trace, feedback and runtime reuse the same fact."""
    if _KEY not in ctx.extra:
        ctx.extra[_KEY] = env.get_observation()
    return ctx.extra[_KEY]
