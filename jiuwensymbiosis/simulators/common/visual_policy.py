"""Model-facing projection; simulator truth remains private to the host."""

OBSERVATION_POLICY = "head-camera-only-v1"


def visual_history(history: list[dict[str, str]] | None) -> list[dict[str, str]]:
    """Keep only turns produced under this policy, never legacy state-grounded turns."""
    filtered = []
    for item in history or []:
        if (
            item.get("observation_policy") == OBSERVATION_POLICY
            and item.get("role") in {"user", "assistant"}
            and isinstance(item.get("content"), str)
        ):
            filtered.append({"role": item["role"], "content": item["content"]})
    return filtered


def motion_feedback(result: dict) -> dict:
    """Allow outcomes, not poses, targets, contacts, measured errors or free-form diagnostics."""
    reason = result.get("reason")
    if not isinstance(reason, str):
        reason = None
    feedback = {
        "ok": result.get("ok") is True,
        "settled": result.get("settled") is True,
        "recoverable": result.get("recoverable") is True,
        "reason": reason
        if reason in {"reached", "motion_timeout", "cancelled", "cancelled_or_lease_expired"}
        else "unavailable",
    }
    if feedback["recoverable"]:
        feedback["next_observation"] = "capture_camera(head_cam)"
    return feedback


def policy_feedback(result: dict) -> dict:
    """Action outcomes only; joint angles, world height and contact truth stay private."""
    feedback = motion_feedback(result)
    if result.get("reason") in {
        "gripper_requires_open",
        "unsafe_start",
        "policy_timeout",
        "policy_safe",
        "policy_state_invalid",
        "gripper_state_unknown",
        "policy_transition_failed",
        "flight_unconfirmed",
    }:
        feedback["reason"] = result["reason"]
    if result.get("kind") == "jump":
        feedback["airborne"] = result.get("flight_s", 0) >= 0.02
    elif result.get("kind") == "set_gripper":
        feedback.update(side=result.get("side"), closed=result.get("closed"), grasp_verified=False)
    return feedback
