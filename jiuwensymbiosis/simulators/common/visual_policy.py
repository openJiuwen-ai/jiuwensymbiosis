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
        if reason in {"reached", "motion_timeout", "cancelled", "cancelled_or_lease_expired", "recovery_required"}
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
        "kick_unconfirmed",
        "roll_unconfirmed",
        "skate_unconfirmed",
        "seat_unconfirmed",
        "sprint_short",
        "stand_required",
        "policy_unavailable",
    }:
        feedback["reason"] = result["reason"]
    if result.get("kind") in {"jump", "ref_free_jump"}:
        feedback["airborne"] = result.get("flight_s", 0) >= 0.02
    elif result.get("kind") == "set_gripper":
        feedback.update(side=result.get("side"), closed=result.get("closed"), grasp_verified=False)
    elif result.get("kind") == "claw_aperture" and result.get("ok"):
        feedback.update(side=result.get("side"), opening=result.get("opening"), grasp_verified=False)
    elif result.get("kind") == "sprint_2m" and feedback["settled"] and "forward_displacement_m" in result:
        feedback["forward_displacement_m"] = result["forward_displacement_m"]
    return feedback


def motion_abort_message(action: str, result: dict) -> str:
    """Describe a safety stop using known simulator outcomes, never raw backend text."""
    reason = result.get("reason")
    details = {
        "recovery_required": "检测到机器人姿态不稳，需要恢复站姿",
        "stand_required": "机器人尚未站起",
        "motion_timeout": "动作超时",
        "policy_timeout": "策略动作超时",
        "unsafe_start": "机器人起始姿态未达到动作要求",
        "cancelled": "动作收到停止请求",
        "cancelled_or_lease_expired": "动作收到停止请求或控制租约失效",
        "policy_unavailable": "当前仿真场景没有加载该策略",
        "stop_unconfirmed": "无法确认机器人已停稳",
    }
    cause = details.get(reason, "仿真未确认动作成功") if isinstance(reason, str) else "仿真未确认动作成功"
    stopped = "机器人已停稳" if result.get("settled") is True else "尚未确认机器人停稳"
    return f"{action}时{cause}，本轮已中止。{stopped}。"
