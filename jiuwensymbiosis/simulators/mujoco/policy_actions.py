"""Input contracts for simulation-only controls, shared by the worker, tools and workbench."""

from __future__ import annotations

from dataclasses import dataclass
from math import radians
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class NoParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Claw(NoParameters):
    side: Literal["left", "right"] = "left"


class Gripper(Claw):
    closed: bool = False


class ClawAperture(Claw):
    opening: float = Field(default=1, ge=0, le=1, description="夹爪开度，0 闭合，1 完全张开")


class ArmPose(Claw):
    pose: Literal["rest", "thumb_up", "thumb_down", "web_up"] = "rest"


class FiveFootStep(Claw):
    direction: Literal[
        "forward",
        "backward",
        "left",
        "right",
        "turn_left",
        "turn_right",
        "pitch_down",
        "pitch_up",
        "roll_left",
        "roll_right",
        "twist_left",
        "twist_right",
    ]
    duration_s: float = Field(default=1, gt=0, le=3)


class JumperPosture(NoParameters):
    pitch: float = Field(default=0, ge=-radians(20), le=radians(20), description="俯仰 rad，正值低头")
    roll: float = Field(default=0, ge=-radians(15), le=radians(15), description="横滚 rad")
    twist: float = Field(default=0, ge=-radians(30), le=radians(30), description="躯干扭转 rad")
    height: float = Field(default=0.10647, ge=0.07, le=0.15, description="机身高度 m")


class DuckHead(NoParameters):
    neck_pitch: float = Field(default=0, ge=-1.1, le=1.1, description="颈部俯仰 rad")
    head_pitch: float = Field(default=0, ge=-1.1, le=1.1, description="头部俯仰 rad")
    head_yaw: float = Field(default=0, ge=-1.4, le=1.4, description="头部偏航 rad")
    head_roll: float = Field(default=0, ge=-0.31, le=0.31, description="头部横滚 rad")


class DuckBody(NoParameters):
    x: float = Field(default=0, ge=-0.02, le=0.02, description="前后偏移 m")
    y: float = Field(default=0, ge=-0.02, le=0.02, description="横向偏移 m")
    z: float = Field(default=0, ge=-0.03, le=0.03, description="高度偏移 m")
    roll: float = Field(default=0, ge=-radians(30), le=radians(30), description="横滚 rad")
    pitch: float = Field(default=0, ge=-radians(30), le=radians(30), description="俯仰 rad")
    yaw: float = Field(default=0, ge=-radians(30), le=radians(30), description="偏航 rad")


class Skate(NoParameters):
    forward: float = Field(default=0.6, ge=-0.5, le=0.6, description="前进速度 m/s，负值制动")
    heading: float = Field(default=0, ge=-1, le=1, description="航向偏差 rad")
    duration_s: float = Field(default=1, gt=0, le=5)


class Mouth(NoParameters):
    opening: float = Field(default=0, ge=0, le=1, description="喙部开度，0 闭合，1 完全张开")


class Rolls(NoParameters):
    repetitions: int = Field(default=1, ge=1, le=10, description="连续前滚翻次数")


@dataclass(frozen=True)
class PolicyAction:
    label: str
    parameters: type[NoParameters] = NoParameters


RECORDINGS = {
    "dance_crab": "螃蟹舞",
    "dance_brazilian": "巴西舞",
    "dance_maze": "迷宫舞",
    "dance_dream_wings": "梦的翅膀",
    "dance_waist": "扭腰舞",
    "gesture_hello": "挥手问好",
    "gesture_bow": "鞠躬",
    "gesture_paw": "伸爪",
    "gesture_salute": "敬礼",
}
JUMPER_ACTIONS = {
    **{name: PolicyAction(label) for name, label in RECORDINGS.items()},
    "jump": PolicyAction("固定轨迹跳跃"),
    "ref_free_jump": PolicyAction("无参考跳跃"),
    "set_gripper": PolicyAction("开合夹爪", Gripper),
    "claw_aperture": PolicyAction("指定夹爪开度", ClawAperture),
    "claw_pose": PolicyAction("五足支撑·前臂姿态", ArmPose),
    "claw_step": PolicyAction("五足支撑·移动或倾身", FiveFootStep),
    "stow_claw": PolicyAction("收回已张开的夹爪"),
    "jumper_posture": PolicyAction("调整躯干姿态", JumperPosture),
}
ARENA_GAITS = {
    "sprint_2m": "向前跑两米",
    "hill_climb_1m5": "爬坡步态",
    "sock_heist_6m": "走停挑战步态",
    "roller_sprint_2m": "轮滑短跑步态",
}
DUCK_ACTIONS = {
    "mouth": PolicyAction("开合鸟喙", Mouth),
    **{name: PolicyAction(label) for name, label in ARENA_GAITS.items()},
    "roller_gait": PolicyAction("使用标准轮滑步态"),
    "got_a_talent": PolicyAction("才艺秀·官方二十秒坐立编舞"),
    "head_pose": PolicyAction("转动颈部和头部", DuckHead),
    "body_pose": PolicyAction("调整身体姿态", DuckBody),
    "sit": PolicyAction("坐下"),
    "stand": PolicyAction("站起"),
    "recover_stance": PolicyAction("跌倒后起身"),
    "ground_pick": PolicyAction("低头啄取地面并站起"),
    "kick_left": PolicyAction("左脚踢球"),
    "kick_right": PolicyAction("右脚踢球"),
    "roulade": PolicyAction("前滚翻", Rolls),
    "velstand_gait": PolicyAction("使用 velstand 行走和站立"),
    "classic_gait": PolicyAction("使用经典行走和站立策略"),
    "roller_crouch": PolicyAction("轮滑俯身滑行"),
    "roller_skate": PolicyAction("轮滑前进、转向或制动", Skate),
}
ACTIONS = {**JUMPER_ACTIONS, **DUCK_ACTIONS}


def validate_action(kind: str, values: dict) -> dict:
    if not isinstance(kind, str) or kind not in ACTIONS:
        raise ValueError("Unsupported simulation policy action")
    return ACTIONS[kind].parameters.model_validate(values).model_dump()


def action_controls(names) -> list[dict]:
    return [
        {"name": name, "label": ACTIONS[name].label, "parameters": ACTIONS[name].parameters.model_json_schema()}
        for name in names
    ]


def action_wait_s(kind: str) -> float:
    # The longest published recording is 235.68 s. Allow real-time playback and slower rendering.
    return 600 if kind in RECORDINGS else 100
