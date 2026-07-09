# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""One-shot skill planner — the single LLM inference of the fast path.

``plan_skills`` makes **one** OpenAI-compatible chat call: it hands the LLM the
user's task plus the available skill catalogue (name + args + description — this
is the "把 skill 描述/skill.md 传入 LLM" step) and asks for a structured, ordered
plan ``[{"skill": ..., "args": {...}}, ...]``. After this single inference the
plan executor (``plan.run_plan``) runs the whole perceive→act closed loop with
no further LLM round-trips.

Endpoint config comes from the same ``ModelSpec`` the slow agent path uses, so
``--server-url`` / ``--model`` overrides apply uniformly.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from typing import Any, cast

from jiuwensymbiosis.agent.fast.sequence import SequenceError, parse_sequence

logger = logging.getLogger(__name__)

_PLANNER_SYSTEM = (
    "你是机器人技能规划器。给定用户的自然语言任务和一组【可用技能】"
    "（每个技能带名称、参数、用途描述），你只做一次规划："
    "**根据任务内容和每个技能的描述，判断需要用到哪些技能、以什么顺序、各自的参数**，"
    "只从清单里真实存在的技能中选取，覆盖完成任务所需的全部步骤；用不到的技能不要列，"
    "也不要臆造清单之外的技能。技能的参数（如要操作的物体名）由你从用户任务里识别后填入。"
    "只输出一个 JSON 数组，不要任何解释或 markdown 代码块。"
    "格式示例（仅示意结构，实际技能名与参数以【可用技能】清单和任务为准）："
    '[{"skill": "<技能名>", "args": {"<参数名>": "<值>"}}, ...]'
)


def _format_skills(available_skills: list[dict[str, Any]]) -> str:
    """Render the skill catalogue as a compact bullet list for the prompt."""
    lines = []
    for s in available_skills:
        args = ", ".join(s.get("args", []))
        lines.append(f"- {s['name']}(args: {args}): {s.get('description', '')}")
    return "\n".join(lines)


def _extract_json_array(text: str) -> list | None:
    """Best-effort parse of a JSON array from an LLM reply (tolerates fences/prose)."""
    if not text:
        return None
    # Strip ```json ... ``` fences if present.
    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    try:
        data = json.loads(candidate)
    except Exception:  # noqa: BLE001 - fall back to bracket extraction
        m = re.search(r"\[.*\]", candidate, re.DOTALL)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except Exception:  # noqa: BLE001 - give up parse; return None
            return None
    return data if isinstance(data, list) else None


def _validate_plan(data: list, available_skills: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only well-formed steps referencing known skills."""
    known = {s["name"] for s in available_skills}
    plan: list[dict[str, Any]] = []
    for step in data:
        if not isinstance(step, dict):
            continue
        skill = step.get("skill")
        args = step.get("args") or {}
        if skill in known and isinstance(args, dict):
            plan.append({"skill": skill, "args": args})
    return plan


def _chat(
    system: str,
    user: str,
    *,
    api_base: str,
    api_key: str,
    model_name: str,
    timeout_s: float,
    temperature: float,
    proxy: str | None,
    attempts: int,
    max_tokens: int,
) -> str:
    """One OpenAI-compatible chat call with retries; returns the reply text.

    Default DIRECT connection — the LLM endpoints used here are domestic and
    reachable directly; a proxy is used ONLY when explicitly passed, never
    auto-picked from the environment. Raises ``RuntimeError`` if all attempts fail.
    """
    import httpx

    url = api_base.rstrip("/").removesuffix("/chat/completions") + "/chat/completions"
    payload = {
        "model": model_name,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    last_exc: Exception | None = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            client_kwargs: dict[str, Any] = {"timeout": timeout_s}
            if proxy:
                client_kwargs["proxy"] = proxy
            with httpx.Client(**client_kwargs) as client:
                resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            return cast(str, resp.json()["choices"][0]["message"]["content"])
        except Exception as exc:  # noqa: BLE001 - retry on transient LLM/HTTP failure
            last_exc = exc
            logger.warning("[planner] attempt %d/%d failed: %s", attempt, attempts, exc)
    raise RuntimeError(f"planner LLM call failed after {attempts} attempts: {last_exc}") from last_exc


def plan_skills(
    query: str,
    *,
    available_skills: list[dict[str, Any]],
    api_base: str,
    api_key: str = "",
    model_name: str,
    timeout_s: float = 20.0,
    temperature: float = 0.0,
    proxy: str | None = None,
    attempts: int = 4,
) -> list[dict[str, Any]]:
    """Single LLM inference → validated ordered skill plan (possibly empty)."""
    user = f"任务：{query}\n\n可用技能：\n{_format_skills(available_skills)}\n\n请输出技能计划 JSON 数组。"
    text = _chat(
        _PLANNER_SYSTEM,
        user,
        api_base=api_base,
        api_key=api_key,
        model_name=model_name,
        timeout_s=timeout_s,
        temperature=temperature,
        proxy=proxy,
        attempts=attempts,
        max_tokens=512,
    )
    data = _extract_json_array(text)
    if data is None:
        logger.warning("[planner] could not parse a plan from reply: %r", text[:200])
        return []
    plan = _validate_plan(data, available_skills)
    logger.info("[planner] task=%r → plan=%s", query, plan)
    return plan


# --------------------------------------------------------------------------- #
# C1 compiler: one LLM call → full action sequence (the fast path's only LLM call)
# --------------------------------------------------------------------------- #
_COMPILER_SYSTEM = (
    "你是机器人技能编译器。给定用户任务和一组【可用技能】（每个技能是一份 SKILL.md，"
    "含其标准 workflow）、一份【可用动作】清单，你只做一次编译：\n"
    "1) 根据任务和各 SKILL.md 的用途，判断需要用到哪些技能；\n"
    "2) 把这些技能的 workflow **按执行顺序展开成一条扁平的动作序列**（多个技能首尾相接成一条）。\n\n"
    "动作序列是一个 JSON 数组，每个元素是一步：\n"
    '  {"op": "<动作名>", "params": {<参数>}, "bind": "<可选:绑定名>"}\n'
    "规则：\n"
    "- op 只能用【可用动作】清单里的名字，或特殊动作 track_detect。\n"
    '- track_detect{"object_name":"<目标>"} 必须带 bind：它实时检测并追踪该目标，'
    '把检测结果绑定到 <bind>；后续步骤用 "<bind>.字段" 读取（如 <bind>.x、<bind>.y、<bind>.z，'
    "或该技能 SKILL.md 指定的其它检测字段）。用哪个字段完全由该技能 SKILL.md 决定，"
    "不要臆造检测没有的字段。\n"
    "- **object_name 只来自用户任务本身**：从用户这句话里识别要操作/检测的目标，用它的自然语言"
    "描述（颜色/形状/大小/类别/材质/位置等任意特征的组合，也可能只有类别），转成英文——照各 "
    "SKILL.md “检测目标来自用户任务”一节。开放词表检测器对英文区分准，中文易把不同目标识别成同一个。"
    "**不要借用示例或 SKILL.md 里出现过的任何具体目标名，目标一律以用户任务为准。**\n"
    "- **bind 与引用必须逐字一致**：bind 是合法标识符（字母/数字/下划线，**不能含空格**），"
    "由 object_name 转写而来——把其中的空格换成下划线。object_name 是自由字符串、bind 是变量名，"
    "用途不同；后续每一处 <bind>.字段 里的 <bind> 都必须和某个 track_detect 的 bind 一模一样，"
    "否则运行时读不到该检测结果。\n"
    "- params 的值可以是数字，或对【已绑定检测】的算术表达式（只允许 + - * / 和 字段/下标），"
    '例如 "obj.z"、"obj.position[0]"（obj 指代你取的某个 bind 名）。目标名等字符串原样写。\n'
    "- **严格照 SKILL.md 的 workflow 表逐行展开，不要自己加额外步骤或偏移**。"
    '若某技能 SKILL.md 明确给了某个偏移的数值范围，按它取一个具体数字写成字面量（如 "obj.z + 30"）；'
    '没写偏移就直接用 SKILL.md 指定的字段（如 "obj.z"），不要臆造 SKILL.md 里没有的符号名。\n'
    "- 不要臆造清单外的动作；不要输出绝对坐标数值（xy/工作高度都来自运行时检测）。\n"
    "只输出这一个 JSON 数组，不要任何解释或 markdown 代码块。"
)


def _format_skills_md(skills_md: Sequence[dict[str, str]]) -> str:
    """Render the candidate skills' full SKILL.md text for the compiler prompt."""
    blocks = []
    for s in skills_md:
        blocks.append(f"### 技能：{s['name']}\n{s.get('markdown', '').strip()}")
    return "\n\n".join(blocks)


def compile_sequence(
    query: str,
    *,
    skills_md: Sequence[dict[str, str]],
    action_vocab: Sequence[str],
    allowed_ops: Any,
    api_base: str,
    api_key: str = "",
    model_name: str,
    timeout_s: float = 90.0,
    temperature: float = 0.0,
    proxy: str | None = None,
    attempts: int = 4,
) -> list[dict[str, Any]]:
    """One LLM inference → a validated action sequence (the C1 fast path).

    The same call that *reads* the candidate SKILL.md to pick skills also *emits*
    their compiled action sequence — there is no separate compile round-trip.

    Args:
        query: the user's natural-language task.
        skills_md: candidate skills as ``[{"name", "markdown"}]`` (full SKILL.md).
        action_vocab: op names the robot exposes (the @robot_tool index keys).
        allowed_ops: collection used to validate the result via ``parse_sequence``.
        api_base/api_key/model_name/...: LLM endpoint config (as ``plan_skills``).

    Returns:
        The validated raw step list ``[{"op", "params", "bind"?}, ...]`` (the
        caller turns it into ``ActionStep`` via ``parse_sequence`` again, or uses
        these dicts directly).

    Raises:
        RuntimeError: if the LLM call fails, or every attempt yields a sequence
            that fails schema validation.
    """
    user = (
        f"任务：{query}\n\n"
        f"【可用技能】(SKILL.md)：\n{_format_skills_md(skills_md)}\n\n"
        f"【可用动作】：{', '.join(action_vocab)}, track_detect\n\n"
        "请输出展开后的动作序列 JSON 数组。"
    )
    last_err: str | None = None
    # Corrective feedback appended to the prompt on the next attempt: re-sampling
    # the identical prompt (temperature is often low) tends to repeat the same
    # mistake, so we tell the model exactly what was wrong and let it self-fix.
    correction = ""
    for attempt in range(1, max(1, attempts) + 1):
        try:
            text = _chat(
                _COMPILER_SYSTEM,
                user + correction,
                api_base=api_base,
                api_key=api_key,
                model_name=model_name,
                timeout_s=timeout_s,
                temperature=temperature,
                proxy=proxy,
                attempts=1,
                max_tokens=1500,
            )
        except RuntimeError as exc:
            # Transient LLM/HTTP failure (e.g. read timeout on a slow generation).
            # _chat raises instead of returning; catch it here so it counts as one
            # attempt and retries, rather than aborting the whole compile on a
            # single slow/failed response. correction is left unchanged.
            last_err = str(exc)
            logger.warning("[compiler] attempt %d: LLM call failed, retrying: %s", attempt, exc)
            continue
        data = _extract_json_array(text)
        if data is None:
            last_err = f"no JSON array in reply: {text[:200]!r}"
            logger.warning("[compiler] attempt %d: %s", attempt, last_err)
            correction = "\n\n上次回复没有包含合法的 JSON 数组。请只输出一个 JSON 数组，不要任何解释或 markdown。"
            continue
        try:
            parse_sequence(data, allowed_ops=allowed_ops)  # validate only; keep raw dicts
        except SequenceError as exc:
            last_err = str(exc)
            logger.warning("[compiler] attempt %d: invalid sequence: %s", attempt, exc)
            correction = (
                f"\n\n上次输出的动作序列无效：{exc}\n"
                "请修正后重新输出完整的 JSON 数组。特别注意：每个 track_detect 都必须带 bind；"
                "后续步骤引用检测字段（如 <bind>.grasp_z、<bind>.position[0]）时，其中的 <bind> "
                "必须与某个 track_detect 的 bind 名逐字一致（用下划线，不含空格；object_name 是另一个独立的自由字符串）。"
            )
            continue
        logger.info("[compiler] task=%r → %d steps: %s", query, len(data), data)
        return data
    raise RuntimeError(f"sequence compiler produced no valid sequence after {attempts} attempts: {last_err}")
