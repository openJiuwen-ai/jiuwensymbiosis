# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Two-tier task planning — compose skills if the library covers the task, derive
an action sequence from the contracts if it does not.

``plan_task`` is the entry point. Tier 1 (``compile_sequence``) both *picks* the
skills and *expands* them in **one** inference, so the common path costs exactly
one round trip — the same as before there was a second tier. Tier 2
(``compose_actions``, no SKILL.md at all) is a second call, paid only when Tier 1
declines or its output cannot be repaired. Both tiers emit the same thing — a flat
action sequence that ``parse_sequence`` has validated — so the runner sees no
difference, and a Tier 2 sequence that ran successfully is exactly what a new skill
would be distilled from.

Neither tier is told an order. Every action and skill carries its own
pre-conditions and effects (``api.state``), rendered into the prompt as data; the
planner derives an order that satisfies them and the validator rejects one that
does not, feeding the failure back for self-repair. That is why nothing here names
a specific action — a new body with different actions needs no change in this file.

Endpoint config comes from the same ``ModelSpec`` the slow agent path uses, so
``--server-url`` / ``--model`` overrides apply uniformly.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from jiuwensymbiosis.agent.cancel import CancelToken, RunCancelled, cancellable_call
from jiuwensymbiosis.agent.fast.sequence import SequenceError, parse_sequence, qualifiers_for
from jiuwensymbiosis.perception.scene3d import SPATIAL_RELATIONS

logger = logging.getLogger(__name__)


def _format_effects(entry: Mapping[str, Any]) -> str:
    """Render one action's / skill's pre-conditions and effects as a compact clause."""
    bits = []
    for key, label in (("requires", "需要"), ("provides", "产生"), ("invalidates", "作废")):
        values = entry.get(key)
        if values:
            bits.append(f"{label}{list(values)}")
    if entry.get("produces_location"):
        bits.append("产生位置")
    if entry.get("consumes_location"):
        bits.append("需已有位置")
    if entry.get("invalidates_locations"):
        bits.append("使已有位置全部失效")
    return " ".join(bits)


def format_action_contracts(action_index: Mapping[str, Any]) -> str:
    """Render every action's full contract — params, result fields, pre-conditions, effects.

    This is what lets a planner order actions itself: it can see that an action
    needs a location and which actions produce one, without any of that being
    written down as a workflow.

    The name / params / result / contract come from the shared ``ActionSpec`` and mean
    the same thing on every body. Two things are rendered apart from them on purpose:
    the **capability gate** (why this action is present here, hence whether a plan using
    it would survive being moved) and the **body note** (implementation detail true of
    this robot only). A reader that cannot tell those apart cannot write a skill that
    transfers.
    """
    from jiuwensymbiosis.api.decorators import ToolMeta

    lines: list[str] = []
    for name in sorted(action_index):
        meta = getattr(action_index[name], "__robot_tool__", None)
        if not isinstance(meta, ToolMeta):
            lines.append(f"- {name}()")
            continue
        props = (meta.input_params or {}).get("properties") or {}
        required = set((meta.input_params or {}).get("required") or ())
        sig = ", ".join(f"{p}{'' if p in required else '?'}" for p in props)
        fields = sorted(meta.result_fields())
        head = f"- {name}({sig})"
        if fields:
            head += " -> {" + ", ".join(fields) + "}"
        if meta.capability:
            head += f"  [能力门 {meta.capability}]"
        lines.append(head)
        effects = _format_effects(
            {
                "requires": meta.requires,
                "provides": meta.provides,
                "invalidates": meta.invalidates,
                "produces_location": meta.produces_location,
                "consumes_location": meta.consumes_location,
                "invalidates_locations": meta.invalidates_locations,
            }
        )
        detail = "; ".join(x for x in (meta.description, effects) if x)
        if detail:
            lines.append(f"    {detail}")
    return "\n".join(lines)


def _format_capabilities(api_capabilities: Any) -> str:
    """Render the robot's capability set as a prompt line ('' when absent).

    Lets the planner/compiler resolve capability-gated SKILL.md steps: a step
    written "若你的 api_capabilities 含 X 则…" expands only when X is listed here.
    """
    if not api_capabilities:
        return ""
    return "【机器人能力】(api_capabilities)：" + ", ".join(sorted(api_capabilities)) + "\n\n"


def _format_reach_prior(prior: dict) -> str:
    """Render the body's URDF-estimated reach envelope — a no-target planning prior."""
    if not prior.get("reachable"):
        return "【本体可达域】(URDF 估算) 当前站姿下几乎无处可达——抓取前必须先移动/调整躯干靠近目标。"
    f = prior.get("forward_m") or [0.0, 0.0]
    la = prior.get("lateral_m") or [0.0, 0.0]
    h = prior.get("height_m") or [0.0, 0.0]
    return (f"【本体可达域】(URDF 估算，当前站姿下双臂大致可达范围)：前向 {f[0]:.2f}~{f[1]:.2f}m、"
            f"横向 {la[0]:.2f}~{la[1]:.2f}m、高度 {h[0]:.2f}~{h[1]:.2f}m。超出此范围的目标须先移动靠近再抓。")


def _format_scene(scene: Any) -> str:
    """Render the pre-plan scene perception as a prompt block ('' when empty).

    Lets LLM② decide skills from perception (e.g. target far → plan a move first,
    多个目标 → 编循环). ``scene`` is the analyze_scene-shaped dict
    ``{count, objects:[...], reach_prior?}``. ``reach_prior`` (no-target body reach envelope)
    is rendered even with zero detections so URDF still informs planning.
    """
    if not isinstance(scene, dict):
        return ""
    objs = scene.get("objects") or []
    refs = scene.get("references") or []
    missing = scene.get("missing") or []
    unqual = scene.get("unqualified") or []
    prior = scene.get("reach_prior")
    if not any((objs, refs, missing, unqual)) and not isinstance(prior, dict):
        return ""

    def _one(i: int, obj: dict) -> str:
        name = obj.get("object", "?")
        dist = obj.get("distance_mm")
        c = obj.get("center_mm")
        dist_s = f"{float(dist):.0f}mm" if isinstance(dist, (int, float)) else "?"
        c_s = f"[{c[0]:.0f},{c[1]:.0f},{c[2]:.0f}]" if isinstance(c, (list, tuple)) and len(c) == 3 else "?"
        r = obj.get("reachable")
        reach_s = "" if r is None else f", 可达={'是' if r else '否'}"
        return f"  - #{i} {name}: 距离 {dist_s}, base系中心 {c_s}mm{reach_s}"

    lines: list[str] = []
    if objs:
        lines.append(f"【场景感知】(规划前检测到 {len(objs)} 个目标实例，按最近优先)：")
        lines.extend(_one(i, obj) for i, obj in enumerate(objs))
    if refs:
        # Referenced objects are NOT what the task grasps — say so, or the planner will happily
        # plan to pick up the drawer the apple is in.
        lines.append("【参照物】(用于定位目标，不是抓取对象)：")
        lines.extend(_one(i, obj) for i, obj in enumerate(refs))
    if missing:
        # "Looked for and not found" ≠ "not there": the body may simply not be facing it. Saying
        # which is which is what lets the planner choose to search instead of reaching blindly.
        lines.append(f"【未见】规划前找过但当前视野内没有：{', '.join(missing)}（可能只是没朝向它）")
    for u in unqual:
        # Report the rejected candidate rather than deleting it. The geometry test can be wrong
        # (a drawer open just a crack, a tight containment margin); handing the planner the
        # measurement lets it overrule us, whereas silence would leave it planning against a
        # scene we quietly edited.
        near = u.get("nearest_mm")
        near_s = f"，最近的距离 {float(near):.0f}mm" if isinstance(near, (int, float)) else ""
        lines.append(
            f"【限定不符】视野里有 {u.get('count')} 个 {u.get('object')}{near_s}，"
            f"但没有一个满足任务的限定「{u.get('relation')} {u.get('reference')}」。"
        )
    if isinstance(prior, dict):
        lines.append(_format_reach_prior(prior))
    return "\n".join(lines) + "\n\n"


def _format_grounding(grounding: Mapping[str, Any] | None) -> str:
    """Render the task's own spatial qualifiers ('' when it stated none).

    LLM① already read these off the task and every step is validated against them, so
    leaving them out of the prompt makes LLM② derive the same thing a second time from
    the same sentence — and the only way it learns its answer differs is by being
    rejected, one round trip per disagreement. Rendering what was already decided costs
    a few lines and removes the second derivation entirely.
    """
    if not grounding:
        return ""
    lines = [
        f"  - {name}：{q['relation']} {q['reference']}"
        for name in sorted(grounding)
        for q in qualifiers_for(grounding, name)
    ]
    if not lines:
        return ""
    return (
        "【任务的空间限定】(读作「目标 关系 参照物」。带限定的目标，凡是能接 reference/relation 的动作"
        "都必须原样带上这两个参数——少写或改写会被判无效)：\n" + "\n".join(lines) + "\n\n"
    )


def _format_blocked(blocked: Mapping[str, str] | None) -> str:
    """Render what the look found in the way of what ('' when nothing was).

    The validator rejects a plan that reaches through a barrier, but a rejection costs a
    round trip. Saying it up front is the same fact, delivered earlier. Derived entirely
    from measurement — no noun here is written down anywhere in the framework.
    """
    if not blocked:
        return ""
    lines = [f"  - {thing}：被 {barrier} 挡住" for thing, barrier in sorted(blocked.items())]
    return (
        "【够不到】规划前的观察发现下面这些东西现在拿不到（要碰它们，先用能清障的动作处理挡它的那个"
        "——打开/掀开/或者把挡的东西拿走）：\n" + "\n".join(lines) + "\n\n"
    )


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


def _extract_json_object(text: str) -> dict | None:
    """Best-effort parse of a JSON object from an LLM reply (tolerates fences/prose)."""
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    try:
        data = json.loads(candidate)
    except Exception:  # noqa: BLE001 - fall back to brace extraction
        m = re.search(r"\{.*\}", candidate, re.DOTALL)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except Exception:  # noqa: BLE001 - give up parse; return None
            return None
    return data if isinstance(data, dict) else None


# --------------------------------------------------------------------------- #
# LLM① task parser: natural-language task → structured intent
# --------------------------------------------------------------------------- #
_PARSER_SYSTEM = (
    "你是机器人任务解析器。把用户的自然语言任务解析成结构化意图 JSON："
    '{"targets": ["<英文物体名>", ...], "references": ["<英文物体名>", ...], '
    '"grounding": {"<目标名>": [{"reference": "<英文物体名>", "relation": "on|under|in|beside|near"}, ...]}, '
    '"destination": "<放置目的地或方位，无则 null>", '
    '"mode": "single|multi|continuous", "count": <整数或 null>, "intent": "pick|pick_place|carry"}。'
    "规则：targets 是要操作/抓取的物体，转成英文（开词表检测友好，中文易混）；"
    "references 是任务里用来**定位**目标的实物——参照物、容器、支撑面，以及放置目的地那个实物"
    "（「抽屉里的苹果」→ targets=['apple'], references=['drawer']；"
    "「帽子旁边的盒子搬到蓝色货架上」→ targets=['carton'], references=['hat','blue shelf']），"
    "同样转英文，无则 []。它们不被抓取，但规划前会先看一眼在不在，好让规划器知道现场长什么样；"
    "grounding 记录**任务限定了哪个目标在哪个参照物的什么方位**——「抽屉里的苹果」→ "
    "grounding={'apple': [{'reference':'drawer','relation':'in'}]}。现场可能有多个同名物体"
    "（桌上一个苹果、抽屉里一个苹果），限定词是唯一能把它们分开的东西，丢了就会抓错；"
    "**同一个名字被任务限定了多次就全部列出**——「香蕉旁的箱子搬到紫桌、柜子里的箱子搬到白桌」是两个箱子，"
    "grounding={'carton': [{'reference':'banana','relation':'beside'}, {'reference':'cabinet','relation':'in'}]}，"
    "只写一个会让另一个箱子无法指认；"
    "**每条 grounding 一律读作「object_name <relation> reference」**——主语是被限定的那个目标，不是参照物："
    "先定谁是 object_name（要抓的、或要放上去的那个东西），再判断**它相对 reference** 的方位。"
    "**key 只能是 targets 里的目标或 destination 那个实物**，绝不能拿参照物当 key："
    "「放到放着香蕉的白桌上」要限定的是**白桌**（现场可能不止一张白桌），"
    "写 {'white table': [{'reference':'banana','relation':'under'}]}；"
    "写成 {'banana': [{'reference':'white table','relation':'on'}]} 虽然读着顺，但限定挂在了香蕉上，"
    "白桌照样分不出是哪张，等于没写。**顺口的方向和有用的方向常常相反，按 key 必须是被操作对象来倒过来写。**"
    "最容易写反的是**用台面上的东西来指认台面**：「放着香蕉的白桌」「香蕉所在的桌子」「有杯子的那张桌子」"
    "——香蕉在桌上，所以桌子在香蕉**下面**，写 {'reference':'banana','relation':'under'}；"
    "写成 'on' 就成了「桌子压在香蕉上面」，几何上永远不成立，现场定位会一直 0 匹配。"
    "每条写完念一遍「<object_name> 在 <reference> 的 <relation>」，念着荒谬就是写反了。"
    "on/under 是**唯一**一对会写反的反向关系（A on B ⟺ B under A）；beside/near 左右对称、没有方向问题；"
    "in 只表达「目标在容器里」——「装着苹果的那个抽屉」这类**以内容物指认容器**的说法词表里没有对应关系"
    "（in 没有反向），这时**宁可不写这条 grounding，也不要写成反向的**；"
    "任务没有限定方位的目标不写进 grounding；relation 只能取 on/under/in/beside/near 之一；"
    "选检测名词要**具体、判别力强**（如 bin/carton/crate/container），避开过泛、易与背景或支撑面"
    "（桌面等矩形面）混淆的词（如 'box'）；目标与其支撑面同色/相邻时尤须如此；"
    "只能把**中心词**换得更具体（box→carton），任务给的**修饰词一个都不能丢**——颜色/大小/材质/新旧等"
    "（「紫色桌子」→ 'purple table' 而非 'table'，「白箱子」→ 'white box'，targets/references/destination "
    "一视同仁）：现场常有多张桌子、多个箱子，修饰词是唯一能分辨的线索，而且下游检测会拿颜色词逐像素核验，"
    "丢了它既选错实例、也关掉了这道核验；"
    "任务是把物体放到某处则 destination 填该处/方位（如 'table behind'）、intent='carry' 或 'pick_place'；"
    "只拿起不放则 intent='pick'、destination=null；"
    "任务含多个/连续（如「10 个」「所有」「依次」）则 mode='continuous'、count 填数字（「所有」则 count=null）；"
    "单个则 mode='single'。只输出这一个 JSON 对象，不要任何解释或 markdown 代码块。"
)

_VALID_MODES = ("single", "multi", "continuous")


def _validate_parsed(data: dict | None) -> dict[str, Any]:
    """Coerce a raw parsed dict into the well-shaped intent, filling safe defaults."""
    d = data if isinstance(data, dict) else {}
    targets = d.get("targets")
    if not isinstance(targets, list):
        targets = [targets] if isinstance(targets, str) and targets else []
    targets = [t for t in targets if isinstance(t, str) and t.strip()]
    refs = d.get("references")
    if not isinstance(refs, list):
        refs = [refs] if isinstance(refs, str) and refs else []
    refs = [r for r in refs if isinstance(r, str) and r.strip() and r not in targets]
    grounding: dict[str, list[dict[str, str]]] = {}
    for k, v in (d.get("grounding") or {}).items() if isinstance(d.get("grounding"), dict) else ():
        if not isinstance(k, str):
            continue
        # One name can be qualified more than once ("the box beside the banana" AND "the box
        # in the cabinet" are two boxes). A bare dict is the one-qualifier shorthand.
        for one in v if isinstance(v, list) else [v]:
            if not isinstance(one, dict):
                continue
            ref, rel = one.get("reference"), one.get("relation")
            # An unknown relation is dropped rather than passed on: relation_holds raises on one,
            # and a qualifier nobody can judge must not masquerade as a judged one.
            if not (isinstance(ref, str) and ref.strip() and rel in SPATIAL_RELATIONS):
                continue
            entry = {"reference": ref, "relation": rel}
            if entry not in grounding.setdefault(k, []):
                grounding[k].append(entry)
            if ref not in refs and ref not in targets:
                refs.append(ref)
    mode = d.get("mode") if d.get("mode") in _VALID_MODES else "single"
    count = d.get("count")
    if not isinstance(count, int):
        count = None
    dest = d.get("destination")
    dest = dest if isinstance(dest, str) and dest.strip() else None
    intent = d.get("intent") if d.get("intent") in ("pick", "pick_place", "carry") else (
        "pick_place" if dest else "pick"
    )
    return {
        "targets": targets, "references": refs, "grounding": grounding, "destination": dest,
        "mode": mode, "count": count, "intent": intent,
    }


def parse_task(
    query: str,
    *,
    api_base: str,
    api_key: str = "",
    model_name: str,
    timeout_s: float = 20.0,
    temperature: float = 0.0,
    proxy: str | None = None,
    attempts: int = 3,
) -> dict[str, Any]:
    """LLM① — parse a natural-language task into structured intent.

    Returns ``{targets, destination, mode, count, intent}`` (always well-shaped;
    safe defaults on a parse failure). This is the first of the two LLM calls;
    its ``targets`` drive the pre-plan perception (which objects to detect) and
    its structure informs the planner (LLM②).
    """
    user = f"任务：{query}\n\n请输出结构化意图 JSON 对象。"
    text = _chat(
        _PARSER_SYSTEM,
        user,
        api_base=api_base,
        api_key=api_key,
        model_name=model_name,
        timeout_s=timeout_s,
        temperature=temperature,
        proxy=proxy,
        attempts=attempts,
        max_tokens=256,
    )
    parsed = _validate_parsed(_extract_json_object(text))
    logger.info("[parser] task=%r → intent=%s", query, parsed)
    return parsed


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
    thinking_mode: str | None = None,
    cancel_token: CancelToken | None = None,
) -> str:
    """One OpenAI-compatible chat call with retries; returns the reply text.

    Default DIRECT connection — the LLM endpoints used here are domestic and
    reachable directly; a proxy is used ONLY when explicitly passed, never
    auto-picked from the environment. Raises ``RuntimeError`` if all attempts fail.

    ``cancel_token`` (GUI-only) makes the in-flight POST abandonable: the socket
    is closed on cancel and ``RunCancelled`` propagates. ``None`` → the original
    synchronous path, unchanged for CLI / tests.
    """
    import httpx

    url = api_base.rstrip("/").removesuffix("/chat/completions") + "/chat/completions"
    payload = {
        "model": model_name,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if thinking_mode is not None:
        payload["thinking"] = {"type": thinking_mode}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    last_exc: Exception | None = None
    for attempt in range(1, max(1, attempts) + 1):
        if cancel_token is not None:
            cancel_token.raise_if_set()
        try:
            client_kwargs: dict[str, Any] = {"timeout": timeout_s}
            if proxy:
                client_kwargs["proxy"] = proxy
            with httpx.Client(**client_kwargs) as client:
                resp = _post(client, url, payload, headers, cancel_token)
            resp.raise_for_status()
            return cast(str, resp.json()["choices"][0]["message"]["content"])
        except RunCancelled:
            raise
        except Exception as exc:  # noqa: BLE001 - retry on transient LLM/HTTP failure
            last_exc = exc
            logger.warning("[planner] attempt %d/%d failed: %s", attempt, attempts, exc)
    raise RuntimeError(f"planner LLM call failed after {attempts} attempts: {last_exc}") from last_exc


def _post(client: Any, url: str, payload: dict, headers: dict, cancel_token: CancelToken | None) -> Any:
    """POST the chat request; when a token is present make the socket read
    abandonable by registering ``client.close`` and polling under
    ``cancellable_call``. ``None`` → a plain blocking ``client.post``.
    """
    if cancel_token is None:
        return client.post(url, json=payload, headers=headers)
    unregister = cancel_token.on_cancel(client.close)
    try:
        return cancellable_call(lambda: client.post(url, json=payload, headers=headers), cancel_token)
    finally:
        unregister()


# --------------------------------------------------------------------------- #
# Action-sequence output contract — shared by both planning tiers.
#
# Body-agnostic by construction: it describes the *format* and the *symbols*, never
# which actions exist or what order they go in. What an action needs, produces and
# returns is data (``ToolMeta``), rendered into 【可用动作】; a step order that
# violates it is rejected by ``parse_sequence`` and fed back for self-repair. So no
# rule here may name a specific action.
# --------------------------------------------------------------------------- #
_SEQUENCE_FORMAT = (
    "动作序列是一个 JSON 数组，每个元素是一步：\n"
    '  {"op": "<动作名>", "params": {<参数>}, "bind": "<可选:绑定名>"}\n'
    "格式规则：\n"
    "- op 只能用【可用动作】清单里的名字，或【特殊动作】中列出的名字；不要臆造清单外的动作。\n"
    "- 每个动作后面标注了它的【返回字段】和【需要/产生】的状态。**一步要读前面某步的结果，"
    '就必须给产生它的那步加 bind，然后用 "<bind>.字段" 引用**；字段名只能取自该动作标注的返回字段。\n'
    "- bind 是合法标识符（字母/数字/下划线，不能含空格），与后续每处引用逐字一致。"
    "它和 object_name 是两回事：object_name 是自由字符串，bind 是变量名。\n"
    "- params 的值可以是数字，或对已绑定结果的算术表达式（只允许 + - * / 和 字段/下标），"
    '例如 "obj.z"、"obj.position[0]"、"obj.z + 30"。目标名等字符串原样写。\n'
    "- **不要把整个 `<bind>` 当参数传**：参数只认字面量或对 bind 的标量访问；整体 `<bind>` 会被当普通"
    "字符串。签名里带 `?` 的可选参数，若紧跟在产生它的那步之后，省略即可——动作会自动用最近一次结果。\n"
    "- 不要输出绝对坐标数值：xy 与工作高度都来自运行时感知。\n"
    "- **object_name 只来自用户任务本身**：从用户这句话识别要操作/感知的目标，用它的自然语言描述"
    "（颜色/形状/大小/类别/材质/位置等任意组合），转成英文——开放词表检测器对英文区分准。"
    "选词要**具体、判别力强**（如 bin/carton/crate/container），避开过泛、易与背景或支撑面混淆的词"
    "（如 'box'）；目标与其支撑面同色/相邻时尤须如此。**不要借用示例里出现过的任何具体目标名。**\n"
    "- 若某动作签名含关系参数（如 `on` / `has`），且任务把目标限定在某参照物之上（'桌上的箱子'）"
    "或把某面限定为其上放着某物（'放着水杯的桌子'），就给该参数传参照物的英文名以消歧同类多目标；"
    "签名无对应参数则忽略此约束。\n"
    "- **多物体/连续任务用循环**：任务是「把所有 / N 个 / 依次」处理多个同类目标（【场景感知】显示有多个实例，"
    "或任务明说数量），**不要**把同一段动作复制 N 遍；改为输出一个循环构造：\n"
    '  {"loop": {"detect": {"op": "<单目标感知动作>", "params": {"object_name": "<目标>"}}, '
    '"bind": "<绑定名>", "body": [<处理这一个目标的动作序列，可用 <绑定名>.字段>], "max_iters": <上限或省略>}}\n'
    "  循环每轮先感知一个目标、感知不到就自动停止；body 是「处理一个目标」的动作。"
    "任务明确 N 个就设 max_iters=N，说「所有」就省略 max_iters。单个目标的任务不要用循环。\n"
    "只输出这一个 JSON 数组，不要任何解释或 markdown 代码块。"
)

# --------------------------------------------------------------------------- #
# Tier 1: pick the skills AND expand them — one inference, not two
# --------------------------------------------------------------------------- #
_COMPILER_SYSTEM = (
    "你是机器人技能编译器（第一层规划）。给定用户任务、机器人【当前状态】、一组【可用技能】"
    "（每个技能是一份 SKILL.md，含其标准 workflow）和一份【可用动作】清单，你只做一次编译：\n"
    "1) 根据任务和各 SKILL.md 的用途，判断需要用到哪些技能；\n"
    "2) 把这些技能的 workflow **按执行顺序展开成一条扁平的动作序列**（多个技能首尾相接成一条）。\n"
    "**如果现有技能覆盖不了这个任务**（没有合适的技能，或从【当前状态】出发补不齐某个技能的前置），"
    "**就输出一个空数组 `[]`**，不要硬凑一个做不成的计划——上层会改用更细粒度的动作级规划来完成它。\n\n"
    + _SEQUENCE_FORMAT
    + "\n展开规则：\n"
    "- **能力门控步骤**：SKILL.md 里凡写「若你的 api_capabilities 含 X 则…」的步骤，"
    "**只在【机器人能力】清单包含 X 时才展开该步**，否则整步跳过。未提供能力清单时按默认分支展开。\n"
    "- **能力映射表 / 占位符**：SKILL.md 常给出按能力分行的映射表，并用 `<抓取>`/`<释放>` 等占位符指代。"
    "展开时必须**按【机器人能力】选对应那一行、并替换成【可用动作】里真正存在的那个动作**；"
    "清单里没有的变体一律不能输出。\n"
    "- **照 SKILL.md 展开**：workflow 表逐行、正文里的注意事项与顺序约束一并遵守，不要自加步骤或偏移。"
    '若 SKILL.md 给了偏移的数值范围，按它取一个具体数字写成字面量（如 "obj.z + 30"）；没写偏移就直接用它'
    "指定的字段。SKILL.md 为可读常把结果写成裸字段名，展开时必须改写成 <bind>.字段 并补上对应的 bind。\n"
    "- 参考【当前状态】与【场景感知】：已经成立的前置不必重复建立；目标标了「可达=是」就不必额外编排长距离移动。"
    "目标压根没出现在【场景感知】里、或【本体可达域】显示它多半在可达范围之外，就要先编排「搜索/靠近」"
    "再动手，不要指望原地就能够到。"
)

# --------------------------------------------------------------------------- #
# Tier 2 composition: no skill covers this — derive a sequence from the contracts
# --------------------------------------------------------------------------- #
_COMPOSER_SYSTEM = (
    "你是机器人动作编排器（第二层规划）。现有技能库覆盖不了这个任务，所以**没有可照抄的 workflow**："
    "你要从机器人【当前状态】出发，只用【可用动作】清单，自己推导出一条能完成任务的动作序列。\n\n"
    "怎么推导：每个动作都标注了它【需要】什么状态、【产生】什么状态、以及是否【需已有位置】/"
    "【产生位置】/【使已有位置全部失效】。一步的「需要」在轮到它时必须已经成立——要么【当前状态】里就有，"
    "要么由前面某一步「产生」。顺序完全由你根据这些前置与效果推出来，没有固定套路。\n"
    "特别注意：\n"
    "- 标了【需已有位置】的动作，前面必须有一步【产生位置】且中间没有【使已有位置全部失效】的动作；"
    "本体移动之后先前感知到的坐标一律作废，要重新感知。\n"
    "- 感知类动作可能失败（目标不在视野）。若任务要求的目标当前【场景感知】里没有，"
    "优先选清单里能**主动搜索**的那类动作，而不是只看当前视野的一次性检测。\n"
    "- 只用【机器人能力】支持的动作；用不到的动作不要列。\n\n"
    + _SEQUENCE_FORMAT
)


def _header_level(line: str) -> int:
    """Markdown header level (number of leading ``#``); 0 if the line isn't a header."""
    s = line.lstrip()
    return len(s) - len(s.lstrip("#")) if s.startswith("#") else 0


def _filter_skills_md_by_capability(
    skills_md: Sequence[dict[str, str]], api_capabilities: Sequence[str] | None
) -> list[dict[str, str]]:
    """Strip SKILL sections / lines gated on capabilities the robot lacks.

    General & maintenance-free: the gate vocabulary is the framework's fixed
    ``KNOWN_CAPABILITIES`` set, **not** action names — a new action added under some
    capability's gated section is kept/dropped with that capability automatically, no
    change here. A line is "gated-absent" when it names ≥1 known capability and **all**
    of them are absent from ``api_capabilities``; a gated-absent header drops its whole
    section (until the next header of same-or-higher level), a gated-absent non-header
    drops that line. Content naming no capability, or ≥1 present capability, is kept
    (so a shared fallback should simply not name a capability). Symmetric across robots.
    """
    caps = set(api_capabilities or ())
    if not caps:
        return list(skills_md)
    from jiuwensymbiosis.env.base import KNOWN_CAPABILITIES

    def gated_absent(line: str) -> bool:
        present = [c for c in KNOWN_CAPABILITIES if c in line]
        return bool(present) and all(c not in caps for c in present)

    out: list[dict[str, str]] = []
    for skill in skills_md:
        # Skill-level gate: a skill's frontmatter `capabilities` is requires-any — a
        # robot with none of them can't use the whole skill (e.g. `transport` needs a
        # movable base/waist). Drop it entirely so its prose can't leak actions either.
        req = set(skill.get("capabilities") or ())
        if req and caps.isdisjoint(req):
            continue
        kept: list[str] = []
        drop_level = 0  # >0 → inside a dropped section headed at this level
        for line in skill.get("markdown", "").splitlines():
            h = _header_level(line)
            if h:
                if drop_level and h > drop_level:
                    continue  # deeper sub-header stays inside the outer dropped section
                # This header is at/above the dropped level (or none active) → it ends
                # any dropped section; it re-opens one only if itself gated-absent.
                drop_level = h if gated_absent(line) else 0
                if drop_level:
                    continue
                kept.append(line)
                continue
            if drop_level or gated_absent(line):
                continue
            kept.append(line)
        out.append({**skill, "markdown": "\n".join(kept)})
    return out


def _strip_frontmatter(md: str) -> str:
    """The SKILL.md body without its YAML frontmatter.

    The frontmatter is data, and the header above already renders every field of it.
    Pasting it in as well says each fact twice — and the second copy is the one that
    goes stale, because nothing checks prose against the fields it restates.
    """
    if not md.startswith("---"):
        return md
    end = md.find("\n---", 3)
    if end == -1:
        return md
    nl = md.find("\n", end + 4)
    return md[nl + 1:] if nl != -1 else ""


def _format_skills_md(skills_md: Sequence[Mapping[str, Any]]) -> str:
    """Render each candidate skill: what it is for, its contract, then its SKILL.md.

    Tier 1 both picks and expands in one call, so the header carries what selection
    needs — purpose, capability gate and pre-conditions/effects — and the body carries
    the workflow. The header is the complete rendering of the frontmatter, which is why
    the body can drop it (see :func:`_strip_frontmatter`).
    """
    blocks = []
    for s in skills_md:
        head = f"### 技能：{s['name']}"
        if s.get("description"):
            head += f" —— {s['description']}"
        gate = s.get("capabilities")
        if gate:
            head += f"\n能力门（有其中任一即可用）：{', '.join(sorted(gate))}"
        effects = _format_effects(s)
        if effects:
            head += f"\n{effects}"
        body = _strip_frontmatter(str(s.get("markdown", ""))).strip()
        blocks.append(f"{head}\n{body}")
    return "\n\n".join(blocks)


def _sequence_correction(exc: SequenceError) -> str:
    """Turn a validation failure into feedback the model can act on.

    ``parse_sequence`` already names the missing token / stale binding and which
    action would supply it, so the bulk of the repair instruction is the error
    itself; this only adds the reminder that matches the failure class.
    """
    text = str(exc)
    hint = ""
    if "unknown op" in text:
        hint = (
            "\n**op 只能取自【可用动作】清单**：不要臆造、也不要用清单外的动作名"
            "（能力映射表/占位符要按【机器人能力】选到清单里真正存在的那个动作）。"
        )
    elif "unbound" in text or "does not return" in text or "stale" in text:
        hint = (
            "\n凡是产生结果供后续步骤读取的那一步都必须带 bind；后续 <bind>.字段 里的 <bind> 要与它逐字一致"
            "（用下划线、不含空格；object_name 是另一个独立的自由字符串），字段名只能取自该动作标注的返回字段。"
            "本体移动会让先前感知到的位置作废——移动之后要重新感知。"
        )
    return f"\n\n上次输出的动作序列无效：{text}{hint}\n请修正后重新输出完整的 JSON 数组。"


def _compile_loop(
    system: str,
    user: str,
    *,
    allowed_ops: Any,
    special_ops: frozenset[str],
    initial_state: Sequence[str] | None,
    api_base: str,
    api_key: str,
    model_name: str,
    timeout_s: float,
    temperature: float,
    proxy: str | None,
    attempts: int,
    label: str,
    grounding: Mapping[str, Mapping[str, str]] | None = None,
    blocked_access: Mapping[str, str] | None = None,
    allow_decline: bool = False,
    thinking_mode: str | None = "disabled",
    cancel_token: CancelToken | None = None,
) -> list[dict[str, Any]]:
    """Sample an action sequence until one validates, feeding each failure back.

    Shared by both tiers — they differ only in the system prompt and in what the
    user block contains, never in how a rejected sequence is repaired. Re-sampling
    the identical prompt at a low temperature tends to repeat the same mistake, so
    every retry carries the validator's own message.

    ``allow_decline`` decides what an empty array means. For Tier 1 it is the answer
    "no skill here covers this task" and returns immediately — that is the decidable
    signal to drop a tier. For Tier 2 there is nothing left to drop to, so an empty
    reply is a failed attempt like any other.

    ``thinking_mode`` defaults to ``"disabled"`` because thinking models can spend the
    whole output budget on ``reasoning_content`` and leave ``message.content`` empty;
    compilation is a constrained JSON transformation, so the budget should go to the
    sequence. ``cancel_token`` (GUI-only) is polled once per attempt and makes the
    in-flight POST abandonable; ``None`` → the original synchronous path.
    """
    last_err: str | None = None
    correction = ""
    for attempt in range(1, max(1, attempts) + 1):
        if cancel_token is not None:
            cancel_token.raise_if_set()
        try:
            text = _chat(
                system,
                user + correction,
                api_base=api_base,
                api_key=api_key,
                model_name=model_name,
                timeout_s=timeout_s,
                temperature=temperature,
                proxy=proxy,
                attempts=1,
                max_tokens=1500,
                thinking_mode=thinking_mode,
                cancel_token=cancel_token,
            )
        except RuntimeError as exc:
            # Transient LLM/HTTP failure (e.g. read timeout on a slow generation):
            # count it as one attempt and retry rather than aborting the compile.
            last_err = str(exc)
            logger.warning("[%s] attempt %d: LLM call failed, retrying: %s", label, attempt, exc)
            continue
        data = _extract_json_array(text)
        if data is None:
            last_err = f"no JSON array in reply: {text[:200]!r}"
            logger.warning("[%s] attempt %d: %s", label, attempt, last_err)
            correction = "\n\n上次回复没有包含合法的 JSON 数组。请只输出一个 JSON 数组，不要任何解释或 markdown。"
            continue
        if not data:
            if allow_decline:
                logger.info("[%s] → declined: no skill covers this task", label)
                return []
            last_err = "reply was an empty sequence"
            logger.warning("[%s] attempt %d: %s", label, attempt, last_err)
            correction = "\n\n上次输出是空序列，但这个任务必须给出可执行的动作。请重新输出完整的 JSON 数组。"
            continue
        try:
            parse_sequence(
                data, allowed_ops=allowed_ops, special_ops=special_ops,
                initial_state=initial_state, grounding=grounding,
                blocked_access=blocked_access,
            )  # validate only; keep raw dicts
        except SequenceError as exc:
            last_err = str(exc)
            logger.warning("[%s] attempt %d: invalid sequence: %s", label, attempt, exc)
            correction = _sequence_correction(exc)
            continue
        logger.info("[%s] → %d steps: %s", label, len(data), data)
        return data
    raise RuntimeError(f"{label} produced no valid sequence after {attempts} attempts: {last_err}")


def compose_actions(
    query: str,
    *,
    action_index: Mapping[str, Any],
    allowed_ops: Any,
    api_base: str,
    api_key: str = "",
    model_name: str,
    special_ops: frozenset[str] | set[str] = frozenset(),
    timeout_s: float = 90.0,
    temperature: float = 0.0,
    proxy: str | None = None,
    attempts: int = 4,
    api_capabilities: list[str] | None = None,
    scene: Any = None,
    world_block: str = "",
    world_tokens: Sequence[str] | None = None,
    grounding: Mapping[str, Mapping[str, str]] | None = None,
    blocked_access: Mapping[str, str] | None = None,
    reason: str = "",
    thinking_mode: str | None = "disabled",
    cancel_token: CancelToken | None = None,
) -> list[dict[str, Any]]:
    """Tier 2 — derive an action sequence from the contracts alone, with no SKILL.md.

    This runs when the skill library does not cover the task. The model gets the
    task, the current world state and every action's full contract (params, result
    fields, pre-conditions, effects) and must find an order that satisfies the
    pre-condition chain itself. ``parse_sequence`` checks that chain, so a sequence
    that returns here is one the state machine accepts.

    Args:
        action_index: name → bound method, as ``_build_action_index`` returns.
        world_tokens: the state the sequence starts from; ``None`` disables the
            self-state check (unknown ≠ false — see ``parse_sequence``).
        reason: why Tier 1 declined, passed through so the model knows what gap
            it is filling.

    Raises:
        RuntimeError: if every attempt fails to produce a valid sequence.
    """
    active_special_ops = frozenset(special_ops)
    special_text = ", ".join(sorted(active_special_ops)) or "无"
    gap = f"【技能库为何不适用】{reason}\n\n" if reason else ""
    user = (
        f"任务：{query}\n\n"
        f"{world_block}"
        f"{gap}"
        f"{_format_capabilities(api_capabilities)}"
        f"{_format_scene(scene)}"
        f"{_format_grounding(grounding)}"
        f"{_format_blocked(blocked_access)}"
        f"【可用动作】(括号内为参数名，? = 可省略；-> 后是该动作的返回字段)：\n"
        f"{format_action_contracts(action_index)}\n\n"
        f"【特殊动作】：{special_text}\n\n"
        "请输出你推导出的动作序列 JSON 数组。"
    )
    return _compile_loop(
        _COMPOSER_SYSTEM,
        user,
        allowed_ops=allowed_ops,
        special_ops=active_special_ops,
        initial_state=world_tokens,
        grounding=grounding,
        blocked_access=blocked_access,
        api_base=api_base,
        api_key=api_key,
        model_name=model_name,
        timeout_s=timeout_s,
        temperature=temperature,
        proxy=proxy,
        attempts=attempts,
        label="composer",
        thinking_mode=thinking_mode,
        cancel_token=cancel_token,
    )


def compile_sequence(
    query: str,
    *,
    skills_md: Sequence[dict[str, str]],
    action_vocab: Sequence[str],
    allowed_ops: Any,
    api_base: str,
    api_key: str = "",
    model_name: str,
    special_ops: frozenset[str] | set[str] = frozenset(),
    timeout_s: float = 90.0,
    temperature: float = 0.0,
    proxy: str | None = None,
    attempts: int = 4,
    api_capabilities: list[str] | None = None,
    action_sigs: dict[str, str] | None = None,
    scene: Any = None,
    action_index: Mapping[str, Any] | None = None,
    world_block: str = "",
    world_tokens: Sequence[str] | None = None,
    grounding: Mapping[str, Mapping[str, str]] | None = None,
    blocked_access: Mapping[str, str] | None = None,
    thinking_mode: str | None = "disabled",
    cancel_token: CancelToken | None = None,
) -> list[dict[str, Any]]:
    """Tier 1 — pick the skills that fit and expand their workflows, in one inference.

    The same call that *reads* the candidate SKILL.md to pick skills also *emits*
    their compiled action sequence, so skill-level planning costs exactly one round
    trip. An **empty** return is not a failure but the decidable answer "no skill
    here covers this task", which is what routes the caller to Tier 2.

    Args:
        query: the user's natural-language task.
        skills_md: candidate skills as ``[{"name", "markdown"}]`` (full SKILL.md).
        action_vocab: op names the robot exposes (the @robot_tool index keys).
        allowed_ops: collection used to validate the result via ``parse_sequence``.
        action_index: name → bound method; when given, 【可用动作】 shows each action's
            full contract instead of only its param names.
        world_tokens: the state the sequence starts from; ``None`` disables the
            self-state check (unknown ≠ false — see ``parse_sequence``).
        api_base/api_key/model_name/...: LLM endpoint config.
        thinking_mode: Thinking-mode request value. Defaults to ``"disabled"``
            because thinking models can spend the whole output budget on
            ``reasoning_content`` and leave ``message.content`` empty;
            compilation is a constrained JSON transformation, so the budget
            should go to the final sequence. Pass ``None`` for an endpoint
            that rejects the ``thinking`` extension.
        cancel_token: GUI-only; makes the compile LLM call abandonable.

    Returns:
        The validated raw step list ``[{"op", "params", "bind"?}, ...]`` (the
        caller turns it into ``ActionStep`` via ``parse_sequence`` again, or uses
        these dicts directly), or ``[]`` when no skill covers the task.

    Raises:
        RuntimeError: if the LLM call fails, or every attempt yields a sequence
            that fails schema validation.
    """
    # Deterministic capability gate: never show the compiler SKILL branches/steps
    # gated on capabilities this robot lacks, so it can't follow them (prompt wording
    # alone doesn't hold). Keyed on the fixed capability set — new actions need no change.
    skills_md = _filter_skills_md_by_capability(skills_md, api_capabilities)
    # 【可用动作】 carries each action's exact param names (from its signature) so the
    # compiler uses them verbatim instead of inventing param names.
    if action_index:
        vocab_block = "(括号内为参数名，? = 可省略；-> 后是该动作的返回字段)：\n" + format_action_contracts(action_index)
    else:
        sigs = action_sigs or {}
        vocab_block = "(动作名后括号是参数名，? = 可省略；只用这些参数名)：" + ", ".join(
            f"{name}{sigs.get(name, '')}" for name in action_vocab
        )
    active_special_ops = frozenset(special_ops)
    special_text = ", ".join(sorted(active_special_ops)) or "无"
    user = (
        f"任务：{query}\n\n"
        f"{world_block}"
        f"{_format_capabilities(api_capabilities)}"
        f"{_format_scene(scene)}"
        f"{_format_grounding(grounding)}"
        f"{_format_blocked(blocked_access)}"
        f"【可用技能】(SKILL.md)：\n{_format_skills_md(skills_md)}\n\n"
        f"【可用动作】{vocab_block}\n"
        f"【特殊动作】：{special_text}\n\n"
        "特殊动作的用途、参数和 workflow 替换规则只按 SKILL.md 执行；"
        "未被 SKILL.md 选用的特殊动作不要自行插入。\n\n"
        "请输出展开后的动作序列 JSON 数组。"
    )
    return _compile_loop(
        _COMPILER_SYSTEM,
        user,
        allowed_ops=allowed_ops,
        special_ops=active_special_ops,
        initial_state=world_tokens,
        grounding=grounding,
        blocked_access=blocked_access,
        api_base=api_base,
        api_key=api_key,
        model_name=model_name,
        timeout_s=timeout_s,
        temperature=temperature,
        proxy=proxy,
        attempts=attempts,
        label="compiler",
        allow_decline=True,
        thinking_mode=thinking_mode,
        cancel_token=cancel_token,
    )


# --------------------------------------------------------------------------- #
# The two-tier loop: try the skill library first, derive from actions if it can't.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PlanResult:
    """A compiled sequence plus how it was arrived at.

    ``tier`` is ``"skill"`` when the library covered the task and ``"action"`` when
    it did not and the sequence was derived from the action contracts. ``reason``
    records why Tier 1 was declined — it is what a distiller needs to know that this
    sequence is a candidate new skill. ``skills`` are the candidates Tier 1 was given,
    not a claim about which it used: selection happens inside the compile call, so
    the only honest provenance is what passed the capability gate.
    """

    sequence: list[dict[str, Any]]
    tier: str
    skills: tuple[str, ...] = ()
    reason: str = ""


def plan_task(
    query: str,
    *,
    skills_md: Sequence[dict[str, Any]],
    action_index: Mapping[str, Any],
    allowed_ops: Any,
    api_base: str,
    api_key: str = "",
    model_name: str,
    special_ops: frozenset[str] | set[str] = frozenset(),
    timeout_s: float = 90.0,
    temperature: float = 0.0,
    proxy: str | None = None,
    attempts: int = 4,
    api_capabilities: list[str] | None = None,
    action_sigs: dict[str, str] | None = None,
    scene: Any = None,
    world_block: str = "",
    world_tokens: Sequence[str] | None = None,
    grounding: Mapping[str, Mapping[str, str]] | None = None,
    blocked_access: Mapping[str, str] | None = None,
    thinking_mode: str | None = "disabled",
    cancel_token: CancelToken | None = None,
) -> PlanResult:
    """Plan a task with the skill library first, falling back to the action contracts.

    Tier 1 (``compile_sequence``) picks skills and expands them in **one** inference,
    so the common path is a single LLM round trip — the second tier costs nothing
    until it is actually needed. Tier 2 (``compose_actions``) then derives a sequence
    from the action contracts alone, and takes over on exactly three **decidable**
    conditions — never on the model's own say-so mid-compile:

    1. no skill survives the capability gate;
    2. Tier 1 answers with an explicit empty sequence (it was told to, when nothing
       in the library fits);
    3. Tier 1 exhausts its correction retries — which includes the case where the
       chosen skills' pre-conditions cannot hold from ``world_tokens``, since
       ``parse_sequence`` rejects the expansion and feeds that back.

    A Tier 1 call that merely *fails* to reach the endpoint also lands in (3): with
    no sequence in hand there is nothing else to do but try the cheaper vocabulary.

    Raises:
        RuntimeError: if Tier 2 also fails to produce a valid sequence.
    """
    common = {
        "allowed_ops": allowed_ops,
        "special_ops": special_ops,
        "api_capabilities": api_capabilities,
        "scene": scene,
        "world_block": world_block,
        "world_tokens": world_tokens,
        "grounding": grounding,
        "blocked_access": blocked_access,
        "api_base": api_base,
        "api_key": api_key,
        "model_name": model_name,
        "timeout_s": timeout_s,
        "temperature": temperature,
        "proxy": proxy,
        "attempts": attempts,
        "thinking_mode": thinking_mode,
        "cancel_token": cancel_token,
    }
    usable_md = _filter_skills_md_by_capability(skills_md, api_capabilities)

    reason = "skill_library_empty"
    if usable_md:
        names = tuple(s["name"] for s in usable_md)
        try:
            raw = compile_sequence(
                query,
                skills_md=usable_md,
                action_vocab=sorted(action_index),
                action_sigs=action_sigs,
                action_index=action_index,
                **common,
            )
        except RuntimeError as exc:
            raw, reason = [], f"expanding {list(names)} produced no valid sequence: {exc}"
            logger.warning("[plan] tier 1 expansion failed, dropping to action-level planning: %s", exc)
        else:
            if raw:
                logger.info("[plan] tier 1 (skills offered %s) → %d steps", list(names), len(raw))
                return PlanResult(sequence=raw, tier="skill", skills=names)
            reason = f"none of {list(names)} covers this task"

    logger.info("[plan] tier 2 (no usable skill: %s)", reason)
    raw = compose_actions(query, action_index=action_index, reason=reason, **common)
    return PlanResult(sequence=raw, tier="action", reason=reason)
