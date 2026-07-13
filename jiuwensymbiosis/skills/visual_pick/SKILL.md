---
name: visual_pick
description: 视觉引导抓取技能 —— 用相机识别目标物体并抓起（平行夹爪或吸盘按机器人能力自动选择），结束时物体已被抓住、机械臂悬于搬运高度，等待与 visual_place 衔接。
---

# visual_pick — 视觉引导抓取（夹爪 / 吸盘通用）

## 何时启用

满足以下**全部**条件时启用本 skill：

1. 用户指令含"把 X 抓起来 / 夹住 / 吸住 / 拿起来"等抓取意图，目标可由一段文本描述（颜色、形状或品类）。
2. 机器人 `api.capabilities` 同时包含 `motion.cartesian`、`vision.detection`，且包含**至少一种抓取能力**：`grasp.parallel`（平行夹爪）**或** `grasp.suction`（吸盘）。
3. 已注册 `robot_control` 工具。
4. 当前**夹具空载**（夹爪张开 / 吸盘未吸住任何物体；如果上一步刚做过 visual_place 或 release，可放心启用）。

任务里若还需要把物体放到指定位置，**先用本 skill 抓起来，再 chain 调用 visual_place**——两者解耦，互不重复职责。

## 抓取 / 释放动作（按机器人能力选择）

本 skill 的"抓取"与"释放"是抽象动作。请从你的 `api_capabilities` 选对应的 `robot_control` action 名，**不要**调用不在能力内的动作：

| 你的能力 | 抓取 action | 释放 action |
|---|---|---|
| `grasp.parallel` | `close_gripper` | `open_gripper` |
| `grasp.suction` | `activate_suction` | `deactivate_suction` |

下文用 **`<抓取>`** / **`<释放>`** 指代上表里你机器人对应的那一个。

## 检测目标来自用户任务（不要用户传参数）

要抓的物体名（下文的 `<目标语义>`）**由用户的自然语言任务决定**：从用户这句话里识别"要抓的东西"，
用它的自然语言描述（颜色/形状/大小/类别/材质/位置等任意特征的组合，也可能只有类别）作为
`get_grasp_info_simple` 的 `object_name`。用户只说任务、不会再单独传物体参数；**物体一律以用户任务为准，
不要套用本文出现过的任何示例物体名**。

**腕部相机(eye-in-hand)注意**：相机装在机械臂腕部，一旦移动就可能被遮挡或视角改变。如果任务
**接着还要放置**（chain visual_place），请在 **home 处一次性**把"要抓的物体"和"放置目标"两个
坐标都 `get_grasp_info_simple` 读好存下来，再开始移动——否则抓取后相机就拍不到放置目标了。标准 Workflow 可以因这条规则发生一点点变化，即把 visual_place 中的 get_grasp_info_simple 挪动到前面来和本 skill 中的 get_grasp_info_simple 一起执行。

## 抓取高度直接用检测给的 `grasp_z`，不要自己算

`get_grasp_info_simple` 返回里已经有**确定化算好**的抓取点，**你不要自己拿 `position` 的 z 去加减**：

```
{
  "ok": true,
  "position":       [x, y, top_z],     # 物体顶面（俯视看到的那一面）——只用来取 x,y，别拿它当夹取高度
  "grasp_z":        <数>,              # ★夹取高度：下降就降到这（已按抓取深度+桌面安全算好）
  "grasp_position": [x, y, grasp_z],   # = 直接 goto 到这里就是夹取位
  "bottom_z":       <数>,              # 物体底面（放置时若需堆叠，供 visual_place 用）
  "height_mm":      <数>, "score": <数>
}
```

平行夹爪要夹**物体主体/侧壁**（夹顶面夹不住），`grasp_z` 已经是"顶面下方合适深度、且不低于桌面"的安全夹取高度。**下降一律降到 `grasp_z`**。

## 标准 Workflow

按顺序调 `robot_control(action=..., params=...)`。每一步如果返回 `success=False`，立即跳到"失败处理"。

| # | action            | params                                       | 目的                                              |
|---|-------------------|----------------------------------------------|---------------------------------------------------|
| 1 | `home`            | `{}`                                         | 回到拍照位姿，给视觉一个稳定的深度基线。          |
| 2 | `<释放>`          | `{}`                                         | 先把夹具置于空载状态（夹爪张开 / 确认未吸）。      |
| 3 | `get_grasp_info_simple` | `{"object_name": "<目标语义>"}`        | 检测目标，拿到 `x,y` 与 **`grasp_z`**（见上）。   |
| 4 | `goto_xyzr`       | `{"x": x, "y": y, "z": grasp_z + approach}` | 移到目标正上方（approach ≈ +30~50mm，相对 `grasp_z`）。 |
| 5 | `goto_xyzr`       | `{"x": x, "y": y, "z": grasp_z}`            | **下降到 `grasp_z`**（检测给的夹取高度，别改）。SafetyRail 会拦越界 z。 |
| 6 | `<抓取>`          | `{}`                                         | 到位后闭合夹爪 / 开吸。                            |
| 7 | `goto_xyzr`       | `{"x": x, "y": y, "z": grasp_z + lift}`     | 提起到搬运高度（lift ≈ +50~80mm）。              |

> `get_grasp_info_simple` 是检测+投影+确定化抓取高度的一站式封装，常态首选；**不要**用 `analyze_scene`+`pixel_to_base_xyz` 手算（那样拿不到 `grasp_z`）。

完成后**结束状态**：夹具抓住目标，TCP 位于 `(x, y, grasp_z + lift)`。**不要**再 home，把后续动线交给 visual_place 或上层 agent。

## 失败处理

- 任一步返回 `success=False`：
  1. 简要记录错误（给中文一句话，不需要念完整堆栈）。
  2. **必须**调 `<释放>`（即便未确认是否抓住，也先释放避免拖件）。
  3. 调 `home` 返回安全位姿。
  4. 向用户报告"在第 N 步（动作名）失败：<原因>"，**不要**重试相同参数。
- `get_grasp_info_simple` / `analyze_scene` 置信度 `score < 0.4` 视为识别失败：把物体描述换得更具体一次（补上颜色/形状/大小/位置等特征），仍失败则放弃并报告。

## 与 Rails 的协作

- **SafetyRail**：监视 `goto_xyzr` 是否越出工作空间 / 越过 z_min_safe。它会用 ValueError 中断你；该错误**不应**被吞，照"失败处理"流程处理。
- **VisualFeedbackRail**：在 `motion` tag 工具调用后自动注入观测；**不要**重复 `get_image`，会浪费带宽。
- **RecoveryRail**：异常时它会兜底执行 home + release，但**你仍要按"失败处理"显式调用** `<释放>` + `home`，保证语义清晰。

## 参数取值约定

- `approach`（接近高度偏移）= 30 ~ 50 mm（默认 40）。物体偏高用上限。
- `lift`（搬运高度偏移）= 50 ~ 80 mm。横移距离大时用上限。
- 所有坐标走 mm / deg，**不要**用 m。`depth_m` 是个例外（视觉返回米），仅作为 `pixel_to_base_xyz` 的输入。

## Anti-patterns（不要做）

- ❌ 跳过 `home` 直接检测：相机基线不稳，深度噪声大。
- ❌ 在本 skill 末尾 `<释放>`（夹爪 open / 吸盘 deactivate）：会让物体掉在原位，让 visual_place 拿到空夹具。
- ❌ 把放置点的 goto 写进本 skill：放置职责在 visual_place，本 skill 只负责"拿起来"。
- ❌ 调用不在 `api_capabilities` 内的抓取动作（如吸盘机器人去调 `close_gripper`）：会返回 unknown action。
- ❌ 用 `run_python`/`robot_control` 之外的工具实现本 workflow：会绕过 SafetyRail 与 VisualFeedbackRail。
- ❌ 在 `success=False` 后继续后续 action：状态已脱锚，必须先 home。
