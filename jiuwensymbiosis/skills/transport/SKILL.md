---
name: transport
description: 通用移动技能 —— 让机器人移动/转向到目标方位（底盘前进转向 / 腰部旋转，按机器人能力自动选择）。可在抓取前先移动到物体附近、可持物搬运、也可单独移动；不要求手上必须有物体。
capabilities: [motion.base, motion.waist]
invalidates_locations: true
---

# transport — 通用移动 / 换向（能力门控，移动机型专用）

## 何时启用

满足以下条件时启用本 skill：

1. 任务需要**移动机器人本体或改变其朝向**：例如「先到 X 处」「转向面向 Y」「把手上的东西搬到某方位」「目标较远、先驶近再抓」。
2. 机器人 `api.capabilities` 含 **`motion.base`** 或 **`motion.waist`** 至少一种（固定机械臂机型无移动/腰转能力 → 本 skill 不适用，不要启用）。
3. 已注册 `robot_control` 工具。

**本 skill 不要求手上有物体**：它可以**先移动到物体附近再抓**（在 visual_pick 之前）、可以**持物搬运**（visual_pick 之后、visual_place 之前）、也可以**单独移动**。用哪种由任务和上层编排决定，本 skill 只负责"把机器人移到/转到目标方位"。

## 移动 / 换向动作（按机器人能力选择）

从你的 `api_capabilities` 选**你机器人有的**那个，**不要**调用不在能力内的动作：

| 你的能力 | 移动 / 换向 action | 说明 |
|---|---|---|
| `motion.base` | `rotate_base(dyaw_rad)` | 底盘原地转（`+`=左转，弧度）；整机转向。**仅用于用户给了明确转角**（如"向左转 90°"= `π/2`）。**不要**用它盲转固定角度去对准某个可感知的物体/放置面。 |
| `motion.base` | `navigate_relative(dx_m, dyaw_rad)` | 底盘相对移动（前进 `dx_m` 米 + 转 `dyaw_rad`）。驶近某方位用它。 |
| `motion.waist` | `turn_waist(delta_rad)` | 只转躯干腰部（`+`=左转）；小幅、精细换向。 |

- 需要**驶近某方位**：`navigate_relative` 前进；用户**给了明确转角**：`rotate_base`；需要**小幅正对**：`turn_waist`。
- ⚠️ **对准一个"可感知的物体/放置面"**（如"面向桌子""正对箱子"）**不在本 skill 盲转**——交给 pick 侧的 `approach_for_grasp` / place 侧的 `approach_for_place`，它们**按感知到的真实方位**转正、不在视野内会**扫掠搜索**，比盲转固定角度更准也常更省转角。

## 标准 Workflow

1. 从任务识别移动需求：用户给的**明确转角**、或"驶近某方位"这类前进量。（"面向桌子/正对箱子"这类**对准可感知目标**的需求**不在这里盲转**——由 place 的 `approach_for_place` / pick 的 `approach_for_grasp` 感知搜索完成。）
2. 按上表能力选动作执行到位（可组合：先转向、再 `navigate_relative` 前进）。
3. 结束——把后续动线交给上层编排（visual_pick 抓 / visual_place 放）。

## 结束状态

机器人已移动/转到目标方位。若移动前手上持物，则持物状态不变（本 skill 不抓不放）。

## 失败处理

- 移动/换向动作返回失败（越界 / 雷达急停 / odom 异常）：简短报告"移动失败"，把控制权交回上层 agent；上层可重试或改路径。**若手上持物，不要因移动失败而放手。**

## Anti-patterns（不要做）

- ❌ 在本 skill 内 `<抓取>` / `<释放>`：抓取是 visual_pick、放置是 visual_place 的职责，本 skill 只负责移动/换向。
- ❌ 调用不在 `api_capabilities` 内的动作（如固定臂机型调 `rotate_base`）：会返回 unknown action。
- ❌ 固定机械臂机型启用本 skill：无 `motion.base`/`motion.waist`，本 skill 不适用。
- ❌ 为对准一个**可感知的目标/放置面**而 `rotate_base` **盲转固定角度**（如 `rotate_base(π)` 假设"在身后"）：真实方位未知，应交给 place 的 `approach_for_place` / pick 的 `approach_for_grasp`——它们感知目标真实方位、不在视野内会扫掠搜索。
