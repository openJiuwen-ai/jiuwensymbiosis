---
name: push
description: 推合技能 —— 把门 / 抽屉 / 柜门一类可推动的物体朝远离自己的方向推回关上，或把物体推到一边。手上必须空载；结束时被推物处于关闭 / 移开状态。
capabilities: [motion.base]
requires: [payload.clear]
invalidates: [body.home]
---

# push — 推合（关门 / 推回抽屉 / 推开挡路物）

## 何时启用

同时满足：

1. 任务需要**把某物推回 / 推开**：关上柜门、推回抽屉、把挡路的东西推到一边。
2. 手上**空载**（`payload.clear`）。
3. 已注册 `robot_control`。

**只在任务真的要求关闭 / 推开时才启用**。任务只说"把柜子里的箱子拿出来放到桌上"时，
收尾**不需要**关门——不要自作主张补一个 `push`。

## 标准 Workflow

| # | action | params | 目的 |
|---|---|---|---|
| 1 | `approach_for_grasp` | `{"object_name": "<被推物>"}` | 驶到可操作距离。无底盘本体跳过。 |
| 2 | `push` | `{"object_name": "<被推物>"}` | 推合 / 推开。`distance_m` 可省略。 |

## 结束状态

被推物已关闭 / 移开；手上仍空载。

## 失败处理

- `push` 返回失败：报"推合失败"，交回上层，不要反复重试。

## Anti-patterns

- ❌ 持物时 `push`（契约 `requires: payload.clear` 会直接拒绝）。
- ❌ 任务没要求关门却在末尾补 `push`：多余动作，也可能把刚放好的东西碰倒。
