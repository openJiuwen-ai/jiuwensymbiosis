---
name: slot_pick
description: 仅用于「把同一类物体一个接一个反复抓放、直到没有为止」的批量循环，且工具列表里确实有名为 slot_pick 的工具时。单次抓放（把 X 放到 Y 上）请勿使用本技能——那是 visual_pick + visual_place 的事。
---

# slot_pick — 通用多轮抓放循环（本体无关 / 任务无关）

## 何时启用（先看这两条硬条件）

1. **任务是批量重复**：同一类物体一个接一个地抓起放到目标处，直到检测不到为止；**单次抓放不算**。
2. **工具列表里确实有 `slot_pick` 工具**。

**任一条不满足，就不要用本技能**——尤其"把黑盒放到白盒上"这种**单次**任务，请直接用
`visual_pick` + `visual_place`（通过 `robot_control` 调底层动作），**不要**来找 slot_pick 工具。

动态循环已经封装在 `slot_pick` 工具里：每轮重新检测"要抓的物体"和"放置目标"，抓起、放下、继续，
直到检测不到要抓的物体（或达到最大循环次数）。模型只负责判断是否调用、读返回结果、失败时汇报或 fallback。

> 检测是**通用**的（同 get_grasp_info_simple，按物体名检测任意目标，无任务专用过滤）；抓取/释放由适配器
> 注入的 strategy 决定（平行夹爪 `close/open_gripper`，吸盘本体 `activate/deactivate_suction`）。
> 模型不需要关心机器人本体。

## 检测目标来自用户任务（不要用户传参数）

每轮检测什么不是写死的，由 `chip_object_name`（要抓的物体）和 `slot_object_name`（放置目标）决定。
**这两个名字由你从用户的自然语言任务里识别**，用户只说任务、不会单独传物体参数。

- 用户点名了具体物体（如"把**蓝色铁片**逐个放到**金属凹槽**里"）→ 调用 `slot_pick` 时传
  `{"chip_object_name": "蓝色铁片", "slot_object_name": "金属凹槽"}`。
- 用户没点名、YAML 里已配好默认目标 → 不传，沿用配置。

坐标全部由检测得到，**不要**自己编坐标。

## 标准 Workflow

1. 调用 `slot_pick`：检测目标按上一节由用户任务决定；观察位姿、厚度、放置偏移、最大循环次数等其余参数由 YAML 提供。
2. 返回 `ok=true`：任务完成或达到停止条件（检测不到要抓物体 / 检测不到放置目标 / 达到最大循环次数），简要汇报 `cycles_done` 及每轮的 `slot_xyz`、`chip_xyz`、`place_tip_z`。
3. 返回 `ok=false` 且 `fallback_recommended=true`：失败发生在检测阶段；可换措辞后再调一次，或用低层视觉工具排查。
4. 返回 `ok=false` 且 `fallback_recommended=false`：不要继续低层补动作；按 `stage`/`reason` 汇报，等待人工检查或复位。

## 返回值约定

- `ok`: 是否正常完成（含"检测不到要抓物体 / 放置目标"这类干净结束）。
- `stage`: `done_no_chip`、`done_no_slot`、`done_max_cycles` 或失败阶段，例如 `cycle_1_detect_slot`、`cycle_2_close_gripper`。
- `fallback_recommended`: 是否建议模型接管做视觉 fallback。
- `cycles_done`: 已完成的抓放轮数。
- `cycles`: 每轮的检测结果、要抓物体/放置目标坐标和放置结果。
- `last_slot_detection` / `last_chip_detection`: 失败或结束时的最近一次检测结果。
- `slot_xyz` / `chip_xyz`: 每轮放置目标 / 要抓物体的基座坐标，单位 mm。
- `pick_r` / `place_r`: 抓取/放置时的末端旋转角（place_r = pick_r + place_r_delta_deg）。
- `place_tip_z`: 放置时指尖目标高度。

## Anti-patterns

- 不要把本任务拆成多轮 `goto_xyzr`、`get_grasp_info_simple`、抓取/释放调用；这会回到慢路径。
- 不要沿用上一轮的检测结果；每轮必须重新检测当前目标。
- 不要在非视觉失败后继续尝试低层动作；复合工具已经到达不确定硬件状态。
- 不要手动加减 tool offset；`goto_xyzr` 接受 TIP frame，内部会处理 flange offset。
