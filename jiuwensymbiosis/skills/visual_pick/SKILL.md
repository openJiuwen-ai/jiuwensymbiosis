---
name: visual_pick
description: 视觉引导抓取技能 —— 用相机识别目标物体并抓起（平行夹爪 / 吸盘 / 双臂按机器人能力自动选择），结束时物体已被抓住、悬于搬运高度，等待与 visual_place 衔接。
requires: [payload.clear]
provides: [payload.held]
invalidates: [body.home]
invalidates_locations: true
---

# visual_pick — 视觉引导抓取

## 何时启用

同时满足：

1. 用户指令含"把 X 抓起来 / 夹住 / 吸住 / 拿起来"等抓取意图，X 可由一段文本（颜色 / 形状 / 品类…）描述。
2. `api.capabilities` 含 `vision.detection`，且含**至少一种抓取能力**：`grasp.parallel` / `grasp.suction` / `grasp.dual_arm`。
3. 已注册 `robot_control`；当前末端**空载**。

任务若还要把物体放到某处：**先用本 skill 抓起，再 chain 调用 `visual_place`**，两者解耦。

## 检测目标来自用户任务

要抓的物体名从用户这句任务里识别（颜色 / 形状 / 大小 / 类别 / 材质 / 位置等特征的自然语言组合，也可能只有类别），作为检测动作的 `object_name`。用户只说任务、不另传参数；**不要**套用本文示例里出现过的任何物体名。

## 动作映射表（按你的能力选行，占位符照此替换）

下文用 `<检测目标>` / `<抓取>` / `<释放>` 这类占位符指代动作。**按【机器人能力】选你那一行**，替换成【可用动作】清单里真正存在的那个名字；清单里没有的变体一律不能输出。

| 你的能力 | `<检测目标>` | `<抓取>` | `<释放>` |
|---|---|---|---|
| `grasp.parallel` | `get_grasp_info_simple` | `close_gripper` | `open_gripper` |
| `grasp.suction` | `get_grasp_info_simple` | `activate_suction` | `deactivate_suction` |
| `grasp.dual_arm` | `locate_for_grasp` | `dual_arm_grasp` | `dual_arm_place` |

会移动的本体（`motion.base`）另有一个占位符：

- `<搜索并驱近目标>`（`motion.goal`）→ `approach_for_grasp`：**一步完成**"看不到就原地扫掠搜索 → 按感知方位转正 → 驶入作业带边驶边复检对中"。收敛后缓存最新检测，供 `<抓取>` 直接使用。

> **`_for_grasp` 是按「接下来要抓」选的，不是按「那东西是不是物体」选的。**
> 要判的是**抓取点**（近面在哪、面法向朝哪、贴多近夹得住）。同一个盒子，
> 要搬走它时用 `_for_grasp`，要往它上面放东西时用 `_for_place`（见 visual_place）。

### 用参照物锁定「哪一个」（`reference` + `relation`）

任务里常带一个**空间关系**来指认目标：「棕盒**上**的白箱」「帽子**旁边**的盒子」。这时给检测 / 搜索类动作加两个参数：

- `reference`：参照物的英文名（**只在任务真的点了参照物时才加**——写一个现场没有的参照物会让检测直接失败，不会退回普通搜索）。
- `relation`：`object_name` 相对 `reference` 的关系，取值**只能**是 `on` / `under` / `beside` / `near`，默认 `on`。

| 任务里的说法 | relation | 例子 |
|---|---|---|
| 在…上面、放在…上 | `on` | 「棕盒上的白箱」→ `object_name="white box", reference="brown box", relation="on"` |
| 在…下面；**有…在上面的那个** | `under` | 「放着杯子的那张桌子」= 杯子下面的桌子 → `object_name="table", reference="water cup", relation="under"` |
| 旁边、挨着、紧邻 | `beside` | 「帽子旁边的盒子」→ `object_name="box", reference="hat", relation="beside"` |
| 附近、周围、那一带 | `near` | 「门口附近的箱子」→ `object_name="box", reference="door", relation="near"` |

- `beside` 要求两者**高度相当**（挨在同一张台面上）；只是水平方向离得近、高度差很大时用 `near`。
- 关系一律读作「`object_name` <relation> `reference`」——方向别写反。
- 只有这四个值合法，`left_of` / `in_front_of` 这类**依赖观察视角**的说法不在其中（机器人一移动就不成立了）；遇到时改用最接近的合法值或干脆不加参照物。

## 抓取方式（按你的抓取能力选一支，选定后只看你这一支）

### 支 A — 夹爪 / 吸盘臂（`motion.cartesian` + `grasp.parallel` 或 `grasp.suction`）

用 `get_grasp_info_simple(object_name)` 一站式拿抓取点，**不要**用 `analyze_scene` + `pixel_to_base_xyz` 手算。返回的 `grasp_z` 是**已算好**的夹取高度（顶面下方合适深度、且不低于桌面）——**下降一律降到 `grasp_z`，别自己拿 `position.z` 加减**。

| # | action | params | 目的 |
|---|---|---|---|
| 1 | `home` | `{}` | 回拍照位姿，给视觉稳定的深度基线。 |
| 2 | `<释放>` | `{}` | 先把末端置空载。 |
| 3 | `<检测目标>` | `{"object_name": "<目标>"}` | 拿 `x,y` 与 `grasp_z`。 |
| 4 | `goto_xyzr` | `{"x":x,"y":y,"z":"grasp_z + 40"}` | 到目标正上方（approach ≈ +30~50mm）。 |
| 5 | `goto_xyzr` | `{"x":x,"y":y,"z":grasp_z}` | 降到 `grasp_z`（检测给的，别改）。 |
| 6 | `<抓取>` | `{}` | 到位后闭合 / 开吸。 |
| 7 | `goto_xyzr` | `{"x":x,"y":y,"z":"grasp_z + 60"}` | 提到搬运高度（lift ≈ +50~80mm）。 |

**腕部相机(eye-in-hand)**：若接着要 `visual_place`，请在 home 处**一次性**把"要抓物"和"放置目标"两个坐标都 `get_grasp_info_simple` 读好再移动——移动后腕部相机就拍不到放置目标了。

#### 支 A 的 Fast 闭环特殊动作（仅在【特殊动作】清单中出现时使用）

fast 编译器会另外给出本次机器人可用的【特殊动作】。这些动作不是所有机器人都有；**只有名字明确出现在清单中时**才能使用，并按以下优先级改写上面支 A 的标准流程：

1. 若有 `track_grasp`，优先用它把步骤 3～5 合并成一个持续视觉闭环步骤：

   ```json
   {"op": "track_grasp", "params": {"object_name": "<目标语义>", "approach_mm": 40}, "bind": "<目标变量名>"}
   ```

   - `track_grasp` 持续检测目标，以绝对 base-frame 抓取点完成 approach 和下降到 `grasp_z`。
   - `approach_mm` 必须是 30～100 mm 的数字字面量；通常取 40。
   - 必须带 `bind`，供后续 lift 步骤读取 `<bind>.x`、`<bind>.y`、`<bind>.grasp_z`。
   - 使用后**不要**再生成步骤 3 的 `<检测目标>` 或步骤 4～5 的两个 `goto_xyzr`；步骤 1～2、6～7 仍保留。

2. 否则，若有 `track_detect`，用它替换步骤 3 的 `<检测目标>`：

   ```json
   {"op": "track_detect", "params": {"object_name": "<目标语义>"}, "bind": "<目标变量名>"}
   ```

   `track_detect` 持续跟踪目标并绑定最新检测结果；之后仍按步骤 4～7 执行 approach、下降、抓取和 lift。

3. 若两者都没有，完整使用支 A 标准流程，不要臆造任何特殊动作。

以上选择规则属于本 skill 的抓取策略；其它 skill 是否使用这些特殊动作，只能以各自 SKILL.md 为准。

### 支 B — 双臂（`grasp.dual_arm`）

`<抓取>` 是复合动作（对位 → 夹紧 → 力确认打包成一步），**自动使用最近一次成功检测的结果**——检测步跑完直接调它，几何字段不必回传，参数省略即可。`<释放>` 属于 visual_place，不在本 skill 出现。

目标的获取按【有无底盘】走对应的**一条完整流程**：

#### 有底盘（`motion.base`）

目标常**不在正前方、或较远够不到**，所以**必做**下面这串（`<搜索并驱近目标>` 打头就是"看不到就搜索"；**不是可选步、不能跳过**）。
**`<搜索并驱近目标>` 是目标获取的唯一入口**：它内部完成搜索（原地扫掠）、转正、边驶边复检，并把收敛的检测留给后续 `<抓取>` 自动使用。**绝不要在它前面再加一个 `<检测目标>`/bind 去"先检测目标"**——那是下面「无底盘」分支的做法；有底盘时目标常在远处、起点根本核验不了，`<检测目标>` 会当场失败并让整条序列退出，搜索永远等不到。

| # | action | params | 目的 |
|---|---|---|---|
| 1 | `<搜索并驱近目标>` | `{"object_name": "<目标>"}`（**若任务用参照物指认了目标**，如"棕盒上的白箱""帽子旁边的盒子"，按上面的表加 `"reference"` + `"relation"`；参照关系**只并入这一步**，不要另开检测步） | 搜索→对准→驶入抓取带，一步到位。失败 `object_not_found` / `too_close` 等则**不抓**、转「失败处理」。 |
| 2 | `<抓取>` | `{}` | 双臂夹取（用上面收敛的检测）。 |
| 3 | `lift_to_clearance` | `{}` | （若含 `motion.lift`）抓住后抬到搬运净空高度。 |

#### 无底盘 / 固定本体

本体不能移动、无法转身搜索或驶近，只用当前视野：

| # | action | params | 目的 |
|---|---|---|---|
| 1 | `<检测目标>` | `{"object_name": "<目标>"}` | 检测当前视野内目标；不在视野则失败、**不硬抓**、转「失败处理」。 |
| 2 | `<抓取>` | `{}` | 双臂夹取。 |
| 3 | `lift_to_clearance` | `{}` | （若含 `motion.lift`）抬到搬运净空高度。 |

## 结束状态

末端抓住目标、悬于搬运高度。**不要**在本 skill 末尾释放，也**不要**再 `home`——把后续动线交给 `visual_place` 或上层。

## 失败处理

任一步返回失败，先记一句错因（不念堆栈），再**按载荷状态**处理（"释放"= 你这支自己的释放 / 回位动作）：

- **已持物**（抓取动作已成功 / `holding_payload=true`）：**禁止释放**。保持夹持，由 RecoveryRail 安全回 home；若恢复失败，立即上报并等待人工处理。
- **未确认持物**（`holding_payload=false` 或状态不可用）：恢复流程会先尝试**释放**再回 home，避免接触检测假阴性时闭爪拖件。
- **载荷未知**（例如抓取动作本身失败）：保守执行一次**释放**，再回 home。
- **运动下发前被拒绝**（参数 / IK / 安全检查失败，未发生实际运动）：不释放、不回 home；修正参数或换路径。
- RecoveryRail 已完成的释放或 home **不要重复执行**。
- 最后报"第 N 步（动作）失败：<原因>"，不要原样重试。

检测 `score < 0.4` 视为识别失败：把目标描述换得更具体一次（补颜色 / 形状 / 大小 / 位置），仍失败则放弃并报告。

## 与 Rails 的协作

- **SafetyRail**：在运动下发前拒绝越界 / 越过 `z_min_safe` 的目标；其 `ValueError` 按「失败处理」，**不要**吞，也不要为此释放或回 home。
- **VisualFeedbackRail**：`motion` 工具后自动注入观测，**不要**重复 `get_image`。
- **RecoveryRail**：按载荷三态恢复——持物运动失败时保持夹持并回 home，明确空载时不重复释放，载荷未知时才保守释放；预派发拒绝不触发恢复。**不要**重复它已完成的动作。

## Anti-patterns

- ❌ 跳过 `home` 直接检测（支 A）：相机基线不稳、深度噪声大。
- ❌ 在本 skill 末尾释放：物体会掉回原位，visual_place 拿到空末端。
- ❌ 把放置点动作写进本 skill：放置是 visual_place 的职责。
- ❌ 双臂（`grasp.dual_arm`）有底盘时在 `<搜索并驱近目标>` 前面加 `<检测目标>`/bind"先检测目标"：目标常在远处、grounded 核验必失败即退出，搜索永不执行。有底盘只靠 `<搜索并驱近目标>`，`<抓取>` 自动用其检测。
- ❌ 失败后不判断载荷就释放或重复 home：已持物时释放会丢件，也会制造与 RecoveryRail 重复的多余动作。
- ❌ 输出不在你能力/动作清单内的动作（含未列入【特殊动作】清单的 `track_grasp`/`track_detect`）。
