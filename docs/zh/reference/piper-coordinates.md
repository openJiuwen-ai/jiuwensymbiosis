# Piper 坐标、安全边界与迁移说明

本页描述 Piper 工具偏移修复后的契约。位置单位为 mm，角度单位为度；TIP 是工具尖端，FLANGE 是法兰原点，两者的位置均以基座坐标系表达。

## 工具变换

`tool_offset_mm = L` 是沿 **flange-local +Z** 的非负工具长度。所有读写和驱动安全检查共用 `adapters/piper/geometry.py:tip_offset_in_base`：

```text
R = Rz(rz) @ Ry(ry) @ Rx(rx)
offset = R @ [0, 0, L]
tip = flange + offset
flange = tip - offset
```

这是 extrinsic xyz 约定，即 SciPy `Rotation.from_euler("xyz", ..., degrees=True)`。厂商 SDK 的矩阵转 RPY 公式与此一致，参见 [Piper SDK FK](https://github.com/agilexrobotics/piper_sdk/blob/master/piper_sdk/kinematics/piper_fk.py) 和 [SciPy 欧拉角约定](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_euler.html)。软件约定的一致性不等于安装工具及具体固件已经通过真机验收。

`goto_xyzr`、`servo_to_tip` 固定 `rx=180, ry=30`，保留指定或当前的 yaw。此时：

```text
offset = (-0.5*L*cos(rz), -0.5*L*sin(rz), -sqrt(3)/2*L)
```

例如 `L=95`，TIP 目标 `(200, 20, 100)`：yaw=0 时法兰为 `(247.5, 20, 182.2724...)`；yaw=90 时法兰为 `(200, 67.5, 182.2724...)`。`get_pose()` 使用实时的完整姿态，支持任意 rx/ry/rz。

## 接口和 home

| 接口 | 坐标语义 |
|---|---|
| `api.get_pose()` / `api.get_home_pose()` | TIP；home 使用存储目标的姿态进行变换 |
| `api.goto_xyzr()` / `api.servo_to_tip()` | TIP 目标；固定上述工具倾角 |
| `api.move_direction()` | 保持当前完整姿态进行基座轴向平移；预检和返回的 `pose` 均为 TIP |
| `env.get_observation().pose` | TIP，与规划状态、状态 CLI 和 trace 中的位姿一致 |
| `env.get_observation().extra["flange_pose"]` | 同一次采样的原始 FLANGE 位姿 |
| `env.get_flange_pose()` / `env.home_pose` / `driver.home_pose` | FLANGE |
| `env.move_to_flange()` / `env.servo_to_flange()` | FLANGE 目标，驱动再检查工具尖端 |

没有校准锚点时，`home_pose_xyzrxryrz_mm_deg` 已是 FLANGE 配置。有旧版 `object.xyz_base_mm` 锚点时，home 的法兰 Z 保持旧物理目标：`anchor_z + home_lift_mm + L`，XY 为锚点位置，姿态继承连接时的读数。`home_use_init_pose=True` 优先选择连接时的完整法兰位姿。摄像头标定文件是否存在，不决定 home 的坐标语义；是否有锚点才决定配置来源。

`get_home_pose()` 只读取配置目标，不驱动硬件；未连接或已断开时抛出说明未连接的 `RuntimeError`。`home()` 下发存储的法兰目标，并经过与其他笛卡尔运动相同的检查。超出这些目标限制的 home 会被拒绝。

初始化会提前诊断 home 是否满足同一套目标边界及指令量化要求：静态配置 home 在 CAN 预检查前检查；启动快照或继承实时朝向的 home 在读数完成后检查。超界会警告具体原因，并提示普通及自动恢复回位将被拒绝，但保留连接用于观测或关节标定。配置格式、非有限值等错误仍会拒绝初始化。诊断不修改 home，也不替代每次 `home()` 的运行时检查；通过检查不保证 IK 可达或回位轨迹无碰撞。

`move_direction()` 从一次实时法兰采样推导目标，保留 rx/ry/rz，再换算 TIP 进行预检。`env.move_to_flange()` 与原始法兰伺服共用位姿归一化：接受原生 `FlangePose`、`Mapping`（含只读映射）及属性对象；字段为 `x/y/z/rx/ry/rz`，`r` 是 yaw 别名，同时提供时 `rz` 优先。阻塞运动要求完整位置和朝向；伺服只允许缺失朝向从一次实时读数补齐。完整目标的归一化不依赖回读。标定仍使用独立的法兰读写接口。

`api.servo_to_tip()` 要求映射中包含完整且非 `None` 的 XYZ；缺失字段会在回读和下发前明确报错。TIP 和 FLANGE 入口共用偏航别名解析，实时朝向的补齐仍由各入口按需执行。

观测中的 TIP `pose` 与 `extra["flange_pose"]` 成对发布，来自同一次法兰采样。读取或 TIP 转换失败时，两者均为 `None` 并记录诊断日志；相机和关节观测独立采集。

笛卡尔运动当前始终使用 MOVE_P，不承诺直线轨迹。驱动的 `joint` 参数仅为兼容旧调用保留，已废弃且不影响运动模式。

## 安全边界

- `z_min_safe` 始终是 **TIP 目标点的 Z 下限**。有锚点时取 `anchor_z + z_safe_margin_mm`；否则取 `z_min_safe_mm`，与 home 的存储方式无关。它不是整个机械臂的碰撞安全平面，也不必等于对象顶面。工具朝上时，只要 TIP 满足下限及其他目标限制，法兰低于该数值本身不会导致拒绝。
- 驱动按**目标姿态**计算 TIP 并检查 Z 下限；倾斜工具不再被全长 L 的固定补偿误拒绝。`get_grasp_info_simple()` 把 `grasp_z` 钳到该下限后，只要其他边界也满足，该高度可以下发。
- 配置的 X/Y 边界同时约束 TIP 和 FLANGE；`z_max_mm` 约束 FLANGE。单侧 `None` 关闭该侧检查；两侧都配置时须满足 `min <= max`，反向边界在 CAN 连接前被拒绝。越界在 `EndPoseCtrl` 之前抛出 `SafetyViolationError`，错误码为 `safety_rejected`，不再静默钳位。伺服循环将报告 `stopped` 和该错误码。
- `flange_z_min_safe = z_min_safe + L` 是全长补偿得到的保守参考值，用于满足 TIP 下限；它不是根据法兰或夹爪外形计算的避障高度。实际命令使用完整目标姿态，因此可接受比它低、仍满足 TIP 下限的法兰目标。
- SDK 指令精度为 0.001 mm / 0.001°。先检查原始请求，再检查量化后的实际目标；贴近 TIP 下限时，Z 按量化后的姿态向安全侧取整。若取整导致越过上限或 XY 边界，拒绝发送，不移动其他轴补偿。
- 原始法兰伺服支持 `FlangePose`、`Mapping`（含只读映射）及属性对象。映射/属性对象必须提供完整 XYZ，只允许省略的姿态字段从实时位姿补齐；回读失败时拒绝执行。缺少 XYZ 或无法识别的对象直接报错，不再用当前位姿替代请求；完整目标不依赖回读。
- 标定文件加载、有效 home 配置及 TIP 下限推导和有限性校验发生在 CAN 预检查、SDK 构造和使能之前。锚点覆盖的配置回退值不参与校验；`home_use_init_pose=True` 时不解析被覆盖的 home 配置，但仍校验有效安全下限。依赖实时姿态的初始化在连接后完成。

这些是笛卡尔**目标点**检查；它们不验证 MOVE_P 的整段轨迹、工具整体碰撞或关节运动的扫掠体积。完整碰撞检测需要机器人和环境几何模型、IK 构型及实际执行轨迹的校验；当前 Piper 的 IK 和 MOVE_P 运动由固件处理，本适配器不提供这些保证。

## 升级时的数值变化

1. 旧 `get_pose()` 近似返回 `(flange_x, flange_y, flange_z-L)`；现在返回 `flange + R@[0,0,L]`。在 `rx=180, ry=30, yaw=0, L=95` 时，同一物理构型的读数相对旧版变化为 `(-47.5, 0, +12.7276)` mm；yaw=90 时变化为 `(0, -47.5, +12.7276)` mm。使用旧读数记录的目标需要按来源和姿态重新核对。
2. 旧 `get_home_pose()` 直接暴露驱动存储值，随配置混合了旧版 home 坐标与 FLANGE。现在统一返回实际 home 的 TIP，物理 home 目标保持原值。需要法兰坐标的脚本应改读 `env.home_pose`。
3. yaw=0 的 `goto_xyzr` 偏移方向保持原实现；非零 yaw 现在旋转工具的 XY 偏移。检查脚本是否额外添加过旧版补偿，避免重复补偿。已有手眼标定矩阵不应仅因 API 读数变化而手工平移。
4. 无锚点的原始 FLANGE 运动现在也检查工具尖端；以前通过的低位目标可能被拒绝。边缘目标以前会被驱动钳位，现在报错，需要重新选择同时满足 TIP/FLANGE 边界的目标。
5. `get_observation().pose`、WorldState、状态 CLI 和新 trace 中的位姿从 FLANGE 改为 TIP。在 `L=135.8、RPY=(180,30,0)` 时，相对旧观测变化为 `(-67.9, 0, -117.60625)` mm；这与上面新旧 `get_pose()` 的差值不同。旧 trace 不自动转换，分析历史数据时须区分版本；新观测的原始法兰值可从 `extra["flange_pose"]` 获取。
6. `move_direction()` 修复了传参类型导致的驱动崩溃，并统一返回 TIP 目标；它保留当前完整姿态，不采用 `goto_xyzr()` 的固定倾角。
7. `servo_to_tip()` 同时收到 `r` 和 `rz` 时，改为优先使用非 `None` 的 `rz`，与法兰位姿归一化一致；`rz=None` 时使用 `r`，两者都缺失或为 `None` 时才回读当前 yaw。旧脚本若携带相互冲突的两个值，实际偏航目标会变化；`rz=0` 是有效目标。
8. fast 跟踪的 `track_detect` 按回读格式生成偏航目标：Piper 使用 `rz`，仅回读 `r` 的接口继续使用 `r`。这样入场偏航会参与角度限速和到位判断，避免伺服合并位姿时 live `rz` 覆盖目标 `r`。

无锚点时，旧驱动实际把 `z_min_safe_mm` 直接用于法兰 Z；SafetyRail 原本按 TIP 检查。现在两层统一为 TIP 下限，这修复了原始法兰运动绕过工具高度检查的问题。没有自动换算旧配置：升级时应按实际安全平面核对该值，同时检查配置 home 或启动 home 的 TIP 高度。

例如 `z_min_safe_mm=50`、`L=135.8`、`RPY=(180,30,0)` 时，法兰 Z=100 对应 TIP Z≈−17.606 mm，必须拒绝；满足该 TIP 下限的法兰 Z 至少为约 167.60625 mm，按 SDK 精度向上取整为 167.607 mm，还需满足其他边界及可达条件。RecoveryRail 的自动 home 也经过相同检查；失败会被捕获、记录为 `home_ok=False`，不会绕过安全限制，也不保证恢复成功。不要仅为维持旧运动范围而自动降低 TIP 下限。

连接完成后，可先只读检查两个坐标表示：

```python
flange_home = session.env.home_pose
tip_home = session.api.get_home_pose()
print("FLANGE home:", flange_home.as_tuple())
print("TIP home:", tip_home)
obs = session.env.get_observation()
print("TIP observation:", obs.pose)
print("FLANGE observation:", obs.extra.get("flange_pose"))
```

## 标定轨迹与工作框

GUI 内置 Piper profile 默认使用 `trajectory.space: joint`。示教采集关节角，执行时走关节命令，不经过本页的笛卡尔 XY/Z 检查；因此默认关节标定路径不受笛卡尔钳位改为拒绝的影响。`z_min_safe_mm` 也不构成对关节轨迹桌面净空的验证。

改用或导入 Cartesian 轨迹时，执行器按轨迹档案的空间分派，驱动会拒绝框外法兰目标，异常会中止该次采集。若已确认标定需要更大的工作区，应显式调整 Piper 的 `x_min_mm/x_max_mm/y_min_mm/y_max_mm` 及所需高度限制；SO-101 的 `workspace_bounds: null` 不是 Piper 的配置字段。旧版静默钳位并不保证回到示教位置。

## 真机验收记录

本次自动测试使用假 CAN，覆盖独立数值预期、命令量化、越界拒绝和伺服错误传播，不能证明物理工具轴或固件姿态解释正确。验收时应记录 SDK/固件版本、工具长度和局部轴方向；在确认可达且具有足够净空的区域，分步验证 yaw=0、正负 yaw（包括可达的 ±90°）下的工具位置。位置证据应来自独立测量或已验收的相机标定，不能仅依赖命令回显与 `get_pose()` 相等。仓库中的初始手眼参数及仅覆盖 yaw=0 的触碰记录不构成 yaw 扫描验收。
