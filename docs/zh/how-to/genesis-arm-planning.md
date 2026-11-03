# Genesis 通用机械臂规划

cuRobo V2 从机器人模型、完整关节状态和静态碰撞场景生成轨迹；现有 Genesis 场景负责电机控制、物理步进、反馈和停止确认。规划进程不创建 Genesis 场景。当前移动操作采用“底盘停稳 → 重新采样 → 规划机械臂”，不提供底盘与双臂同时运动的全身规划。

## 配置与环境

在专用 Genesis 环境安装 `.[genesis,genesis-planning]`。cuRobo V2 需单独安装并固定版本；此接入实测版本为 `0.8.0.post1.dev43`，接口为 `MotionPlanner` / `MotionPlannerCfg`，不兼容旧版 `MotionGen`。不将 CUDA 规划依赖引入默认 core 环境。

模板位于 `configs/simulators/planning/moz1.json` 和 `panda.json`。它们声明规划基座、任意数量的关节组及工具、显式速度/加速度/jerk/力矩限值、地面支撑链节和碰撞近似来源。限值是仿真调试参数，不是厂家硬件规格。新机器人需要匹配其 URDF 的配置，以及物理控制器提供的电机控制与读回；仅加载网格不能确定这些语义。

模板引用外部碰撞球资产，不复制资产内容。先设置 `MOZ1_XRDF` 或 `CUROBO_FRANKA_CONFIG` 为本机源文件路径，在启动后台前解析成独立文件：

```python
import json
from pathlib import Path
from jiuwensymbiosis.simulators.genesis.planning_profile import load_profile

out = Path("outputs/arm-planning")
out.mkdir(parents=True, exist_ok=True)
profile = load_profile(Path("configs/simulators/planning/moz1.json"))
resolved = (out / "profile.json").resolve()
resolved.write_text(json.dumps(profile, indent=2))
```

`load_profile` 支持 JSON/YAML；碰撞来源路径相对于配置文件解析，可展开环境变量。解析后的文件保留来源路径、SHA256 和忽略理由。**后台不继承任意环境变量**，所以传入解析后的文件，不依赖后台展开模板变量。

把 `planning_profile=str(resolved)` 传入 `PreviewSettings` 或 `GenesisPreview.configure`，同时使用已有的 `robot_mode="moz1_wheels"`。空路径保持原有底盘功能。Panda 已验证同一规划器、碰撞检查器和 `ArmMotion` 执行器的复用；GUI 的机器人加载器仍只有现有 MOZ1 移动控制模式，没有新增任意机器人一键加载入口。

## 公共动作与坐标

配置被实际场景接纳后，ground-truth Agent 才获得 `motion.joint` 和 `motion.tool_pose`。纯视觉与 RGB-D 感知模式本批不开放机械臂动作。

- `move_joint(targets)` 沿用共享动作，按名字指定绝对弧度值；省略的关节必须保持。规划器如果产生了省略关节的偏移，轨迹会被拒绝。
- `move_tool_poses(group, poses, speed=1.0)` 为一个组中的所有工具联合求解。`poses` 格式为 `{tool_name: [x,y,z,qw,qx,qy,qz]}`，长度单位米、单位四元数，参考系为配置的 **URDF `base_link`**。它不是底盘导航的 forward/left 坐标，也不是旧 `goto_pose` 的 mm/deg 格式。
- `sim_get_robot_state` 提供实际关节、组/工具名称、规划参考系和工具实测位姿。末端默认配置为真实链节；需要 TCP 时应在模型中明确添加并配置工具坐标。

结果区分 `plan_success`、`goal_reached`、`settled` 和 `stop_confirmation`。完整轨迹、几何检查、速度重定时和失败阶段保存在场景证据目录 `arm-plans/<scene>/<plan>/`。主机与 resident RPC 沿用现有排他运动所有权、请求身份、场景/配置绑定和续租机制；兼容旧传输字段 `base_motion`，用 `kind="arm"` 区分请求。

## 执行与失败语义

规划进程采用独立 CUDA 上下文，避免 cuRobo CUDA Graph 捕获干扰 Genesis 步进。取消和 90 秒规划期限会终止该规划进程；过期结果不进入执行。物理场景持续运行，仍只有一个物理时钟。

轨迹按名称映射电机，使用端点速度/加速度为零的五次样条。根据解析导数极值检查速度、加速度和 jerk，随后再次检查插值曲线的关节边界和碰撞。`speed=1.5` 是请求倍率；超出约束时延长时间，不裁剪轨迹、不保证每段都达到请求倍率。控制器仅写有限力矩的电机目标，不通过设置关节位置执行轨迹。

执行前校验完整测量状态、模型和场景身份；执行中监控关节跟踪、底座漂移和非规划关节变化。失败时保持测得的关节位置并制动，持续测量后才能确认停止。关节终点容差为 0.01 rad/m，速度阈值 0.02 rad/s 或 m/s，稳定窗口 0.25 s；工具动作还要求位置误差不超过 5 mm、姿态误差不超过 0.05 rad。未知停止结果会锁住后续动作和控制权，不能当成普通规划失败继续。

MOZ1 源模型的被动球形轮会在底盘停稳后自由旋转。`uncontrolled_support_joints` 仅允许连续旋转、无下游关节、碰撞为关节原点处球体的支撑叶节点；它们的朝向不影响支撑几何。独立检查使用包住实际网格的球体。该例外不适用于机械臂、躯干或任意未受控关节。

## 碰撞证据与范围

原有碰撞球不是全模型的保守包络。每次规划保留逐链节覆盖抽样报告；缺失球、未覆盖点不会被报告为安全证明。轨迹必须额外经过 FCL 对 **Genesis 实际加载的碰撞形状** 的检查，使用引擎实际碰撞配对表；外部球配置的忽略列表不能绕过该检查。允许明确列出的支撑链节接触地面，不豁免它们与其他障碍物的碰撞。

世界模型当前接受固定障碍物，用包围盒提供给 cuRobo、实际形状提供给独立检查；动态障碍物直接拒绝。路径检查包含插值中间点，是离散抽样检查，不是连续碰撞证明。引擎已有的 adjacent/neutral 配对过滤仍然生效，并随模型审计记录；“自碰撞开启”不表示所有链节对都参与检测。

本批不包含抓取确认、负载附着/释放、载荷恢复、动态避障、移动全身控制或硬件验证。新增机器人需重新完成模型/FK/碰撞/物理执行验收，不能凭配置文件宣称已支持真机。
