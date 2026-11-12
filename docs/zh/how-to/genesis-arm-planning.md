# Genesis 机械臂规划与夹持

cuRobo V2 规划；现有 Genesis 场景独占电机、物理时钟和实测停止。底盘须先停稳，不支持移动全身规划。ArmMotion/EndEffectors/GraspEvidence 无机型分支（底盘轮系仍为 MOZ1 专用），MOZ1/Panda 关节、工具、增益及碰撞来源仅在 YAML 中，不可作为其他机器人或真实硬件的默认参数。

## 安装和入口

专用环境执行 `pip install -e '.[genesis,genesis-planning]' --extra-index-url https://download.pytorch.org/whl/cu130`。另行安装 [cuRobo V2](https://github.com/NVlabs/curobo/tree/78fd485fa82d9b9a063fb4985e371814587e666a)：检出该 commit，按上游 CUDA 安装说明构建（验证版本 `0.8.0.post1.dev43`）；使用 `MotionPlanner`/`MotionPlannerCfg`，不兼容旧 `MotionGen`。源码许可证为 Apache-2.0；本仓库/wheel 不分发它。core 不加载 CUDA 依赖，dev 包含 CPU python-fcl 碰撞测试。
设置 `MOZ1_XRDF` 指向与实际 URDF 匹配的外部 XRDF；`CUROBO_FRANKA_CONFIG` 指向该 cuRobo checkout 的 Franka 配置。配置字段见[后端参考](../reference/genesis-physical-scene.md)。外部机器人资产需自行取得并验收，不随本仓库分发。加载器记录来源 SHA-256/忽略理由，支持 JSON/YAML、相对路径和环境变量。
```python
import json
from pathlib import Path
from jiuwensymbiosis.simulators.genesis.planning_profile import load_profile
out = Path("outputs/arm-planning/profile.json").resolve()
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(load_profile(Path("configs/simulators/planning/moz1.yaml")), indent=2))
```
通过 Python `PreviewSettings`/`GenesisPreview.configure` 传入 `planning_profile=str(out)` 和 `robot_mode="moz1_wheels"`；后台不继承任意环境变量。GUI 尚无规划配置入口，固定臂加载模式尚未实现；`panda.yaml` 仅为模型/规划校验参考。新机器人仍需物理控制器及模型/FK/碰撞/执行验收。MOZ1 夹爪要求资产有有效限值（示例 effort=16.5、velocity≥0.6、lower≤-0.61）；旧零限值导出被拒绝，不能绕过校验。

## 动作和恢复

仅已接纳配置的 ground-truth 会话开放 `motion.joint`/`motion.tool_pose`，有夹爪再开放 `grasp.parallel`；纯视觉/RGB-D 不开放。`sim_get_robot_state` 返回实测关节、工具和逐夹爪证据。`move_joint(targets)` 为命名绝对 rad/m，省略关节保持；`move_tool_poses(group, poses, speed=1)` 要求组内所有工具的 `[x,y,z,qw,qx,qy,qz]`（配置的 planning base_link、米、单位四元数）；`open_gripper(width_mm=80, *, effector=None)` / `close_gripper(force_n=None, *, effector=None)` 与真机共用动作词汇；多夹爪必须指定配置名，旧单夹爪适配器保留位置参数调用并拒绝命名选择。Genesis 忽略宽度/力提示。命名动作按 `payload.clear:<名称>` / `payload.held:<名称>` 检查与更新；整机 held 表示任一手持物，clear 必须所有手实测空手，未知仍为未知，不把闭合当成抓取成功。
规划使用独立 CUDA 进程，取消或90秒超时终止进程并丢弃过期结果；RPC 共用排他所有权与续租。五次样条检查位置及解析导数极值，超限延长时间而不裁剪路径；电机不瞬移。执行绑定场景、模型、基座、非规划关节、负载和静止障碍物；失败制动并实测。关节容差0.01 rad/m、速度0.02 rad/s或m/s、稳定0.25 s；工具误差≤5 mm/0.05 rad。结果区分 plan_success/goal_reached/settled/stop_confirmation。
未知停止锁住后续动作/控制权；未知持物阻止规划、底盘移动和复位。先查看状态中的 reason/逐夹爪证据并保留 状态目录（默认 `~/.jiuwensymbiosis/genesis-preview`）下的 `arm-plans/<scene_instance_id>/<plan_id>/` 及同层 `execution-<uuid>.json`（每次复位新建实例目录）。若能重新测得稳定支撑，可显式释放；若读回持续失败、物体消失或停车无法确认，操作员须停止并重启仿真后台，再重新验收世界。没有清除证据的旁路，也不能由 LLM 自动张开或退离。官方宿主禁用自动 Recovery/HOME；直接复用 Session 的宿主也须排除 home 并禁用自动恢复，自动回家尚未实现。证据不自动删除，停止后台后由操作员归档清理。
执行器声明 tool、独立 joints、至少两组不重叠 contact_groups；电机 open/close/velocity/effort/kp/kd/tolerance 及可选 preload 必须满足模型限值，并与规划组分离。mimic follower 保持被动；可选 noslip_iterations（示例5）为实验参数。Agent/脚本决定顺序：到位→闭合→接触0.15 s→试抬升≥15 mm并保持相对位姿→机械臂带载移动→支撑释放。预紧仅施加一次；取消保持夹持。
开放到位、无夹持接触、外部向上支撑和稳定速度持续0.25 s才确认放置；掉落撤销确认。接触丢失0.1 s、相对平移>3 mm/旋转>0.05 rad、非夹取链节接触或穿透>2 mm均拒绝成功。每夹爪仅跟踪一个刚体；共享对象双手操作、吸盘、柔性手、铰接载荷不支持。
碰撞球只作近似和覆盖诊断；FCL检查实际加载几何及引擎允许碰撞对，外部忽略列表不能绕过。仅配置支撑链节可触地；自由支撑关节须为连续、无下游关节且原点球形碰撞的叶节点。动态障碍须静止并持续监测；负载使用实测工具相对变换/网格，规划器用保守球。已测固定支撑可从粗模型移除，FCL仍只允许负载与它近竖直法向、穿透≤1 mm且接触数未截断的接触，本体不豁免；新支撑面可能被拒绝。离散采样和SDK adjacent/neutral过滤不是连续碰撞或稳定接触证明。历史单物体实验不代表本提交、双臂独立持物、LLM自主任务或硬件验收。

为持续检测滑移，逐 tick 读取夹爪接触与负载状态；大型场景开销尚未测量，当前不承诺实时性能。
