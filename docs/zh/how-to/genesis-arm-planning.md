# Genesis 机械臂规划与夹持

cuRobo V2 规划；现有 Genesis 场景独占电机、物理时钟和实测停止。底盘须先停稳，不支持移动全身规划。ArmMotion/EndEffectors/GraspEvidence 无机型分支（底盘轮系仍为 MOZ1 专用），MOZ1/Panda 关节、工具、增益及碰撞来源仅在 YAML 中，不可作为其他机器人或真实硬件的默认参数。


## 当前验收边界（2026-11-20）

CPU 物理（八子步、noslip=5）＋ GPU Nyx FastPathTracer 32 spp 的单次 Chrome 验证中，MOZ1 从双臂对称收拢姿态抓取并抬升约 8.2 cm，通过 2 秒保持确认。Agent 随后返回空回复，未执行放置；持续悬空约 48 秒后，相对平移超过 3 mm，触发 `payload_slip`，但双侧接触和闭爪目标仍存在。该次完整抓放闭环未通过，不能据此宣称稳定抓取或放置已验收。原始录像、状态和耗时保存在本地 `outputs/simulation-clean-verify-20261120/`，不随仓库分发。

## 安装和入口

专用环境执行 `pip install -e '.[genesis]' --extra-index-url https://download.pytorch.org/whl/cu130`。另行安装 [cuRobo V2](https://github.com/NVlabs/curobo/tree/78fd485fa82d9b9a063fb4985e371814587e666a)：检出该 commit，按上游 CUDA 安装说明构建（验证版本 `0.8.0.post1.dev43`）；使用 `MotionPlanner`/`MotionPlannerCfg`，不兼容旧 `MotionGen`。源码许可证为 Apache-2.0；本仓库/wheel 不分发它。core 不加载 CUDA 依赖。
设置 `MOZ1_XRDF` 指向与实际 URDF 匹配的外部 XRDF；`CUROBO_FRANKA_CONFIG` 指向该 cuRobo checkout 的 Franka 配置。配置字段见[后端参考](simulaiton-guide.md)。外部机器人资产需自行取得并验收，不随本仓库分发。加载器记录来源 SHA-256/忽略理由，支持 JSON/YAML、相对路径和环境变量。
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

真值快照中的 `robot.effectors[<名称>].entity_id` 使用与 `sim_list_entities`、`sim_get_entity_state` 相同的逻辑实体 ID。底层数值 `object_id` 只保留在物理证据中，不出现在该快照里。未注册的持物返回 `entity_id=null`，不改变实测持有状态。释放后仍可保留此 ID 用于放置核验，因此应读取 `holding_payload` 和 `placement_confirmed`，不能仅凭 ID 判断正在持物。
规划使用独立 CUDA 进程，取消或90秒超时终止进程并丢弃过期结果；RPC 共用排他所有权与续租。五次样条检查位置及解析导数极值，超限延长时间而不裁剪路径；电机不瞬移。执行绑定场景、模型、基座、非规划关节、负载和静止障碍物；失败制动并实测。关节容差0.01 rad/m、速度0.02 rad/s或m/s、稳定0.25 s；工具误差≤5 mm/0.05 rad。结果区分 plan_success/goal_reached/settled/stop_confirmation。
未知停止锁住后续动作/控制权；未知持物阻止规划、底盘移动和复位。先查看状态中的 reason/逐夹爪证据并保留 状态目录（默认 `~/.jiuwensymbiosis/genesis-preview`）下的 `arm-plans/<scene_instance_id>/<plan_id>/` 及同层 `execution-<uuid>.json`（每次复位新建实例目录）。若能重新测得稳定支撑，可显式释放；若读回持续失败、物体消失或停车无法确认，操作员须停止并重启仿真后台，再重新验收世界。没有清除证据的旁路，也不能由 LLM 自动张开或退离。官方宿主禁用自动 Recovery/HOME；直接复用 Session 的宿主也须排除 home 并禁用自动恢复，自动回家尚未实现。证据不自动删除，停止后台后由操作员归档清理。
执行器声明 tool、独立 joints、至少两组不重叠 contact_groups；电机 open/close/velocity/effort/kp/kd/tolerance 及可选 preload 必须满足模型限值，并与规划组分离。mimic follower 保持被动；`noslip_iterations` 是主求解器之后的抑制滑移/漂移后处理，Genesis 默认 0，官方建议操作类任务设为 5（本仓库 `moz1`/`panda` 配置取 5）；它是实验特性且会降低仿真速度，`friction_cone` 的 `elliptic` 模式与之不兼容。Agent/脚本决定顺序：到位→闭合→接触0.15 s→试抬升≥15 mm并保持相对位姿→机械臂带载移动→支撑释放。预紧仅施加一次；取消保持夹持。
开放到位、无夹持接触、外部向上支撑和稳定速度持续0.25 s才确认放置；掉落撤销确认。接触丢失0.1 s、相对平移>3 mm/旋转>0.05 rad、非夹取链节接触或穿透>2 mm均拒绝成功。每夹爪仅跟踪一个刚体；共享对象双手操作、吸盘、柔性手、铰接载荷不支持。
cuRobo 使用配置的机器人碰撞球与场景障碍近似进行自碰撞和环境避障规划；规划成功后直接交给 Genesis 执行，不再独立做 FCL 初态、端点或路径复核。自由支撑关节须为连续、无下游关节且原点球形碰撞的叶节点。动态障碍须静止并持续监测；负载按实测工具相对变换生成保守球。抬升时从规划障碍中排除已测接触支撑，以允许负载离开支撑面；物理接触仍由 Genesis 处理。碰撞球和障碍包围盒是近似模型，规划成功不等于实际网格无碰撞或抓取成功；后者由执行后的实测反馈报告。

cuRobo 使用规划配置自身的 `self_collision_ignore`，不追加独立几何复核。

持物规划为每个配置夹爪预留独立的固定负载连杆和一个保守球，空载时球半径为 cuRobo 约定的负值；持物时按实测工具相对变换更新包围球。只有该负载与已命名抓持接触链节之间的粗球检查被豁免；工具自身的碰撞球不受影响。这些连杆仅存在于规划模型，不修改机器人资产或物理世界。固定槽位保持 CUDA 张量形状，抓取/释放不必重新预热规划器。

为持续检测滑移，逐 tick 读取夹爪接触与负载状态；大型场景开销尚未测量，当前不承诺实时性能。

规划工作进程由当前场景持有，串行处理请求并复用 cuRobo 预热结果；每次使用最新状态、模型和障碍规划。非规划关节的实测变换和世界障碍通过原地更新刷新；负载球和缓存容量以内的障碍增减原地更新；模型、规划组、碰撞拓扑或障碍缓存容量变化时重建后端。取消、超时、世界释放会终止进程，后续请求使用新的进程。每次请求保留独立结果文件，`timing` 记录工作进程 PID、初始化是否复用、准备/规划及轨迹时长；结果原子发布，未完成文件不参与执行。`collision_validation.method=curobo_motion_planner` 表示由规划器完成避障，不表示执行过独立网格复核。

### 仿真抓取的带避障执行

`pick_target` 保留现有点云来源和 GraspGenX 姿态候选。cuRobo IK 筛选预抓取位、抓取位与抬升位的可达性；选定候选后分别调用 cuRobo MotionPlanner 规划“当前位置到预抓取位、预抓取位到抓取位、抓取位到抬升位”。每段由 cuRobo 完成自碰撞和环境避障规划，成功即交给 Genesis 受限关节控制跟踪执行，中间没有独立 FCL 复核。运动开始后失败直接返回，不换候选继续运动。桌面演示配置 `moz1-graspgenx.json` 的双臂初态左右对称收拢，不预置右臂到桌面附近。

到抓取位后，先确认感知位置处的目标仍受支撑，建立负载碰撞模型并规划抬升；规划通过才闭合夹爪并执行抬升。只有配置的夹持链节允许与负载接触，负载与环境及其他机器人链节的检查保留。抬升仍需双侧接触、离开支撑面、达到高度并稳定持有的实测证据；闭合命令本身不算成功。

IK 和轨迹规划各自由场景拥有的常驻子进程串行处理，复用初始化结果；二者都使用已安装 NVIDIA 运行库的匹配 NVRTC 搜索路径。候选 IK 不代替运动规划。执行记录标记 `execution_mode=curobo_planned`；`ik-*.json` 和每段 `result.json` 分别保留求解、初始化复用、碰撞检查和轨迹耗时。Linux 下 Genesis 物理固定使用 CPU，Nyx 渲染和 cuRobo 求解仍使用 CUDA；MOZ1/Panda 配置恢复为 `noslip_iterations=5`。修改已解析的运行配置后，重新加载仿真后台才能应用物理求解器设置。

机器人加载时先为 `grasp_pose_backends` 声明的规划组预热常驻 MotionPlanner，再用当前工具位姿预热 IK；只计算，不下发轨迹。预热完成后才发布 `arm_planning_ready`，加载阶段的 `arm_planning_warmup` 和 `warmup.json` 单独记录这项开销。后续请求按组复用后端，仍更新当前关节和障碍。`prepare_s` 包含 `backend_create_s`、`warmup_s` 等准备开销；`solve_s`、`postprocess_s` 和 `plan_including_prepare_s` 在成功与异常路径均保存。IK 另存每次求解和请求总耗时；执行记录保存 `command_wall_s` 与 `execution_and_settle_s`。初始化只是提前到加载阶段，并未从总耗时中消失。

MOZ1 的 `collision_source.adjacent_self_collision_pairs` 对齐 RoboSim 的肩腰相邻球关系：`waist03` 与 `left02/03`、`right02/03`。这些球存在设计重叠，排除关系属于机器人模型配置，并记录在 `collision_provenance`；其余自碰撞及环境避障仍由 cuRobo 执行，不增加 FCL 复核。

`moz1-graspgenx.json` 通过 `collision_sphere_overrides` 引用 `moz1-gripper-spheres.yaml`，仅替换双侧末节夹指和与夹爪固定连接的腕部/掌部几何。球体由 cuRobo 官方 MorphIt 工具根据实际加载的碰撞几何拟合；左右夹指的物理凸分解策略不同，分别拟合，不直接镜像。无实体几何的 `*_gripper_base_link` 球体并入对应 `left07/right07` 的拟合结果，工具坐标系和控制参数不变。配置记录 0.15–0.50 mm 的夹指覆盖补偿、来源和拟合参数；不增加自碰撞排除。

覆盖文件绑定原始碰撞源 SHA-256 和 prepared model SHA-256。源文件、覆盖文件或准备后的机器人模型变化时拒绝复用，需重新拟合并导出解析后的规划配置。该绑定适用于上述抓取配置的碰撞策略；普通 `moz1.yaml/json` 仍使用原始 XRDF。球体拟合是近似，静态几何数据不能替代 Chrome 抓放验收；完全空爪闭合仍存在对向夹指球重叠，详见保留的几何报告。

静止关节绑定检查允许夹爪从动关节偏移 0.02 rad，其余非执行关节保持 0.01 rad；执行记录保留从动关节最大偏移。它不改变物理接触与稳定持物成功判据。


### 独立 Nyx 渲染循环（Linux）

Genesis CPU 物理与动作控制由一个线程拥有；Nyx 在进程主线程独立运行，图像回读、NVENC 编码和视频发布也在该线程执行。Nyx 与 cuRobo 使用 GPU，物理使用 CPU；不提供串行渲染或 GPU 物理切换开关。目前只支持刚体场景；不支持的可变形几何会明确拒绝加载。

视频只渲染界面当前选中的相机；选择其他相机后更换渲染目标。使用 Nyx 原生 `render_frame(camera_index)` 的单环境单相机接口，输出从共享颜色缓冲起始位置读取，不使用插件全相机批量渲染的偏移。纯相机选择只更新视频代次与编码器，不改变世界配置代次或取消正在执行的动作。Agent 按需取图只渲染请求的相机一帧，随后视频继续使用界面选中的相机。FastPathTracer、32 spp 及现有镜头校正保持不变。

物理线程在完成步进和 GPU 状态复制后，发布不可变的几何及相机状态。交接槽只保留最新状态；渲染中的旧缓冲由引用保活，消费完成后释放。复位、场景切换、RGB-D 抓拍和销毁通过主线程屏障排空视频工作，避免读到一半更新的物理状态。按需 RGB-D 仍等待同一物理状态的 RGB、深度及位姿完成。

每帧 `step`、`sim_time = step * dt_s`、`reset_generation` 来自同一状态缓冲。重复渲染旧状态不会推进仿真时间；WebRTC 播放用独立的单调墙钟时间戳，避免重复物理状态产生重复播放时间戳。这对应固定物理时间步、独立渲染频率的时间语义，不意味着物理与渲染帧率相同或 Genesis 与 Isaac Sim 的内部实现相同。

状态接口保留 `physics_clock`、`video_schedule.recent_source_fps` 和 `fresh_state_fps`；`frame_clock` 记录物理状态完成、快照就绪、渲染和编码完成时间、状态年龄及落后物理步数。`timings_ms` 记录快照、渲染、回读、编码、发布和状态年龄的 P50/P95。页面中的“源帧状态延迟”从物理状态完成计算到源帧编码完成，不包含网络传输及浏览器解码/显示排队；浏览器呈现 FPS 单独统计。
