# Jumper 仿真接入

设计起点：Symbiosis `53fdba7`；Mac 基础适配与策略机器人均包含在本 PR 中；
KingKongRobotics/jumper `198686eac0d188815d0943e9fee50de415edc80a`。

用户需要在 M4 16 GB 上用现有 Symbiosis Agent 控制 Jumper。
官方模型是 MuJoCo MJCF，配有 ONNX 步态和 macOS 原生 FSM；先保留这套
物理与策略，避免把跨引擎迁移误当成资产格式转换。初版支持站立、前后、
横移、转向、停止和机载 RGB。本轮按用户要求增加跳跃与左右夹爪；舞蹈仍不开放。

资产索引新增显式 `jumper` 扫描器：配置根是用户下载的官方仓库，关联
MJCF、bundle.json、motor_config.yaml。仍只接受 library ID，不允许 HTTP
请求提供任意模型/动态库路径；扫描不执行上游代码。安装示例脚本只登记
资产，不修改上游文件，不覆盖现有资产根。

工作台保留 GenesisPreview 的进程、IPC、租约、视频、聊天所有权。
worker 按显式 robot_mode 创建 JumperRuntime；它独占 MuJoCo model/data、
FSM、ONNX 和 GL renderer。原 Genesis RuntimeScene 与 MOZ1 行为保留。
机器人控制在 200 Hz 更新，MuJoCo 以 bundle 的 1 ms 步长积分；显示帧率
与控制时钟独立。每个关节以名称匹配，22 个关节均执行 FSM 输出的目标和
增益，并按官方测量的扭矩速度曲线及热模型限幅。接触参数跟随官方
common/constants.py 的 HYBRID_COLLISION，保留原碰撞网格、质量及惯量。

JumperPolicy 读取 bundle 自带的本机扩展和所有合约/轨迹，不导入上游
训练栈。加载文件校验 bundle 摘要；使用新的 FSM 回放 reference.json，
另以本机 ONNX 对照录制动作。启动失败即不可控制。状态中公开策略来源、
bundle notes 和回放误差；这只证明消费链路一致，不证明实机控制可靠。

Agent 沿用 NAVIGATE_RELATIVE / ROTATE_BASE 及 motion.base 门控，不增加
第二套动作。RelativeMotion 的目标、取消、租约与成功语义共用；Jumper
声明自身的收敛容差与最小步态速度。机器人观测提供其真实 +X 前向姿态，
不套用 MOZ1 的根 -Y。成功必须真实到达并停车，失去租约只请求零速，
不能伪报停车。原始世界状态仍不进入视觉 Agent 上下文。

observer 是可移动外部相机；head_cam 对应 MJCF onboard 相机，保留镜头
视场和机身安装位姿，降采样到 640×480。帧在 worker 内渲染，软件 H264
沿用现有 WebRTC；重设画质仅替换 renderer，不重置物理状态。关闭在同一
线程释放 GL；部分初始化失败也清理资源。

验证覆盖：索引边界/缺失资产；策略摘要与参考回放；真实模型站立及双轴
平移/转向/停止；共享 Agent 工具调用及实测结果；两路相机、重设画质、
取消/租约；既有 Genesis、资产与工作台回归。没有模型服务凭证时，使用
可控模型驱动真实 Agent 工具循环，并明确区分该检查与在线 LLM 验证。

## 跳跃和夹爪扩展

共享词表新增 `JUMP`（`motion.jump`），执行一次机身已验收的固定跳跃，并要求实际腾空、落地、停稳。
夹爪复用 `OPEN_GRIPPER` / `CLOSE_GRIPPER`（`grasp.parallel`），通过可选 `side` 选择左右侧。
闭合到位不等于抓住物体；Jumper 没有载荷证据，不建立 `payload.held`，保留主仓的抓取确认规则。
单夹爪适配器仅接受默认侧，原有调用不变；内部 `set_gripper` 消息继续复用官方策略。
JumperApi/Env 仅在当前 worker 公布这两种能力时使用；只读模式不发放动作。

JumperMotion 复用现有 base_motion 请求、场景绑定、单所有者和租约；相对
行走仍交给 RelativeMotion。跳跃走 bundle 的模式键和 residual ONNX，夹爪
走 five_foot 模式与原生任务 hook（包括右侧镜像、开合端点、扭矩与增益）。
按键两条边必须落在不同的 200 Hz 控制 tick，不直接写关节目标。

原生扩展的 `operators` 开关只能在构造时指定。夹爪需要它来读取任务按键，
导航需要直接物理速度输入，因此只在已收臂并回到稳定 locomotion 时重建
FSM 来切换输入方式；物理 world、伺服热状态、模型权重和原始合约均保留。
这一选择避免修改上游合约、重编原生扩展或重做手柄轴映射。

闭爪在动作/会话结束后保留，停止不会自动松爪。换侧、行走和跳跃前必须
显式张开当前爪；打开后由原生 FSM 收臂，再开始下一动作。当前只验收空爪
开合和站立平衡，不承诺持物搬运。取消已触发的跳跃等待本次录制轨迹落地，
不启动下一跳；未触发的请求和过期租约不得产生新的模式按键。

结果保存实际最高机身高度、离地时间、支撑接触和爪关节反馈到私有证据；
模型仅收到成功/停稳、是否确认腾空和夹爪命令状态，仍不接收仿真真值坐标。
验收扩展覆盖两侧完整行程、跳跃前后继续导航、收臂交接、去重/失租/取消，
以及真实共享 Agent 工具和机载相机循环。


## 维护者检视后的职责收口

`preview_backends.py` 唯一声明预览模式、运行时工厂、平台渲染/编码、依赖组合与深度/真值/点选支持。
沿用 `robot_mode` 为兼容的预览模式选择字段，本体身份由索引资产提供。
主机与页面读取稳定的 `runtime_backend`/`backend_features`，展示名称不参与能力判断。

`MujocoPolicyRuntime` 拥有共用物理时钟、底盘观测、相机元数据和渲染生命周期；
Jumper FSM/电机热限制与 Microduck BAM/ONNX 物理步保留在各本体，避免混合控制器。
`SimulationRuntime` 声明 worker 接口；不支持的深度、真值与点选显式报错。
复位先构建替代世界再关闭旧世界，以保证重建失败可保留原世界；短暂共存是明确的回滚代价。

夹爪复用共享动作但只允许一个活动爪，不支持独立多载荷。全局 `payload.*` 不侧别化。
Jumper 禁止自动释放；闭合或未知状态按保守 `holding_payload` 阻止掉载/盲目回位，
闭合/未知时 `holding_payload=None`，已确认张开时为 `False`；禁止自动释放的策略单独声明，避免把闭爪写成持物证据。跳跃要求 `payload.clear` 并声明实测结果形状。

Microduck 指令保留原幅值，导航显式使用已验收的仿真策略范围；通用手动限速不变。
运行前验证脚本和 ONNX 摘要。资产和二进制不随源码发布，单独保留三方件与非商业模型许可说明。
