# Genesis 物理场景后端

本页描述 `jiuwensymbiosis.simulators.genesis` 的宿主/worker 接口。
安装与页面操作见 [安装教程](../tutorial/03-install-genesis.md)和
[GUI 指南](../how-to/gui-simulator.md)。这里提供后端能力；新的 Agent 动作绑定与工作台入口另行接入。


## 原始 USD 渲染与单层地面物理

静态 USD 背景仅创建一个不可见的水平物理平面。原始 USD 继续提供视觉场景；
房屋网格只用于支撑面选择、导航准入和真值查询。逐网格碰撞导出、凸分解、
房屋 SDF 构建和房屋碰撞缓存已移除，机器人模型碰撞准入保持独立。

默认选择源几何中可支撑的水平面；通过 `floor_height_m`（世界坐标米）指定其他
有源几何支撑的高度，null 恢复自动选择。这是单层近似，不支持楼梯、斜坡、
多层物理或墙壁家具接触。物理平面无限延伸，导航准入仍受源几何约束。
接触数据明确标注 `scope: ground_only`，无物理碰撞不代表没有视觉障碍物。

配置 `robot_mode: virtual_camera`、`camera_name: scan_cam` 可加载独立的水平
640×480 标定相机，无需机器人。action 模式 RGB-D 采集提供虚拟位姿和同步信息，
不伪造底盘实测状态；机器人 spatial_memory 采集仍遵循原有物理机器人契约。
复位恢复初始相机位姿，视口重建保留物理世界。旧外部房屋缓存不再读取。

## 同一世界内复位

宿主调用 `GenesisPreview.reset_scene()`，保留物理场景与渲染器，恢复初始稳定状态。
运动任务、图像采集或场景查询占用时拒绝复位。支持移动机器人的场景须先实测低速持续
0.6 仿真秒（线速度小于 0.008 m/s、角速度小于 0.02 rad/s）；无法确认或取消时不恢复检查点。

成功恢复会更新 `scene_instance_id` 和 `reset_generation`，清空选择与真值快照。
宿主等待恢复状态的新帧后才确认完成。15 秒超时表示结果未确认，不自动重试，也不保证后台已取消。
旧帧、旧查询及旧控制会话不能跨越复位继续使用；只改观察分辨率则保留世界和位姿。

## RGB-D 采集

`GenesisPreview.capture_image(camera_name, include_depth=True, depth_stride=8)` 仅接受机器人相机，
要求受支持的移动机器人当前没有活动运动；不依赖视频连接，不负责替调用者建立实测停车证据。
采样在拥有场景的 worker 线程执行，采样期间不推进物理状态，并分块检查取消、截止时间和配置变化。

返回 JPEG、对齐的 `rgb`/`depth_m`、相机标定、`base_sample`、`body_bounds_m` 和场景/帧绑定信息。
深度为相机光轴 Z，单位米；`T_base_camera_mm` 的平移为毫米；本体包围盒为底座前/左/上坐标系的米。
`depth_stride` 接受整数 4–32，未采样或无有效命中的像素保留 NaN，不能补成可通行空间。
Nyx 投影仅接受代码已验证的 SDK 0.1.7 / plugin 0.1.9 组合；其他版本须重新验证后支持。
宿主检查结果绑定、新鲜度及数组形状/类型，拒绝过期、配置变化和超出 32 MiB 的 RGB-D 负载。

## 显式真值与导航规划

`GenesisPreview.query_scene(operation, params=None, cancelled=None)` 在物理步之间查询：

| operation | 参数与行为 |
|---|---|
| `list_entities` | `query`、`kind` 过滤；`limit` 为 1–50，`offset` 为非负整数；返回下一页偏移。 |
| `robot_state` | 实测移动状态；规划模式还返回 joint_positions/joint_velocities/joint_units（rad/m）、planning_groups、planning_base_link、tool_poses_base_m_wxyz（米、wxyz）和 effectors；不支持时失败。 |
| `entity_state` | 指定 `entity_id`，`frame` 为 `world` 或 `base`；位置、姿态和包围盒一并变换。 |
| `plan_navigation` | 指定 `x_m/y_m/yaw_rad`，或静态有界 `entity_id` 与 `stand_off_m`；只产生规划，不执行运动。 |

返回值明确标注 `source="simulation_ground_truth"`，包含单位、场景实例、复位代次、步数和时间。
实体名称来自资产作者，不能当作模型识别标签。携带同一个 `snapshot_id` 可跨查询复用同一步的数据；
最多保留 16 个快照，60 秒或复位后失效。宿主请求参数最多 2048 字节，查询期限为 60 秒。
真值及基于场景网格的导航图不能冒充 RGB-D 感知、未知空间探索或安全运动完成证据。

导航图将支撑面与障碍分开，按本体范围和网格余量收缩可用空间；无支撑、障碍、地图过大或动态
目标会拒绝相应规划。得到路径不等于机器人已执行、停稳或到达。MOZ1 上身通过物理控制保持初始
关节姿态，实际负载、接触和运行效果仍需单独物理验收。

## 材质转换与占据图导出

`RuntimeScene.info["appearance"]["material_repair"]` 是跨 IPC 的摘要：
`converted` / `textures` / `remaining`（计数）/ `issues`（按种类计数）/
`baked_textures`（计数）/ `report_path`（指向输出目录的 `material-repair.json`，
含逐材质明细）。声明了 `material_rules` 的资产根在渲染前做非破坏转换——源 USD
不改写，烘焙产物内容寻址缓存并随场景缓存持久化（分辨率重建渲染器时复用）；
单个材质失败只记入 `issues`，不中断整场。

`occupancy.build_occupancy` 从同一批房间网格导出离线占据图（png/pgm/npz/yaml
+ 依赖哈希报告）。它是评估导出：`sensor_derived=False`、无足迹膨胀、不证明与
物理碰撞等价；运行时规划以 `navigation.NavigationMap` 为准（两者使用相同的
高度带边界，占据图的地面层按面积投票独立推导）。

## 机器人模型准入

移动模式在电机初始化前准备和审计碰撞模型；失败阻止进入运动状态。默认不加载 MOZ1 示例策略。
`info.model_admission` 提供报告路径，`robot_model_instance_id` 在 reset 后保持不变。
持久化证据位于状态目录 `robot-models/`，关闭后标记 `closed`，由运维归档，不随渲染缓存删除。
支持范围、实际导入参数与清理要求见[模型准入指南](../how-to/audit-robot-models.md)。

规划 profile：`planning_profile.load_profile` 要求 `schema_version=1`、`base_link`、`groups`（独立 joints 与 tools）。`limits`/`default_limits` 提供 velocity/acceleration/jerk（rad 或 m 的一至三阶时间导数）及 effort（N/Nm）；覆盖项须属于组。可选 `collision_source`（format/path/ignore_rationale）导入并哈希碰撞球；否则显式 collision_spheres。`CuroboPlanner` 验证 `uncontrolled_support_joints` 为原点球形、连续、无下游的支撑叶关节；`CollisionGate` 仅允许 `support_links` 接触地面。可选 `effectors` 由 EndEffectors 消费，字段见夹持指南；`contact_solver.noslip_iterations` 是场景求解器迭代数。


## 通用源光照与显示策略

所有静态 USD 自动读取源 DomeLight、RectLight 的颜色、强度、曝光、单位及世界变换，
不再按 Home 0 或其他场景 ID 选择配置，也不自动添加检查补光。
没有 DomeLight 时使用零环境光，让源矩形灯或发光材质决定亮度。
SphereLight 转为点光、DistantLight 转为平行光，有限尺寸软阴影和独立漫反射/高光权重不能等价保留，逐灯记录近似。多个 DomeLight、其余未实现灯型、IES/聚光整形等明确报错，不静默丢灯。

根据 OmniPBR 的 enable_emission 或 UsdPreviewSurface 的 emissiveColor 自动寻找向下的
水平发光灯具面。只有距离、唯一性、面积和几何检查通过才贴合；未匹配或歧义时保留
源灯面并记录原因。所有矩形灯统一应用可审计的 60000 峰值辐亮度上限，保留灯间比例；
这是 Nyx 数值范围的近似，不能视为严格光度等价。

全局使用 Neutral 色调映射，普通 USD 默认相机曝光 −8 EV，源 presentation 契约使用
其原有 0 EV 基线；曝光在整个轨迹中固定。全局显示调整位于包内
data/genesis_appearance.yaml，旧 scenes 映射不再接受，genesis_lighting.yaml 已移除。
这不是逐帧自动曝光，也不是保证所有资产同样明亮。

每个 USD 都检查材质：Nyx 原生支持的 OmniPBR/UsdPreviewSurface 保持不变，
不支持的 shader 写入 material-repair.json 并在状态中计数。已有根目录声明的特殊
材质转换规则仍独立存在；本次未把专用 MDL 模板规则当作通用转换。
OmniGlass 转换会请求原生双面薄壁透射：只将已标记的玻璃网格分到独立 USD 可见性层，保留源几何和变换，并显式启用 Nyx 双面材质及背面法线翻转。它修复单面玻璃壳的黑背面；体积吸收、厚度仍是近似，不代表完整玻璃光学还原。混合面材质的网格暂不分离，计入 `skipped_subset_meshes`。


## 资产材质规则传递与审计

根配置的 material_rules 由扫描器写入 Asset，随 SQLite record 保存，经搜索/get/resolve
传到正式 RuntimeScene → SceneRenderer。与 root id 或目录名无关。索引 schema/scanner
升级为 3；旧索引必须通过资产 build 重建，避免默默丢失规则。InteriorAgent 声明的
规则只对选中的已知 MDL 模板生效；传统 UE4 模板发光按源 OmniUe4Base 的 20×128 单位换算。

完整源资产缺失引用列表保存到 scene-audits，状态消息只发送计数及最多 8 条示例；
完整材质转换及灯光近似另有审计。转换数量不等价于纹理全部有效或玻璃/软阴影完全还原。
