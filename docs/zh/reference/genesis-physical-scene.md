# Genesis 物理场景后端

本页描述 `jiuwensymbiosis.simulators.genesis` 的宿主/worker 接口。
安装与页面操作见 [安装教程](../tutorial/03-install-genesis.md)和
[GUI 指南](../how-to/gui-simulator.md)。这里提供后端能力；新的 Agent 动作绑定与工作台入口另行接入。

## 场景缓存

`RoomSceneCache.prepare()` 缓存固定房间碰撞实体及原始网格，不缓存机器人、动态物体或运行中的世界。
加载缓存后仍须添加机器人并构建场景。静态 USD 背景也加载碰撞，不等同于可穿过的视觉背景。

默认目录为 `$XDG_CACHE_HOME/jiuwensymbiosis/genesis/scenes`，未设置 XDG 时使用
`~/.cache/jiuwensymbiosis/genesis/scenes`。可用绝对路径环境变量
`JIUWENSYMBIOSIS_GENESIS_SCENE_CACHE` 覆盖；相对路径会拒绝。

缓存键包含源文件与 USD 依赖的内容摘要、物理配置、相关 SDK 版本、缓存和碰撞处理代码。
每次命中核对 manifest 和负载校验和；损坏条目重建。相同键使用进程锁，完成后原子发布；
源数据在构建中变化会失败，不发布旧数据。`scene.scene_cache` 区分 `hit`、`built` 及重建原因。
缓存只复用预处理结果，不证明场景碰撞等价或物理验收通过。

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
| `robot_state` | 返回实测移动机器人状态；不支持时明确失败。 |
| `entity_state` | 指定 `entity_id`，`frame` 为 `world` 或 `base`；位置、姿态和包围盒一并变换。 |
| `plan_navigation` | 指定 `x_m/y_m/yaw_rad`，或静态有界 `entity_id` 与 `stand_off_m`；只产生规划，不执行运动。 |

返回值明确标注 `source="simulation_ground_truth"`，包含单位、场景实例、复位代次、步数和时间。
实体名称来自资产作者，不能当作模型识别标签。携带同一个 `snapshot_id` 可跨查询复用同一步的数据；
最多保留 16 个快照，60 秒或复位后失效。宿主请求参数最多 2048 字节，查询期限为 60 秒。
真值及基于场景网格的导航图不能冒充 RGB-D 感知、未知空间探索或安全运动完成证据。

导航图将支撑面与障碍分开，按本体范围和网格余量收缩可用空间；无支撑、障碍、地图过大或动态
目标会拒绝相应规划。得到路径不等于机器人已执行、停稳或到达。MOZ1 上身通过物理控制保持初始
关节姿态，实际负载、接触和运行效果仍需单独物理验收。
