# 空间记忆基础版

空间记忆按显式调用启用。当前入口支持 Genesis/MOZ1 RGB-D 记录、YOLOE prompt-free 检测、跨帧物体关联、SQLite 持久化、离线回放和可视化。Agent 查询工具尚未注册。

## 安装

```bash
pip install -e '.[spatial-memory]'
# 需要本地 YOLOE 推理时，选择适合主机的 Torch/CUDA 环境，再安装
pip install -e '.[spatial-memory-yoloe]'
```

默认关联使用三维表面重叠与 RGB 直方图。可选 CLIP/DINOv3 描述子需要独立安装模型运行依赖和本地权重，基础功能不会加载它们。DINOv3 尚无实际模型验收结果。

## 记录 RGB-D

先通过已有 Genesis 入口加载场景、选择 MOZ1，等待后台就绪。以下命令连接已有后台，在同一场景会话内记录头部相机数据。

```bash
jiuwensymbiosis-memory capture outputs/memory-room/capture \
  --map-frame room-world --frames 30 --interval 0.5 --depth-stride 4 --allow-motion
```

`--allow-motion` 允许底盘运动中取样。路线和底盘运动由现有控制入口执行，记录器只采样。省略该参数时要求停稳。每帧的 RGB、深度与相机位姿取自同一物理状态，采样期间物理推进暂停。间隔是每次采样完成后的等待时间，不能解释为真实相机帧率。机械臂运动期间会拒绝 RGB-D 采样。

输出包含 `sequence.json`、每帧的 `rgbd.npz`、`observation.json`、`capture.json` 和 `capture-report.json`。失败立即结束记录，已经发布的帧仍可回放。后台关闭、场景切换和取消会传播为错误。退出记录器会释放自身连接。输出目录必须是新目录。

## 检测、建库和回放

权重由操作者预先准备并核对 SHA256。使用 4585 类的 YOLOE prompt-free 分割权重，文件名须以 `-pf.pt` 结尾。

```bash
jiuwensymbiosis-memory replay outputs/memory-room/capture outputs/memory-room/result \
  --weights /absolute/path/to/yoloe-11l-seg-pf.pt --sha256 VERIFIED_SHA256 --device 0

# 对同一 RGB-D 和已保存的检测做确定性回放
jiuwensymbiosis-memory replay outputs/memory-room/capture outputs/memory-room/replayed \
  --detections outputs/memory-room/result/detections

jiuwensymbiosis-memory visualize outputs/memory-room/result/memory.sqlite \
  outputs/memory-room/view.html --sequence outputs/memory-room/capture

jiuwensymbiosis-memory status outputs/memory-room/result/memory.sqlite
```

可视化 HTML 包含逐帧图像、观测点云、相机轨迹、实体 ID 和分配证据。`--snapshot` 额外导出静态图，需要 matplotlib。源码环境也可使用 `python scripts/spatial_memory.py` 执行同一组命令。

## 数据与更新约定

- `Observation` 保存不可变 RGB、米制 optical-Z 深度、相机内参、相机到地图变换、时间戳及会话/地图/时钟/标定标识。位姿来源随观测保存。
- `Detections` 保留原始掩膜、标签、置信度和模型来源。YOLOE 推理失败会报错，成功但未检测到物体才写入空检测。
- `SpatialMemory.ingest()` 在一个事务中写入观测、关联结果与实体。相同观测的相同输入可幂等重试，冲突输入、倒序时间戳和范围不兼容的输入会拒绝，失败事务回滚。
- 一个数据库对应一个静态场景会话、地图坐标系、时钟及标定。重启数据库读取保留身份；重新启动仿真会产生新会话，需要另建数据库。自动跨会话地图对齐尚未提供。
- 身份关联优先使用已观测三维表面，在证据不足或冲突时保留未决记录。历史记录和数据库内原有补看候选字段用于追溯，当前入口未调度主动补看。
- 真实相机可构造相同 `Observation`，但真机时序同步、标定及里程计误差需要单独验收。

## Rule 单房间扫描

在已有 Genesis/MOZ1 常驻后台就绪后，可运行正式的有界闭环入口（不加载或调用 KEV/LLM）：

```bash
jiuwensymbiosis-memory scan outputs/memory-room/scan \
  --map-frame room-world --weights /absolute/path/to/yoloe-26x-seg-pf.pt \
  --sha256 VERIFIED_SHA256 --max-frames 16 --max-seconds 600 --max-distance 2
```

可用 `--state-dir` 指定独立常驻后台。CLI 只连接后台，不启动或停止外部服务；RobotSession 拥有运动租约。SIGINT/SIGTERM 请求取消，现有执行器负责停车，只有实测速度满足停车阈值才记录 stopped。单次命令仍有自己的执行超时；预算在观测和动作之间检查，不会在推理中间强行杀模型。

每轮保存 RGB-D → YOLOE 检测 → 同一 SpatialMemory 事务提交 → 在线地图与补看候选 → Rule 选择 → 整机执行门控 → 新观测。每个候选最多推进 0.2m 或转 30° 后重新观测；候选排序沿用 R0 效用，但这种有界执行/逐帧感知不是原实验的逐字节复现。记忆的 open 补看任务参与候选，deferred 仍是缺口。截短路线后达不到所需视角变化的补看不进入候选。

本次执行适配器明确使用 **simulator_readback 位姿**；整机门控复用 Genesis 的静态场景包络及地面支撑检查，拒绝需要绕路的候选。真值不用于选择观测收益或判断完成。本入口尚不代表轮编码器定位、未知环境导航或真机验收。机器人上身姿态在本轮固定；不支持动态障碍的完整安全保证。

输出 `capture/sequence.json`、`detections/`、`memory.sqlite`、`receipts.jsonl`、`moves.json`、`online_map.npz` 和 `scan-report.json`。采集先保存；感知异常不会写空成功检测。固定检测可通过 replay 命令重放。status 中的 `scan_assessment` 提供扫描结论；数据库之后接收新观测会失效该旧结论。

结束原因区分 `budget_exhausted`、`no_candidates`、`blocked`、`cancelled`、`failed` 和 `stop_unconfirmed`。预算结束不等于扫描完成。`scope=unknown` 时明确保留未知；即使观测到封闭连通分量且几何收敛，也仅描述该观测范围，不声称整套房屋。未决/延期补看阻止宣称观测范围内扫描完成；`inventory_complete` 不会自动置为 true。CLI 退出码：0 有界正常结束（可能不完整），1 执行/感知或停车失败，2 受阻/取消。

## 关联版本与剩余边界

v7 使用独立场景深度剔除被前景遮住的历史表面，避免遮挡造成额外实体；无深度不作遮挡证据。IoU≥0.90 的同帧近重复掩膜保留原始记录，以较高置信度代表参与一次关联，其余记录 `duplicate_of`，不作为额外多视角确认。较低重叠、容纳关系及身份冲突继续保留未决。

旧 v2–v5 数据库可只读；改变关联版本后必须回放到新的数据库，不能混写旧库。一次局部质量修复并不证明全房间召回达标。跨房间身份、视觉回环后的重关联和真机验收仍需单独工作。
