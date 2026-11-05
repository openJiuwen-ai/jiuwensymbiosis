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

## 遗留问题

1. 桌类重叠掩膜处理暂留在实验目录。十五房间实验存在 6 条已分配观测变为未决，规则收益和退化尚需逐例评估。
2. 主动补看与扫描结束判断后续再做。需将具体疑问、新视角证据和最终结论对应起来，并处理仍有歧义、目标未看见和预算耗尽。走完路线不能证明房间物体已记录完整。
