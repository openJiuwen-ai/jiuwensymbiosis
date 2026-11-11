# 仿真资产配置

[English](README.en.md)

`jiuwensymbiosis/simulators/assets/data/assets.example.yaml` 是随 wheel 分发的示例；
`assets.yaml` 保存本机资产位置，不纳入 Git。
索引由本仓库扫描本地源文件生成，不依赖其他项目的代码、配置或数据库。

## 构建与查询

源码运行可在仓库根目录执行；首次使用显式生成配置，再修改 `roots[].path`：

```bash
python -m jiuwensymbiosis.simulators.assets init-config
python jiuwensymbiosis/simulators/assets build
python jiuwensymbiosis/simulators/assets status
python jiuwensymbiosis/simulators/assets search --root-id geniesim --kind environment
python jiuwensymbiosis/simulators/assets show library://geniesim/background/kitchen/kitchen_1/background
python jiuwensymbiosis/simulators/assets probe library://geniesim/background/kitchen/kitchen_1/background --usd
```

目录入口只需 Pydantic 和 PyYAML，不加载 Agent 或 Genesis SDK，适合读取纯 JSON。
也可用 `python -m jiuwensymbiosis.simulators.assets`，但框架包可能先输出启动日志。
`probe --usd` 额外需要 `usd-core`，只检查元数据和本地引用，不启动仿真。

配置和默认索引相对于同一个仿真数据根目录解析，优先级为：

1. 启动前设置的 `JIUWENSYMBIOSIS_SIMULATOR_HOME`（绝对路径，支持 `~`）。
2. 源码 checkout 根目录（同时具有 `pyproject.toml` 和 `configs/simulators/`）。
3. wheel 安装时的 `~/.jiuwensymbiosis/simulators/`。

默认配置位于数据根下 `configs/simulators/assets.yaml`，索引位于
`outputs/simulator-assets/index.sqlite`。默认位置不依赖启动目录；
源码运行保留仓库内路径，wheel 不会向 site-packages 写数据。
`init-config` 只生成示例，不扫描资产、不覆盖已有配置。wheel 安装后可从任意目录运行
`python -m jiuwensymbiosis.simulators.assets init-config`；其余命令也使用模块入口。
目录入口是轻量方式，不初始化 Agent SDK。
自定义构建用 `--config /path/to/custom.yaml build`，查询用 `--index /path/to/index.sqlite search`；
这些全局参数放在子命令前。查询不接受 `--config`，也不自动扫描或重建。

- `root.path`、`index` 支持绝对路径、相对于 YAML 的路径和 `~`，不展开环境变量。
- `content_path` 相对于资产根目录，文件关联必须留在根目录内；不支持 URL 或自动下载。
- 根可用 `material_rules` 声明该来源使用哪套材质转换规则（如
  `material_rules: interioragent`）；行为由这个声明决定，不由根 ID 命名决定。
- 更改资产位置、内容或扫描规则后重新 `build`。扫描或写库失败会保留旧索引。
- SQLite 保存路径、加载选项与材质摘要，不包含 USD、URDF、网格或贴图内容；加载仍需源文件。

## 扫描规则

顶层仅接受 `version`、`index`、`roots`。每个根声明 `id`、`path`、`scanner`，以及可选的
`exclude`。支持 `geniesim`、`spiritai` 和 `content_tree` 三种扫描器。
所有调用使用查询返回的 `library://...` ID，不转换旧短 ID。
ID 由库 ID 和相对入口路径生成，去掉 USD 扩展名并对特殊字符作 URL 编码。

`content_tree` 由本机配置声明目录分类，例如：

```yaml
scanner: content_tree
content_path: content
families:
  Scenes:
    kind: environment
    preview_depth: 2
    preview_depth_overrides: {Campus: 3}
  Machines: {kind: robot, min_depth: 1}
```

深度相对于各分类目录，包含文件名；默认 `min_depth: 1`、`preview_depth: 2`。
辅助材质、网格、物理目录不作入口；环境下的 Props 收录为物体，深层组件保守归为 `other`。
同名 USD 后缀变体按固定规则选入口，不代表它们等价。GenieSim 的 `Aligned.usda`
只有明确引用同目录 `Aligned.usd` 且声明 Z-up 时才优先。SpiritAI 优先采用机器人
`config.yaml` 的 `asset_path`；入口不明确时记录诊断，不执行资产目录内的 Python 脚本。

## 机器人关联与校验

扫描器在 USD 同目录及 `urdf/` 中查找 `export_manifest.json`，以 `source` 匹配 USD，
再读取 `urdf` 指向的模型。无匹配清单时，只关联同目录或 `urdf/` 下的同名 URDF；
多个候选会报错。清单可使用不同文件名，例如：

```json
{"source": "../robot.usda", "urdf": "kinematics.urdf", "file_meshes_are_zup": true}
```

`file_meshes_are_zup` 描述导出网格坐标；导出清单缺省为 `true`，普通同名 URDF 为 `false`。
物理 URDF 旁的 `materials/manifest.json` 通过 `physics_urdf_sha256`、`render_urdf` 和
`files_sha256` 关联并校验渲染 URDF、网格及贴图。清单损坏、引用越界或摘要不符会使构建失败。
关联和摘要写入索引；加载时按快照校验资源，不再解析清单或构建 YAML。

## Python 查询

```python
from jiuwensymbiosis.simulators.assets import AssetLibrary

library = AssetLibrary()  # 可显式传入其他索引路径
page = library.search(root_id="geniesim", kind="environment", limit=20, offset=0)
if page.items:
    asset = library.get(page.items[0].id)
    source_usd = library.resolve(asset.id).paths["source"]
```

`search` 支持关键词、`root_id`、`kind`、`category`、`preview_mode` 过滤，返回
`total/items/limit/offset`；`resolve` 重新检查入口位置、大小和修改时间。
`preview_mode` 仅标记候选类型：环境为 `static_usd`，关联 URDF 的机器人为 `urdf`。
已提供 USD 场景处理、表面拾取、独立 worker 和 WebRTC 后端；完整浏览器页面与工作台入口由下一批交付。
`probe` 检查本地资源引用和材质摘要；USD 未解析列表也可能包含 SDK 自带的 MDL 模块。
索引成功或预览可用不代表碰撞、控制、规划或画质已经通过验收。

## 资产准备边界

资产准备需提供上述 export manifest 或匹配 URDF。本批不提供 USD→URDF 转换器，
不自动下载资产，也不执行资产目录附带的脚本。

## USD 场景与可选渲染依赖

CPU 场景解析和表面拾取可单独安装，不需要 Genesis、Torch 或 GPU：

```bash
python -m pip install -e ".[simulator-scenes]"
python -m jiuwensymbiosis.simulators.genesis.usd_probe --inventory --output outputs/usd-inventory
```

先按上文构建本地资产索引。探针只接受索引返回的 `library://...` ID，
不自动扫描或下载资产。`--inventory` 只做 USD 元数据和依赖检查，不初始化渲染器。
源码可编辑安装后，`genesis_appearance.yaml`、`genesis_lighting.yaml` 从源码包内 `data/` 加载。
生成的静态场景覆盖层和报告写入显式 `--output`，不覆盖源 USD。

GPU 渲染桥接使用独立 Python 3.12 环境和固定版本的 Genesis/Nyx：

```bash
python -m pip install -e ".[genesis]" --extra-index-url https://download.pytorch.org/whl/cu130
python -m jiuwensymbiosis.simulators.genesis.usd_probe --scene <资产ID> --output outputs/usd-preview
```

渲染需要受支持的 NVIDIA GPU/驱动；安装完成不等于 GPU 验收通过。
`genesis` 固定 Torch 2.11.0+cu130、Genesis 1.4.4、Nyx plugin 0.1.9/Nyx 0.1.7、Pillow 11.3.0。
不要与固定其他 Torch 版本的 `full`、`vision-server`、`voice-local` 合装，
也不要与要求 Pillow>=12 的 `dev`、`remote` 合装；CPU 测试环境可使用 `.[dev,gui,simulator-scenes]`。
本批尚不提供 WebRTC 或工作台仿真界面。

源 USD 与贴图文件保持不变；对声明了 `material_rules` 的资产根，渲染前做非破坏的
材质转换与贴图烘焙（见下文「源材质渲染」），逐材质明细写入输出目录的
`material-repair.json`。
曝光和色彩映射只作用于相机显示；灯光转换和拾取测试验证 USD 数值/几何，
不代表画质、接触物理或机器人任务验收。

配套批处理与导出工具（`.[genesis]` 环境，输出目录按资产 ID 建档）：

```bash
python scripts/simulator/genesis/genesis_interioragent_batch.py --output outputs/interioragent-batch
python scripts/simulator/genesis/genesis_material_acceptance.py --output outputs/material-acceptance
python scripts/simulator/genesis/genesis_scene_occupancy.py --scene <资产ID> --output outputs/scene-map
```

批处理逐场景独立进程渲染并构建离线地图；验收脚本汇总转换/剩余/烘焙计数，
任一场景渲染失败以非零退出；占据图脚本导出 png/pgm/npz/yaml 与依赖哈希报告
（也可在 `.[simulator-scenes]` CPU 环境运行）。


## 独立运行与媒体后端

`GenesisPreview` 拥有一个独立 worker。worker 内才加载 Genesis、CUDA 和 Nyx；
父进程可查询状态、切换场景/画质和请求相机图像。`PreviewStream` 提供单观看者
WebRTC 信令和租约，不自行绑定网络端口。视频断开保留物理世界；结束时必须调用
`PreviewStream.stop()` 或 `GenesisPreview.stop()`，等待 worker 退出。

CPU 媒体回归可使用独立环境，不需要安装 Genesis/CUDA：

```bash
python -m pip install -e ".[dev,simulator-scenes,simulator-media]"
python -m pytest tests/unit_tests/simulators/genesis/test_stream.py -q
node --test tests/unit_tests/simulators/genesis/metrics.test.mjs
```

GPU 组合仍在独立环境安装 `.[genesis]`。需要 aiortc 1.15.0 的预编码 H264
关键帧桥接和 av 17.1.0；直接依赖声明见根目录 Notice。PyAV 的 BSD 许可不覆盖
所选 FFmpeg/codec 二进制的独立许可；`python -m av --version` 显示实际库版本和许可。
本仓库不打包这些外部二进制。

以下 GPU 检查使用内置演示场景，不需要配置外部资产，结果保留到指定目录：

```bash
JIUWEN_GENESIS_INTEGRATION=1 JIUWEN_GENESIS_EVIDENCE="$PWD/outputs/genesis-gpu" \
python -m pytest tests/integration/test_genesis_preview.py -q \
  -k 'real_nyx_motion_camera_reconnect_and_cleanup or real_nvenc_quality_and_forced_idr'
```

这两项检查覆盖动态图像、相机、重连、退出及 NVENC 编码，不代表室内接触物理、
机器人任务或浏览器界面已经验收。

## 源材质渲染

对在 `assets.yaml` 中声明了 `material_rules` 的资产根（如 InteriorAgent 根声明
`material_rules: interioragent`），渲染前会把已知不支持的 MDL 模板族转换为
`UsdPreviewSurface`：非破坏地在独立覆盖层写入新材质，源 USD 与贴图文件不改写。
转换计数、烘焙明细、剩余未支持项与近似标注（如菲涅尔双瓣取正向分支）逐材质
记录在输出目录的 `material-repair.json`；未声明规则的根或不在规则内的模板族
保持渲染器默认行为并在报告中列为 `remaining`。

贴图按 UV 仿射变换重采样烘焙，输出内容寻址缓存（分辨率/画质重建渲染器时复用，
不重复烘焙）；EXR/HDR 线性源按线性处理，不做二次伽马。单个材质转换失败只影响
该材质（记入 `issues`，保留常量回退），不会导致整场打不开。

材质等价性未经物理验收：转换是显式近似，报告中的 `issues` 逐条标注偏离；
占据图等评估产物不证明与物理碰撞等价，也不直接授权执行导航（运行时规划以
`navigation.NavigationMap` 为准）。

## 机器人模型准备与准入

CPU 几何准备使用 `.[simulator-collision]`（POSIX 锁，Linux x86_64）；加载审计另需 Genesis 1.4.4。
入口为 `python jiuwensymbiosis/simulators/genesis/robot_model.py <source.urdf> --output outputs/robot-audit/example`，
加 `--load-audit` 只 build/读回，不推进物理。资产需单根树、无环 mimic、fixed/revolute/continuous/prismatic 关节，
碰撞几何限 box/sphere/cylinder 和 OBJ/STL/PLY，正 scale。运行时不提供 package URI 映射；应先解析为文件路径。
不支持的资产会启动失败，报告位于状态目录 `robot-models/<robot_model_instance_id>/loaded-model.json`。
MOZ1 示例策略在 [collision/](collision/README.md)，GUI 不自动加载。
参数、生命周期和物理验收边界见[指南](../../docs/zh/how-to/audit-robot-models.md)。
