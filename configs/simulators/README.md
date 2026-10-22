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
本批只提供目录和文件探测，不包含预览界面或仿真执行。
`probe` 检查本地资源引用和材质摘要；USD 未解析列表也可能包含 SDK 自带的 MDL 模块。
索引成功或预览可用不代表碰撞、控制、规划或画质已经通过验收。

## 资产准备边界

资产准备需提供上述 export manifest 或匹配 URDF。本批不提供 USD→URDF 转换器，
不自动下载资产，也不执行资产目录附带的脚本。
