# 准备和审计机器人碰撞模型

`robot_model.prepare_robot_model()` 准备 URDF 与资源清单；`collision_audit.audit_loaded_model()`
读取已 build 的 Genesis 实体，在控制器初始化前拒绝不一致或不可验证的模型。两者均不推进物理或发送运动命令。

## 通用能力与机型配置

| 模块 | 职责 |
| --- | --- |
| `robot_model` | URDF 结构、资源解析、哈希与派生清单 |
| `collision_preparation` | 按显式策略分解几何、验证缓存，不包含机型名称或默认排除项 |
| `collision_audit` | 加载后的几何、处理参数和碰撞过滤读回 |
| `configs/simulators/collision/moz1_*_collision.yaml` | MOZ1 示例策略，须由 CLI 或代码显式加载 |
| `Moz1Base` | MOZ1 底盘及上身控制器的准备和准入接入点 |

GUI 没有碰撞策略配置入口，也不会自动选择示例 YAML；普通移动模式传入的策略为 None，保留源碰撞。
其他机器人需要自己的策略和构建生命周期接入；共享模型接口不等于获得 MOZ1 控制器或已完成物理验收。

## 安装与支持范围

`pip install -e '.[simulator-collision]'` 只安装 CPU 几何依赖，不含 Genesis。
加载审计需 Genesis 1.4.4；GUI 场景可安装 `.[gui,genesis,simulator-collision]`。
当前分解缓存使用 POSIX 文件锁，验证平台为 Linux x86_64。

支持单根树 URDF，fixed/revolute/continuous/prismatic 关节、无环 mimic；碰撞形状为
box/sphere/cylinder 或 OBJ/STL/PLY 三角网格，可有多个碰撞体。要求正 mesh scale，尺寸单位为米、弧度、千克。
保留碰撞 origin、scale、惯量和关节，不自动猜测单位。负 scale、闭环、未知形状/资源协议会明确报错；
资产作者应先导出受支持的 URDF。USD 动态机器人、柔性结构和多环境审计不在范围内。
视觉纹理及间接依赖不属于碰撞资源完整性保证。

## 准备与缓存

从仓库根目录运行独立脚本可避免初始化核心框架：

```bash
python jiuwensymbiosis/simulators/genesis/robot_model.py /absolute/robot.urdf \
  --output outputs/robot-audit/example --package robot_description=/absolute/robot_description
```

相对/绝对网格路径可直接使用；package URI 必须显式映射。`model.json` 记录源/派生 URDF、网格 SHA-256，
关节树、限值、mimic、惯量、碰撞变换、策略及工具版本。无限限值编码为字符串。源文件不被修改，加载前后复核哈希。
资产目录的导出清单负责源物理/视觉资产配对；`model.json` 负责本次实际加载的派生碰撞模型，两者不能互相替代。

使用 `--policy /absolute/policy.yaml` 显式启用策略，例如：

```yaml
schema_version: 1
name: example_gripper_v1
decomposed_links: [finger_a, finger_b]
disabled_links: {}
parameters: {threshold: 0.03, max_convex_hull: 32}
```

禁用项必须是 `{link_name: "具体依据"}`；缺失连杆、无效参数和分解失败均报错，不回退单凸包。
CoACD threshold 是搜索目标，不是表面误差上界。MOZ1 示例分别分解三个手部连杆、排除两个 loop 连杆；
排除原因记录在 YAML 中。示例描述特定源资产，使用者必须核对自己的源 USD 碰撞开关，不能按相似名称复制。

输出目录内 `collision_parts/` 的键包含源字节、类型、完整参数、工具版本及准备代码哈希。
相同输出目录可跨调用复用；不同实例目录不共享缓存。命中时复核 SHA、凸性、闭合、绕序、体积和坐标范围；
损坏缓存拒绝使用。使用进程锁及原子目录发布，部件文件名含完整键，避免导入器 basename 冲突。

## 加载准入与生命周期

加 `--load-audit` 创建 CPU 场景，默认固定基座、无简化、关闭自碰撞；`--floating --decimate`
选择移动运行时的浮动/简化方式；`--self-collision` 显式启用自碰撞。CLI 完成或失败均释放场景和 Genesis。
共享导入设置保留 authored 连杆坐标（align=False），不合并固定连杆、启用 collision/convexify、watertighten=5、关闭隐式分解；实际设置写入报告。

已有 Genesis 场景的调用示例（`source`、`output` 是 Path，`scene` 为调用方拥有的场景）：

```python
import genesis as gs
from jiuwensymbiosis.simulators.genesis.robot_model import prepare_robot_model, admission_morph_options
from jiuwensymbiosis.simulators.genesis.collision_audit import audit_loaded_model

model = prepare_robot_model(source, output)
entity = scene.add_entity(gs.morphs.URDF(file=model["urdf"], **admission_morph_options(fixed=False, decimate=True)))
scene.build()
report = audit_loaded_model(entity, model, output / "loaded-model.json")
# 仅在通过后初始化本机型控制器；调用方负责 scene/Genesis 的关闭。
```

`RuntimeScene → Moz1Base.build()` 在电机初始化之前审计，空碰撞模型也拒绝准入。
`info.model_admission` 提供状态和报告路径；`robot_model_instance_id` 是不可变构建 ID，reset 不改变它。
证据在状态目录的 `robot-models/<robot_model_instance_id>/`，独立于临时渲染目录。
关闭场景后写 `closed` 标记；不会自动删除成功或失败证据。运维可在确认无需留存后，仅归档/删除带 `closed`
标记的目录；活动实例目录不可清理。应为状态目录设置存储配额或定期归档。直接使用 `Moz1Base` 必须提供持久化目录。
纯视觉 `RobotPreview` 不走物理准入。

报告保存几何数量、来源实例、顶点/面指纹、范围、26 方向支撑匹配、掩码、solver 开关及允许碰撞对。
未经简化时，球/圆柱允许半径 2.5% 的离散化误差；重处理分支统一使用 10 微米阈值，并记录 processed_index，
不假设它等于 URDF collision_index。重处理检查加载一致性，不独立验证 Genesis 算法的准确性。
读回缺失记录为 `unverifiable` 并保留异常原因，与几何不一致的 `error` 区分；两者都拒绝启动控制器。
`pass` 不证明凹面逐点一致、接触稳定或抓取成功；物理运动、刹停、复位和抓放需另外验收。
