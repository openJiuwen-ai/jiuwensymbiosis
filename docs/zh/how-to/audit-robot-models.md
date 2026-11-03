# 准备和审计机器人碰撞模型

机器人模型准备现在通过 `robot_model.prepare_robot_model()` 和
`collision_audit.audit_loaded_model()` 两个接口完成。前者不创建仿真场景，
后者读取调用方已经 build 的 Genesis 实体，不推进物理、不发送运动命令。
两者不依赖机器人名称，也不引入新的动作执行器。

## 当前支持范围

- URDF 单根树；fixed、revolute、continuous、prismatic 关节和无环 mimic 引用。
- 碰撞体：box、sphere、cylinder，以及 OBJ/STL/PLY 三角网格；一个连杆可有多个碰撞体。
- 固定或浮动基座；Genesis 1.4.4，未合并固定连杆，正 mesh scale，一份环境（`n_envs=0` 或 `1`）。
- URDF 的米、弧度和千克约定；保留各碰撞体 origin 和 mesh scale，不猜测或自动修正单位。
- 相对/绝对文件路径；`package://` 通过调用方显式提供包目录映射解析。

这不意味着自动获得其他机器人的控制器。轮式底盘运动、关节增益、工具坐标和动作能力仍由
各机器人适配器声明。当前不承诺 USD 动态机器人、闭环机构、柔性结构、多环境审计或实机效果。
纯运动学 URDF 不可用于证明接触质量。视觉纹理及其间接依赖不属于碰撞资源完整性保证。

## 准备与缓存

安装可选的 CPU 几何依赖 `.[simulator-collision]`；加载审计另需项目固定版本的 Genesis 环境。
核心包导入不会加载 Genesis、CoACD 或 cuRobo。

```bash
python -m jiuwensymbiosis.simulators.genesis.robot_model /absolute/robot.urdf \
  --output outputs/robot-audit/example \
  --package robot_description=/absolute/robot_description
```

`model.json` 记录源 URDF、直接网格依赖和派生 URDF 的 SHA-256，关节树、限值、mimic、惯量原文、
碰撞形状/变换/缩放、策略与工具版本。无限关节限值用字符串显式表示，JSON 不使用非标准的 Infinity。
源资产不被写入。模型及资源在加载审计前后再次校验哈希。

默认保留 authored 碰撞几何，不自动分解或删除。使用 `--policy /absolute/policy.yaml` 显式选择策略：

```yaml
schema_version: 1
name: example_gripper_v1
decomposed_links: [finger_a, finger_b]
disabled_links: {}
parameters:
  threshold: 0.03
  max_convex_hull: 32
```

禁用项格式为 `disabled_links: {link_name: "有依据的具体原因"}`，不能只列名称。
缺失连杆、错误参数或分解失败均报错，不回退到单凸包。
CoACD 的 threshold 是搜索目标，不是最大表面误差保证。

缓存位于输出目录的 `collision_parts/`，键包含源字节、文件类型、完整参数、CoACD/Trimesh 版本与缓存版本。
部件文件名包含完整键，避免导入器按 basename 合并不同网格。
每次命中复核 SHA、凸性、闭合性、绕序、体积及坐标范围；损坏缓存报错。
首次结果使用进程锁和临时目录验证后发布，失败不会发布半套缓存。

MOZ1 的左右手分别配置在 `genesis/data/moz1_right_collision.yaml` 和
`genesis/data/moz1_left_collision.yaml`。右手保留旧实验有依据的 RoboSim loop 排除规则；
左手分别核对源 USD 的 authored `physics:collisionEnabled=false` 后配置两个 loop 排除项，
并非仅按名称镜像复制；核对证据在本轮 outputs 的 `source-policy-evidence.json`。
这些策略通过 `Moz1Base(..., collision_policy=...)` 显式传入；普通移动模式默认不改变原碰撞分组。
左右手的关节控制器不在这个通用模块中。

## 加载准入与报告

准备后加 `--load-audit` 可建立一个仅用于检查的 CPU 场景，默认固定基座，`--floating` 改为浮动基座。
该入口不会执行任何物理步或运动。

在已有运行链路中，应依次调用：

```python
model = prepare_robot_model(source, output, policy=policy)
# 用 model["urdf"] 添加机器人，merge_fixed_links=False、decimate=False；scene.build()。
audit_loaded_model(entity, model, output / "loaded-model.json")
# 审计通过后再初始化控制器。
```

当前 `RuntimeScene → Moz1Base.build()` 已在所有电机初始化之前执行准入。
模型、缓存和报告保留在调用方输出根目录的 `robot-models/<scene-instance-id>/`，
不会随着临时渲染目录清理而删除。审计失败会写出报告并抛出异常，场景无法进入运动状态。
固定基座的纯视觉 `RobotPreview` 仍是无碰撞的运动学预览，不代表物理准入。

报告区分 `errors`、`warnings`、`unverifiable`：

- 实际几何数量、来源实例、运行时顶点/三角面指纹、局部和世界范围；同名错误网格不会仅因数量一致而通过。
- 每个碰撞体在连杆坐标系下的 26 方向支撑值，按最小误差匹配，无需假设导入顺序不变。
  网格阈值 10 微米；球/圆柱另允许半径的 2.5% 离散化误差。
  普通移动模式保留原有 decimation 设置；审计记录实际处理参数，并从独立读取的源网格重新执行
  Genesis 的凸化/简化步骤作为期望值。它检查加载一致性，不独立证明简化算法的物理精度。
- 实际碰撞掩码、solver 开关以及构建后的允许碰撞对；被相邻/中立姿态等规则过滤的对不等于真实无相交。
- 读回不可用或来源不明确时，不能用源网格冒充已加载几何，准入失败。

`pass` 只代表上述模型准入检查通过。支撑值无法证明凹面逐点一致；碰撞对可用不证明接触稳定，
也不提供持续避碰规划。物理运动、刹停、复位、抓放和视频必须另外验收。

本轮复现脚本、分阶段失败记录和最终结果保存在 `outputs/robot-model-audit/20261103/`。
