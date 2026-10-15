# 接入 RoboSim 仿真后端

RoboSim 在专用 Python 环境中运行并管理仿真世界。JiuwenSymbiosis 在独立 Agent 环境中通过本机 HTTP 接入，断开连接只释放本地客户端，不关闭 RoboSim 服务。

当前接入支持 MOZ1 所选单臂的关节控制、TCP 运动和夹爪操作。对象抓放、底盘导航、相机输入和双臂协同不在本阶段的能力集合中。

## 启动独立服务

在 JiuwenSymbiosis 源码根目录，使用 RoboSim 专用环境的 Python 启动服务：

```bash
<robosim-python> scripts/run_robosim_service.py --repo <robosim-repo> --headless
```

`<robosim-python>` 是已安装 RoboSim 运行依赖的 Python 可执行文件，`<robosim-repo>` 是包含 `sim_service/main.py` 的 RoboSim 源码根目录。

服务默认监听 `127.0.0.1:8001`。启动器复用 RoboSim 的主循环，并添加只读观测与实例身份接口；它不修改 RoboSim 源码。Agent 环境继续使用本项目原有依赖。

## 配置与工作区查询

所有仿真配置统一放在 `configs/simulators/`，当前入口为 `configs/simulators/moz1.yaml`。`backend: robosim` 经仿真后端发现入口解析到 `jiuwensymbiosis.simulators.robosim`，现有硬件仍位于 `adapters/`。

`backend` 选择 `simulators/` 下的仿真后端；`adapter` 只选择 `adapters/` 下的真实硬件适配器。二者互斥，不根据目录名或模块是否存在自动回退。仿真配置应使用 `backend: robosim`；此前 PR 草稿中的 `adapter: robosim` 需要改写，否则将按硬件路径解析并报错。

`prepare_binding()` 是共用的运行配置准备入口：它按选择字段加载对应的会话构建器，校验并冻结配置，计算资源准入标识；此阶段不连接硬件或仿真服务。仿真与硬件共用会话生命周期和准入保护，选择字段及发现命名空间保持分离。`RobosimConfig.from_dict` 只提取自身负责的连接和运动参数；`backend` 由运行时读取，`agent` 由 Agent 配置读取。

| 字段 | 消费方与作用 |
| --- | --- |
| `backend` | `prepare_binding` 只从 `simulators/robosim` 选择仿真会话构建器。 |
| `name` | 通用会话构建器设置 `RobotSession.name`，用于会话标识及默认工作区选择。 |
| `service_url` | HTTP 客户端连接地址；运行时还用服务端口生成准入资源标识。 |
| `robot_type` | 机器人加载与身份核验；当前支持 `moz1`。 |
| `side` | 选择左/右臂的观测和控制资源。 |
| `load_robot` | 控制会话连接时是否允许在空服务中加载机器人。 |
| `request_timeout_s` | 单次 HTTP 请求超时。 |
| `joint_timeout_s` | 关节到位读回等待上限。 |
| `joint_tolerance_rad` | 关节到位误差阈值，单位为弧度。 |
| `position_tolerance_mm` | TCP 位置读回误差阈值，单位为毫米。 |
| `orientation_tolerance_deg` | TCP 姿态读回误差阈值，单位为度。 |
| `agent.modules.visual_feedback.enabled` | Agent 配置关闭视觉反馈；本阶段不提供相机输入。 |

这些字段均有用途。部分值与后端默认值一致，示例显式保留它们，方便部署时核对连接、加载行为和到位阈值。

```python
from jiuwensymbiosis.simulators.robosim import RobosimConfig, RobosimWorkspace

workspace = RobosimWorkspace(RobosimConfig(load_robot=False))
try:
    print(workspace.refresh())
    print(workspace.scene_state())
finally:
    workspace.close()
```

构造、查询和关闭工作区不会加载机器人。`workspace.load_robot()` 是显式加载入口：空场景加载 MOZ1，已有 MOZ1 保留，其他型号拒绝替换。`scene_state()` 返回登记状态，不能用作物体实时位姿。

## 机器人会话与动作

通用入口可直接读取带 `backend: robosim` 的 YAML。CLI 的 `--robot` 只覆盖硬件 `adapter`，不能覆盖仿真配置；GUI 注册仿真本体时同样使用 `backend`。示例配置的 `load_robot: true` 允许会话连接时在空服务中加载 MOZ1；只连接已有机器人时设置为 `false`。

```python
from jiuwensymbiosis.introspect import prepare_binding
from jiuwensymbiosis.runtime.admission import admitted_session

binding = prepare_binding("configs/simulators/moz1.yaml")
with admitted_session(binding) as session:
    print(session.env.get_observation())
```

已有工作区可通过 `bind_robosim_session(workspace, workspace.refresh())` 绑定会话和命名查询工具。会话借用工作区传输；断开会话只关闭本地驱动，Env 保留借入客户端和绑定身份。重连同一实例仍借用原客户端；实例变化、工作区关闭或首次连接失败后重试不会自动转成独立会话。独立会话和工作区绑定会话都校验服务与机器人实例身份，同型号重载后旧会话拒绝操作。

- 关节使用名称与弧度，限位来自 USD，运动完成依据实测关节读回。`env.move_named_joints(targets, timeout_s=...)` 可覆盖本次到位等待期限；省略时使用 `joint_timeout_s`。
- TCP API 使用基坐标系毫米和 XYZ 欧拉角度数，经 `RobosimEnv.get_tcp_pose/move_to_tcp` 调用 `TcpDriver`，驱动转换为后端世界坐标米与弧度。当前未提供 TCP 到法兰的几何转换，Env 的 FLANGE 查询和运动接口明确拒绝调用。
- 夹爪闭合本身不证明抓取成功；没有所选机械臂的抓持绑定时返回 `grasp_not_confirmed`。
- `SafetyRail` 在正常 Agent 入口执行。直接调用 API 不自动触发 rail；诊断脚本必须显式校验。

所有机器人动作经 `RobosimApi → RobosimEnv → RobosimLowLevel`。关节与 TCP 动作直接返回同一命令范围内的实测结果，API 不再在命令结束后追加一次无保护读回。夹爪的抓持确认由 Env 保留所选机械臂的证据。

## 通信失败与安全边界

HTTP 仅允许 loopback，禁用代理与重定向，不自动重试失败请求。写入超时、损坏响应、中断，或写入已接受后的读回/到位确认失败造成结果未知后，同进程内同端口的后续写入被阻断，查询仍可进行；清空客户端不会解除保护。只有已识别后台换成新服务实例才可恢复。

写响应必须包含 `success: true` 和非空对象 `data`；只读桥接接口仍允许原始 JSON。夹爪响应还必须包含机器人型号、开合和抓持状态；归位响应必须确认所选机械臂完成，不能是部分成功。缺失这些证据时保留结果未知，不提交动作成功效果。

官方 CLI 和 `robosim_smoke` 的运行时准入锁按服务端口串行化本机调用，同一服务的左右臂和 loopback 别名共用资源标识。该锁不替代 RoboSim 的结果未知保护：本阶段操作协调器仍是进程内状态，不能跨进程或跨重启保存未知操作。运行时保持单操作者，不通过重启 Agent 绕过未知状态；先核对现场，再显式重启自有后台。不要将本地 `completed` 视为独立物理验收。

RobotControlTool、直接执行器和 ability executor 保留 `outcome_unknown` 与 `skip_recovery`，序列在失败处停止，不追加自动回零或松爪。

## 验证

无需启动 RoboSim 服务的边界测试：

```bash
python -m pytest tests/unit_tests/simulators/robosim tests/unit_tests/test_backends.py -q
python -m scripts.validate_adapter --module jiuwensymbiosis.simulators.robosim
```

RoboSim 确定性 smoke 会加载或连接仿真机器人并执行运动，应在空载、专用仿真中运行：

```bash
python -m scripts.robosim_smoke --config configs/simulators/moz1.yaml --cartesian \
  --output outputs/robosim_smoke/pr1.json
```

报告的 `robot_backend` 固定为 `robosim`，保存动作参数、实测读回和失败状态，不调用 LLM。通用 `smoke_test_adapter.py` 的桩驱动没有抓持证据，因此 `close_gripper` 应拒绝成功；通用脚本也缺少 `goto_pose` 的安全参数。对应行为由本后端的针对性测试覆盖，不能将通用桩结果当作物理验收。
