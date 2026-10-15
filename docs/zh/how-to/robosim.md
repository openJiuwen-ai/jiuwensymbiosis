# 接入 RoboSim 仿真后端

RoboSim 在独立的 Isaac Python 环境中运行。JiuwenSymbiosis 通过本机 HTTP 连接它，Agent 环境不导入 Isaac，也不负责关闭仿真世界。

当前接入支持 MOZ1 所选单臂的关节控制、TCP 运动和夹爪操作。对象抓放、底盘导航、相机输入和双臂协同不在本阶段的能力集合中。

## 启动独立服务

在 JiuwenSymbiosis 源码根目录，用已配置 RoboSim 的 Isaac Python 启动服务：

```bash
<isaac-python> scripts/run_robosim_service.py --repo <RiemannRoboSim_Isaac路径> --headless
```

服务默认监听 `127.0.0.1:8001`。启动器复用 RoboSim 的主循环，并添加只读观测与实例身份接口；它不修改 RoboSim 源码。Agent 环境继续使用本项目原有依赖。

## 配置与工作区查询

使用 `configs/robosim/moz1.yaml`。`adapter: robosim` 经统一发现入口解析到 `jiuwensymbiosis.simulators.robosim`，现有硬件仍位于 `adapters/`。

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

通用入口可直接读取 YAML。示例配置的 `load_robot: true` 允许会话连接时在空服务中加载 MOZ1；只连接已有机器人时设置为 `false`。

```python
from jiuwensymbiosis.introspect import prepare_binding
from jiuwensymbiosis.runtime.admission import admitted_session

binding = prepare_binding("configs/robosim/moz1.yaml")
with admitted_session(binding) as session:
    print(session.env.get_observation())
```

已有工作区可通过 `bind_robosim_session(workspace, workspace.refresh())` 绑定会话和命名查询工具。会话借用工作区传输；断开会话只关闭本地驱动。独立会话和工作区绑定会话都校验服务与机器人实例身份，同型号重载后旧会话拒绝操作。

- 关节使用名称与弧度，限位来自 USD，运动完成依据实测关节读回。
- TCP API 使用基坐标系毫米和 XYZ 欧拉角度数，驱动转换为后端世界坐标米与弧度。
- 夹爪闭合本身不证明抓取成功；没有所选机械臂的抓持绑定时返回 `grasp_not_confirmed`。
- `SafetyRail` 在正常 Agent 入口执行。直接调用 API 不自动触发 rail；诊断脚本必须显式校验。

## 通信失败与安全边界

HTTP 仅允许 loopback，禁用代理与重定向，不自动重试失败请求。写入超时、损坏响应或中断造成结果未知后，同进程内同端口的后续写入被阻断，查询仍可进行；清空客户端不会解除保护。只有已识别后台换成新服务实例才可恢复。

官方 CLI 的运行时准入锁按服务端口串行化本机调用，同一服务的左右臂和 loopback 别名共用资源标识。该锁不替代 RoboSim 的结果未知保护：本阶段操作协调器仍是进程内状态，不能跨进程或跨重启保存未知操作。运行时保持单操作者，不通过重启 Agent 绕过未知状态；先核对现场，再显式重启自有后台。不要将本地 `completed` 视为独立物理验收。

RobotControlTool、直接执行器和 ability executor 保留 `outcome_unknown` 与 `skip_recovery`，序列在失败处停止，不追加自动回零或松爪。

## 验证

无需 Isaac 的边界测试：

```bash
python -m pytest tests/unit_tests/simulators/robosim tests/unit_tests/test_backends.py -q
python scripts/validate_adapter.py --module jiuwensymbiosis.simulators.robosim
```

真实 Isaac 确定性 smoke 会加载或连接机器人并实际运动，应在空载、专用仿真中运行：

```bash
python -m scripts.robosim_smoke --config configs/robosim/moz1.yaml --cartesian \
  --output outputs/robosim_smoke/pr1.json
```

它保存动作参数、实测读回和失败状态，不调用 LLM。通用 `smoke_test_adapter.py` 的桩驱动没有抓持证据，因此 `close_gripper` 应拒绝成功；通用脚本也缺少 `goto_pose` 的安全参数。对应行为由本后端的针对性测试覆盖，不能将通用桩结果当作物理验收。
