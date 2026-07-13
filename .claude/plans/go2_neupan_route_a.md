# unitree_go2 导航:路线 A —— NEUPAN + ROS2 cmd_vel(SDK 软化保留)

## 目标
把 go2 的 `_move_to_xy_yaw`([lowlevel.py:266](jiuwensymbiosis/adapters/unitree_go2/lowlevel.py#L266),当前 SDK TODO 桩)换成 NEUPAN 避障导航 + ROS2 cmd_vel 发布闭环。和 ubetech 同构,但**运动学不同**。

## 已确认的关键差异(go2 vs ubetech)
| | ubetech(已做) | go2(本次) |
|---|---|---|
| 运动学 | diff 差速 | **mecanum 麦轮**(go2_point_nav.py 用 `go2_mecanum` yaml) |
| action 维度 | 2 维 `[vx, omega]`,vy=0 | **3 维 `[vx, vy, omega]`**,vy 发出去 |
| 运动执行 | `_Ros2CmdVel`(私有类) | `_Ros2CmdVel`(提到 `_common` 后复用) |
| SDK | 无 | **有,软化保留** |

## 改动清单

### 1. 新建 `jiuwensymbiosis/adapters/_common/ros2_cmd_vel.py`
把 ubetech lowlevel 里的 `_Ros2CmdVel` 类(第 71-241 行)**原样提到公共件**,改名 `Ros2CmdVel`(去下划线,变公共)。镜像 `Ros2Odom`/`Ros2Scan` 的结构(lazy rclpy、`SingleThreadedExecutor` daemon、`_safe_teardown`、构造不抛、`start()` 返回 bool、`publish_twist(vx,vy,wz)`、`_CMD_VEL_MSG_KINDS` 支持 twist/twist_stamped)。
- 模块 docstring 说明:纯写者,速度命令由调用方(驱动)算出来后通过本件发布到 cmd_vel topic;真正的运动执行(底盘怎么响应 cmd_vel)是机器人端 bridge 的事。

### 2. 改 `jiuwensymbiosis/adapters/ubetech_cruzr_s2/lowlevel.py`(适配公共件)
- 删掉文件内的 `_Ros2CmdVel` 类(71-241 行)和 `_CMD_VEL_MSG_KINDS` 字典(移到公共件了)。
- import 改 `from jiuwensymbiosis.adapters._common.ros2_cmd_vel import Ros2CmdVel`。
- 把 `_Ros2CmdVel` → `Ros2CmdVel`(2 处引用:301、303 行的类型注解和构造)。
- 行为**完全不变**(纯搬家)。

### 3. 改 `jiuwensymbiosis/adapters/unitree_go2/lowlevel.py`(核心)
- **import**:加 `Ros2CmdVel`、`Ros2Scan`、`time`。
- **构造函数**加 kwargs:`ros2_cmd_vel_topic`、`ros2_cmd_vel_msg_kind`、`ros2_scan_topic`、`neupan_config_path`、`neupan_arrive_threshold_m`、`neupan_max_collision_count`、`neupan_control_hz`(和 ubetech 同名,便于将来 `_common` 化 nav 逻辑)。存 `self._cmd_vel`/`self._scan`/`self._planner`/`self._neupan_*`。
- **构造体**:建 `self._cmd_vel: Ros2CmdVel | None`(topic 设了才建)、`self._scan: Ros2Scan | None`、`self._planner = None`。SDK 字段(`self._sdk`、`_dds_factory_initialized`)保留不动。
- **connect()**:
  - 加 `self._cmd_vel.start()` / `self._scan.start()`(降级不抛,像 camera/odom)。
  - **SDK 软化**:`from unitree_sdk2py... import ChannelFactoryInitialize` 那段改成 `try/except ImportError` → 缺 SDK 只 `logger.warning("[Go2] SDK not installed — SDK motion path unavailable; NEUPAN/cmd_vel path still works.")`,**不 raise**。SDK init 成功就保留 `self._sdk = True`,失败/缺包就 `self._sdk = None`,都不阻塞。
  - 加 NEUPAN 规划器构建(照 ubetech:`neupan_config_path` 设了才建,缺 neupan 包 raise 带指引)。
- **close()**:加 `self._cmd_vel.stop()`、`self._scan.stop()` teardown(best-effort,像 camera/odom)。
- **加 `get_scan()`**:委托 `self._scan.grab_scan()`(ubetech 同款)。
- **重写 `_move_to_xy_yaw`**:走 NEUPAN + cmd_vel。结构和 ubetech `_nav_loop` 同构,但 **mecanum 取 3 维**:
  - `vx=action[0,0]`、`vy=action[1,0]`、`wz=action[2,0]`(3 维,vy 不恒 0)。
  - clamp:vx/vy clamp 到 `max_linear`、wz clamp 到 `max_angular`。
  - `publish_twist(vx, vy, wz)`(3 维全发)。
  - 其余(odom→state、goal_theta=atan2、update_initial_path_from_goal、ipath.arrive_threshold override、双重到达 arrive+距离、碰撞计数 cap、finally 停车、等 odom/scan)和 ubetech 完全一致。
- 把 ubetech 的 `_odom_state`/`_sleep_remaining` 辅助方法也照搬(go2 还没有这俩)。
- 现有 SDK 桩的 TODO 注释删掉(被 NEUPAN 实现替代),但 SDK init 代码留着。

### 4. 改 `jiuwensymbiosis/adapters/unitree_go2/config.py`
加 7 个字段(带中文注释):
- `ros2_cmd_vel_topic: str = "/cmd_vel"`、`ros2_cmd_vel_msg_kind: str = "twist"`(和 ubetech 同款默认)
- `ros2_scan_topic: str | None = None`
- `neupan_config_path: str | None = None`、`neupan_arrive_threshold_m: float = 0.1`、`neupan_max_collision_count: int = 100`、`neupan_control_hz: float = 10.0`
`from_yaml` 加 `neupan_config_path` 相对路径解析(照 ubetech/calib_path 同款)。

### 5. 改 `jiuwensymbiosis/adapters/unitree_go2/env.py`
- **connect() kwargs**:把 `dict(...)` 改成字面量 `{...}`(顺带修掉遗留的 C408 + 误导注释 "conditionally extended below"),加 7 个新字段透传。
- **get_observation()**:extra 加 `"scan"` 上报(照 ubetech,`getattr` + `# type: ignore[attr-defined]` 同款理由)。

### 6. 新建 `configs/unitree_go2/planner.yaml`
基于 ubetech 的 planner.yaml,但运动学改 **mecanum**:
- `robot.kinematics: 'mecanum'`、`max_speed: [0.8, 0.4, 0.6]`(照 go2_point_nav 的 go2_mecanum yaml,3 维 [vx,vy,omega])
- 其 MPC/PAN/adjust 参数照 ubetech 默认值,Cruzr→Go2 量级微调。
- 注释说明:go2 端需跑 `cmd_vel_bridge`(把 `/cmd_vel` 转 SDK SportClient),见 README。

### 7. 改 `config_template.yaml` + `configs/unitree_go2/default.yaml`
加 cmd_vel / scan / NEUPAN 段(照 ubetech 模板结构)。template 注释强调:
- cmd_vel topic 默认 `/cmd_vel`,需 go2 端 bridge 在跑。
- SDK 现已软化(缺 SDK 不报错,SDK 路径不可用但 NEUPAN/cmd_vel 仍工作)。
- neupan 用户自装。

### 8. 改 `pyproject.toml`
`[nav]` extra 已存在(ubetech 加的),go2 共用,不用动。确认 `[unitree]` extra 仍在(SDK 可选)。

### 9. 改 `README.md`
在 go2 段补 "NEUPAN navigation" 说明:cmd_vel bridge 前置 + planner.yaml(mecanum)+ SDK 软化语义。

## 测试

### `tests/unit_tests/adapters/common/test_ros2_cmd_vel.py`(新建)
镜像 `test_ros2_scan.py`:构造不抛、`start()` 缺 rclpy 返回 False、`publish_twist` 未 start 返回 False、`publish_twist` start 后发出去(用记录型 fake publisher 验证 msg 字段填对)、twist/twist_stamped 两种 msg_kind、回调路径。不依赖 rclpy。

### `tests/unit_tests/adapters/unitree_go2/test_lowlevel_neupan.py`(新建)
镜像 ubetech 的 test_lowlevel_neupan.py:mock 掉 NEUPAN + ROS2,子类化 `UnitreeGo2Driver` 注入 fake planner + fake cmd_vel + 可编程 odom/scan。验证:
1. 到位(距离 + arrive flag 双路径)
2. goal_theta = atan2
3. 碰撞 cap raise
4. 碰撞计数重置
5. **mecanum 3 维:vy 不恒 0**(和 ubetech 差速的关键差异测试)—— 断言 action[1] 被原样发到 publish_twist 的第二维
6. vx/vy/wz clamp
7. 无 scan 等待不崩
8. 无初始 odom raise
9. planner 未建 raise
10. cmd_vel 未 running raise
11. 非有限值 raise
12. planner 异常仍停车
13. **SDK 缺失不阻塞 connect**(go2 特有:mock 掉 unitree_sdk2py import 失败,connect() 不 raise,NEUPAN 路径仍可用)

### 更新 `tests/unit_tests/adapters/unitree_go2/test_env.py`
`_MockLowLevel` 加 `get_scan`;加 scan 上报到 extra 的断言(照 ubetech test_env)。

## 不做的事(surgical)
- 不动 `api.py`(`goto_xyzr` 链路已通)。
- 不删 SDK 代码(软化保留:connect 不因缺 SDK raise,但 SDK init 代码原样留)。
- 不把 nav 闭环逻辑提成 `_common`(go2 和 ubetech 的 `_nav_loop` 结构同构但运动学不同——2 维 vs 3 维,抽象要带运动学参数,等第三个底盘再说)。
- 不动 ubetech 的行为(只把 `_Ros2CmdVel` 搬家 + 改 import,纯 refactor)。

## 验证
1. `source ~/venvs/jiuwen/bin/activate && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -p anyio tests/unit_tests/adapters/unitree_go2/ tests/unit_tests/adapters/ubetech_cruzr_s2/ tests/unit_tests/adapters/common/test_ros2_cmd_vel.py tests/unit_tests/adapters/common/test_ros2_scan.py` — 全绿(含 ubetech 回归,确保搬家没破坏)。
2. `ruff check` + `ruff format --check` 相关文件。
3. `mypy jiuwensymbiosis/adapters/unitree_go2/ jiuwensymbiosis/adapters/_common/ros2_cmd_vel.py jiuwensymbiosis/adapters/ubetech_cruzr_s2/`。
4. `python scripts/validate_adapter.py --module jiuwensymbiosis.adapters.unitree_go2` — 100%。
5. `python scripts/validate_adapter.py --module jiuwensymbiosis.adapters.ubetech_cruzr_s2` — 100%(回归)。
