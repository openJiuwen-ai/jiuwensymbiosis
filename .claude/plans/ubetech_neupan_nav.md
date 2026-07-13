# ubetech_cruzr_s2 导航:集成 NEUPAN 避障规划器

## 目标
把 ubetech 的 `_move_to_xy_yaw`([lowlevel.py:427](jiuwensymbiosis/adapters/ubetech_cruzr_s2/lowlevel.py#L427))从「只发零速度的 TODO 桩」换成真正的 NEUPAN 避障导航闭环。智能体发 odom 系目标坐标 → driver 用 NEUPAN 算 `(vx, omega)` → `_Ros2CmdVel` 发 cmd_vel → 读 odom+scan 闭环到位。

## 已确认的关键事实(来自本地源码核实)
1. **NEUPAN 是纯库**,返回速度标量,不自己发 cmd_vel → 现有 `_Ros2CmdVel` publisher **复用,不冗余**。
2. **必须喂 LaserScan** → `planner.scan_to_point(state, scan_dict, ...)` 转障碍点云,scan_dict schema:`{ranges, angle_min, angle_max, range_max, range_min}`。
3. **差速模型**(用户确认)→ `action` 是 **2 维** `[vx, omega]`([neupan.py:156](file:///home/ubuntu/NeuPAN/neupan/neupan.py#L156)),**vy 恒 0**。注意:go2_point_nav.py 是 mecanum 用 3 维,别照搬那处的 `action[1,0]=vy`。
4. **API**:`neupan.init_from_yaml(yaml)` 建规划器;`update_initial_path_from_goal(start_3x1, goal_3x1)` 设路径(theta 弧度);`planner(state_3x1, points_2xN, None) -> (action_2x1, info)`;`info['arrive']`、`info['stop']`;`planner.min_distance`;`planner.ipath.arrive_threshold`。
5. **NEUPAN 自带 arrive 有缺陷**:`check_curve_arrive` 要求 point_index 接近路径末端,避障偏移后常失效。go2_point_nav.py:220-225 加了**纯距离补充检查**——照搬。
6. **重依赖**(torch/cvxpy/scipy/sklearn)→ 用户自己装,lazy import,缺了报错带指引(和 unitree_sdk2py 同款)。
7. **scan 读取范式** = 和 Ros2Odom/Ros2Camera 完全同构(async 订阅 + Lock 守 latest-slot + sync `grab_scan()`),属跨厂商件 → 放 `_common/`。

## 改动清单

### 1. 新建 `jiuwensymbiosis/adapters/_common/ros2_scan.py`
`Ros2Scan` 类,镜像 `Ros2Odom` 的全部结构(lazy rclpy、`SingleThreadedExecutor` daemon 线程、`_safe_teardown`、构造不抛、`start()` 返回 bool、`grab_scan()` 返回 dict 或 None)。
- 订阅 `sensor_msgs.msg.LaserScan`。
- 回调把 msg 转成 `{ranges: list[float], angle_min, angle_max, range_max, range_min}` 存 latest-slot(纯 python 提取,`SimpleNamespace` 可测,不依赖 rclpy)。
- `grab_scan()` 返回该 dict 或 None。
- 模块 docstring 说明:纯消费者,scan 由机器人端雷达 driver 发布。

### 2. 改 `jiuwensymbiosis/adapters/ubetech_cruzr_s2/lowlevel.py`
- `__init__` 新增 kwargs:`ros2_scan_topic`、`neupan_config_path`、`neupan_arrive_threshold_m`(默认 0.1)、`neupan_max_collision_count`(默认 100)、`neupan_control_hz`(默认 10.0)。
- 新增 `self._scan: Ros2Scan | None`(topic 设了才建)。
- `connect()`:`start()` scan 订阅(降级不抛);**lazy 建 NEUPAN 规划器**——`from neupan import neupan as _np; self._planner = _np.init_from_yaml(config_path)`。缺 neupan 包 → raise RuntimeError 带 `pip install` 指引(运动主能力,不降级)。`init_from_yaml` 失败也 raise。
- `close()`:加 scan `.stop()` teardown(同 camera/odom 的 `except Exception as e: logger.debug`)。
- 新增 `get_scan() -> dict | None`:委托 `self._scan.grab_scan()`。
- **重写 `_move_to_xy_yaw`**(替换 TODO 桩):
  - 入口校验:connected、cmd_vel running、planner 已建、target 有限。
  - 读 start pose(`get_odom_pose()`;None → 报错"等 odom")。
  - `goal_theta = atan2(target_y - start_y, target_x - start_x)`;`goal = [[target_x],[target_y],[goal_theta]]`;`planner.update_initial_path_from_goal(start_3x1, goal)`。
  - 若 `arrive_threshold` 显式配置 → `planner.ipath.arrive_threshold = ...`。
  - 循环 `while running`(按 `control_hz` sleep):
    - `state = odom → 3x1`(yaw 从 `yaw_deg` 转 rad);无 odom → sleep 继续。
    - `scan = get_scan()`;无 scan 或 ranges 空 → sleep 继续(不发速度,等数据)。
    - `points = planner.scan_to_point(state, scan, [0,0,0], [-pi,pi], 2)`。
    - `action, info = planner(state, points, None)`。
    - 到达:`info['arrive']` **或** `hypot(state-target) < arrive_threshold` → 发 (0,0,0) 退出。
    - 碰撞:`info['stop']` → 发 (0,0,0)、collision_count++、超 max → raise RuntimeError;否则清零。
    - 正常:`vx=action[0,0]`、`wz=action[1,0]`、`vy=0.0`(差速);**clamp 到 max_linear/max_angular**(安全边界在驱动层);`publish_twist(vx,0,wz)`。
  - `finally` 无条件 `publish_twist(0,0,0)` 停车(防悬留速度)。
  - 用 `time.monotonic()` + 超时上限(可选,先不加硬超时,靠碰撞计数兜底)。

### 3. 改 `config.py`
加 5 个字段(带中文注释 `[选填]`):
- `ros2_scan_topic: str | None = None`(NEUPAN 避障必需;留空则不订阅 scan)
- `neupan_config_path: str | None = None`(NEUPAN planner.yaml 路径;相对 config 文件解析,仿 calib_path)
- `neupan_arrive_threshold_m: float = 0.1`
- `neupan_max_collision_count: int = 100`
- `neupan_control_hz: float = 10.0`
`from_yaml` 里把 `neupan_config_path` 按相对路径解析(复用 calib_path 同款逻辑)。

### 4. 改 `env.py`
- `connect()` kwargs 字面量加这 5 个字段(保持字面量形态,不引入 dict()+noqa)。
- `get_observation()` 的 `extra` 加 `"scan"`(可选,镜像 odom 模式;方便 trace/调试)。需 driver 有 `get_scan()`——env 里 `getattr(self._low_level, "get_scan", None)` 安全取。

### 5. 新建 `configs/ubetech_cruzr_s2/planner.yaml`
基于 `/home/ubuntu/NeuPAN/example/pf/diff/planner.yaml` 改的 diff 模型默认配置:`kinematics: 'diff'`、`max_speed` 调到 ubetech 量级(如 `[1.0, 1.5]`)、`ref_speed: 0.8`、`arrive_threshold: 0.1`、`d_max/d_min/eta` 适中、`device: 'cpu'`。注释说明这是起点,真机要调。

### 6. 改 `config_template.yaml` + `configs/ubetech_cruzr_s2/default.yaml`
加 NEUPAN 段:scan topic、neupan_config_path(指向同目录 planner.yaml)、3 个调参字段。template 带安装/调参说明。

### 7. 改 `pyproject.toml`
加 `[nav]` extra(可选依赖,用户自选装):`nav = ["neupan"]`。但 neupan 不在 PyPI 上(本地 `-e` 装),所以 extra 注释说明「从本地 NeuPAN 仓库 `pip install -e .`」。

### 8. 改 `README.md`
加 "NEUPAN navigation (optional)" 段:说明 ubetech 导航用 NEUPAN 避障;装 neupan(本地仓库)+ scan topic 必需 + planner.yaml 位置 + 调参指引。

## 测试
### `tests/unit_tests/adapters/common/test_ros2_scan.py`(新建)
镜像 `test_ros2_odom.py`:构造不抛、`start()` 缺 rclpy 返回 False、`grab_scan()` None 直到首消息、回调丢畸形消息、msg→dict 提取正确。不依赖 rclpy。

### `tests/unit_tests/adapters/ubetech_cruzr_s2/test_lowlevel_neupan.py`(新建)
mock 掉真 NEUPAN + 真 ROS2:子类化 `UbetechCruzrS2Driver`,注入 fake planner(记录每次 `__call__` 的 state/points,返回可编程的 `(action, info)` 序列)+ fake cmd_vel(记录 publish)+ 可编程 odom/scan。验证:
1. `test_move_calls_planner_and_publishes` — 到达前每 tick 调 planner、发非零 cmd_vel、最后发 (0,0,0)。
2. `test_move_arrives_on_info_arrive` — planner 第 N 次 `info['arrive']=True` → 退出且发停车。
3. `test_move_arrives_on_distance` — planner 不报 arrive 但 odom 距离 < threshold → 退出(照搬 go2_point_nav 的纯距离补充)。
4. `test_move_collision_raises_after_max` — planner 连续 `info['stop']=True` 超 max_collision_count → raise RuntimeError,且发过 (0,0,0)。
5. `test_move_diff_model_vy_zero` — 断言所有 publish 的 vy==0(差速)。
6. `test_move_speed_clamped` — planner 返回超大 action → publish 的 vx/wz 不超 max。
7. `test_move_stops_on_exception` — planner 中途抛 → finally 仍发 (0,0,0)。
8. `test_move_waits_for_odom_and_scan` — 首几 tick 无 odom/scan → 不调 planner、不发速度、不崩。
9. `test_goal_theta_from_atan2` — 目标在第象限 → goal_theta 正确传给 planner。
10. `test_raises_when_planner_not_built` — 未 connect → raise RuntimeError。

全部不依赖 rclpy/neupan/torch。

### 更新 `tests/unit_tests/adapters/ubetech_cruzr_s2/test_env.py`
`_MockLowLevel` 加 `get_scan` 方法;加 1-2 个 scan 上报到 extra 的断言。

## 不做的事(surgical)
- 不动 `api.py`(`goto_xyzr` 链路已通,语义正确)。
- 不动 unitree_go2(本次只 ubetech;go2 SDK 路径不同,且 go2 已有独立的 go2_point_nav 脚本)。
- 不写 P 控制回退(用户明确:纯 NEUPAN,缺则报错)。
- 不把 NEUPAN 提到 `_common/`(它当前只 ubetech 用;等第二个避障底盘到来再说——但 `Ros2Scan` 提到 `_common`,因为它是通用 ROS2 消费者)。

## 验证
1. `source ~/venvs/jiuwen/bin/activate && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -p anyio tests/unit_tests/adapters/ubetech_cruzr_s2/ tests/unit_tests/adapters/common/test_ros2_scan.py` — 全绿。
2. `ruff check` + `ruff format --check` 相关文件。
3. `mypy jiuwensymbiosis/adapters/ubetech_cruzr_s2/ jiuwensymbiosis/adapters/_common/ros2_scan.py`(建议性)。
4. `python scripts/validate_adapter.py --module jiuwensymbiosis.adapters.ubetech_cruzr_s2` — 100%。
