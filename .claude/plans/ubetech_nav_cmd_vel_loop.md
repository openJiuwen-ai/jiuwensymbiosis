# ubetech_cruzr_s2 导航:补全 cmd_vel 比例控制循环

## 背景 / 现状

ubetech 脚手架已经搭好,链路通了:

```
智能体 → goto_xyzr(x,y,r)            [api.py]
     → env.move_to_flange(pose)      [env.py]
     → driver.move_to_pose_blocking  [lowlevel.py:412]
     → _move_to_xy_yaw()             [lowlevel.py:427]  ← 唯一缺口
```

缺口在 `_move_to_xy_yaw`([lowlevel.py:427-453](jiuwensymbiosis/adapters/ubetech_cruzr_s2/lowlevel.py#L427)):现在它只发一个零速度就返回,根本没走到目标。读端已就绪:`get_pose()` 读 `Ros2Odom` 返回当前 `x/y/yaw`([lowlevel.py:376](jiuwensymbiosis/adapters/ubetech_cruzr_s2/lowlevel.py#L376));写端 `_Ros2CmdVel.publish_twist(vx,vy,wz)` 也已就绪([lowlevel.py:175](jiuwensymbiosis/adapters/ubetech_cruzr_s2/lowlevel.py#L175))。

只差「目标 → 一串 cmd_vel → 读 odom 闭环到位」这个控制循环本身。这是上一轮确认的**方案 A**:闭环逻辑内联在驱动层,契合 `RobotDriver.move_to_pose_blocking` 的 blocking 契约,不碰 Env/Api 分层。符合 karpathy 原则(不预抽象到 `_common/`,等第二个移动底盘真正到来再说)。

## 坐标系约定(关键,易错)

- 里程计坐标系(odom frame):`x/y` 米,`yaw` 度。目标 `target_x/y/yaw` 也是 odom frame 绝对坐标(智能体发布的就是 odom 系目标)。
- cmd_vel 是**基座坐标系**(base frame,随底盘转向):`vx` 前进、`vy` 侧移、`wz` 自转。
- 所以控制律要先把 odom 系的位置误差旋到 base 系:
  - 位置误差(odom 系):`dx = target_x - cur_x`,`dy = target_y - cur_y`
  - 旋到 base 系:`err_x = cos(yaw)*dx + sin(yaw)*dy`,`err_y = -sin(yaw)*dx + cos(yaw)*dy`
  - 前进速度 `vx = clamp(kp_linear * err_x, ±vmax)`;侧移 `vy = clamp(kp_linear * err_y, ±vmax)`(差速/全向底盘才用 vy;这个驱动先都发,不用的底盘收到 vy 自然忽略)
  - 航向误差(短角):`err_yaw = wrap(target_yaw - cur_yaw)` 到 `[-180,180]`;`wz = clamp(kp_angular * deg2rad(err_yaw), ±wmax)`
- 到位判据:`hypot(err_x_odom, err_y_odom) < xy_tol_m` **且** `abs(err_yaw) < yaw_tol_deg`。

## 改动清单(4 个文件)

### 1. `config.py` — 新增控制律可调参数
在 `max_angular_speed_radps` 之后、`home_xy_yaw_m_deg` 之前加一组:
- `nav_kp_linear: float = 1.0` 位置→速度比例增益(无量纲:m 误差 → m/s 命令,再 clamp)
- `nav_kp_angular: float = 2.0` 航向→角速度增益(deg 误差 → rad/s 命令,再 clamp)
- `nav_xy_tol_m: float = 0.05` 到位线容差(米)
- `nav_yaw_tol_deg: float = 5.0` 到位角容差(度)
- `nav_timeout_s: float = 30.0` 单次导航超时(秒),超时 raise RuntimeError(防止 odom 丢失时无限发速度)
- `nav_control_hz: float = 20.0` 控制循环频率

都带中文注释,标 `[选填]`,说明是集成调参接缝。

### 2. `lowlevel.py` — 实现 `_move_to_xy_yaw` 控制循环
- `UbetechCruzrS2Driver.__init__` 新增同名 6 个 kwargs 存为 `self._nav_*`。
- 把 `_move_to_xy_yaw` 的 TODO 桩替换成真正的比例闭环:
  - 入口校验保持(connected/cmd_vel running/finite)。
  - `deadline = monotonic() + timeout`;循环 `while not 到位 and monotonic() < deadline`:
    - `cur = self.get_pose()`(读 odom;若 None 用 home_pose 退避)
    - 算 base 系误差 → `vx/vy/wz` → `publish_twist`(失败记 debug 继续)
    - `sleep(1/control_hz)`
  - 循环退出后**无条件发一次 `publish_twist(0,0,0)` 停车**(无论到位/超时/异常),确保底盘不悬留速度。
  - 超时 raise `RuntimeError("[CruzrS2] nav timeout ...")`。
- 用 `time.monotonic()`(不是 `time.time()`,不受时钟跳变影响)。
- 注意 `import time` 加到文件头。
- **速度限幅在驱动层**(`clamp(v, -vmax, vmax)`),符合 security.md「速度/力限制属于 driver」——不发超过 `max_linear_speed_mps`/`max_angular_speed_radps` 的命令。

### 3. `env.py` — connect() kwargs 透传
`connect()` 的 `kwargs` dict 加 6 个 `nav_*` 字段(照现有 `max_*` 同款透传)。**这里正是上一轮悬而未决的 C408 问题**——现 env.py:107 已是字面量 `{...}`(用户改过),加字段就加进字面量,保持字面量形态(不引入 `dict()`+noqa)。

### 4. `config_template.yaml` + `configs/ubetech_cruzr_s2/default.yaml`
两个 YAML 加 `nav_*` 段(注释 + 默认值),template 带调参说明,default.yaml 给定值。

## 测试(`tests/unit_tests/adapters/ubetech_cruzr_s2/test_lowlevel.py` — 新建)

参照 `test_env.py` 的 `_MockLowLevel` 风格,但反过来:这里被测对象是 **driver 本身**,mock 掉 `_Ros2CmdVel`(不发真 ROS2)和 odom(给可编程的位姿序列)。

关键:用一个**可编程 odom**——让 `get_pose()` 返回一个由「已发布的 cmd_vel 累积积分」驱动的虚拟位姿,这样闭环能真正收敛(每 tick 用发布的 vx/vy/wz 推进位姿),验证:
1. `test_move_converges_to_target` — 目标 (2,1,90°),从 (0,0,0) 出发,虚拟 odom 按发布的速度积分,断言最终 `publish_twist` 被以非零速度调用过、且最后那次是 (0,0,0) 停车、循环退出时位姿在容差内。
2. `test_move_stops_on_timeout` — odom 永远卡在原点(不积分),`nav_timeout_s` 取极小值,断言 raise RuntimeError 且最后发了 (0,0,0)。
3. `test_move_clamps_velocity_to_max` — 目标极远,断言单次 `publish_twist` 的 `vx` 不超过 `max_linear_speed_mps`、`wz` 不超过 `max_angular_speed_radps`。
4. `test_move_raises_when_cmd_vel_not_running` — 未 connect,断言 raise RuntimeError。
5. `test_move_publishes_stop_on_exception` — 让 `publish_twist` 中途抛异常,断言仍尝试发停车(不悬留)。
6. `test_non_finite_target_raises` — target 含 NaN,断言 ValueError(入口校验)。

mock 策略:子类化 `UbetechCruzrS2Driver`,覆写 `__init__` 跳过 `_Ros2CmdVel`/`Ros2Odom` 构造,注入一个记录型 fake publisher + 可编程 odom reader。**不碰 rclpy**——所有测试无 ROS2 环境可跑(和 test_ros2_odom 一致)。

## 不做的事(surgical)
- 不动 `api.py`(`goto_xyzr` 链路已通,语义正确)。
- 不动 `_common/`(不预抽象 `Ros2CmdVel` 出去——ubetech 的 `_Ros2CmdVel` 是模块内私有类,留着;等第二个移动底盘真正到来再提)。
- 不动 unitree_go2(本次只集成 ubetech;go2 的 SDK 路径是另一回事)。
- 不引入新依赖(纯 `math`/`time`/`threading`,已有)。

## 验证
1. `source ~/venvs/jiuwen/bin/activate && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -p anyio tests/unit_tests/adapters/ubetech_cruzr_s2/` — 全绿(含新 test_lowlevel)。
2. `ruff check jiuwensymbiosis/adapters/ubetech_cruzr_s2/ tests/unit_tests/adapters/ubetech_cruzr_s2/` — 无新增违规。
3. `mypy jiuwensymbiosis/adapters/ubetech_cruzr_s2/` — 无新增类型错误(建议性)。
4. `python scripts/validate_adapter.py --module jiuwensymbiosis.adapters.ubetech_cruzr_s2` — 100%。
