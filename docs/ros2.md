# ROS2 后端使用指南（相机 / 里程计 / 底盘运动）

> JiuwenSymbiosis 框架的 ROS2 后端是**机器人无关**的桥接层：把 ROS2 异步 pub/sub 模型（消息只在 executor spin 时到达）桥接为框架要求的**同步**契约——一次非阻塞调用返回最新帧 / 最新位姿 / 一次速度命令发布，或 `None`。所有后端共享同一套 lazy-rclpy + 守护线程 + 优雅降级模式，缺失 rclpy 时**永不 raise**，只退化为 `None`。

---

## 一、总览：三个 ROS2 后端

| 后端 | 模块 | 方向 | 用途 | 使用者 |
|---|---|---|---|---|
| **相机** | `adapters/_common/ros2_camera.py:Ros2Camera` | 订阅 | 读 RGBD 帧（`sensor_msgs/Image`） | piper、unitree_go2、ubetech_cruzr_s2 |
| **里程计** | `adapters/_common/ros2_odom.py:Ros2Odom` | 订阅 | 读位姿（`nav_msgs/Odometry` 等） | piper、unitree_go2、ubetech_cruzr_s2 |
| **底盘运动** | `adapters/ubetech_cruzr_s2/lowlevel.py:_Ros2CmdVel` | 发布 | 发速度命令（`geometry_msgs/Twist` 等） | ubetech_cruzr_s2 |

相机与里程计后端位于 `adapters/_common/`，所有适配器共用；底盘运动后端目前仅 `ubetech_cruzr_s2` 使用（纯 ROS2、无 vendor SDK 的运动方案），实现在适配器包内。

---

## 二、安装

三个后端的依赖（`rclpy` 及各消息包）**都不在 PyPI**——它们随 ROS2 本身发布。因此框架的 `[ros2]`-style extra **故意不包含**它们：不能在普通 `pip install` 里拉到。需从 ROS apt 源安装并在**每个**运行框架的 shell 里 source ROS 环境。

```bash
# 1. 安装 ROS2（推荐 Humble）：https://docs.ros.org/en/humble/Installation.html
#    这会带入 rclpy + sensor_msgs。或只装最小包：
sudo apt install ros-humble-rclpy ros-humble-sensor-msgs ros-humble-std-msgs
#    里程计后端额外需要：
sudo apt install ros-humble-nav-msgs ros-humble-geometry-msgs
#    底盘运动后端额外需要（geometry_msgs 已随上面装上）：
#    （Twist / TwistStamped 都在 geometry_msgs 里，无需额外包）

# 2. 在【每个】运行框架的 shell 里激活 ROS 环境（这样 Python 才能 import rclpy）。
#    最好加进 ~/.bashrc 或 venv 的 activate 脚本：
source /opt/ros/humble/setup.bash

# 3. 在 YAML 里把对应字段指向你的 ROS2 topic（见下文各后端章节）。
```

> ⚠️ **解释器兼容性**：`rclpy` 是绑定了 ROS2 发行版 Python 的 C 扩展。若你的 conda/venv 用的 Python 版本与系统 ROS2 的 Python 不一致（例如 ROS2 Humble 用 3.10，而你建了 3.11 环境），`import rclpy` 会失败。两种解法：
> - 用系统 Python（`/usr/bin/python3.10`）建 venv：`python3.10 -m venv ~/venvs/jiuwen`；
> - 或在 conda 环境里 `pip install` 一套匹配的 `rclpy`（部分 ROS2 发行版有 conda-forge 包，但兼容性需自行验证）。
>
> 失败时所有后端**优雅降级**为 `None`（见下文），框架不会崩。

---

## 三、ROS2 相机后端（`Ros2Camera`）

把 ROS2 `sensor_msgs/Image` 流桥接为同步 `grab_frames() -> (rgb_uint8, depth_m_float32) | None`。

### 3.1 配置字段

| 字段 | 默认 | 说明 |
|---|---|---|
| `camera_source` | `"ros2"` | 选 `"ros2"` 启用此后端（其他值走 USB RealSense 路径，需适配器自行实现） |
| `ros2_rgb_topic` | `None` | RGB topic 名；**留空则相机不启动** |
| `ros2_depth_topic` | `None` | 深度 topic 名（可选；留空则只回 RGB） |
| `ros2_depth_scale_m` | `0.001` | 16UC1 原始单位 → 米（RealSense 默认 1mm） |
| `ros2_camera_info_topic` | `None` | `sensor_msgs/CameraInfo` topic（可选；提供内参） |
| `ros2_intrinsics` | `None` | 无 camera_info 时，手填行优先 9 元素 3x3 K 矩阵 |

### 3.2 帧转换

纯 numpy 实现（**不用 cv_bridge**——它对 numpy>=2 有兼容问题）。支持的 encoding：`rgb8`、`bgr8`、`rgba8`、`bgra8`（RGB 输出）；`16uc1`、`32fc1`（深度输出）。编码不识别或 buffer 长度不匹配时返回 `None`，不 raise。

### 3.3 降级行为

`rclpy` 不可 import、`start()` 失败、或尚未收到任何帧时：`grab_frames()` 返回 `None`。框架把"无相机"和"无帧"同等对待，走 `ok=False, reason=no_camera` 的回退链——与缺 RealSense 时完全一致。

---

## 四、ROS2 里程计后端（`Ros2Odom`）

把一个携带位姿的 ROS2 topic 桥接为同步 `grab_pose() -> dict | None`。

### 4.1 配置字段

| 字段 | 默认 | 说明 |
|---|---|---|
| `ros2_odom_topic` | `None` | odom topic 名；**留空则禁用 odom** |
| `ros2_odom_msg_kind` | `"odometry"` | 消息类型（见下表） |

`msg_kind` 取值：

| `msg_kind` | ROS2 消息类型 | 位姿取自 |
|---|---|---|
| `"odometry"` | `nav_msgs/Odometry` | `msg.pose.pose` |
| `"pose_stamped"` | `geometry_msgs/PoseStamped` | `msg.pose` |
| `"pose_with_covariance_stamped"` | `geometry_msgs/PoseWithCovarianceStamped` | `msg.pose.pose` |

### 4.2 返回值

```python
{
    "x": float, "y": float, "z": float,     # 米
    "qx": float, "qy": float, "qz": float, "qw": float,  # 四元数
    "yaw_deg": float,                        # 便捷平面航向（deg）
}
```

位姿是**原始 ROS 单位**（米 + 四元数），**不会**自动换算到机械臂 flange 帧。消费者（如 `unitree_go2` / `cruzr_s2` 的 driver）把它包成 6-DoF `SimpleNamespace` 供 `get_flange_pose()` 透传；Api 层决定各分量语义。

### 4.3 SLAM 责任边界（重要）

⚠️ 框架是 odom topic 的**纯消费者**——**自身不跑任何 SLAM / 定位 / 建图**。topic 上发布的位姿必须由**机器人端**的外部 SLAM / 里程计栈产生，由集成方在框架之外部署：

- LiDAR SLAM 节点（cartographer、slam_toolbox、FAST-LIO、LIO-SAM…）
- 视觉惯性里程计（VINS-Fusion、ORB-SLAM3、RTAB-Map…）
- 轮式编码器 + IMU 融合（robot_localization EKF）

**先把这个栈起起来并收敛**，再把 `ros2_odom_topic` 指向它发布的 topic。若 SLAM 没跑或没收敛，没有消息发布，`grab_pose()` 就返回 `None`——框架照常工作，只是没有外部位姿输入（与缺相机同等的"无数据"回退）。

### 4.4 降级行为

`rclpy` 不可 import、`start()` 失败、或尚未收到消息时：`grab_pose()` 返回 `None`。driver 的 `get_pose()` 在 odom 为 None 时**回退到 home 位姿**，保证调用方总能拿到一个合法 pose 对象。

---

## 五、ROS2 底盘运动后端（`_Ros2CmdVel`）

把一次速度命令发布到 ROS2 topic（`geometry_msgs/Twist` 或 `TwistStamped`）。目前仅 `ubetech_cruzr_s2` 适配器使用——它是"纯 ROS2、不装任何 vendor SDK"的运动方案。与 `Ros2Odom` 同模式（lazy-rclpy + 守护线程 + 降级）。

### 5.1 配置字段

| 字段 | 默认 | 说明 |
|---|---|---|
| `ros2_cmd_vel_topic` | `"/cmd_vel"` | 速度命令发布到的 topic 名（**可配置**，按你的 driver 改） |
| `ros2_cmd_vel_msg_kind` | `"twist"` | 速度消息类型（见下表） |
| `max_linear_speed_mps` | `1.0` | 线速度上限 m/s（driver 在发布前钳制） |
| `max_angular_speed_radps` | `1.5` | 角速度上限 rad/s |

`msg_kind` 取值：

| `msg_kind` | ROS2 消息类型 | 速度取自 |
|---|---|---|
| `"twist"` | `geometry_msgs/Twist` | `msg.linear.x/y` + `msg.angular.z` |
| `"twist_stamped"` | `geometry_msgs/TwistStamped` | `msg.twist.linear.x/y` + `msg.twist.angular.z`（带 header） |

topic 名与消息类型**均可配置**，以适配不同机器人 ROS2 driver 暴露的接口。

### 5.2 运动控制模型

driver 的 `move_to_pose_blocking(pose)` 取 `pose.x/y`（米）+ `pose.rz`（yaw deg），用比例速度控制器向目标驱动：每 tick 发 `v=clamp(kp*err, vmax)`，轮询 `get_pose()`（来自 odom）直至收敛或超时。

> ⚠️ **集成 seam**：`lowlevel.py:_move_to_xy_yaw` 的控制增益 / 容差 / 超时是按实际部署调参的接缝。当前为骨架（发布零速度标记），标注 `TODO(integration)`——把 `kp` / `tol` / `timeout` 按你的机器人调好再上线。

### 5.3 降级行为

与 camera/odom 不同：**运动是底盘的主能力**，缺 rclpy 时不能"静默降级为 None"后假装能动。`connect()` 阶段 rclpy 缺失**不 raise**（与 camera/odom 一致降级，因为纯 ROS2 方案下三者一起缺），但调用 `move_to_pose_blocking()` / `home()` 时会 raise `RuntimeError("[CruzrS2] cmd_vel not running ...")`——让"运动不可用"在调用点显式暴露，而非静默吞掉。

---

## 六、各适配器的 ROS2 后端使用情况

| 适配器 | 相机 | 里程计 | 运动 | 说明 |
|---|---|---|---|---|
| **piper** | ✅ `ros2` 可选 | ✅ 可选 | ❌（走 `piper_sdk` CAN） | 6-DoF 臂 + 夹爪 + 腕部相机；运动不走 ROS2 |
| **unitree_go2** | ✅ `ros2` | ✅ 可选 | ❌（走 `unitree_sdk2py` Cyclone DDS） | 四足底盘；运动走官方 SDK，图像/odom 走 ROS2 |
| **ubetech_cruzr_s2** | ✅ `ros2` | ✅ 可选 | ✅ `cmd_vel` | 纯 ROS2 方案，**无需任何 vendor SDK** |

---

## 七、设计要点

1. **构造永不 raise**：`Ros2Camera` / `Ros2Odom` / `_Ros2CmdVel` 的 `__init__` 只存配置；未知 `msg_kind` 降级为默认值 + warning，而非 `ValueError`。
2. **lazy rclpy**：`rclpy` 与消息包只在 `start()` 里 import。缺失时 `start()` 返回 `False` 并 log warning，后续调用降级为 `None` / no-op。
3. **守护线程 spin**：每个后端用 `SingleThreadedExecutor` + daemon 线程 spin，把异步消息桥接为同步读取。`stop()` 幂等、best-effort teardown。
4. **优雅降级链**：缺 rclpy → `start()=False` → `grab_*/publish` 返回 `None`/no-op → 框架走"无数据"回退链，不崩。运动后端在**调用点** raise 而非静默（运动是主能力，不能假装能动）。
5. **SLAM 责任在机器人侧**：框架只消费 odom topic，不跑定位/建图。先把 SLAM 栈起起来并收敛，再指向它发布的 topic。

详见各模块 docstring：`adapters/_common/ros2_camera.py`、`adapters/_common/ros2_odom.py`、`adapters/ubetech_cruzr_s2/lowlevel.py`。
