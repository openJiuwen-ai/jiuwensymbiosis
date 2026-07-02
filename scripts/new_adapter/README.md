# 本体适配器生成向导（new_adapter）

把一个新机器人的「本体适配」从头手写 2K+ 适配代码，简化成一次问答式生成和少量 SDK 对接。通常只需要补齐 `lowlevel.py` / `api.py` 里真正依赖硬件的函数，最终手写代码可以控制在 **200 行以内**。

生成程序：[scripts/new_adapter/main.py](main.py)，会产出 `jiuwensymbiosis/adapters/<name>/` 和 `configs/<name>/default.yaml`。

**它主要做什么**

- 自动生成符合框架契约的 `config.py`、`lowlevel.py`、`env.py`、`api.py`、`session.py`、`__init__.py`；
- 根据自由度、夹爪/吸盘、相机、检测、连接方式，自动声明 capabilities 和 API mixin；
- 把 `configs/<name>/default.yaml` 的连接参数一路传到 low-level driver；
- 给每个待接入函数放入可运行 mock 和参考形状，让新本体能先离线通过结构校验；
- 用 `GENERATED-MOCK` 标记剩余待填项，重跑命令即可继续扫描和复检。

**什么时候使用**

- 要接入一个新机器人、本体、夹爪或相机组合；
- 想先得到一个能 `validate` / `smoke` 的最小可运行骨架；
- 不想手动复制 piper 等适配器的大量框架胶水代码；
- 已经有机器人 SDK，但还不确定 `Env`、`Api`、capabilities、builder 应该怎么写。

---

## 目录

1. [准备工作](#1-准备工作)
2. [开始生成（三步）](#2-开始生成三步)
3. [补全真实 SDK](#3-补全真实-sdk)
4. [测试与复检](#4-测试与复检)
5. [生成物说明](#5-生成物说明)
6. [常见问题](#6-常见问题)
7. [常用参数速查](#7-常用参数速查)

---

## 1. 准备工作

### 1.1 进入仓库根目录

所有命令都从仓库根目录运行：

```bash
cd /path/to/jiuwensymbiosis
```

如果你使用本项目已有 conda 环境，建议带上 `PYTHONPATH=$PWD`：

```bash
PYTHONPATH=$PWD conda run -n jiuwensymbiosis python -m scripts.new_adapter.main --help
```

如果当前 shell 已经激活了正确环境，也可以直接：

```bash
python -m scripts.new_adapter.main --help
```

### 1.2 想清楚几个硬件问题

向导会问这些问题，不确定时可以在交互里输入 `?` 查看解释：

| 问题 | 说明 |
|------|------|
| 适配器名字 | 小写包名，如 `my_arm`、`demo_scara` |
| 自由度 | `4` 表示 SCARA 风格 `x,y,z,r`；`6` 表示六轴 `x,y,z,rx,ry,rz` |
| 是否支持关节运动 | SDK 是否有类似 `move_joint(q)` 的接口 |
| 末端执行器 | `none` / `parallel` / `suction` |
| 是否有相机 | 是否能取 RGB 或 RGB-D 图像 |
| 是否做目标检测 | 是否需要 `object_name -> 3D 抓取点` |
| 工具安装方式 | 垂直向下还是有倾斜/偏移 |
| 连接方式 | 当前 CAN 会生成较完整模板；其它方式先生成空模板 |

---

## 2. 开始生成（三步）

### 第 1 步：运行交互式向导

```bash
PYTHONPATH=$PWD conda run -n jiuwensymbiosis python -m scripts.new_adapter.main
```

程序会一路询问本体信息，然后进入阶段 A：

```text
[阶段 A] 生成骨架
  能力: motion.cartesian, motion.joint, grasp.parallel
  连接方式: can
  + jiuwensymbiosis/adapters/my_arm/config.py
  + jiuwensymbiosis/adapters/my_arm/lowlevel.py
  ...
  [✓] validate (静态结构)
```

生成后的骨架仍然是 mock，但已经是框架可识别、可校验的本体适配器。

### 第 2 步：按清单补函数

阶段 B 会列出还需要补的函数，例如：

```text
还有 7 处需要你用机器人的真实 SDK / 标定来补充。

jiuwensymbiosis/adapters/my_arm/lowlevel.py
    - connect()   打开硬件连接，必须幂等。
    - get_pose()  读当前末端位姿。
    - home()      阻塞式回零/回原点。
```

打开提示的文件，找到对应函数，把函数体里标着：

```python
# >>> GENERATED-MOCK: replace with real hardware <<<
```

的 mock 实现替换成真实 SDK 调用。每完成一个函数，就删除这行 `GENERATED-MOCK` 标记。

### 第 3 步：回车复检，或稍后续跑

补完一轮后，在向导里直接回车，程序会重新扫描标记并跑一次结构校验。

如果暂时不想继续，输入：

```text
q
```

以后重跑同一个名字即可续跑阶段 B：

```bash
PYTHONPATH=$PWD conda run -n jiuwensymbiosis python -m scripts.new_adapter.main --name my_arm
```

---

## 3. 补全真实 SDK

### 3.1 先改配置

先打开：

```text
configs/<name>/default.yaml
```

CAN 模板会包含：

```yaml
connection: "can"
can_port: "can0"
can_bitrate: 1000000
move_speed: 50
tool_offset_mm: 0.0
home_pose_xyzrxryrz_mm_deg: [200.0, 0.0, 250.0, 0.0, 90.0, 0.0]
```

这些值会通过 `config.py -> env.py -> lowlevel.py` 传到 driver。真实硬件参数优先改 YAML，不建议直接改 `lowlevel.py` 里的默认值；那些默认值只是离线 mock 的兜底。

### 3.2 再改 lowlevel.py

`lowlevel.py` 是真正碰硬件的地方。优先补这些函数：

| 函数 | 要做什么 |
|------|----------|
| `connect()` | 创建 SDK client、打开连接、使能机器人；必须幂等 |
| `disconnect()` | 释放连接；任意状态调用都应安全 |
| `get_pose()` | 返回 FLANGE 系末端位姿，单位保持 mm/deg |
| `home()` | 阻塞式回零或移动到 home |
| `move_to_pose_blocking()` | 阻塞式笛卡尔运动 |
| `move_joint_blocking()` | 如果启用 `--joint`，补关节运动 |
| `set_gripper()` / `set_suction()` | 如果有末端执行器，补开合/吸附释放 |
| `grab_frames()` | 如果有相机，返回 `(rgb, depth_m)` 或 `None` |

CAN 模板的 `connect()` docstring 里会给一个参考形状：

```python
from robot_sdk import RobotClient
self._client = RobotClient(channel=self.can_port, bitrate=self.can_bitrate)
self._client.connect()
self._client.enable()
self._connected = True
```

这不是可直接运行的 SDK 名称，只是告诉你应把“创建 client、连接、使能、记录状态”放在这个函数里。

### 3.3 有检测时再改 api.py

如果生成时选择了 `--detection`，还需要补：

| 函数 | 要做什么 |
|------|----------|
| `get_grasp_info_simple()` | 根据 `object_name` 检测目标，输出抓取点 |
| `pixel_to_base_xyz()` | 像素和深度反投影到基座坐标 |
| `analyze_scene()` | 返回高层场景分析结果 |

这几项通常依赖检测服务、相机内参、手眼标定文件。手眼标定流程见 [docs/hand-eye-calibration.md](../../docs/hand-eye-calibration.md)。

---

## 4. 测试与复检

### 4.1 只检查生成器本身

```bash
python3 -m py_compile scripts/new_adapter/render.py
```

### 4.2 跑 new_adapter 单测

当前环境没有 `black` 时，先跳过 `black_clean`：

```bash
PYTHONPATH=$PWD conda run -n jiuwensymbiosis python -m pytest \
    tests/unit_tests/scripts/test_new_adapter.py -q -k 'not black_clean'
```

预期结果类似：

```text
11 passed, 1 deselected
```

### 4.3 校验某个生成出的本体

结构校验：

```bash
PYTHONPATH=$PWD conda run -n jiuwensymbiosis python scripts/validate_adapter.py \
    --module jiuwensymbiosis.adapters.my_arm --errors-only
```

如果还没接真机，保留 mock 时可以先只看结构校验。等真实 SDK 函数补完后，再根据本体自己的调试脚本或 session 进行真机 smoke。

### 4.4 非交互生成一个测试本体

```bash
PYTHONPATH=$PWD conda run -n jiuwensymbiosis python -m scripts.new_adapter.main \
    --name my_arm --dof 6 --joint --end-effector parallel --connection can \
    --non-interactive --force
```

注意：`--force` 会覆盖同名生成物，只适合确认要重建的场景。

---

## 5. 生成物说明

| 文件 | 作用 |
|------|------|
| `config.py` | 配置 dataclass，负责从 YAML 加载硬件参数 |
| `lowlevel.py` | 真实 SDK 调用层，所有硬件 I/O 都应收敛在这里 |
| `env.py` | 框架环境层，把 config 传给 driver，并实现观测/安全边界 |
| `api.py` | 对外工具 API，处理 tip/flange 几何、视觉抓取等高层语义 |
| `session.py` | 一行从 YAML 构建 session 的入口 |
| `__init__.py` | 包导出 |
| `configs/<name>/default.yaml` | 真实部署时优先修改的配置文件 |

**经验顺序**

1. 先让 `connect()` / `disconnect()` 能稳定反复调用。
2. 再让 `get_pose()` 返回真实位姿。
3. 然后接 `home()` 和 `move_to_pose_blocking()`。
4. 最后接夹爪、相机、检测。

这样每一步都能用向导复检，不会一次性把问题堆到最后。

---

## 6. 常见问题

**Q：为什么生成后还有很多 `GENERATED-MOCK`？**
这些是故意留下的待填标记。生成器负责框架胶水和可运行 mock，真实 SDK 调用仍需要你按机器人实际接口补齐。

**Q：`can_port` 应该在哪里改？**
改 `configs/<name>/default.yaml`。`lowlevel.py` 里的默认值只是离线/mock 兜底，避免空配置时程序直接崩。

**Q：非 CAN 连接能用吗？**
可以生成，但目前 `serial` / `tcp` / `usb` / `ros` 先是空连接模板，后续会实现更完整模板。现在需要你按硬件 SDK 手动补 `config.py` / YAML / `lowlevel.py`。

**Q：什么时候选 `custom`？**
连接方式不属于 CAN、串口、TCP、USB、ROS，或者 SDK 初始化方式很特殊时选 `custom`。它会生成最空模板，由你完全填充。

**Q：我已经生成过了，重跑会覆盖吗？**
不加 `--force` 时，同名适配器会进入续跑阶段 B，不会重写已有文件。加 `--force` 才会覆盖。

**Q：为什么 mock 阶段能通过 smoke？**
生成器会写入内存态 mock：位置存在 `_pose` 里，`connect()` 只标记已连接。这让框架结构先跑通，等你逐个函数替换成真实 SDK。

**Q：补完一个函数后为什么要删 `GENERATED-MOCK`？**
向导靠这行标记判断“还有哪里没填”。不删的话，它会继续把该函数列为待办。

---

## 7. 常用参数速查

| 参数 | 说明 |
|------|------|
| `--name <name>` | 适配器名字，小写字母/数字/下划线，且以字母开头 |
| `--dof {4,6}` | 自由度，默认 `6` |
| `--joint` | 生成关节空间运动接口 |
| `--end-effector {none,parallel,suction}` | 末端执行器，默认 `none` |
| `--camera` | 生成相机取图接口 |
| `--detection` | 生成目标检测和像素反投影接口，自动蕴含 `--camera` |
| `--tool {straight_down,tilted}` | 工具几何，默认 `straight_down` |
| `--connection {can,serial,tcp,usb,ros,custom}` | 硬件连接方式，默认 `can` |
| `--non-interactive` | 不进入问答，完全使用命令行参数 |
| `--force` | 目标已存在时覆盖重写 |

完整参数见：

```bash
PYTHONPATH=$PWD conda run -n jiuwensymbiosis python -m scripts.new_adapter.main --help
```
