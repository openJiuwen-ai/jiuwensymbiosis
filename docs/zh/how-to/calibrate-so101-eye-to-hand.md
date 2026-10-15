# SO-101 固定相机手眼标定使用指南

[English](../../en/how-to/calibrate-so101-eye-to-hand.md) | 中文

本指南只适用于以下组合：

- 机械臂：SO-101，5 个臂关节；
- 相机：固定在桌面、支架或机器人基座附近，不随机械臂运动；
- 标定方式：eye-to-hand；
- 标定板：固定在夹爪上，随机械臂一起运动；
- 求解结果：`T_base_cam`，即相机坐标系到机器人 base 坐标系的固定变换。

整个流程分为三个阶段：

| 阶段 | CLI 模式 | 作用 | 是否需要硬件 |
|---|---|---|---|
| 1 | `--collect-poses` | 手动示教一组关节 waypoint | 是 |
| 2 | `--auto` | 自动回放 waypoint、采集图像、求解并尝试发布正式标定 | 是 |
| 3 | `--replay` | 使用 station 归档离线重新求解 | 否 |

阶段 1 和阶段 2 是首次标定的必需步骤；阶段 3 用于换算法、调阈值或复查结果。

## 先看看图形界面

如果这是你第一次做标定，建议改用图形界面的标定向导，而不是本文的命令行：

```bash
python -m jiuwensymbiosis_gui --gui workbench    # 打开后进入「工具 → 手眼标定」
```

向导把本文的全部步骤串成四步，并补上了命令行没有的三件事：**没有标定板时可直接生成
可打印 PDF**（含校验标尺，防止打印缩放导致的静默误差）、**示教时实时显示标定板识别到
了多少角点**（命令行只能盲按回车，废姿态要等采集完才暴露）、以及**质量门禁失败时给出
对应的改进动作**。它与本文调用的是同一套标定代码，产出的标定文件也完全相同。

本文适合需要精细控制参数、批量脚本化或离线重解的场景。

---

## 1. 安全与硬件准备

开始前确认：

1. SO-101 已完成 LeRobot 电机标定，配置中的 `robot_id` 与 LeRobot 标定时使用的 ID 一致。
LeRobot 电机标定参考官网视频：https://huggingface.co/docs/lerobot/so101#calibrate
2. 用 `lerobot-find-port` 确认串口路径。
3. 固定相机支架牢固，整个采集期间相机不得移动、转动或重新对焦。
4. 标定板刚性固定在夹爪上，不能在采集过程中松动。
5. 相机视野覆盖 SO-101 的主要工作区域，USB 连接稳定。
6. 急停或断电手段触手可及；自动采集前清空机械臂周围障碍物。
7. 手动示教会释放 5 个臂关节的力矩。SO-101 的肘关节受重力影响明显，必须用另一只手托住机械臂。

> 标定配置会使用较宽的关节限位和放宽的工作空间限制，只适合受控标定。不要把这些安全参数直接复制到日常运行配置。

---

## 2. 安装依赖

SO-101 适配器要求 Python 3.12。建议在项目环境中一次安装标定和 SO-101 依赖：

```bash
python -m pip install -e ".[gui,calib,so101]"
```

这里的 `gui` extra 安装现有工作台的 NiceGUI 依赖；只运行下方无界面 CLI 工作流时，可改装 `.[calib,so101]`。

如需运行标定模块测试：

```bash
pytest tests/unit_tests/calibration/
```

---

## 3. 准备 ChArUco 标定板

推荐使用 5×7 的 ChArUco 板。没有现成标定板时，可调用共享板生成器：

```bash
python -c 'from scripts.calibrate.handeye_board import BoardSpec, generate_board_image; generate_board_image(BoardSpec("charuco", 5, 7, 20.86, 15.2), "tmp/so101_charuco.png")'
```

打印和安装要求：

1. 按 100% 比例打印，关闭“适应页面”或自动缩放。
2. 打印后用尺子测量实际方格边长和 marker 边长。
3. 将实测值用于后续所有命令的 `--square-size-mm` 和 `--marker-size-mm`。
4. 把纸张平整贴在硬质平板上，避免弯曲、翘边和反光。
5. 将硬板刚性固定在夹爪上，尽量避免夹爪遮挡 marker 或角点。

以下示例统一假设实测参数为：

```text
board=charuco
squares-x=5
squares-y=7
square-size-mm=20.86
marker-size-mm=15.2
```

如果实际尺寸不同，必须同步修改阶段 2 的采集命令。方格尺寸错误会直接造成平移尺度错误。

---

## 4. 查询 SO-101 串口和相机序列号

查询机械臂串口：
这一步按照lerobot提示需要拔掉电脑上的机械臂连接线

```bash
lerobot-find-port
```

查询 RealSense 序列号：

```bash
rs-enumerate-devices | grep "Serial Number"
```

也可以使用 Python：

```bash
python -c "import pyrealsense2 as rs; c=rs.context(); print([d.get_info(rs.camera_info.serial_number) for d in c.devices])"
```

---

## 5. 配置标定环境

以项目内的 `scripts/calibrate/so101_calibrate.yaml` 为模板。建议复制成本机配置，避免把设备路径和序列号提交到仓库：

```bash
cp scripts/calibrate/so101_calibrate.yaml scripts/calibrate/so101_calibrate.local.yaml
```

至少检查以下字段：

```yaml
env:
  cfg:
    low_level:
      name: "so101"
      port: "/dev/your_port"
      robot_id: "so101_left"
      calibration_dir: null

      # 标定时以连接瞬间的实际关节角作为 home。
      home_use_init_pose: true
      disable_torque_on_disconnect: false

      joint_limits:
        shoulder_pan: [-180.0, 180.0]
        shoulder_lift: [-180.0, 180.0]
        elbow_flex: [-180.0, 180.0]
        wrist_flex: [-180.0, 180.0]
        wrist_roll: [-180.0, 180.0]

      # 固定相机的挂载方式。SO-101 标定只允许 eye_to_hand。
      camera_mount: "eye_to_hand"
      camera_serial: "<your-camera-serial>"
      camera_resolution: [640, 480]
      camera_fps: 30

calibration:
  adapter_module: "jiuwensymbiosis.adapters.so101"
  trajectory:
    space: joint
  output: "tmp/so101_eye_to_hand.json"

  observability:
    min_relative_rotation_deg: 5.0
    min_axis_separation_deg: 15.0
    min_max_rotation_deg: 20.0
    min_translation_baseline_mm: 30.0
    duplicate_rotation_deg: 2.0
    duplicate_translation_mm: 5.0

  # SO-101 舵机在重力负载下可能存在约 3.5° 稳态偏差。
  capture_gate:
    reach_rotation_deg: 4.0
    reach_translation_mm: 0.5
    exposure_rotation_deg: 1.0
    exposure_translation_mm: 0.5
```

不要把 `camera_mount` 放到顶层 `calibration:` 中；它的唯一权威来源是 `env.cfg.low_level.camera_mount`。

后续命令统一使用：

```bash
export CALIB_CONFIG=scripts/calibrate/so101_calibrate.local.yaml
```

如果不想使用 shell 变量，可把命令中的 `$CALIB_CONFIG` 直接替换为配置文件路径。

---

## 6. 阶段 1：手动示教 waypoint

注意：运行下列指令前需要先托住机械臂，防止突然松开力矩机械臂掉落，运行命令后机械臂会松开力矩，建议先阅读下方交互过程和示教姿态要求再开始操作
运行：

```bash
jiuwensymbiosis-calibrate-eye-to-hand \
  --config "$CALIB_CONFIG" \
  --collect-poses tmp/so101_wp.npz
```

`--collect-poses` 只记录关节 waypoint，不拍摄标定图像，因此此阶段的板参数不参与计算。

交互过程：

1. CLI 连接 SO-101。
2. 程序关闭 5 个臂关节的力矩，夹爪力矩保持开启。
3. 托住肘部，将机械臂缓慢移动到第一个姿态。
4. 确认标定板完整出现在固定相机视野中，然后按 Enter 记录。
5. 重复移动和记录，建议准备 12～20 个 waypoint。
6. 完成后输入 `q` 并按 Enter。
7. 程序先同步当前关节目标，再恢复力矩并写出 `tmp/so101_wp.npz`。

示教姿态要求：
- 注意机械臂运动过程中标定板不要碰到物体或者桌面，否则导致标定板移动或者变形会影响标定结果
- 至少覆盖两条不同的旋转轴，不要只绕单一轴转动。
- 最大相对旋转建议达到 20° 以上。
- 标定板在相机坐标系中的平移范围建议超过 30 mm。
- 覆盖视野中央、左右、前后和不同距离，避免所有姿态集中在小区域。
- 每个姿态都要保证板完整、清晰、不过曝，并尽量减少夹爪遮挡。
- 不要重复记录几乎相同的姿态。

如果力矩恢复失败，程序会报 `ManualGuidanceRecoveryError`。此时继续托住机械臂，停止后续自动流程，检查电机总线并人工恢复安全姿态。

---

## 7. 阶段 2：预演与自动采集

### 7.1 先执行 dry-run

dry-run 会检查 archive 类型、adapter、mount、关节顺序、单位、周期性和轨迹插值，但不会运动或拍照：

```bash
jiuwensymbiosis-calibrate-eye-to-hand \
  --config "$CALIB_CONFIG" \
  --auto tmp/so101_wp.npz \
  --n-stations 20 \
  --dry-run \
  --out tmp/so101_eye_to_hand.json
```

只有 dry-run 成功后才进入真机自动采集。

### 7.2 自动采集、求解和发布

确认急停可达、工作区无障碍物后运行：

```bash
jiuwensymbiosis-calibrate-eye-to-hand \
  --config "$CALIB_CONFIG" \
  --board charuco \
  --squares-x 5 --squares-y 7 \
  --square-size-mm 20.86 --marker-size-mm 15.2 \
  --auto tmp/so101_wp.npz \
  --n-stations 20 \
  --confirm-estop \
  --cross-check \
  --min-corners 16 \
  --out tmp/so101_eye_to_hand.json
```

自动流程会依次执行：

1. 校验 waypoint archive 的 adapter、mount 和关节元数据。
2. 按默认不超过 5° 的关节步长插值轨迹。
3. 在第一次运动前抓取一帧，验证固定相机、内参和图像数据可用。
4. 移动至采样位置并等待稳定。
5. 检查实际姿态与目标姿态的偏差。
6. 拍摄图像并检测 ChArUco 板。
7. 检查曝光前后机械臂是否漂移。
8. 记录通过 gate 的 station，并写出 station 归档。
9. 求解 `T_base_cam`，执行观测性和刚性一致性门禁。
10. 通过 SO-101 运行时 loader 做 reload smoke；全部通过后才发布正式标定。

`--n-stations` 是目标采样数和尝试上限，不保证最终接受同样数量的 station。板检测失败、到位偏差过大或曝光期间漂移的站点会被跳过。

### 7.3 输出文件和退出码

成功时产生：

- `tmp/so101_eye_to_hand.json`：正式 schema-2 标定，顶层包含 `T_base_cam`；
- `tmp/so101_eye_to_hand.stations.npz`：自描述 station 归档，可用于离线重解。

如果求解完成但质量门禁或 reload smoke 未通过，会产生：

- `tmp/so101_eye_to_hand.candidate.json`：仅供 REVIEW，矩阵位于 `candidate.T_base_cam`，运行时 loader 不会加载它。

如果有效 station 少于求解下限，程序可能只返回 REVIEW 状态而不生成 candidate 文件；此时应查看日志和 station 归档，重新示教或改善成像后再采集。

退出码：

| 退出码 | 含义 |
|---|---|
| `0` | 正式发布成功，或 dry-run 成功 |
| `1` | 执行错误 |
| `2` | preflight 契约检查失败 |
| `3` | 只得到 REVIEW/candidate，未发布正式标定 |

只有退出码为 `0` 且输出 JSON 顶层存在 `T_base_cam` 时，才可作为运行时标定使用。

---

## 8. 阶段 3：离线重解（可选）

### 8.1 不带配置：只生成 candidate

```bash
jiuwensymbiosis-calibrate-eye-to-hand \
  --replay tmp/so101_eye_to_hand.stations.npz \
  --method HORAUD \
  --cross-check \
  --out tmp/so101_eye_to_hand_horaud.json
```

没有 `--config` 时不会发布正式标定，只会生成 `tmp/so101_eye_to_hand_horaud.candidate.json`，正常退出码为 `3`。

### 8.2 带配置：允许正式发布

```bash
jiuwensymbiosis-calibrate-eye-to-hand \
  --replay tmp/so101_eye_to_hand.stations.npz \
  --config "$CALIB_CONFIG" \
  --method HORAUD \
  --cross-check \
  --out tmp/so101_eye_to_hand_horaud.json
```

该模式无需连接机械臂和相机，但会使用配置确认 adapter 与 `camera_mount`，并通过 SO-101 loader 做 reload smoke。只有质量门禁和 reload smoke 均通过时，才会写出正式 JSON。

常用求解方法包括 `PARK`、`TSAI`、`HORAUD`、`ANDREFF` 和 `DANIILIDIS`。建议以 `PARK` 为主，并用 `--cross-check` 检查不同方法之间是否存在明显分歧；不要仅因为某种方法能通过门禁就忽略采样质量。

---

## 9. 应用到 SO-101 运行配置

先备份正式标定文件到稳定位置，例如：

```bash
mkdir -p configs/so101/calibration
cp tmp/so101_eye_to_hand.json configs/so101/calibration/so101_eye_to_hand.json
```

然后在 SO-101 运行配置的 `env.cfg.low_level` 中设置：

```yaml
camera_mount: "eye_to_hand"
camera_serial: "<与标定时相同的固定相机序列号>"
calib_path: "configs/so101/calibration/so101_eye_to_hand.json"
```

使用前检查正式 JSON：

```bash
python -c 'import json; p=json.load(open("configs/so101/calibration/so101_eye_to_hand.json")); assert p.get("schema_version")==2 and "T_base_cam" in p; print("calibration artifact OK")'
```

标定文件与物理安装是一一对应的。以下任一情况发生后都必须重新标定：

- 固定相机或支架被移动；
- SO-101 基座位置或朝向改变；
- 标定板在夹爪中的安装发生松动或变化后重新采集；
- 相机更换、内参发生变化或分辨率模式改变。

---

## 10. 常见故障排查

| 现象 | 常见原因 | 处理建议 |
|---|---|---|
| `ManualGuidanceRecoveryError` | 示教结束后力矩恢复失败 | 托住肘部，停止自动流程，检查电机总线并人工恢复安全姿态 |
| `camera preflight failed` | 序列号错误、USB 不稳定或内参不可读 | 检查 `camera_serial`、USB 连接、RealSense 驱动和相机占用情况 |
| `board not detected` | 板出视野、角点被遮挡、反光、模糊或尺寸参数错误 | 重新示教姿态，改善光照，降低运动速度，核对板规格 |
| `reach gate failed` | 舵机稳态偏差过大或 waypoint 不可达 | 检查实际姿态和机械负载；不要盲目继续放宽 `reach_rotation_deg` |
| `exposure drift` | 拍摄期间舵机仍在漂移 | 检查机械负载、供电和关节稳定性，改善 settle 状态 |
| `observability_flange_axes` | 姿态只围绕单一轴变化 | 重新示教，增加至少另一条旋转轴的变化 |
| `observability_camera_axes` | 标定板相对固定相机的运动退化 | 增加板的倾斜、旋转和视野内平移，确认板与夹爪刚性连接 |
| `observability_*_trans` | 平移覆盖太小 | 扩大工作区内的前后、左右或远近变化 |
| `observability_duplicates` | waypoint 过于相似 | 删除重复姿态并重新采集 |
| 只生成 candidate | 质量门禁或 reload smoke 失败 | 查看 candidate 的 `reasons`，修复采样问题后重新采集或重解 |
| 标定后投影存在固定偏差 | 板尺寸错误、相机/基座移动或坐标安装不一致 | 重新测量板尺寸，确认硬件未移动，然后重新标定 |

调试时可在命令末尾增加 `--debug` 获取更详细日志。不要直接手工把 candidate 中的矩阵复制到正式 schema-2 文件，以绕过质量门禁和 reload smoke。

---

## 11. 命令速查

```bash
# 1. 示教 waypoint
jiuwensymbiosis-calibrate-eye-to-hand \
  --config "$CALIB_CONFIG" \
  --collect-poses tmp/so101_wp.npz

# 2. 无运动预检
jiuwensymbiosis-calibrate-eye-to-hand \
  --config "$CALIB_CONFIG" \
  --auto tmp/so101_wp.npz --n-stations 20 --dry-run \
  --out tmp/so101_eye_to_hand.json

# 3. 真机采集、求解、发布
jiuwensymbiosis-calibrate-eye-to-hand \
  --config "$CALIB_CONFIG" \
  --board charuco --squares-x 5 --squares-y 7 \
  --square-size-mm 20.86 --marker-size-mm 15.2 \
  --auto tmp/so101_wp.npz --n-stations 20 \
  --confirm-estop --cross-check \
  --out tmp/so101_eye_to_hand.json

# 4. 离线重解并重新执行发布门禁
jiuwensymbiosis-calibrate-eye-to-hand \
  --replay tmp/so101_eye_to_hand.stations.npz \
  --config "$CALIB_CONFIG" \
  --method HORAUD --cross-check \
  --out tmp/so101_eye_to_hand_horaud.json
```
