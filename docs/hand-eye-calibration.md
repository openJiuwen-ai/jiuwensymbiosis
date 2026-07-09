# 手眼标定指南（Hand-Eye Calibration）

让装在机械臂腕部的相机「知道」自己相对机械臂的位置——标定后，相机看到的物体能被准确换算成机械臂坐标，视觉抓取才会准。

标定程序：[scripts/calibrate/calibrate_hand_eye.py](../scripts/calibrate/calibrate_hand_eye.py)，产出标定文件 `configs/piper/piper_calib.json`。

**什么时候需要做标定**

- 第一次装好相机/机械臂；
- 相机、镜头或相机支架被挪动过；
- 视觉抓取出现稳定的偏移（比如总是偏左、偏低同样的量）。

---

## 目录

1. [准备工作](#1-准备工作)
2. [开始标定（三步）](#2-开始标定三步)
3. [判断标定好不好](#3-判断标定好不好)
4. [复验](#4-复验)
5. [常见问题](#5-常见问题)
6. [常用参数速查](#6-常用参数速查)

---

## 1. 准备工作

### 1.1 安装依赖

```bash
pip install -e ".[calib,piper]"
```

`[calib]` 提供 OpenCV 和 RealSense 支持，`[piper]` 提供 piper 机械臂驱动（如果你使用的机械臂并非 piper，需自行安装相对应的依赖）。

### 1.2 准备标定板

需要一块印有 **ChArUco**（推荐）或**棋盘格**图案的平板。

**没有现成的板？让程序生成一张可打印图：**

```bash
python scripts/calibrate/calibrate_hand_eye.py --generate-board board.png \
    --board charuco --squares-x 5 --squares-y 7 --square-size-mm 30 --marker-size-mm 22
```

**自制标定板的建议：**

1. 用 A4 纸 **100% 原始比例**打印，关闭「适应页面/缩放」。
2. 打印后**用尺子分别测量大黑色方格和小黑色方格的实际边长**，把真实毫米数分别填给后面的 `--square-size-mm`和`--marker-size-mm`。**打印缩放是头号误差来源**。
3. **平整裱在硬板上**（KT 板/亚克力/铝板），不能弯、不能翘。
4. 表面**不要反光**（哑光纸最好）。
5. 记住你的板参数（方格数、方格边长、marker 边长）——**生成和标定必须用同一组参数**。

对于大黑色方格，可以沿一行连续测量若干个大格子的总长再除以数目，比单格量得更准。

### 1.3 配置机器人

编辑 [scripts/calibrate/calibrate.yaml](../scripts/calibrate/calibrate.yaml)，填入你的 RealSense 序列号：

```yaml
      camera_serial: "你的相机序列号"
```

或临时用环境变量：`export CAMERA_SERIAL=你的序列号`。

> 这份配置专为标定准备，**不要**用 `piper.yaml`——那会指向正在生成的标定文件。

---

## 2. 开始标定（三步）

把机械臂上电、CAN 接好、标定板摆在相机视野里，运行：

```bash
python scripts/calibrate/calibrate_hand_eye.py --config scripts/calibrate/calibrate.yaml \
    --board charuco --squares-x 5 --squares-y 7 --square-size-mm 30 --marker-size-mm 22
```

请务必记得 --square-size-mm 以及 --marker-size-mm 改为你实际测量出来的那个值。

程序会**全程中文向导**带你走完。

### 第 1 步：自检与确认

程序先检查相机、机械臂是否就绪，并打印一张配置确认卡。看一眼板参数、输出文件对不对，回车继续。

### 第 2 步：采集多个角度（最关键）

手动模式下，你来摆姿势、程序来拍：

| 按键 | 作用 |
|------|------|
| 回车 | 采集当前这一帧 |
| `s`  | 采够了，开始求解 |
| `u`  | 撤销最近一帧（拍错了） |
| `q`  | 放弃退出 |

**采集要领——必须让手腕的「转动」有变化，光平移没用：**

- 每拍一帧，就**换一个姿态**：手腕往不同方向倾斜 ±20~30°、绕轴转一转、远近也变一变。
- 始终让**整块板**清楚地出现在画面里。
- 建议采 **10~15 帧**。程序顶部会实时显示「已采几帧、旋转跨度多少度」，跨度太小它会提醒你多转转。
- 每帧拍完会告诉你「✓ 采纳」还是「✗ 未采纳」及原因。

> 想让机械臂自己转着拍？加 `--auto`（会驱动机械臂运动，**确保 E-stop 在手边**；可先用 `--auto-dry-run` 看它打算去哪些位姿）。

### 第 3 步：求解并写文件

按 `s` 后程序自动求解、打印精度报告（见下一节），确认覆盖后写入 `configs/piper/piper_calib.json`（旧文件自动备份成 `.bak`）。

---

## 3. 判断标定好不好

报告里每项都带 ✅/⚠️/❌。三个核心指标：

| 指标 | 含义 | 达标（✅） | 要重做（❌） |
|------|------|-----------|-------------|
| **重投影 RMS** | 板检测得准不准 | < 1 px | > 2 px |
| **手眼一致性**（旋转 / 平移） | 各帧解出的结果一不一致 | < 0.5° / < 2~3 mm | 明显更大 |
| **板原点一致性** | 机械臂和相机对同一点的认知误差 | std < 2~3 mm | > 3 mm |

**不达标怎么办？** 绝大多数是这两个原因：

- **旋转角度不够**：回去重采，手腕多换姿态（别只平移）。
- **板尺寸填错了**：用尺子重量方格边长，确认 `--square-size-mm` 是真实值。

---

## 4. 复验

强烈建议真机验一下。在标定命令后追加 `--verify-touch`，标定完成后会立即复验——它让**指尖悬停在板中心上方约 30mm（默认不接触）**，你肉眼看指尖是否对准板中心：xy 对得上就说明标定良好。

**末端是裸法兰（没装工具）：**

```bash
python scripts/calibrate/calibrate_hand_eye.py --config scripts/calibrate/calibrate.yaml \
    --board charuco --squares-x 5 --squares-y 7 --square-size-mm 30 --marker-size-mm 22 \
    --verify-touch
```
> 配置里 `tool_offset_mm=0` 时，程序会**强警告并要你确认**末端确实无工具——因为它会按「法兰=指尖」算悬停高度。

**末端装了工具/夹爪：** 必须用 `--verify-tool-offset-mm` 传入真实的法兰→指尖长度（mm），否则工具会比预期更低、可能撞向标定板：

```bash
... --verify-touch --verify-tool-offset-mm 95   # 例：夹爪长 95mm
```

- 悬停余量可用 `--verify-hover-mm` 调整（默认 30mm，越大越保守）。
- 不想动机械臂、只看数字：把 `--verify-touch` 换成 `--verify`，它只打印换算出的机械臂坐标供你目测（零运动）。

也可以直接跑抓取演示看整体效果：

```bash
piper-pick-demo --config configs/piper/piper.yaml
```

---

## 5. 常见问题

**Q：一直提示「未检测到标定板」？**
板要完整入镜、别太斜太远、避免反光；确认命令里的板参数和你手上的板一致。

**Q：提示「相机内参不可用」？**
非 RealSense 相机没有出厂内参，加 `--calibrate-intrinsics`（用采集到的图一起标定内参），或用 `--intrinsics fx fy ppx ppy` 手动指定。

**Q：提示「有效视图不足」？**
至少要 3 帧成功，推荐 10~15 帧。多换姿态多拍几张。

**Q：用棋盘格而不是 ChArUco？**
把 `--board charuco --marker-size-mm ...` 换成 `--board chessboard`，其余一样。ChArUco 对遮挡/模糊更鲁棒，优先用它。

**Q：没有硬件，想先熟悉一下？**
跑 `python scripts/calibrate/calibrate_hand_eye.py --selftest`，用合成数据离线验证程序本身（无需机械臂和相机）。

---

## 6. 常用参数速查

| 参数 | 说明 |
|------|------|
| `--config <yaml>` | 机器人配置（如 `scripts/calibrate/calibrate.yaml`） |
| `--board {charuco,chessboard}` | 标定板类型（默认 charuco） |
| `--squares-x / --squares-y` | 板的方格行列数 |
| `--square-size-mm` | 方格实测边长（mm，务必准确） |
| `--marker-size-mm` | ChArUco 的 marker 边长（mm） |
| `--auto` | 自动驱动机械臂采集（注意安全） |
| `--calibrate-intrinsics` | 顺便标定相机内参（非 RealSense 用） |
| `--out <path>` | 输出标定文件（默认 `configs/piper/piper_calib.json`） |
| `--verify` / `--verify-touch` | 标定后真机复验（仅打印坐标 / 指尖悬停目视） |
| `--verify-tool-offset-mm` | verify-touch 用的法兰→指尖长度（mm，装了工具必填） |
| `--verify-hover-mm` | verify-touch 指尖悬停余量（mm，默认 30，不接触） |
| `--generate-board <png>` | 生成可打印标定板图 |
| `--selftest` | 离线自检（无需硬件） |

完整参数见 `python scripts/calibrate/calibrate_hand_eye.py --help`。
