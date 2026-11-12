# 在 Mac 上用 Symbiosis Agent 控制 Jumper

支持 Apple Silicon（本机验收为 M4、16 GB）。在 Symbiosis 仿真工作台选择
Jumper 后，使用官方 MJCF、原生 FSM 和 ONNX 步态策略运行 MuJoCo 接触物理。
工作台、Agent、相对动作契约、视频和取消机制与已有仿真共用。

与主仓 `53fdba7` 合并后，请选择「相机观察 · RGB-only」观测模式。
Jumper 尚未实现新版 RGB-D 感知或真值导航；跳跃和夹爪仍按能力门控开放。
新版工作台关闭后常驻世界可继续运行；需要完全停止时点击「释放后台资源」。
「复位当前世界」会先确认停车，再重建官方策略世界，并使旧场景控制绑定失效。

## 安装与运行

已有 `.venv-genesis` 时只需加装 `jumper`：

```bash
uv pip install --python .venv-genesis/bin/python -e '.[jumper]'
```

新机器按 [Genesis 安装教程](../tutorial/03-install-genesis.md) 建立独立环境，
将安装项设为 `.[genesis-macos,jumper,gui]`。不需要训练环境、CUDA、Docker
或 Rust 编译。消费的是官方 bundle 自带的 macOS universal2 扩展。

在仓库根目录下载并登记资产（已存在的下载无需再 clone）：

```bash
git clone --depth 1 https://github.com/KingKongRobotics/jumper.git outputs/vendor/jumper
# 本次验收版本：198686eac0d188815d0943e9fee50de415edc80a
# 需要复现时，在该下载目录 checkout 此提交。
.venv-genesis/bin/python examples/jumper_sim.py --source outputs/vendor/jumper
.venv-genesis/bin/python -m jiuwensymbiosis_gui --gui workbench --port 8771 --no-browser
```

登记会追加 `configs/simulators/assets.yaml` 中的 Jumper 根并重建索引，保留
其他资产根；原模型和上游仓库不被修改。大资产位于被 Git 忽略的 `outputs/`。
来源与许可证见 [官方 Jumper 仓库](https://github.com/KingKongRobotics/jumper)
及下载目录内的 `LICENSE`、`NOTICE`；不将这些资产复制进 Symbiosis 源码。

打开 `http://127.0.0.1:8771` 的仿真页：

1. 场景选 `jumper · Jumper`，会自动使用「Jumper 六足步态物理」。
2. 启动后台，等待就绪，连接视频。推荐观察视角 540p / 30 fps。
3. 相机视角可切换「观察视角」和「头部相机」。后者是 MJCF 的 `onboard`，
   输出 640×480，保留 95.7° 垂直视场及机身安装位姿。
4. 在「模型」页设置支持图片与工具调用的模型服务，应用到工作台。
5. 打开「允许 Agent 控制仿真」，例如输入：
   「先看相机，然后前进 20 厘米，左移 15 厘米，左转 0.5 弧度，最后再看相机。」

Agent 使用共享 `navigate_relative(dx_m, dy_m, dyaw_rad)`、
`rotate_base(dyaw_rad)` 和已有的相机工具。+X 前进、+Y 左移、正偏航左转；
组合动作先转再平移。手动按钮提供短时速度命令，停止按钮也可取消 Agent。
每次只允许一个控制会话，断开视频后手动命令失效，Agent 仍有独立动作租约。

## 跳跃和夹爪

Jumper 支持共享跳跃动作，并复用主仓的夹爪动作：

```python
jump()                                    # 一次官方固定轨迹跳跃，等待落地和停稳
close_gripper(side="left")                 # 左夹爪闭合，右侧用 side="right"
open_gripper(side="left")                  # 张开指定侧夹爪
```

可直接对 Agent 说：「先看相机，原地跳一次，落地后左夹爪闭合再张开，
然后右夹爪闭合再张开，最后前进 15 厘米并看相机。」
需要先在“模型”页应用支持图片与工具的模型，并开启 Agent 控制。

跳跃复用官方 `jump.onnx`、录制轨迹及原生 FSM；不提供任意高度/距离参数。
成功需确认实际离地至少 20 ms、恢复地面支撑，并连续停稳 1 s。
取消已触发的跳跃会等本次跳跃结束落地，不启动下一次。

夹爪复用 `five_foot` 策略和官方原生 hook：左、右模式一次只抬起一侧，
其他五足支撑；右侧由官方镜像实现。闭合命令完成后保持夹爪状态，停止
或会话退出不会自动松爪。**换侧、行走、跳跃前须显式张开当前夹爪**，
之后由官方状态机收臂再执行动作；夹爪模式下的手动移动按钮会拒绝命令。
当前验收空爪开合与支撑平衡，未验收抓持物体或持物搬运。返回的
`grasp_verified=false` 表示开合成功不等于抓住物体，不写入持物状态。

夹爪与行走的输入方式只在稳定的 locomotion 交接时切换；无需修改上游
合约、重新训练、改写关节轨迹或重编原生扩展。物理与伺服热状态保持连续。

## 当前范围与验收

支持站立、前后移动、左右横移、转向、停止，以及观察/机载 RGB 视频。
初始场景有地面和红绿蓝路标，均参与接触物理。当前没有把 Jumper 转换到
Genesis，也没有把它放入 USD 房间；这是保留原步态物理的原生 MuJoCo 接入。
已开放跳跃与左右夹爪开合；舞蹈及姿态切换仍未开放。没有自动避障、视觉抓取或深度工具。

运动采用真实机身反馈闭环。步态停止后，位置误差需保持在 3 cm 内、
偏航误差在 0.08 rad 内，并连续稳定 1 s 才成功；这是仿真验收容差。
取消和超时先发送零速，再读回停车结果，不将命令接收当作动作完成。

加载时校验消费文件的 SHA-256，并回放官方 `reference.json`；状态详情的
`scene.policy` 包含控制器提交、bundle 摘要、原始 notes 和参考误差。
参考只覆盖 locomotion 的录制帧，不能据此推断其他策略或实机已验证。

可重复运行本机验收：

```bash
JIUWEN_JUMPER_INTEGRATION=1 \
JIUWEN_JUMPER_SOURCE="$PWD/outputs/vendor/jumper" \
JIUWEN_JUMPER_EVIDENCE="$PWD/outputs/jumper" \
.venv-genesis/bin/pytest tests/integration/test_jumper_macos.py -q
```

验收覆盖真实步态物理、共享 Agent 工具循环、相机图片注入、四个移动动作的
实测到达/停车、两次跳跃与双爪开合、取消跳跃后落地、H264/WebRTC、
重设画质时保留世界状态。
模型端是可控的本机 HTTP 测试服务，在线 LLM 对话需另配置模型服务。
证据位于 `outputs/jumper/acceptance.json`、相机截图和 `agent/` 的执行轨迹。
跳跃和夹爪的证据位于 `outputs/jumper/skills/acceptance.json`，包含私有的
实际最高机身高度、离地时间和爪关节反馈；这些真值不进入模型上下文。
