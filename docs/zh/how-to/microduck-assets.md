# Microduck 本地仿真与 Agent 控制

资产来自 Pollen Robotics 的 [microduck_rl 官方仓库](https://github.com/pollen-robotics/microduck_rl)。
本次验证版本：`0daab160d3fb1c2cbaba16a4a19052f36709dd9e`。
使用原始 MJCF、STL 和官方 CPU `PolicyInference`，不安装训练依赖。
行走与站立使用官方 [velstand 策略](https://huggingface.co/pollen-robotics/microduck-policies)，
权重版本固定为 `283771bcb3690a0a94a288c108365d9b630d21e4`。

在独立的 Python 3.12 仿真环境安装：

```bash
pip install -e ".[genesis-macos,microduck,gui]"
```

Microduck 使用 CPU MuJoCo + ONNX Runtime + BAM XL330 电机模型，
复用现有仿真进程、WebRTC、相机采样、控制权租约和 Agent 动作接口。
不需要 CUDA、训练、Rust 守护进程或第二套工作台。

下载模型目录（无需 CUDA）：

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/pollen-robotics/microduck_rl.git outputs/vendor/microduck_rl
git -C outputs/vendor/microduck_rl sparse-checkout set src/mjlab_microduck/robot/microduck
git -C outputs/vendor/microduck_rl sparse-checkout add scripts
mkdir -p outputs/vendor/microduck_rl/policies
curl -L --fail https://huggingface.co/pollen-robotics/microduck-policies/resolve/283771bcb3690a0a94a288c108365d9b630d21e4/velstand.onnx -o outputs/vendor/microduck_rl/policies/velstand.onnx
curl -L --fail https://huggingface.co/pollen-robotics/microduck-policies/resolve/283771bcb3690a0a94a288c108365d9b630d21e4/manifest.json -o outputs/vendor/microduck_rl/policies/manifest.json
python examples/microduck_sim.py
```

注册脚本保留已有资产根和索引位置，幂等地添加 Microduck。
等价的本机 `configs/simulators/assets.yaml` 根配置如下（`path` 使用绝对路径）：

```yaml
- id: microduck
  path: /你的项目路径/outputs/vendor/microduck_rl
  scanner: microduck
```

使用项目的仿真 Python 环境重建、查询索引：

```bash
python -m jiuwensymbiosis.simulators.assets build
python -m jiuwensymbiosis.simulators.assets search Microduck
python -m jiuwensymbiosis.simulators.assets show library://microduck/src/mjlab_microduck/robot/microduck/robot_walk
```

仅下载模型时仍可索引；安装完整策略后才会进入场景选择器。
机器人身份沿用 `robot_walk`，运行场景使用官方 `scene.xml` / `robot_groundcontact.xml`。

启动工作台：

```bash
python -m jiuwensymbiosis_gui --gui workbench --port 8771 --no-browser
```

访问 `http://127.0.0.1:8771` 的仿真页，在场景下拉框选择 `microduck · Microduck`。
界面自动选择 `Microduck 双足步态物理`；启动后台并连接视频后，可用下方控制条移动。
观测模式选择「相机观察 · RGB-only」。新版 RGB-D 感知和真值导航尚未接入 Microduck，
选用这些模式会明确提示不可用。关闭工作台可保留常驻世界；完全停止请点击「释放后台资源」。
复位会先确认停车，再重建策略世界，使旧场景控制绑定失效。
在模型页配置支持图片和工具调用的模型，打开“允许 Agent 控制仿真”，例如输入：

> 先看头部相机，向前走 20 厘米，向左转 0.5 弧度，然后停车并重新观察。

复用 `capture_camera(head_cam)`、`navigate_relative(dx_m, dy_m, dyaw_rad)` 和
`rotate_base(dyaw_rad)`；前后、横移、转向按机身前/左/上坐标执行。
没有接入 Microduck 的其他技能，工具列表不包含 Jumper 的跳跃或夹爪。

适配范围：

- 原始 61 维观测、14 维动作、50 Hz 推理、200 Hz 物理和动作增益保持官方定义。
  BAM 使用包含 CPU 自由度摩擦索引修复的 1.0.3，保留电压、力矩与摩擦物理。
  电源压降使用该版本的电阻接口，设为 0.1 Ω，标称电压 7.4 V。
- 发布策略在 CPU 仿真有明显启动死区。宿主将非零平移命令映射为模长 0.4 的步态指令，
  非零转向映射为 ±1.5；距离和转角通过物理反馈闭环执行。
  这是仿真导航的步态启动配置，不保证策略指令值等于实际速度。
  当前到达后的稳定范围为 4 cm / 0.1 rad，连续实测停稳 1 秒才确认成功。
- 官方 CAD 导出的 `head_camera` 指向镜头内部。运行时仅修正渲染相机朝向和向外 5 mm 的位置；
  不改源文件、刚体、惯量或碰撞。640×480 内参来自渲染 FOV，不代表实机标定。
- 倾倒、非有限状态或未确认停车会报错，不用传送位置代替动作；倾倒后需重启场景。

Mac 验收覆盖真实 Agent 工具调度、前进/后退/横移/转向、相机、H264/WebRTC、
取消后停车和切换分辨率保留场景。模型服务使用确定性的本地 HTTP 测试端点，
在线模型的视觉规划质量仍取决于所配置的模型。
证据在 `outputs/microduck/integration-final/acceptance.json` 及同目录的图片和日志。

模型归 Pollen Robotics 所有。STL 按
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) 授权，
其余文件按 Apache-2.0 授权；原始 `LICENSE` 和 `LICENSE-MODELS.md` 随下载保留。
本次未修改上游文件。
