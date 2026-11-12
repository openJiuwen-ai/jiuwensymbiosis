# Microduck 本地仿真与 Agent 控制

资产来自 Pollen Robotics 的 [microduck_rl 官方仓库](https://github.com/pollen-robotics/microduck_rl)。
本次验证版本：`0daab160d3fb1c2cbaba16a4a19052f36709dd9e`。
使用原始 MJCF、STL 和官方 CPU `PolicyInference`，不安装训练依赖。
行走与站立使用官方 [velstand 策略](https://huggingface.co/pollen-robotics/microduck-policies)，
权重版本固定为 `283771bcb3690a0a94a288c108365d9b630d21e4`。

在独立的 Python 3.12 仿真环境安装：

```bash
uv pip install --python .venv-genesis/bin/python -e ".[genesis-macos,microduck,gui]"
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
.venv-genesis/bin/python examples/microduck_sim.py
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
.venv-genesis/bin/python -m jiuwensymbiosis.simulators.assets build
.venv-genesis/bin/python -m jiuwensymbiosis.simulators.assets search Microduck
.venv-genesis/bin/python -m jiuwensymbiosis.simulators.assets show library://microduck/src/mjlab_microduck/robot/microduck/robot_walk
```

仅下载模型时仍可索引；安装完整策略后才会进入场景选择器。
机器人身份沿用 `robot_walk`，运行场景使用官方 `scene.xml` / `robot_groundcontact.xml`。

启动工作台：

```bash
.venv-genesis/bin/python -m jiuwensymbiosis_gui --gui workbench --port 8771 --no-browser
```

访问 `http://127.0.0.1:8771` 的仿真页，在场景下拉框选择 `microduck · Microduck`。
界面自动选择 `Microduck 双足步态物理`；启动仿真并连接视频后，可用下方控制条移动。
观测模式选择「相机观察 · RGB-only（兼容）」。新版 RGB-D 感知和真值导航尚未接入 Microduck，
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
- 发布策略在 CPU 仿真有明显启动死区，运行时原样传递已校验的速度幅值，不隐式放大小指令。
  本体原生策略指令范围显式声明为平移不超过 0.4 m/s、转向不超过 1.5 rad/s；
  相对导航使用 0.4 m/s / 1.5 rad/s 的起步指令，再按实测距离和转角闭环停车。
  通用手动控制仍限制为 0.3 m/s / 0.6 rad/s，小指令可能不足以起步。
  策略指令不是实测速率；达不到目标时返回超时，不声称动作成功。
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


运行时会执行固定版本的 `scripts/infer_policy.py`，导入前校验脚本和 ONNX 的 SHA256。
请将官方仓库检出到本页标明的提交：

```bash
git -C outputs/vendor/microduck_rl checkout 0daab160d3fb1c2cbaba16a4a19052f36709dd9e
```

脚本 SHA256：`03bfd0642a880cd379e057536ee819bd1377d4af7b5206a57907d10afd1d4270`。
ONNX SHA256：`ce894d30b3343036d499bfac1102bdaa58d59a21c1b58401ecb70fb6a2f63e0a`。
校验失败时拒绝执行，不自动接受修改后的资产代码。BAM 由 Rhoban 发布，见 `Open_Source_Software_Notice.txt`。
STL 的非商业许可不因资产登记而改变；未经另行授权，不得随商业产品分发。
Linux 使用 `.[genesis,microduck,gui]`，Mac 使用 `.[genesis-macos,microduck,gui]`。

可复现验收命令（证据目录由环境变量指定）：

```bash
JIUWEN_MICRODUCK_INTEGRATION=1 JIUWEN_SIM_RESIDENT_INTEGRATION=1 \
JIUWEN_MICRODUCK_SOURCE="$PWD/outputs/vendor/microduck_rl" \
JIUWEN_MICRODUCK_EVIDENCE="$PWD/outputs/microduck/integration-final" \
.venv-genesis/bin/python -m pytest tests/integration/test_jumper_macos.py -k microduck -q
```
