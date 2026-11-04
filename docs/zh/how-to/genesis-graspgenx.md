# 在 Genesis 中用 GraspGenX 抓取桌面物体

MOZ1 右夹爪支持两种点云输入。两种模式都向同一个 GraspGenX 服务请求抓取位姿，再由 cuRobo 和 Genesis 执行。一次会话只使用一种输入。

| 会话的 observation_mode | pick_target 的 target | 点云 |
| --- | --- | --- |
| perception | 可见且唯一的英文描述，如 small orange cube | 头部相机的 RGB、GroundingDINO + SAM2 分割、同帧有效深度和标定。目标太小时，在新一帧对分割区域加密采样、重新分割。只能看到朝向相机的表面。 |
| ground_truth | sim_list_entities 返回的 entity_id | 从已注册、可移动的单刚体当前碰撞网格按表面积采样，包含背面。资产使用碰撞代理时采到的是该代理的表面。 |

## 启用服务和规划配置

在 Genesis 环境安装 `.[genesis,genesis-planning,genesis-graspgenx]` 和工程已验证的 cuRobo V2。在推理环境单独准备 GraspGenX 软件和权重。主工程只实现客户端，请求 ZMQ 上的 `infer_object`。使用上游服务时将 `--config` 指向实际模型配置，`--assets_dir` 指向部署使用的夹爪资源目录，`--host 127.0.0.1 --port 5558` 对应示例客户端。感知模式还需要可访问的 `gdino_sam2` 分割服务。

使用本地已有实验部署时，可以开两个终端分别运行以下命令。服务生命周期由启动者管理。

```bash
outputs/graspgenx-integration/.venv/bin/python outputs/graspgenx-integration/scripts/serve.py
python -m jiuwensymbiosis.serving.grounding_dino_sam2_server --port 18114 --box-threshold 0.25
```

设置 `MOZ1_XRDF` 为本机 MOZ1 碰撞配置的绝对路径。从仓库目录启动 Genesis 时，在现有移动机器人配置中指定规划配置。该示例在单机器人场景初始化一张桌子和一个可移动方块，并给机械臂设置固定的初始准备姿态。初始姿态与随后推断的抓姿是独立的。

```python
from pathlib import Path
from jiuwensymbiosis.simulators.genesis.camera import PreviewSettings
from jiuwensymbiosis.simulators.genesis.preview import GenesisPreview

settings = PreviewSettings(
    scene_id="library://spiritai/moz1/moz1_omni_gripper_double_narrow",
    robot_mode="moz1_wheels",
    planning_profile=str(Path("configs/simulators/planning/moz1-graspgenx.json").resolve()),
)
preview = GenesisPreview(settings=settings)
preview.start()
```

后台状态为 `ready` 后，用已有 Agent 会话或直接用公开 API 执行。以下示例在桌面目标已进入工作空间时运行。

```python
from jiuwensymbiosis.simulators.genesis.agent import make_session
from jiuwensymbiosis.perception.config import DetectorConfig, HttpEndpointConfig

detector = DetectorConfig(
    mode="remote",
    endpoint=HttpEndpointConfig(url="http://127.0.0.1:18114", request_timeout_s=30),
)
with make_session(preview, observation_mode="perception", detector_config=detector) as session:
    result = session.api.pick_target("small orange cube")
```

真值模式在新的、夹爪空闲的场景中独立运行。

```python
with make_session(preview, observation_mode="ground_truth") as session:
    entities = session.api.sim_list_entities(query="orange cube")["entities"]
    assert len(entities) == 1
    result = session.api.pick_target(entities[0]["entity_id"])
```

结束后调用 `preview.stop()`。工具层按动作授权、规划器、夹爪以及输入模式注册 `pick_target`。感知失败会返回失败或抛出异常；场景真值不会替代缺失的相机点。

## 位姿与成功标准

后端收到规划基座系中的 N×3 点云，单位米，以及 MOZ1 的开口与半开几何描述，返回一组 4×4 刚体变换和置信度。执行端通过 `T_base_gripper @ T_gripper_control` 得到 URDF 工具目标。示例优先尝试接近方向向下的高分候选；倾角和绕竖轴的转角均来自模型。当前适用于桌面上受支撑的单个刚体。

动作先检查预抓、抓取和抬升的逆解，再用 cuRobo 和加载的碰撞几何检查路径。抵达抓点后先规划约 8 cm 的抬升。夹爪以速度限制下发全闭合位置，持续施力，指令到达闭合端点后直接执行抬升。仿真内部用目标位置关联唯一的刚体来检查碰撞并测量结果；物体依靠接触动力学随动。

确认抓起需要轨迹到位、机器人停稳、目标升高至少 5 cm、失去外部支撑、双指仍接触以及相对位姿稳定。接触读数异常、夹空、规划失败、场景改变等会返回失败。新夹爪需提供其自身的开口描述、工具坐标变换和执行配置。使用真机前还需独立验证相机外参、夹爪模型和接触控制。
