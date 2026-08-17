# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Cruzr adapter config.

The Cruzr SDK in ``Cruzr_ws`` exposes ROS 2 messages. This config keeps the
ROS topic names, joint names, and simple arm-gesture trajectory parameters in
one place so the agent-facing API can stay semantic.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class DetectorServerConfig:
    """如何连接(或拉起)开放词汇检测服务(GroundingDINO+SAM2)。cruzr 自有副本，不依赖 piper。"""

    url: str = "http://127.0.0.1:8114"
    spawn: bool = True
    host: str = "127.0.0.1"
    port: int = 8114
    device: str = "cuda"
    startup_timeout_s: float = 300.0
    gdino_model_id: str = "IDEA-Research/grounding-dino-base"
    sam2_model_id: str = "facebook/sam2.1-hiera-large"
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    use_sam2: bool = True


def _extract_detector_from_api_servers(api_servers: list) -> DetectorServerConfig:
    """从 YAML 的 api_servers 列表里识别检测服务条目并复制连接/模型参数。"""
    for s in api_servers or []:
        if not isinstance(s, dict):
            continue
        target = str(s.get("_target_", "")).lower()
        if "grounding_dino" not in target and "gdino" not in target:
            continue
        host = s.get("host", "127.0.0.1")
        port = int(s.get("port", 8114))
        defaults = DetectorServerConfig()
        return DetectorServerConfig(
            url=f"http://{host}:{port}",
            spawn=True,
            host=host,
            port=port,
            device=s.get("device", "cuda"),
            gdino_model_id=s.get("gdino_model_id", defaults.gdino_model_id),
            sam2_model_id=s.get("sam2_model_id", defaults.sam2_model_id),
            box_threshold=float(s.get("box_threshold", defaults.box_threshold)),
            text_threshold=float(s.get("text_threshold", defaults.text_threshold)),
            use_sam2=bool(s.get("use_sam2", defaults.use_sam2)),
        )
    return DetectorServerConfig()


@dataclass
class CruzrConfig:
    """Configuration for the Cruzr ROS 2 adapter.

    Positions are in radians because the Cruzr SDK examples command
    ``JointCmd.position`` in radians.
    """

    # ROS 2 environment. ``ros_workspace`` is informational for examples and
    # diagnostics; Python code expects the caller to source ROS setup files
    # before running when using real hardware.
    ros_workspace: str = "/home/linyiqi/Cruzr_ws"
    command_topic: str = "/mc/sdk/robot_command"
    state_topic: str = "/mc/sdk/robot_state"
    developer_mode_service: str = "/sys/task/developer_mode"
    ros_python: str = "/usr/bin/python3"  # interpreter for the camera worker subprocess (needs rclpy)

    # Body joint names from the SDK demos.
    left_shoulder_pitch_joint: str = "L_shoulder_pitch_joint"
    right_shoulder_pitch_joint: str = "R_shoulder_pitch_joint"
    waist_yaw_joint: str = "waist_yaw_joint"
    default_arm: str = "left"

    # Safe simple gesture: raise shoulder pitch and optionally return home.
    home_position_rad: float = 0.0
    raise_position_rad: float = 1.0
    left_raise_position_rad: Optional[float] = None
    right_raise_position_rad: Optional[float] = -1.0
    control_rate_hz: float = 500.0
    ramp_duration_s: float = 2.0
    # 头部专用 ramp 时长(s)：头是轻负载 2-DOF、不搬箱，用不着手臂那条 1.5s 慢 ramp。
    # 搜索/接近时每任务要 set_head 十几~几十次,这里单独调快(默认 0.4)大幅缩短每次"看一眼"的
    # 头动耗时,且不影响抓取/举升的轻柔运动(那条仍走 ramp_duration_s)。set_head 走这条。
    head_ramp_duration_s: float = 0.4
    # 夹取/放置里【不夹箱的空载移动】(举到 ready、下探到夹取高度外侧、放好松开后抬离)专用快 ramp。
    # 这些动作没有接触/夹紧,不涉及夹变形或接触力核验,可比慢 ramp 快得多;smoothstep 末速为 0,加速
    # 只是中段峰值更快、落点不变。夹紧那一下(clamp)+放置的下放到位仍走 ramp_duration_s 保精细。
    arm_transit_ramp_duration_s: float = 0.6
    # 夹取/放置里【和箱子接触】那几下(夹紧 clamp、放置下压到位)专用 ramp。这两步涉及夹变形/掉箱/
    # 接触力核验,故与空载 transit 分开、单独一个旋钮:想快就压小(接触峰值速度随之升高——压太狠会夹
    # 变形/掉箱),想稳就调大。默认 0.6(比原来的 1.0 快)。不影响腰/举升/回家(那些仍走 ramp_duration_s)。
    arm_contact_ramp_duration_s: float = 0.6
    # 放置【下压把箱子放到桌面】那一下专用 ramp,从夹紧 clamp 拆出来(两者原本共用 arm_contact_ramp_duration_s)。
    # 放置下压带着箱子往桌面落,太快会磕桌/顿一下;单独一个旋钮 → 只把放置放慢,夹紧 clamp 不受影响仍走
    # arm_contact_ramp_duration_s。想更轻柔就调大。默认 1.2(比夹紧的 0.6 慢一倍,箱子落得更稳)。
    place_contact_ramp_duration_s: float = 1.2
    # 放置下压的【保夹距分段路点数】。下压那一步不再一次性关节直线插值(端点保夹距、中途夹距会鼓开→掉箱),
    # 而是把夹板 TCP 在笛卡尔空间从当前持有位到落点直线分成 N 段、逐段解 IK,再作为一条连续轨迹流式下发
    # (总时长仍= place_contact_ramp_duration_s,不变慢、无顿挫)。N 越大路径越贴合、夹距越稳,但下发前的
    # IK 预算越多(N×2 次)。1=退回旧单段行为。默认 4(真机可调)。
    place_lower_waypoints: int = 4
    # 举升柱(lifter_pitch)/直起躯干专用 ramp 时长(s)。set_lifter 及 lift_to_clearance/dual_arm_place 倾身
    # 都走它,与手臂 transit/contact ramp 分开。默认 2.0=保守(=旧全局默认,零行为变化);真机在 YAML 逐步压
    # (如 1.2)缩短「base 停→手臂抓」死时间。物理安全:带箱下压/直起过快有磕桌/掉箱风险,须逐步验证。
    lifter_ramp_duration_s: float = 2.0
    # Preset ABSOLUTE carry height (base-frame z, m) of the paddle TCPs / box centre after
    # lift_to_clearance. The box is stood up (lifter->0) then raised to exactly this z (x/y &
    # gap preserved), so the box always clears the waist camera regardless of grasp height.
    # Tune to the real waist-camera geometry.
    transit_lift_z_m: float = 1.15
    step_rad: float = 0.0008
    settle_s: float = 0.05
    open_loop_hold_s: float = 0.25

    # Optional readback tolerance used when state topic is available.
    reach_tolerance_rad: float = 0.03

    # --- 腰部 RGBD 相机（waist_front_rgbd）
    waist_color_topic: str = "/sensor/camera/waist_front_rgbd/color/raw"
    waist_depth_topic: str = "/sensor/camera/waist_front_rgbd/depth/raw"
    waist_camera_info_topic: str = "/sensor/camera/waist_front_rgbd/color/info"
    color_msg_type: str = "shm_msgs/msg/Image1m"
    depth_msg_type: str = "shm_msgs/msg/Image1m"
    depth_scale: float = 0.001
    camera_grab_timeout_s: float = 8.0
    # Camera workers use Fast DDS default transports. The robot performs stereo
    # rectification, depth and point-cloud synthesis and publishes standard ROS messages.
    head_camera_rmw_implementation: str = "rmw_fastrtps_cpp"
    camera_fastdds_profiles_file: str = ""
    head_camera_ros_domain_id: int = 0
    head_camera_discovery_range: str = "SUBNET"
    # 每次抓帧新起子进程 worker，头/腰相机是 BEST_EFFORT/VOLATILE：只收发现之后新发布的帧。运行刚开始时
    # 头部立体相机节点/DDS 发现尚未就绪，头一两次抓帧会在 timeout 内一帧都收不到(no_color_frame)——表现为
    # 【只有头部、只在开头】没图，过一会儿流起来就正常(waist 启动快不受影响)。拿不到帧时用【全新子进程】
    # 重试：fresh 子进程重做一次完整 DDS 发现，等流起来即成。retries=额外重试次数(总尝试=retries+1)；
    # settle=两次尝试间的稳定等待(s)。稳定后首帧即中、无重试开销。
    camera_grab_retries: int = 2
    camera_grab_retry_settle_s: float = 0.5
    # 每次抓帧新起 worker，其 tf2 监听要几秒才把 base_link 装进 TF 树；2s 常不够 →
    # locate_for_grasp 报 no_live_tf、交接失败。放宽到 8s 稳定拿到实时外参(命中即返回，不总等满)。
    camera_tf_timeout_s: float = 8.0
    camera_calib_path: Optional[str] = None
    base_frame: str = "base_link"
    camera_optical_frame: str = "waist_front_rgbd_color_optical_frame"
    # Path to the Cruzr URDF used by the FK checker and future motion planning.
    urdf_path: str = "/home/riemann/Robot/Cruzr_ws/cruzr_s2_description/cruzr_s2_description/urdf/cruzr_s2_v1/cruzr_s2_v1.urdf"
    # Mesh package root so pinocchio+coal can resolve package:// collision meshes for self-collision checks.
    urdf_package_dir: str = "/home/riemann/Robot/Cruzr_ws/cruzr_s2_description"
    # End-effector link names for FK validation and grasp-frame construction.
    left_arm_leaf: str = "L_sixforce_link"
    right_arm_leaf: str = "R_sixforce_link"
    # Local axis of the palm that points "outward" (away from the palm surface).
    # Default (0, 0, 1) is a placeholder — FINALIZE by running scripts/cruzr/fk_check.py
    # on real hardware and observing which tool local axis points palm-outward.
    palm_normal_local: tuple = (0.0, 0.0, 1.0)

    # --- 手部力传感器话题（WrenchStamped）
    left_hand_ft_topic: str = "/mc/ft_states/L_hand_ft"
    right_hand_ft_topic: str = "/mc/ft_states/R_hand_ft"
    # Clamp-verify force gate: both hands must read >= this before the lift. Lowered
    # to pair with the gentler grasp_inset_mm below (a shallow grip presses lightly);
    # still confirms real contact vs a miss.
    contact_force_threshold_n: float = 2.5
    # Per-side LATERAL clamp depth (mm) the paddles press INTO each box side face (sign per
    # geometry.py:129). Grip gap = box_width - 2*inset: POSITIVE = paddles cross the face inward
    # (press/grip); NEGATIVE = they stop OUTSIDE it (gap -> no contact / "没够到"). Tune on hardware
    # with contact_force_threshold_n.
    grasp_inset_mm: float = 5.0
    # 放置【下压到桌面】时,双夹板在"保持抓取实测夹距"基础上再各向内过盈总量的一半(mm,总量,左右各半)。
    # 抓取靠 grasp_inset_mm 让夹板越过箱面产生夹紧力;放置只保持夹距会零过盈→桌前手臂内向刚度弱→夹力不足→下压时掉箱。
    # 此值给放置补回夹紧过盈,像抓取一样压住。太大压变形/够不到,太小仍掉。真机调(配合 grasp_inset_mm)。0=只保持夹距(旧行为)。
    place_squeeze_mm: float = 20.0
    # Keep the whole box this far (mm) INSIDE every sensed table edge when placing, so it
    # never lands on the rim / overhanging. Used by dual_arm_place(surface=...).
    place_edge_margin_mm: float = 20.0
    # Max lifter pitch (rad) dual_arm_place may lean to reach the table. This is only a CEILING:
    # the search picks the SMALLEST lean that reaches, so a bigger ceiling just lets it reach
    # farther/lower tables without forcing a big lean. Lower it if the torso dips onto the
    # table; if the arms still can't reach within it, place returns unreachable (drive closer).
    place_max_lift_lean_rad: float = 0.8
    # Arm-home CLEARANCE waypoint (arm joint -> rad): _safe_home_arms first abducts the arms
    # OUT to the sides via this pose, THEN descends to 0, so the forearms come down along the
    # body's sides instead of dragging across the torso (the self-collision model does not
    # always catch that contact). Only non-zero joints need listing (rest -> 0). shoulder_roll
    # NEGATIVE = abducted out (URDF: the near-blocked +roll side is inward). Tune the magnitude
    # per hardware; set to {} to disable (falls back to the direct/one-arm/ready escalation).
    home_clearance_arm_q: dict = field(default_factory=lambda: {
        "L_shoulder_roll_joint": -0.7,
        "R_shoulder_roll_joint": -0.7,
    })

    # --- 检测服务
    detector: DetectorServerConfig = field(default_factory=DetectorServerConfig)

    # --- 头部双目相机（stereo，仅用于方位搜索；无 depth / 无标定 / 无 TF）
    head_left_topic: str = "/sensor/camera/stereo_left/image/raw"
    head_color_msg_type: str = "shm_msgs/msg/Image2m"
    head_aligned_color_topic: str = "/sensor/camera/stereo/left/image_rect_color/compressed"
    head_aligned_color_msg_type: str = "sensor_msgs/msg/CompressedImage"
    head_yaw_joint: str = "head_yaw_joint"
    head_pitch_joint: str = "head_pitch_joint"
    # 近似水平视场角(rad)，把像素横向误差换成粗方位；标注为近似值，迭代重检纠偏。
    head_hfov_rad: float = 1.7
    # 头部扫掠采集：目标不在初始视野时只转头(不转底盘)遍历这些 yaw。均在 ±1.658 限位内。
    head_search_yaw_positions_rad: tuple = (0.0, 0.6, -0.6, 1.2, -1.2)
    head_search_pitch_rad: float = -0.35  # 低头看地面/低台箱体(实机确认: +pitch=抬头, -pitch=低头)
    head_forward_yaw_rad: float = 0.0
    head_forward_pitch_rad: float = 0.0
    # 粗定位驱近期间头部固定低头角，让头部视野与腰部视野衔接。-pitch=低头。
    head_approach_pitch_rad: float = -0.35
    # 粗定位驱近期间【头部偏航跟踪】：每帧把头按 gain×画面方位β转回去对准目标，抵消一次性开环转身的
    # 欠/过冲(否则头钉在正前方看不正目标→grounded 复检"不在参照物上"→head_hits=0 卡死)。底盘按
    # head_yaw+β 的真实底盘系方位转向。gain≈0.9 一步基本对正；调小更稳、调大更快但可能过冲抖动。
    head_yaw_track_gain: float = 0.9
    # 头部跟踪偏航限位(rad)：约束跟踪时头偏航幅度，勿超头部机械限位(±1.658)。
    head_yaw_track_max_rad: float = 1.2

    # --- 头部搜索用【右目原始图】= 厂商右目直发话题(实测在发、Publisher=1)，非左目 rect-color 压缩流。
    # 类型是 shm_msgs/msg/Image2m(固定 2MB 缓冲、yuv422)：header+载荷经 DDS 网络可达；worker 的
    # _import_msg 能导入它、decode_image_msg 按 data[:h*step] 切缓冲并解 yuv422→RGB，无需机器人端 republish 栈。
    # 搜索纯 2D：无 depth / 无点云 / 无 TF。grounded("X 在 Y 上")只用两者 bbox 的 2D 重叠度判定关系。
    head_right_topic: str = "/sensor/camera/stereo_right/image/raw"
    head_right_msg_type: str = "shm_msgs/msg/Image2m"
    # grounded 头部搜索的 2D 判定阈值：目标 bbox 落在参照 bbox 之内的比例(|t∩r|/|t|, 用框不用 mask)≥此值才算"在其上"。
    # 用 bbox 而非像素 mask：小物摞在正视高箱顶面时 mask 几乎不相交(重叠≈0)，但目标框完全落在参照框内；
    # 严格 IoU(交/并)对"小物摞大面"也会偏低(并集被大参照主导)。头部只做粗筛(把底盘朝对的候选驱近)、
    # 腰部 RGBD 做最终精确判定，故取偏松默认。tunable。
    head_on_overlap_min: float = 0.15

    # --- (遗留/搜索已不用) 头部点云 3D 验证字段：search_target 已改走右目 2D bbox 重叠，不再抓点云。
    # 仍保留供 grab_head_frame / _head_cloud_cmd 这条独立相机能力使用(其自带单测)。
    # 点云与头部检测图同光学系(stereo_left_rgb_optical_link)，organized、单位米、含 NaN 空洞。
    # 空 topic / 拿不到 cloud|TF / 有效点不足 → 优雅降级回纯 bearing 搜索(无回归铁律)。
    head_pointcloud_topic: str = "/sensor/camera/stereo/pointcloud/raw"
    head_pointcloud_msg_type: str = "sensor_msgs/msg/PointCloud2"
    head_optical_frame: str = "stereo_left_rgb_optical_link"   # 点云→base 的 TF 源系(与头检测图同系)
    head_cloud_min_valid_points: int = 30    # mask 落到点云上的最少有效(非 NaN)点，低于此判无 3D → 降级
    head_ground_verify: bool = True          # 总开关：False 强制走纯 bearing(调试/点云未标定时降级用)
    # grounded(on/has)时若头部点云/TF 拿不到、核验不出"目标在参照物上"，底盘怎么办：
    #   False(默认, fail-open) = 降级为纯 bearing 继续粗驱近(点云还没接好时靠方向先开近，腰部做最终判定)。
    #   True(fail-closed/严格) = 核验不了就【不算命中、不前进】：搜索阶段不 acquire、驱近循环判丢失即中止，
    #     绝不靠纯 bearing 盲开——完全贴合"条件满足才前进"。开启后必须先接通头部点云，否则 grounded 抓取
    #     会在搜索阶段就以 object_not_found 中止(并打印 head_cloud DEGRADED WARNING 指出缺哪块)。
    head_grounded_strict: bool = False
    # on_surface_margin_mm / base_frame 复用腰部 grounded 字段，不重复定义。

    # --- 底盘差速轮速控制（/mc/sdk/robot_command driving_wheel + MODE_VELOCITY + odom 闭环）
    # 弃用 Nav2：底盘真正执行的是 SDK 轮速指令，不是 /mc/cmd_vel（那条要厂商 chassis_controller）。
    nav_python: str = "/usr/bin/python3"       # 跑轮速 worker 的解释器（子进程需 rclpy）
    odom_topic: str = "/mc/odom"
    front_lidar_topic: str = "/sensor/lidar/front"
    back_lidar_topic: str = "/sensor/lidar/back"
    left_wheel_joint: str = "driving_wheel_left_joint"
    right_wheel_joint: str = "driving_wheel_right_joint"
    base_k_rot: float = 1.5                     # 转向时(远离目标)轮速(rad/s)
    base_k_rot_min: float = 0.25               # 接近目标角时的最小轮速(rad/s)，在底盘死区之上、减少过冲
    base_k_rot_slow_rad: float = 0.5           # 距目标角小于此(rad)开始按比例减速
    base_k_fwd: float = 0.8                     # 前进时轮速(rad/s)
    base_k_fwd_min: float = 0.25               # 前进接近目标距离时的最小轮速(rad/s)，死区之上、减过冲
    base_k_fwd_slow_m: float = 0.25            # 距目标距离小于此(m)开始按比例减速(减少顶到目标的过冲)
    # approach_for_grasp/approach_for_place 精定位专用【温和】底盘参数(与全局 base_k_* 分离):搜索大转身
    # 可用快 base_k_*(过冲无所谓),但精定位过冲会把腰部相机边缘的箱子甩出视野→重检丢失→抓取静默跳过→
    # 空放。默认=真机验证过的稳值(与 base_k_* 一致);approach_for_grasp/approach_for_place 用它们覆盖全局。
    approach_k_rot: float = 1.5                 # 精定位转向轮速(rad/s)
    approach_k_rot_slow_rad: float = 0.5        # 精定位减速尾段(rad):越大减速越早、过冲越小
    approach_k_fwd: float = 1.0                 # 精定位前进轮速(rad/s):提速(0.8→1.0)缩短精定位耗时;远段有 _forward_step 预留、末段按 base_k_fwd_slow_m 减速到 k_fwd_min 落位,提速不改变落地的柔和/防撞
    base_yaw_tol_rad: float = 0.05             # "正对"容差(rad,≈2.9°),三处共用:①离散驶近的转向相位转到此即开始直行——收紧=更接近正对才走直线,直接消除到位后的本体夹角(旧 0.08≈4.6° 就是实测残留夹角)②plan_grasp_pose_goal 判 in_band 的居中门槛 ③approach_for_place 相位1方位门槛。噪声地板≈0.037(cy测量+转向里程计),更小易在转向相位振荡(有超时/max_iters 兜底);0.05 稳,要更正可试 0.04
    grasp_yaw_align_tol: float = 0.20          # 抓取靠近"正对可见面"的对齐容差(≈11°);大于此则沿面法向绕正
    grasp_face_flatness_max: float = 0.15       # 面法向可信阈值 λmin/λmid：小于此才认为看到平整正面、据其绕正；否则退回径向靠近
    grasp_face_step_m: float = 0.12             # 到位后绕正的每步位移封顶(米)：小步走弧线,避免切近; 每步复检
    # 注:以下 grasp_square_min_aspect / grasp_square_tie_eps 已不再被 live 抓取路径
    # (approach_for_grasp → _grasp_near_face_normal)读取——该路径改用“迟滞锁面”(见 api.py):
    # 首帧选最朝向机器人的面、后续帧只在其附近选,近方形/角对着也稳定正对某个面,不再按 aspect/tie 退回径向。
    # 这两个字段仅剩 motion/base_goal.py 的 footprint_face_normal(测试用平行实现)引用,保留不删。
    grasp_square_min_aspect: float = 1.2        # (legacy) footprint 长/短轴比阈:footprint_face_normal 判近方形→不可信
    grasp_square_tie_eps: float = 0.1           # (legacy) 近面明确门槛:最近/次近面法向点积之差
    # 抓取正对(先原地绕正对准某面、再 turn=0 直线进)的 square 容差(rad,≈15°)。必须 > footprint yaw 帧间
    # 抖动(近方形箱实测 ~0.26),否则远处近方形箱每帧 |square_turn|≈抖动 > 容差 → 只原地转、永不前进(死锁);
    # 故不能沿用 place 侧的 place_square_tol_rad=0.15。细长箱 yaw 稳定、转一次即 ≈0,会自然收敛到远小于此值;
    # 近方形箱最终 square 残差以此为界(任意面都可抓故可接受)。真机可调。
    grasp_square_tol_rad: float = 0.26
    # 绕正“朝向一致性”门槛(≈25°)：仅当连续两帧 footprint yaw 一致(mod π，90°轴翻转判为不一致)才据其绕正。
    # 真机日志实测近方形/mask 污染时 yaw 逐帧在 ±30° 跳变；此门控让底盘只对稳定朝向绕正、不追单帧噪声尖峰，
    # 否则退回径向对准中心(实测已能对到 ±2.5°)。太小会打断长方形箱正常收敛(每步 yaw 本就会渐变)，太大放进噪声。
    grasp_yaw_consistency_tol_rad: float = 0.44
    # 近方形箱“远距离先对齐再直进”的安全间隙(米)：绕正只在“驶近该步后底盘到箱心的建模最近距离≥此值”时才走。
    # 只有较远(有弧线空间)时该门才通过、绕正生效；进抓取带(≈0.40)后门槛挡住、改纯径向靠近——这样即便面法向被
    # mask 污染而错，绕正也只会在安全距离外侧向错位、绝不驶入箱子(这是把 in-band 绕正撞箱回退后改到远处对齐的关键护栏)。
    grasp_square_min_clearance_m: float = 0.50
    base_pos_tol_m: float = 0.05               # 到达目标距离容差
    base_safe_dist_m: float = 0.45             # 行进方向扇区内雷达距离小于此即急停
    base_lidar_self_floor_m: float = 0.25      # 忽略比此更近的雷达回波(自身/近场噪声)
    base_lidar_sector_deg: float = 25.0        # 只看行进方向 ±此角度内的雷达
    base_move_timeout_s: float = 20.0          # 单段(转/进)超时

    # --- 传感器引导换向（face_surface/face_object）：转向目标/放置面时按感知方位转，替代盲转固定角
    face_surface_tol_rad: float = 0.10         # |感知方位| 小于此即视为已正对，不转
    # 目标不在视野时【连续自转】边转边检测(无90°步进盲扇区)，一见到即停。为真正消盲区，
    # 自转须够慢:轮速×检测周期 ≤ 相机FOV(否则相邻检测帧的视野仍留缝)。worker 自设上限一圈+超时,
    # 父进程死亡时经 stdin EOF 兜底停车,底盘不会转飞。
    search_spin_wheel_rs: float = 0.7         # 连续搜索自转轮速(rad/s);越慢越不漏、越慢越久
    search_spin_max_rad: float = 6.6           # 自转搜索上限(≈一圈+余量),到此仍未见到即判未找到
    search_spin_timeout_s: float = 40.0        # 自转搜索超时(s)
    search_spin_dir: float = 1.0               # 自转方向(+左/CCW, -右/CW)
    # 空间关系判定（relation=on/under/beside/near，见 perception/scene3d.py:relation_holds）
    on_surface_margin_mm: float = 80.0         # on/under 判"在参照面之上"的 footprint/高度容差
    beside_max_gap_mm: float = 400.0           # beside：两者 footprint 之间允许的最大水平间隙
    near_max_dist_mm: float = 1000.0           # near：两者中心的最大水平距离（不看高度）

    # --- 头部粗定位驱近：连续前进（--forward worker，匀速开、雷达急停、自设上限；父进程命中即停）
    # 头部看到锚点后不再走一步看一步，而是匀速前进，直到腰部拿到精确坐标接管。安全同自转:stdin停/EOF/
    # SIGTERM/自设上限，前向单线雷达急停。
    approach_drive_wheel_rs: float = 0.8       # 连续驱近前进轮速(rad/s):提速(0.5→0.8)吃掉大段行程;粗驱近在~1.2-1.5m 远处就交接给腰部(早交接)+前向雷达急停,提速仍安全;过快则腰命中→停车的过冲变大(靠停车后重检 + approach_for_grasp 收敛吸收)
    # 下面两个是【安全兜底上限】,不是正常停车点:正常由父进程在"腰部接管/头部丢锁/雷达防撞"时停车。
    # 设得宽松,让底盘一直逼近到腰部真能看清为止(别在目标还在前方时就固定 2.5m 放弃);父死时它才靠此自停。
    approach_drive_max_m: float = 5.0          # 驱近前进安全兜底距离(m,tunable):走到此仍没接管才判失败
    approach_drive_timeout_s: float = 60.0     # 驱近前进安全兜底超时(s,tunable)
    # 驱近不是直着开:父进程每轮把头部实时方位(bearing)喂给 worker,叠成左右轮差速,弧线朝目标转向前进。
    approach_drive_steer_gain: float = 0.6     # 转向增益(轮rad/s per bearing-rad):方位越偏、差速越大 → 朝目标转
    approach_drive_steer_max_rs: float = 0.4   # 单侧差速上限(rad/s),防大方位时过转/原地打转(0=退化为直行)

    # --- 靠近循环（approach）
    center_tol_frac: float = 0.08          # |横向像素误差/宽| 小于此即视为已对中
    approach_step_m: float = 0.4           # 每次前进步长(m)
    approach_max_iterations: int = 12
    # approach_for_grasp 迭代收敛上限:每轮驶近后复检修正,直到箱子居中且在抓取带内(in_band)。
    # 偏置越大、一次驶近的 odom 转动残差越大,靠多轮收敛到"正好可抓"。
    # 5:前进用"预留末段减速"策略(见 approach_forward_step_m),提前交接后的整段前进(0.4-0.75m)只需
    # ≤2 次驶近即收敛,再留 1-2 轮给方位/正对(squaring)微调即可。
    approach_converge_iters: int = 5
    # 每步前进【硬截断】上限(m):单帧误检也只走这一小步、复检修正,防大前冲撞物。用于【不可信帧】的前进——
    # approach_for_grasp 无 footprint 的径向兜底、以及单发驶近的末段对齐(_forward_step)。正对法线后的直行不用它,
    # 改用下面的 grasp_approach_forward_reserve_m 预留策略(避免每 0.25m 停车+重检的卡顿)。
    approach_forward_step_m: float = 0.25
    # approach_for_grasp 正对法线后【直行】的末段减速预留距离(m),对称于 place 侧 place_approach_forward_reserve_m:
    # 远段一次连续开到离抓取待命距此值处(单次 navigate_relative,即便 odom 过冲也停在箱前不顶箱),末段≤此值落在
    # base_k_fwd_slow_m 减速斜坡内轻柔到位 → 整段≤2 次前进,取代旧的每 0.25m 硬截断(远距离 3-4 次停车+重检=卡顿)。
    grasp_approach_forward_reserve_m: float = 0.15
    probe_bbox_frac: float = 0.10          # 头部框高/图高 ≥ 此值才去探测腰部(靠得够近)
    grasp_forward_min_m: float = 0.30      # handoff：箱心前向距离可抓带下界
    grasp_forward_max_m: float = 0.50      # handoff：上界
    place_approach_edge_m: float = 0.35    # 放箱前底盘停在桌子近边(front_x)前方此距离，让手臂够到桌面
    place_approach_forward_reserve_m: float = 0.15  # 放置正对后"直入"的末段预留(m):远段一次开到离桌边此距离(单次驶近,即便odom过冲也停在桌前不顶桌),末段≤此值轻柔落位→整段≤2次前进,取代旧的每0.25m硬截断(用户反馈的"分好几小段/慢")
    # approach_for_place 正对桌面:桌面水平→无可信正面法向,改用地面近边法向让底盘垂直正对。优先用桌面
    # 近边直线拟合法向(near_edge_line:对近边最靠机器人的前轮廓拟合直线,取中点+垂线,对大桌/局部可见更稳),
    # 不可信时退回整块 footprint 主轴法向(_grasp_near_face_normal)。
    place_square_tol_rad: float = 0.15     # "已正对近边"阈值(≈9°);大于此则原地旋转对齐(定位到位后)
    place_max_turn_step_rad: float = 0.7   # 放置驶近每步底盘转角封顶(rad,≈40°) fail-safe:即便某帧近边法向翻转给出~90°+的错误转角,单步也只转此值,复检再修,绝不带着箱子大幅甩身(防甩箱)。真机正常对正<40°不受影响
    place_edge_quality_max: float = 0.25   # 近边拟合可信阈值 edge_quality=残差厚度/边长;≤此才用近边法向对正,否则退回 footprint。越小越严
    place_edge_min_len_mm: float = 150.0   # 近边拟合最短边长(mm);短于此认为看到的桌边太少、法向不可信,退回 footprint
    # (legacy) 已不再被 live 放置路径读取——approach_for_place 改用迟滞锁面(_grasp_near_face_normal),
    # 近方形/局部可见的桌面 footprint 也稳定正对某条近边,不再按 aspect 跳过(见 grasp_square_min_aspect 同款处理)。
    # 仅剩 motion/base_goal.py 的 plan_surface_square_turn(测试用平行实现)引用,保留不删。
    place_square_min_aspect: float = 1.2   # (legacy) footprint 长/短轴比阈值;仅 plan_surface_square_turn 仍读
    lost_target_max: int = 3               # 连续丢失目标次数上限
    # 头部俯仰跟踪：贴近时目标下移出高处头部视野 → 随之低头把它留在视野。
    head_pitch_track_gain: float = -0.5    # 负: 箱子在画面下方(v_err>0) → 低头(pitch 减小)。实机确认 -pitch=低头
    head_pitch_track_tol: float = 0.10     # |竖直像素误差/高| 小于此不调
    head_pitch_min_rad: float = -0.78      # head_pitch 下限(URDF)
    head_pitch_max_rad: float = 0.52       # head_pitch 上限(URDF)

    # --- 调试：把每次检测(腰/头两路: prompt+实时画面+mask 叠加)可视化到一个 cv2 窗口。
    # 默认关；YAML 里设 true 或运行前 export JIUWEN_CRUZR_VIZ=1 打开。无 cv2/无 DISPLAY 自动降级为 no-op。
    viz_detections: bool = False

    # --- 底盘接近（approach_for_grasp）停车设定点
    grasp_target_forward_m: float = 0.40        # 到位后盒子距底盘前向间距（须在 grasp_forward_min..max）

    # --- 弧线驶近（arc-to-line）：用点云/footprint 法向定对齐面，规划一段弧把底盘搬到"面法线延长线上的
    # 待命点",随后原地小幅正对 −n + 视觉纠正直入。差速底盘上"正对某面+居中"只有在法线线上才一致,弧线负责
    # "上线"这一步(离散原地转+直行凑不出)。默认【关】=退回现有离散 square-first;隔离验证 drive_arc 通过再开。
    grasp_arc_enabled: bool = False             # 弧线驶近总开关(关=完全现状,天然回归基线)
    # 直角路线终点(待命点 E)比 grasp_d 再远此值 = 【预留距离】。接近段是开环执行、末端会前冲,且 E 从【箱心】算、
    # 箱子前面还凸出半个箱深 → 预留太小会在末端蹭到箱子(真机实测)。留足此距离让路线停在安全外圈,由随后
    # 【里程计闭环+视觉纠正】的直入去精确收尾,并给终检的位置微调留空间。真机调:还蹭箱就加大。
    grasp_arc_standoff_m: float = 0.25
    grasp_arc_min_radius_m: float = 0.35        # 弧最小半径(m):小于此判过紧(可能贴箱急拐)→拒绝、退回离散
    grasp_arc_max_dyaw_rad: float = 1.2         # 弧单次最大转角(rad,≈69°):超过则箱易出腰部视野→拒绝、退回离散
    arc_k_fwd: float = 0.6                      # 弧名义前进轮速(rad/s)
    arc_k_fwd_min: float = 0.25                 # 弧接近目标转角时的最小前进轮速(死区之上)
    arc_curv_gain: float = 0.5                  # 弧曲率反馈 P 增益(轮差速 per odom 曲率误差)
    arc_len_safety: float = 1.5                 # 弧长自停兜底 = R·|dyaw|·此(父死也不会转飞)
    # 直角路线【横向那一段】(_approach_single_shot 的 lateral leg,驶上箱面法线线)的距离标定增益。
    # 几何算的横向距离 = 机器人到法线线的垂距 |s|,理论精确;但真机上里程计/轮滑/感知(箱心 y 偏视野内侧)
    # 会让【实际横移偏短】→ 停不到线上、仍偏一侧。此增益只乘在【横向段】(前进/驶近段不动,故对正常的
    # 前向距离无影响),>1 补偿欠行、把实际横移拉回 |s|。过冲由随后的终检重检+对正吸收,安全。1.0=不变。真机调。
    grasp_lat_gain: float = 1.0

    # --- 末段连续视觉伺服（continuous visual servo，默认关=现状）：近距抓取末段"边开边检不停车",复用
    # start/steer/hold/stop_base_drive 连续行驶原语,消除 _approach_single_shot 末段那两次静止重检的
    # 冻结卡顿;顺带获得末段移动目标跟踪。关时 _approach_single_shot 完全走今天的两次重检(天然回归基线)。
    grasp_servo_enabled: bool = False           # 末段连续视觉伺服总开关(关=完全现状)
    grasp_servo_creep_k_fwd: float = 1.2        # 末段爬行轮速(rad/s,抓取&放置伺服共用):0.6→0.75→1.2 提速逼近;仍<粗驱近 approach_drive_wheel_rs(yaml 1.95)留"边开边检"余量,几何 fwd_max(到点+reserve)自停兜底防冲。冲箱/冲桌/甩箱则调低
    grasp_servo_fwd_max_m: float = 1.2          # 伺服前进【绝对上限】(m,几何安全兜底):伺服现负责整段前进,实际自停距离=min(到抓取点+reserve, 此值);此值只在种子检测异常偏远时封顶,须 < 到箱心距离
    grasp_servo_commit_dist_m: float = 0.30     # 提交距离(m):锁定后箱离开视野且剩余前进≤此→开环走完末段(须再重检才抓);否则丢框即 hold。须≤2×grasp_approach_forward_reserve_m 才能让正常 commit 走满
    grasp_servo_max_polls: int = 40             # 伺服循环轮询上限(安全):检测卡死时防死循环
    grasp_servo_lost_hold: bool = True          # 锁定后丢失且未进提交距离时是否 hold(暂停轮子不盲进);False=继续直行到 fwd_max
    # 放置侧连续伺服:approach_for_place 正对近边后的"直入"段并入一次连续直行爬行(消除放置侧逐段硬停卡顿)。
    # 载箱→纯直行不 steer(打弯会转偏已正对底盘、甩箱)。数值旋钮复用上面 grasp_servo_*(creep_k_fwd/fwd_max_m/
    # commit_dist_m/max_polls/lost_hold)+ grasp_approach_forward_reserve_m + place_approach_edge_m。关=走离散直入(现状)。
    place_servo_enabled: bool = False           # 放置侧末段连续直行伺服总开关(关=完全现状)

    # --- 常驻 worker（性能，默认关=现状）：False=每次抓帧/移动新起子进程(付 rclpy/DDS 发现/TF 重填冷启动)。
    # True=camera/wheel worker 常驻(rclpy/订阅/TF/odom 保活,按 stdin 请求响应),消除每次冷启动税。开启需真机
    # 验证 FastDDS/SDK 稳定性(进程内持久订阅曾拖垮 SDK;独立常驻子进程有隔离但未验证)——验证通过再开。照
    # grasp_arc_enabled 先例:关时代码路径完全等同现状。
    resident_workers: bool = False

    # Session metadata.
    name: str = "cruzr"
    task_prompt: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CruzrConfig":
        """Accept a flat dict or ``env.cfg.low_level`` style YAML."""
        ll = data.get("env", {}).get("cfg", {}).get("low_level", {}) if isinstance(data.get("env"), dict) else None
        prompt = data.get("env", {}).get("cfg", {}).get("prompt") if isinstance(data.get("env"), dict) else None
        api_servers = data.get("api_servers") or []
        detector_cfg = _extract_detector_from_api_servers(api_servers)
        if isinstance(ll, dict) and ll:
            kw = {k: v for k, v in ll.items() if not k.startswith("_")}
        else:
            kw = dict(data)

        valid = {f.name for f in dataclasses.fields(cls)}
        clean = {k: v for k, v in kw.items() if k in valid}
        clean["detector"] = detector_cfg
        if prompt is not None:
            clean["task_prompt"] = prompt
        return cls(**clean)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CruzrConfig":
        """Load config from a YAML file."""
        path = Path(path).resolve()
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    def joint_name_for_arm(self, arm: str | None = None) -> str:
        """Resolve ``left`` / ``right`` to the configured shoulder pitch joint."""
        arm_name = (arm or self.default_arm).lower()
        if arm_name in {"left", "l"}:
            return self.left_shoulder_pitch_joint
        if arm_name in {"right", "r"}:
            return self.right_shoulder_pitch_joint
        raise ValueError(f"unknown arm {arm!r}; expected 'left' or 'right'")

    def raise_position_for_arm(self, arm: str | None = None) -> float:
        """Return the configured raise target for the selected arm."""
        arm_name = (arm or self.default_arm).lower()
        if arm_name in {"left", "l"}:
            return float(self.raise_position_rad if self.left_raise_position_rad is None else self.left_raise_position_rad)
        if arm_name in {"right", "r"}:
            return float(
                self.raise_position_rad if self.right_raise_position_rad is None else self.right_raise_position_rad
            )
        raise ValueError(f"unknown arm {arm!r}; expected 'left' or 'right'")
