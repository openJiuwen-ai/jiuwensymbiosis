# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""「手眼标定」工具面板(NiceGUI):四步向导。

面向没做过标定的人。四步分别对应标定链上四个必须由人参与的环节:

  0 准备   自助生成可打印标定板 + 硬件与安全确认
  1 示教   松力矩手动摆姿态,实时看板检测结果,逐个记录 waypoint
  2 采集   先无运动预演,通过后再真机采集 → 求解 → 发布
  3 结果   质量指标、失败原因与补救建议、一键写回运行配置

界面只做展示与编排,标定逻辑全在 ``CalibrationEngine`` 背后的
``jiuwensymbiosis.calibration`` 工作流里(与命令行同一套代码)。事件靠 ``drain()``
轮询消费,与「感知测试」一致。
"""

from __future__ import annotations

import dataclasses
import shutil
import tempfile
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any

from nicegui import ui

from jiuwensymbiosis.utils.logging import get_logger
from jiuwensymbiosis_gui.workbench import board_print, registry
from jiuwensymbiosis_gui.workbench.app_state import AppState
from jiuwensymbiosis_gui.workbench.board_print import BoardParams, BoardParamsError
from jiuwensymbiosis_gui.workbench.calibration_engine import CalibrationEngine, CalibrationSetup
from jiuwensymbiosis_gui.workbench.maintenance import MaintenanceOwner

logger = get_logger(__name__)

__all__ = ["CalibrationView"]

_STEPS = [
    ("prepare", "准备"),
    ("teach", "示教姿态"),
    ("capture", "采集与求解"),
    ("result", "结果"),
]

# 建议的示教姿态数下限/上限。少于下限求解不稳,多于上限收益递减、耗时明显变长。
_MIN_WAYPOINTS = 8
_RECOMMENDED_WAYPOINTS = 16

# 板参数两行输入的共同宽度。三个控件的那行按此平分后仍放得下「ChArUco(推荐)」。
_FORM_WIDTH_PX = 460

# 板参数输入的防抖(ms)。渲染一次预览要生成板图并合成整页(~70ms),不值得为键入的
# 每个字符跑一遍;300ms 停手即刷新,点上下箭头连调几下也只渲染最后那次。
_INPUT_DEBOUNCE_MS = 300

# 相机安装方式的说法。它决定用户该把相机摆在哪,但对每个机型是固定的(适配器 config
# 里是 Literal),所以只陈述事实,不写成可选项。
_MOUNT_TEXT = {
    "eye_to_hand": "固定在外部，不随机械臂移动",
    "eye_in_hand": "装在机械臂手腕上，随机械臂移动",
}

# 标定板该装在哪,是相机安装方式的物理蕴含:相机不动就得让板跟着臂动,相机跟着臂动
# 就得让板不动。装反了相机与板的相对运动会退化,解出来必是废值 —— 但这条原理不写进
# 界面,准备阶段只给动作。
_BOARD_SETUP_TEXT = {
    "eye_to_hand": "把打印件贴在平整硬板上，再夹进或贴在夹爪上。",
    "eye_in_hand": "把打印件贴在平整硬板上，平放固定在工作区里。",
}
_BOARD_CHECK_TEXT = {
    "eye_to_hand": "标定板已夹紧，板面平整",
    "eye_in_hand": "标定板已固定在工作区，板面平整",
}

# 示教方式的指引与危险级别。键是 ``resolve_teaching_mode`` 的返回值 —— 判据是设备实现
# 了哪个端口(``ManualGuidance``),不是机型名,所以新本体实现与否都会自动落到正确一档。
_TEACH_MODE_TEXT = {
    "manual_guidance": (
        "⚠️ 确认后机械臂会立刻失去力矩并下坠 —— 请先用手托住，再确认。",
        True,
    ),
    "external_snapshot": (
        "本机型不支持松力矩示教:力矩保持开启。请用示教器或厂商的拖拽模式把机械臂移到位，再点「记录当前姿态」。",
        False,
    ),
}

# 示教全程双手都在托机械臂,操作靠盲按键盘。空格固定表示"改变机械臂的软硬":变硬
# (暂停)一下就够,变软(确认开始 / 继续)必须双击 —— 那一下的后果是机械臂失力矩下坠,
# 不能被一次误触触发。间隔短于此算双击,长于此算两次互不相干的按键。
_DOUBLE_TAP_S = 0.8
_KEY_HINT_ARMED = "再按一下空格确认。"

# 示教中可以按键盘的三个阶段(其余阶段按键一律不响应)。
_KEY_STAGES = ("armed", "teaching", "paused")

# 横幅只有"危险"和"说明"两种样式,同一类信息一律复用同一种。
_BANNER_COLORS = {
    "danger": "background:#fee2e2; color:#991b1b; font-weight:600;",
    "warn": "background:#fef3c7; color:#92400e;",
    "ok": "background:#dcfce7; color:#166534;",
    "info": "background:#eff6ff; color:#1e40af;",
}


def _banner_style(kind: str) -> str:
    """横幅样式。四种语义各一套色,整个向导共用 —— 每处现调一个色号会调出五种黄。"""
    return _BANNER_COLORS[kind] + " padding:8px; border-radius:4px;"


_DANGER_STYLE = _banner_style("danger")
_INFO_STYLE = _banner_style("info")

_RESUME_WARNING = "⚠️ 继续示教过程中机械臂会失去力矩并下坠 —— 请先用手托住，再点击继续示教或双击空格。"
_PAUSED_STATUS = "示教暂停中：机械臂已托住，可以松手。"

# 重投影门限的兜底值(引擎正常会把本轮验收实际用的那组门限随结果一起送上来)。
_REPROJ_GOOD_PX, _REPROJ_WARN_PX = 1.0, 2.0

# 硬件清单里"还没配好"那一行的字色。
_WARN_COLOR = "color:#b45309;"

# 相机序列号在配置里的点分路径;「配置」链接按它定位到配置页的对应输入框。
_CAMERA_SERIAL_PATH = "env.cfg.low_level.camera_serial"

# 技术详情里原样保留的完整数值(含界面不单独解读的 AX=XB 残差)。
_RAW_METRIC_LINES = [
    ("reprojection_px", "reprojection: mean {mean:.3f} px / max {max:.3f} px"),
    ("axxb", "AX=XB residual: rot {rotation_deg:.4f} deg / trans {translation_mm:.3f} mm"),
    ("rigidity", "{invariant_frame} consistency: trans {translation_mm:.3f} mm / rot {rotation_deg:.4f} deg"),
]


def _verdict(value: float, good: float, bad: float) -> tuple[str, str]:
    """把一个数值判成 良好 / 一般 / 偏大,返回 (文案, 颜色)。"""
    if value <= good:
        return "✓ 良好", "#15803d"
    if value <= bad:
        return "一般", "#b45309"
    return "⚠ 偏大", "#b91c1c"


# 质量门禁失败项 → 操作者能直接执行的补救动作。键取自两套验收策略产出的
# ``CalibrationDecision.failed_checks``(eye-in-hand 与 eye-to-hand 各一套),按最长前缀
# 匹配。本体不会引入新键 —— 它只是二选一地挑一套策略,所以这张表不随机型增长;策略新增
# 门禁时由 tests/gui/workbench/components/test_calibration_view.py 的覆盖断言拦下。
# 未收录的检查项原样显示 reasons 文本,不臆造建议。
_REMEDY = {
    # 两套策略共有
    "reproj_invalid": "这轮没拿到有效的重投影误差，通常是几乎所有照片都没认出标定板。"
    "检查板是否在视野内、对焦与曝光是否正常，然后重新采集。",
    "quality_nonfinite": "质量指标算出了非法值，通常是有效姿态太少或姿态高度重复。重新示教，把姿态数量和差异都加大。",
    # eye-in-hand:重投影 / AX=XB / 板原点散布
    "reproj": "角点重投影误差偏大。检查板是否平整、有无反光与运动模糊，并核对方格/marker 实测尺寸。",
    "axxb_rot": "机械臂位姿与相机观测对不上(旋转)。检查标定板在采集途中有没有被碰动，以及机械臂重复定位是否正常。",
    "axxb_trans": "机械臂位姿与相机观测对不上(平移)。检查标定板固定是否松动，并用尺子复核方格实测边长。",
    "board_origin_spread": "标定板相对基座不固定。腕部相机标定要求板全程不动 —— 确认它没被碰到、底座没有滑移。",
    # eye-to-hand:可观测性(法兰侧 + 相机侧)+ 刚性
    "observability_flange_axes": "示教姿态只绕单一轴在转。重新示教，让手腕至少绕两条不同的轴明显转动。",
    "observability_flange_sep": "手腕的几条旋转轴方向太接近。换成绕明显不同方向翻转的姿态，别只在一个平面里转。",
    "observability_flange_rot": "机械臂整体旋转幅度不足。加大姿态之间的角度差(建议最大相对旋转超过 20°)。",
    "observability_flange_trans": "机械臂末端平移范围太小。把姿态拉开到工作区的前后、左右、远近。",
    "observability_camera_axes": "标定板相对相机的运动太单一。增加板的倾斜与旋转，并确认板被夹紧、没有相对夹爪松动。",
    "observability_camera_sep": "标定板的旋转方向太集中。让板绕不同方向翻转，而不是一直朝同一边倾。",
    "observability_camera_rot": "标定板在画面里的旋转幅度不足。加大板的翻转角度。",
    "observability_camera_trans": "标定板在画面里的位置太集中。让板走遍视野中央与四周，并改变与相机的距离。",
    "observability_duplicates": "有多个几乎重复的姿态。删掉重复的，换成差异更大的姿态重新采集。",
    "target_consistency": "板与法兰之间不够刚性，或板尺寸参数与实物不符。检查夹持是否松动，并用尺子复核方格实测边长。",
}


def _board_input(control: Any, *, width: str = "grow") -> Any:
    """给板参数输入框接上防抖,并返回它本身以便链式赋值。

    变更回调绑在值本身而不是 ``blur``:点上下箭头调数不会让输入框失焦,只绑 blur 的话
    预览要等用户去点一下别处才更新。防抖是为了避开键入过程中的中间态 —— 输入 "20"
    会先经过 "2",那一瞬间板参数是非法的,不防抖就会闪一下红字。
    """
    return control.props(f"debounce={_INPUT_DEBOUNCE_MS}").classes(width)


def _is_placeholder(value: Any) -> bool:
    """配置项是否还是模板占位。

    随包配置里用 ``<your-camera-serial>`` 这类尖括号占位提示用户填写;原样显示会让人
    以为已经配好了,连接时才失败。
    """
    if not value:
        return True
    text = str(value).strip()
    return not text or (text.startswith("<") and text.endswith(">"))


def _hardware_row(name: str, value: str, *, muted: bool = False) -> None:
    """一行「名称 值」。对齐的键值行比 `|` 拼出来的长句好扫。"""
    with ui.row().classes("items-center gap-2 no-wrap"):
        ui.label(name).classes("text-sm text-gray-500 w-24 shrink-0")
        label = ui.label(value).classes("text-sm")
        if muted:
            label.classes("text-gray-700")


def _hardware_row_link(name: str, value: str, link: str, on_click: Callable[[], None]) -> None:
    """同 ``_hardware_row``,但值用告警配色,并把其中 ``link`` 那一段(而非整句)做成可点的链接。"""
    head, _, tail = value.partition(link)
    with ui.row().classes("items-center gap-2 no-wrap"):
        ui.label(name).classes("text-sm text-gray-500 w-24 shrink-0")
        with ui.row().classes("items-center gap-0 no-wrap"):
            ui.label(head).classes("text-sm").style(_WARN_COLOR)
            anchor = ui.label(link).classes("text-sm cursor-pointer underline").style("color:#1d4ed8;")
            anchor.on("click", lambda _e: on_click())
            ui.label(tail).classes("text-sm").style(_WARN_COLOR)


def _warn_line(text: str) -> None:
    """一条"动手之前要知道"的提示。同一类提示只有这一种长相。"""
    ui.label("⚠️ " + text).classes("text-sm").style(_WARN_COLOR)


def _blur_active_element() -> None:
    """把焦点从刚点过的按钮上挪开。

    焦点粘在按钮上时 ``ui.keyboard`` 会忽略按键(``ignore`` 默认含 button),而空格/回车
    又会被浏览器当成"再点一次这个按钮"——鼠标点过一次之后快捷键就全乱了。
    """
    try:
        ui.run_javascript("document.activeElement?.blur()")
    except Exception as exc:
        logger.debug("失焦请求未发出(没有已连接的客户端): %s", exc)


def _key_button(label: str, on_click: Callable[[], None]) -> Any:
    """示教动作按钮:点完顺手失焦,让键盘快捷键继续可用。"""

    def handle() -> None:
        _blur_active_element()
        on_click()

    return ui.button(label, on_click=handle)


@contextmanager
def _step_card(number: str, title: str):
    """一张带序号的步骤卡。

    准备阶段的三件事(做板 / 打印装夹 / 开工前确认)之间隔着离开电脑去打印的中断,
    平铺加分隔线会糊成一片。用卡片把每件事围起来,序号做成实心圆角标,一眼能数清
    还剩几步。
    """
    with ui.card().classes("w-full gap-2").style("border:1px solid #e5e7eb; box-shadow:none;"):
        with ui.row().classes("items-center gap-2"):
            ui.label(number).classes("text-sm font-bold").style(
                "background:#1e40af; color:#fff; width:22px; height:22px; border-radius:11px;"
                " display:flex; align-items:center; justify-content:center;"
            )
            ui.label(title).classes("text-base font-bold")
        with ui.column().classes("w-full gap-2 pl-8"):
            yield


class CalibrationView:
    """「手眼标定」工具面板。由 ``ToolsView`` 在工具列表中挂载。"""

    def __init__(self, state: AppState, *, on_open_config: Callable[[str], None] = lambda _path: None) -> None:
        """构建四步向导并挂上事件轮询定时器;``on_open_config`` 跳到「配置」页的某个点分路径。"""
        self._state = state
        self._on_open_config = on_open_config
        self._engine: CalibrationEngine | None = None
        self._setup: CalibrationSetup | None = None
        self._step = "prepare"
        self._board = BoardParams()
        self._waypoints = 0
        self._teaching_done = False
        # 收尾需二次触发:第一次只警告,避免误触提前结束示教。
        self._finish_warned = False
        self._stations_ok = 0
        self._result: dict[str, Any] | None = None
        self._pdf_path: Path | None = None
        self._space_at = 0.0
        self._paused = False
        self._can_pause = False
        self._teach_running_msg = ""
        self._saved_path: Path | None = None

        self._dispatch: dict[str, Callable[[Any], None]] = {
            "preview": self._on_preview,
            "detect": self._on_detect,
            "waypoint": self._on_waypoint,
            "teach_mode": self._on_teach_mode,
            "phase": self._on_phase,
            "log": self._on_log,
            "station": self._on_station,
            "paused": self._on_paused,
            "aborted": self._on_aborted,
            "done": self._on_done,
            "error": self._on_error,
            "gap": self._on_gap,
        }
        self._panels: dict[str, Any] = {}
        self._build()
        # repeating=False:按住空格不放不该被当成"连按两下"。
        ui.keyboard(on_key=self._on_key, repeating=False)
        ui.timer(0.1, self._drain)

    # ================================================================== 布局
    def _build(self) -> None:
        with ui.column().classes("w-full gap-3"):
            with ui.row().classes("w-full items-center gap-3"):
                ui.label("手眼标定").classes("text-lg font-bold")
                ui.space()
                self._body_label = ui.label("").classes("text-sm text-gray-500")

            self._blocker = ui.label().classes("w-full").style(_banner_style("warn"))
            self._blocker.set_visibility(False)

            # 步骤条可点回退:走到后面几步才发现板参数填错或姿态不够时,不该只能从头再来。
            # 只允许往回点,往前必须走各步自己的出口(它们带着前置校验)。
            self._stepper_row = ui.row().classes("w-full gap-2 no-wrap")
            self._step_chips: dict[str, Any] = {}
            with self._stepper_row:
                for index, (key, label) in enumerate(_STEPS, start=1):
                    chip = ui.label(f"{index}. {label}").classes("text-sm px-2 py-1 rounded")
                    chip.on("click", lambda _e, k=key: self._goto_back(k))
                    self._step_chips[key] = chip

            for key, _label in _STEPS:
                panel = ui.column().classes("w-full gap-2")
                self._panels[key] = panel
            with self._panels["prepare"]:
                self._build_prepare()
            with self._panels["teach"]:
                self._build_teach()
            with self._panels["capture"]:
                self._build_capture()
            with self._panels["result"]:
                self._build_result()
        self._show_step("prepare")
        self.refresh()

    # ------------------------------------------------------------ 第 0 步:准备
    def _build_prepare(self) -> None:
        with _step_card("1", "做一块标定板"):
            # 预览在左、参数在右:预览是竖长的,放右边会在它左下方留出一块空白。
            with ui.row().classes("w-full items-stretch gap-6 no-wrap"):
                self._board_preview = (
                    ui.image().classes("rounded border shrink-0 self-start").style("width:180px; background:#fff;")
                )
                # 绝大多数人直接下载默认板,所以默认视图只有「下载 + 怎么打印」;
                # 板型格数这些收起来,想改的人才展开。顶部对齐,按钮与预览图上沿齐平。
                with ui.column().classes("gap-3 grow"):
                    self._btn_pdf = ui.button("下载标定板 PDF", on_click=self._make_pdf).props("color=primary")
                    with ui.column().classes("gap-1"):
                        ui.label("打印时选「实际大小 / 100%」，不要选「适应页面」。").classes("text-sm text-gray-500")
                        self._pdf_note = ui.label("").classes("text-sm text-gray-500")
                    # 折叠区放在最后:收起时它下面没有别的东西,空白就落在卡片底部,而不是
                    # 在两段文字之间豁开一道。用 label + 可见性切换而不是 ui.expansion ——
                    # 后者的 header 自带左右 padding,文字会比同列其它说明缩进一截。
                    self._params_toggle = ui.label("").classes("text-sm text-gray-600 cursor-pointer select-none")
                    self._params_toggle.on("click", self._toggle_params)
                    # 两行输入等宽、各自平分:三个控件和两个控件的行不等长时,右边缘
                    # 参差不齐,一眼看过去像是分了好几组。
                    self._params_box = ui.column().classes("gap-3").style(f"width:{_FORM_WIDTH_PX}px")
                    with self._params_box:
                        with ui.row().classes("items-end gap-3 no-wrap w-full"):
                            self._in_kind = ui.select(
                                {"charuco": "ChArUco(推荐)", "chessboard": "棋盘格"},
                                value="charuco",
                                label="板型",
                                on_change=lambda _e: self._on_board_change(),
                            ).classes("grow")
                            self._in_sx = _board_input(
                                ui.number("横向格数", value=5, min=3, max=20, format="%d", on_change=self._board_edited)
                            )
                            self._in_sy = _board_input(
                                ui.number("纵向格数", value=7, min=3, max=20, format="%d", on_change=self._board_edited)
                            )
                        with ui.row().classes("items-end gap-3 no-wrap w-full"):
                            self._in_square = _board_input(
                                ui.number(
                                    "方格边长 mm",
                                    value=20.0,
                                    min=5,
                                    max=100,
                                    step=0.1,
                                    on_change=self._square_size_edited,
                                )
                            )
                            self._in_marker = _board_input(
                                ui.number(
                                    "marker 边长 mm",
                                    value=15.0,
                                    min=1,
                                    max=100,
                                    step=0.1,
                                    on_change=self._board_edited,
                                )
                            )
                        # 参数摘要只在有问题时出现:正常时它只是把上面几个输入框念一遍。
                        self._board_note = ui.label("").classes("text-sm")
                        self._board_note.set_visibility(False)
                    self._sync_params_toggle()

        with _step_card("2", "测量方格边长"):
            ui.label("纸上标尺应为 50 mm，使用默认方格边长即可。否则需要重新实测方格边长。").classes(
                "text-sm text-gray-600"
            )
            # 量单个方格误差太大(边界上有 marker,尺子不好对齐),量一整排再除更准。
            self._measure_hint = ui.label("").classes("text-sm text-gray-600")
            self._in_measured = _board_input(
                ui.number("实测方格边长 mm", value=20.0, min=1, step=0.01, on_change=self._measured_edited),
                width="w-44",
            )

        with _step_card("3", "开始前确认"):
            # 板装夹爪还是装桌面取决于相机安装方式,在 _render_hardware 里按 mount 填。
            self._board_setup_label = ui.label("").classes("text-sm text-gray-600")
            with ui.column().classes("gap-0"):
                # 文案随相机安装方式变(见 _render_hardware);先给通用说法,免得还没选
                # 本体时显示一个没有文字的勾选框。
                self._ck_board = ui.checkbox("标定板已固定，板面平整")
                self._ck_estop = ui.checkbox("急停可达，工作区内无障碍物与人")
            for box in (self._ck_board, self._ck_estop):
                box.on("update:model-value", lambda _e: self._refresh_prepare_gate())
            self._btn_to_teach = ui.button("下一步:示教姿态", on_click=lambda: self._goto("teach")).props(
                "color=primary"
            )
            self._btn_to_teach.disable()

    # ------------------------------------------------------------ 第 1 步:示教
    def _build_teach(self) -> None:
        self._back_to_prepare = ui.button("← 上一步", on_click=lambda: self._goto("prepare")).props("flat dense")
        # 相机信息放在这一步:它要回答的是"画面该出现什么"、"取不到画面时查哪里",
        # 而这两个问题都是站在机械臂前面摆姿态时才会问的。
        self._hw_rows = ui.column().classes("gap-1")
        ui.label(
            f"摆 {_RECOMMENDED_WAYPOINTS} 个左右的姿态(至少 {_MIN_WAYPOINTS} 个):"
            "让手腕绕两条以上不同的轴转动，板走遍画面中央与四周、远近都要有，别重复相似的姿态。"
        ).classes("text-sm text-gray-600")

        with ui.row().classes("w-full no-wrap gap-4"):
            with ui.column().classes("gap-1"):
                self._teach_image = (
                    ui.interactive_image(size=(640, 480)).classes("rounded").style("width:520px; background:#111;")
                )
                self._detect_label = ui.label("等待相机画面…").classes("text-sm")
            with ui.column().classes("grow gap-2"):
                with ui.card().classes("w-full gap-1"):
                    ui.label("已记录姿态").classes("font-bold")
                    self._wp_label = ui.label("0").classes("text-2xl font-mono")
                    self._wp_hint = ui.label("").classes("text-sm text-gray-600")
                self._build_teach_actions()
        self._teach_status = ui.label("").classes("text-sm text-gray-500")

    def _build_teach_actions(self) -> None:
        """动作区:示教的几个状态各占一块,由 ``_set_teach_stage`` 挑一块显示。

        ``paused`` 没有自己那一块 —— 它复用 ``teaching`` 的,只换掉其中一个按钮并亮出警告条,
        这样暂停前后整排按钮不会跳位置。
        """
        self._teach_stages = {key: ui.column().classes("w-full gap-2") for key in ("idle", "busy", "armed", "teaching")}
        with self._teach_stages["idle"]:
            ui.button("开始示教", on_click=self._start_teaching).props("color=primary")
        with self._teach_stages["busy"]:
            ui.spinner(size="2em")
        with self._teach_stages["armed"]:
            # 示教方式(松不松力矩)取决于设备实现了哪个端口,连上才知道。所以这里先留空,
            # 等引擎的 teach_mode 事件到了再填;该事件严格早于力矩发生变化,危险提示因此
            # 仍然在"按下去会掉"之前出现。
            self._teach_hint = ui.label("").classes("w-full")
            self._btn_teach_confirm = _key_button("确认，开始示教（双击空格）", self._confirm_teaching).props(
                "color=negative"
            )
            self._teach_key_hint = ui.label("").classes("text-xs text-amber-700")
        with self._teach_stages["teaching"]:
            self._pause_hint = ui.label(_RESUME_WARNING).classes("w-full").style(_DANGER_STYLE)
            # 暂停与继续二选一显示,所以这一行任何时候都是四个按钮。
            with ui.row().classes("items-center gap-1"):
                self._btn_record = _key_button("记录当前姿态（Enter）", self._record).props("flat dense")
                self._btn_pause = _key_button("暂停（空格）", self._pause_teaching).props("flat dense")
                self._btn_resume = _key_button("继续示教（双击空格）", self._resume_teaching).props("flat dense")
                self._btn_teach_done = _key_button("完成示教（S）", self._finish_teaching).props("flat dense")
                _key_button("中止（Esc）", self._abort_teaching).props("flat dense")
            self._pause_key_hint = ui.label("").classes("text-xs text-amber-700")
        self._set_teach_stage("idle")

    def _set_teach_stage(self, stage: str) -> None:
        """切到示教的某个状态:只显示它那块动作区,并清掉半截的双击 / 中止确认状态。"""
        self._space_at = 0.0
        self._abort_warned = False
        self._paused = stage == "paused"
        for hint in (self._teach_key_hint, self._pause_key_hint):
            hint.set_text("")
            hint.set_visibility(False)
        visible_panel = "teaching" if self._paused else stage
        for key, panel in self._teach_stages.items():
            panel.set_visibility(key == visible_panel)
        self._pause_hint.set_visibility(self._paused)
        self._btn_pause.set_visibility(not self._paused and self._can_pause)
        self._btn_resume.set_visibility(self._paused)
        # 暂停期间只放行一次记录:机械臂被托住不动,再记只会得到同一个姿态。
        self._paused_recorded = False
        self._btn_record.enable()

    def _teach_stage(self) -> str:
        """当前摆在动作区里的是哪一块;``paused`` 是 ``teaching`` 的子状态。"""
        if self._teach_stages["teaching"].visible:
            return "paused" if self._paused else "teaching"
        return next((key for key, panel in self._teach_stages.items() if panel.visible), "")

    # ------------------------------------------------------------ 第 2 步:采集
    def _build_capture(self) -> None:
        """采集这一步的两个状态:没开跑(参数 + 开始)和跑起来了(进度)。"""
        self._back_to_teach = ui.button("← 重新示教", on_click=lambda: self._goto("teach")).props("flat dense")
        ui.label("⚠️ 机械臂会自动沿示教轨迹移动并逐个拍照，开始前请离开工作区。").classes("w-full").style(_DANGER_STYLE)

        self._capture_stages = {key: ui.column().classes("w-full gap-2") for key in ("idle", "running")}
        with self._capture_stages["idle"]:
            with ui.row().classes("w-full items-end gap-6"):
                self._in_stations = ui.number(
                    "拍照次数", value=20, min=3, max=60, format="%d", on_change=lambda _e: self._refresh_capture_note()
                ).classes("w-32")
                self._in_corners = ui.number("每张照片最少角点数", value=16, min=6, max=100, format="%d").classes(
                    "w-44"
                )
            self._capture_note = ui.label("").classes("text-sm text-gray-600")
            self._btn_run = ui.button("开始采集", on_click=self._start_capture).props("color=negative")
        with self._capture_stages["running"]:
            self._capture_progress = ui.linear_progress(value=0.0, show_value=False).classes("w-full")
            self._progress_label = ui.label("").classes("text-sm")

        self._capture_status = ui.label("").classes("text-sm")
        with ui.expansion("运行日志").classes("w-full text-sm"):
            self._log_box = ui.log(max_lines=200).classes("w-full h-64 text-xs")
        self._set_capture_stage("idle")

    def _set_capture_stage(self, stage: str) -> None:
        for key, panel in self._capture_stages.items():
            panel.set_visibility(key == stage)

    # ------------------------------------------------------------ 第 3 步:结果
    def _build_result(self) -> None:
        self._result_banner = ui.label("").classes("w-full text-base")
        self._result_detail = ui.column().classes("w-full gap-1")
        self._build_save_box()
        self._apply_row = ui.column().classes("w-full gap-1")
        with self._apply_row:
            self._btn_apply = ui.button("启用这份标定", on_click=self._apply_to_config).props("color=primary")
            self._apply_note = ui.label("").classes("text-sm text-gray-600")
        with ui.row().classes("gap-2"):
            ui.button("← 重新采集", on_click=lambda: self._goto("capture")).props("flat")
            ui.button("重新标定", on_click=lambda: self._goto("prepare")).props("flat")

    def _build_save_box(self) -> None:
        """保存位置 + 两个保存按钮。求解产物停在工作区暂存目录,这两颗按钮是它进入用户目录的唯一入口。"""
        self._save_box = ui.column().classes("w-full gap-2")
        with self._save_box:
            self._in_save_path = ui.input("保存位置", on_change=lambda _e: self._refresh_save_gate()).classes("w-full")
            self._save_warns = ui.column().classes("w-full gap-0")
            with ui.row().classes("items-center gap-2"):
                self._btn_save = ui.button("保存", on_click=lambda: self._save_result(backup=False)).props(
                    "color=primary"
                )
                self._btn_save_backup = ui.button(
                    "备份同名文件后保存", on_click=lambda: self._save_result(backup=True)
                ).props("color=primary")
            self._save_note = ui.label("").classes("text-sm text-gray-600")
        self._save_box.set_visibility(False)

    def _refresh_save_gate(self) -> None:
        """重算保存前要交代的事:先说整份结果的状态,再说这个位置上的状态。"""
        target = self._save_target()
        self._save_warns.clear()
        with self._save_warns:
            if self._saved_path is None:
                _warn_line("结果还没写盘，点「保存」后才能启用。")
            if target is not None and target.exists():
                _warn_line(f"{target} 已存在，点「保存」会覆盖它。")
            if target is None:
                _warn_line("请填写保存位置。")
        self._btn_save.set_enabled(target is not None)
        self._btn_save_backup.set_enabled(target is not None and target.exists())

    def _save_target(self) -> Path | None:
        """输入框里的位置解析成绝对路径;相对路径按本体配置所在目录算,与 ``calib_path`` 同基准。"""
        text = str(self._in_save_path.value or "").strip()
        if not text:
            return None
        path = Path(text).expanduser()
        return path if path.is_absolute() else self._config_path().parent / path

    def _save_result(self, *, backup: bool) -> None:
        """把暂存的求解产物复制到用户指定位置。"""
        result = self._result or {}
        pending = result.get("artifact_path")
        target = self._save_target()
        if not pending or target is None:
            return
        try:
            backed_up = _copy_result(Path(str(pending)), target, backup=backup)
        except OSError as exc:
            logger.exception("保存标定结果失败")
            ui.notify(f"保存失败:{exc}", type="negative")
            return
        self._saved_path = target
        self._btn_apply.enable()
        note = f"已保存到 {target}"
        # 备份文件名是回滚时唯一的线索,不能被随后的「已写入配置」回执顶掉,所以各用各的标签。
        self._save_note.set_text(f"{note}（原文件已备份为 {backed_up}）" if backed_up else note)
        self._refresh_save_gate()

    def _config_path(self) -> Path:
        """当前生效的本体配置文件路径。"""
        if self._state.current_config_file:
            return Path(self._state.current_config_file)
        return registry.get_body(str(self._state.current_body)).config_path()

    # ================================================================== 步骤切换
    def _show_step(self, key: str) -> None:
        self._step = key
        for name, panel in self._panels.items():
            panel.set_visibility(name == key)
        order = [step for step, _ in _STEPS]
        current = order.index(key)
        for index, (name, _label) in enumerate(_STEPS):
            chip = self._step_chips[name]
            if name == key:
                chip.style("background:#dbeafe; color:#1e40af; font-weight:600; cursor:default;")
            elif index < current:
                chip.style("background:#f3f4f6; color:#374151; font-weight:400; cursor:pointer;")
            else:
                chip.style("background:#f3f4f6; color:#9ca3af; font-weight:400; cursor:default;")

    def _goto_back(self, key: str) -> None:
        """点步骤条回到已经走过的步骤;点当前或未到达的步骤无效。"""
        order = [step for step, _ in _STEPS]
        if order.index(key) >= order.index(self._step):
            return
        self._goto(key)

    def _goto(self, key: str) -> None:
        if key == "prepare":
            self._reset()
        elif key == "teach":
            if not self._prepare_ready():
                ui.notify("请先确认标定板已夹紧、工作区安全。", type="warning")
                return
            # 回到示教就要重来一轮:停掉可能仍连着的会话,并清空上一轮的姿态计数。
            self._restart_teaching()
        self._show_step(key)

    def _restart_teaching(self) -> None:
        self.stop(wait=True)
        self._waypoints = 0
        self._teaching_done = False
        # 收尾需二次触发:第一次只警告,避免误触提前结束示教。
        self._finish_warned = False
        self._wp_label.set_text("0")
        self._wp_hint.set_text("")
        self._teach_status.set_text("")
        self._detect_label.set_text("等待相机画面…")
        self._detect_label.style("color:inherit;")
        self._set_teach_stage("idle")
        self._set_capture_stage("idle")
        self._finish_warned = False

    def _reset(self) -> None:
        self.stop(wait=True)
        self._waypoints = 0
        self._teaching_done = False
        # 收尾需二次触发:第一次只警告,避免误触提前结束示教。
        self._finish_warned = False
        self._result = None
        self._saved_path = None
        self._wp_label.set_text("0")
        self._wp_hint.set_text("")
        self._set_capture_stage("idle")
        self._btn_apply.disable()
        self._save_box.set_visibility(False)
        self._capture_status.set_text("")
        self._teach_status.set_text("")

    # ================================================================== 前置条件
    def refresh(self) -> None:
        """进入页面或状态变化时重算前置条件(选了哪个本体、该本体是否接入标定)。"""
        body_key = self._state.current_body
        if body_key is None:
            self._blocker.set_text("⚠️ 请先在主页选择一个本体 —— 标定针对具体机器人，与任务无关。")
            self._blocker.set_visibility(True)
            self._body_label.set_text("")
            return
        body = registry.get_body(body_key)
        self._body_label.set_text(f"本体:{body.display_name}")
        if registry.calibration_profile(body_key) is None:
            self._blocker.set_text(
                f"⚠️ 本体「{body.display_name}」尚未接入标定向导"
                "(缺少内置标定档案:轨迹空间与安全放宽项)。请使用命令行标定。"
            )
            self._blocker.set_visibility(True)
            return
        missing = board_print_dependency_hint()
        if missing is not None:
            self._blocker.set_text("⚠️ " + missing)
            self._blocker.set_visibility(True)
            return
        self._blocker.set_visibility(False)
        self._render_hardware(body_key)
        self._on_board_change()
        self._refresh_prepare_gate()

    def _resolve_mount(self, body_key: str) -> str:
        """本次标定的相机安装方式;运行配置是权威,内置档案只是兜底显示值。

        现有两个适配器的 config 里 ``camera_mount`` 是 ``Literal``,写不写都改不动它;
        但优先级仍以配置为准,这样日后出现同时支持两种安装方式的本体时,改配置就能切,
        不必回来动界面。配置缺这一项时读档案,而不是显示一个让人以为填错了的问号。
        """
        low_level = self._low_level(body_key)
        profile = registry.calibration_profile(body_key) or {}
        return str(low_level.get("camera_mount") or profile.get("camera_mount") or "")

    def _render_hardware(self, body_key: str) -> None:
        """列出这次标定要用的硬件,并按相机安装方式说明标定板该装在哪。"""
        mount = self._resolve_mount(body_key)
        serial = self._low_level(body_key).get("camera_serial")

        self._board_setup_label.set_text(_BOARD_SETUP_TEXT.get(mount, "把打印件贴在平整硬板上并固定好。"))
        self._ck_board.set_text(_BOARD_CHECK_TEXT.get(mount, "标定板已固定，板面平整"))

        self._hw_rows.clear()
        with self._hw_rows:
            _hardware_row("相机位置", _MOUNT_TEXT.get(mount, "未知"), muted=True)
            if _is_placeholder(serial):
                _hardware_row_link(
                    "相机序列号",
                    "还没填，先到「配置」页填上，否则取不到画面",
                    "「配置」",
                    lambda: self._on_open_config(_CAMERA_SERIAL_PATH),
                )
            else:
                _hardware_row("相机序列号", str(serial), muted=True)

    def _low_level(self, body_key: str) -> dict[str, Any]:
        config = self._config_data(body_key)
        env = config.get("env")
        cfg = env.get("cfg") if isinstance(env, dict) else None
        low_level = cfg.get("low_level") if isinstance(cfg, dict) else None
        return low_level if isinstance(low_level, dict) else {}

    def _config_data(self, body_key: str) -> dict[str, Any]:
        """取该本体当前生效的配置 dict(标定与任务无关,任务未选时也要能拿到)。"""
        task_key = self._state.current_task
        if task_key is not None:
            return self._state.config_for(body_key, task_key).data
        from jiuwensymbiosis_gui.workbench.config_model import ConfigModel

        path = (
            Path(self._state.current_config_file)
            if self._state.current_config_file
            else registry.get_body(body_key).config_path()
        )
        try:
            return ConfigModel.from_yaml_text(path.read_text(encoding="utf-8")).data
        except (OSError, ValueError) as exc:
            logger.warning("标定读取本体配置 %s 失败: %s", path, exc)
            return {}

    def _prepare_ready(self) -> bool:
        return bool(self._ck_board.value and self._ck_estop.value and self._pdf_ok())

    def _pdf_ok(self) -> bool:
        """板参数合法即可进入下一步 —— 用户可能早就有一块板,不强制生成 PDF。"""
        try:
            self._board_spec().validate()
        except BoardParamsError:
            return False
        return True

    def _refresh_prepare_gate(self) -> None:
        if self._prepare_ready():
            self._btn_to_teach.enable()
        else:
            self._btn_to_teach.disable()

    # ================================================================== 第 0 步动作
    def _board_spec(self) -> BoardParams:
        """要**画出来**的板规格:全部取界面设定值。

        预览与 PDF 只看这一份。实测值绝不能倒灌回来 —— 否则量完一改,图跟着变,再打
        一次又是另一个尺寸,越修越偏。
        """
        return BoardParams(
            kind=str(self._in_kind.value),
            squares_x=int(self._in_sx.value or 0),
            squares_y=int(self._in_sy.value or 0),
            square_size_mm=float(self._in_square.value or 0),
            marker_size_mm=float(self._in_marker.value or 0),
        )

    def _measured_board(self) -> BoardParams:
        """标定**实际使用**的板规格:按实测边长等比缩放。

        打印缩放是等比的 —— 方格小了多少,marker 就小了多少。只改 square 会凭空改变
        marker/square 比例:既让求解用错 marker 尺寸,又会误报「marker 不能大于方格」。
        """
        spec = self._board_spec()
        measured = self._in_measured.value
        if not measured or float(measured) <= 0 or not spec.square_size_mm:
            return spec
        scale = float(measured) / spec.square_size_mm
        return dataclasses.replace(
            spec,
            square_size_mm=float(measured),
            marker_size_mm=spec.marker_size_mm * scale,
        )

    def _toggle_params(self) -> None:
        """展开/收起标定板参数。"""
        self._params_box.set_visibility(not self._params_box.visible)
        self._sync_params_toggle()

    def _sync_params_toggle(self) -> None:
        self._params_toggle.set_text(("▾ " if self._params_box.visible else "▸ ") + "修改标定板参数")

    def _board_edited(self, _event: Any = None) -> None:
        """板参数输入框的变更回调(事件参数用不上)。"""
        self._on_board_change()

    def _measured_edited(self, _event: Any = None) -> None:
        """实测值只喂给标定,不参与画图 —— 所以这里不重绘预览,只更新「默认」标记。"""
        self._mark_measured_default()

    def _mark_measured_default(self) -> None:
        """实测值还等于设定值时,在标签上标「默认」。

        标定用的是实测值,所以要让人一眼看出这个数是量过的还是照抄的默认值 —— 两者
        数值相同但含义完全不同。标在 label 而不是 Quasar 的 ``suffix``:后者是 input
        的兄弟节点,只会落在数字输入框的上下箭头右边,离数字很远。
        """
        nominal = self._in_square.value
        measured = self._in_measured.value
        is_default = nominal is not None and measured is not None and float(measured) == float(nominal)
        self._in_measured.props(f'label="实测方格边长 mm{"（默认）" if is_default else ""}"')

    def _square_size_edited(self, _event: Any = None) -> None:
        """改了设定边长,实测值跟着重置 —— 新尺寸还没打印,更谈不上量过。"""
        self._in_measured.value = self._in_square.value
        self._on_board_change()

    def _on_board_change(self) -> None:
        self._in_marker.set_visibility(str(self._in_kind.value) == "charuco")
        # 「默认」标记只看两个数值相不相等,与板参数合不合法无关 —— 放在校验之前,
        # 免得参数一报错就卡在上一次的状态不再更新。
        self._mark_measured_default()
        try:
            board = self._board_spec()
            board.validate()
        except BoardParamsError as exc:
            self._board_note.set_text(str(exc))
            self._board_note.style("color:#b91c1c;")
            self._board_note.set_visibility(True)
            self._board_preview.set_source("")
            self._refresh_prepare_gate()
            return
        # 沿更长的那条边量,格子多、相对误差小。
        longest = max(board.squares_x, board.squares_y)
        self._measure_hint.set_text(f"实测方法:量出 {longest} 个格子的总长，除以 {longest} 得到单个方格边长。")
        notes = board.warnings()
        # 参数没问题就什么都不说:输入框和左侧预览已经把结果摆在那儿了。
        self._board_note.set_visibility(bool(notes))
        if notes:
            self._board_note.set_text("　".join(notes))
            self._board_note.style("color:#b45309;")
        try:
            self._board_preview.set_source(board_print.render_preview_data_uri(board))
        except Exception as exc:
            # 预览渲染不出来就说明板也生成不出来,不能只留一个空白框让人猜。
            logger.exception("标定板预览渲染失败")
            self._board_note.set_text(f"标定板预览生成失败:{exc}")
            self._board_note.style("color:#b91c1c;")
            self._board_note.set_visibility(True)
            self._board_preview.set_source("")
        self._refresh_prepare_gate()

    def _make_pdf(self) -> None:
        """生成 PDF 并交给浏览器下载。

        落盘只是为了把文件交给 ``ui.download``,所以写临时目录、也不把这个路径显示
        出来 —— 界面可能开在另一台机器上,服务器磁盘上的路径对用户没有意义,他要的
        是浏览器下载目录里的那份。
        """
        name = "calibration_board.pdf"
        try:
            board = self._board_spec()
            out = board_print.write_board_pdf(board, Path(tempfile.gettempdir()) / name)
        except BoardParamsError as exc:
            ui.notify(str(exc), type="warning")
            return
        except Exception as exc:
            logger.exception("生成标定板 PDF 失败")
            ui.notify(f"生成失败:{exc}", type="negative")
            return
        self._pdf_path = out
        self._pdf_note.set_text(f"已下载 {name}")
        ui.download.file(out, name)

    # ================================================================== 第 1 步动作
    def _start_teaching(self) -> None:
        engine = self._new_engine()
        if engine is None:
            return
        self._waypoints = 0
        self._wp_label.set_text("0")
        self._set_teach_stage("busy")
        self._teach_status.set_text("正在连接…")
        self._teach_status.style("color:inherit;")  # 抹掉上一轮中止/失败留下的颜色
        self._state.host.start_tool(engine, engine.start_teaching)

    def _confirm_teaching(self) -> None:
        """用户已读过本机型的示教指引:放行工作线程,并换上记录/完成两个动作。"""
        if self._engine is None:
            return
        self._set_teach_stage("teaching")
        self._engine.confirm_teaching()

    def _on_key(self, e: Any) -> None:
        """示教期间的键盘操作。

        只在示教的三个可交互阶段、且工作线程确实还在等命令时才认;``ui.keyboard`` 默认
        忽略输入框里的按键,所以在采集数量那类数字框里敲空格不会误触。
        """
        if not e.action.keydown or e.action.repeat:
            return
        stage = self._teach_stage()
        if stage not in _KEY_STAGES or self._engine is None or not self._engine.is_running():
            return
        if e.key.space:
            self._on_space(stage)
        elif e.key.escape:
            self._abort_teaching()
        elif e.key.enter and stage in ("teaching", "paused"):
            self._record()
        elif str(e.key.name).lower() == "s" and stage in ("teaching", "paused"):
            self._finish_teaching()

    def _on_space(self, stage: str) -> None:
        """空格 = 改变机械臂的软硬。变软的方向(确认 / 继续)要双击。"""
        if stage == "teaching":
            self._pause_teaching()
        elif stage == "armed":
            if self._double_tapped(self._teach_key_hint):
                self._confirm_teaching()
        elif self._double_tapped(self._pause_key_hint):
            self._resume_teaching()

    def _double_tapped(self, hint: Any) -> bool:
        """双击判定;第一下只是上膛,并把提示改成"再按一下"。"""
        now = time.monotonic()
        if now - self._space_at > _DOUBLE_TAP_S:
            self._space_at = now
            hint.set_text(_KEY_HINT_ARMED)
            hint.set_visibility(True)
            return False
        return True

    def _pause_teaching(self) -> None:
        """请求托住机械臂。界面等引擎回报 ``paused`` 再改 —— 力矩还没恢复就说"已托住"会害人松手。"""
        if self._engine is not None:
            self._engine.pause_teaching()

    def _resume_teaching(self) -> None:
        """请求重新松力矩。同样等引擎回报,不抢在力矩真的掉之前改界面。"""
        if self._engine is not None:
            self._engine.resume_teaching()

    def _abort_teaching(self) -> None:
        """中止这一轮。已记录过姿态时要按两次,免得一下 Esc 丢掉十几个姿态。"""
        if self._engine is None:
            return
        if self._waypoints and not self._abort_warned:
            self._abort_warned = True
            ui.notify(f"再按一次将丢弃已记录的 {self._waypoints} 个姿态。", type="warning")
            return
        self._set_teach_stage("busy")
        self._teach_status.set_text("正在中止并恢复力矩…")
        self._engine.abort_teaching()

    def _record(self) -> None:
        if self._engine is None or (self._paused and self._paused_recorded):
            return
        if self._paused:
            # 托住不动就只有一个姿态可记,再记是重复项 —— 重复姿态会被可观测性门禁判失败。
            self._paused_recorded = True
            self._btn_record.disable()
        self._engine.record_waypoint()

    def _finish_teaching(self) -> None:
        if self._waypoints < _MIN_WAYPOINTS:
            ui.notify(
                f"只记录了 {self._waypoints} 个姿态，少于建议下限 {_MIN_WAYPOINTS} 个，求解可能不稳定。"
                "要就这样结束，请再按一次。",
                type="warning",
            )
            # 二次触发才真正结束:第一次只是警告,避免误触提前收尾。
            if not self._finish_warned:
                self._finish_warned = True
                return
        if self._engine is not None:
            self._engine.finish_teaching()
        self._set_teach_stage("busy")
        self._teach_status.set_text("正在恢复力矩并写出姿态归档…")

    # ================================================================== 第 2 步动作
    def _start_capture(self) -> None:
        """开始真机采集。

        动机械臂之前的校验由 ``execute_calibration`` 自己做全(轨迹、数据契约、运动能力),
        界面不必先跑一趟 dry-run 再跑真的。
        """
        # 按当前界面值重建,而不是复用示教时那个 engine:拍照次数、角点下限、实测边长
        # 都在示教之后才轮到用户填,沿用旧快照会拿老参数静默跑一遍。
        if self._engine is not None and self._engine.is_running():
            ui.notify("上一步还在进行中，请稍候。", type="warning")
            return
        engine = self._new_engine()
        if engine is None:
            return
        self._set_capture_stage("running")
        self._stations_ok = 0
        self._capture_progress.set_value(0.0)
        self._progress_label.set_text("正在连接…")
        self._capture_status.set_text("")
        self._capture_status.style("color:inherit;")
        self._state.host.start_tool(engine, lambda: engine.start_capture(dry_run=False))

    # ================================================================== 引擎装配
    def _new_engine(self) -> CalibrationEngine | None:
        body_key = self._state.current_body
        if body_key is None:
            ui.notify("请先在主页选择一个本体。", type="warning")
            return None
        profile = registry.calibration_profile(body_key)
        if profile is None:
            ui.notify("该本体尚未接入标定向导。", type="warning")
            return None
        body = registry.get_body(body_key)
        config_source = self._config_path()
        config_data = self._config_data(body_key)
        workspace = Path(self._state.workspace) / "calibration"
        workspace.mkdir(parents=True, exist_ok=True)
        # 产物落在暂存目录而非 configs/:configs/ 下那份可能正被运行配置指着用,换掉它要等
        # 用户在结果页点「保存」。
        pending_dir = workspace / "pending"
        pending_dir.mkdir(parents=True, exist_ok=True)
        mount = self._resolve_mount(body_key)
        out_path = pending_dir / Path(_output_relpath(body.adapter, mount, profile)).name
        setup = CalibrationSetup(
            adapter_module=body.adapter_module(),
            config_data=config_data,
            board=self._measured_board(),
            out_path=out_path,
            waypoint_path=workspace / f"{body.adapter}_waypoints.npz",
            calibration_profile=profile,
            min_corners=int(self._in_corners.value or 16),
            n_stations=int(self._in_stations.value or 20),
        )
        self._setup = setup
        if self._engine is not None:
            self._engine.close()
        admission = MaintenanceOwner(
            self._state.host.runtime(self._state.workspace), config_source, self._state.workspace, adapter=body.adapter
        )
        self._engine = CalibrationEngine(setup, admission=admission)
        return self._engine

    def stop(self, *, wait: bool = False) -> bool:
        """停止后台标定并释放硬件(离开页面 / 重新开始时调用;幂等),返回是否真的停下了。

        停止请求只在示教的取指令循环里被检查:自动采集阶段(``execute_calibration``,
        机械臂正沿轨迹走位)中途停不下来。所以调用方不能假定调用完硬件就归还了 ——
        还在跑时也不 ``close``,那会把工作线程仍在读的临时 profile 删掉。
        """
        engine = self._engine
        if engine is None:
            return True
        if engine.is_running():
            engine.stop()
            if wait:
                engine.join(timeout=5.0)
            if engine.is_running():
                return False
        engine.close()
        return True

    # ================================================================== 事件
    def _drain(self) -> None:
        engine = self._engine
        if engine is None:
            return
        for tag, payload in engine.drain():
            handler = self._dispatch.get(tag)
            if handler is not None:
                handler(payload)

    def _on_preview(self, uri: str) -> None:
        self._teach_image.set_source(uri)

    def _on_detect(self, payload: dict) -> None:
        if payload.get("ok"):
            count = int(payload.get("n_corners", 0))
            self._detect_label.set_text(f"✓ 已检测到标定板({count} 个角点)")
            self._detect_label.style("color:#15803d;")
        else:
            reason = str(payload.get("reason") or "未检测到标定板")
            self._detect_label.set_text(f"✗ {reason}")
            self._detect_label.style("color:#b91c1c;")

    def _on_waypoint(self, payload: dict) -> None:
        self._waypoints = int(payload.get("count", 0))
        self._wp_label.set_text(str(self._waypoints))
        if self._waypoints < _MIN_WAYPOINTS:
            self._wp_hint.set_text(f"还需 {_MIN_WAYPOINTS - self._waypoints} 个")
        elif self._waypoints < _RECOMMENDED_WAYPOINTS:
            self._wp_hint.set_text(f"够用了，补到 {_RECOMMENDED_WAYPOINTS} 个更稳")
        else:
            self._wp_hint.set_text("数量充足")

    def _on_teach_mode(self, payload: dict) -> None:
        """已连接、力矩尚未变化:按设备**实际**的示教方式给出指引,并亮出确认按钮。

        模式来自 ``resolve_teaching_mode``(判据是设备实现了哪个端口),所以不实现
        ``ManualGuidance`` 的本体不会看到"力矩会掉"这种对它而言是假话的警告。
        """
        mode = str(payload.get("mode") or "")
        text, dangerous = _TEACH_MODE_TEXT.get(mode, ("已连接。确认工作区安全后开始示教。", False))
        self._teach_hint.set_text(text)
        self._teach_hint.style(_DANGER_STYLE if dangerous else _INFO_STYLE)
        # 暂停要设备实现了 GuidanceHold 才给得出来;给不出来时不摆一个按下去没反应的按钮。
        self._can_pause = bool(payload.get("can_pause"))
        self._set_teach_stage("armed")
        self._teach_status.set_text("")

    def _on_paused(self, payload: dict) -> None:
        """引擎回报力矩已恢复 / 已再次松开,此时才切换界面说法。"""
        paused = bool(payload.get("paused"))
        self._set_teach_stage("paused" if paused else "teaching")
        self._teach_status.set_text(_PAUSED_STATUS if paused else self._teach_running_msg)

    def _on_aborted(self, payload: dict) -> None:
        """中止:力矩已在退出示教时恢复,这一轮的姿态不写归档,计数清零重来。"""
        logger.info("示教中止: %s", payload.get("reason", ""))
        self._waypoints = 0
        self._wp_label.set_text("0")
        self._wp_hint.set_text("")
        self._set_teach_stage("idle")
        self._teach_status.set_text("已中止，这一轮记录的姿态没有保存。")
        self._teach_status.style("color:#b45309;")

    def _on_phase(self, payload: dict) -> None:
        msg = str(payload.get("msg", ""))
        if payload.get("phase") == "teaching":
            # 记下来:暂停时这句被"示教暂停中…"顶掉,继续时要原样放回。
            self._teach_running_msg = msg
            self._teach_status.set_text(msg)
        elif payload.get("phase") in ("capture", "dry_run"):
            self._capture_status.set_text(msg)
        else:
            self._teach_status.set_text(msg)
            self._capture_status.set_text(msg)

    def _on_log(self, payload: dict) -> None:
        self._log_box.push(f"{payload.get('level', '')} {payload.get('msg', '')}")

    def _on_station(self, payload: dict) -> None:
        """引擎每拍完一个采集点发一次(挂在 ``detect_fn`` 上,见 ``CalibrationEngine._detect_station``)。"""
        index, total = int(payload.get("index", 0)), max(int(payload.get("total", 0)), 1)
        self._stations_ok += int(bool(payload.get("ok")))
        self._capture_progress.set_value(min(index / total, 1.0))
        self._progress_label.set_text(f"第 {index} / {total} 个位置 · 认出标定板 {self._stations_ok} 张")

    def _on_done(self, payload: dict) -> None:
        if payload.get("phase") == "teaching":
            self._on_teaching_done(payload)
        else:
            self._on_capture_done(payload)

    def _refresh_capture_note(self) -> None:
        self._capture_note.set_text(
            f"机械臂会在你示教的 {self._waypoints} 个姿态之间等距取 {int(self._in_stations.value or 20)} 个位置停下拍照。"
            "拍得多结果更稳，也更慢。"
        )

    def _on_teaching_done(self, payload: dict) -> None:
        self._teaching_done = True
        count = int(payload.get("count", 0))
        # 「力矩已恢复」只对真的松过力矩的本体成立。
        restored = "力矩已恢复。" if payload.get("mode") == "manual_guidance" else ""
        self._teach_status.set_text(f"✓ 已记录 {count} 个姿态。{restored}")
        self._teach_status.style("color:#15803d;")
        self._set_teach_stage("idle")
        self._refresh_capture_note()
        self._show_step("capture")

    def _on_capture_done(self, payload: dict) -> None:
        self._result = payload
        self._set_capture_stage("idle")
        self._render_result(payload)
        self._show_step("result")

    def _on_error(self, payload: dict) -> None:
        reason = str(payload.get("reason", ""))
        self._set_capture_stage("idle")
        self._set_teach_stage("idle")
        if payload.get("fatal"):
            self._blocker.set_text(
                "🛑 力矩没能恢复，机械臂可能还是软的。请立刻用手托住，不要再点任何按钮，"
                f"检查电机连线后手动把它放回安全位置。详情:{reason}"
            )
            self._blocker.style(_banner_style("danger"))
            self._blocker.set_visibility(True)
            return
        self._teach_status.set_text("✗ " + reason)
        self._teach_status.style("color:#b91c1c;")
        self._capture_status.set_text("✗ " + reason)
        self._capture_status.style("color:#b91c1c;")

    def _on_gap(self, payload: dict) -> None:
        dropped = int(payload.get("dropped", 0))
        message = f"界面事件流有缺口，已丢弃 {dropped} 条旧事件。"
        self._teach_status.set_text(message)
        self._capture_status.set_text(message)

    # ================================================================== 结果渲染
    def _render_result(self, payload: dict) -> None:
        self._result_detail.clear()
        candidate = bool(payload.get("candidate"))
        artifact = payload.get("artifact_path")
        self._saved_path = None
        self._btn_apply.disable()
        self._apply_note.set_text("")
        self._save_note.set_text("")
        usable = not candidate and bool(artifact)
        self._save_box.set_visibility(usable)
        # 结果不能用时「启用」永远不会亮,整块收起来,而不是摆一颗点不动的按钮。
        self._apply_row.set_visibility(usable)
        if usable:
            self._result_banner.set_text("✓ 标定成功")
            self._result_banner.style(_banner_style("ok"))
            self._in_save_path.set_value(self._default_save_path())
            self._refresh_save_gate()
        else:
            self._result_banner.set_text("⚠ 这次标定没通过质量检查，结果不能用")
            self._result_banner.style(_banner_style("warn"))
        with self._result_detail:
            self._render_metrics(payload)
            self._render_reasons(payload, artifact)

    def _render_metrics(self, payload: dict) -> None:
        """质量指标 + 好坏判断。

        每一行显示的门限就是这一轮验收实际用的那条线(引擎从 ``acceptance_policy`` 取,
        随 mount 与档案覆盖而变),所以不会出现标着"良好"却被判不通过。
        """
        quality = payload.get("quality") or {}
        limits = payload.get("limits") or {}
        stations = payload.get("n_stations")
        if stations:
            ui.label(f"用了 {stations} 个位置的数据").classes("text-sm text-gray-600")

        reprojection = quality.get("reprojection_px")
        if reprojection:
            self._limit_row(
                "图像识别精度",
                float(reprojection["mean"]),
                "px",
                limit=float(limits.get("reproj_warn_px", _REPROJ_WARN_PX)),
                good=float(limits.get("reproj_good_px", _REPROJ_GOOD_PX)),
            )
        rigidity = quality.get("rigidity")
        if rigidity and "translation_std_mm" in limits:
            self._limit_row(
                "标定板位置波动",
                float(rigidity["translation_std_mm"]),
                "mm",
                limit=float(limits["translation_std_mm"]),
            )
            self._limit_row(
                "标定板角度偏差",
                float(rigidity["rotation_deg"]),
                "°",
                limit=float(limits["rotation_spread_deg"]),
            )

        raw = [template.format(**quality[key]) for key, template in _RAW_METRIC_LINES if quality.get(key)]
        if payload.get("method"):
            raw.insert(0, f"solver method: {payload['method']}")
        if not raw:
            return  # 站点太少时求解没跑,展开了也是空的
        with ui.expansion("技术详情").classes("text-sm w-full"):
            for line in raw:
                ui.label(line).classes("text-xs font-mono text-gray-600")

    @staticmethod
    def _limit_row(name: str, value: float, unit: str, *, limit: float, good: float | None = None) -> None:
        """一行指标:数值 / 合格线 / 判定。``good`` 缺省取合格线的一半,与重投影那两档同构。"""
        text, color = _verdict(value, limit / 2 if good is None else good, limit)
        with ui.row().classes("gap-2 items-center"):
            ui.label(name).classes("text-sm w-32 text-gray-600")
            ui.label(f"{value:.2f} {unit}（限 {limit:g}）").classes("text-sm font-mono w-52")
            ui.label(text).classes("text-sm").style(f"color:{color};")

    @staticmethod
    def _render_reasons(payload: dict, report_path: Any = None) -> None:
        """列出改进建议;求解器的英文原文收进折叠区。

        ``reasons`` 是标定子系统的英文诊断串(如 ``min_axis_separation_deg 8.3 < 15.0``),
        对操作者没有可执行性。主体显示中文建议,英文留给排查时展开。
        """
        reasons = [str(reason) for reason in payload.get("reasons") or []]
        failed = [str(check) for check in payload.get("failed_checks") or []]
        if not reasons and not failed:
            return
        # 去重:一条建议常对应多个检查项(如 target_consistency 的 _trans 与 _rot),
        # 不去重就会把同一句话原样列两遍。
        found = (_remedy_for(check) for check in failed)
        remedies = list(dict.fromkeys(remedy for remedy in found if remedy))
        if remedies:
            ui.label("怎么改进").classes("font-bold mt-2")
            for remedy in remedies:
                ui.label("• " + remedy).classes("text-sm")
        if reasons or report_path:
            # 与指标那块的「技术详情」并列,标题必须区分得开,否则用户点开一个就以为看全了。
            with ui.expansion("未通过的检查项").classes("text-sm w-full"):
                for reason in reasons:
                    ui.label(reason).classes("text-xs font-mono text-gray-600")
                if report_path:
                    ui.label(f"诊断报告:{report_path}").classes("text-xs font-mono text-gray-600")

    def _default_save_path(self) -> str:
        """保存位置的默认值:约定路径去掉本体那一段,基准与 ``calib_path`` 一致。"""
        body_key = str(self._state.current_body)
        body = registry.get_body(body_key)
        profile = registry.calibration_profile(body_key) or {}
        relpath = PurePosixPath(_output_relpath(body.adapter, self._resolve_mount(body_key), profile))
        if relpath.parts and relpath.parts[0] == body.adapter:
            return str(relpath.relative_to(body.adapter))
        return str(relpath)

    # ================================================================== 应用到配置
    def _apply_to_config(self) -> None:
        body_key = self._state.current_body
        if body_key is None or self._saved_path is None:
            return
        config_path = self._config_path()
        artifact = self._saved_path
        try:
            written = _write_calib_path(config_path, artifact)
        except Exception as exc:
            logger.exception("写回标定路径失败")
            ui.notify(f"写入配置失败:{exc}", type="negative")
            return
        self._apply_note.set_text(f"已写入 {config_path}:calib_path = {written}")
        ui.notify("已应用到运行配置。", type="positive")


def _copy_result(pending: Path, target: Path, *, backup: bool) -> str:
    """把暂存的求解产物复制到 ``target``,返回备份文件名(没备份则空串)。

    站点归档跟着标定 JSON 一起搬:它是唯一能重新求解的原始数据,分开放等于丢掉。
    ``backup`` 时先把同名文件改名成 ``<原名>.<时间戳>.bak`` 再写。
    """
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backed_up: list[str] = []
    target.parent.mkdir(parents=True, exist_ok=True)
    for source, dest in _result_file_pairs(pending, target):
        if not source.is_file():
            continue
        if backup and dest.exists():
            keep = dest.with_name(f"{dest.name}.{stamp}.bak")
            dest.replace(keep)
            backed_up.append(keep.name)
        shutil.copy2(source, dest)
    return "、".join(backed_up)


def _result_file_pairs(pending: Path, target: Path) -> list[tuple[Path, Path]]:
    """(暂存文件, 目标文件) 对:标定 JSON 与同名的站点归档。"""
    stations = ".stations.npz"
    return [
        (pending, target),
        (pending.with_name(pending.stem + stations), target.with_name(target.stem + stations)),
    ]


def _output_relpath(adapter: str, mount: str, profile: dict[str, Any]) -> str:
    """正式标定文件的落盘位置(相对 ``configs/``)。

    按约定推导 ``<adapter>/calibration/<adapter>_<mount>.json``:mount 进文件名,同一本体
    日后同时支持两种安装方式时两份产物不会互相覆盖,新本体也不必在档案里再写一行。档案里
    的 ``output_relpath`` 仍可覆盖这个约定。
    """
    override = str(profile.get("output_relpath") or "").strip()
    if override:
        return override
    suffix = f"_{mount}" if mount else ""
    return f"{adapter}/calibration/{adapter}{suffix}.json"


def _remedy_for(check: str) -> str | None:
    """按失败项前缀匹配补救建议(检查项名可能带后缀,如 observability_flange_axes_x)。

    取**最长**匹配前缀:``reproj`` 同时是 ``reproj_invalid`` 的前缀,只按声明顺序取首个
    命中会让两者共用一条建议,且结果随字典书写顺序漂移。
    """
    best_key: str | None = None
    best_text: str | None = None
    for key, text in _REMEDY.items():
        if check.startswith(key) and (best_key is None or len(key) > len(best_key)):
            best_key, best_text = key, text
    return best_text


def _write_calib_path(config_path: Path, artifact: Path) -> str:
    """把标定文件路径写进本体运行配置的 ``env.cfg.low_level.calib_path``。

    只改这一个键,用行级替换而非 YAML 重写 —— 重写会丢掉整份配置里的注释,而这些
    注释记录了各字段的物理含义与验收状态,对现场排障比格式统一重要得多。
    """
    text = config_path.read_text(encoding="utf-8")
    try:
        value = str(artifact.relative_to(config_path.parent))
    except ValueError:
        value = str(artifact)
    lines = text.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("calib_path:"):
            indent = line[: len(line) - len(line.lstrip())]
            lines[index] = f'{indent}calib_path: "{value}"'
            _backup_and_write(config_path, "\n".join(lines) + "\n")
            return value
    raise ValueError(
        f'{config_path} 里没有 calib_path 键,无法自动写回。请手工在 env.cfg.low_level 下添加 calib_path: "{value}"。'
    )


def _backup_and_write(path: Path, text: str) -> None:
    """写配置前留一份 ``.bak`` —— 这是用户手工维护的文件,不能只留新版本。"""
    shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
    path.write_text(text, encoding="utf-8")


def board_print_dependency_hint() -> str | None:
    """检查标定需要的依赖,缺失时返回中文指引;都就绪返回 ``None``。

    除了 OpenCV,还检查 ``scripts.calibrate`` 能否导入 —— 标定板的生成与识别都在那里。
    它由 pyproject 声明为包,但旧的 editable 安装没有它的路径映射,于是只有恰好在仓库
    根目录启动时才能导入。不显式拦下的话,用 console script 从别处启动就只会看到一个
    空白预览框,没有任何线索。
    """
    import importlib.util

    if importlib.util.find_spec("cv2") is None:
        return '标定需要 OpenCV(生成与识别标定板),当前环境未安装。请运行:pip install -e ".[calib]"'
    try:
        importlib.import_module("scripts.calibrate.handeye_board")
    except ImportError:
        return '标定组件未正确安装到当前环境。请在仓库目录下运行:pip install -e ".[calib]"'
    return None
