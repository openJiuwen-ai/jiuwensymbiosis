# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""运行页:实时监看(软件核心),NiceGUI 版。

把「看得懂」放中央:大号相机画面 + 一句话当前动作 + 右侧可点开的步骤时间线;原始日志 /
安全事件 / 错误诊断收进底部默认折叠的抽屉。所有更新来自 ``RunEngine`` 的事件队列——由
``ui.timer`` 周期 ``drain()`` 后应用到界面元素,跨线程只经这一个队列。

相机用 ``interactive_image``:窗口任意宽度按比例完整显示、不裁剪,并预留 SVG 叠加层
(检测框/mask 的基础设施);点击历史步骤可回看当时那一帧。
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from nicegui import ui

from jiuwensymbiosis.gui import local_models
from jiuwensymbiosis.gui.diagnostics import FIX_USE_HF_MIRROR, FIX_USE_LOCAL_MODEL, Diagnosis, diagnose
from jiuwensymbiosis.gui.run_status import STATUS_COLORS, outcome_from_result

__all__ = ["RunView"]


class RunView:
    """实时监看页。``on_stop`` 停止本次运行,``on_fix`` 把一键修复补丁沉淀进配置。"""

    def __init__(self, *, on_stop: Callable[[], None], on_fix: Callable[[dict], None]) -> None:
        """搭建状态条、主视觉区、步骤时间线与技术抽屉,并挂上拉取事件的定时器。"""
        self._on_stop = on_stop
        self._on_fix = on_fix
        self._engine: Any = None
        self._running = False
        self._t0 = 0.0
        self._rows: dict[int, Any] = {}
        self._details: dict[int, str] = {}
        self._step_frames: dict[int, str] = {}
        self._latest_uri = ""
        self._viewing_past = False
        self._selected = -1

        self._dispatch: dict[str, Callable[[Any], None]] = {
            "run_started": self._on_run_started,
            "step_started": self._on_step_started,
            "step_finished": self._on_step_finished,
            "frame": self._on_frame,
            "narration": self._on_narration,
            "safety_event": self._on_safety_event,
            "log": self._on_log,
            "run_finished": self._on_run_finished,
        }

        self._build()
        ui.timer(0.1, self._drain)
        ui.timer(0.5, self._tick)

    # ------------------------------------------------------------------ 布局
    def _build(self) -> None:
        with ui.row().classes("w-full items-center gap-3"):
            self._header = ui.label("准备就绪").classes("text-base grow")
            self._badge = ui.label("准备中")
            self._set_badge("准备中")
            self._timer_label = ui.label("00:00").classes("font-mono")
            self._stop_btn = ui.button("■ 停止", on_click=self._on_stop_clicked).props("color=negative")
            self._stop_btn.disable()

        self._banner = (
            ui.label().classes("w-full").style("background:#fff3cd; color:#7a5b00; padding:6px; border-radius:4px;")
        )
        self._banner.set_visibility(False)

        # 相机框固定高度(不随有/无图片跳变);内层 <img> 用 object-fit:contain 等比缩放、
        # 不裁剪、不拉伸,空缺处用深色背景补齐。高度用 vh 随窗口自适应,给展开的「运行详情」
        # 留出空间——想更大/更小只改下面那一处 height 值即可。
        ui.add_head_html("<style>.jw-cam img{object-fit:contain;}</style>")
        with ui.row().classes("w-full no-wrap gap-4"):
            with ui.column().classes("w-1/3 gap-1"):
                self._camera = (
                    ui.interactive_image().classes("w-full rounded jw-cam").style("background:#111; height:42vh;")
                )
                self._live_btn = ui.button("↩ 回到实时画面", on_click=self._back_to_live).props("flat dense")
                self._live_btn.set_visibility(False)
                self._narration = (
                    ui.label("—").classes("w-full text-center font-bold").style("font-size:16px; padding:6px;")
                )
            with ui.column().classes("w-2/3 gap-1"):
                ui.label("执行步骤(点击展开原始细节):").classes("text-sm text-gray-600")
                # 与相机框一致用 vh 固定高度,画面比例协调;内容超出时框内自带滚动条。
                with ui.scroll_area().classes("w-full border rounded").style("height:30vh"):
                    self._timeline = ui.column().classes("w-full gap-1")
                with ui.scroll_area().classes("w-full border rounded p-2").style("height:12vh"):
                    self._detail = ui.label("点击左侧某一步查看原始工具调用与参数…").classes(
                        "whitespace-pre-wrap font-mono text-xs text-gray-700"
                    )

        self._drawer = ui.expansion("运行详情(原始日志 / 安全事件 / 错误诊断)", value=False).classes("w-full")
        with self._drawer:
            with ui.tabs() as self._tabs:
                log_tab = ui.tab("原始日志")
                safety_tab = ui.tab("安全事件")
                self._diag_tab = ui.tab("错误诊断")
            with ui.tab_panels(self._tabs, value=log_tab).classes("w-full"):
                with ui.tab_panel(log_tab):
                    self._log = ui.log(max_lines=1000).classes("w-full font-mono text-xs").style("height:15vh")
                with ui.tab_panel(safety_tab):
                    self._safety = ui.log(max_lines=400).classes("w-full font-mono text-xs").style("height:15vh")
                with ui.tab_panel(self._diag_tab):
                    self._build_diagnosis()

    def _build_diagnosis(self) -> None:
        self._diag_title = ui.label("").classes("text-red-700 font-bold text-base")
        self._diag_cause = ui.label("").classes("whitespace-pre-wrap")
        self._diag_steps = ui.html("").classes("whitespace-pre-wrap")

        with ui.column().classes("w-full gap-1 mt-3") as self._method1_box:
            ui.label("解决方法一:填入本机已下好的视觉检测模型目录").classes("font-bold")
            with ui.row().classes("w-full items-center no-wrap gap-2"):
                ui.label("GroundingDINO:")
                self._gdino_input = ui.input(placeholder="GroundingDINO 模型目录").classes("grow")
                ui.button("自动检测", on_click=self._detect_gdino).props("flat dense")
            with ui.row().classes("w-full items-center no-wrap gap-2"):
                ui.label("SAM2:")
                self._sam2_input = ui.input(placeholder="SAM2 模型目录").classes("grow")
                ui.button("自动检测", on_click=self._detect_sam2).props("flat dense")
            ui.button("使用这些地址", on_click=self._use_local)
        self._method1_box.set_visibility(False)

        with ui.column().classes("w-full gap-1 mt-3") as self._method2_box:
            ui.label("解决方法二:更换成国内镜像源重新下载").classes("font-bold")
            ui.button("一键更换", on_click=self._use_mirror)
        self._method2_box.set_visibility(False)

        self._diag_hint = ui.label("如果你熟悉本系统,也可以切到「原始日志」标签查看更详细的报错。").classes(
            "text-gray-500 text-xs"
        )
        self._diag_hint.set_visibility(False)

    # ------------------------------------------------------------------ 生命周期
    def attach(self, engine: Any) -> None:
        """接管一次新运行:清空视图、置「运行中」、起表,然后启动引擎线程。"""
        self._reset()
        self._engine = engine
        self._running = True
        self._t0 = time.monotonic()
        self._set_badge("运行中")
        # 首次运行要连接硬件、拉起检测服务、编译动作序列,第一条指令出现前会有空档;
        # 先在当前动作行给出加载提示,第一步的叙述到达后自动替换。
        self._narration.set_text("加载中，请稍等…")
        self._stop_btn.enable()
        engine.start()

    def show_model_help(self, missing: list[str]) -> None:
        """真机运行前发现视觉模型缺失:直接展示「错误诊断」的自动检测/镜像/填目录,而非空跑到超时。"""
        self._reset()
        self._set_badge("未完成")
        names = "、".join(missing)
        self._narration.set_text(
            f"视觉检测模型未就绪({names})。请在下方「错误诊断」用「自动检测」定位本机已下好的模型目录、"
            "或「一键更换」国内镜像后,重新点击「运行」。"
        )
        diag = Diagnosis(
            title="视觉检测模型未就绪",
            cause=f"未在本机缓存找到 {names};直接联网下载 GroundingDINO/SAM2 可能很慢或卡住(连不上 huggingface.co)。",
            steps=(
                "点「自动检测」在本机缓存里定位已下好的模型目录,确认后点「使用这些地址」。",
                "或点「一键更换」切到国内镜像重新下载。",
            ),
            fixes=(FIX_USE_LOCAL_MODEL, FIX_USE_HF_MIRROR),
        )
        self._show_diagnosis(diag)
        self._drawer.open()
        self._tabs.set_value(self._diag_tab)

    def _reset(self) -> None:
        self._rows.clear()
        self._details.clear()
        self._step_frames.clear()
        self._latest_uri = ""
        self._viewing_past = False
        self._selected = -1
        self._timeline.clear()
        self._log.clear()
        self._safety.clear()
        self._detail.set_text("点击左侧某一步查看原始工具调用与参数…")
        self._banner.set_visibility(False)
        self._live_btn.set_visibility(False)
        self._narration.set_text("—")
        self._clear_diagnosis()

    def _drain(self) -> None:
        engine = self._engine
        if engine is None:
            return
        for tag, payload in engine.drain():
            handler = self._dispatch.get(tag)
            if handler is not None:
                handler(payload)

    # ------------------------------------------------------------------ 事件处理
    def _on_run_started(self, meta: dict) -> None:
        mode = "模拟" if meta.get("mock") else "真机"
        self._header.set_text(f"任务:{meta.get('task', '')}   本体:{meta.get('body', '')}   [{mode}]")
        self._set_badge("运行中")
        self._running = True
        self._stop_btn.enable()
        self._t0 = time.monotonic()

    def _on_step_started(self, info: dict) -> None:
        idx = int(info.get("index", len(self._rows) + 1))
        label = info.get("label", info.get("tool", ""))
        with self._timeline:
            row = ui.row().classes("w-full items-center cursor-pointer rounded px-2 py-1 hover:bg-gray-100")
            with row:
                lbl = ui.label(f"⏳ {idx}. {label}")
        row.on("click", lambda _e, i=idx: self._select(i))
        self._rows[idx] = lbl
        self._details[idx] = self._format_detail(info, running=True)

    def _on_step_finished(self, info: dict) -> None:
        idx = int(info.get("index", len(self._rows)))
        ok = bool(info.get("ok"))
        icon = "✅" if ok else "❌"
        label = info.get("label", info.get("tool", ""))
        text = f"{icon} {idx}. {label}   {info.get('duration_s', 0.0):.2f}s"
        lbl = self._rows.get(idx)
        if lbl is not None:
            lbl.set_text(text)
        self._details[idx] = self._format_detail(info, running=False)
        self._step_frames[idx] = self._latest_uri
        if self._selected == idx:
            self._detail.set_text(self._details[idx])

    def _on_frame(self, uri: str) -> None:
        self._latest_uri = uri
        if not self._viewing_past:
            self._camera.set_source(uri)

    def _on_narration(self, text: str) -> None:
        self._narration.set_text(text)

    def _on_safety_event(self, info: dict) -> None:
        detail = info.get("detail", "")
        self._banner.set_text(f"⚠️ 安全护栏拦截:{detail}")
        self._banner.set_visibility(True)
        self._safety.push(f"[{info.get('rail', '')}/{info.get('kind', '')}] {detail}")

    def _on_log(self, record: dict) -> None:
        self._log.push(f"{record.get('level', '')} {record.get('name', '')}: {record.get('msg', '')}")

    def _on_run_finished(self, result: dict) -> None:
        self._running = False
        self._stop_btn.disable()
        outcome = outcome_from_result(result)
        self._set_badge(outcome.status)
        self._narration.set_text(outcome.narration)
        if outcome.is_failure:
            err = str(result.get("error", "")).strip()
            self._show_diagnosis(diagnose(err, str(result.get("log_tail", ""))))
            if err:
                self._log.push(f"ERROR 运行失败: {err}")
            self._drawer.open()
            self._tabs.set_value(self._diag_tab)
            return
        payload = result.get("result")
        summary = payload.get("output") if isinstance(payload, dict) else payload
        if outcome.status == "未完成" and summary:
            self._log.push(f"WARNING 未完成: {summary}")

    # ------------------------------------------------------------------ 步骤/相机交互
    def _select(self, idx: int) -> None:
        self._selected = idx
        self._detail.set_text(self._details.get(idx, ""))
        frame = self._step_frames.get(idx)
        if frame:
            self._camera.set_source(frame)
            self._viewing_past = True
            self._live_btn.set_visibility(True)

    def _back_to_live(self) -> None:
        self._viewing_past = False
        self._live_btn.set_visibility(False)
        if self._latest_uri:
            self._camera.set_source(self._latest_uri)

    def _on_stop_clicked(self) -> None:
        self._stop_btn.disable()
        self._narration.set_text("正在停止…")
        self._on_stop()

    # ------------------------------------------------------------------ 错误诊断
    def _show_diagnosis(self, diag: Diagnosis) -> None:
        self._diag_title.set_text(f"❌ {diag.title}")
        self._diag_cause.set_text(diag.cause)
        steps = "".join(f"• {s}<br>" for s in diag.steps)
        self._diag_steps.set_content(f"<b>怎么办:</b><br>{steps}" if steps else "")
        self._method1_box.set_visibility(FIX_USE_LOCAL_MODEL in diag.fixes)
        self._method2_box.set_visibility(FIX_USE_HF_MIRROR in diag.fixes)
        self._diag_hint.set_visibility(True)

    def _clear_diagnosis(self) -> None:
        self._diag_title.set_text("")
        self._diag_cause.set_text("")
        self._diag_steps.set_content("")
        self._gdino_input.value = ""
        self._sam2_input.value = ""
        self._method1_box.set_visibility(False)
        self._method2_box.set_visibility(False)
        self._diag_hint.set_visibility(False)

    def _detect_gdino(self) -> None:
        self._detect_into(
            self._gdino_input, local_models.GDINO_REPO, local_models.looks_like_gdino_dir, "GroundingDINO"
        )

    def _detect_sam2(self) -> None:
        self._detect_into(self._sam2_input, local_models.SAM2_REPO, local_models.looks_like_sam2_dir, "SAM2")

    def _use_local(self) -> None:
        gdino = (self._gdino_input.value or "").strip()
        sam2 = (self._sam2_input.value or "").strip()
        if not gdino and not sam2:
            ui.notify("请至少填写(或自动检测)一个模型目录。", type="warning")
            return
        patch: dict[str, str] = {}
        if gdino:
            if not local_models.looks_like_gdino_dir(Path(gdino)):
                ui.notify("GroundingDINO 目录缺少 config.json 或权重文件,请确认路径。", type="negative")
                return
            os.environ["GDINO_MODEL_ID"] = gdino  # 当次运行立即生效(config 优先读此环境变量)
            patch["gdino_model_id"] = gdino
        if sam2:
            if not local_models.looks_like_sam2_dir(Path(sam2)):
                ui.notify("SAM2 目录缺少 config.json / processor_config.json 或权重文件,请确认路径。", type="negative")
                return
            os.environ["SAM2_MODEL_ID"] = sam2
            patch["sam2_model_id"] = sam2
        self._on_fix(patch)
        ui.notify("下次运行将使用你指定的本地模型,请重新点击「运行」。", type="positive")

    def _use_mirror(self) -> None:
        os.environ["HF_ENDPOINT"] = local_models.HF_MIRROR  # 子进程继承,当次生效
        self._on_fix({"hf_endpoint": local_models.HF_MIRROR})
        ui.notify("已切换到国内镜像源,下次运行会用它重新下载模型,请重新点击「运行」。", type="positive")

    # ------------------------------------------------------------------ 内部
    def _set_badge(self, text: str) -> None:
        color = STATUS_COLORS.get(text, "#888")
        self._badge.set_text(text)
        self._badge.style(f"color:white; background:{color}; border-radius:8px; padding:2px 12px; font-weight:bold;")

    def _tick(self) -> None:
        if not self._running:
            return
        elapsed = int(time.monotonic() - self._t0)
        self._timer_label.set_text(f"{elapsed // 60:02d}:{elapsed % 60:02d}")

    @staticmethod
    def _detect_into(field: Any, repo_id: str, validator: Any, name: str) -> None:
        found = local_models.detect_local_model(repo_id, validator)
        if found is None:
            ui.notify(f"没在常见位置找到已下好的 {name} 模型,请手动填入目录。", type="warning")
            return
        field.value = str(found)
        ui.notify(f"找到本地 {name} 模型,请点「使用这些地址」确认使用。", type="positive")

    @staticmethod
    def _format_detail(info: dict, *, running: bool) -> str:
        lines: list[str] = []
        thought = str(info.get("assistant_text", "")).strip()
        if thought:
            lines.append(f"AI 说明:{thought}")
            lines.append("")
        lines += [
            f"工具:{info.get('tool', '')}",
            f"参数:{json.dumps(info.get('params', {}), ensure_ascii=False)}",
        ]
        if running:
            lines.append("状态:进行中…")
        else:
            lines.append(f"状态:{'成功' if info.get('ok') else '失败'}   用时:{info.get('duration_s', 0.0):.2f}s")
            lines.append(f"返回:{info.get('output', '')}" if info.get("ok") else f"错误:{info.get('error', '')}")
        return "\n".join(lines)
