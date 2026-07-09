# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""GUI 入口:``python -m jiuwensymbiosis.gui`` / ``jiuwensymbiosis-gui``。

先做代理卫生(必须在导入 openjiuwen 之前),再预检图形界面依赖,最后启动浏览器界面。
预检失败时用 tkinter 弹图形对话框(纯标准库),这样即使没有终端窗口(如从应用菜单/
桌面图标启动),用户也能看到"缺什么、怎么装"的提示。
"""

from __future__ import annotations

import logging
import sys

# 兜底弹窗优先使用的中文字体(按优先级)。挑第一个"tkinter 认得"的:
# - Xft 构建的 Tk 走 fontconfig,认得 Noto / 文泉驿 / 雅黑 等 TrueType 家族;
# - 无 Xft(仅 X11 核心位图字体)的 Tk 只认得 song ti / fangsong ti / gothic 等。
# 必须显式挑一个"自带中文"的字体:tkinter 默认字体的行距按拉丁字形算(≈11px),
# 而中文字形要 ≈14–16px,回退渲染时各行就会叠在一起。
_PREFERRED_CJK_FONTS: tuple[str, ...] = (
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "WenQuanYi Micro Hei",
    "WenQuanYi Zen Hei",
    "Microsoft YaHei",
    "SimHei",
    "song ti",
    "fangsong ti",
    "gothic",
)

# 兜底弹窗的目标字号(像素)。用像素而非磅值:磅值会随屏幕 DPI 换算,在 4K 高分屏
# 上被放大插值,位图字体就糊了。位图字体只有落在"原生像素尺寸"时才清晰,故据此对齐。
_DIALOG_FONT_PX = 26


def _preferred_cjk_font(available: set[str]) -> str | None:
    """从已安装(tkinter 可见)的字体里挑一个中文行距正常的;都没有则返回 None。"""
    return next((f for f in _PREFERRED_CJK_FONTS if f in available), None)


def _nearest_native_px(target_px: int, native_px: list[int]) -> int:
    """从位图字体的原生像素尺寸里挑最接近 target 的;没有原生尺寸(如 TrueType)则用 target。"""
    if not native_px:
        return target_px
    return min(native_px, key=lambda px: (abs(px - target_px), px))


def _show_error_dialog(title: str, message: str) -> None:
    """尽力弹一个 tkinter 错误框(纯标准库);失败则静默(调用方已打到 stderr)。

    图形界面走浏览器模式,若连 NiceGUI 都没装(或启动阶段异常),又是从桌面图标/菜单
    启动(无终端),就用只依赖标准库的 tkinter 弹窗把"缺什么、怎么装"告诉用户。

    自绘对话框而非 ``messagebox.showerror``,原因是为了防止:
    1. 挤成一团 —— 默认字体行距按拉丁字形算,中文各行重叠;这里显式挑自带中文的字体。
    2. 字糊 —— 此 Tk 无 Xft、只有位图字体,用磅值会被 DPI 放大插值;改用原生像素尺寸。
    3. 要能复制指令 —— 用只读的 ``Text`` 呈现,可鼠标划选、Ctrl+C 复制那条 apt 指令。
    """
    try:
        import tkinter
        from tkinter import font as tkfont
    except Exception as exc:  # 没有 tkinter 就放弃弹窗(主错误已在 stderr/日志)
        logging.getLogger(__name__).debug("tkinter unavailable, skip fallback dialog: %s", exc)
        return

    try:
        root = tkinter.Tk()
        root.title(title)
        root.resizable(False, False)

        family = _preferred_cjk_font(set(tkfont.families(root)))
        text_font = tkfont.Font(root=root, size=-_DIALOG_FONT_PX)
        if family:
            text_font.configure(family=family)
        # 位图字体只在原生像素尺寸下清晰(linespace==请求像素即未被插值),探测并对齐;
        # Xft/TrueType 任意尺寸都平滑,探测不到原生尺寸时保持目标像素即可。
        natives = []
        for px in range(max(_DIALOG_FONT_PX - 8, 8), _DIALOG_FONT_PX + 9):
            text_font.configure(size=-px)
            if text_font.metrics("linespace") == px:
                natives.append(px)
        text_font.configure(size=-_nearest_native_px(_DIALOG_FONT_PX, natives))

        frame = tkinter.Frame(root, padx=44, pady=36)
        frame.pack(fill="both", expand=True)

        # 只读但可划选/复制的文本区:按内容像素宽度换算列数,避免裁掉那条 apt 指令。
        lines = message.split("\n")
        char_px = text_font.measure("0") or 1
        cols = max((text_font.measure(line) for line in lines), default=1) // char_px + 3
        box = tkinter.Text(
            frame,
            font=text_font,
            width=cols,
            height=len(lines),
            wrap="none",
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            background=root.cget("background"),
            insertwidth=0,  # 隐藏光标:这是只读展示,不是输入框
            spacing3=4,  # 行后留一点空隙,读起来不局促
            padx=0,
            pady=0,
        )
        box.insert("1.0", message)
        # 屏蔽键盘编辑但放行 Ctrl+C 复制 / Ctrl+A 全选;鼠标划选不走 <Key>,照常可用。
        box.bind("<Key>", lambda e: None if (int(e.state) & 0x4 and e.keysym.lower() in ("c", "a")) else "break")
        box.bind("<Button-2>", lambda _e: "break")  # 屏蔽中键粘贴
        box.pack(fill="both", expand=True)

        tkinter.Button(frame, text="确定", width=10, font=text_font, command=root.destroy).pack(pady=(32, 0))

        root.update_idletasks()  # 先算出实际尺寸,再据此居中
        x = max((root.winfo_screenwidth() - root.winfo_width()) // 2, 0)
        y = max((root.winfo_screenheight() - root.winfo_height()) // 3, 0)
        root.geometry(f"+{x}+{y}")
        root.mainloop()
    except Exception as exc:  # 弹窗失败无所谓(主错误已在 stderr/日志)
        logging.getLogger(__name__).debug("fallback dialog failed: %s", exc)


def _startup_failure_message(exc: BaseException) -> str:
    """把一个"启动阶段"的异常翻译成面向用户的中文指引。

    preflight 已覆盖已知的常见缺失(NiceGUI);走到这里的是"没预料到"的启动失败
    (依赖装了但坏、openjiuwen 缺失、其它 import 链问题等),所以带上异常摘要 +
    通用处理建议,而不是抛裸 traceback。
    """
    return (
        "图形界面启动失败。\n"
        f"\n原因:{type(exc).__name__}: {exc}\n"
        "\n常见原因与处理:\n"
        '  • 缺少图形界面依赖 → 在已激活的 conda 环境执行:pip install -e ".[gui]"\n'
        "  • 激活了错误的 conda 环境 → conda activate jiuwensymbiosis\n"
        "\n详细堆栈见终端 / 日志。"
    )


def main() -> int:
    """入口:清理代理环境变量,预检系统库,然后启动图形界面。"""
    from jiuwensymbiosis.utils.proxy import clear_proxy_env

    clear_proxy_env()

    # 事前预检:NiceGUI 缺失会在导入那一刻抛裸 traceback;提前拦下并给出清晰中文指引,
    # 优雅退出(终端 + 图形弹窗),而不是让用户对着 ImportError 堆栈。
    from jiuwensymbiosis.gui.preflight import preflight_message

    message = preflight_message()
    if message is not None:
        logging.getLogger(__name__).error(message)  # 预检早于日志配置;无 handler 时经 lastResort 落 stderr
        _show_error_dialog("无法启动 Jiuwen Symbiosis", message)  # 无终端也可见
        return 1

    # 兜底安全网:预检放行后,导入 / 启动阶段仍可能有没预料到的异常(依赖坏、openjiuwen
    # 缺失等)。捕获后弹窗 + 打印堆栈,而不是让用户对着裸 traceback。
    try:
        from jiuwensymbiosis.gui.app import run

        return run()
    except Exception as exc:  # 启动失败要弹窗告知,而非静默/裸崩
        import traceback

        traceback.print_exc()  # stderr 给开发者留完整堆栈
        _show_error_dialog("无法启动 Jiuwen Symbiosis", _startup_failure_message(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
