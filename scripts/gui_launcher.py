#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""GUI 引导器:在能否导入 ``jiuwensymbiosis`` 之前先兜底。

``launch_gui.sh`` 调用本脚本(而非 ``python -m jiuwensymbiosis.gui``),是为了兜住
**更早一层**的失败:若连 ``jiuwensymbiosis`` 包本身都导入不了(装错 conda 环境 /
没 ``pip install``),``python -m`` 会在 runpy 阶段抛一个只有终端里才看得见的
traceback——而从 ``.desktop``/桌面图标启动时根本没有终端。这里用**纯标准库 +
tkinter** 弹一个中文对话框告诉用户怎么办。

包能导入之后,一切交给 ``jiuwensymbiosis.gui.__main__.main()``:后者用它自己那套更
完善的弹窗处理 NiceGUI 缺失、以及启动阶段的其它异常。
"""

from __future__ import annotations

import logging
import os
import sys

# 本脚本在 <repo>/scripts/,直接运行时 repo 根不在 sys.path 上。追加进去,保持
# launch_gui.sh"跑仓库实时源码、改完即生效"的语义(不依赖是否 pip 安装)。
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)


def _import_failure_message(exc: BaseException) -> str:
    """把"包导入失败"翻译成面向用户的中文指引。"""
    return (
        "无法启动 Jiuwen Symbiosis:未能导入 jiuwensymbiosis。\n"
        f"\n原因:{type(exc).__name__}: {exc}\n"
        "\n多为未激活正确的 conda 环境,或尚未安装本项目。请执行:\n"
        "    conda activate jiuwensymbiosis\n"
        '    pip install -e ".[gui]"\n'
        "\n然后重新运行本程序。"
    )


def _dialog(title: str, message: str) -> None:
    """尽力弹一个自包含的 tkinter 对话框;失败则静默(stderr 已有提示)。

    这里刻意不复用 ``jiuwensymbiosis.gui.__main__`` 里的弹窗——本函数触发的前提正是
    "那个包导不进来"。故只依赖标准库,并显式挑一个自带中文的字体(默认字体在部分
    Linux 上渲染中文会行距塌缩、糊成一团)。
    """
    try:
        import tkinter
        from tkinter import font as tkfont
    except Exception as exc:  # 没有 tkinter 就放弃弹窗(主错误已在 stderr/日志)
        logging.getLogger(__name__).debug("tkinter unavailable, skip fallback dialog: %s", exc)
        return

    # 优先挑一个"tkinter 认得且自带中文"的字体;Xft 构建认 TrueType 家族,无 Xft 的
    # 只认 song ti / fangsong ti 等核心字体;都没有就退回默认。
    preferred = (
        "Noto Sans CJK SC",
        "WenQuanYi Micro Hei",
        "Microsoft YaHei",
        "SimHei",
        "song ti",
        "fangsong ti",
    )
    try:
        root = tkinter.Tk()
        root.title(title)
        root.resizable(False, False)
        available = set(tkfont.families(root))
        family = next((f for f in preferred if f in available), None)
        text_font = tkfont.Font(root=root, size=-24)
        if family:
            text_font.configure(family=family)

        frame = tkinter.Frame(root, padx=44, pady=36)
        frame.pack(fill="both", expand=True)
        tkinter.Label(frame, text=message, justify="left", anchor="w", font=text_font).pack(fill="both", expand=True)
        tkinter.Button(frame, text="确定", width=10, font=text_font, command=root.destroy).pack(pady=(32, 0))

        root.update_idletasks()
        x = max((root.winfo_screenwidth() - root.winfo_width()) // 2, 0)
        y = max((root.winfo_screenheight() - root.winfo_height()) // 3, 0)
        root.geometry(f"+{x}+{y}")
        root.mainloop()
    except Exception as exc:  # 弹窗失败无所谓(主错误已在 stderr/日志)
        logging.getLogger(__name__).debug("fallback dialog failed: %s", exc)


def main() -> int:
    """导入 GUI 主入口并运行;导入失败则弹窗指引并返回 1。"""
    try:
        from jiuwensymbiosis.gui.__main__ import main as gui_main
    except Exception as exc:  # 包/依赖导不进,兜底弹窗而非裸崩
        message = _import_failure_message(exc)
        logging.getLogger(__name__).error(message)  # 无 handler 时经 lastResort 落 stderr
        _dialog("无法启动 Jiuwen Symbiosis", message)
        return 1
    return gui_main()


if __name__ == "__main__":
    sys.exit(main())
