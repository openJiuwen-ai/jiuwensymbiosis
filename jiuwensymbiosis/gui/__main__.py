# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""兼容入口：将旧命令转发给统一 GUI 启动器。

``python -m jiuwensymbiosis.gui`` 仅在执行入口时导入新的 launcher，普通导入旧
命名空间不会装载 GUI 插件或 NiceGUI。
"""

from __future__ import annotations

import sys


def main() -> int:
    from jiuwensymbiosis_gui.__main__ import main as launcher_main

    return launcher_main()


if __name__ == "__main__":
    sys.exit(main())
