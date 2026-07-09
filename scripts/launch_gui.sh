#!/usr/bin/env bash
# 启动 Jiuwen Symbiosis 图形界面 —— 跑仓库的实时源码,改代码后下次启动即生效
# (无需重新打包)。
#
# conda 环境名默认 jiuwensymbiosis;本地若用别的名字,设 JIUWEN_CONDA_ENV 覆盖:
#   JIUWEN_CONDA_ENV=jiuwen scripts/launch_gui.sh
#
# 依赖:已按 README 安装图形界面依赖(pip install -e ".[gui]",含 NiceGUI);浏览器
# 模式无需额外系统库(缺 NiceGUI 时程序会弹图形对话框提示如何安装)。

# 自动定位仓库根(本脚本位于 <repo>/scripts/)。
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${JIUWEN_CONDA_ENV:-jiuwensymbiosis}"

# 找到 conda 并激活目标环境(尽量兼容常见安装位置)。
CONDA_BASE=""
if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base 2>/dev/null)"
else
    for d in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/mambaforge" /opt/conda; do
        if [ -f "$d/etc/profile.d/conda.sh" ]; then
            CONDA_BASE="$d"
            break
        fi
    done
fi
cd "$REPO_DIR" || exit 1
# 清掉外部 PYTHONPATH(如 ROS)干扰,用仓库源码运行。经引导器启动:即便连
# jiuwensymbiosis 包都导不进(装错环境/没装),它也能弹图形窗提示,而非裸 traceback。
#
# 有 conda 就用 `conda run` 直接在目标环境里跑(免 `source conda.sh`,ShellCheck 无需
# 抑制注释);conda 不可用则回退当前 PATH 上的 python。
if [ -n "$CONDA_BASE" ] && [ -x "$CONDA_BASE/bin/conda" ]; then
    exec env -u PYTHONPATH "$CONDA_BASE/bin/conda" run --no-capture-output -n "$CONDA_ENV" \
        python "$REPO_DIR/scripts/gui_launcher.py"
fi
exec env -u PYTHONPATH python "$REPO_DIR/scripts/gui_launcher.py"
