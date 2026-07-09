#!/usr/bin/env bash
# 一键把 Jiuwen Symbiosis 图形界面装进"应用菜单"(生成 .desktop)。
#
# 关键:Exec / Icon 两个地址都按**本机实际路径**动态生成(基于本脚本位置推导仓库根),
# 所以在任何机器/任何克隆路径下运行都对——不写死。
#
# 用法:
#   bash scripts/install_desktop_entry.sh              # 安装到 ~/.local/share/applications
#   bash scripts/install_desktop_entry.sh --uninstall  # 卸载
#
# 装好后:在活动/应用列表里搜 "Jiuwen Symbiosis" 即可点开,也可右键固定到 Dock。
# 桌面图标(~/Desktop)不在本脚本范围:GNOME 需再对该文件右键 "允许启动"。

set -euo pipefail

# 自动定位仓库根(本脚本位于 <repo>/scripts/)。
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAUNCHER="$REPO_DIR/scripts/launch_gui.sh"
ICON="$REPO_DIR/jiuwensymbiosis/gui/assets/app_icon.png"

APPS_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
DESKTOP_FILE="$APPS_DIR/jiuwensymbiosis.desktop"

# 卸载分支。
if [ "${1:-}" = "--uninstall" ]; then
    if [ -f "$DESKTOP_FILE" ]; then
        rm -f "$DESKTOP_FILE"
        command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$APPS_DIR" 2>/dev/null || true
        echo "已卸载:$DESKTOP_FILE"
    else
        echo "未发现已安装的桌面项:$DESKTOP_FILE(无需卸载)"
    fi
    exit 0
fi

# 安装前校验:启动器必须存在;图标可选(缺失则不写 Icon 行,用系统默认)。
if [ ! -f "$LAUNCHER" ]; then
    echo "错误:找不到启动脚本 $LAUNCHER" >&2
    exit 1
fi
chmod +x "$LAUNCHER"

icon_line=""
if [ -f "$ICON" ]; then
    icon_line="Icon=$ICON"
else
    echo "警告:找不到图标 $ICON,将使用系统默认图标。" >&2
fi

mkdir -p "$APPS_DIR"

# 生成 .desktop。Exec 用双引号包住绝对路径,符合 Desktop Entry 规范(路径含空格也安全)。
cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Version=1.0
Name=Jiuwen Symbiosis
Comment=九问共生 · 基于 openjiuwen 的具身智能体框架
Exec="$LAUNCHER"
${icon_line}
Terminal=false
Categories=Science;
StartupNotify=true
EOF

chmod +x "$DESKTOP_FILE"  # 部分桌面要求 .desktop 可执行才肯启动
command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$APPS_DIR" 2>/dev/null || true

echo "已安装到应用菜单:$DESKTOP_FILE"
echo
echo "  Exec = $LAUNCHER"
echo "  Icon = ${ICON:-（系统默认）}"
echo
echo "现在可以:在活动/应用列表里搜 \"Jiuwen Symbiosis\" 点开,或右键固定到 Dock。"
echo "卸载:bash scripts/install_desktop_entry.sh --uninstall"
