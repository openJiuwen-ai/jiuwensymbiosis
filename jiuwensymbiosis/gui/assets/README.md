# GUI 资源

- `favicon.svg` —— 浏览器标签页图标(openJiuwen 官方品牌标,矢量)。`app.py`
  在存在时传给 `ui.run(favicon=...)`;缺失时回落到 NiceGUI 默认图标,不报错。
- `app_icon.png` —— 应用图标(窗口/任务栏/应用菜单)。放一张方形 PNG(建议
  256×256 或 512×512)在此文件名下即生效:`app.py` 会在存在时自动加载,
  `~/.local/share/applications/jiuwen-symbiosis.desktop` 的 `Icon=` 也指向它。
  缺失时回落到系统默认图标,不报错。
