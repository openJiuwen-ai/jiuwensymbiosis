# 查看 Genesis 启动耗时与编译缓存

Genesis 后台状态的 `startup` 字段与 worker 日志记录初次启动的阶段耗时、SDK 版本和实际 Quadrants 编译缓存配置。该诊断不会改变物理时间步、碰撞模型或渲染参数。

`startup.quadrants_cache` 来自 `gs.init()` 后的实际运行配置，包含：

- `offline_cache`：编译缓存是否开启。
- `offline_cache_file_path`：本次 worker 使用的绝对目录。
- `offline_cache_max_size_of_files`：内核缓存容量上限，单位为字节。
- `offline_cache_cleaning_policy`、`offline_cache_cleaning_factor`：清理策略与比例。

`available=false` 表示读取不到该 SDK 版本的配置；不要把它解释为缓存已关闭。

## 显式覆盖缓存配置

通过启动 GUI 的进程环境传入以下两项。以下命令仅展示变量设置；随后从同一终端运行正常的 Symbiosis 启动命令：

```bash
export QD_OFFLINE_CACHE_FILE_PATH="$HOME/.cache/jiuwensymbiosis/quadrants"
export QD_OFFLINE_CACHE_MAX_SIZE_OF_FILES=2147483647
```

容量必须是 `1..2147483647` 范围内的十进制整数；整 `2147483648` 超过当前 Quadrants 字段的上限。路径可不存在，但不能为空或指向普通文件。相对路径在父进程当前目录下解析成绝对路径，再交给 worker，避免 worker 工作目录改变含义。设置变量不会复制旧缓存；改到新目录后的首次启动可能需要重新生成缓存。

未设置时，Symbiosis 保留已安装 Genesis / Quadrants 的默认策略，不自行创建、清空或扩大缓存。更改环境变量只影响下一次新 worker；已运行的 worker 不会被动态改写。

验证时检查 worker 返回的实际值，不只检查终端环境变量。其他 `QD_*` 变量不会因为这两项的支持而全部放行。

## 耗时口径

`phases_ms` 记录墙钟调用区间，例如 SDK 导入、Genesis 初始化、碰撞与机器人实体创建、`scene_build`、稳定落地、Nyx renderer 和编码器初始化。

- 阶段可以嵌套。`runtime_scene` 已包含实体、物理构建、相机和 Nyx 等阶段，不能把父子阶段相加。
- `scene_build` 包含 Genesis SDK 内部的求解器构建及首次步进；正式诊断不会为拆分它们而修改 SDK 或执行顺序。
- 起点位于 worker 的 `run_preview`、SDK 导入之前，不包括解释器启动及此前模块导入；从点击启动到后台 ready 的完整时间应由调用方测量。
- `total_ms` 在首帧 RGB 读取与同步完成后冻结，随后才发布 ready；`start_boundary` 和 `end_boundary` 明确测量边界。`first_frame_ms` 单独记录从相同起点到首次完成 RGB 的时间，尚未完成时为 `null`。`loading` 对每次启动、切换或复位分别记录阶段和 operation_id。
- 后台 ready 与浏览器的 WebRTC 连接、解码和显示是不同终点，不能互相代替。
- 外部 profiler 如果提前导入 SDK，会使正式诊断的 `imports` 接近零；比较性能时必须保持相同测量入口。
- 后续场景/画质热切换保留初次 `startup` 记录，切换本身仍查看已有切换时间字段。

## 对照实验

对缓存容量进行 A/B 时，每组从相同缓存种子复制独立目录，并各自连续继承多次正常退出后的缓存。保持版本、资产、物理与渲染设置相同；记录首次与后续重复启动、有效配置、退出码和缓存增删。

Genesis 资产缓存、Quadrants 编译缓存与 NVIDIA driver 缓存是不同层次，可分别通过已有的 `GS_CACHE_FILE_PATH`、本页的 `QD_OFFLINE_CACHE_FILE_PATH` 和已有的 `CUDA_CACHE_PATH` 指定隔离副本。不要删除用户原缓存制造冷启动，也不要把整个 Quadrants 目录大小当作 `.qdc` 内核容量占用。

容量变大不代表必然更快。若原缓存没有淘汰工作集，实体创建、GPU 数据初始化及 Nyx 场景加载仍需执行。
