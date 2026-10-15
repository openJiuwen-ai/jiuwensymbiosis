# 仿真配置

所有仿真后端的运行配置统一放在 `configs/simulators/`，硬件配置继续使用各本体目录。
当前提供 `moz1.yaml`，通过 `adapter: robosim` 选择 RoboSim 后端。

目录名仅用于分类，通用加载器根据 YAML 顶层的 `adapter` 字段发现后端；不要删除该字段或依赖文件路径选择后端。新增机器人、场景或任务的仿真配置也应放入此目录，并使用明确的文件名。

配置字段及启动方式见 [RoboSim 接入指南](../../docs/zh/how-to/robosim.md)。
