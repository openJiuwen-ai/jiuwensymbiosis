# 标定审查中的术语

仅在改动涉及标定时读取。以下是消歧提示，具体定义仍以目标 revision 的
设计文档、接口和实现为依据。

| 术语 | 报告中的用法 |
|---|---|
| Execution Trace replay | 执行轨迹回放，对应 `jiuwensymbiosis-replay` |
| 标定 CLI 的 `--replay` | 从 station archive 离线重新求解，不暗示机器人重复运动 |
| waypoint archive | 首次解释为示教轨迹文件，沿用项目原词 |
| station archive | 首次解释为标定采样数据集，不另造“事实档案”等概念 |
| `Station` | 一组标定采样，不能译作物理工位或工作站 |
| candidate | 写作 REVIEW/candidate 报告或 `*.candidate.json`，明确不能作为正式运行时标定加载 |
| schema | 文件格式约束；描述写出符合目标 schema 的标定 JSON，不写“写正式 schema” |
| 发布 | 说明实际写入的路径及副作用，本地文件写入不等于上传或部署 |

前后版本发生更名或语义变化时分别说明，不能用最新术语掩盖历史差异。
图表与正文固定使用同一组名称；存在跨子系统同名词时加限定词。
