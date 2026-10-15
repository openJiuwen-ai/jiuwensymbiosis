# 仿真 backend 与硬件 adapter 的选择边界

## 问题与基线

基于 PR1 `b246b55`。原方案借用 `adapter: robosim`，并按模块存在与否优先查找仿真包。这使硬件适配器与仿真后端共用一个含义不准确的字段，也可能因同名包安装状态改变解析结果。用户要求以 `backend` 明确表达仿真后端。

## 配置契约

| 选择字段 | 命名空间 | 示例 |
| --- | --- | --- |
| `adapter` | `jiuwensymbiosis.adapters` | `adapter: piper` |
| `backend` | `jiuwensymbiosis.simulators` | `backend: robosim` |

两个字段必须且只能出现一个。空值、无效模块名、同时存在时均拒绝。按字段直接解析命名空间，不自动回退；旧 PR 草稿的 `adapter: robosim` 改为 `backend: robosim`，不保留含混别名。硬件配置沿用已有字段。

## 职责与数据流

`backends.SessionTarget` 封装选择字段、名称、模块路径和快照匹配校验。`session_target()` 只解析声明，不加载后端。

`prepare_binding()` 继续负责配置冻结、源路径解析和资源身份准备，但按 SessionTarget 加载对应的会话构建器。`BindingSnapshot.target` 提供明确身份；保留已有硬件构造参数 `adapter`，仿真时该值为 None，新字段 `backend` 保存仿真后端名。指纹、维护绑定和运行事件保留对应字段，仿真不会被重写为硬件。

会话构建器仍遵循 `build_<name>_session`、`config_factory`、`resource_keys` 的共用契约。仿真复用现有配置解析、资源准入及会话组装工具；这些工具目前位于 `adapters/_common`，不做硬件后端选择或硬件连接。本次不移动该公共工具包，避免扩大成无关的全仓目录迁移。

CLI 保留硬件缺省 piper 与 `--robot` 覆盖行为；仿真由 YAML `backend` 显式指定，拒绝与 `--robot` 混用。GUI 本体注册、运行快照和维护快照使用相同选择契约；硬件标定发现入口不接受仿真本体。

## 失败语义与验证

选择错误必须在构建/连接前失败；GUI 已选本体与配置选择不一致时拒绝，不能通过补齐字段静默切换命名空间。配置后续修改不得改变已冻结的目标或准入资源。

- `tests/unit_tests/test_backends.py`：无导入解析、非法/冲突字段、错误命名空间、仿真绑定与维护副本的身份和快照。
- `tests/unit_tests/runtime/test_cli_admission.py`：真实 CLI 配置准备保留 backend，仍先准入后连接；无硬件或外部模型调用。
- `tests/unit_tests/runtime/test_jobs.py`：仿真任务事件写 backend，不写 adapter。
- GUI registry、RunEngine、MaintenanceOwner 用例：配置传递、冲突拒绝及原有硬件回归。

本方案不改变 RoboSim HTTP 协议、运动单位、结果未知保护或恢复行为。验收结果另记在 output 中，不以设计文档代替测试证据。
