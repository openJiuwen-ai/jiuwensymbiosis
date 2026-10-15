# RoboSim PR1 动作边界与失败语义修复

## 起点与目标

基于 `9eaa399` 复核七项问题：API 绕过 Env、TCP 被当作 FLANGE、命名关节缺少 timeout_s、写后读回失败丢失未知结果、重连丢失绑定、无有效响应仍成功，以及 smoke 未取得资源准入。保留 `backend: robosim` 的独立发现路径和现有 ActionSpec，不改 PR2。

## 职责与接口

调用链固定为 `RobosimApi → RobosimEnv → RobosimLowLevel → RobosimClient`。API 绑定动作词表并包装结果，Env 提供本体公开动词、持有生命周期和所选臂抓持确认，Driver 负责协议、坐标转换与实测完成证据。

- 命名关节动作使用已有 `Env.move_joint`；带控制参数的调用沿用 `move_named_joints`。Driver 接受 `timeout_s`，省略时取配置等待期限；非法值在读取和发令前拒绝。
- 当前服务只有 TCP 观测与控制，未提供可靠的 TCP↔FLANGE 转换。因此添加能力切片 `TcpDriver`，与 `CartesianDriver` 作为 `motion.cartesian` 的两种坐标接口。RoboSim 明确实现 `get_tcp_pose/move_tcp_blocking`，Env 提供 `get_tcp_pose/move_to_tcp`；FLANGE 接口抛出明确错误，不伪造零工具偏移。
- 能力检查映射和通用桩驱动同步支持 TCP 切片，原硬件 FLANGE 接口保持原义。未新增 ActionSpec，也不要求硬件迁移。
- 关节、TCP 完成读回在同一 Driver command 中完成并返回；API 复用结果，避免完成记录之后再触发一次未受保护的查询。
- Env 的夹爪接口保留 Driver 返回的抓持证据；开合与抓取成功不同，空夹爪闭合仍抛出 `GraspNotConfirmedError`。

## 状态与所有权

CommandCoordinator 在最外层命令开始时清除本次发令标记；Client 完整解析成功写响应后标记已接受写入。随后任何未确认完成的异常都将整个命令升级为 `outcome_unknown`，保留错误码并设置 `skip_recovery`，后续写入阻断。首次发令前的校验、纯读查询、明确拒绝写入不因此污染写租约。KeyboardInterrupt 等中断保持原异常，不吞掉取消；已有未知状态直到观察到新服务实例才解除。

传输层要求写响应有成功 envelope 和非空 data 对象，仍允许只读桥接 JSON；Driver 另外验证夹爪状态和归位完成字段。关节/TCP 的成功还要求独立实测到位，HTTP 200 本身不能提交完成效果。

`RobosimEnv.bind_transport` 持有不可自动替换的工作区客户端和实例令牌，Driver 只是一次连接生命周期的执行对象。断开或首次连接失败会关闭 Driver，但不丢弃绑定。重连只能使用同一工作区客户端、同一机器人实例；工作区关闭后也不能偷偷创建新的自有客户端。独立会话的自有客户端生命周期不变。

smoke 先 `prepare_binding`，再 `admitted_session(operation="robosim-smoke")`，取得同端口资源租约后才构建和连接。清理沿用运行时 CleanupReport；资源释放不代表外部未知操作已经完成。跨重启持久化仍留在 PR4。

## 验证

- API 仅拿到 Env 公开动词也能执行全部基础动作；FLANGE 调用拒绝且零 Driver I/O。
- 命名关节 timeout_s 的生效、非法值、TCP 坐标往返和 Protocol 能力映射。
- 真实客户端配合隔离 HTTP 替身：写后 GET 失败跨 API/runner 保持未知，不追加释放/归位；查询仍可用。
- 同实例重连保留借用；替换实例及失败后重试拒绝；工作区关闭不能转为独立会话。
- 缺失 envelope、data、夹爪字段、归位证据时拒绝成功；已有正常只读协议不受影响。
- 同进程及独立进程持有服务端口租约时，smoke 零连接零发令；连接失败后正确清理。

单元、类型和通用桩验证不替代真实服务时序及物理验收；执行结果保存在 output。
