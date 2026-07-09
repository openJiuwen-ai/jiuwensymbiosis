# JiuwenSymbiosis 开发指南

本文档面向新加入仓库的开发者，覆盖：Git/GitCode 协作流程、开发行为规则、提交前的格式化 / 扫描 / 测试 / 文档更新。

> 项目约定以 `AGENTS.md` 为跨工具（Cursor / Copilot / Claude）共享的唯一事实来源，`pyproject.toml` 是 Python/工具链配置的唯一事实来源。本指南是二者在"日常开发流"上的浓缩，遇冲突以 `AGENTS.md` 与 `pyproject.toml` 为准。

---

## 一、环境准备

### 1.1 Python 环境

```bash
# 编辑式安装（二选一）
pip install -e ".[dev]"                                          # 核心 + 测试依赖
pip install -e ".[full]" --extra-index-url https://download.pytorch.org/whl/cu128  # + 视觉/GPU 依赖
pip install -e ".[piper]"                                        # + piper 硬件 SDK
```

- Python 3.11+（`pyproject.toml` 限定 `requires-python = ">=3.11,<3.14"`）。
- 默认使用 conda 环境 `jiuwensymbiosis`（Makefile 默认走 `conda run -n jiuwensymbiosis`）。
- 工具链：`ruff`（唯一格式化 + lint 工具，Black 兼容）、`mypy`（类型检查，结果为建议性、不阻断）、`pytest`。

### 1.2 关键依赖与代理清理

`clear_proxy_env()`（`jiuwensymbiosis/utils/proxy.py`，由 `jiuwensymbiosis.utils` / `jiuwensymbiosis` 导出）**必须在 `import openjiuwen` 之前调用**。HTTP 代理环境变量会导致 `httpx` 依赖 `socksio`，并把 localhost 流量绕到代理，破坏本地 vLLM / 检测调用。`tests/conftest.py` 已为测试自动处理；自写入口脚本时需自行调用。

---

## 二、GitCode 协作流程（Fork & PR）

仓库地址：
- 主仓（upstream）：`git@gitcode.com:openJiuwen/jiuwensymbiosis.git`
- 个人 Fork（origin）：`git@gitcode.com:<你的用户名>/jiuwensymbiosis.git`

### 2.1 生成并配置 SSH Key

GitCode 使用 SSH 推送，需先在本地生成密钥并添加到 GitCode 账户。

```bash
# 1. 生成 ed25519 密钥（推荐；用真实邮箱替换）
ssh-keygen -t ed25519 -C "your_email@example.com"

# 一路回车，默认生成：
#   ~/.ssh/id_ed25519      （私钥，勿外传）
#   ~/.ssh/id_ed25519.pub  （公钥，贴到 GitCode）

# 2. 启动 ssh-agent 并加载私钥
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# 3. 复制公钥内容
cat ~/.ssh/id_ed25519.pub
```

登录 GitCode → **设置 → SSH 密钥 → 添加 SSH 密钥**，粘贴公钥。

```bash
# 4. 验证连通性（看到 "Welcome to GitCode, <用户名>!" 即成功）
ssh -T git@gitcode.com
```

### 2.2 Fork 主仓

登录 GitCode，打开 `openJiuwen/jiuwensymbiosis` 主仓页面 → 右上角 **Fork** → 选择自己的命名空间 → 生成 `<你的用户名>/jiuwensymbiosis`。

### 2.3 Clone 个人 Fork（origin）

```bash
# clone 你自己的 fork，默认远程名为 origin
git clone git@gitcode.com:<你的用户名>/jiuwensymbiosis.git
cd jiuwensymbiosis
```

### 2.4 设置主仓为 upstream 远程

```bash
# 添加主仓为 upstream（仅 fetch，不直接 push）
git remote add upstream git@gitcode.com:openJiuwen/jiuwensymbiosis.git

# 确认远程配置
git remote -v
# origin    git@gitcode.com:<你>/jiuwensymbiosis.git (fetch)
# origin    git@gitcode.com:<你>/jiuwensymbiosis.git (push)
# upstream  git@gitcode.com:openJiuwen/jiuwensymbiosis.git (fetch)
# upstream  git@gitcode.com:openJiuwen/jiuwensymbiosis.git (push)
```

> 约定：**只向 `origin` 推送，从不直接 push 到 `upstream`**。贡献通过 PR 回流。

### 2.5 从主仓同步代码（fetch + rebase）

```bash
# 1. 抓取 upstream 最新引用（不改动工作区）
git fetch upstream

# 2. 切到本地主分支并同步
git checkout main
git rebase upstream/main      # 把本地提交 rebase 到 upstream 最新之上
#   或用 merge：git merge upstream/main

# 3. 把同步后的 main 推到自己的 fork
git push origin main

# 4. 回到功能分支，把主仓更新 rebase 进来（避免后续 PR 落后）
git checkout <feature-branch>
git rebase main
```

**rebase 冲突处理**：

```bash
# 冲突时 Git 会暂停并标记冲突文件
# 1. 手动编辑冲突文件，解决 <<<<<<< / ======= / >>>>>>> 标记
# 2. 标记已解决
git add <已解决的文件>
# 3. 继续 rebase
git rebase --continue
#    跳过当前提交：git rebase --skip
#    放弃 rebase：git rebase --abort
```

> ⚠️ 已推送到 `origin` 并被他人拉取的分支慎用 `rebase`（会改写历史）。功能分支尚属个人开发期时 rebase 安全；公开分支优先用 `merge`。

### 2.6 提交与发起 PR

```bash
# 功能分支开发
git checkout -b feat/<short-description>
# ... 编码 + 提交前检查（见第六节）...
git add <files>
git commit -m "<type>: <subject>"   # type: feat/fix/refactor/docs/test/chore
git push origin feat/<short-description>
```

随后在 GitCode 上发起 **`origin/feat/<...>` → `upstream/main`** 的 Pull Request。约定 commit 前缀建议：`feat` 新功能、`fix` 修复、`refactor` 重构、`docs` 文档、`test` 测试、`chore` 杂项。

---

## 三、开发行为规则（摘自 `.claude/rules/karpathy-principles.md`）

四条贯穿全程的行为准则（`alwaysApply: true`，所有文件生效）：

1. **先思考再编码**：不臆测、不隐藏困惑。假设显式声明；存在多种理解时摆出来再选；有更简方案要直说；不清楚就停下来问。本仓库分层抽象（capability gating、mixin MRO、Card/Config 拆分、safety rails）交互微妙，遇歧义先问。
2. **简洁优先**：用解决问题的最小代码量，不写投机性功能 / 抽象 / 可配置项。新硬件仅需 YAML + 6 个 adapter 文件，在第二个形态出现前不要泛化 mixin 层级。"资深工程师会不会觉得过度复杂？"——会，就简化。
3. **外科式改动**：只动必须动的。不"顺手"改邻近代码 / 注释 / 格式；不重构没坏的东西；匹配既有风格。只清理自己改动产生的孤儿引用，不删既有死代码（可提一句，但不删）。capability mixin 与其 `@robot_tool` 方法跨 MRO 强耦合，改动一行都应能追溯到需求。
4. **目标驱动**：把任务转成可验证目标——"加校验"→"写失败用例再让其通过"；"修 bug"→"写复现测试再修"。多步任务先列计划：`步骤 → verify: 检查`。

---

## 四、代码风格规则（摘自 `.claude/rules/code-style.md`）

作用范围：`jiuwensymbiosis/**/*.py`。

- Python 3.11+，行宽 **120**（与 `pyproject.toml` 的 `[tool.ruff] line-length` 一致）。
- `ruff` 是 lint 与 format 的**唯一**工具（Black 兼容，无需单独装 `black`）。lint：`ruff check .`；格式化：`ruff format .`；自动修复：`ruff check --fix .`。
- 新公共 API 加类型注解；docstring 与同模块风格对齐。
- **禁用 `print()`**：库代码用 `get_logger(name)`（`jiuwensymbiosis.utils.logging`），统一走 `configure_logging`，确保 `TraceLogHandler` 与文件 handler 正确挂载。遗留 `logging.getLogger(__name__)` 仍合法，新代码优先 `get_logger`。
- 异步安全：库代码保持 async-safe；阻塞 I/O 在异步路径中慎用。本仓库多为同步（硬件 I/O 天然阻塞），不要无理由给 adapter/driver 撒 `async`。
- 命名遵循 PEP 8；capability 字符串用点号 `"<域>.<动词>"`（如 `motion.cartesian`、`grasp.suction`、`vision.detection`）；Config 类型 `<Feature>Config`，env 子类 `BaseRobotEnv`，api 子类 `BaseRobotApi`，driver 子类 `RobotDriver`。
- 导入用绝对导入；禁 wildcard import；分组 stdlib / 三方 / 本地；`clear_proxy_env()` 必须在 `import openjiuwen` 前调用。
- 一模块一公共类优先；私有细节以 `_` / `__` 起名；`__init__.py` 仅导出公共面、保持精简。

---

## 五、安全规则（摘自 `.claude/rules/security.md`）

作用范围：`jiuwensymbiosis/**/*.py`、`configs/**/*.yaml`。本仓库无 agent-core 那类沙箱/注入面，安全重点在**凭据、物理安全、依赖审查、代理清理**。

- **凭据**：永不在源码 / YAML 硬编码 API key、token、真实硬件端点；凭据一律来自环境变量或运行时配置。测试与 `--mock` 路径用 `build_mock_model()`（`jiuwensymbiosis/agent/mock_model.py`）/ `MockDriver` / `MockArmEnv`。
- **物理安全（最重要）**：
  - 永不绕过 `SafetyRail`：Z 下限 `z_min_safe`、XY 工作空间边界、关节软限位 `joint_limits` 分别在 `goto_xyzr` / `goto_pose` / `move_joint` 前运行，禁止直接调驱动运动方法跳过 rail。
  - `z_min_safe` 是硬下限，env 子类须如实反映真实机械臂碰撞极限，**不得**为"让测试过"而设宽松值。`joint_limits` 同理：限位值以官方手册为准，未配置 `joint_limits`（`env.joint_limits is None`）时 SafetyRail 跳过越限检查但保留 q 缺失/类型/finite 检查——**不要**为"让测试过"而硬编码未核实的宽限位。
  - 保留 `RecoveryRail`（失败自动回零 + 释放末端）；不要以吞异常方式跳过恢复。
  - 速度 / 力限放驱动层（`lowlevel.py`）在硬件边界强制，而非 Python 层"尽力而为"。
- **`.env` 与代理**：`.env` / `.env.*` 不得提交（已 gitignore）；`clear_proxy_env()` 必须在 `import openjiuwen` 前调用。
- **依赖审查**：新增依赖前评审 `pyproject.toml` 与安全影响；新网络面依赖需评审；合并前跑 `pip-audit`。
- **高敏区**：`jiuwensymbiosis/rails/`、`adapters/*/lowlevel.py`、`serving/` 的改动需额外评审与测试。提交前完整清单见 `skills/security-review`。

---

## 六、测试规则（摘自 `.claude/rules/testing.md`）

作用范围：`tests/**/*.py`。

- 单测路径镜像源码：`jiuwensymbiosis/tools/build_robot_tools.py` → `tests/unit_tests/tools/test_build_robot_tools.py`。
- `tests/unit_tests/`：快、确定性、无硬件/GPU、CI 跑；`tests/integration/`：需真硬件/GPU/外部服务，CI 常跳过；`tests/mocks/`：共享 `MockApi` / `MockEnv` / `MockDriver` / `MockScene`，单测用它保持无硬件。
- 选型：碰 serial/CAN/socket / 真相机 / 检测子进程 → integration；单函数 / mixin / rail 隔离 → unit（用 `MockEnv` / `MockApi`）；改 capability gating 或 tool emission → 在 `tests/unit_tests/api/` 与 `tests/unit_tests/tools/` 补测。
- `pytest` + `asyncio_mode = "auto"`（无需 `@pytest.mark.asyncio` 模板）；`pytest-mock` 可用，优先 `mocker` fixture；测试类命名 `Test<Feature>`。
- 凭据：库代码无真实凭据面，保持如此；测试 LLM 用离线 mock 模型（`build_mock_model()`，来自 `jiuwensymbiosis.agent.mock_model`），通过 `RobotAgentConfig(model=build_mock_model())` 传入 `build_robot_agent`，禁硬编码真实硬件端点。
- 新公共 API（新 `@robot_tool` / mixin 方法 / env 属性）需对应测试更新；用户可见行为变更需同步更新 `examples/` 与 `docs/`。

运行（用 Makefile 目标，默认走 conda 环境 `jiuwensymbiosis`）：

```bash
make test        # pytest tests/unit_tests/（无硬件/GPU，CI 跑）
make test-all    # 全量 pytest（含 integration，通常跳过）

# Makefile 不提供单文件 / 过滤测试的便捷目标，直接用底层 pytest：
pytest tests/unit_tests/tools/test_build_robot_tools.py  # 单文件
pytest -k "test_capabilities"                            # 按名过滤

# adapter 相关（脚本，非 make 目标）
python scripts/smoke_test_adapter.py                                          # adapter 运行时冒烟
python scripts/validate_adapter.py --module jiuwensymbiosis.adapters.<name>   # adapter 静态检查
```

---

## 七、提交前检查清单

全部用 Makefile 目标，按顺序执行，**全绿后再 `git commit`**。Makefile 默认走 conda 环境 `jiuwensymbiosis`（`make help` 查全部目标）。

可用目标（见 `Makefile`）：

| 目标 | 作用 |
|---|---|
| `make fix` | `ruff format` + `ruff check --fix`（自动修复格式与可修 lint） |
| `make format` | `ruff format --check`（仅检查格式，不改文件） |
| `make lint` | `ruff check --show-fixes`（仅 lint，提示可修项） |
| `make type-check` | `mypy`（仅类型检查，advisory，不阻断） |
| `make check` | `ruff format --check` + `ruff check` + `mypy` 一键自检（mypy advisory） |
| `make test` | `pytest tests/unit_tests/`（无硬件/GPU） |
| `make test-all` | 全量 `pytest`（含 integration，常跳过） |

```bash
# 0. 暂存待提交文件（Makefile 默认检查 staged .py；无暂存改动会报错并提示）
git add <files>

# 1. 自动修复：格式化 + 可修复 lint（会改文件，需重新暂存）
make fix

# 2. 一键自检：format check + lint + mypy（mypy 仅建议性，不阻断）
make check

# 3. 单元测试
make test

# 4.（按需）全量测试（含 integration，通常跳过）
make test-all

# 5.（按需）adapter 改动 → 脚本冒烟（非 make 目标）
python scripts/validate_adapter.py --module jiuwensymbiosis.adapters.<name>
python scripts/smoke_test_adapter.py
python examples/piper_pick_demo.py --config configs/piper/piper.yaml --mock  # 端到端冒烟（无硬件/LLM）

# 6.（新增依赖时）依赖审计
pip-audit

# 全绿后提交（fix 改过文件，重新暂存）
git add <files>
git commit -m "<type>: <subject>"
git push origin <branch>
```

> `make` 的检查目标仅作用于**已暂存的 .py 文件**（`--diff-filter=ACMR`）。若暂存为空会报 `NOTE: no staged .py changes` 并退出 1——先 `git add` 再跑，或用 `COMMITS=N` 检查最近 N 个提交。
>
> ⚠️ `make check` / `format` / `lint` / `type-check` 的命令在 Makefile 中带 `-` 前缀（GNU make 的"忽略失败"语义），ruff/mypy 报错时 `make` 仍返回 0；唯一 `exit 1` 的是暂存为空时的前置检查。

检查范围与环境的灵活性：

| 场景 | 命令 |
|---|---|
| 检查已暂存文件（默认） | `make check` |
| 检查最近 N 个提交的改动 | `make check COMMITS=1` |
| 用 conda 环境（默认） | `make check CONDA_ENV=jiuwensymbiosis` |
| 用 PATH 工具（不走 conda） | `make check CONDA_ENV=` |
| 只看格式是否合规 | `make format` |
| 只跑 lint | `make lint` |
| 只跑类型检查 | `make type-check` |

### 7.1 文档更新

提交前自检：是否需要同步文档？

- **用户可见行为变更**（新 `@robot_tool`、新 mixin 方法、新 env 属性、capability 增减、CLI/配置项变化）→ 同步更新 `examples/`、`docs/`，并视情况更新 `AGENTS.md` 与 `CLAUDE.md` 的相关章节。
- **新 adapter** → 更新 `AGENTS.md` "Source Tree Layout"，必要时补 `docs/hardware-porting-guide.md`。
- **新安全/物理安全相关** → 复核 `.claude/rules/security.md` 与 `skills/security-review` 是否需要补充。
- 深参考手册按需查读：`docs/architecture.md`、`docs/hardware-porting-guide.md`、`docs/logging.md`、`docs/trace.md`、`docs/hand-eye-calibration.md`。

---

## 八、Skills 速查（按需调用）

`.claude/skills/` 是按需加载的深度参考，不要每次编辑都翻；遇到对应模式时再查：

| Skill | 何时用 |
|---|---|
| `python-patterns` | Python 惯用法：frozen dataclass、Protocol、异常层级、async、装饰器、包布局 |
| `python-testing` | 深度 pytest：TDD、fixtures、factory fixtures、mocking、async、adapter 冒烟 |
| `security-review` | PR 前清单：secrets、物理安全、subprocess、依赖、log/trace 卫生 |

---

## 九、一页速查

```bash
# === 日常循环 ===
git checkout -b feat/xxx
# ...编码...
git add <files>                    # make 默认检查 staged .py，先暂存
make fix                           # ruff format + ruff check --fix（改文件后重新暂存）
git add <files>
make check                         # format check + lint + mypy(advisory)
make test                          # pytest tests/unit_tests/
# ...按需更新 docs/examples...
git commit -m "feat: xxx"
git push origin feat/xxx
# 在 GitCode 发起 origin/feat/xxx → upstream/main 的 PR

# === 同步主仓 ===
git fetch upstream
git checkout main && git rebase upstream/main && git push origin main
git checkout feat/xxx && git rebase main     # 把主仓更新并进功能分支
```

| 规则 | 要点 |
|---|---|
| 行为 | 先思考、简洁、外科式、目标驱动 |
| 格式 | `ruff`（120 行宽），禁 `print()` 用 `get_logger` |
| 安全 | 不绕 `SafetyRail`，`z_min_safe` 如实，凭据走 env |
| 测试 | 单测镜像源码路径，用 `Mock*` 保持无硬件 |
| 提交 | `make fix` → `make check` → `make test` → 更新文档 → commit |
