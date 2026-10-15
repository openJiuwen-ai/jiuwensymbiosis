# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""本体(机器人)与任务的注册表 —— GUI 多本体/多任务的扩展点。

一个**本体**(``RobotBody``)= 一种机器人形态,知道如何构建它的真机会话,并持有它那份
**与具体任务无关**的默认真机配置(``default_config_relpath``)。一个**任务**
(``TaskDef``)= 一份**本体无关**的意图预设(自然语言指令 + agent 行为默认);
``bodies`` 为空即适用所有本体(如"视觉拾取"可在任意机械臂上执行)。首页据此渲染本体
下拉与任务卡片:选中本体 × 选中任务 = 用该本体的配置执行该意图。

**数据化**:本体/任务清单是数据,不硬编码——启动时从随包只读数据文件
``gui/data/{bodies,tasks}.yaml`` 加载,任务再与用户目录 ``~/.jiuwensymbiosis/gui/tasks.yaml``
合并(用户可覆盖同 key、可「另存为新任务」)。数据文件与面向 terminal 的 ``configs/``
分开,属 GUI 内部资产。

**约定式会话构建**:本体只需在数据里声明 ``adapter`` 名,真机会话由约定推导——
在 ``jiuwensymbiosis.simulators`` 或 ``jiuwensymbiosis.adapters`` 下惰性 import
``<adapter>.build_<adapter>_session`` 并 ``from_dict`` 构建。故"接入一种新硬件"=
6 个适配器文件 + 一条 ``bodies.yaml`` 数据,注册表零改动。
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import yaml

import jiuwensymbiosis
from jiuwensymbiosis.agent.session import RobotSession
from jiuwensymbiosis.backends import backend_module_name

logger = logging.getLogger(__name__)

__all__ = [
    "RobotBody",
    "TaskDef",
    "configs_dir",
    "user_data_dir",
    "list_bodies",
    "get_body",
    "list_tasks",
    "get_task",
    "tasks_for_body",
    "calibration_profile",
    "alternate_configs",
    "is_body_config_text",
    "body_config_path",
    "save_body_config",
    "add_user_task",
]


def _repo_root() -> Path:
    """定位仓库根(可编辑安装下 ``<repo>/jiuwensymbiosis/__init__.py`` 的上两级)。"""
    return Path(jiuwensymbiosis.__file__).resolve().parent.parent


def configs_dir() -> Path:
    """返回 ``configs/`` 目录(优先仓库内,回落到当前工作目录)。"""
    root = _repo_root() / "configs"
    if root.is_dir():
        return root
    return Path.cwd() / "configs"


def user_data_dir() -> Path:
    """GUI 用户数据目录(存用户「另存为新任务」的任务与配置)。"""
    return Path.home() / ".jiuwensymbiosis" / "gui"


@dataclass(frozen=True)
class RobotBody:
    """一种机器人本体的注册项。

    Attributes:
        key: 稳定标识(下拉/任务引用用)。
        display_name: 界面显示名。
        adapter: 后端包名段,从 ``simulators`` 或 ``adapters`` 解析。标定工具据此
            找到该本体的标定包装器与标定档案,无需另立映射。
        default_config_relpath: 本体的、**与具体任务无关**的**默认**真机配置(相对 ``configs/``
            或绝对路径);运行时套上所选任务的指令与 agent 默认。用户后续可指向别的配置。
        build_real_session: 传入界面编辑过的配置 dict,返回真机会话(惰性导入硬件依赖);
            关键字 ``include_sidecars=False`` 用于只碰机械臂、不需要检测服务的工具。
    """

    key: str
    display_name: str
    adapter: str
    default_config_relpath: str
    build_real_session: Callable[..., RobotSession]

    def config_path(self) -> Path:
        """返回本体默认配置文件的绝对路径(绝对路径时直接用)。"""
        path = Path(self.default_config_relpath)
        return path if path.is_absolute() else configs_dir() / self.default_config_relpath

    def adapter_module(self) -> str:
        """该本体的适配器包路径(标定子系统据此发现标定包装器)。"""
        return backend_module_name(self.adapter)


@dataclass(frozen=True)
class TaskDef:
    """一个**本体无关**的任务意图预设。

    任务只描述"做什么"(指令 + agent 行为),不含任何本体专属配置——配置由
    所选本体(``RobotBody.default_config_relpath``)提供。这样"夹香蕉/视觉拾取"这类意图可在任意
    机械臂上执行,无需为每种本体重复一份。

    Attributes:
        key: 稳定标识(唯一)。
        display_name / description: 界面文案。
        bodies: 适用的本体 key;**空=适用所有本体**,非空则限定(能力天然不匹配时的逃生阀)。
        default_query: 缺省任务指令(配置无 prompt 时用)。
        agent_defaults: 本任务在配置里缺省启用的 ``agent.*`` 项;配置未显式设置时由 GUI
            填入,使界面显示 = 实际运行。
    """

    key: str
    display_name: str
    description: str
    bodies: tuple[str, ...] = ()
    default_query: str = ""
    agent_defaults: dict = field(default_factory=dict)


# ------------------------------------------------------------------ 约定式会话构建
def _real_session_builder(adapter: str) -> Callable[..., RobotSession]:
    """按约定返回某 adapter 的真机会话构建器(惰性导入,无 SDK 时仅在真跑时才失败)。

    约定:``simulators`` 或 ``adapters`` 下的后端包导出 ``build_<adapter>_session``,后者
    带 ``.from_dict``(见 ``adapters/_common/builder.make_builder``)。直接采用**界面编辑过
    的配置 dict**(而非重新读盘),使配置页改动对真机运行同样生效。
    """

    def build(config_data: dict, *, include_sidecars: bool = True) -> RobotSession:
        module = importlib.import_module(backend_module_name(adapter))
        factory = getattr(module, f"build_{adapter}_session")
        return cast(RobotSession, factory.from_dict(config_data, include_sidecars=include_sidecars))

    return build


def _adapter_available(adapter: str) -> bool:
    """该 adapter 包是否存在(不执行其重依赖;仅 find_spec 定位)。"""
    try:
        return importlib.util.find_spec(backend_module_name(adapter)) is not None
    except (ImportError, ValueError):
        return False


# ------------------------------------------------------------------ 数据加载
def _data_dir() -> Path:
    """随包的内置数据目录(``gui/data``)。"""
    return Path(__file__).resolve().parent / "data"


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    """读一个 YAML 映射;文件缺失/损坏/非映射都返回空(记 warning,不抛)。"""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        logger.warning("跳过损坏的数据文件 %s: %s", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def _body_from_dict(entry: dict) -> RobotBody | None:
    """把一条本体数据构造成 ``RobotBody``;adapter 包不存在则跳过(记 warning)。"""
    adapter = str(entry.get("adapter", ""))
    if not adapter or not _adapter_available(adapter):
        logger.warning("本体 %r 引用了未知 adapter %r,跳过", entry.get("key"), adapter)
        return None
    return RobotBody(
        key=str(entry["key"]),
        display_name=str(entry.get("display_name", entry["key"])),
        adapter=adapter,
        default_config_relpath=str(entry.get("default_config_relpath", "")),
        build_real_session=_real_session_builder(adapter),
    )


def _task_from_dict(entry: dict) -> TaskDef:
    """把一条任务数据构造成 ``TaskDef``(``bodies`` 缺省=适用所有本体)。"""
    return TaskDef(
        key=str(entry["key"]),
        display_name=str(entry.get("display_name", entry["key"])),
        description=str(entry.get("description", "")),
        bodies=tuple(str(b) for b in entry.get("bodies") or ()),
        default_query=str(entry.get("default_query", "")),
        agent_defaults=dict(entry.get("agent_defaults") or {}),
    )


def _load_bodies() -> dict[str, RobotBody]:
    """从随包数据文件加载本体;空/全失败则回落到内置最小默认。"""
    bodies: dict[str, RobotBody] = {}
    for entry in _read_yaml_mapping(_data_dir() / "bodies.yaml").get("bodies") or []:
        if isinstance(entry, dict) and "key" in entry:
            body = _body_from_dict(entry)
            if body is not None:
                bodies[body.key] = body
    return bodies or _fallback_bodies()


def _load_tasks() -> dict[str, TaskDef]:
    """内置 + 用户任务合并加载(用户可覆盖同 key);空/全失败则回落到内置最小默认。"""
    tasks: dict[str, TaskDef] = {}
    for source in (_data_dir() / "tasks.yaml", user_data_dir() / "tasks.yaml"):
        for entry in _read_yaml_mapping(source).get("tasks") or []:
            if isinstance(entry, dict) and "key" in entry:
                task = _task_from_dict(entry)
                tasks[task.key] = task
    return tasks or _fallback_tasks()


def _fallback_bodies() -> dict[str, RobotBody]:
    """最后兜底:数据文件读不出时,至少给一个 piper 本体,保证界面可用。"""
    return {
        "piper": RobotBody(
            key="piper",
            display_name="Piper 六轴机械臂",
            adapter="piper",
            default_config_relpath="piper/piper.yaml",
            build_real_session=_real_session_builder("piper"),
        )
    }


def _fallback_tasks() -> dict[str, TaskDef]:
    """最后兜底:数据文件读不出时,至少给一个本体无关的拾放任务。"""
    return {
        "pick_box": TaskDef(
            key="pick_box",
            display_name="拾取盒子",
            description="把黑色盒子抓起来放到白色盒子上。",
            default_query="把黑色盒子抓起来放到白色盒子上。",
            agent_defaults={"enable_skill": True},
        )
    }


_BODIES: dict[str, RobotBody] = _load_bodies()
_TASKS: dict[str, TaskDef] = _load_tasks()


# ------------------------------------------------------------------ 查询接口
def list_bodies() -> list[RobotBody]:
    """返回所有已注册本体。"""
    return list(_BODIES.values())


def get_body(key: str) -> RobotBody:
    """按 key 取本体;不存在时抛 ``KeyError``。"""
    return _BODIES[key]


def list_tasks() -> list[TaskDef]:
    """返回所有已注册任务。"""
    return list(_TASKS.values())


def get_task(key: str) -> TaskDef:
    """按 key 取任务;不存在时抛 ``KeyError``。"""
    return _TASKS[key]


def tasks_for_body(body_key: str) -> list[TaskDef]:
    """返回适用于某本体的任务(``bodies`` 为空的本体无关任务对所有本体可见)。"""
    return [t for t in _TASKS.values() if not t.bodies or body_key in t.bodies]


def calibration_profile(body_key: str) -> dict[str, Any] | None:
    """返回该本体的内置标定档案;未接入标定的机型返回 ``None``。

    档案见 ``gui/data/calibration_profiles.yaml``,含 ``profile``(标定 YAML 的
    ``calibration:`` 段)、``env_overrides``(标定用软限位放宽),以及兜底的
    ``camera_mount`` 与可选的 ``output_relpath``(缺省按 adapter+mount 约定推导)。
    """
    adapter = get_body(body_key).adapter
    entry = (_read_yaml_mapping(_data_dir() / "calibration_profiles.yaml").get("profiles") or {}).get(adapter)
    return dict(entry) if isinstance(entry, dict) else None


def _is_body_config_mapping(data: dict[str, Any]) -> bool:
    """该配置映射能否直接当本体配置用:顶层 ``env.cfg.low_level`` 是映射。

    判据取自 ``<Adapter>Config.from_dict`` 与界面表单共同依赖的嵌套形状——标定 json、
    纯任务片段等据此排除。
    """
    env = data.get("env")
    cfg = env.get("cfg") if isinstance(env, dict) else None
    low_level = cfg.get("low_level") if isinstance(cfg, dict) else None
    return isinstance(low_level, dict)


def is_body_config_text(text: str) -> bool:
    """YAML 文本能否直接当本体配置用(供拖入文件时判定)。"""
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        logger.debug("拖入的 YAML 解析失败: %s", exc)
        return False
    return isinstance(data, dict) and _is_body_config_mapping(data)


def _is_body_config(path: Path) -> bool:
    return _is_body_config_mapping(_read_yaml_mapping(path))


def body_config_path(body_key: str, name: str) -> Path:
    """该本体配置目录下、名为 ``name`` 的配置文件路径。

    ``name`` 只取文件名部分(挡住 ``../`` 之类的目录穿越),缺扩展名时补 ``.yaml``。
    """
    stem = Path(name.strip()).name
    if not stem:
        raise ValueError("配置名不能为空。")
    if not stem.endswith((".yaml", ".yml")):
        stem = f"{stem}.yaml"
    return get_body(body_key).config_path().parent / stem


def save_body_config(body_key: str, name: str, text: str) -> Path:
    """把 YAML 文本存进该本体的配置目录,返回落盘路径(同名即覆盖)。"""
    path = body_config_path(body_key, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def alternate_configs(body_key: str) -> list[Path]:
    """返回该本体默认配置**同目录**下、其余可用作本体配置的 YAML(按文件名排序)。

    供主页「配置文件」下拉在「默认」之外列出候选(如本机专属的 ``*.local.yaml``)。
    """
    default = get_body(body_key).config_path()
    directory = default.parent
    if not directory.is_dir():
        return []
    return [
        path
        for path in sorted(directory.iterdir())
        if path.suffix in (".yaml", ".yml") and path.name != default.name and path.is_file() and _is_body_config(path)
    ]


def add_user_task(
    *, display_name: str, description: str, default_query: str = "", bodies: tuple[str, ...] = ()
) -> TaskDef:
    """把一个用户新任务(意图)落盘并注册(供「另存为新任务」)。

    任务本体无关——只存"做什么"(指令 + 适用本体范围);配置属本体,不在此捆绑。做两件事:
    ①追加一条任务到用户 ``tasks.yaml``;②注册进内存 ``_TASKS``。``key`` 自动生成唯一 id。
    返回新建的 ``TaskDef``。
    """
    import uuid

    udir = user_data_dir()
    udir.mkdir(parents=True, exist_ok=True)
    key = f"user_{uuid.uuid4().hex[:8]}"

    entry: dict[str, Any] = {
        "key": key,
        "display_name": display_name,
        "description": description,
        "default_query": default_query,
    }
    if bodies:
        entry["bodies"] = list(bodies)
    tasks_path = udir / "tasks.yaml"
    existing = _read_yaml_mapping(tasks_path).get("tasks")
    tasks_list = list(existing) if isinstance(existing, list) else []
    tasks_list.append(entry)
    tasks_path.write_text(
        yaml.safe_dump({"tasks": tasks_list}, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    task = _task_from_dict(entry)
    _TASKS[task.key] = task
    return task
