# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""本体(机器人)与任务的注册表 —— GUI 多本体/多任务的扩展点。

一个**本体**(``RobotBody``)= 一种机器人形态,知道如何构建它的真机会话与模拟
会话;一个**任务**(``TaskDef``)= 绑定某本体的一份预设(配置文件 + 默认指令 +
模拟脚本)。首页据此渲染本体下拉与任务卡片。

**数据化**:任务/本体清单是数据,不硬编码——启动时从随包只读数据文件
``gui/data/{bodies,tasks}.yaml`` 加载,并与用户目录 ``~/.jiuwensymbiosis/gui/tasks.yaml``
合并(用户可覆盖同 key、可「另存为新任务」)。数据文件与面向 terminal 的 ``configs/``
分开,属 GUI 内部资产。

唯一留在代码里的是 ``_ADAPTERS``:``adapter 名 → (模拟会话构建, 真机会话构建)``——
这两个是必须 import 适配器类的**可执行回调**,天然属于代码;数据里的本体用 ``adapter``
名引用它。故"接入一种新硬件"仍需写适配器代码,但其展示元数据已数据化。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import yaml

import jiuwensymbiosis
from jiuwensymbiosis.agent.session import RobotSession
from jiuwensymbiosis.gui.mock_sessions import build_mock_robot_session

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
        description: 一句话说明。
        capability_badges: 首页展示的能力徽章(纯文案)。
        build_mock_session: 返回一个未连接的模拟会话。
        build_real_session: 传入界面编辑过的配置 dict,返回真机会话(惰性导入硬件依赖)。
    """

    key: str
    display_name: str
    description: str
    capability_badges: tuple[str, ...]
    build_mock_session: Callable[[], RobotSession]
    build_real_session: Callable[[dict], RobotSession]


@dataclass(frozen=True)
class TaskDef:
    """一个任务预设。

    Attributes:
        key: 稳定标识(唯一)。
        body_key: 所属本体。
        display_name / description: 界面文案。
        config_relpath: 配置文件路径(相对 ``configs/`` 或绝对路径)。
        default_query: 缺省任务指令(配置无 prompt 时用)。
        mock_script: 模拟运行时脚本化模型逐轮返回的工具序列(仅内置演示任务需要)。
        mock_final_text: 脚本走完后的收尾文本。
        agent_defaults: 本任务在配置里缺省启用的 ``agent.*`` 项;配置未显式设置时由 GUI
            填入,使界面显示 = 实际运行。
    """

    key: str
    body_key: str
    display_name: str
    description: str
    config_relpath: str
    default_query: str = ""
    mock_script: tuple[dict, ...] = ()
    mock_final_text: str = "任务完成。"
    agent_defaults: dict = field(default_factory=dict)

    def config_path(self) -> Path:
        """返回配置文件的绝对路径(``config_relpath`` 为绝对路径时直接用)。"""
        path = Path(self.config_relpath)
        return path if path.is_absolute() else configs_dir() / self.config_relpath


# ------------------------------------------------------------------ 适配器注册表(代码)
def _build_piper_real_session(config_data: dict) -> RobotSession:
    """真机 Piper 会话:惰性导入适配器,避免无 piper_sdk 时导入本模块即失败。

    直接采用**界面编辑过的配置 dict**(经 ``from_dict``,而非重新读盘),使配置页里
    改的相机序列号 / 运动速度 / 工具偏置等对真机运行同样生效(否则真机会话读的是磁盘
    原文件,界面改动全被丢弃)。
    """
    from jiuwensymbiosis.adapters.piper import build_piper_session

    return cast(RobotSession, build_piper_session.from_dict(config_data))


# adapter 名 → (模拟会话构建, 真机会话构建)。数据文件里的本体用 adapter 名引用这里。
_ADAPTERS: dict[str, tuple[Callable[[], RobotSession], Callable[[dict], RobotSession]]] = {
    "piper": (lambda: build_mock_robot_session("piper_mock"), _build_piper_real_session),
}


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
    """把一条本体数据构造成 ``RobotBody``;引用未知 adapter 则跳过(记 warning)。"""
    adapter = entry.get("adapter")
    builders = _ADAPTERS.get(str(adapter))
    if builders is None:
        logger.warning("本体 %r 引用了未知 adapter %r,跳过", entry.get("key"), adapter)
        return None
    build_mock, build_real = builders
    return RobotBody(
        key=str(entry["key"]),
        display_name=str(entry.get("display_name", entry["key"])),
        description=str(entry.get("description", "")),
        capability_badges=tuple(str(b) for b in entry.get("capability_badges", ())),
        build_mock_session=build_mock,
        build_real_session=build_real,
    )


def _task_from_dict(entry: dict) -> TaskDef:
    """把一条任务数据构造成 ``TaskDef``。"""
    return TaskDef(
        key=str(entry["key"]),
        body_key=str(entry["body"]),
        display_name=str(entry.get("display_name", entry["key"])),
        description=str(entry.get("description", "")),
        config_relpath=str(entry.get("config_relpath", "")),
        default_query=str(entry.get("default_query", "")),
        mock_script=tuple(entry.get("mock_script") or ()),
        mock_final_text=str(entry.get("mock_final_text", "任务完成。")),
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
            if isinstance(entry, dict) and "key" in entry and "body" in entry:
                task = _task_from_dict(entry)
                tasks[task.key] = task
    return tasks or _fallback_tasks()


def _fallback_bodies() -> dict[str, RobotBody]:
    """最后兜底:数据文件读不出时,至少给一个 piper 本体,保证界面可用。"""
    build_mock, build_real = _ADAPTERS["piper"]
    return {
        "piper": RobotBody(
            key="piper",
            display_name="Piper 六轴机械臂(示例)",
            description="AgileX Piper 六轴机械臂 + 平行夹爪 + 腕部相机(示例本体)。",
            capability_badges=("运动", "夹爪", "视觉"),
            build_mock_session=build_mock,
            build_real_session=build_real,
        )
    }


def _fallback_tasks() -> dict[str, TaskDef]:
    """最后兜底:数据文件读不出时,至少给一个 pick_box 任务(无模拟脚本)。"""
    return {
        "pick_box": TaskDef(
            key="pick_box",
            body_key="piper",
            display_name="拾取盒子",
            description="把黑色盒子抓起来放到白色盒子上。",
            config_relpath="piper/piper.yaml",
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
    """返回属于某本体的任务。"""
    return [t for t in _TASKS.values() if t.body_key == body_key]


def add_user_task(*, display_name: str, description: str, body_key: str, config_yaml: str) -> TaskDef:
    """把一个用户新任务落盘并注册(供「另存为新任务」)。

    做三件事:①把配置写成用户目录下的一份 yaml;②追加一条任务到用户 ``tasks.yaml``;
    ③注册进内存 ``_TASKS``。``key`` 自动生成唯一 id;``config_relpath`` 存该配置 yaml 的
    绝对路径。返回新建的 ``TaskDef``。
    """
    import uuid

    udir = user_data_dir()
    (udir / "configs").mkdir(parents=True, exist_ok=True)
    key = f"user_{uuid.uuid4().hex[:8]}"
    config_file = udir / "configs" / f"{key}.yaml"
    config_file.write_text(config_yaml, encoding="utf-8")

    entry = {
        "key": key,
        "body": body_key,
        "display_name": display_name,
        "description": description,
        "config_relpath": str(config_file),  # 绝对路径,config_path() 会直接采用
    }
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
