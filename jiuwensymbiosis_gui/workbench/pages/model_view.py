"""One secure connection-profile editor for every WebUI agent entry point."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace
from typing import Any
from urllib.parse import urlsplit

from nicegui import run, ui

from jiuwensymbiosis.agent.config import ModelSpec
from jiuwensymbiosis.utils.logging import get_logger
from jiuwensymbiosis_gui.workbench.model_catalog import deepseek_model_spec, fetch_model_options, model_preset
from jiuwensymbiosis_gui.workbench.model_preferences import ModelPreferenceError, ModelPreferences

__all__ = ["ModelView"]

logger = get_logger(__name__)


class ModelView:
    """Keep editable drafts separate from the connection used by new runs."""

    def __init__(self, *, model_config: Callable[[], dict[str, Any]]) -> None:
        self._model_config = model_config
        self._fetching_models = False
        self._restoring_model = True
        self._saving_model = False
        self._preferences = ModelPreferences()
        self._updating_model = True
        self._model = ModelSpec(**model_config())
        self._active: ModelSpec | None = None
        self._key_origin = self._api_origin(self._model.api_base)
        ui.label("统一模型设置").classes("text-2xl font-semibold")
        ui.label().classes("text-sm text-primary").bind_text_from(
            self,
            "_active",
            backward=lambda spec: (
                f"当前生效：{spec.provider} / {spec.model_name}" if spec else "当前使用各任务配置的模型"
            ),
        )
        with ui.column().classes("w-full max-w-3xl gap-3"):
            self._preset = ui.select(
                {"custom": "自定义 / 当前配置", "deepseek": "DeepSeek 官方"},
                label="服务商预设",
                value=model_preset(self._model),
                on_change=self._apply_preset,
            ).classes("w-full")
            with ui.element("div").classes("w-full grid grid-cols-1 md:grid-cols-2 gap-3"):
                self._provider = ui.input(
                    "接口适配器", value=self._model.provider, on_change=self._model_endpoint_changed
                )
                self._base = ui.input(
                    "模型 API 基础地址", value=self._model.api_base, on_change=self._model_endpoint_changed
                )
                self._name = ui.select(
                    [self._model.model_name] if self._model.model_name else [],
                    label="模型名称",
                    value=self._model.model_name or None,
                    with_input=True,
                    new_value_mode="add-unique",
                )
                self._key = ui.input(
                    "API Key",
                    value=self._model.api_key,
                    password=True,
                    password_toggle_button=True,
                    on_change=self._model_endpoint_changed,
                )
            self._verify_ssl = ui.switch("验证 TLS 证书", value=self._model.verify_ssl)
            ui.label(
                "接口适配器用于选择客户端；OpenAI 也可连接 DeepSeek 等兼容服务。官方 URL 填在 API 基础地址。"
            ).classes("text-xs text-gray-500")
            with ui.row().classes("items-center"):
                self._fetch = ui.button("获取模型列表", icon="download", on_click=self._fetch_models).props(
                    "flat dense"
                )
                self._import = ui.button("读取任务模型到表单", on_click=self._import_model).props("flat dense")
            self._catalog_status = ui.label("可手动输入模型 ID，或填写地址和 Key 后获取当前接口的可用模型。").classes(
                "text-xs text-gray-500"
            )
            with ui.row().classes("items-center"):
                self._apply_model_button = ui.button("仅本次应用", on_click=self._apply_model).props("flat dense")
                self._save_model_button = ui.button("应用并保存模型", icon="save", on_click=self._save_model).props(
                    "flat dense"
                )
                self._task_model_button = ui.button("使用任务自身模型", on_click=self._use_task_model).props(
                    "flat dense"
                )
                self._forget_model_button = ui.button(
                    "忘记已保存设置", icon="delete_outline", on_click=self._forget_model
                ).props("flat dense")
            self._memory_status = ui.label("正在读取上次的模型设置…").classes("text-xs text-gray-500")
            ui.label(
                "应用后，当前页面启动和重新执行的任务使用此模型；正在运行的任务不受影响。仅本次应用不保存 Key，应用并保存使用系统钥匙串，不写回 YAML。"
            ).classes("text-xs text-gray-500")
            ui.label(
                "使用任务自身模型仅取消当前页面的统一覆盖。刷新后重新读取已保存设置；要删除保存记录，请点击忘记已保存设置。"
            ).classes("text-xs text-gray-500")
            ui.label("任务的温度、Token 上限等参数仍来自任务配置。修改表单或获取列表不会立即切换模型。").classes(
                "text-xs text-gray-500"
            )
            ui.label("本地 OpenAI 兼容服务若无需鉴权，API Key 可填 EMPTY。").classes("text-xs text-gray-500")
        self._controls: list[Any] = [
            self._provider,
            self._base,
            self._name,
            self._key,
            self._verify_ssl,
            self._preset,
            self._fetch,
            self._import,
            self._apply_model_button,
            self._save_model_button,
            self._task_model_button,
            self._forget_model_button,
        ]
        self._updating_model = False
        for control in self._controls:
            control.disable()
        ui.timer(0.1, self._restore_model, once=True)

    @property
    def ready(self) -> bool:
        return not self._restoring_model

    def selected(self, default: ModelSpec | None = None) -> ModelSpec:
        """Snapshot the applied connection without copying secrets into task YAML."""
        if not self.ready:
            raise RuntimeError("正在恢复统一模型设置，请稍后再试。")
        base = default if default is not None else ModelSpec(**self._model_config())
        if self._active is None:
            return deepcopy(base)
        return deepcopy(
            replace(
                base,
                **{
                    name: getattr(self._active, name)
                    for name in ("provider", "api_base", "model_name", "api_key", "verify_ssl")
                },
            )
        )

    def _draft_model(self) -> ModelSpec:
        return replace(
            self._model,
            provider=str(self._provider.value or "").strip(),
            api_base=str(self._base.value or "").strip(),
            model_name=str(self._name.value or "").strip(),
            api_key=str(self._key.value or ""),
            verify_ssl=bool(self._verify_ssl.value),
        )

    @staticmethod
    def _api_origin(address: str) -> tuple[str, str | None, int | None] | None:
        try:
            url = urlsplit(address)
            return url.scheme.lower(), url.hostname, url.port or (443 if url.scheme == "https" else 80)
        except ValueError:
            return None

    async def _restore_model(self) -> None:
        try:
            saved = await run.io_bound(self._preferences.load, self._model)
            if saved is not None:
                self._set_model(saved)
                self._active = saved
                self._memory_status.set_text("已恢复上次的模型设置（系统钥匙串）。")
            else:
                self._memory_status.set_text("尚未保存模型设置；可点击保存，下次自动恢复。")
        except ModelPreferenceError as exc:
            self._memory_status.set_text(str(exc))
        finally:
            self._restoring_model = False
            for control in self._controls:
                control.enable()

    async def _remember_model(self, spec: ModelSpec) -> bool:
        try:
            await run.io_bound(self._preferences.save, spec)
            self._active = deepcopy(spec)
            self._memory_status.set_text("已应用到当前页面的新任务，并保存到系统钥匙串；下次打开自动恢复。")
            return True
        except ModelPreferenceError as exc:
            self._memory_status.set_text(str(exc) + " 当前生效模型未改变。")
            return False

    def _apply_model(self) -> None:
        if self._fetching_models or self._restoring_model or self._saving_model:
            return
        spec = self._draft_model()
        try:
            self._preferences.validate(spec)
        except ModelPreferenceError as exc:
            self._memory_status.set_text(str(exc) + " 当前生效模型未改变。")
            return
        self._active = deepcopy(spec)
        self._memory_status.set_text("已仅应用到当前页面的新任务；未保存，刷新后重新读取已保存设置。")

    def _use_task_model(self) -> None:
        if self._fetching_models or self._restoring_model or self._saving_model:
            return
        self._active = None
        self._memory_status.set_text("当前页面的新任务使用各任务自身模型；表单草稿和已保存设置保持不变。")

    async def _save_model(self) -> None:
        if self._fetching_models or self._restoring_model or self._saving_model:
            return
        self._saving_model = True
        for control in self._controls:
            control.disable()
        try:
            await self._remember_model(self._draft_model())
        finally:
            self._saving_model = False
            for control in self._controls:
                control.enable()

    async def _forget_model(self) -> None:
        if self._fetching_models or self._restoring_model or self._saving_model:
            return
        self._saving_model = True
        for control in self._controls:
            control.disable()
        try:
            await run.io_bound(self._preferences.clear)
            self._key.set_value("")
            self._model.api_key = ""
            if self._active is not None:
                self._active = replace(self._active, api_key="")
            self._memory_status.set_text(
                "已忘记保存的模型设置，并清空表单及统一模型 Key；任务自身配置和运行中的任务不变。"
            )
        except ModelPreferenceError as exc:
            self._memory_status.set_text(str(exc))
        finally:
            self._saving_model = False
            for control in self._controls:
                control.enable()

    def _import_model(self) -> None:
        self._set_model(ModelSpec(**self._model_config()))
        ui.notify("已读取到表单，点击“仅本次应用”或“应用并保存模型”后生效。")

    def _set_model(self, spec: ModelSpec) -> None:
        self._updating_model = True
        try:
            self._model = spec
            self._key_origin = self._api_origin(spec.api_base)
            self._provider.set_value(spec.provider)
            self._base.set_value(spec.api_base)
            self._name.set_options(
                {spec.model_name: spec.model_name} if spec.model_name else {}, value=spec.model_name or None
            )
            self._key.set_value(spec.api_key)
            self._verify_ssl.set_value(spec.verify_ssl)
            self._preset.set_value(model_preset(spec))
            self._catalog_status.set_text("可手动输入模型 ID，或填写地址和 Key 后获取当前接口的可用模型。")
        finally:
            self._updating_model = False

    def _apply_preset(self) -> None:
        if not self._updating_model and self._preset.value == "deepseek":
            self._set_model(deepseek_model_spec(self._draft_model()))

    def _model_endpoint_changed(self) -> None:
        if self._updating_model:
            return
        self._updating_model = True
        try:
            spec = self._draft_model()
            origin = self._api_origin(spec.api_base)
            if origin != self._key_origin:
                self._key.set_value("")
                self._verify_ssl.set_value(True)
            self._key_origin = origin
            self._name.set_options(
                {spec.model_name: spec.model_name} if spec.model_name else {}, value=spec.model_name or None
            )
            self._preset.set_value(model_preset(spec))
            self._catalog_status.set_text("地址、接口适配器或 Key 已改变，请重新获取模型列表；也可手动输入。")
        finally:
            self._updating_model = False

    async def _fetch_models(self) -> None:
        if self._fetching_models or self._restoring_model or self._saving_model:
            return
        spec = self._draft_model()
        self._fetching_models = True
        for control in self._controls:
            control.disable()
        self._fetch.props("loading")
        self._catalog_status.set_text("正在获取当前接口的模型列表…")
        try:
            options = await run.io_bound(fetch_model_options, spec)
            count = len(options)
            current = spec.model_name or None
            manual = bool(current and current not in options)
            if current and manual:
                options[current] = f"{current}（手动输入，未在本次列表中）"
            self._name.set_options(options, value=current)
            self._catalog_status.set_text(
                f"接口返回 {count} 个模型，展开模型名称即可选择。"
                + ("当前模型未列出，请确认或另选。" if manual else "")
            )
        except ValueError as exc:
            detail = str(exc).replace(spec.api_key, "***") if spec.api_key else str(exc)
            self._catalog_status.set_text(detail)
        except Exception as exc:
            logger.warning("Model catalog fetch failed (%s)", type(exc).__name__)
            self._catalog_status.set_text("获取模型列表失败，请检查配置；仍可手动填写模型名称。")
        finally:
            self._fetching_models = False
            self._fetch.props(remove="loading")
            for control in self._controls:
                control.enable()
