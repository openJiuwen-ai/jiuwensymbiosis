"""Provider presets and explicit model discovery, independent of NiceGUI."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING
from urllib.parse import urljoin, urlsplit

import httpx

if TYPE_CHECKING:
    from jiuwensymbiosis.agent.config import ModelSpec

__all__ = ["deepseek_model_spec", "fetch_model_options", "model_list_url", "model_preset"]

_DEEPSEEK_BASE = "https://api.deepseek.com"
_COMPATIBLE_PROVIDERS = {"OpenAI", "DeepSeek", "SiliconFlow", "OpenRouter", "DashScope"}


def model_preset(spec: ModelSpec) -> str:
    """Identify the official DeepSeek preset without inferring it from model names."""
    if spec.provider in {"OpenAI", "DeepSeek"} and spec.api_base.rstrip("/") in {
        _DEEPSEEK_BASE,
        urljoin(_DEEPSEEK_BASE, "v1"),
    }:
        return "deepseek"
    return "custom"


def deepseek_model_spec(current: ModelSpec) -> ModelSpec:
    """Apply official connection settings; retain credentials only for the same service."""
    return replace(
        current,
        provider="OpenAI",
        api_base=_DEEPSEEK_BASE,
        api_key=current.api_key if model_preset(current) == "deepseek" else "",
        model_name="deepseek-flash",
        verify_ssl=True,
    )


def model_list_url(api_base: str) -> str:
    """Append /models to the configured API base, retaining its version/path prefix."""
    base = api_base.strip().rstrip("/")
    try:
        url = urlsplit(base)
        valid = url.scheme in {"http", "https"} and bool(url.hostname)
        _ = url.port
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("请填写有效的 http(s) 模型 API 基础地址。")
    if url.username or url.password:
        raise ValueError("API 地址不能包含用户名或密码；请单独填写 API Key。")
    if url.query or url.fragment:
        raise ValueError("API 地址不能包含用户名、密码、查询参数或片段；请单独填写 API Key。")
    if url.path.rstrip("/").endswith(("/chat/completions", "/responses", "/models")):
        raise ValueError("请填写 API 基础地址，不要追加 /chat/completions、/responses 或 /models。")
    return urljoin(f"{base}/", "models")


def fetch_model_options(spec: ModelSpec) -> dict[str, str]:
    """Return API model IDs and display labels, never exposing response bodies or keys in errors."""
    if spec.provider not in _COMPATIBLE_PROVIDERS:
        raise ValueError("此接口适配器暂不支持自动获取模型，请手动填写模型名称。")
    endpoint = model_list_url(spec.api_base)
    if not spec.api_key.strip():
        raise ValueError("请先填写 API Key；本地无鉴权服务可填 EMPTY。")
    try:
        with httpx.Client(timeout=10.0, verify=spec.verify_ssl, trust_env=False, follow_redirects=False) as client:
            response = client.get(endpoint, headers={"Authorization": f"Bearer {spec.api_key}"})
    except httpx.TimeoutException:
        raise ValueError("获取模型列表超时，请检查网络或稍后重试。") from None
    except httpx.RequestError:
        raise ValueError("无法连接模型 API，请检查地址、网络和证书配置。") from None
    if response.status_code in {401, 403}:
        raise ValueError("模型接口鉴权失败，请检查 API Key 和账户权限。")
    if response.status_code in {404, 405}:
        raise ValueError("当前地址未提供 /models 接口，请核对 API 基础地址或手动填写模型名称。")
    if 300 <= response.status_code < 400:
        raise ValueError("模型接口返回重定向，请直接填写正确的 API 基础地址。")
    if response.status_code == 429:
        raise ValueError("模型接口请求过于频繁，请稍后重试。")
    if not response.is_success:
        raise ValueError(f"获取模型列表失败（HTTP {response.status_code}），请检查服务状态。")
    try:
        payload = response.json()
    except ValueError:
        raise ValueError("模型接口没有返回有效 JSON，请手动填写模型名称。") from None
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError("模型列表格式不兼容：应包含 data 数组。")
    options: dict[str, str] = {}
    for model in payload["data"]:
        if not isinstance(model, dict) or not isinstance(model.get("id"), str) or not model["id"].strip():
            raise ValueError("模型列表格式不兼容：每项必须包含非空的模型 id。")
        model_id = model["id"]
        name = model.get("name")
        options[model_id] = f"{name} ({model_id})" if isinstance(name, str) and name and name != model_id else model_id
    if not options:
        raise ValueError("接口返回的模型列表为空；请检查账户权限，或手动填写模型名称。")
    return dict(sorted(options.items()))
