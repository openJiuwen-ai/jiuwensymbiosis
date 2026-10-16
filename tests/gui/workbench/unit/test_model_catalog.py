"""Model discovery uses the configured endpoint and keeps credentials out of failures."""

from __future__ import annotations

from dataclasses import replace

import httpx
import pytest
import respx

from jiuwensymbiosis.agent.config import ModelSpec
from jiuwensymbiosis_gui.workbench.model_catalog import deepseek_model_spec, fetch_model_options, model_list_url


@pytest.mark.parametrize(
    ("base", "endpoint"),
    [
        ("https://api.deepseek.com", "https://api.deepseek.com/models"),
        ("https://api.deepseek.com/v1/", "https://api.deepseek.com/v1/models"),
        ("http://127.0.0.1:8110/v1", "http://127.0.0.1:8110/v1/models"),
        ("https://gateway.example/custom/api/v4", "https://gateway.example/custom/api/v4/models"),
        ("http://[::1]:8110/custom/v1/", "http://[::1]:8110/custom/v1/models"),
        ("https://gateway.example/team%2Fone/v1/", "https://gateway.example/team%2Fone/v1/models"),
    ],
)
def test_discovery_preserves_base_path_and_returns_api_ids(base, endpoint):
    spec = ModelSpec(api_base=base, api_key="test-credential", verify_ssl=True)
    with respx.mock as router:
        route = router.get(endpoint).mock(
            return_value=httpx.Response(
                200,
                json={"data": [{"id": "other"}, {"id": "deepseek-flash", "name": "DeepSeek-V4.1-Flash"}]},
            )
        )
        options = fetch_model_options(spec)
        assert options == {"deepseek-flash": "DeepSeek-V4.1-Flash (deepseek-flash)", "other": "other"}
        assert route.call_count == 1
        assert route.calls[0].request.headers["Authorization"] == "Bearer test-credential"


def test_preset_clears_another_services_credentials_but_keeps_request_settings():
    current = ModelSpec(api_key="another-services-key", max_tokens=8192, temperature=0.5)
    spec = deepseek_model_spec(current)
    assert spec.api_key == ""
    assert spec.provider == "OpenAI"
    assert spec.api_base == "https://api.deepseek.com"
    assert spec.model_name == "deepseek-flash"
    assert spec.verify_ssl
    assert spec.max_tokens == current.max_tokens
    assert current.api_key == "another-services-key"
    assert deepseek_model_spec(replace(spec, api_key="deepseek-test-key")).api_key == "deepseek-test-key"
    versioned = replace(spec, api_base="https://api.deepseek.com/v1/", api_key="deepseek-test-key")
    assert deepseek_model_spec(versioned).api_key == "deepseek-test-key"


@pytest.mark.parametrize(
    "base",
    [
        "your own api",
        "file:///tmp/key",
        "https://",
        "https://host:bad",
        "https://user:password@example.com/v1",
        "https://example.com/v1?key=secret",
        "https://example.com/#secret",
        "https://example.com/v1/chat/completions",
    ],
)
def test_invalid_or_credential_bearing_base_is_rejected(base):
    with pytest.raises(ValueError):
        model_list_url(base)


@pytest.mark.parametrize("status", [401, 403, 404, 405, 429, 500])
def test_http_failure_never_echoes_body_credentials_or_retries(status):
    spec = ModelSpec(api_base="https://api.deepseek.com", api_key="private-test-key")
    with respx.mock as router:
        route = router.get("https://api.deepseek.com/models").mock(
            return_value=httpx.Response(status, json={"error": {"message": spec.api_key}})
        )
        with pytest.raises(ValueError) as caught:
            fetch_model_options(spec)
        assert spec.api_key not in str(caught.value)
        assert route.call_count == 1


def test_redirect_is_not_followed_with_credentials():
    with respx.mock as router:
        router.get("https://api.deepseek.com/models").mock(
            return_value=httpx.Response(302, headers={"Location": "https://other.example/models"})
        )
        destination = router.get("https://other.example/models").mock(
            return_value=httpx.Response(200, json={"data": []})
        )
        with pytest.raises(ValueError, match="重定向"):
            fetch_model_options(ModelSpec(api_base="https://api.deepseek.com"))
        assert not destination.called


@pytest.mark.parametrize("payload", [{}, {"data": []}, {"data": {}}, {"data": [{"id": ""}]}, {"data": ["bad"]}])
def test_empty_or_incompatible_list_is_reported(payload):
    with respx.mock as router:
        router.get("https://api.deepseek.com/models").mock(return_value=httpx.Response(200, json=payload))
        with pytest.raises(ValueError, match="列表"):
            fetch_model_options(ModelSpec(api_base="https://api.deepseek.com"))


def test_non_json_response_is_reported_without_echoing_it():
    with respx.mock as router:
        router.get("https://api.deepseek.com/models").mock(return_value=httpx.Response(200, text="private-test-key"))
        with pytest.raises(ValueError, match="JSON") as caught:
            fetch_model_options(ModelSpec(api_base="https://api.deepseek.com", api_key="private-test-key"))
        assert "private-test-key" not in str(caught.value)


@pytest.mark.parametrize("error", [httpx.ConnectError, httpx.ReadTimeout])
def test_transport_error_is_sanitized(error):
    with respx.mock as router:
        route = router.get("https://api.deepseek.com/models").mock(side_effect=error("private-test-key"))
        with pytest.raises(ValueError) as caught:
            fetch_model_options(ModelSpec(api_base="https://api.deepseek.com", api_key="private-test-key"))
        assert "private-test-key" not in str(caught.value)
        assert route.call_count == 1


def test_unsupported_adapter_and_missing_key_do_not_make_requests():
    with respx.mock as router:
        for spec in [ModelSpec(provider="Anthropic"), ModelSpec(api_key="")]:
            with pytest.raises(ValueError):
                fetch_model_options(spec)
        assert not router.calls
