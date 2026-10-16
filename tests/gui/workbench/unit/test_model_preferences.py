"""Credential storage stays endpoint-bound, safe on failure, and independent of YAML."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from jiuwensymbiosis.agent.config import ModelSpec
from jiuwensymbiosis_gui.workbench import model_preferences as preferences


class MemoryKeyring:
    def __init__(self):
        self.entries = {}

    def get_password(self, service, account):
        return self.entries.get((service, account))

    def set_password(self, service, account, password):
        self.entries[service, account] = password

    def delete_password(self, service, account):
        del self.entries[service, account]


@pytest.fixture
def backend():
    return MemoryKeyring()


@pytest.fixture
def model():
    return ModelSpec(
        provider="OpenAI",
        api_base="https://api.deepseek.com",
        model_name="deepseek-flash",
        api_key="test-secret-never-log",
        verify_ssl=True,
    )


def test_new_store_restores_connection_and_key_together_without_replacing_yaml_options(backend, model):
    store = preferences.ModelPreferences(backend=backend)
    store.save(model)
    default = ModelSpec(max_tokens=8192, temperature=0.8, extra_request_kwargs={"option": "from-yaml"})
    restored = preferences.ModelPreferences(backend=backend).load(default)
    assert restored == replace(
        default,
        provider=model.provider,
        api_base=model.api_base,
        model_name=model.model_name,
        api_key=model.api_key,
        verify_ssl=True,
    )
    assert default.api_key == "EMPTY"
    saved = json.loads(next(iter(backend.entries.values())))
    assert "extra_request_kwargs" not in saved["model"]


def test_forget_is_idempotent_and_only_deletes_this_profile(backend, model):
    store = preferences.ModelPreferences(backend=backend)
    other = preferences.ModelPreferences(backend=backend, account="other-application-profile")
    store.save(model)
    other.save(model)
    store.clear()
    store.clear()
    assert store.load(ModelSpec()) is None
    assert other.load(ModelSpec()).api_key == model.api_key


@pytest.mark.parametrize("operation", ["get_password", "set_password", "delete_password"])
def test_backend_errors_never_reflect_keys_or_payloads(backend, model, caplog, operation):
    store = preferences.ModelPreferences(backend=backend)
    store.save(model)
    setattr(backend, operation, Mock(side_effect=RuntimeError(model.api_key)))
    action = {
        "get_password": lambda: store.load(model),
        "set_password": lambda: store.save(model),
        "delete_password": store.clear,
    }[operation]
    with pytest.raises(preferences.ModelPreferenceError) as error:
        action()
    assert model.api_key not in str(error.value)
    assert model.api_key not in caplog.text


@pytest.mark.parametrize("encoded", ["not-json-test-secret", '{"version":2}', '{"version":1,"model":{}}'])
def test_corrupt_records_do_not_partially_restore_credentials(backend, model, encoded):
    backend.set_password("jiuwensymbiosis.gui", "workbench-model", encoded)
    with pytest.raises(preferences.ModelPreferenceError):
        preferences.ModelPreferences(backend=backend).load(model)
    assert model.api_key == "test-secret-never-log"


@pytest.mark.parametrize(
    "address", ["not-an-api", "https://user:password@example.com/v1", "https://example.com?key=secret"]
)
def test_invalid_or_credential_bearing_urls_are_not_saved(backend, model, address):
    with pytest.raises(preferences.ModelPreferenceError):
        preferences.ModelPreferences(backend=backend).save(replace(model, api_base=address))
    assert not backend.entries


def test_plaintext_and_null_backends_are_rejected(monkeypatch):
    import keyring

    for module in ("keyrings.alt.file", "keyring.backends.null", "keyring.backends.fail"):
        backend_type = type("Keyring", (), {"__module__": module, "priority": 5})
        monkeypatch.setattr(keyring, "get_keyring", lambda cls=backend_type: cls())
        with pytest.raises(preferences.ModelPreferenceError, match="明文"):
            preferences.ModelPreferences().save(ModelSpec())


def test_chainer_selects_only_a_supported_system_backend(monkeypatch):
    import keyring

    secretstorage = pytest.importorskip("secretstorage", reason="Linux SecretService backend")
    from keyring.backends.chainer import ChainerBackend

    secure = type("Keyring", (), {"__module__": "keyring.backends.SecretService", "priority": 5})()
    insecure = SimpleNamespace(priority=10)
    monkeypatch.setattr(ChainerBackend, "backends", [insecure, secure])
    chained = ChainerBackend()
    monkeypatch.setattr(keyring, "get_keyring", lambda: chained)
    monkeypatch.setattr(secretstorage, "dbus_init", lambda: None)
    monkeypatch.setattr(secretstorage, "get_default_collection", lambda bus: SimpleNamespace(is_locked=lambda: False))
    assert preferences._system_backend() is secure


def test_locked_secret_service_does_not_attempt_an_unlock_prompt(monkeypatch):
    import keyring

    secretstorage = pytest.importorskip("secretstorage", reason="Linux SecretService backend")

    backend = type("Keyring", (), {"__module__": "keyring.backends.SecretService", "priority": 5})()
    unlock = Mock(side_effect=AssertionError("must not open an unlock dialog"))
    monkeypatch.setattr(keyring, "get_keyring", lambda: backend)
    monkeypatch.setattr(secretstorage, "dbus_init", lambda: None)
    monkeypatch.setattr(
        secretstorage, "get_default_collection", lambda bus: SimpleNamespace(is_locked=lambda: True, unlock=unlock)
    )
    with pytest.raises(preferences.ModelPreferenceError, match="已锁定"):
        preferences.ModelPreferences().load(ModelSpec())
    unlock.assert_not_called()
