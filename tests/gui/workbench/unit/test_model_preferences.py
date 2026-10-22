"""Credential storage stays endpoint-bound, safe on failure, and independent of YAML."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

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


@pytest.mark.parametrize("chained", [False, True])
@pytest.mark.parametrize("backend_class", ["DBusKeyring", "DBusKeyringKWallet4"])
def test_kwallet_open_path_requires_explicit_interaction(monkeypatch, chained, backend_class):
    import keyring
    from keyring.backends import kwallet
    from keyring.backends.chainer import ChainerBackend

    backend_type = getattr(kwallet, backend_class)
    with patch.object(backend_type, "priority", 5):
        backend = backend_type()
        # Real get_password() enters connected(), which opens the wallet. No D-Bus
        # connection is made by this test; False represents cancelling its prompt.
        connected = Mock(return_value=False)
        monkeypatch.setattr(backend, "connected", connected)
        monkeypatch.setattr(ChainerBackend, "backends", [backend])
        monkeypatch.setattr(keyring, "get_keyring", lambda: ChainerBackend() if chained else backend)
        store = preferences.ModelPreferences()
        with pytest.raises(preferences.ModelPreferenceError, match="KWallet"):
            store.load(ModelSpec())
        connected.assert_not_called()
        with pytest.raises(preferences.ModelPreferenceError, match="无法读取"):
            store.load(ModelSpec(), allow_interaction=True)
        connected.assert_called_once()


def test_named_providers_can_be_saved_before_selecting_a_model_and_edited_without_changing_it(backend, model):
    from jiuwensymbiosis_gui.workbench.model_catalog import CustomProvider

    store = preferences.ModelPreferences(backend=backend)
    first = CustomProvider("First", "https://one.example/v1", "one-key", "https://one.example")
    second = CustomProvider("Second", "https://two.example/v1", "two-key")
    store.save_provider(first)
    store.save_provider(second)
    assert store.load(model) is None
    store.save(model)
    renamed = replace(first, name="Renamed", api_key="replacement-key")
    store.save_provider(renamed, original_name=first.name)
    reopened = preferences.ModelPreferences(backend=backend)
    saved, providers = reopened.load_profiles(model)
    assert saved == model
    assert providers == {"Second": second, "Renamed": renamed}
    reopened.delete_provider("Renamed")
    assert reopened.load_profiles(model) == (model, {"Second": second})
    reopened.clear()
    assert reopened.load_profiles(model) == (None, {})


def test_duplicate_or_stale_provider_edits_preserve_saved_profiles(backend, model):
    from jiuwensymbiosis_gui.workbench.model_catalog import CustomProvider

    store = preferences.ModelPreferences(backend=backend)
    provider = CustomProvider("First", "https://one.example/v1", "one-key")
    store.save_provider(provider)
    before = dict(backend.entries)
    with pytest.raises(preferences.ModelPreferenceError, match="已存在"):
        store.save_provider(replace(provider, api_key="different-key"))
    with pytest.raises(preferences.ModelPreferenceError, match="已被删除"):
        store.save_provider(replace(provider, name="renamed"), original_name="missing")
    assert backend.entries == before


def test_old_model_record_loads_and_is_preserved_when_adding_provider(backend, model):
    from jiuwensymbiosis_gui.workbench.model_catalog import CustomProvider

    fields = {name: getattr(model, name) for name in preferences._FIELDS}
    backend.set_password("jiuwensymbiosis.gui", "workbench-model", json.dumps({"version": 1, "model": fields}))
    store = preferences.ModelPreferences(backend=backend)
    assert store.load_profiles(model) == (model, {})
    provider = CustomProvider("First", "https://one.example/v1", "one-key")
    store.save_provider(provider)
    assert store.load_profiles(model) == (model, {"First": provider})


@pytest.mark.parametrize("operation", ["save_provider", "delete_provider"])
def test_provider_write_failure_is_safe_and_leaves_saved_record_unchanged(backend, model, operation):
    from jiuwensymbiosis_gui.workbench.model_catalog import CustomProvider

    store = preferences.ModelPreferences(backend=backend)
    provider = CustomProvider("First", "https://one.example/v1", "one-key")
    store.save_provider(provider)
    before = dict(backend.entries)
    backend.set_password = Mock(side_effect=RuntimeError("one-key"))
    with pytest.raises(preferences.ModelPreferenceError) as error:
        if operation == "save_provider":
            store.save_provider(replace(provider, name="Second"))
        else:
            store.delete_provider("First")
    assert "one-key" not in str(error.value)
    assert backend.entries == before


@pytest.mark.parametrize("providers", [{}, [None], [{"name": "broken", "api_key": "secret"}]])
def test_corrupt_provider_records_do_not_expose_keys(backend, model, providers):
    backend.set_password("jiuwensymbiosis.gui", "workbench-model", json.dumps({"version": 1, "providers": providers}))
    with pytest.raises(preferences.ModelPreferenceError) as error:
        preferences.ModelPreferences(backend=backend).load_profiles(model)
    assert "secret" not in str(error.value)
