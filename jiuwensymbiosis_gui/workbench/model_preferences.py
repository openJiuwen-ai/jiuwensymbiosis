"""Remember the shared WebUI model profile in the operating system's credential store."""

from __future__ import annotations

import json
import threading
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Protocol

from jiuwensymbiosis_gui.workbench.model_catalog import model_list_url

if TYPE_CHECKING:
    from jiuwensymbiosis.agent.config import ModelSpec

__all__ = ["ModelPreferenceError", "ModelPreferences"]

_SERVICE = "jiuwensymbiosis.gui"
_ACCOUNT = "workbench-model"
_LOCK = threading.RLock()
_SECURE_BACKENDS = {
    "keyring.backends.SecretService",
    "keyring.backends.macOS",
    "keyring.backends.Windows",
    "keyring.backends.kwallet",
}
_FIELDS = {"provider", "api_base", "model_name", "api_key", "verify_ssl"}


class _CredentialBackend(Protocol):
    def get_password(self, service: str, account: str) -> str | None:
        """Return the stored credential, or None when no record exists."""
        ...

    def set_password(self, service: str, account: str, password: str) -> None:
        """Store the credential for the service and account."""
        ...

    def delete_password(self, service: str, account: str) -> None:
        """Delete the credential for the service and account."""
        ...


class ModelPreferenceError(RuntimeError):
    """A safe error message that contains neither credentials nor backend output."""


def _system_backend() -> _CredentialBackend:
    try:
        import keyring
        from keyring.backends.chainer import ChainerBackend

        backend = keyring.get_keyring()
        candidates = backend.backends if isinstance(backend, ChainerBackend) else [backend]
        for candidate in candidates:
            if type(candidate).__module__ in _SECURE_BACKENDS and candidate.priority > 0:
                if type(candidate).__module__ == "keyring.backends.SecretService":
                    import secretstorage

                    bus = secretstorage.dbus_init()
                    collection = (
                        secretstorage.Collection(bus, candidate.preferred_collection)
                        if hasattr(candidate, "preferred_collection")
                        else secretstorage.get_default_collection(bus)
                    )
                    # Avoid an unexpected desktop unlock prompt during page restoration.
                    if collection.is_locked():
                        raise ModelPreferenceError("系统钥匙串已锁定；请先解锁，再保存或刷新页面恢复设置。")
                return candidate
    except ModelPreferenceError:
        raise
    except Exception:
        raise ModelPreferenceError("系统钥匙串不可用；模型设置未保存，请检查钥匙串是否已解锁。") from None
    raise ModelPreferenceError("没有可用的系统钥匙串；模型设置未保存，不会回退到明文文件。")


def _validate_fields(fields: object) -> dict[str, Any]:
    if not isinstance(fields, dict) or set(fields) != _FIELDS:
        raise ModelPreferenceError("已保存的模型设置格式不兼容，请重新保存或忘记旧设置。")
    if any(not isinstance(fields[name], str) for name in _FIELDS - {"verify_ssl"}):
        raise ModelPreferenceError("模型设置格式无效，请重新填写。")
    if not isinstance(fields["verify_ssl"], bool) or not fields["provider"].strip() or not fields["model_name"].strip():
        raise ModelPreferenceError("请填写接口适配器和模型名称。")
    try:
        model_list_url(fields["api_base"])
    except ValueError:
        raise ModelPreferenceError("请填写有效的模型 API 基础地址后再保存。") from None
    return fields


class ModelPreferences:
    """Save endpoint-bound credentials as one record; never use a plaintext fallback.

    Only connection fields are remembered. Token limits, temperature and extra
    request parameters continue to come from the currently selected configuration.
    An injected backend/account allows isolated tests without touching user secrets.
    """

    def __init__(self, *, backend: _CredentialBackend | None = None, account: str = _ACCOUNT) -> None:
        self._backend_override = backend
        self._account = account

    def _backend(self) -> _CredentialBackend:
        return self._backend_override if self._backend_override is not None else _system_backend()

    @staticmethod
    def validate(model: ModelSpec) -> None:
        """Validate connection fields without accessing the credential store."""
        _validate_fields({name: getattr(model, name) for name in _FIELDS})

    def load(self, default: ModelSpec) -> ModelSpec | None:
        """Restore the last connection profile, retaining current non-connection options."""
        with _LOCK:
            backend = self._backend()
            try:
                encoded = backend.get_password(_SERVICE, self._account)
            except Exception:
                raise ModelPreferenceError("无法读取系统钥匙串；请检查钥匙串是否已解锁。") from None
            if encoded is None:
                return None
            try:
                if len(encoded) > 32768:
                    raise ValueError("oversized profile")
                record = json.loads(encoded)
                if not isinstance(record, dict) or record.get("version") != 1:
                    raise ValueError("unknown schema")
                fields = _validate_fields(record.get("model"))
            except (TypeError, ValueError):
                raise ModelPreferenceError("已保存的模型设置损坏或版本不兼容，请重新保存或忘记旧设置。") from None
            return replace(default, **fields)

    def save(self, model: ModelSpec) -> None:
        """Remember the connection profile and API key as a single credential entry."""
        fields = _validate_fields({name: getattr(model, name) for name in _FIELDS})
        encoded = json.dumps({"version": 1, "model": fields}, ensure_ascii=False)
        if len(encoded) > 32768:
            raise ModelPreferenceError("模型设置过长，未保存到系统钥匙串。")
        with _LOCK:
            backend = self._backend()
            try:
                backend.set_password(_SERVICE, self._account, encoded)
            except Exception:
                raise ModelPreferenceError("保存失败；请检查系统钥匙串是否已解锁。") from None

    def clear(self) -> None:
        """Forget this application's profile; leave other credential entries untouched."""
        with _LOCK:
            backend = self._backend()
            try:
                if backend.get_password(_SERVICE, self._account) is not None:
                    backend.delete_password(_SERVICE, self._account)
            except Exception:
                raise ModelPreferenceError("忘记设置失败；请检查系统钥匙串是否已解锁。") from None
