# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""A misconfigured ``can_port`` must fail as configuration, never as lost hardware.

``HardwareCleanupError`` makes the runtime persist a blocked resource record that
only an operator can reconcile. That is correct when a real bus was touched and
its state is unknown; it must not be the outcome of a typo in the runtime YAML.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from jiuwensymbiosis.adapters.piper import lowlevel
from jiuwensymbiosis.agent.lifecycle import HardwareCleanupError

_CAN = "280"
_ETHERNET = "1"


def _link(root: Path, name: str, *, link_type: str, operstate: str = "up") -> None:
    """Create one fake ``/sys/class/net`` entry."""
    path = root / name
    path.mkdir()
    (path / "type").write_text(f"{link_type}\n", encoding="utf-8")
    (path / "operstate").write_text(f"{operstate}\n", encoding="utf-8")


@pytest.fixture
def net(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "net"
    root.mkdir()
    monkeypatch.setattr(lowlevel, "_NET_SYSFS", str(root))
    return root


def test_missing_interface_is_a_plain_configuration_error(net: Path) -> None:
    _link(net, "can0", link_type=_CAN)
    _link(net, "wlan0", link_type=_ETHERNET)

    with pytest.raises(RuntimeError) as failed:
        lowlevel._require_can_interface("can_left")

    # A blocked record would demand operator reconciliation for a YAML typo.
    assert not isinstance(failed.value, HardwareCleanupError)
    message = str(failed.value)
    assert "can_left" in message
    assert "can0" in message  # names the usable interface, excludes non-CAN links
    assert "wlan0" not in message
    # A usable interface exists, so do not also suggest bringing one up.
    assert "ip link set" not in message


def test_missing_interface_without_any_can_link_suggests_bringing_one_up(net: Path) -> None:
    _link(net, "wlan0", link_type=_ETHERNET)

    with pytest.raises(RuntimeError, match="no CAN interface is present") as failed:
        lowlevel._require_can_interface("can0")

    assert "ip link set can0 up type can" in str(failed.value)


def test_non_can_interface_is_rejected(net: Path) -> None:
    _link(net, "wlan0", link_type=_ETHERNET)

    with pytest.raises(RuntimeError, match="not a CAN interface") as failed:
        lowlevel._require_can_interface("wlan0")

    assert not isinstance(failed.value, HardwareCleanupError)


def test_down_interface_is_rejected_with_the_bring_up_command(net: Path) -> None:
    _link(net, "can0", link_type=_CAN, operstate="down")

    with pytest.raises(RuntimeError) as failed:
        lowlevel._require_can_interface("can0")

    assert not isinstance(failed.value, HardwareCleanupError)
    assert "ip link set can0 up type can" in str(failed.value)


def test_up_can_interface_is_admitted(net: Path) -> None:
    _link(net, "can_left", link_type=_CAN)

    assert lowlevel._require_can_interface("can_left") is None


def test_unknown_operstate_leaves_the_verdict_to_the_sdk(net: Path) -> None:
    """Linux UNKNOWN also represents drivers that do not report operstate."""
    _link(net, "can0", link_type=_CAN, operstate="unknown")

    assert lowlevel._require_can_interface("can0") is None


def test_unreadable_interface_type_leaves_the_verdict_to_the_sdk(net: Path) -> None:
    _link(net, "can0", link_type=_CAN)
    (net / "can0" / "type").unlink()

    assert lowlevel._require_can_interface("can0") is None


def test_unobservable_sysfs_leaves_the_verdict_to_the_sdk(tmp_path: Path, monkeypatch) -> None:
    """Inability to check is not evidence of failure."""
    monkeypatch.setattr(lowlevel, "_NET_SYSFS", str(tmp_path / "absent"))

    assert lowlevel._require_can_interface("can0") is None


def test_unreadable_operstate_leaves_the_verdict_to_the_sdk(net: Path) -> None:
    path = net / "can0"
    path.mkdir()
    (path / "type").write_text(f"{_CAN}\n", encoding="utf-8")

    assert lowlevel._require_can_interface("can0") is None


def test_constructor_rejects_bad_can_port_without_touching_the_sdk(net: Path, monkeypatch) -> None:
    """The pre-check runs before the SDK can leave the bus unconfirmable."""
    _link(net, "can0", link_type=_CAN)

    def _must_not_construct(_port):
        raise AssertionError("SDK must not be constructed for a bad can_port")

    monkeypatch.setenv("JIUWEN_PIPER_CMD_LOG", "0")
    monkeypatch.setitem(sys.modules, "piper_sdk", SimpleNamespace(C_PiperInterface_V2=_must_not_construct))

    with pytest.raises(RuntimeError, match="does not exist") as failed:
        lowlevel.PiperLowLevel(can_port="can_left")

    assert not isinstance(failed.value, HardwareCleanupError)
