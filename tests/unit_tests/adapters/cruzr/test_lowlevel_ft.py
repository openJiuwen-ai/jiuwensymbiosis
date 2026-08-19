# coding: utf-8
import sys
import types
from types import SimpleNamespace

from jiuwensymbiosis.adapters.cruzr.config import CruzrConfig
from jiuwensymbiosis.adapters.cruzr.lowlevel import CruzrLowLevel


def _fake_wrench():
    force = SimpleNamespace(x=1.0, y=2.0, z=2.0)   # |f| = sqrt(1+4+4) = 3
    torque = SimpleNamespace(x=0.0, y=0.0, z=0.0)
    return SimpleNamespace(wrench=SimpleNamespace(force=force, torque=torque))


class _FakeNode:
    """Records the subscribed topic and delivers one wrench frame synchronously."""

    def __init__(self, wrench):
        self.topic = None
        self._wrench = wrench

    def create_subscription(self, msg_type, topic, cb, qos):
        self.topic = topic
        cb(self._wrench)
        return object()

    def destroy_subscription(self, sub):
        pass


def _stub_geometry_msgs(monkeypatch):
    # read_hand_ft does `from geometry_msgs.msg import WrenchStamped`; ROS is off the
    # unit-test path, so inject a stub module.
    geo = types.ModuleType("geometry_msgs")
    geo_msg = types.ModuleType("geometry_msgs.msg")
    geo_msg.WrenchStamped = object
    geo.msg = geo_msg
    monkeypatch.setitem(sys.modules, "geometry_msgs", geo)
    monkeypatch.setitem(sys.modules, "geometry_msgs.msg", geo_msg)


def test_read_hand_ft_selects_topic(monkeypatch):
    _stub_geometry_msgs(monkeypatch)
    ll = object.__new__(CruzrLowLevel)
    ll.cfg = CruzrConfig(left_hand_ft_topic="/mc/ft_states/L_hand_ft",
                         right_hand_ft_topic="/mc/ft_states/R_hand_ft")
    ll._node = _FakeNode(_fake_wrench())

    out = ll.read_hand_ft("left")
    assert out["ok"] is True
    assert ll._node.topic == "/mc/ft_states/L_hand_ft"
    assert abs(out["fmag"] - 3.0) < 1e-9

    ll.read_hand_ft("right")
    assert ll._node.topic == "/mc/ft_states/R_hand_ft"
