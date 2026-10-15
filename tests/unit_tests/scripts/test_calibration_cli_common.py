# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the shared hand-eye calibration CLI plumbing."""

from __future__ import annotations

import logging

import pytest

pytestmark = pytest.mark.usefixtures("isolated_framework_logging")


def test_unified_main_clears_proxy_before_workflow(monkeypatch, tmp_path):
    import scripts.calibrate.hand_eye_calib as cli

    calls: list[str] = []
    monkeypatch.setattr(cli, "clear_proxy_env", lambda: calls.append("proxy"))
    monkeypatch.setattr(
        cli,
        "import_waypoints",
        lambda _source, _output, *, expected_mount: calls.append(f"import:{expected_mount}"),
    )

    result = cli.main(["--collect-poses", str(tmp_path / "out.npz"), "--import-poses", str(tmp_path / "in.npz")])

    assert result == 0
    assert calls == ["proxy", "import:None"]


def test_configure_logging_uses_framework_handler_once():
    import scripts.calibrate._cli_common as cli_common
    from jiuwensymbiosis.utils.logging import _OWNED_TAG

    root = logging.getLogger()
    root_handlers, root_level = root.handlers[:], root.level
    cli_common.configure_logging(debug=True)
    logger = logging.getLogger("jiuwensymbiosis")
    owned = [h for h in logger.handlers if getattr(h, _OWNED_TAG, False)]
    cli_common.configure_logging(debug=True)
    assert [h for h in logger.handlers if getattr(h, _OWNED_TAG, False)] == owned
    assert len(owned) == 1
    assert not logger.propagate
    assert root.handlers == root_handlers
    assert root.level == root_level
    assert logging.getLogger("jiuwensymbiosis.calibration.cli").getEffectiveLevel() == logging.DEBUG
