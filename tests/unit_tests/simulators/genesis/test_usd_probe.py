"""Standalone source entry points keep imports isolated and stdout machine-readable."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from jiuwensymbiosis.simulators.genesis import usd_probe


def test_standalone_catalog_ignores_unrelated_assets_package(tmp_path):
    directory = Path(usd_probe.__file__).parent
    script = """
import sys
from types import ModuleType
sys.path.insert(0, sys.argv[1])
foreign = ModuleType('assets')
sys.modules['assets'] = foreign
before = list(sys.path)
import scene_catalog
assert scene_catalog.AssetLibrary.__module__ != 'assets'
assert sys.modules['assets'] is foreign
assert sys.path == before
assert not {'jiuwensymbiosis', 'openjiuwen', 'torch', 'genesis'} & sys.modules.keys()
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(directory)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("succeeds", [False, True])
def test_probe_writes_json_result_and_failure_exit(tmp_path, monkeypatch, capsys, succeeds):
    def render(*args, **kwargs):
        if not succeeds:
            raise ValueError("test render failure")
        return {"ok": True, "scene_id": "test"}

    monkeypatch.setattr(usd_probe, "render", render)
    monkeypatch.setattr(sys, "argv", ["usd_probe", "--scene", "test", "--output", str(tmp_path)])
    if succeeds:
        usd_probe.main()
    else:
        with pytest.raises(SystemExit) as error:
            usd_probe.main()
        assert error.value.code == 1
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is succeeds
    assert result == json.loads((tmp_path / "report.json").read_text())
    if not succeeds:
        assert result["error"] == "test render failure"
