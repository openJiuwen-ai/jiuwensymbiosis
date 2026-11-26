# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""``--json`` introspection commands keep stdout machine-readable (#18).

Importing the agent stack makes openjiuwen's console logging write to stdout.
The console scripts must not let those lines precede the JSON payload, and
``import jiuwensymbiosis`` itself must stay free of that import side effect.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_script(script: str) -> subprocess.CompletedProcess[str]:
    """Run a fresh interpreter so import-time logging is exercised honestly."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _run_entry(name: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", f"from jiuwensymbiosis.cli import {name}; raise SystemExit({name}())", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_package_import_does_not_load_the_agent_or_write_stdout():
    """The root cause of #18: importing the package must not start openjiuwen logging."""
    result = _run_script(
        """
        import sys
        import jiuwensymbiosis
        loaded = sorted(name for name in sys.modules if name.startswith("openjiuwen"))
        assert not loaded, loaded
        """
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout == ""


def test_actions_vocabulary_json_is_parseable():
    result = _run_entry("actions_main", "--vocabulary", "--json")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert isinstance(payload, list) and payload
    assert "| INFO |" not in result.stdout


def test_skills_json_keeps_import_logs_off_stdout():
    result = _run_entry("skills_main", "--json")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert isinstance(payload, list) and payload
    assert "| INFO |" not in result.stdout


def test_state_json_is_parseable_with_a_faked_admission_boundary():
    """The live command's stdout contract, without connecting hardware."""
    result = _run_script(
        """
        import sys
        from jiuwensymbiosis import cli, introspect

        class Session:
            def connect(self): pass
            def disconnect(self): pass
            def cleanup_report(self):
                from jiuwensymbiosis.agent.lifecycle import CleanupReport
                return CleanupReport()

        class Binding:
            resources = ()
            def build_session(self): return Session()

        class Lease:
            resources = ()

        class Manager:
            def acquire(self, resources, operation="task"): return Lease()
            def finish(self, lease, report): return True

        introspect.prepare_binding = lambda config: Binding()
        sys.argv = ["jiuwensymbiosis-state", "--config", "fake.yaml", "--json"]
        raise SystemExit(cli.state_main(resource_manager=Manager()))
        """
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert set(payload) >= {"tokens", "pose", "joints", "capabilities", "extra"}
    assert "| INFO |" not in result.stdout


def test_machine_output_redirects_child_stdout_and_restores_after_an_exception():
    result = _run_script(
        """
        import subprocess
        import sys
        from jiuwensymbiosis.cli import _machine_output

        with _machine_output(True) as out:
            subprocess.run([sys.executable, "-c", "print('child-noise')"], check=True)
            out.write('{"ok": true}\\n')
        print("after")
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == '{"ok": true}\nafter\n'
    assert "child-noise" in result.stderr

    failed = _run_script(
        """
        from jiuwensymbiosis.cli import _machine_output

        with _machine_output(True) as out:
            out.write('{"ok": 1}\\n')
            raise RuntimeError("boom")
        """
    )
    assert failed.returncode != 0
    assert failed.stdout == '{"ok": 1}\n'
    assert "RuntimeError: boom" in failed.stderr


def test_public_names_and_submodules_are_lazily_resolvable():
    result = _run_script(
        """
        import jiuwensymbiosis

        assert len(jiuwensymbiosis.__all__) == 22
        for name in jiuwensymbiosis.__all__:
            assert getattr(jiuwensymbiosis, name) is not None
        assert jiuwensymbiosis.clear_proxy_env.__name__ == "clear_proxy_env"
        assert jiuwensymbiosis.api.actions.ACTIONS
        assert jiuwensymbiosis.agent is not None
        try:
            jiuwensymbiosis.does_not_exist
        except AttributeError:
            pass
        else:
            raise AssertionError("unknown attributes must raise AttributeError")
        """
    )
    assert result.returncode == 0, result.stdout + result.stderr
