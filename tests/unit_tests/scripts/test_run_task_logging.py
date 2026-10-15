# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""CLI configuration precedence and final voice output, without hardware/LLMs."""

from __future__ import annotations

import logging
from unittest.mock import Mock

import pytest

from examples import run_task
from tests.helpers import make_mock_session

pytestmark = pytest.mark.usefixtures("isolated_framework_logging")


def test_cli_merges_overrides_before_validation_and_connection(monkeypatch, tmp_path):
    raw = {
        "agent": {
            "execution": {"stepagent": {"max_iterations": 0}},
            "logging": {"dir": str(tmp_path), "level": "typo"},
        }
    }
    monkeypatch.setattr(run_task, "_load_yaml", lambda _: raw)

    def build(*_args):
        assert logging.getLogger("jiuwensymbiosis").level == logging.DEBUG
        assert (tmp_path / "jiuwensymbiosis.log").exists()
        return make_mock_session()

    def execute(_session, _query, config, **_kwargs):
        assert config.execution.stepagent.max_iterations == 5
        assert config.logging.level == "DEBUG"
        return {"output": "done"}

    monkeypatch.setattr(run_task, "_build_session", build)
    monkeypatch.setattr(run_task, "run_robot_task", execute)
    assert run_task.main(["--config", "unused.yaml", "--mock", "--query", "task", "--max-iter", "5", "--debug"]) == 0
    assert raw["agent"]["execution"]["stepagent"]["max_iterations"] == 0


def test_invalid_final_config_fails_before_session_construction(monkeypatch):
    monkeypatch.setattr(
        run_task, "_load_yaml", lambda _: {"agent": {"execution": {"stepagent": {"max_iterations": 0}}}}
    )
    build = Mock(side_effect=AssertionError("must validate before building session"))
    monkeypatch.setattr(run_task, "_build_session", build)
    with pytest.raises(ValueError, match="max_iterations"):
        run_task.main(["--config", "unused.yaml", "--mock", "--query", "task"])
    build.assert_not_called()


def test_missing_query_reports_error_without_creating_log_files(monkeypatch, tmp_path, capsys):
    log_dir = tmp_path / "logs"
    monkeypatch.setattr(
        run_task, "_load_yaml", lambda _: {"agent": {"logging": {"dir": str(log_dir), "level": "CRITICAL"}}}
    )
    build = Mock(side_effect=AssertionError("must reject missing query before building session"))
    monkeypatch.setattr(run_task, "_build_session", build)
    assert run_task.main(["--config", "unused.yaml", "--mock"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "未提供任务" in captured.err
    assert not log_dir.exists()
    build.assert_not_called()


def test_adapter_import_failure_is_still_logged(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(run_task, "_load_yaml", lambda _: {"agent": {"logging": {"dir": str(tmp_path)}}})
    monkeypatch.setattr(run_task, "_build_session", Mock(side_effect=ImportError("adapter-unavailable")))
    assert run_task.main(["--config", "unused.yaml", "--mock", "--query", "task"]) == 2
    assert "adapter-unavailable" in capsys.readouterr().err
    assert "adapter-unavailable" in (tmp_path / "jiuwensymbiosis.log").read_text()


@pytest.mark.parametrize("level", ["INFO", "WARNING"])
@pytest.mark.parametrize("answer", [None, "output", "content", "empty_output"])
def test_fast_success_has_one_final_answer_without_payloads(monkeypatch, tmp_path, capsys, level, answer):
    from contextlib import nullcontext

    monkeypatch.setattr(
        run_task, "_load_yaml", lambda _: {"agent": {"logging": {"dir": str(tmp_path), "level": level}}}
    )
    monkeypatch.setattr("jiuwensymbiosis.runtime.bindings.prepare_binding", lambda *_a, **_kw: object())
    monkeypatch.setattr(
        "jiuwensymbiosis.runtime.admission.admitted_session", lambda *_a, **_kw: nullcontext(make_mock_session())
    )
    result = {"ok": True, "steps_done": 3, "steps": [{"result": "PRIVATE_RESULT"}], "context": "PRIVATE_CONTEXT"}
    if answer == "empty_output":
        result.update(output="", content="FINAL_REPLY")
    elif answer is not None:
        result[answer] = "FINAL_REPLY"

    def execute(_session, _query, config, **_kwargs):
        assert config.execution.mode == "fastagent"
        return result

    monkeypatch.setattr(run_task, "run_robot_task", execute)
    assert run_task.main(["--config", "unused.yaml", "--query", "PRIVATE_TASK"]) == 0
    captured = capsys.readouterr()
    expected = "FINAL_REPLY" if answer else "任务完成，共执行 3 步"
    assert captured.out.splitlines() == [expected]
    for output in (captured.out, captured.err, (tmp_path / "jiuwensymbiosis.log").read_text()):
        assert "PRIVATE_" not in output


@pytest.mark.parametrize("level", ["INFO", "WARNING", "DEBUG"])
@pytest.mark.parametrize("failure_location", ["task", "loop", "action", "exception"])
def test_cli_failure_summary_keeps_full_context_in_debug(
    monkeypatch, tmp_path, capsys, caplog, level, failure_location
):
    from jiuwensymbiosis.agent.fast.runner import run_sequence
    from jiuwensymbiosis.agent.fast.sequence import ActionStep, LoopStep

    monkeypatch.setattr(
        run_task, "_load_yaml", lambda _: {"agent": {"logging": {"dir": str(tmp_path), "level": level}}}
    )
    monkeypatch.setattr(run_task, "_build_session", lambda *_args: make_mock_session())

    def execute(session, *_args, **_kwargs):
        if failure_location == "task":
            result = {"ok": False, "reason": "no_model_spec"}
        else:

            def executor(op, params):
                if failure_location != "loop":
                    assert op == "get_grasp_info_simple"
                    assert params == {"object_name": "PRIVATE_PARAMS"}
                if failure_location == "exception":
                    raise RuntimeError("camera_timeout")
                return {"ok": False, "reason": "camera_timeout"}

            step = (
                LoopStep(detect_op="get_grasp_info_simple")
                if failure_location == "loop"
                else ActionStep(op="get_grasp_info_simple", params={"object_name": "PRIVATE_PARAMS"})
            )
            result = run_sequence(
                session,
                [step],
                executor=executor,
                recovery_enabled=False,
            )
        result["sequence"] = [{"op": "get_grasp_info_simple", "params": {"object_name": "PRIVATE_PARAMS"}}]
        result["context"] = "PRIVATE_RESULT"
        return result

    monkeypatch.setattr(run_task, "run_robot_task", execute)
    assert run_task.main(["--config", "unused.yaml", "--mock", "--query", "PRIVATE_TASK"]) == 0
    captured = capsys.readouterr()
    summary = {
        "task": "Agent failed: no_model_spec",
        "loop": "Agent failed at step 0 (loop:get_grasp_info_simple): camera_timeout",
        "action": "Agent failed at step 0 (get_grasp_info_simple): _StepExecutionError: camera_timeout",
        "exception": "Agent failed at step 0 (get_grasp_info_simple): RuntimeError: camera_timeout",
    }[failure_location]
    assert captured.out == ""
    for output in (captured.err, (tmp_path / "jiuwensymbiosis.log").read_text()):
        assert output.count(summary) == 1
        for marker in ("PRIVATE_TASK", "PRIVATE_PARAMS", "PRIVATE_RESULT"):
            assert (marker in output) == (level == "DEBUG")
    assert any(record.levelno == logging.ERROR and record.getMessage() == summary for record in caplog.records)
    if failure_location in {"action", "exception"}:
        runner_records = [record for record in caplog.records if record.name == "jiuwensymbiosis.agent.fast.runner"]
        assert any(
            record.levelno == logging.WARNING
            and record.getMessage() == "[runner] step 0 failed: get_grasp_info_simple: camera_timeout"
            for record in runner_records
        )
        parameter_records = [record for record in runner_records if "PRIVATE_PARAMS" in record.getMessage()]
        assert len(parameter_records) == int(level == "DEBUG")
        assert all(record.levelno == logging.DEBUG for record in parameter_records)


def test_voice_failure_reports_step_summary_before_reducing_reply(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        run_task,
        "_load_yaml",
        lambda _: {
            "agent": {"logging": {"dir": str(tmp_path), "level": "WARNING"}},
            "voice": {"asr": {"backend": "fixed"}, "tts": {"backend": "null"}},
        },
    )
    monkeypatch.setattr(run_task, "_build_session", lambda *_args: make_mock_session())
    monkeypatch.setattr(
        run_task,
        "run_robot_task",
        lambda *_args, **_kwargs: {
            "ok": False,
            "steps": [
                {"i": 1, "op": "home", "ok": True, "result": {"context": "PRIVATE_SUCCESS_RESULT"}},
                {"i": 2, "op": "loop:get_grasp_info_simple", "ok": False, "reason": "camera_timeout"},
            ],
            "context": "PRIVATE_RESULT",
        },
    )
    assert run_task.main(["--config", "unused.yaml", "--mock", "--voice-text", "task", "--no-wake"]) == 0
    captured = capsys.readouterr()
    assert captured.out.splitlines() == ["抱歉，没能完成"]
    for output in (captured.err, (tmp_path / "jiuwensymbiosis.log").read_text()):
        assert output.count("Agent failed at step 2 (loop:get_grasp_info_simple): camera_timeout") == 1
        assert "Agent failed at step 1" not in output
        assert "PRIVATE_SUCCESS_RESULT" not in output
        assert "PRIVATE_RESULT" not in output


@pytest.mark.parametrize("tts_backend", ["null", "remote"])
@pytest.mark.parametrize("one_shot", [True, False])
@pytest.mark.parametrize("result_kind", ["text", "fast"])
def test_voice_reply_is_user_output_once_for_each_command(
    monkeypatch, tmp_path, capsys, tts_backend, one_shot, result_kind
):
    from jiuwensymbiosis.voice import loop as voice_loop
    from jiuwensymbiosis.voice.tts import NullTTS

    tts_settings = {"backend": tts_backend}
    if tts_backend == "remote":
        tts_settings["endpoint"] = {"url": "http://tts.invalid"}
    raw = {
        "agent": {"logging": {"dir": str(tmp_path), "level": "WARNING"}},
        "voice": {"asr": {"backend": "fixed"}, "tts": tts_settings},
    }
    monkeypatch.setattr(run_task, "_load_yaml", lambda _: raw)
    monkeypatch.setattr(run_task, "_build_session", lambda *_args: make_mock_session())
    result = {"context": "PRIVATE_CONTEXT"}
    if result_kind == "fast":
        result.update(ok=True, steps_done=3)
    else:
        result["output"] = "FINAL_REPLY"
    reply = "任务完成，共执行 3 步" if result_kind == "fast" else "FINAL_REPLY"
    monkeypatch.setattr(run_task, "run_robot_task", lambda *_a, **_kw: result)
    speaker = NullTTS() if tts_backend == "null" else Mock()
    monkeypatch.setattr(voice_loop, "build_tts_backend", lambda _config: speaker)
    command_count = 1 if one_shot else 2
    if not one_shot:

        def run_forever(self):
            self.handle_command("PRIVATE_TASK_1")
            self.handle_command("PRIVATE_TASK_2")

        monkeypatch.setattr(voice_loop.VoiceLoop, "run_forever", run_forever)
    flags = ["--voice-text", "PRIVATE_TASK"] if one_shot else ["--voice"]
    assert run_task.main(["--config", "unused.yaml", "--mock", "--no-wake", "--tts", tts_backend, *flags]) == 0
    captured = capsys.readouterr()
    assert captured.out.splitlines() == [reply] * command_count
    for output in (captured.err, (tmp_path / "jiuwensymbiosis.log").read_text()):
        assert reply not in output
        assert "PRIVATE_CONTEXT" not in output
        assert "PRIVATE_TASK" not in output
    spoken = speaker.spoken if isinstance(speaker, NullTTS) else [call.args[0] for call in speaker.speak.call_args_list]
    assert spoken.count(reply) == command_count
