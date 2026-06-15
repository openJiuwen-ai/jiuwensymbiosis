# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for jiuwensymbiosis.tools.inproc_code."""

from __future__ import annotations

from jiuwensymbiosis.tools.inproc_code import InProcessCodeExec, InProcessCodeTool, make_inproc_code_tool
from jiuwensymbiosis.agent.abstractions import LocalFunction


def _provider():
    return {"x": 42, "y": "hello"}


class TestInProcessCodeExec:
    def test_simple_expression(self):
        e = InProcessCodeExec(_provider)
        r = e.run("1 + 1")
        assert r["ok"] is True
        assert r["result"] is None

    def test_result_variable(self):
        e = InProcessCodeExec(_provider)
        r = e.run("RESULT = x * 2")
        assert r["ok"] is True
        assert r["result"] == 84

    def test_globals_available(self):
        e = InProcessCodeExec(_provider)
        r = e.run("RESULT = x + len(y)")
        assert r["ok"] is True
        assert r["result"] == 47

    def test_stdout_capture(self):
        e = InProcessCodeExec(_provider)
        r = e.run("print('hello')")
        assert r["ok"] is True
        assert "hello" in r["stdout"]

    def test_stderr_capture(self):
        e = InProcessCodeExec(_provider)
        r = e.run("import sys; print('err', file=sys.stderr)")
        assert r["ok"] is True
        assert "err" in r["stderr"]

    def test_exception(self):
        e = InProcessCodeExec(_provider)
        r = e.run("1 / 0")
        assert r["ok"] is False
        assert "ZeroDivisionError" in r["error"]
        assert "ZeroDivisionError" in r["stderr"]

    def test_elapsed(self):
        e = InProcessCodeExec(_provider)
        r = e.run("pass")
        assert r["elapsed_s"] >= 0


class TestInProcessCodeTool:
    def test_run(self):
        t = InProcessCodeTool(_provider)
        r = t.run("RESULT = x + 10")
        assert r["ok"] is True
        assert r["result"] == 52

    def test_as_openjiuwen_tool(self):
        t = InProcessCodeTool(_provider)
        tool = t.as_openjiuwen_tool()
        assert isinstance(tool, LocalFunction)


class TestMakeInprocCodeTool:
    def test_returns_local_function(self):
        tool = make_inproc_code_tool(_provider)
        assert isinstance(tool, LocalFunction)
        assert tool.card.name == "run_python"
