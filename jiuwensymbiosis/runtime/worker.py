"""Single-task execution through the existing agent, planner, rails and session."""

from __future__ import annotations

import logging
import math
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from jiuwensymbiosis.agent import ModelSpec, RobotAgentConfig, run_robot_task
from jiuwensymbiosis.agent.cancel import CancelToken, RunCancelled
from jiuwensymbiosis.agent.lifecycle import CleanupReport, HardwareCleanupError
from jiuwensymbiosis.errors import error_code
from jiuwensymbiosis.utils.logging import get_logger

if TYPE_CHECKING:
    from jiuwensymbiosis.runtime.bindings import BindingSnapshot
    from jiuwensymbiosis.runtime.jobs import Runtime
    from jiuwensymbiosis.runtime.resources import ResourceLease

logger = get_logger(__name__)
_PRIVATE_KEYS = {"api_key", "password", "secret", "token", "access_token", "authorization"}
_FORBIDDEN_OPTIONS = {"model", "model_spec", "extra_rails", "rails", "workspace", "exec_config"}


def build_agent_config(binding, options):
    if not isinstance(options, dict):
        raise ValueError("agent_options must be an object")
    data = binding.config_data()
    agent = dict(data.get("agent") or {})
    agent.update(options)
    forbidden = _FORBIDDEN_OPTIONS.intersection(agent)
    if forbidden:
        raise ValueError(f"runtime agent options do not accept {sorted(forbidden)}")
    config = RobotAgentConfig.from_dict(agent)
    config.model_spec = ModelSpec(**(data.get("model") or {}))
    config.workspace = str(binding.workspace)
    return config


class Sanitizer:
    """Bounded JSON facts with configured credentials removed before persistence."""

    def __init__(self, config):
        self.secrets = set()

        def collect(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if str(key).lower() in _PRIVATE_KEYS and isinstance(item, str) and item:
                        self.secrets.add(item)
                    else:
                        collect(item)
            elif isinstance(value, list):
                for item in value:
                    collect(item)

        collect(config)

    def __call__(self, value, depth=0):
        if depth > 12:
            return "[depth limit]"
        if isinstance(value, dict):
            return {
                str(k): "[redacted]" if str(k).lower() in _PRIVATE_KEYS else self(v, depth + 1)
                for k, v in list(value.items())[:1000]
            }
        if isinstance(value, (list, tuple)):
            return [self(v, depth + 1) for v in value[:1000]]
        if value is None or isinstance(value, (bool, int)):
            return value
        if isinstance(value, float) and math.isfinite(value):
            return value
        text = value if isinstance(value, str) else repr(value)
        for secret in self.secrets:
            text = text.replace(secret, "[redacted]")
        return text[:16000]


class EventEmitter:
    def __init__(self, runtime, job_id, token, sanitize):
        self.runtime, self.job_id, self.token, self.sanitize = runtime, job_id, token, sanitize

    def event(self, kind, data):
        try:
            self.runtime.store.append_event(self.job_id, kind, self.sanitize(data))
        except Exception as exc:
            self.runtime.storage_failed(exc)
            self.token.set()
            raise

    def step_started(self, data):
        self.event("step_started", data)

    def step_finished(self, data):
        self.event("step_finished", data)

    def safety_event(self, data):
        self.event("safety_event", data)

    def frame(self, rgb):
        ref = self.runtime.artifacts.frame(self.job_id, rgb)
        if ref:
            self.event("frame", ref)

    def step_frame(self, index, rgb):
        ref = self.runtime.artifacts.frame(self.job_id, rgb, slot=str(index))
        if ref:
            self.event("step_frame", {"index": index, "artifact": ref})


class EventLogHandler(logging.Handler):
    def __init__(self, emitter):
        super().__init__(logging.INFO)
        self.emitter = emitter
        self.tail = deque(maxlen=400)
        self._accepting = True

    def emit(self, record):
        if not self._accepting:
            return
        try:
            message = record.getMessage()
            if record.exc_info:
                message += "\n" + "".join(traceback.format_exception(*record.exc_info)).rstrip()
            data = self.emitter.sanitize({"level": record.levelname, "name": record.name, "msg": message})
            self.tail.append(f"{data['level']} {data['name']}: {data['msg']}")
            self.emitter.event("log", data)
        except Exception:
            # event() already closes admission and signals cancellation. Never
            # recurse into logging from a handler failure.
            return

    def close(self) -> None:
        # A logging call may already hold a reference when its logger detaches
        # us. Serialize with emit() and reject that late call after close.
        self.acquire()
        try:
            self._accepting = False
        finally:
            self.release()
        super().close()


def _apply_fast_config(session, config):
    if config.exec_mode == "fastagent" and config.exec_config is None:
        from jiuwensymbiosis.agent.fast import SkillExecConfig, servo_config_from_session

        servo = servo_config_from_session(session)
        if servo is not None:
            config.exec_config = SkillExecConfig(servo=servo)


@dataclass(frozen=True)
class JobRequest:
    """Everything one background task owns: identity, binding, lease and cancel."""

    runtime: Runtime
    job_id: str
    binding: BindingSnapshot
    query: str
    options: dict
    lease: ResourceLease
    token: CancelToken


def run_job(request: JobRequest) -> None:
    runtime, job_id, binding, query, options, lease, token = (
        request.runtime,
        request.job_id,
        request.binding,
        request.query,
        request.options,
        request.lease,
        request.token,
    )
    from jiuwensymbiosis.runtime.events import TaskEventRail
    from jiuwensymbiosis.runtime.results import normalize_result

    sanitize = Sanitizer(binding.config_data())
    emitter = EventEmitter(runtime, job_id, token, sanitize)
    handler = EventLogHandler(emitter)
    log = get_logger("jiuwensymbiosis")

    def detach_log() -> None:
        log.removeHandler(handler)
        handler.close()

    session = None
    config = None
    unrecoverable_cleanup: tuple[str, ...] = ()
    outcome = {"ok": False, "error": "execution did not start"}
    log.addHandler(handler)
    try:
        token.raise_if_set()
        config = build_agent_config(binding, options)
        session = binding.build_session()
        session.cancel_token = token
        session.motion_log_dir = config.motion_log_dir
        _apply_fast_config(session, config)
        config.extra_rails = [TaskEventRail(emitter, session, should_stop=token.is_set)]
        runtime.store.update(job_id, {"phase": "connecting"}, event_kind="connecting", event_data={})
        session.connect()
        token.raise_if_set()
        runtime.store.update(
            job_id,
            {"phase": "running"},
            event_kind="run_started",
            event_data={"query": sanitize(query), "adapter": binding.adapter},
        )
        from jiuwensymbiosis.env.protocol import JointDriver

        try:
            observation = session.env.get_observation()
        except Exception as exc:
            logger.debug("initial observation unavailable: %s", exc)
        else:
            if isinstance(getattr(session.env, "low_level", None), JointDriver) and observation.joints:
                pose = {
                    "binding_id": binding.binding_id,
                    "adapter": binding.adapter,
                    "joints": [float(v) for v in observation.joints],
                }
                runtime.store.update(job_id, {"start_pose": pose}, event_kind="start_pose", event_data=pose)
            if observation.rgb is not None:
                emitter.frame(observation.rgb)
        result = run_robot_task(session, query, config, conversation_id=f"gui-{job_id}", cancel_token=token)
        outcome = {
            "ok": True,
            "result": sanitize(result),
            "conversation_id": f"gui-{job_id}",
            "workspace": str(binding.workspace),
        }
    except RunCancelled:
        outcome = {
            "ok": True,
            "result": {"result_type": "stopped", "output": "cancelled"},
            "workspace": str(binding.workspace),
        }
    except Exception as exc:
        if isinstance(exc, HardwareCleanupError):
            unrecoverable_cleanup = (str(exc),)
        logger.exception("Task execution failed")
        outcome = {
            "ok": False,
            "error": sanitize(f"{type(exc).__name__}: {exc}"),
            "error_type": type(exc).__name__,
            "error_code": error_code(exc),
        }
    finally:
        # Wait for real helper completion before attempting device disconnect.
        # If the bounded wait expires, publish blocked while the owner remains
        # alive to finish cleanup when that helper eventually returns.
        report = CleanupReport(errors=unrecoverable_cleanup)
        try:
            idle = token.wait_for_idle(runtime.cleanup_timeout)
            if session is not None:
                session.disconnect()
                report = session.cleanup_report()
                if unrecoverable_cleanup:
                    report = CleanupReport(
                        pending_work=report.pending_work,
                        connected=report.connected,
                        errors=(*report.errors, *unrecoverable_cleanup),
                    )
            if not idle:
                report = CleanupReport(
                    pending_work=token.pending_work, connected=report.connected, errors=report.errors
                )
            if config is not None:
                trace_root = Path(config.trace_dir) if config.trace_dir else Path(binding.workspace) / "traces"
                references = [
                    runtime.artifacts.register_trace(job_id, path)
                    for path in sorted(trace_root.glob(f"gui-{job_id}_*.json"))[:128]
                ]
                if references:
                    runtime.store.update(
                        job_id, {"artifacts": references}, event_kind="artifacts", event_data={"references": references}
                    )
            outcome["log_tail"] = "\n".join(handler.tail)
            phase = normalize_result(outcome)
            runtime.store.update(
                job_id,
                {"phase": phase if report.released else "blocked", "result": sanitize(outcome)},
                event_kind="run_finished" if report.released else "blocked",
                event_data=sanitize(outcome)
                if report.released
                else {
                    "cleanup": {
                        "pending_work": report.pending_work,
                        "errors": report.errors,
                        "connected": report.connected,
                        "released": False,
                    }
                },
            )
            # Final facts and log capture belong to this operation. Finish them
            # before another task can acquire the process slot.
            if report.released:
                detach_log()
            released = runtime.record_cleanup(job_id, lease, report)
            if not released and report.pending_work:
                token.wait_for_idle(None)
                if session is not None:
                    session.disconnect()
                    report = session.cleanup_report()
                else:
                    report = CleanupReport()
                if unrecoverable_cleanup:
                    report = CleanupReport(
                        pending_work=report.pending_work,
                        connected=report.connected,
                        errors=(*report.errors, *unrecoverable_cleanup),
                    )
                if report.released:
                    runtime.store.update(
                        job_id, {"phase": phase}, event_kind="run_finished", event_data=sanitize(outcome)
                    )
                    detach_log()
                runtime.record_cleanup(job_id, lease, report)
        except Exception as exc:
            runtime.storage_failed(exc)
            # A persistence failure must not skip resource cleanup. Preserve
            # uncertain reports, while clean hardware can still be released.
            if session is not None and not lease.released:
                try:
                    session.disconnect()
                    report = session.cleanup_report()
                except Exception as cleanup_error:
                    report = CleanupReport(
                        errors=(str(cleanup_error),), pending_work=token.pending_work, connected=True
                    )
            if unrecoverable_cleanup:
                report = CleanupReport(
                    pending_work=report.pending_work,
                    connected=report.connected,
                    errors=(*report.errors, *unrecoverable_cleanup),
                )
            try:
                if report.released:
                    detach_log()
                runtime.resources.finish(lease, report)
            except Exception:
                logger.exception("Resource cleanup could not be recorded")
        finally:
            detach_log()
