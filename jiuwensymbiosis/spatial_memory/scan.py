"""Bounded Rule observation loop, independent of a robot or inference backend."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from jiuwensymbiosis.spatial_memory.artifacts import write_detections, write_observation
from jiuwensymbiosis.spatial_memory.completion import assess_completion, observed_scope
from jiuwensymbiosis.spatial_memory.exploration import OnlineMap
from jiuwensymbiosis.spatial_memory.memory import SpatialMemory
from jiuwensymbiosis.spatial_memory.models import Detections, Observation, canonical


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(canonical(value) + "\n")
    temporary.replace(path)


class ScanPort(Protocol):
    """Caller owns connection; execute must independently admit the full robot."""

    def capture(self) -> Observation: ...
    def execute(self, choice: dict, observation: Observation) -> dict: ...
    def confirm_stopped(self) -> bool: ...


class Detector(Protocol):
    def detect(self, rgb: np.ndarray) -> Detections: ...


@dataclass(frozen=True)
class ScanBudget:
    max_frames: int = 16
    max_seconds: float = 600
    max_distance_m: float = 2

    def __post_init__(self) -> None:
        if type(self.max_frames) is not int or not 1 <= self.max_frames <= 1000:
            raise ValueError("max_frames must be in [1, 1000]")
        if any(not np.isfinite(v) or v <= 0 for v in (self.max_seconds, self.max_distance_m)):
            raise ValueError("Time and distance budgets must be finite and positive")


def run_scan(port: ScanPort, detector: Detector, output: Path, *, budget: ScanBudget, cancelled=None) -> dict:
    """Persist every observation before inference; failures remain failed attempts."""
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    capture = output / "capture"
    capture.mkdir()
    online = OnlineMap()
    names: list[str] = []
    gains: list[int] = []
    moves: list[dict] = []
    started = time.monotonic()
    distance = 0.0
    last = None
    reason = "budget_exhausted"
    error = None
    scope = {"kind": "unknown", "boundary_unknown_cells": None, "cells": 0}
    stats = {"frontier_cells": 0}
    stopped = False
    budget_ended = False
    with SpatialMemory(output / "memory.sqlite") as memory:
        try:
            while True:
                if cancelled is not None and cancelled.is_set():
                    reason = "cancelled"
                    break
                if len(names) >= budget.max_frames or time.monotonic() - started >= budget.max_seconds:
                    break
                observation = port.capture()
                if last is not None:
                    distance += float(np.linalg.norm(observation.map_from_camera[:2, 3] - last.map_from_camera[:2, 3]))
                last = observation
                name = f"{len(names):03d}"
                write_observation(capture / name, observation)
                names.append(name)
                _write_json(capture / "sequence.json", {"schema_version": 1, "frames": names})
                detected = detector.detect(observation.rgb)
                write_detections(output / "detections" / name, detected, observation.depth_m.shape)
                receipt = memory.ingest(observation, detected)
                with (output / "receipts.jsonl").open("a") as stream:
                    stream.write(canonical(receipt) + "\n")
                online.integrate(
                    observation.rgb, observation.depth_m, observation.intrinsics, observation.map_from_camera
                )
                gains.append(online.last_new)
                choices, stats = online.candidates(observation.map_from_camera, memory.get_view_tasks())
                # Candidate projection cannot certify complete geometric coverage.
                center, _, free, occupied = online.navigation(observation.map_from_camera, observed_projection=False)
                scope = observed_scope(center, free, occupied)
                if (
                    len(names) >= budget.max_frames
                    or time.monotonic() - started >= budget.max_seconds
                    or distance >= budget.max_distance_m
                ):
                    break
                if cancelled is not None and cancelled.is_set():
                    reason = "cancelled"
                    break
                if not choices:
                    reason = "no_candidates"
                    break
                # A rejection is evidence about executability, not about coverage.
                # Known, localizable memory gaps must not starve behind frontier
                # gain. A refused revisit still falls through to other choices.
                for choice in sorted(choices, key=lambda c: (c["kind"] != "reobserve", -c["utility"])):
                    if time.monotonic() - started >= budget.max_seconds:
                        budget_ended = True
                        break
                    if cancelled is not None and cancelled.is_set():
                        reason = "cancelled"
                        break
                    choice = {**choice, "remaining_distance_m": max(0, budget.max_distance_m - distance)}
                    online.record_choice(choice)
                    result = port.execute(choice, observation)
                    moves.append({"choice": choice, "result": result})
                    _write_json(output / "moves.json", moves)
                    if result.get("motion_executed"):
                        if not result.get("ok") or not result.get("stopped"):
                            raise RuntimeError("motion_failed_or_stop_unconfirmed")
                        break
                    if result.get("stopped") is not True:
                        raise RuntimeError("stop_unconfirmed")
                else:
                    reason = "blocked"
                    break
                if reason == "cancelled" or budget_ended:
                    break
        except Exception as exc:
            reason = "cancelled" if cancelled is not None and cancelled.is_set() else "failed"
            error = f"{type(exc).__name__}: {exc}"
        except BaseException:
            reason = "cancelled"
            raise
        finally:
            try:
                stopped = port.confirm_stopped()
            except Exception as exc:
                error = f"{error or ''} stop confirmation: {exc}".strip()
            if not stopped:
                reason = "stop_unconfirmed"
            report = assess_completion(
                reason=reason,
                scope=scope,
                frontier_cells=stats["frontier_cells"],
                gains=gains,
                tasks=memory.get_view_tasks(),
                stopped=stopped,
            )
            report.update(
                error=error,
                budget=asdict(budget),
                frames=len(names),
                committed_frames=memory.get_status()["observation_count"],
                executed_actions=sum(bool(m["result"].get("motion_executed")) for m in moves),
                measured_camera_path_m=distance,
                elapsed_s=time.monotonic() - started,
                policy="rule_bounded_step",
                pose_source=last.pose_source if last else None,
                geometry=stats,
                memory_status=memory.get_status(),
            )
            online.save(output / "online_map.npz")
            memory.record_scan_report(report)
            _write_json(output / "scan-report.json", report)
    return report
