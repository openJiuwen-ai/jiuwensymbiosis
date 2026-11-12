"""Named trajectories and measured single-clock execution; owner alone steps physics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from scipy.interpolate import PPoly, make_interp_spline


def finite(value, shape=None):
    a = np.asarray(value, dtype=float)
    if not np.isfinite(a).all() or (shape is not None and a.shape != shape):
        raise ValueError("nonfinite or incorrectly shaped motion data")
    return a


@dataclass(frozen=True)
class Trajectory:
    names: tuple[str, ...]
    positions: np.ndarray
    dt: float
    binding: str
    start: dict[str, float]
    velocity_limits: np.ndarray
    acceleration_limits: np.ndarray
    jerk_limits: np.ndarray
    stationary_names: tuple[str, ...] = ()

    def __post_init__(self):
        q = finite(self.positions)
        invalid_names = not self.names or len(set(self.names)) != len(self.names)
        invalid_shape = q.ndim != 2 or q.shape[1] != len(self.names) or len(q) < 2
        if invalid_names or invalid_shape:
            raise ValueError("trajectory requires unique names and at least two samples")
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("invalid trajectory timestep")
        q = q.copy()
        q.setflags(write=False)
        object.__setattr__(self, "positions", q)
        for name in ("velocity_limits", "acceleration_limits", "jerk_limits"):
            limits = finite(getattr(self, name), (len(self.names),)).copy()
            if np.any(limits <= 0):
                raise ValueError("motion limits must be positive")
            limits.setflags(write=False)
            object.__setattr__(self, name, limits)
        start = {name: float(value) for name, value in self.start.items()}
        if set(self.names) - start.keys() or not np.isfinite(list(start.values())).all():
            raise ValueError("complete finite trajectory start required")
        if not np.allclose(q[0], [start[n] for n in self.names], atol=1e-5, rtol=0):
            raise ValueError("trajectory does not start at measured state")
        object.__setattr__(self, "start", MappingProxyType(start))
        zeros = np.zeros(len(self.names))
        spline = make_interp_spline(
            np.arange(len(q)) * self.dt, q, k=5, bc_type=([(1, zeros), (2, zeros)], [(1, zeros), (2, zeros)])
        )
        object.__setattr__(self, "_spline", spline)
        watched = tuple(start) if not self.stationary_names else tuple(self.stationary_names)
        if set(watched) - start.keys() or set(self.names) - set(watched):
            raise ValueError("invalid stationary joint set")
        object.__setattr__(self, "stationary_names", watched)
        self.validate_dynamics()

    @property
    def duration(self):
        return (len(self.positions) - 1) * self.dt

    def sample(self, elapsed):
        return self._spline(np.clip(elapsed, 0.0, self.duration))

    def _extrema_values(self, column, order):
        curve = PPoly.from_spline((self._spline.t, self._spline.c[:, column], self._spline.k)).derivative(order)
        roots = curve.derivative().roots(extrapolate=False)
        interior = roots[np.isfinite(roots) & (roots >= 0) & (roots <= self.duration)]
        return curve(np.r_[0.0, self.duration, interior, self._spline.t])

    def position_bounds(self):
        values = [self._extrema_values(column, 0) for column in range(len(self.names))]
        return np.array([(v.min(), v.max()) for v in values])

    def derivative_maxima(self):
        return [
            np.array([np.max(np.abs(self._extrema_values(column, order))) for column in range(len(self.names))])
            for order in (1, 2, 3)
        ]

    def validate_dynamics(self):
        for values, limits in zip(
            self.derivative_maxima(), (self.velocity_limits, self.acceleration_limits, self.jerk_limits), strict=True
        ):
            if np.any(values > limits * (1 + 1e-5)):
                raise ValueError("trajectory exceeds velocity/acceleration/jerk limits")


class TrajectoryFollower:
    """Finite-force targets only; failures brake and require measured settling."""

    def __init__(
        self,
        read: Callable,
        write: Callable,
        binding: Callable,
        *,
        tracking_limit=0.08,
        goal_tolerance=0.01,
        stopped_velocity=0.02,
        settle_s=0.25,
        stop_timeout_s=4.0,
    ):
        self.read, self.write, self.binding = read, write, binding
        self.tracking_limit, self.goal_tolerance = tracking_limit, goal_tolerance
        self.stopped_velocity, self.settle_s, self.stop_timeout_s = stopped_velocity, settle_s, stop_timeout_s
        self.trajectory: Trajectory | None = None
        self.result: dict = {"status": "idle", "ok": False}
        self.terminal = False
        self.phase = "idle"
        self.elapsed = self.waited = self.stable = 0.0
        self.failure: str | None = None
        self.target = np.empty(0)

    @property
    def active(self):
        return self.phase in {"running", "settling", "braking"}

    def start(self, trajectory: Trajectory):
        if self.active or self.terminal:
            raise RuntimeError("executor busy or terminal")
        trajectory.validate_dynamics()
        q, dq = self.read()
        finite([q[n] for n in trajectory.stationary_names])
        finite([dq[n] for n in trajectory.stationary_names])
        if self.binding() != trajectory.binding:
            raise ValueError("stale model/scene/base binding")
        if any(
            name not in q or abs(q[name] - value) > 0.01
            for name, value in trajectory.start.items()
            if name in trajectory.stationary_names
        ):
            raise ValueError("stale planning start state")
        if any(abs(dq[n]) > self.stopped_velocity for n in trajectory.stationary_names):
            raise ValueError("robot must be measured stopped before execution")
        self.trajectory = trajectory
        self.elapsed = self.waited = self.stable = 0.0
        self.phase, self.failure = "running", None
        self.target = trajectory.positions[0].copy()
        self.result = {
            "status": "running",
            "ok": False,
            "goal_reached": False,
            "settled": False,
            "stop_confirmation": "unknown",
            "max_tracking_error": 0.0,
        }
        self.write(dict(zip(trajectory.names, self.target, strict=True)))

    def cancel(self, reason="cancelled"):
        if not self.active:
            return
        if self.trajectory is None:
            raise RuntimeError("active executor has no trajectory")
        q, _ = self.read()
        self.target = finite([q[n] for n in self.trajectory.names])
        self.failure, self.phase, self.waited, self.stable = reason, "braking", 0.0, 0.0
        self.write(dict(zip(self.trajectory.names, self.target, strict=True)))

    def tick(self, dt: float):
        if not self.active:
            return
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("invalid control timestep")
        t = self.trajectory
        if t is None:
            raise RuntimeError("active executor has no trajectory")
        try:
            qmap, vmap = self.read()
            q, dq = finite([qmap[n] for n in t.names]), finite([vmap[n] for n in t.names])
        except (ValueError, KeyError, RuntimeError):
            self.write(dict(zip(t.names, self.target, strict=True)))
            self.terminal, self.phase = True, "failed"
            self.result.update(status="completed", reason="readback_unavailable", stop_confirmation="unknown")
            return
        if self.binding() != t.binding and self.phase != "braking":
            self.cancel("scene_or_base_changed")
        error = float(np.max(np.abs(q - self.target)))
        self.result["max_tracking_error"] = max(self.result["max_tracking_error"], error)
        if error > self.tracking_limit and self.phase != "braking":
            self.cancel("tracking_error")
        if self.phase == "running":
            self.elapsed += dt
            self.target = t.sample(self.elapsed)
            self.write(dict(zip(t.names, self.target, strict=True)))
            if self.elapsed >= t.duration:
                self.phase = "settling"
        else:
            self.waited += dt
            settled = bool(
                np.max(np.abs(dq)) < self.stopped_velocity and np.max(np.abs(q - self.target)) < self.goal_tolerance
            )
            self.stable = self.stable + dt if settled else 0.0
            if self.stable >= self.settle_s:
                success = self.failure is None
                self.phase = "completed"
                self.result.update(
                    status="completed",
                    ok=success,
                    goal_reached=success,
                    settled=True,
                    stop_confirmation="measured",
                    reason=self.failure,
                    final_positions=dict(zip(t.names, q.tolist(), strict=True)),
                )
            elif self.waited > self.stop_timeout_s:
                if self.phase == "settling":
                    self.cancel("settling_timeout")
                else:
                    self.terminal, self.phase = True, "failed"
                    self.result.update(
                        status="completed", ok=False, reason="stop_unconfirmed", stop_confirmation="unknown"
                    )
