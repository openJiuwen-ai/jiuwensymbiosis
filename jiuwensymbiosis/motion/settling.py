"""Continuous measured stopping from fresh velocity samples; stdlib only.

ROS workers load this file directly under the system interpreter. No transport,
clock, sleep or command publication is owned here.
"""

from __future__ import annotations

import math


class StopMonitor:
    """Require advancing measurements to remain below both limits for a duration."""

    def __init__(self, *, linear_limit: float, angular_limit: float, hold_s: float, max_age_s: float):
        for value in (linear_limit, angular_limit, hold_s, max_age_s):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("Stopping thresholds must be finite and positive")
        self.linear_limit = linear_limit
        self.angular_limit = angular_limit
        self.hold_s = hold_s
        self.max_age_s = max_age_s
        self._stamp: int | None = None
        self._sample_at: float | None = None
        self._low_since: float | None = None

    def update(
        self,
        *,
        now: float,
        sample_at: float | None,
        stamp: int | None,
        linear_speed: float | None,
        angular_speed: float | None,
    ) -> bool:
        """Only a new, fresh sample can confirm stopping; repeated packets cannot."""
        values = (now, sample_at, linear_speed, angular_speed)
        if (
            any(v is None or isinstance(v, bool) or not math.isfinite(v) for v in values)
            or type(stamp) is not int
            or not 0 <= now - sample_at <= self.max_age_s
        ):
            self._low_since = None
            return False
        if stamp == self._stamp:
            # Timestamp repetition is not new evidence, even if the callback was called again.
            if self._sample_at is None or now - self._sample_at > self.max_age_s:
                self._low_since = None
            return False
        if (
            self._stamp is not None
            and stamp < self._stamp
            or self._sample_at is not None
            and not 0 < sample_at - self._sample_at <= self.max_age_s
        ):
            self._low_since = None
        self._stamp, self._sample_at = stamp, sample_at
        if abs(linear_speed) >= self.linear_limit or abs(angular_speed) >= self.angular_limit:
            self._low_since = None
            return False
        if self._low_since is None:
            self._low_since = sample_at
        return sample_at - self._low_since >= self.hold_s
