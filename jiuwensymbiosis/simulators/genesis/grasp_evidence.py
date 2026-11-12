"""Measured grasp lifecycle: contact needs lift; lost payload needs supported release."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GraspEvidence:
    contact_s: float = 0.15
    release_s: float = 0.25
    loss_s: float = 0.1
    lift_m: float = 0.015
    slip_m: float = 0.003
    slip_rad: float = 0.05
    state: str = "clear"
    object_id: int | None = None
    contact_time: float = 0.0
    loss_time: float = 0.0
    support_time: float = 0.0
    released_time: float = 0.0
    reference: np.ndarray | None = None
    height: float | None = None
    reason: str | None = None
    translation_error: float = 0.0
    rotation_error: float = 0.0

    @property
    def holding(self) -> bool | None:
        return True if self.state == "held" else False if self.state == "clear" else None

    def observe(self, *, dt, object_id, relative, height, contact, supported, speed, angular_speed, opened):
        from scipy.spatial.transform import Rotation

        relative = np.asarray(relative, dtype=float)
        invalid_pose = relative.shape != (4, 4) or not np.isfinite(relative).all()
        invalid_motion = not np.isfinite([dt, height, speed, angular_speed]).all()
        invalid_ranges = dt <= 0 or speed < 0 or angular_speed < 0
        if invalid_pose or invalid_motion or invalid_ranges:
            self.state = "unknown"
            self.reason = "invalid_readback"
            raise ValueError("invalid grasp observation")
        resting = supported and speed < 0.015 and angular_speed < 0.05
        self.support_time = self.support_time + dt if resting else 0.0
        if self.object_id is None and contact and not opened:
            self.object_id, self.height = object_id, height
            self.reference = relative.copy()
            self.state = "contact" if supported else "unknown"
            self.reason = None if supported else "initial_support_unconfirmed"
        same = object_id == self.object_id
        if self.object_id is None:
            return
        slipped = False
        if same and self.reference is not None:
            delta = np.linalg.inv(self.reference) @ relative
            self.translation_error = float(np.linalg.norm(delta[:3, 3]))
            self.rotation_error = float(Rotation.from_matrix(delta[:3, :3]).magnitude())
            slipped = self.translation_error > self.slip_m or self.rotation_error > self.slip_rad
        good = same and contact and not slipped
        self.contact_time = self.contact_time + dt if good else 0.0
        self.loss_time = 0.0 if good else self.loss_time + dt
        if self.state in {"contact", "held"} and (slipped or self.loss_time >= self.loss_s):
            self.state = "unknown"
            self.reason = "payload_slip" if slipped else "payload_contact_lost"
        lifted = self.height is not None and height - self.height >= self.lift_m and not supported
        if self.state == "contact" and self.contact_time >= self.contact_s and lifted:
            self.state = "held"
        # Opening alone does not certify either an empty hand or a stable place.
        released = same and opened and not contact and resting
        self.released_time = self.released_time + dt if released else 0.0
        if self.released_time >= self.release_s:
            self.state = "clear"
            # Observe until next close so a fall during retreat revokes placement confirmation.
            self.reference = self.height = None
            self.contact_time = self.loss_time = 0.0
            self.reason = None

    def report(self):
        return {
            "state": self.state,
            "object_id": self.object_id,
            "holding_payload": self.holding,
            "grasp_confirmed": self.holding is True,
            "contact_confirmed": self.contact_time >= self.contact_s,
            "support_confirmed": self.support_time >= self.release_s,
            "placement_confirmed": self.released_time >= self.release_s,
            "reason": self.reason,
            "translation_error_m": self.translation_error,
            "rotation_error_rad": self.rotation_error,
        }
