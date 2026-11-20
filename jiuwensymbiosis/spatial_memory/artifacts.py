"""Portable immutable RGB-D bundles and successful-inference replay records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from jiuwensymbiosis.spatial_memory.models import (
    Detection,
    Detections,
    Observation,
    canonical,
    observation_from_arrays,
    pack,
    unpack,
)


def write_observation(directory: Path, observation: Observation) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    payload = pack(**observation.arrays())
    (directory / "rgbd.npz").write_bytes(payload)
    (directory / "observation.json").write_text(
        canonical(
            {
                "schema_version": 1,
                "observation": observation.metadata(),
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
        + "\n"
    )


def read_observation(directory: Path) -> Observation:
    metadata = json.loads((directory / "observation.json").read_text())
    payload = (directory / "rgbd.npz").read_bytes()
    if metadata["schema_version"] != 1 or hashlib.sha256(payload).hexdigest() != metadata["payload_sha256"]:
        raise ValueError("Observation bundle schema or digest mismatch")
    return observation_from_arrays(metadata["observation"], unpack(payload))


def write_detections(directory: Path, detections: Detections, shape: tuple[int, int]) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    payload = pack(
        masks=np.stack([d.mask for d in detections.instances]) if detections.instances else np.empty((0, *shape), bool)
    )
    (directory / "masks.npz").write_bytes(payload)
    (directory / "detections.json").write_text(
        canonical(
            {
                "schema_version": 1,
                "instances": [d.metadata() for d in detections.instances],
                "provenance": json.loads(detections.provenance_json),
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
        + "\n"
    )


def read_detections(directory: Path) -> Detections:
    metadata = json.loads((directory / "detections.json").read_text())
    payload = (directory / "masks.npz").read_bytes()
    if metadata["schema_version"] != 1 or hashlib.sha256(payload).hexdigest() != metadata["payload_sha256"]:
        raise ValueError("Detection bundle schema or digest mismatch")
    masks = unpack(payload)["masks"]
    return Detections(
        tuple(Detection(**record, mask=mask) for record, mask in zip(metadata["instances"], masks, strict=True)),
        canonical(metadata["provenance"]),
    )
