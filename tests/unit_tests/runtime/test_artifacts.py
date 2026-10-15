import json
import uuid

import numpy as np
import pytest

from jiuwensymbiosis.runtime.artifacts import ArtifactStore, decode_frame


def test_preview_cache_overwrites_latest_and_bounds_step_images(tmp_path):
    store = ArtifactStore(tmp_path)
    job_id = uuid.uuid4().hex
    ref = store.frame(job_id, np.zeros((2, 2, 3), dtype=np.uint8))
    store.frame(job_id, np.ones((2, 2, 3), dtype=np.uint8))
    assert decode_frame(store.read(ref)).tolist() == np.ones((2, 2, 3)).tolist()
    assert len(list((tmp_path / job_id).glob("*.npy"))) == 1
    assert store.frame(job_id, np.zeros((1, 1, 3)), slot="129") is None
    assert store.frame(job_id, np.zeros((1, 1, 3), dtype=object)) is None


def test_artifact_reads_reject_paths_and_symlink_escape(tmp_path):
    store = ArtifactStore(tmp_path / "frames")
    job_id = uuid.uuid4().hex
    with pytest.raises(ValueError):
        store.read({"job_id": "../../secret", "slot": "latest"})
    with pytest.raises(ValueError):
        store.read({"job_id": job_id, "slot": "../../secret"})
    secret = tmp_path / "secret"
    secret.write_bytes(b"not a frame")
    (store.directory / job_id).mkdir()
    (store.directory / job_id / "latest.npy").symlink_to(secret)
    with pytest.raises(ValueError):
        store.read({"job_id": job_id, "slot": "latest"})


def test_registered_trace_is_readable_after_reopening_store(tmp_path):
    store = ArtifactStore(tmp_path / "frames")
    job_id = uuid.uuid4().hex
    trace = tmp_path / f"gui-{job_id}_timestamp.json"
    trace.write_text(json.dumps({"steps": []}))
    ref = store.register_trace(job_id, trace)
    reopened = ArtifactStore(tmp_path / "frames")
    assert json.loads(reopened.read(ref)) == {"steps": []}
    with pytest.raises(ValueError):
        reopened.read({"job_id": job_id, "kind": "trace", "slot": str(trace)})
