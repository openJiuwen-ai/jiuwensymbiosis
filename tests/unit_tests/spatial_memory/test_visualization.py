"""The viewer exports committed temporal evidence without mutating memory."""

import hashlib
import json
import re

from jiuwensymbiosis.spatial_memory.memory import SpatialMemory
from jiuwensymbiosis.spatial_memory.visualization import write_viewer
from tests.unit_tests.spatial_memory.test_memory import sample


def test_viewer_contains_separate_frame_evidence_and_escapes_labels(tmp_path):
    database = tmp_path / "memory.sqlite"
    with SpatialMemory(database) as memory:
        memory.ingest(*sample(label="</script><script>alert(1)</script>"))
        memory.ingest(*sample(1, shift=5))
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    result = write_viewer(database, tmp_path / "view.html")
    html = (tmp_path / "view.html").read_text()
    payload = re.search(r'<script id="memory-data" type="application/json">(.*?)</script>', html, re.S)[1]
    data = json.loads(payload)
    assert len(data["frames"]) == 2 and len(data["objects"]) == 2
    assert all(p["status"] == "created" for p in data["frames"][0]["parts"])
    assert all(p["status"] == "matched" for p in data["frames"][1]["parts"])
    assert data["frames"][0]["image"].startswith("data:image/jpeg;base64,")
    assert "</script><script>alert(1)</script>" not in html
    assert result["source_sha256"] == before == hashlib.sha256(database.read_bytes()).hexdigest()
