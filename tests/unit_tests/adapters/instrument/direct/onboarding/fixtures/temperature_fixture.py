"""Document → Skill → Runtime sequence → feedback, using only an owned HTTP fixture."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from jiuwensymbiosis.adapters.instrument.common.naming import tool_name
from jiuwensymbiosis.adapters.instrument.direct.config import load_config
from jiuwensymbiosis.adapters.instrument.direct.onboarding import add_device
from jiuwensymbiosis.adapters.instrument.direct.skill_verification import verify_device_skills
from jiuwensymbiosis.agent.lifecycle import CleanupReport
from jiuwensymbiosis.runtime.jobs import Runtime

FIXTURES = Path(__file__).resolve().parent


def verify(output: str | Path, *, inject_failure: bool = False) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    requests = []
    active_write = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append({"method": "GET", "path": self.path})
            if self.path == "/status":
                self.respond({"status": "IDLE"})
            elif self.path == "/temperature":
                self.respond({"temperature_c": 20.0})
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path != "/setpoint":
                self.send_error(404)
                return
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append({"method": "POST", "path": self.path, "body": body})
            active_write.set()
            try:
                self.respond({"accepted": not inject_failure})
            finally:
                active_write.clear()

        def respond(self, value):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(value).encode("utf-8"))

        def log_message(self, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    runtime = Runtime(output / "workspace", resource_directory=output / "locks")
    job = None
    try:
        candidate = json.loads((FIXTURES / "temperature.candidate.json").read_text())
        candidate["device"].update(configured=True, configuration_evidence="Local software fixture only")
        candidate["device"]["connection"]["base_url"] = f"http://127.0.0.1:{server.server_port}"
        candidate_path = output / "candidate.json"
        candidate_path.write_text(json.dumps(candidate))
        deployment = output / "deployment.json"
        installed = add_device(
            FIXTURES / "temperature.source.md", candidate_path, deployment, skills_dir=output / "skills"
        )
        installation = load_config(deployment)
        skills = verify_device_skills(installation, output / "skills")
        names = {name: tool_name("temperature-demo", name) for name in candidate["device"]["operation_ids"]}
        config = {
            "adapter": "instrument",
            "instrument": {
                "access": "direct",
                "installation": installation,
                "permissions": {"local": list(names.values())},
            },
        }
        path = output / "runtime.json"
        path.write_text(json.dumps(config))
        binding = runtime.prepare_binding(path)
        if requests:
            raise AssertionError("offline installation/binding opened hardware")
        steps = [
            {"op": names["temperature_read"], "params": {}, "bind": "reading"},
            {"op": names["temperature_setpoint_request"], "params": {"temperature_c": "reading.temperature_c + 15"}},
            {"op": names["temperature_read"], "params": {}, "bind": "after"},
        ]
        job = runtime.submit_actions(binding.binding_id, "fixture-task", steps)
        result = runtime.wait_for_job(job["job_id"], 10)
        feedback = {
            "job": runtime.get_job(job["job_id"]),
            "invocations": runtime.list_invocations(job["job_id"]),
            "events": runtime.read_events(job["job_id"]),
        }
        calls_before = list(requests)
        same = runtime.submit_actions(binding.binding_id, "fixture-task", steps)
        assert same["job_id"] == job["job_id"] and requests == calls_before
        writes = [r for r in requests if r["method"] == "POST"]
        reads = sum(r["path"] == "/temperature" for r in requests)
        expected = "unknown" if inject_failure else "succeeded"
        invocations = feedback["invocations"]
        passed = (
            bool(invocations)
            and invocations[-1]["status"] == expected
            and len(writes) == 1
            and reads == (1 if inject_failure else 2)
        )
        report = {
            "status": "PASS" if passed else "FAIL",
            "result": result,
            "feedback": feedback,
            "hardware_verified": False,
            "skills": skills,
            "installation": installed,
            "requests": list(requests),
            "temperature_reads": reads,
            "ordering_owner": "Runtime/fastagent",
        }
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        return report
    finally:
        # Only this in-memory HTTP fixture is externally verified idle here.
        # We never reinterpret a protocol failure as success or reissue its write.
        if job is not None:
            runtime.wait_for_job(job["job_id"], 10)
            for record in runtime.list_invocations(job["job_id"]):
                if record["status"] == "unknown" and not active_write.is_set():
                    runtime.reconcile_invocation(
                        record["invocation_id"],
                        status="failed",
                        completion_evidence={
                            "source": "owned_software_fixture",
                            "idle": True,
                            "physical_device": False,
                        },
                        cleanup_report=CleanupReport(),
                    )
        runtime.close()
        server.shutdown()
        server.server_close()
        worker.join(2)
