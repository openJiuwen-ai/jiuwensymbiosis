"""A tiny loopback-only task UI using the public Runtime facade."""

from __future__ import annotations

import json
import logging
import re
import webbrowser
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from jiuwensymbiosis.runtime import (
    RequestConflictError,
    ResourceBlockedError,
    ResourceBusyError,
    RuntimeClosedError,
)

_JOB_PATH = re.compile(r"^/api/jobs/([a-f0-9]{32})$")
_CANCEL_PATH = re.compile(r"^/api/jobs/([a-f0-9]{32})/cancel$")
_MAX_REQUEST_BYTES = 16 * 1024
_POLL_EVENT_LIMIT = 128
logger = logging.getLogger(__name__)

_PAGE = r"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
body {{ font: 16px system-ui, sans-serif; margin: 2rem auto; max-width: 54rem; padding: 0 1rem; }}
textarea {{ box-sizing: border-box; min-height: 6rem; width: 100%; }}
button {{ margin: .5rem .5rem .5rem 0; padding: .45rem .8rem; }}
pre {{ background: #f4f4f4; max-height: 22rem; overflow: auto; padding: 1rem; white-space: pre-wrap; }}
.status {{ font-weight: 600; }}
</style>
<h1>{title}</h1>
<p>This local example submits one task to the shared Runtime, polls its status and events, and can request cancellation.</p>
<label for="query">Task</label>
<textarea id="query" maxlength="4000">Describe the task to run.</textarea>
<div><button id="submit">Submit task</button><button id="cancel" disabled>Cancel current task</button></div>
<p>Runtime: <span class="status" id="runtime">checking</span></p>
<p>Job: <code id="job">none</code> — <span class="status" id="phase">idle</span></p>
<pre id="events" aria-live="polite"></pre>
<script>
let jobId = "";
let cursor = 0;
let pendingSubmission = null;
const show = (id, value) => {{ document.getElementById(id).textContent = value; }};
async function refresh() {{
  try {{
    const stateResponse = await fetch("/api/state", {{cache: "no-store"}});
    const state = await stateResponse.json();
    show("runtime", state.blocked ? "blocked" : (state.busy ? "busy" : "idle"));
    if (!jobId) return;
    const response = await fetch(`/api/jobs/${{jobId}}?after=${{cursor}}`, {{cache: "no-store"}});
    if (!response.ok) throw new Error(`job query failed (${{response.status}})`);
    const data = await response.json();
    show("phase", data.job.phase);
    document.getElementById("cancel").disabled = ["succeeded", "failed", "cancelled", "incomplete", "blocked"].includes(data.job.phase);
    if (data.events.gap) document.getElementById("events").textContent += "[event window expired; showing current job snapshot]\n";
    for (const event of data.events.events) {{
      document.getElementById("events").textContent += `${{event.seq}} ${{event.kind}} ${{JSON.stringify(event.data)}}\n`;
    }}
    const eventOutput = document.getElementById("events");
    if (eventOutput.textContent.length > 200000) eventOutput.textContent = eventOutput.textContent.slice(-200000);
    cursor = data.events.next_seq;
  }} catch (error) {{ show("runtime", String(error)); }}
}}
document.getElementById("submit").addEventListener("click", async () => {{
  if (!pendingSubmission) {{
    pendingSubmission = {{request_id: crypto.randomUUID(), query: document.getElementById("query").value}};
  }}
  try {{
    const response = await fetch("/api/jobs", {{
      method: "POST", headers: {{"Content-Type": "application/json"}},
      body: JSON.stringify(pendingSubmission)
    }});
    const data = await response.json();
    if (!response.ok) {{ pendingSubmission = null; show("phase", data.error || "request rejected"); return; }}
    pendingSubmission = null;
    jobId = data.job.job_id; cursor = 0;
    show("job", jobId); show("phase", data.job.phase);
    document.getElementById("events").textContent = "";
    document.getElementById("cancel").disabled = false;
    refresh();
  }} catch (error) {{ show("phase", "submission response unavailable; click again to retry safely"); }}
}});
document.getElementById("cancel").addEventListener("click", async () => {{
  if (!jobId) return;
  const response = await fetch(`/api/jobs/${{jobId}}/cancel`, {{method: "POST"}});
  const data = await response.json();
  show("phase", data.phase || data.error || `cancel rejected (${{response.status}})`);
}});
refresh(); window.setInterval(refresh, 1000);
</script>
</html>"""


def create_server(runtime, binding, *, host: str, port: int, title: str) -> ThreadingHTTPServer:
    """Create the example server; it deliberately refuses non-loopback binds."""
    if host != "127.0.0.1":
        raise ValueError("the example GUI only binds to 127.0.0.1")

    class Handler(BaseHTTPRequestHandler):
        server_version = "JiuwenExampleGUI/1"

        def _expected_host(self) -> str:
            return f"127.0.0.1:{self.server.server_port}"

        def _local_request(self) -> bool:
            return self.headers.get("Host", "") == self._expected_host()

        def _send_json(self, status: int, value: dict) -> None:
            content = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(content)

        def _send_error(self, status: int, message: str) -> None:
            self._send_json(status, {"error": message})

        def _json_body(self) -> dict:
            if self.headers.get_content_type() != "application/json":
                raise ValueError("Content-Type must be application/json")
            raw_length = self.headers.get("Content-Length", "")
            if not raw_length.isdigit():
                raise ValueError("Content-Length is required")
            length = int(raw_length)
            if not 0 < length <= _MAX_REQUEST_BYTES:
                raise ValueError(f"request body must be 1..{_MAX_REQUEST_BYTES} bytes")
            value = json.loads(self.rfile.read(length))
            if not isinstance(value, dict):
                raise ValueError("request body must be a JSON object")
            return value

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler protocol
            if not self._local_request():
                self._send_error(403, "Host must be 127.0.0.1 on this port")
                return
            parsed = urlparse(self.path)
            if parsed.path == "/":
                content = _PAGE.format(title=escape(title)).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                    "connect-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'",
                )
                self.end_headers()
                self.wfile.write(content)
                return
            if parsed.path == "/api/state":
                self._send_json(200, runtime.get_runtime_state())
                return
            match = _JOB_PATH.fullmatch(parsed.path)
            if match is None:
                self._send_error(404, "not found")
                return
            after_text = parse_qs(parsed.query).get("after", ["0"])[0]
            if not after_text.isdigit():
                self._send_error(400, "after must be a non-negative integer")
                return
            job_id = match.group(1)
            try:
                job = runtime.get_job(job_id)
                events = runtime.read_events(job_id, after_seq=int(after_text), limit=_POLL_EVENT_LIMIT)
            except KeyError:
                self._send_error(404, "unknown job")
                return
            self._send_json(200, {"job": job, "events": events})

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler protocol
            expected_origin = f"http://127.0.0.1:{self.server.server_port}"
            if not self._local_request() or self.headers.get("Origin") != expected_origin:
                self._send_error(403, "same-origin requests from 127.0.0.1 are required")
                return
            parsed = urlparse(self.path)
            if parsed.path == "/api/jobs":
                try:
                    body = self._json_body()
                except ValueError as exc:
                    self._send_error(400, str(exc))
                    return
                query = body.get("query")
                if not isinstance(query, str) or not query.strip() or len(query) > 4000:
                    self._send_error(400, "query must contain 1..4000 characters")
                    return
                request_id = body.get("request_id")
                if not isinstance(request_id, str) or not request_id.strip() or len(request_id) > 128:
                    self._send_error(400, "request_id must contain 1..128 characters")
                    return
                try:
                    job = runtime.submit_task(binding.binding_id, request_id, query)
                except ResourceBlockedError as exc:
                    self._send_error(423, str(exc))
                except ResourceBusyError as exc:
                    self._send_error(409, str(exc))
                except RequestConflictError as exc:
                    self._send_error(409, str(exc))
                except RuntimeClosedError:
                    self._send_error(503, "runtime is closing")
                except ValueError as exc:
                    self._send_error(400, str(exc))
                except Exception:
                    logger.exception("example GUI rejected task submission")
                    self._send_error(503, "task submission failed")
                else:
                    self._send_json(202, {"job": job})
                return

            match = _CANCEL_PATH.fullmatch(parsed.path)
            if match is None:
                self._send_error(404, "not found")
                return
            try:
                result = runtime.cancel(match.group(1))
            except KeyError:
                self._send_error(404, "unknown job")
                return
            self._send_json(200, result)

        def log_message(self, format: str, *args) -> None:
            logger.info("example GUI HTTP %s", format % args)

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def serve(runtime, binding, *, host: str, port: int, title: str, open_browser: bool) -> None:
    """Serve the local task page until shutdown; always close the listening socket."""
    server = create_server(runtime, binding, host=host, port=port, title=title)
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"Example GUI listening at {url}")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
