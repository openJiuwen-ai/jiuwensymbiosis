"""Dependency-free container readiness check."""

import argparse
import json
import urllib.request


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="HTTP service base URL")
    parser.add_argument("service", choices=("vision", "speech"))
    args = parser.parse_args(argv)
    headers = {"X-Request-ID": "container-health"}
    try:
        request = urllib.request.Request(args.url.rstrip("/") + "/v1/health", headers=headers)
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=3) as response:
            raw = response.read(65537)
        if len(raw) > 65536:
            return 1
        data = json.loads(raw)
        result = data["result"]
        return int(
            data.get("schema_version") != 1
            or data.get("request_id") != "container-health"
            or "error" in data
            or result.get("service") != args.service
            or not (result.get("ready") is True or result.get("status") == "ready")
        )
    except Exception:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
