"""Launch the unmodified RoboSim service using its Isaac Python environment.

Uvicorn's automatic uvloop selection replaces the process-wide event-loop
policy, breaking Kit callbacks on the main thread. Keep the asyncio policy
installed by Kit and let RoboSim retain ownership of its main-thread loop.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


def main() -> None:
    """Load RoboSim from a separate checkout and serve its existing HTTP API."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--streaming", action="store_true", help="Run without a local window, with WebRTC rendering")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    repo = args.repo.resolve()
    if not (repo / "sim_service" / "main.py").is_file():
        parser.error("--repo must point to a RiemannRoboSim_Isaac checkout")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    logging.basicConfig(level=logging.INFO)

    import uvicorn
    from sim_service.main import create_app, run_service_loop

    if args.streaming:
        sys.argv.append("--/exts/omni.kit.livestream.app/primaryStream/enableEventTracing=false")
    headless = args.headless or args.streaming
    app = create_app(headless=headless, streaming=args.streaming, shutdown_sim_on_lifespan=False)

    from robosim_identity import install_identity_route
    from robosim_observation import install_observation_route

    install_observation_route(app)
    install_identity_route(app)

    import omni.kit.app

    def serve(server_app):
        """Keep Uvicorn from replacing Kit's asyncio event-loop policy."""
        uvicorn.run(server_app, host="127.0.0.1", port=args.port, loop="asyncio")

    run_service_loop(app, headless, serve, omni.kit.app.get_app())


if __name__ == "__main__":
    main()
