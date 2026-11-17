"""Lifecycle wiring for the installed memory scan command, without an LLM."""

import signal
import threading

from jiuwensymbiosis.agent.config import RobotAgentConfig
from jiuwensymbiosis.simulators.common.agent import make_session
from jiuwensymbiosis.simulators.common.resident import ResidentPreview
from jiuwensymbiosis.simulators.genesis.memory_scan import GenesisScanPort
from jiuwensymbiosis.spatial_memory.scan import ScanBudget, run_scan
from jiuwensymbiosis.spatial_memory.yoloe import YoloEPromptFree


def scan_resident(args) -> dict:
    """Attach only; the external resident service remains its owner's resource."""
    budget = ScanBudget(args.max_frames, args.max_seconds, args.max_distance)
    cancelled = threading.Event()
    handlers = {}
    preview = ResidentPreview(state_dir=args.state_dir)
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            handlers[sig] = signal.signal(sig, lambda *_: cancelled.set())
        preview.connect()
        if preview.status().get("state") != "ready":
            raise RuntimeError("Start the Genesis mobile scene before scanning")
        detector = YoloEPromptFree(args.weights, args.sha256, device=args.device)
        session = make_session(preview, cancelled=cancelled, observation_mode="ground_truth")
        RobotAgentConfig().prepare_session(session)
        with session:
            port = GenesisScanPort(session, map_frame=args.map_frame, depth_stride=args.depth_stride)
            return run_scan(port, detector, args.output, budget=budget, cancelled=cancelled)
    finally:
        preview.close()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
