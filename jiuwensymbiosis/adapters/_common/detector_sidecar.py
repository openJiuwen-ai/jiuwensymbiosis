# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Detector sidecar — spawn the open-vocabulary detection server
(``jiuwen_robotics.serving.grounding_dino_sam2_server``) as a subprocess.

The server is intentionally NOT imported in-process: it loads heavy CUDA
models (GroundingDINO + SAM2) and conflicts with vLLM/torch state if hosted in
the same process.

If the chosen port is already accepting connections, we assume an external
detector instance is already running and just attach to it (no subprocess
spawned).
"""

from __future__ import annotations

import logging
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from typing import Iterator, Optional

logger = logging.getLogger(__name__)


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0


def _wait_for_port(host: str, port: int, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _port_open(host, port, timeout=1.0):
            return True
        time.sleep(1.0)
    return False


@contextmanager
def detector_subprocess(
    *,
    host: str = "127.0.0.1",
    port: int = 8114,
    device: str = "cuda",
    startup_timeout_s: float = 300.0,
    log_stdout: bool = True,
    gdino_model_id: Optional[str] = None,
    sam2_model_id: Optional[str] = None,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    use_sam2: bool = True,
) -> Iterator[Optional[subprocess.Popen]]:
    """Start (or attach to) the GroundingDINO(+SAM2) detection server.

    Yields the ``Popen`` we spawned, or ``None`` if we attached to an external
    instance. Always tears down spawned children on context exit.

    The first spawn downloads the model weights from HuggingFace, so
    ``startup_timeout_s`` defaults high.
    """
    if _port_open(host, port, timeout=0.5):
        logger.info("detector already running at %s:%d, attaching", host, port)
        yield None
        return

    cmd = [
        sys.executable,
        "-m", "jiuwensymbiosis.serving.grounding_dino_sam2_server",
        "--host", host,
        "--port", str(port),
        "--device", device,
        "--box-threshold", str(box_threshold),
        "--text-threshold", str(text_threshold),
    ]
    if gdino_model_id:
        cmd += ["--gdino-model-id", gdino_model_id]
    if sam2_model_id:
        cmd += ["--sam2-model-id", sam2_model_id]
    if not use_sam2:
        cmd += ["--no-sam2"]
    logger.info("Spawning detector server: %s", " ".join(cmd))

    stdout = None if log_stdout else subprocess.DEVNULL
    stderr = subprocess.STDOUT if log_stdout else subprocess.DEVNULL
    proc = subprocess.Popen(cmd, stdout=stdout, stderr=stderr)

    try:
        if not _wait_for_port(host, port, startup_timeout_s):
            raise RuntimeError(
                f"detector server did not start on {host}:{port} within {startup_timeout_s}s"
            )
        logger.info("detector ready at %s:%d (pid=%d)", host, port, proc.pid)
        yield proc
    finally:
        if proc.poll() is None:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
            except Exception as exc:  # noqa: BLE001
                logger.warning("detector shutdown failed: %s", exc)
        logger.info("detector server stopped")
