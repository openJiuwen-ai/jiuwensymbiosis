# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Detector sidecar — spawn the open-vocabulary detection server
(``jiuwensymbiosis.serving.grounding_dino_sam2_server``) as a subprocess.

The server is intentionally NOT imported in-process: it loads heavy CUDA
models (GroundingDINO + SAM2) and conflicts with vLLM/torch state if hosted in
the same process.

The local process owns its listening port. An occupied port is a configuration
error; externally managed services use remote mode, including on localhost.
"""

from __future__ import annotations

import logging
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from types import TracebackType

from jiuwensymbiosis.agent.cancel import CancelToken
from jiuwensymbiosis.agent.lifecycle import CleanupReport, HardwareCleanupError
from jiuwensymbiosis.errors import DetectorStartError, InferenceServiceError

logger = logging.getLogger(__name__)


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0


def _wait_for_port(host: str, port: int, timeout: float, *, cancel_token: CancelToken | None = None) -> bool:
    # A present token means the model-load wait must be interruptible: poll faster
    # and raise RunCancelled on cancel (the caller's finally terminates the proc).
    # None → the original 1.0s cadence, unchanged for CLI.
    poll = 0.1 if cancel_token is not None else 1.0
    from jiuwensymbiosis.perception.config import DetectorConfig
    from jiuwensymbiosis.perception.detector_client import create_detector_client
    from jiuwensymbiosis.utils.service_http import HttpEndpointConfig

    deadline = time.monotonic() + timeout
    config = DetectorConfig(
        mode="remote",
        endpoint=HttpEndpointConfig(url=f"http://{host}:{port}", connect_timeout_s=0.5, request_timeout_s=1.0),
    )
    with create_detector_client(config) as client:
        client.bind_cancel_token(cancel_token)
        while time.monotonic() < deadline:
            if cancel_token is not None:
                cancel_token.raise_if_set()
            if _port_open(host, port, timeout=0.5):
                try:
                    client.readiness()
                    return True
                except InferenceServiceError:
                    pass  # Model loading is not readiness; retry within startup budget.
            time.sleep(min(poll, max(0, deadline - time.monotonic())))
        return False


@dataclass
class _DetectorSidecar:
    """Own the child and expose rollback evidence even when entering fails."""

    command: list[str]
    host: str
    port: int
    startup_timeout_s: float
    log_stdout: bool
    cancel_token: CancelToken | None
    _proc: subprocess.Popen | None = field(default=None, init=False)
    _released: bool = field(default=True, init=False)

    def __enter__(self) -> subprocess.Popen | None:
        if _port_open(self.host, self.port, timeout=0.5):
            raise DetectorStartError(
                f"detector port {self.host}:{self.port} is occupied; use mode: remote for externally managed services"
            )
        logger.info("Spawning detector server: %s", " ".join(self.command))
        self._released = False
        stdout = None if self.log_stdout else subprocess.DEVNULL
        stderr = subprocess.STDOUT if self.log_stdout else subprocess.DEVNULL
        self._proc = subprocess.Popen(self.command, stdout=stdout, stderr=stderr)
        try:
            if not _wait_for_port(self.host, self.port, self.startup_timeout_s, cancel_token=self.cancel_token):
                raise DetectorStartError(
                    f"detector server did not start on {self.host}:{self.port} within {self.startup_timeout_s}s"
                )
        except BaseException as original:
            try:
                self.close()
            except Exception as cleanup:
                raise HardwareCleanupError("detector startup", original, (cleanup,)) from original
            raise
        logger.info("detector ready at %s:%d (pid=%d)", self.host, self.port, self._proc.pid)
        return self._proc

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.close()

    def cleanup_report(self) -> CleanupReport:
        return CleanupReport(connected=not self._released)

    def close(self) -> None:
        if self._released:
            return
        proc = self._proc
        if proc is None:
            raise HardwareCleanupError("detector spawn", cleanup_errors=(RuntimeError("no child handle"),))
        try:
            if proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=10)
                except Exception:
                    # A failed terminate/wait is recoverable only if kill + wait
                    # confirms the detector child has exited.
                    proc.kill()
                    proc.wait(timeout=5)
            if proc.poll() is None:
                raise RuntimeError("detector process is still running")
        except Exception as cleanup:
            raise HardwareCleanupError("detector shutdown", cleanup_errors=(cleanup,)) from cleanup
        self._released = True
        self._proc = None
        logger.info("detector server stopped")


def detector_subprocess(
    *,
    host: str = "127.0.0.1",
    port: int = 8114,
    device: str = "cuda",
    startup_timeout_s: float = 300.0,
    log_stdout: bool = True,
    gdino_model_id: str | None = None,
    sam2_model_id: str | None = None,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    use_sam2: bool = True,
    cancel_token: CancelToken | None = None,
) -> _DetectorSidecar:
    """Start an explicitly owned GroundingDINO(+SAM2) detection server.

    The context yields the spawned ``Popen``. Its cleanup report confirms successful startup rollback; failed
    shutdown raises rather than silently declaring the child stopped.

    The first spawn downloads the model weights from HuggingFace, so
    ``startup_timeout_s`` defaults high.
    """
    cmd = [
        sys.executable,
        "-m",
        "jiuwensymbiosis.serving.grounding_dino_sam2_server",
        "--host",
        host,
        "--port",
        str(port),
        "--device",
        device,
        "--box-threshold",
        str(box_threshold),
        "--text-threshold",
        str(text_threshold),
    ]
    if gdino_model_id:
        cmd += ["--gdino-model-id", gdino_model_id]
    if sam2_model_id:
        cmd += ["--sam2-model-id", sam2_model_id]
    if not use_sam2:
        cmd += ["--no-sam2"]
    return _DetectorSidecar(cmd, host, port, startup_timeout_s, log_stdout, cancel_token)
