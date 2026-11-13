"""Local signaling and static player, with no eager Genesis or media SDK imports."""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import asdict
from hashlib import sha256
from html import escape
from pathlib import Path
from typing import Any, Literal

from fastapi import HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field
from starlette.requests import Request

from jiuwensymbiosis.simulators.assets import AssetError, AssetLibrary
from jiuwensymbiosis.simulators.common.preview_backends import MODES, POLICY_MODES, PhysicsBackend, RobotMode
from jiuwensymbiosis.simulators.common.scene_catalog import scene_catalog
from jiuwensymbiosis.simulators.common.stream import PreviewStream, StreamError
from jiuwensymbiosis.simulators.common.web.local_request import require_local_action

_ASSETS = Path(__file__).parent / "assets"


class Offer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sdp: str = Field(min_length=1, max_length=65536)
    type: Literal["offer"]


class Session(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: str = Field(min_length=32, max_length=32, pattern="^[a-f0-9]+$")


class CameraCommand(Session):
    action: Literal["orbit", "pan", "zoom", "reset", "fly", "focus"]
    dx: float = Field(default=0, ge=-1, le=1, allow_inf_nan=False, strict=True)
    dy: float = Field(default=0, ge=-1, le=1, allow_inf_nan=False, strict=True)
    delta: float = Field(default=0, ge=-1, le=1, allow_inf_nan=False, strict=True)
    forward: float = Field(default=0, ge=-1, le=1, allow_inf_nan=False, strict=True)
    strafe: float = Field(default=0, ge=-1, le=1, allow_inf_nan=False, strict=True)
    lift: float = Field(default=0, ge=-1, le=1, allow_inf_nan=False, strict=True)


class SelectionCommand(Session):
    action: Literal["pick", "clear"]
    u: float = Field(default=0, ge=0, le=1, allow_inf_nan=False, strict=True)
    v: float = Field(default=0, ge=0, le=1, allow_inf_nan=False, strict=True)
    scene_instance_id: str = Field(min_length=32, max_length=32, pattern="^[a-f0-9]+$")
    config_revision: int = Field(ge=0, strict=True)
    camera_revision: int = Field(ge=0, strict=True)


class BaseCommand(Session):
    # The preview enforces the current robot's declared envelope, not a body-independent cap.
    vx: float = Field(default=0, allow_inf_nan=False, strict=True)
    vy: float = Field(default=0, allow_inf_nan=False, strict=True)
    wz: float = Field(default=0, allow_inf_nan=False, strict=True)
    scene_instance_id: str = Field(min_length=32, max_length=32, pattern="^[a-f0-9]+$")
    config_revision: int = Field(ge=0, strict=True)


class StartOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    resolution: Literal["960x540", "1280x720", "1920x1080"] = "1280x720"
    fps: Literal[30, 60] = 60
    render_quality: Literal["reference"] = "reference"
    backend: PhysicsBackend = "genesis"
    robot_mode: RobotMode = "inspection"
    robot_id: str = Field(default="", max_length=1024, pattern=r"^(|library://[a-zA-Z0-9_-]+/[a-zA-Z0-9_./%~-]+)$")
    camera_name: str = Field(default="observer", max_length=64, pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$")
    scene_id: str = Field(
        default="demo",
        min_length=1,
        max_length=1024,
        pattern=r"^(demo|[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+|library://[a-zA-Z0-9_-]+/[a-zA-Z0-9_./%~-]+)$",
    )


class CaptureOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    camera_name: str = Field(default="observer", max_length=64, pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def install_simulation_routes(app: Any, service: PreviewStream | None = None) -> PreviewStream:
    """Install local preview routes; the application must await stop() on shutdown."""
    from fastapi import APIRouter
    from fastapi.responses import HTMLResponse, Response
    from fastapi.staticfiles import StaticFiles

    class RevalidatedAssets(StaticFiles):
        async def get_response(self, path: str, scope):
            response = await super().get_response(path, scope)
            # Local files change in place; every module/CSS response must be revalidated.
            response.headers["Cache-Control"] = "no-cache"
            return response

    service = service or PreviewStream()
    router = APIRouter()

    def guard(request: Request, action: str) -> None:
        require_local_action(
            request,
            action,
            header="x-simulation-action" if request.url.path.startswith("/simulation/") else "x-genesis-action",
        )

    @router.get("/backend")
    def status():
        return service.status()

    @router.get("/scenes")
    def scenes():
        try:
            entries = scene_catalog()
            if sys.platform == "darwin":
                entries = [
                    entry
                    for entry in entries
                    if entry["id"] == "demo" or entry.get("preview_mode") in ("mjcf", *POLICY_MODES)
                ]
            return entries
        except AssetError as exc:
            raise HTTPException(409, str(exc)) from exc

    @router.post("/capture")
    async def capture(request: Request, body: CaptureOptions):
        guard(request, "capture")
        try:
            result = await asyncio.to_thread(service.preview.capture_image, body.camera_name)
        except (ValueError, RuntimeError, TimeoutError) as exc:
            raise HTTPException(409, str(exc)) from exc
        jpeg = result.pop("jpeg")
        return Response(
            jpeg,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "no-store",
                "X-Genesis-Camera": json.dumps(result, separators=(",", ":")),
            },
        )

    @router.get("/library")
    def library(
        query: str = "",
        root_id: str | None = None,
        kind: str | None = None,
        limit: int = Query(default=30, ge=1, le=1000),
        offset: int = Query(default=0, ge=0),
    ):
        try:
            return asdict(AssetLibrary().search(query=query, root_id=root_id, kind=kind, limit=limit, offset=offset))
        except AssetError as exc:
            raise HTTPException(409, str(exc)) from exc

    @router.post("/backend/start")
    async def start(request: Request, body: StartOptions | None = None):
        guard(request, "start")
        try:
            return await service.start(body.model_dump() if body is not None else None)
        except (ValueError, RuntimeError, TimeoutError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @router.post("/backend/stop")
    async def stop(request: Request):
        guard(request, "stop")
        try:
            return await service.stop()
        except (RuntimeError, TimeoutError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @router.post("/backend/reset")
    async def reset_scene(request: Request):
        guard(request, "reset")
        try:
            return await service.reset_scene()
        except StreamError as exc:
            raise HTTPException(exc.status_code, str(exc)) from exc

    @router.post("/backend/configure")
    async def configure(request: Request, body: StartOptions):
        guard(request, "configure")
        try:
            return await service.configure(body.model_dump())
        except StreamError as exc:
            raise HTTPException(exc.status_code, str(exc)) from exc

    @router.post("/offer")
    async def offer(request: Request, body: Offer):
        guard(request, "offer")
        try:
            return await service.offer(body.sdp)
        except StreamError as exc:
            raise HTTPException(exc.status_code, str(exc)) from exc

    @router.post("/disconnect")
    async def disconnect(request: Request, body: Session):
        guard(request, "disconnect")
        await service.disconnect(body.session_id)
        return {"disconnected": True}

    @router.post("/heartbeat")
    async def heartbeat(request: Request, body: Session):
        guard(request, "heartbeat")
        try:
            service.heartbeat(body.session_id)
            return {"alive": True}
        except StreamError as exc:
            raise HTTPException(exc.status_code, str(exc)) from exc

    @router.post("/camera")
    async def camera(request: Request, body: CameraCommand):
        guard(request, "camera")
        try:
            return await service.camera(
                body.session_id, body.action, **body.model_dump(exclude={"session_id", "action"})
            )
        except StreamError as exc:
            raise HTTPException(exc.status_code, str(exc)) from exc

    @router.get("/viewer")
    def viewer():
        html = (_ASSETS / "viewer.html").read_text(encoding="utf-8")
        options = "".join(
            f'<option data-engines="{"mujoco" if mode in POLICY_MODES else "genesis" if mode == "moz1_wheels" else "genesis,mujoco"}" value="{escape(mode)}">{escape(spec.label)}</option>'
            for mode, spec in MODES.items()
        )
        html = html.replace("{{ROBOT_MODE_OPTIONS}}", options)
        for name in ("viewer.js", "viewer.css"):
            version = sha256((_ASSETS / name).read_bytes()).hexdigest()[:16]
            html = html.replace(f'/simulation/assets/{name}"', f'/simulation/assets/{name}?v={version}"')
        return HTMLResponse(
            html,
            headers={
                "Cache-Control": "no-store",
                "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self'; "
                "connect-src 'self'; media-src 'self' blob:; img-src 'self' data:; "
                "frame-ancestors 'self'; object-src 'none'; base-uri 'none'",
            },
        )

    @router.post("/base")
    async def base(request: Request, body: BaseCommand):
        guard(request, "base")
        try:
            return await service.base_command(body.session_id, **body.model_dump(exclude={"session_id"}))
        except StreamError as exc:
            raise HTTPException(exc.status_code, str(exc)) from exc

    @router.post("/selection")
    async def selection(request: Request, body: SelectionCommand):
        guard(request, "selection")
        try:
            return await service.select(body.session_id, **body.model_dump(exclude={"session_id"}))
        except StreamError as exc:
            raise HTTPException(exc.status_code, str(exc)) from exc

    for prefix in ("/simulation", "/genesis"):
        app.include_router(router, prefix=prefix)
        app.mount(prefix + "/assets", RevalidatedAssets(directory=_ASSETS), name=prefix[1:] + "-assets")
    return service
