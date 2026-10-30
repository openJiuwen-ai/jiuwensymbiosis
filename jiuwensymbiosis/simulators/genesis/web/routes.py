"""Local signaling and static player, with no eager Genesis or media SDK imports."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from fastapi import HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field
from starlette.requests import Request

from jiuwensymbiosis.simulators.assets import AssetError, AssetLibrary
from jiuwensymbiosis.simulators.genesis.scene_catalog import scene_catalog
from jiuwensymbiosis.simulators.genesis.stream import PreviewStream, StreamError
from jiuwensymbiosis.simulators.genesis.web.local_request import require_local_action

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
    vx: float = Field(default=0, ge=-0.3, le=0.3, allow_inf_nan=False, strict=True)
    vy: float = Field(default=0, ge=-0.3, le=0.3, allow_inf_nan=False, strict=True)
    wz: float = Field(default=0, ge=-0.6, le=0.6, allow_inf_nan=False, strict=True)
    scene_instance_id: str = Field(min_length=32, max_length=32, pattern="^[a-f0-9]+$")
    config_revision: int = Field(ge=0, strict=True)


class StartOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    resolution: Literal["1280x720", "1920x1080"] = "1280x720"
    fps: Literal[30, 60] = 60
    render_quality: Literal["reference"] = "reference"
    robot_mode: Literal["inspection", "moz1_wheels"] = "inspection"
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


def install_genesis_routes(app: Any, service: PreviewStream | None = None) -> PreviewStream:
    """Install local preview routes; the application must await stop() on shutdown."""
    from fastapi.responses import HTMLResponse, Response
    from fastapi.staticfiles import StaticFiles

    class RevalidatedAssets(StaticFiles):
        async def get_response(self, path: str, scope):
            response = await super().get_response(path, scope)
            # Local files change in place; every module/CSS response must be revalidated.
            response.headers["Cache-Control"] = "no-cache"
            return response

    service = service or PreviewStream()

    def guard(request: Request, action: str) -> None:
        require_local_action(request, action, header="x-genesis-action")

    @app.get("/genesis/backend")
    def status():
        return service.status()

    @app.get("/genesis/scenes")
    def scenes():
        try:
            return scene_catalog()
        except AssetError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/genesis/capture")
    async def capture(request: Request, body: CaptureOptions):
        guard(request, "capture")
        try:
            result = await asyncio.to_thread(service.preview.capture_image, body.camera_name)
        except (ValueError, RuntimeError) as exc:
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

    @app.get("/genesis/library")
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

    @app.post("/genesis/backend/start")
    async def start(request: Request, body: StartOptions | None = None):
        guard(request, "start")
        try:
            return await service.start(body.model_dump() if body is not None else None)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/genesis/backend/stop")
    async def stop(request: Request):
        guard(request, "stop")
        return await service.stop()

    @app.post("/genesis/backend/configure")
    async def configure(request: Request, body: StartOptions):
        guard(request, "configure")
        try:
            return await service.configure(body.model_dump())
        except StreamError as exc:
            raise HTTPException(exc.status_code, str(exc)) from exc

    @app.post("/genesis/offer")
    async def offer(request: Request, body: Offer):
        guard(request, "offer")
        try:
            return await service.offer(body.sdp)
        except StreamError as exc:
            raise HTTPException(exc.status_code, str(exc)) from exc

    @app.post("/genesis/disconnect")
    async def disconnect(request: Request, body: Session):
        guard(request, "disconnect")
        await service.disconnect(body.session_id)
        return {"disconnected": True}

    @app.post("/genesis/heartbeat")
    async def heartbeat(request: Request, body: Session):
        guard(request, "heartbeat")
        try:
            service.heartbeat(body.session_id)
            return {"alive": True}
        except StreamError as exc:
            raise HTTPException(exc.status_code, str(exc)) from exc

    @app.post("/genesis/camera")
    async def camera(request: Request, body: CameraCommand):
        guard(request, "camera")
        try:
            return await service.camera(
                body.session_id, body.action, **body.model_dump(exclude={"session_id", "action"})
            )
        except StreamError as exc:
            raise HTTPException(exc.status_code, str(exc)) from exc

    @app.get("/genesis/viewer")
    def viewer():
        html = (_ASSETS / "viewer.html").read_text(encoding="utf-8")
        for name in ("viewer.js", "viewer.css"):
            version = sha256((_ASSETS / name).read_bytes()).hexdigest()[:16]
            html = html.replace(f'/genesis/assets/{name}"', f'/genesis/assets/{name}?v={version}"')
        return HTMLResponse(
            html,
            headers={
                "Cache-Control": "no-store",
                "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self'; "
                "connect-src 'self'; media-src 'self' blob:; img-src 'self' data:; "
                "frame-ancestors 'self'; object-src 'none'; base-uri 'none'",
            },
        )

    @app.post("/genesis/base")
    async def base(request: Request, body: BaseCommand):
        guard(request, "base")
        try:
            return await service.base_command(body.session_id, **body.model_dump(exclude={"session_id"}))
        except StreamError as exc:
            raise HTTPException(exc.status_code, str(exc)) from exc

    @app.post("/genesis/selection")
    async def selection(request: Request, body: SelectionCommand):
        guard(request, "selection")
        try:
            return await service.select(body.session_id, **body.model_dump(exclude={"session_id"}))
        except StreamError as exc:
            raise HTTPException(exc.status_code, str(exc)) from exc

    app.mount("/genesis/assets", RevalidatedAssets(directory=_ASSETS), name="genesis-assets")
    return service
