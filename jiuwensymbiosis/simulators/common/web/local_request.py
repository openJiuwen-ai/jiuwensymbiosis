"""Same-origin, loopback guard for GUI-owned local process controls."""

from urllib.parse import urlsplit

from fastapi import HTTPException
from starlette.requests import Request


def require_local_action(request: Request, action: str, *, header: str) -> None:
    """Reject cross-origin forms, remote callers and rebinding hosts before side effects."""
    try:
        origin = urlsplit(request.headers.get("origin", ""))
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="页面来源无效。") from exc
    if request.client is None or request.client.host not in {"127.0.0.1", "::1"}:
        raise HTTPException(status_code=403, detail="只允许本机 GUI 页面管理后台。")
    if (
        request.url.hostname not in {"127.0.0.1", "localhost", "::1"}
        or origin.netloc != request.url.netloc
        or origin.scheme != request.url.scheme
    ):
        raise HTTPException(status_code=403, detail="只允许本机 GUI 页面管理后台。")
    if request.headers.get(header) != action:
        raise HTTPException(status_code=403, detail="只允许本机 GUI 页面管理后台。")
