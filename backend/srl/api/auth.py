"""Access token. When SRL_AUTH_TOKEN is set, every /api route except OPEN_PATHS needs it."""

from __future__ import annotations

import hmac

from fastapi import APIRouter, Body, FastAPI, Request
from fastapi.responses import JSONResponse

from .. import settings
from ..errors import ApiError
from .errors import error_response

AUTH_COOKIE = "srl_token"
OPEN_PATHS = ("/api/health", "/api/auth/status", "/api/auth/login")

router = APIRouter(tags=["auth"])


def authed(request: Request) -> bool:
    if not settings.AUTH_TOKEN:
        return True
    given = request.cookies.get(AUTH_COOKIE) or request.headers.get("x-auth-token") or ""
    return hmac.compare_digest(given.encode(), settings.AUTH_TOKEN.encode())


def install(app: FastAPI) -> None:
    @app.middleware("http")
    async def _auth(request: Request, call_next):
        path = request.url.path
        if path.startswith("/api/") and path not in OPEN_PATHS and not authed(request):
            return error_response(401, "Sign in with the access token printed when the app started")
        return await call_next(request)


@router.get("/api/health")
def health():
    return {"ok": True}


@router.get("/api/auth/status")
def auth_status(request: Request):
    return {"required": bool(settings.AUTH_TOKEN), "authenticated": authed(request)}


@router.post("/api/auth/login")
def login(token: str = Body(..., embed=True)):
    if not settings.AUTH_TOKEN:
        return {"authenticated": True}
    if not hmac.compare_digest(token.encode(), settings.AUTH_TOKEN.encode()):
        raise ApiError("That token does not match the one printed when the app started", 401)
    resp = JSONResponse({"authenticated": True})
    resp.set_cookie(AUTH_COOKIE, token, httponly=True, samesite="strict")
    return resp
