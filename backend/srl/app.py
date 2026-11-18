"""FastAPI application: REST for actions, Server-Sent Events for a run's event stream.

This module only assembles: the app state, how errors leave (srl/api/errors.py), the access
token check, the routers of srl/api/ and the built frontend. Routes that call AWS are plain
``def`` handlers so FastAPI runs them in its thread pool; boto3 is blocking.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import settings
from .api import ROUTERS, auth, errors
from .state import AppState


def create_app() -> FastAPI:
    app = FastAPI(title="Serverless RL InstantStart", docs_url="/api/docs", openapi_url="/api/openapi.json")
    app.state.srl = AppState.load()
    errors.install(app)
    auth.install(app)
    for router in ROUTERS:
        app.include_router(router)
    _frontend(app)
    return app


def _frontend(app: FastAPI) -> None:
    """Registered after every API route: unknown /api paths are a JSON 404, anything else is the single-page app."""

    @app.api_route("/api/{rest:path}", methods=["GET", "POST", "PUT", "DELETE"], include_in_schema=False)
    def api_not_found(rest: str):
        return errors.error_response(404, f"no API route /api/{rest}")

    index = settings.STATIC_DIR / "index.html"
    if index.exists():
        app.mount("/assets", StaticFiles(directory=settings.STATIC_DIR / "assets"), name="assets")

        @app.get("/{rest:path}", include_in_schema=False)
        def spa(rest: str):
            candidate = (settings.STATIC_DIR / rest).resolve()
            if rest and candidate.is_file() and settings.STATIC_DIR.resolve() in candidate.parents:
                return FileResponse(candidate)
            return FileResponse(index)
    else:
        @app.get("/")
        def no_frontend():
            return errors.error_response(503, f"Frontend not built: {index} is missing. Run `npm run build` in frontend/.")
