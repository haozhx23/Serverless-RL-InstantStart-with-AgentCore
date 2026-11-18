"""How errors leave the app: always as {"error": "<what happened and what to do>"}.

ApiError and its subclasses carry their own status. A failed AWS call is 502 with the
operation and the service's error code. Anything else is a bug and leaves as 500 naming the
exception; it is not turned into "not found" or "bad request" for being a KeyError or ValueError.
"""

from __future__ import annotations

from botocore.exceptions import BotoCoreError, ClientError
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ..errors import ApiError


def error_response(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


def install(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(_, e: ApiError):
        return error_response(e.status, str(e))

    @app.exception_handler(ClientError)
    async def _client_error(_, e: ClientError):
        code = e.response["Error"].get("Code")
        return error_response(502, f"AWS {e.operation_name} failed: {code}: {e.response['Error'].get('Message')}")

    @app.exception_handler(BotoCoreError)
    async def _boto_error(_, e: BotoCoreError):
        return error_response(502, f"AWS call failed: {e}")

    @app.exception_handler(Exception)
    async def _bug(request: Request, e: Exception):
        # Starlette raises the exception again after sending this, so the server log keeps the traceback.
        return error_response(500, f"Internal error in {request.method} {request.url.path}: {type(e).__name__}: {e}. "
                                   "This is a bug in the app; the traceback is in the server log.")
