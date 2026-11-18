"""What routes take besides the request body: the app state, the UI language and the AWS context.

Routes declare them as parameter types, e.g. ``def get_artifacts(ctx: Identity)``; FastAPI
fills them in. Sync dependencies run in the thread pool like the routes, so resolving the
context (which may call STS) does not block the event loop.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Header, Request

from ..aws import context as aws_context
from ..aws.context import AwsContext
from ..errors import ApiError
from ..i18n import request_lang
from ..state import AppState


def _state(request: Request) -> AppState:
    return request.app.state.srl


def _lang(accept_language: str | None = Header(None)) -> str:
    """The UI's language, from Accept-Language (docs/architecture.md 3.7). Errors stay in English."""
    return request_lang(accept_language)


def _ctx() -> AwsContext:
    return aws_context.resolve()


def _identity() -> AwsContext:
    ctx = aws_context.resolve()
    if not ctx.ready:
        raise ApiError(f"No AWS identity: {ctx.identity_error}")
    return ctx


State = Annotated[AppState, Depends(_state)]
Lang = Annotated[str, Depends(_lang)]
# The selected region and the caller's identity, which may be missing: not every route needs one.
Ctx = Annotated[AwsContext, Depends(_ctx)]
# The same, for routes that only make sense with an identity: 409 without one.
Identity = Annotated[AwsContext, Depends(_identity)]
