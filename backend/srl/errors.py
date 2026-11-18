"""Errors a request can end in for a reason the client can act on.

Routes and the modules they call raise these; the app sends them as {"error": message} with
their status (srl/api/errors.py). Any other exception that escapes a route is a bug in the app
and leaves as 500 naming it: a KeyError or ValueError from a bug never reads as "not found"
or "bad request".
"""

from __future__ import annotations


class ApiError(Exception):
    """A request that cannot be honored now; the message says why and what to do."""

    status = 409

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        if status is not None:
            self.status = status


class NotFound(ApiError):
    status = 404


class Invalid(ApiError):
    """The request itself is wrong: a value outside what the endpoint accepts."""

    status = 400
