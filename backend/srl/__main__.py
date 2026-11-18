"""python -m srl: serve the API and the built frontend."""

from __future__ import annotations

import ipaddress
import sys

import uvicorn

from . import settings


def _loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def main() -> None:
    if not _loopback(settings.HOST) and not settings.AUTH_TOKEN:
        sys.exit(f"Refusing to listen on {settings.HOST} without SRL_AUTH_TOKEN: anyone who reaches the port could "
                 "start training sessions with this machine's AWS credentials. Set SRL_AUTH_TOKEN, or keep the "
                 "default 127.0.0.1 and use SSH or SSM port forwarding.")
    reload = "--reload" in sys.argv
    uvicorn.run("srl.app:create_app", factory=True, host=settings.HOST, port=settings.PORT, reload=reload,
                reload_dirs=[str(settings.ROOT / "backend" / "srl")] if reload else None, log_level="info")


if __name__ == "__main__":
    main()
