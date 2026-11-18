"""What the app keeps in memory while it serves: the scenario catalog, the run manager, and
what the latest account checks saw.

All of it is rebuilt at startup. What has to survive a restart is on disk: runs and settings
in data/, the scenarios in templates/.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from . import settings
from .aws import context as aws_context
from .aws import readiness as aws_readiness
from .aws.context import AwsContext
from .errors import ApiError
from .estimate import CostBasis
from .runs import RunManager
from .scenarios import ScenarioRegistry

MODELS_TTL_S = 600


class AccountCache:
    """The latest readiness result and model lists, behind one lock.

    The readiness result belongs to the region that was selected when it ran; switching regions
    forgets it. Model lists are kept per region: readiness fills them, and a live launch fetches
    them again once they are older than MODELS_TTL_S.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._readiness: dict | None = None
        self._models: dict[str, tuple[float, list[str]]] = {}
        self._mtrl_models: dict[str, tuple[float, list[str]]] = {}

    def readiness(self) -> dict:
        """The cached result when it is for the selected region, else a fresh one."""
        with self._lock:
            cached = self._readiness
        if cached and cached["region"] == aws_context.current_region():
            return cached
        return self.refresh_readiness()

    def refresh_readiness(self) -> dict:
        ctx = aws_context.resolve(refresh=True)
        result = aws_readiness.run_all(ctx)
        with self._lock:
            self._readiness = result
            if result["models"]:
                self._models[ctx.region] = (time.time(), result["models"])
            if result["mtrl_models"]:
                self._mtrl_models[ctx.region] = (time.time(), result["mtrl_models"])
        return result

    def forget_readiness(self) -> None:
        with self._lock:
            self._readiness = None

    def models(self, region: str) -> list[str] | None:
        """Trainable models last seen in `region`, however old (for display)."""
        with self._lock:
            cached = self._models.get(region)
        return cached[1] if cached else None

    def mtrl_models(self, region: str) -> list[str] | None:
        with self._lock:
            cached = self._mtrl_models.get(region)
        return cached[1] if cached else None

    def live_models(self, ctx: AwsContext) -> list[str]:
        """Trainable models in ctx.region for a live launch, fetched again once older than MODELS_TTL_S."""
        with self._lock:
            cached = self._models.get(ctx.region)
        if cached and time.time() - cached[0] < MODELS_TTL_S:
            return cached[1]
        check, models = aws_readiness.check_models(ctx)
        if not models:
            raise ApiError(f"Cannot list trainable models in {ctx.region}: {check['observed']}")
        with self._lock:
            self._models[ctx.region] = (time.time(), models)
        return models


@dataclass
class AppState:
    registry: ScenarioRegistry
    runs: RunManager
    account: AccountCache
    basis: CostBasis  # completed live runs, what cost estimates are based on (srl/estimate.py)

    @classmethod
    def load(cls) -> AppState:
        if not (settings.TEMPLATES_DIR / "_shared").is_dir():
            raise RuntimeError(f"run templates not found: {settings.TEMPLATES_DIR / '_shared'} does not exist. The image "
                               "does not contain templates/; mount the checkout's templates/ there (scripts/start.sh "
                               "does), or point SRL_TEMPLATES at it.")
        registry = ScenarioRegistry(settings.TEMPLATES_DIR)
        runs = RunManager(settings.DATA_DIR, registry)
        return cls(
            registry=registry, runs=runs, account=AccountCache(),
            basis=CostBasis(lambda: [runs.runs_dir, *runs.recording_roots()]),
        )
