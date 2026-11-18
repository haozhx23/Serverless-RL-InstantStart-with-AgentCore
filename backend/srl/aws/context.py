"""Which account and region the app works in, and the resource names derived from them."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass

import boto3

from .. import settings
from ..errors import Invalid

_lock = threading.Lock()
_identity: dict[str, tuple[float, dict]] = {}
IDENTITY_TTL_S = 300


@dataclass
class AwsContext:
    region: str
    account: str | None
    caller_arn: str | None
    identity_error: str | None
    role_name: str = settings.ROLE_NAME
    model_package_group: str = settings.MODEL_PACKAGE_GROUP

    @property
    def ready(self) -> bool:
        return self.account is not None

    @property
    def role_arn(self) -> str | None:
        return f"arn:aws:iam::{self.account}:role/service-role/{self.role_name}" if self.account else None

    @property
    def model_package_group_arn(self) -> str | None:
        return (f"arn:aws:sagemaker:{self.region}:{self.account}:model-package-group/{self.model_package_group}"
                if self.account else None)

    @property
    def bucket(self) -> str | None:
        return f"sagemaker-{self.region}-{self.account}" if self.account else None

    @property
    def s3_output_path(self) -> str | None:
        return f"s3://{self.bucket}/{settings.S3_PREFIX}/" if self.account else None

    def run_aws_config(self) -> dict:
        """The aws block written into a run's config.json."""
        return {"region": self.region, "role_arn": self.role_arn,
                "model_package_group_arn": self.model_package_group_arn,
                "s3_output_path": self.s3_output_path, "session_name_prefix": settings.SESSION_PREFIX}

    def public(self) -> dict:
        d = asdict(self)
        d.update(role_arn=self.role_arn, model_package_group_arn=self.model_package_group_arn,
                 bucket=self.bucket, s3_output_path=self.s3_output_path,
                 supported_regions=settings.SUPPORTED_REGIONS, session_prefix=settings.SESSION_PREFIX)
        return d


def _settings_path():
    return settings.DATA_DIR / "settings.json"


def load_settings() -> dict:
    """data/settings.json: region, MTRL selections per region, last MTRL access probe."""
    path = _settings_path()
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_settings(data: dict) -> None:
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _settings_path().with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp.replace(_settings_path())


def current_region() -> str:
    return load_settings().get("region") or settings.DEFAULT_REGION


def set_region(region: str) -> None:
    if region not in settings.SUPPORTED_REGIONS:
        raise Invalid(f"Training sessions are available in {', '.join(settings.SUPPORTED_REGIONS)}; "
                         f"{region!r} is not one of them")
    data = load_settings()
    data["region"] = region
    save_settings(data)


def resolve(refresh: bool = False) -> AwsContext:
    region = current_region()
    with _lock:
        cached = _identity.get(region)
    if cached and not refresh and time.time() - cached[0] < IDENTITY_TTL_S:
        ident = cached[1]
    else:
        try:
            r = boto3.client("sts", region_name=region).get_caller_identity()
            ident = {"account": r["Account"], "caller_arn": r["Arn"], "identity_error": None}
        except Exception as e:  # no credentials, expired token, no network: say which
            ident = {"account": None, "caller_arn": None, "identity_error": f"{type(e).__name__}: {e}"}
        with _lock:
            _identity[region] = (time.time(), ident)
    return AwsContext(region=region, **ident)


def client(service: str, ctx: AwsContext):
    return boto3.client(service, region_name=ctx.region)
