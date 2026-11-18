"""Model packages in the tool's model package group: what runs saved, and deleting them."""

from __future__ import annotations

import re

from botocore.exceptions import ClientError

from ..errors import Invalid
from .context import AwsContext, client

DESCRIBE_LIMIT = 100
RUN_ID = re.compile(r"\br-\d{6}-\d{6}-[0-9a-f]{4}\b")


def _describe(sm, arn: str) -> dict:
    d = sm.describe_model_package(ModelPackageName=arn)
    containers = (d.get("InferenceSpecification") or {}).get("Containers") or []
    first = containers[0] if containers else {}
    s3 = first.get("ModelDataUrl") or ((first.get("ModelDataSource") or {}).get("S3DataSource") or {}).get("S3Uri")
    return {"kind": "CHECKPOINT" if first.get("IsCheckpoint") else "MODEL", "s3_uri": s3,
            "status": d.get("ModelPackageStatus")}


def list_packages(ctx: AwsContext) -> dict:
    sm = client("sagemaker", ctx)
    # ListModelPackages returns an empty list for a group that does not exist, so ask directly.
    try:
        sm.describe_model_package_group(ModelPackageGroupName=ctx.model_package_group)
    except ClientError as e:
        if "does not exist" in e.response["Error"].get("Message", "").lower():
            return {"group": ctx.model_package_group, "group_exists": False, "packages": [], "errors": []}
        raise
    rows, token = [], None
    while True:
        kw = {"ModelPackageGroupName": ctx.model_package_group, "MaxResults": 100,
              "SortBy": "CreationTime", "SortOrder": "Descending", **({"NextToken": token} if token else {})}
        r = sm.list_model_packages(**kw)
        rows += r.get("ModelPackageSummaryList", [])
        token = r.get("NextToken")
        if not token:
            break
    out, errors = [], []
    for i, p in enumerate(rows):
        desc = p.get("ModelPackageDescription") or ""
        m = RUN_ID.search(desc)
        row = {"arn": p["ModelPackageArn"], "version": p.get("ModelPackageVersion"), "description": desc,
               "created": p.get("CreationTime"), "run_id": m.group(0) if m else None,
               "kind": None, "s3_uri": None, "status": p.get("ModelPackageStatus")}
        if i < DESCRIBE_LIMIT:
            try:
                row.update(_describe(sm, p["ModelPackageArn"]))
            except ClientError as e:
                errors.append(f"{p['ModelPackageArn']}: {e.response['Error'].get('Message')}")
        out.append(row)
    return {"group": ctx.model_package_group, "group_exists": True, "packages": out, "errors": errors}


def delete(ctx: AwsContext, arn: str) -> dict:
    """Delete the package and its S3 objects; the package is the only reference to them."""
    if f":model-package/{ctx.model_package_group}/" not in arn:
        raise Invalid(f"{arn} is not in this tool's model package group {ctx.model_package_group}")
    from sagemaker.train.training_session import ServiceClient
    svc = ServiceClient(region=ctx.region, role_arn=ctx.role_arn, model_package_group_arn=ctx.model_package_group_arn)
    r = svc.delete_model_package(arn, delete_artifacts=True)
    return {"arn": arn, "deleted": r.model_package_deleted, "objects_deleted": r.objects_deleted,
            "artifacts_retained": r.artifacts_retained, "retained_reason": r.retained_reason, "s3_uri": r.s3_uri}
