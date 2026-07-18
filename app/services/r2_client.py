"""Cloudflare R2 (S3-compatible) access via boto3.

boto3 is an optional dependency — install it and set the R2_* env vars to enable
real uploads. Until then, presigned-URL calls return a clearly-marked dev stub so
the upload UI flow still works locally.
"""
from __future__ import annotations

from app.config import settings


class R2Client:
    def __init__(self) -> None:
        self._client = None

    @property
    def configured(self) -> bool:
        return bool(
            settings.r2_account_id
            and settings.r2_access_key
            and settings.r2_secret_key
        )

    def _get_client(self):
        if self._client is None:
            import boto3  # imported lazily so the app runs without boto3 installed

            self._client = boto3.client(
                "s3",
                endpoint_url=f"https://{settings.r2_account_id}.r2.cloudflarestorage.com",
                aws_access_key_id=settings.r2_access_key,
                aws_secret_access_key=settings.r2_secret_key,
                region_name="auto",
            )
        return self._client

    def presign_put(self, key: str, content_type: str, expires: int = 3600) -> str:
        if not self.configured:
            return f"https://r2.dev-stub.local/PUT/{key}?expires={expires}"
        return self._get_client().generate_presigned_url(
            "put_object",
            Params={
                "Bucket": settings.r2_bucket_name,
                "Key": key,
                "ContentType": content_type,
            },
            ExpiresIn=expires,
        )

    def put_fileobj(self, key: str, fileobj, content_type: str) -> bool:
        """Server-side upload — streams a file object straight to R2. Used by
        the backend-proxied upload path so the browser never has to satisfy R2
        CORS. Returns False (no-op) when R2 isn't configured (dev)."""
        if not self.configured:
            return False
        self._get_client().upload_fileobj(
            fileobj,
            settings.r2_bucket_name,
            key,
            ExtraArgs={"ContentType": content_type},
        )
        return True

    def presign_get(self, key: str, expires: int = 3600) -> str:
        if not self.configured:
            return f"https://r2.dev-stub.local/GET/{key}?expires={expires}"
        return self._get_client().generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.r2_bucket_name, "Key": key},
            ExpiresIn=expires,
        )

    def count_objects(self, prefix: str) -> int:
        if not self.configured:
            return 0
        resp = self._get_client().list_objects_v2(
            Bucket=settings.r2_bucket_name, Prefix=prefix
        )
        return resp.get("KeyCount", 0)

    def list_keys(self, prefix: str) -> list[str]:
        if not self.configured:
            return []
        resp = self._get_client().list_objects_v2(
            Bucket=settings.r2_bucket_name, Prefix=prefix
        )
        return [o["Key"] for o in resp.get("Contents", [])]

    def object_size(self, key: str) -> int:
        if not self.configured:
            return 0
        resp = self._get_client().head_object(
            Bucket=settings.r2_bucket_name, Key=key
        )
        return int(resp.get("ContentLength", 0))

    def read_range(self, key: str, start: int, end: int) -> bytes:
        """Inclusive byte range — lets us read a ZIP's central directory
        without downloading the whole archive."""
        if not self.configured:
            return b""
        resp = self._get_client().get_object(
            Bucket=settings.r2_bucket_name,
            Key=key,
            Range=f"bytes={start}-{end}",
        )
        return resp["Body"].read()

    def find_dataset_archive(self, prefix: str) -> str | None:
        """The uploaded dataset ZIP under a project's prefix, if any."""
        for key in self.list_keys(prefix):
            if key.lower().endswith((".zip", ".tar", ".tar.gz")):
                return key
        return None


r2 = R2Client()
