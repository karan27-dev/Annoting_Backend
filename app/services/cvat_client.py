"""All CVAT REST API calls live here.

CVAT runs on YOUR server. The platform talks to it ONLY through this client —
never the CVAT database. If CVAT credentials aren't configured, methods raise
CvatNotConfigured so callers can degrade gracefully in dev.
"""
from __future__ import annotations

import httpx

from app.config import settings


class CvatNotConfigured(RuntimeError):
    pass


class CvatClient:
    def __init__(self) -> None:
        self.base = settings.cvat_url.rstrip("/")
        self.user = settings.cvat_api_user
        self.password = settings.cvat_api_password
        self._token: str | None = None

    @property
    def configured(self) -> bool:
        return bool(self.user and self.password)

    def _ensure(self) -> None:
        if not self.configured:
            raise CvatNotConfigured(
                "CVAT_API_USER / CVAT_API_PASSWORD not set — configure CVAT to enable this."
            )

    async def _auth_headers(self) -> dict[str, str]:
        self._ensure()
        if self._token is None:
            async with httpx.AsyncClient(timeout=30) as c:
                r = await c.post(
                    f"{self.base}/api/auth/login",
                    json={"username": self.user, "password": self.password},
                )
                r.raise_for_status()
                self._token = r.json()["key"]
        return {"Authorization": f"Token {self._token}"}

    async def _request(self, method: str, path: str, **kwargs):
        headers = await self._auth_headers()
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.request(method, f"{self.base}{path}", headers=headers, **kwargs)
            r.raise_for_status()
            return r.json() if r.content else {}

    async def _request_bytes(self, path: str) -> bytes:
        headers = await self._auth_headers()
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.get(f"{self.base}{path}", headers=headers)
            r.raise_for_status()
            return r.content

    # ── Projects / tasks / jobs ────────────────────────────────────────────────
    async def create_project(self, name: str, labels: list[dict]) -> dict:
        return await self._request(
            "POST", "/api/projects", json={"name": name, "labels": labels}
        )

    async def create_task(
        self,
        name: str,
        project_id: int,
        segment_size: int | None = None,
        overlap: int = 0,
    ) -> dict:
        # segment_size 0/None = don't split: CVAT keeps the entire dataset in
        # one job, so an annotator sees the full container at once.
        payload: dict = {"name": name, "project_id": project_id, "overlap": overlap}
        if segment_size:
            payload["segment_size"] = segment_size
        return await self._request("POST", "/api/tasks", json=payload)

    async def create_ground_truth_job(self, task_id: int, frame_count: int) -> dict:
        return await self._request(
            "POST",
            "/api/jobs",
            json={
                "task_id": task_id,
                "type": "ground_truth",
                "frame_selection_method": "random_uniform",
                "frame_count": frame_count,
            },
        )

    async def assign_job(self, cvat_job_id: int, cvat_user_id: int) -> dict:
        return await self._request(
            "PATCH", f"/api/jobs/{cvat_job_id}", json={"assignee": cvat_user_id}
        )

    async def get_quality_report(self, task_id: int) -> dict:
        return await self._request("GET", f"/api/quality/reports?task_id={task_id}")

    async def trigger_export(self, task_id: int, fmt: str = "COCO 1.0") -> dict:
        return await self._request(
            "POST", f"/api/tasks/{task_id}/exports", json={"format": fmt}
        )

    async def attach_data_remote(
        self, task_id: int, url: str, image_quality: int = 70
    ) -> dict:
        """Ingest data into a task from a remote URL (e.g. a presigned R2 zip).
        CVAT downloads + extracts it asynchronously. Returns {rq_id: ...}."""
        return await self._request(
            "POST",
            f"/api/tasks/{task_id}/data",
            json={
                "remote_files": [url],
                "image_quality": image_quality,
                "use_zip_chunks": True,
                "use_cache": True,
            },
        )

    async def task_status(self, task_id: int) -> dict:
        """{'state': 'Queued'|'Started'|'Finished'|'Failed', 'message': ...}."""
        return await self._request("GET", f"/api/tasks/{task_id}/status")

    async def get_task(self, task_id: int) -> dict:
        return await self._request("GET", f"/api/tasks/{task_id}")

    async def list_task_jobs(self, task_id: int) -> list[dict]:
        data = await self._request(
            "GET", f"/api/jobs?task_id={task_id}&page_size=1000"
        )
        return data.get("results", [])

    async def register_user(
        self, username: str, email: str, password: str
    ) -> dict:
        """Create a CVAT account for an annotator (no auth needed)."""
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(
                f"{self.base}/api/auth/register",
                json={
                    "username": username,
                    "email": email,
                    "password1": password,
                    "password2": password,
                    "first_name": username,
                    "last_name": "Annotator",
                },
            )
            r.raise_for_status()
            return r.json()

    async def find_user_id(self, username: str) -> int | None:
        data = await self._request(
            "GET", f"/api/users?search={username}&page_size=10"
        )
        for u in data.get("results", []):
            if u.get("username") == username:
                return u.get("id")
        return None

    # ── Review: read a job's annotations + frames ───────────────────────────────
    async def get_job_annotations(self, job_id: int) -> dict:
        """{'shapes': [...], 'tracks': [...], 'tags': [...]} for a job."""
        return await self._request("GET", f"/api/jobs/{job_id}/annotations")

    async def get_job_labels(self, job_id: int) -> list[dict]:
        data = await self._request(
            "GET", f"/api/labels?job_id={job_id}&page_size=1000"
        )
        return data.get("results", [])

    async def get_job_meta(self, job_id: int) -> dict:
        """Frame metadata incl. per-frame width/height and names."""
        return await self._request("GET", f"/api/jobs/{job_id}/data/meta")

    async def get_frame_bytes(
        self, job_id: int, frame: int, quality: str = "compressed"
    ) -> bytes:
        return await self._request_bytes(
            f"/api/jobs/{job_id}/data?type=frame&quality={quality}&number={frame}"
        )

    def job_deep_link(self, cvat_job_id: int) -> str:
        """Direct link an annotator opens to work on a specific job."""
        return f"{self.base}/tasks/jobs/{cvat_job_id}"


cvat = CvatClient()
