"""Zoom API tools for creating, reading, updating, and deleting meetings.

Uses Zoom Server-to-Server OAuth. Set in .env:
  ZOOM_ACCOUNT_ID, ZOOM_CLIENT_ID, ZOOM_CLIENT_SECRET
"""
import os
import base64
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import requests

# ─── Token cache ─────────────────────────────────────────────────────────────
_zoom_token_cache: dict = {"token": None, "expires_at": 0}


def zoom_available() -> bool:
    """Return True if Zoom credentials are configured."""
    return bool(
        os.getenv("ZOOM_ACCOUNT_ID")
        and os.getenv("ZOOM_CLIENT_ID")
        and os.getenv("ZOOM_CLIENT_SECRET")
    )


def _get_zoom_token() -> str:
    """Get Zoom Server-to-Server OAuth access token (cached for ~50 min)."""
    if not zoom_available():
        raise ValueError(
            "Zoom credentials not set. Add ZOOM_ACCOUNT_ID, ZOOM_CLIENT_ID, ZOOM_CLIENT_SECRET to .env. "
            "Create a Server-to-Server OAuth app at https://marketplace.zoom.us/"
        )

    # Return cached token if still valid (with 60s safety margin)
    if _zoom_token_cache["token"] and time.time() < _zoom_token_cache["expires_at"] - 60:
        return _zoom_token_cache["token"]

    account_id = os.getenv("ZOOM_ACCOUNT_ID")
    client_id = os.getenv("ZOOM_CLIENT_ID")
    client_secret = os.getenv("ZOOM_CLIENT_SECRET")

    auth = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    resp = requests.post(
        "https://zoom.us/oauth/token",
        params={"grant_type": "account_credentials", "account_id": account_id},
        headers={
            "Authorization": f"Basic {auth}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()

    # Cache token (Zoom tokens are typically valid for 1 hour)
    _zoom_token_cache["token"] = data["access_token"]
    _zoom_token_cache["expires_at"] = time.time() + data.get("expires_in", 3600)

    return data["access_token"]


def _zoom_request(
    method: str,
    path: str,
    *,
    json_body: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Low-level helper for Zoom REST calls."""
    token = _get_zoom_token()
    url = f"https://api.zoom.us/v2{path}"
    resp = requests.request(
        method.upper(),
        url,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=json_body,
        params=params,
        timeout=20,
    )
    # For DELETE endpoints Zoom may return 204 with empty body
    if resp.status_code == 204:
        return {}
    resp.raise_for_status()
    try:
        return resp.json()
    except ValueError:
        return {}


def create_zoom_meeting(
    topic: str,
    start_time: datetime,
    duration_minutes: int = 30,
) -> Dict[str, Any]:
    """Create a scheduled Zoom meeting. Returns dict with join_url, start_url, id, etc."""
    # Zoom API expects ISO 8601 in UTC
    if start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=timezone.utc)
    start_str = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")

    body = {
        "topic": topic[:200],
        "type": 2,  # scheduled meeting
        "start_time": start_str,
        "duration": duration_minutes,
        "timezone": "UTC",
    }
    return _zoom_request("POST", "/users/me/meetings", json_body=body)


def update_zoom_meeting(
    meeting_id: str,
    *,
    topic: Optional[str] = None,
    start_time: Optional[datetime] = None,
    duration_minutes: Optional[int] = None,
    agenda: Optional[str] = None,
) -> Dict[str, Any]:
    """Update basic meeting fields (topic, start_time, duration, agenda)."""
    body: Dict[str, Any] = {}
    if topic is not None:
        body["topic"] = topic[:200]
    if start_time is not None:
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)
        body["start_time"] = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        body["timezone"] = "UTC"
    if duration_minutes is not None:
        body["duration"] = duration_minutes
    if agenda is not None:
        body["agenda"] = agenda[:2000]
    if not body:
        return {}
    return _zoom_request("PATCH", f"/meetings/{meeting_id}", json_body=body)

