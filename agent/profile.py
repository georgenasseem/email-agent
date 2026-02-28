"""User profile management backed by SQLite + profile.json."""
import json
from pathlib import Path
from typing import Any, Dict, Optional

from tools.gmail_tools import get_profile_email
from agent.memory_store import upsert_user_profile, get_user_profile


PROFILE_PATH = Path(__file__).parent.parent / "data" / "profile.json"

_profile_cache: Dict[str, Any] = {}  # Simple per-process cache for load_profile


def _load_all_profiles() -> Dict[str, Any]:
    if not PROFILE_PATH.exists():
        return {}
    try:
        with open(PROFILE_PATH, "r") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception:
        return {}
    return {}


def _save_all_profiles(profiles: Dict[str, Any]) -> None:
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROFILE_PATH, "w") as f:
        json.dump(profiles, f, indent=2, sort_keys=True)


def get_current_user_email() -> str:
    try:
        email = get_profile_email()
        return email or "me@example.com"
    except Exception:
        return "me@example.com"


def load_profile(email: Optional[str] = None) -> Dict[str, Any]:
    """Load merged profile for the given (or current) user from DB + profile.json (cached)."""
    if not email:
        email = get_current_user_email()

    if email in _profile_cache:
        return _profile_cache[email]

    base: Dict[str, Any] = {}
    row = get_user_profile(email)
    if row:
        base.update(
            {
                "email": row.get("email"),
                "display_name": row.get("display_name"),
                "style_notes": row.get("style_notes"),
                "preferences_json": row.get("preferences_json"),
                "roles_json": row.get("roles_json"),
                "working_hours_json": row.get("working_hours_json"),
            }
        )

    profiles = _load_all_profiles()
    from_file = profiles.get(email) or {}
    base.update(from_file)
    _profile_cache[email] = base
    return base


def save_style_notes(style: str, email: Optional[str] = None) -> None:
    """Persist style_notes into both SQLite user_profile and profile.json."""
    if not style:
        return

    if not email:
        email = get_current_user_email()

    # Update DB
    upsert_user_profile(email=email, style_notes=style)

    # Update JSON snapshot
    profiles = _load_all_profiles()
    prof = profiles.get(email) or {}
    prof["email"] = email
    prof["style_notes"] = style
    profiles[email] = prof
    _save_all_profiles(profiles)

