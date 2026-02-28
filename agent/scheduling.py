"""Scheduling flow: extract meeting intent, find free slots, create events."""
from datetime import datetime, timedelta, timezone
from typing import Any

from tools.calendar_tools import find_free_slots, create_event
from tools.gmail_tools import extract_email_address
from agent.meeting_extractor import extract_meeting_intent


def propose_meeting_times(
    email: dict,
    days_ahead: int = 7,
    duration_minutes: int | None = None,
    max_slots: int = 5,
    force: bool = False,
) -> dict[str, Any]:
    """Extract meeting intent from email and propose free time slots.

    When *force* is True (e.g. user explicitly clicked Schedule), skip the
    LLM intent gate and build a sensible fallback intent from the email
    metadata so scheduling always proceeds.

    Returns dict with meeting_intent, free_slots (list of (start, end) as ISO strings), error.
    """
    intent = extract_meeting_intent(email)
    if not intent.get("has_meeting_intent"):
        if not force:
            return {"meeting_intent": intent, "free_slots": [], "error": "No meeting intent detected"}
        # User explicitly asked to schedule — build fallback intent
        sender = extract_email_address(email.get("sender", ""))
        subject = (email.get("subject") or "Meeting")[:100]
        intent = {
            "has_meeting_intent": True,
            "title": subject,
            "duration_minutes": duration_minutes or 30,
            "attendees": [sender] if sender else [],
            "notes": "",
        }

    # Ensure sender is in attendees if not already
    sender_email = extract_email_address(email.get("sender", ""))
    attendees = intent.get("attendees") or []
    if sender_email and sender_email.lower() not in [a.lower() for a in attendees]:
        attendees = [sender_email] + attendees
    intent["attendees"] = attendees

    duration = duration_minutes or intent.get("duration_minutes", 30)
    tz = timezone.utc
    now = datetime.now(tz)
    # Start from next working-hours window (tomorrow at 9 AM UTC)
    time_min = (now + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
    time_max = time_min + timedelta(days=days_ahead)

    try:
        slots = find_free_slots(
            time_min, time_max,
            duration_minutes=duration,
            respect_working_hours=True,
        )
        # Convert to ISO strings for UI
        slot_strs = [(s[0].isoformat(), s[1].isoformat()) for s in slots[:max_slots]]
        return {"meeting_intent": intent, "free_slots": slot_strs, "error": None}
    except Exception as e:
        return {"meeting_intent": intent, "free_slots": [], "error": str(e)}


def propose_meeting_times_simple(
    duration_minutes: int = 30,
    days_ahead: int = 7,
    max_slots: int = 5,
    attendees: list[str] | None = None,
    title: str = "Meeting",
) -> dict[str, Any]:
    """Propose meeting times without an email — for direct scheduling from the UI.

    Returns dict with meeting_intent, free_slots, error.
    """
    tz = timezone.utc
    now = datetime.now(tz)
    time_min = (now + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
    time_max = time_min + timedelta(days=days_ahead)

    intent = {
        "has_meeting_intent": True,
        "title": title,
        "duration_minutes": duration_minutes,
        "attendees": attendees or [],
        "notes": "",
    }

    try:
        slots = find_free_slots(
            time_min, time_max,
            duration_minutes=duration_minutes,
            respect_working_hours=True,
        )
        slot_strs = [(s[0].isoformat(), s[1].isoformat()) for s in slots[:max_slots]]
        return {"meeting_intent": intent, "free_slots": slot_strs, "error": None}
    except Exception as e:
        return {"meeting_intent": intent, "free_slots": [], "error": str(e)}


def create_event_from_slot(
    meeting_intent: dict,
    start_iso: str,
    end_iso: str,
    add_zoom: bool = False,
) -> dict:
    """Create a calendar event from a selected slot. Optionally create a Zoom meeting and add link to description. Returns the created event dict."""
    start = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
    end = datetime.fromisoformat(end_iso.replace("Z", "+00:00"))
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    description = meeting_intent.get("notes") or ""
    if add_zoom:
        try:
            from tools.zoom_tools import create_zoom_meeting, zoom_available
            if zoom_available():
                duration_minutes = max(15, int((end - start).total_seconds() / 60))
                zoom_meeting = create_zoom_meeting(
                    topic=meeting_intent.get("title", "Meeting"),
                    start_time=start,
                    duration_minutes=duration_minutes,
                )
                join_url = zoom_meeting.get("join_url")
                if join_url:
                    description = (description + "\n\nZoom: " + join_url).strip()
        except Exception:
            pass

    return create_event(
        summary=meeting_intent.get("title", "Meeting"),
        start=start,
        end=end,
        attendees=meeting_intent.get("attendees") or [],
        description=description or None,
    )
