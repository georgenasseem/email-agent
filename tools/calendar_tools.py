"""Google Calendar API tools for listing events, finding free slots, and creating events."""
from datetime import datetime, timedelta, timezone
from typing import Optional

from googleapiclient.discovery import build

from tools.google_auth import get_google_credentials


# ─── Working-hours configuration (override via env or profile later) ────────
WORK_START_HOUR = 9    # 09:00
WORK_END_HOUR = 17     # 17:00
LUNCH_START_HOUR = 12  # 12:00
LUNCH_END_HOUR = 13    # 13:00
SLOT_GAP_MINUTES = 15  # gap between consecutive proposed slots


def _parse_rfc3339(s: str) -> datetime:
    """Parse RFC3339 string to timezone-aware datetime.

    Handles 'Z', '+00:00', and arbitrary offsets like '-05:00'.
    """
    s = s.replace("Z", "+00:00")
    return datetime.fromisoformat(s)


def _to_rfc3339(dt: datetime) -> str:
    """Convert datetime to RFC3339 string for API calls."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    s = dt.isoformat()
    if s.endswith("+00:00"):
        s = s[:-6] + "Z"
    return s


def get_calendar_service():
    """Authenticate and return Google Calendar API service."""
    creds = get_google_credentials()
    return build("calendar", "v3", credentials=creds)


def _is_within_working_hours(dt: datetime, duration_delta: timedelta) -> bool:
    """Check if a slot starting at `dt` and lasting `duration_delta` falls
    within working hours and avoids the lunch break."""
    start_h = dt.hour + dt.minute / 60
    end_dt = dt + duration_delta
    end_h = end_dt.hour + end_dt.minute / 60

    # Must be within work hours
    if start_h < WORK_START_HOUR or end_h > WORK_END_HOUR:
        return False

    # Must not overlap with lunch
    if start_h < LUNCH_END_HOUR and end_h > LUNCH_START_HOUR:
        return False

    return True


def find_free_slots(
    time_min: datetime,
    time_max: datetime,
    duration_minutes: int = 30,
    calendar_id: str = "primary",
    respect_working_hours: bool = True,
) -> list[tuple[datetime, datetime]]:
    """Find free time slots between time_min and time_max.

    Uses the freebusy API to get busy periods, then computes gaps >= duration_minutes.
    When respect_working_hours=True, only returns slots within 9-17 hours,
    skipping the lunch break (12-13) and adding a gap between consecutive slots.

    Returns list of (start, end) datetime tuples for free slots.
    """
    service = get_calendar_service()
    tmin_str = _to_rfc3339(time_min)
    tmax_str = _to_rfc3339(time_max)

    freebusy = (
        service.freebusy()
        .query(
            body={
                "timeMin": tmin_str,
                "timeMax": tmax_str,
                "items": [{"id": calendar_id}],
            }
        )
        .execute()
    )

    busy_list = []
    cal_data = freebusy.get("calendars", {}).get(calendar_id, {})
    tz = time_min.tzinfo or timezone.utc
    for b in cal_data.get("busy", []):
        start = _parse_rfc3339(b["start"])
        end = _parse_rfc3339(b["end"])
        if start.tzinfo is None:
            start = start.replace(tzinfo=tz)
        if end.tzinfo is None:
            end = end.replace(tzinfo=tz)
        busy_list.append((start, end))

    busy_list.sort(key=lambda x: x[0])
    duration_delta = timedelta(minutes=duration_minutes)
    gap_delta = timedelta(minutes=SLOT_GAP_MINUTES) if respect_working_hours else timedelta(0)
    free_slots: list[tuple[datetime, datetime]] = []
    tz = time_min.tzinfo or timezone.utc
    cursor = time_min.replace(tzinfo=tz) if time_min.tzinfo is None else time_min

    def _try_add_slots(from_dt: datetime, until_dt: datetime):
        """Generate slots in a free gap, respecting working hours."""
        nonlocal cursor
        c = from_dt
        while c + duration_delta <= until_dt:
            if respect_working_hours and not _is_within_working_hours(c, duration_delta):
                # Skip to next valid start: either after lunch or next morning
                if c.hour < WORK_START_HOUR:
                    c = c.replace(hour=WORK_START_HOUR, minute=0, second=0, microsecond=0)
                elif c.hour < LUNCH_END_HOUR and c.hour >= LUNCH_START_HOUR:
                    c = c.replace(hour=LUNCH_END_HOUR, minute=0, second=0, microsecond=0)
                elif c.hour >= WORK_END_HOUR:
                    c = (c + timedelta(days=1)).replace(hour=WORK_START_HOUR, minute=0, second=0, microsecond=0)
                else:
                    c += timedelta(minutes=15)  # nudge forward
                continue
            free_slots.append((c, c + duration_delta))
            c += duration_delta + gap_delta
        cursor = max(cursor, c)

    for busy_start, busy_end in busy_list:
        if cursor < busy_start:
            _try_add_slots(cursor, busy_start)
        cursor = max(cursor, busy_end)

    time_max_tz = time_max.replace(tzinfo=tz) if time_max.tzinfo is None else time_max
    if cursor < time_max_tz:
        _try_add_slots(cursor, time_max_tz)

    return free_slots


def list_events(
    time_min: Optional[datetime] = None,
    time_max: Optional[datetime] = None,
    max_results: int = 10,
    calendar_id: str = "primary",
) -> list[dict]:
    """List upcoming calendar events.

    Returns list of dicts with: id, summary, start, end, location, attendees, description.
    """
    service = get_calendar_service()
    now = datetime.now(timezone.utc)
    tmin = _to_rfc3339(time_min or now)
    tmax = _to_rfc3339(time_max or (now + timedelta(days=7)))

    result = (
        service.events()
        .list(
            calendarId=calendar_id,
            timeMin=tmin,
            timeMax=tmax,
            maxResults=max_results,
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
    )

    events = []
    for item in result.get("items", []):
        start_raw = item.get("start", {})
        end_raw = item.get("end", {})
        events.append({
            "id": item.get("id", ""),
            "summary": item.get("summary", "(No title)"),
            "start": start_raw.get("dateTime") or start_raw.get("date", ""),
            "end": end_raw.get("dateTime") or end_raw.get("date", ""),
            "location": item.get("location", ""),
            "attendees": [a.get("email", "") for a in item.get("attendees", [])],
            "description": item.get("description", ""),
        })
    return events


def create_event(
    summary: str,
    start: datetime,
    end: datetime,
    attendees: Optional[list[str]] = None,
    description: Optional[str] = None,
    location: Optional[str] = None,
    calendar_id: str = "primary",
) -> dict:
    """Create a calendar event. Returns the created event dict."""
    service = get_calendar_service()
    # Ensure timezone-aware datetimes
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    body = {
        "summary": summary,
        "start": {"dateTime": _to_rfc3339(start), "timeZone": "UTC"},
        "end": {"dateTime": _to_rfc3339(end), "timeZone": "UTC"},
    }
    if description:
        body["description"] = description
    if location:
        body["location"] = location
    if attendees:
        body["attendees"] = [{"email": e.strip()} for e in attendees if e.strip()]

    event = service.events().insert(calendarId=calendar_id, body=body).execute()
    return event
