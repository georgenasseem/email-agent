"""Extract meeting/scheduling intent from emails using LLM."""
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agent.llm import get_llm


def extract_meeting_intent(email: dict) -> dict[str, Any]:
    """Extract meeting intent from an email. Returns dict with has_meeting_intent, title, duration_minutes, etc.

    Now also extracts specific_time (ISO datetime string) when the email
    contains a concrete meeting time. This lets the scheduling system
    auto-create the event without showing a time picker.
    """
    llm = get_llm(task="meeting_extract")
    subject = (email.get("subject") or "")[:120]
    body = (email.get("body") or email.get("snippet") or "")[:600]
    thread_ctx = (email.get("thread_context") or "")[:300]
    sender = email.get("sender", "")

    system = """Extract meeting/scheduling intent from the email. Output ONLY valid JSON.

If the email is about scheduling a meeting, appointment, or call, output:
{"has_meeting_intent": true, "title": "Meeting with X", "duration_minutes": 30, "attendees": ["email@example.com"], "notes": "optional notes", "specific_time": "ISO datetime or empty string"}

If NOT about scheduling, output:
{"has_meeting_intent": false}

Rules:
- title: short summary (e.g. "Meeting with John", "Study Away advising")
- duration_minutes: 15, 30, 45, or 60
- attendees: array of email addresses mentioned, include sender if relevant
- notes: brief context if useful
- specific_time: If the email mentions a CONCRETE date and time for the meeting (e.g. "Thursday at 3pm", "March 10 at 10:00 AM"), convert it to ISO 8601 format. If no specific time is given or it is vague ("sometime next week", "let me know when"), set to empty string "".
- No other fields. No markdown. Only JSON."""

    prompt = f"""Today's date: {datetime.now().strftime('%A, %B %d, %Y')}

Email:
From: {sender}
Subject: {subject}
Body: {body}
Thread: {thread_ctx}

Extract meeting intent (JSON only):"""

    try:
        raw = llm.invoke([SystemMessage(content=system), HumanMessage(content=prompt)], max_tokens=256)
        text = (raw or "").strip()
        text = re.sub(r"^```[a-zA-Z0-9_]*\s*", "", text)
        text = re.sub(r"```\s*$", "", text).strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            text = text[start:end]
        data = json.loads(text)
        if not isinstance(data, dict):
            return {"has_meeting_intent": False}
        if not data.get("has_meeting_intent"):
            return {"has_meeting_intent": False}

        # Parse specific_time if provided
        specific_time = ""
        raw_time = data.get("specific_time", "")
        if raw_time and raw_time.strip():
            specific_time = _normalize_datetime(raw_time.strip())

        return {
            "has_meeting_intent": True,
            "title": (data.get("title") or subject or "Meeting")[:100],
            "duration_minutes": min(120, max(15, int(data.get("duration_minutes", 30)))),
            "attendees": [a for a in data.get("attendees", []) if isinstance(a, str) and "@" in a],
            "notes": (data.get("notes") or "")[:200],
            "specific_time": specific_time,
        }
    except (json.JSONDecodeError, ValueError, TypeError):
        return {"has_meeting_intent": False}


def _normalize_datetime(raw: str) -> str:
    """Try to normalize a datetime string into ISO 8601 format.

    Returns ISO string on success, empty string on failure.
    """
    if not raw:
        return ""
    # Already ISO format?
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return dt.isoformat()
    except (ValueError, TypeError):
        pass
    # Try common formats
    for fmt in (
        "%B %d, %Y at %I:%M %p",
        "%B %d, %Y at %H:%M",
        "%b %d, %Y at %I:%M %p",
        "%Y-%m-%d %H:%M",
        "%m/%d/%Y %I:%M %p",
        "%B %d at %I:%M %p",
        "%B %d at %H:%M",
        "%b %d at %I:%M %p",
    ):
        try:
            dt = datetime.strptime(raw, fmt)
            # If year is 1900 (missing), use current year
            if dt.year == 1900:
                dt = dt.replace(year=datetime.now().year)
            return dt.isoformat()
        except ValueError:
            continue
    return ""
