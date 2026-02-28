"""Extract meeting/scheduling intent from emails using LLM."""
import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agent.llm import get_llm


def extract_meeting_intent(email: dict) -> dict[str, Any]:
    """Extract meeting intent from an email. Returns dict with has_meeting_intent, title, duration_minutes, etc."""
    llm = get_llm(task="meeting_extract")
    subject = (email.get("subject") or "")[:120]
    body = (email.get("body") or email.get("snippet") or "")[:600]
    thread_ctx = (email.get("thread_context") or "")[:300]
    sender = email.get("sender", "")

    system = """Extract meeting/scheduling intent from the email. Output ONLY valid JSON.

If the email is about scheduling a meeting, appointment, or call, output:
{"has_meeting_intent": true, "title": "Meeting with X", "duration_minutes": 30, "attendees": ["email@example.com"], "notes": "optional notes"}

If NOT about scheduling, output:
{"has_meeting_intent": false}

Rules:
- title: short summary (e.g. "Meeting with John", "Study Away advising")
- duration_minutes: 15, 30, 45, or 60
- attendees: array of email addresses mentioned, include sender if relevant
- notes: brief context if useful
- No other fields. No markdown. Only JSON."""

    prompt = f"""Email:
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
        return {
            "has_meeting_intent": True,
            "title": (data.get("title") or subject or "Meeting")[:100],
            "duration_minutes": min(120, max(15, int(data.get("duration_minutes", 30)))),
            "attendees": [a for a in data.get("attendees", []) if isinstance(a, str) and "@" in a],
            "notes": (data.get("notes") or "")[:200],
        }
    except (json.JSONDecodeError, ValueError, TypeError):
        return {"has_meeting_intent": False}
