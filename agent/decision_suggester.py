"""Suggest contextual quick actions based on email content.

Uses the quick_actions_graph for the main flow. This module provides
a simpler fallback for the pipeline's suggest_decision_node.
"""
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
import json
import logging
import re

from agent.llm import get_llm
from agent.email_memory import build_memory_context, get_sender_history

logger = logging.getLogger(__name__)


def suggest_decision(email: dict) -> dict:
    """Suggest contextual quick actions for an email using LLM.

    Returns {"decision_options": [{"type": "reply"|"todo", "label": "...", "context": "...", ...}]}.
    """
    llm = get_llm(task="decide")
    parser = StrOutputParser()

    system = """You are an email assistant. Analyze the email and suggest short action labels.

Each action is a JSON object with:
- "type": "reply" or "todo"
- "label": SHORT action text (2-6 words). This is what the user sees on the button.
- "context": Brief hidden context for the AI drafter (who, what, when — 1 sentence).
- "has_meeting": true if this involves a meeting/event/scheduling component.
- "meeting_action": "accept" | "decline" | "reschedule" | null. Only when has_meeting is true.

CRITICAL RULES:
- Reply actions are what you'd SAY to the sender. Labels: "Agree to meeting", "Ask for details", "Confirm attendance".
- Todo actions are tasks for the user. Labels: "Submit form by Friday", "Review proposal".
- For meeting/event emails: suggest accept, decline, and optionally reschedule options (all type="reply" with has_meeting=true).
- Do NOT suggest actions for newsletters, automated notifications, marketing emails.
- Labels must be SHORT (2-6 words). Details go in "context".
- Suggest 1-4 actions total.

Output ONLY a JSON array of objects. No other text."""

    subject = email.get("subject", "")[:120]
    body_text = (email.get("clean_body") or email.get("body") or email.get("snippet") or "")[:1200]
    thread_ctx = (email.get("thread_context") or "")[:350]
    related_ctx = (email.get("related_context") or "")[:500]
    category = email.get("category", "normal")
    needs_action = bool(email.get("needs_action"))

    # Pull rich memory context
    memory_ctx = ""
    try:
        memory_ctx = build_memory_context(email, max_linked=5)[:600]
    except Exception:
        pass

    # Pull sender history
    sender_history = ""
    try:
        from tools.gmail_tools import extract_email_address
        sender_addr = extract_email_address(email.get("sender", ""))
        history = get_sender_history(sender_addr, exclude_id=email.get("id", ""), limit=3)
        if history:
            lines = []
            for h in history:
                lines.append(f"- {h.get('subject', '')} ({h.get('category', '')}) [{h.get('date', '')}]")
            sender_history = "Previous emails from this sender:\n" + "\n".join(lines)
    except Exception:
        pass

    prompt = f"""Email:
Subject: {subject}
Category: {category}
Needs action: {needs_action}
Content: {body_text}
Thread context:
{thread_ctx}

Related emails:
{related_ctx}

{(email.get('enriched_context') or '')[:500]}

{memory_ctx}

{sender_history}

Suggest quick actions (JSON array of action objects):"""

    try:
        chain = llm | parser
        raw = chain.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])

        text = (raw or "").strip()
        text = re.sub(r"^```[a-zA-Z0-9_]*\s*", "", text)
        text = re.sub(r"```\s*$", "", text).strip()
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]

        options = json.loads(text)

        if isinstance(options, list) and options:
            cleaned: list[dict] = []
            seen = set()
            for o in options:
                if isinstance(o, dict):
                    label = str(o.get("label", "")).strip()
                    if not label:
                        continue
                    key = label.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    action_type = o.get("type", "reply")
                    if action_type not in ("reply", "todo"):
                        action_type = "reply"
                    action = {
                        "type": action_type,
                        "label": label,
                        "context": str(o.get("context", "")).strip(),
                        "has_meeting": bool(o.get("has_meeting", False)),
                        "meeting_action": o.get("meeting_action") if o.get("has_meeting") else None,
                    }
                    if action["meeting_action"] not in (None, "accept", "decline", "reschedule"):
                        action["meeting_action"] = None
                    cleaned.append(action)
                elif isinstance(o, str):
                    # Backward compat: handle plain string from old format
                    s = o.strip()
                    if not s:
                        continue
                    key = s.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    # Parse old prefix format
                    if s.startswith("Reply:"):
                        cleaned.append({"type": "reply", "label": s[6:].strip(), "context": s[6:].strip(), "has_meeting": False, "meeting_action": None})
                    elif s.startswith("Todo:"):
                        cleaned.append({"type": "todo", "label": s[5:].strip(), "context": s[5:].strip()})
                    elif s.startswith("Schedule:"):
                        cleaned.append({"type": "reply", "label": s[9:].strip(), "context": s[9:].strip(), "has_meeting": True, "meeting_action": "accept"})
                    else:
                        cleaned.append({"type": "reply", "label": s, "context": s, "has_meeting": False, "meeting_action": None})
            if cleaned:
                return {"decision_options": cleaned[:5]}
    except json.JSONDecodeError as e:
        logger.warning("JSON parse error in decision_suggester: %s (raw: %s)", e, (raw or "")[:500])
        return {"decision_options": []}
    except Exception as e:
        logger.warning("Error in decision_suggester: %s", e)
        return {"decision_options": []}

    return {"decision_options": []}
