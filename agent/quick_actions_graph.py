"""Dedicated LangGraph for intelligent quick-action suggestions.

Runs multiple analysis nodes to produce high-quality,
context-aware actions for a single email:

1. gather_context    — Pull memory, sender history, existing todos.
2. reply_analysis    — Should the user reply? What are ALL reasonable reply options?
3. todo_analysis     — Extract actionable tasks. Strict dedup against existing todos.
4. meeting_analysis  — Does the email mention a meeting or scheduling need?
5. merge_actions     — Deduplicate, validate, and rank the final suggestions.

Key design principles:
- Reply options always come in pairs (accept/decline, confirm/reject)
- Reply descriptions are detailed and specific
- Todos must be concrete with deadlines when available
- Meeting detection is generous — better to suggest than miss
- No "Archive" quick-action — there is already a permanent Archive button in the UI.
  Instead, when no actions are warranted, we return a status message.
"""
import json
import logging
import re
from typing import TypedDict

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

from agent.llm import get_llm
from agent.email_memory import (
    build_memory_context,
    get_sender_history,
    get_todo_items,
)

logger = logging.getLogger(__name__)


class QuickActionState(TypedDict, total=False):
    """State flowing through the quick-actions graph."""
    email: dict
    memory_context: str
    sender_history: str
    existing_todos: list[str]
    reply_options: list[str]
    todo_options: list[str]
    meeting_options: list[str]
    no_action_message: str          # replaces archive_option
    final_options: list[str]


# ─── Helper: extract sender address ────────────────────────────────────────

def _sender_addr(email: dict) -> str:
    sender = email.get("sender", "")
    if "<" in sender and ">" in sender:
        return sender.split("<")[1].split(">")[0].strip().lower()
    return sender.strip().lower()


# ─── Node 0: Gather context ────────────────────────────────────────────────

def gather_context_node(state: QuickActionState) -> dict:
    """Pull memory context, sender history, and existing todos."""
    email = state.get("email") or {}
    memory_ctx = ""
    try:
        memory_ctx = build_memory_context(email, max_linked=5)[:600]
    except Exception:
        pass

    sender_history = ""
    try:
        addr = _sender_addr(email)
        history = get_sender_history(addr, exclude_id=email.get("id", ""), limit=3)
        if history:
            lines = [f"- {h.get('subject', '')} ({h.get('category', '')}) [{h.get('date', '')}]" for h in history]
            sender_history = "Previous emails from this sender:\n" + "\n".join(lines)
    except Exception:
        pass

    existing_todos: list[str] = []
    try:
        todos = get_todo_items()
        existing_todos = [t["task"].lower().strip() for t in todos]
    except Exception:
        pass

    return {
        "memory_context": memory_ctx,
        "sender_history": sender_history,
        "existing_todos": existing_todos,
    }


# ─── Node 1: Reply analysis ────────────────────────────────────────────────

def reply_analysis_node(state: QuickActionState) -> dict:
    """Determine if a reply is warranted and suggest ALL reasonable reply options.

    Key improvements:
    - Always suggests PAIRS of opposing options (accept & decline, confirm & deny)
    - Reply descriptions are detailed and specific
    - Only skips replies for truly automated / noreply senders
    """
    email = state.get("email") or {}
    subject = (email.get("subject") or "")[:150]
    body = (email.get("clean_body") or email.get("body") or email.get("snippet") or "")[:1500]
    sender = email.get("sender", "")
    category = email.get("category", "informational")
    thread_ctx = (email.get("thread_context") or "")[:400]
    memory_ctx = state.get("memory_context", "")
    sender_hist = state.get("sender_history", "")

    # Only skip truly automated senders — do NOT skip based on category alone
    sender_lower = sender.lower()
    if any(s in sender_lower for s in ["noreply", "no-reply", "mailer-daemon", "donotreply"]):
        return {"reply_options": []}

    llm = get_llm(task="decide")
    parser = StrOutputParser()

    system = """You analyze emails and suggest SPECIFIC reply options the user might want to send.

CRITICAL RULES:
1. ALWAYS suggest BOTH sides of any decision. If suggesting "Accept the meeting invitation for March 5th", ALSO suggest "Decline the meeting invitation for March 5th". If suggesting "Confirm attendance at the workshop", ALSO suggest "Decline — unable to attend the workshop".
2. Each reply description MUST be detailed and specific — include names, dates, topics from the email. BAD: "Confirm attendance". GOOD: "Confirm attendance at the NYU Career Fair on March 10th".
3. Only suggest replies that are CONVERSATIONAL messages TO the sender. NOT personal notes.
4. Do NOT suggest replies for: automated notifications, receipts, system alerts, mass newsletters, noreply senders.
5. DO suggest replies for: invitations, requests, questions, proposals, meeting requests, RSVPs, confirmations, event registrations, collaborative discussions.
6. Maximum 4 reply suggestions (usually 2 pairs of opposing options).
7. Keep descriptions as email-reply actions: "Accept...", "Decline...", "Ask about...", "Confirm...", "Request...".

Output a JSON array of reply description strings. Empty array [] if truly no reply warranted.

EXAMPLES:
Email about a hackathon registration deadline → ["Confirm registration for the NYUAD Hackathon by the Feb 27th deadline", "Decline — unable to participate in the hackathon"]
Email inviting to a meeting → ["Accept the meeting invitation for Thursday 3pm", "Decline the meeting — suggest alternative time"]
Email asking for a project update → ["Provide the requested project status update", "Ask for more time to prepare the update"]
Email about shelter-in-place alert → []
Newsletter → []"""

    prompt = f"""From: {sender}
Subject: {subject}
Category: {category}
Content: {body}
Thread: {thread_ctx}
{memory_ctx}
{sender_hist}

What replies should the user consider? Include BOTH accepting and declining options where applicable. Be specific with names, dates, and details from the email. (JSON array only):"""

    try:
        raw = (llm | parser).invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
        text = _extract_json_array(raw)
        options = json.loads(text)
        if isinstance(options, list):
            results = []
            for o in options:
                desc = str(o).strip()
                if desc:
                    results.append(f"Reply: {desc}")
            return {"reply_options": results[:4]}
    except Exception as e:
        logger.debug("Reply analysis error: %s", e)
    return {"reply_options": []}


# ─── Node 2: Todo analysis ─────────────────────────────────────────────────

def todo_analysis_node(state: QuickActionState) -> dict:
    """Extract concrete, actionable tasks with deadlines. Strict dedup.

    Key improvements:
    - Tasks must include deadlines / specifics from the email
    - Stricter duplicate detection (both fuzzy + existing list in prompt)
    - Avoids vague actions like "review email" or "check document"
    """
    email = state.get("email") or {}
    subject = (email.get("subject") or "")[:150]
    body = (email.get("clean_body") or email.get("body") or email.get("snippet") or "")[:1500]
    sender = email.get("sender", "")
    existing_todos = state.get("existing_todos", [])

    llm = get_llm(task="decide")
    parser = StrOutputParser()

    existing_block = ""
    if existing_todos:
        existing_block = "\n\nEXISTING TODO ITEMS (do NOT suggest anything similar to these — even if worded differently):\n" + "\n".join(f"- {t}" for t in existing_todos[:15])

    system = f"""You extract actionable tasks from emails and return them as a JSON array.

CRITICAL RULES:
1. Each task MUST be a concrete, specific action with details from the email (names, deadlines, links, amounts).
2. Include deadlines when mentioned. BAD: "Submit form". GOOD: "Submit hackathon speaker form by Feb 27th 11:59 PM EST".
3. Do NOT suggest vague tasks: "check the email", "review this", "look into it", "follow up".
4. Do NOT suggest tasks that duplicate or overlap with existing todos, even if worded differently.
5. Only suggest tasks that require the USER's action — not things the sender will do.
6. Maximum 3 tasks.
7. Do NOT create todo items for newsletters or automated notifications unless they contain a specific deadline or required action.{existing_block}

Output a JSON array of task descriptions. Empty array [] if no concrete tasks.

EXAMPLES:
Email about hackathon form deadline → ["Submit NYUAD Hackathon Speaker Form by Feb 27th 11:59 PM EST"]
Email about project proposal review → ["Review and provide feedback on the research proposal by Monday"]
Email about shelter-in-place → ["Follow shelter-in-place instructions — stay indoors until further notice"]
Newsletter about new tools → []"""

    prompt = f"""From: {sender}
Subject: {subject}
Content: {body}

Extract concrete actionable tasks with specific deadlines and details (JSON array only):"""

    try:
        raw = (llm | parser).invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
        text = _extract_json_array(raw)
        options = json.loads(text)
        if isinstance(options, list):
            results = []
            for o in options:
                task = str(o).strip()
                if not task:
                    continue
                task_lower = task.lower()
                # Strict dedup: word-overlap check against ALL existing todos
                is_dup = any(_fuzzy_match(task_lower, existing) for existing in existing_todos)
                if not is_dup:
                    results.append(f"Todo: {task}")
            return {"todo_options": results[:3]}
    except Exception as e:
        logger.debug("Todo analysis error: %s", e)
    return {"todo_options": []}


# ─── Node 3: Meeting analysis ──────────────────────────────────────────────

def meeting_analysis_node(state: QuickActionState) -> dict:
    """Detect meetings, calls, or scheduling needs. Generous — better to suggest than miss.

    Key improvements:
    - More comprehensive signal detection
    - Extracts specific meeting details (who, what, duration estimate)
    - Also detects implicit scheduling needs ("let's discuss", "can we talk")
    - Detects event invitations that should be added to calendar
    """
    email = state.get("email") or {}
    subject = (email.get("subject") or "").lower()
    body = (email.get("clean_body") or email.get("body") or email.get("snippet") or "")[:1200].lower()
    full_text = f"{subject} {body}"

    # Comprehensive meeting signal detection
    meeting_signals = [
        # Explicit meeting words
        "meeting", "call", "zoom", "teams", "google meet", "conference",
        "catch up", "catch-up", "sync", "1-on-1", "one-on-one", "standup",
        # Scheduling intent
        "schedule", "calendar", "availability", "available", "free time",
        "appointment", "book a time", "book time", "set up a time",
        # Social/casual meeting
        "let's meet", "let's chat", "let's discuss", "let's talk",
        "can we meet", "can we talk", "can we discuss", "can we chat",
        "would like to meet", "want to meet", "want to discuss",
        "get together", "drop by", "come by", "stop by",
        # Formal scheduling
        "workshop", "seminar", "webinar", "office hours", "consultation",
        "interview", "demo", "presentation", "briefing", "orientation",
        # Time references in scheduling context
        "this week", "next week", "tomorrow", "this afternoon",
        # Event registration with time
        "rsvp", "register for", "sign up for",
    ]

    has_signal = any(sig in full_text for sig in meeting_signals)

    # Also check for date/time patterns suggesting a specific event
    time_patterns = re.findall(
        r'\d{1,2}:\d{2}\s*(?:am|pm|AM|PM)|'
        r'\d{1,2}\s*(?:am|pm|AM|PM)|'
        r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2}',
        full_text,
    )
    has_event_time = len(time_patterns) >= 1

    if not has_signal and not has_event_time:
        return {"meeting_options": []}

    # Use LLM for nuanced meeting detection
    llm = get_llm(task="decide")
    parser = StrOutputParser()

    system = """You determine if an email contains a meeting, event, or scheduling need.

RULES:
1. If the email mentions a specific event, meeting, call, or appointment — suggest scheduling it.
2. If the email implies a need to meet or discuss something — suggest scheduling.
3. Include specific details: who to meet with, what the meeting is about, and when if mentioned.
4. Do NOT suggest scheduling for: newsletters, system alerts, automated notifications, general announcements without RSVP.
5. DO suggest scheduling for: invitations (even if time is set — user may want to add to calendar), requests to meet, events requiring attendance.

Output a JSON array with at most 1 meeting description. Include the WHO, WHAT, and WHEN if available.
Example: ["30 min meeting with Prof. Smith to discuss research proposal — suggested: this week"]
Example: ["NYUAD Hackathon orientation on March 1st at 2pm — add to calendar"]
Example: []"""

    prompt = f"""From: {email.get('sender', '')}
Subject: {email.get('subject', '')}
Content: {(email.get('clean_body') or email.get('body') or '')[:800]}

Does this require scheduling or adding an event to calendar? Be specific with details. (JSON array only):"""

    try:
        raw = (llm | parser).invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
        text = _extract_json_array(raw)
        options = json.loads(text)
        if isinstance(options, list) and options:
            desc = str(options[0]).strip()
            if desc:
                return {"meeting_options": [f"Schedule: {desc}"]}
    except Exception as e:
        logger.debug("Meeting analysis error: %s", e)
    return {"meeting_options": []}


# ─── Node 4: Merge and rank ─────────────────────────────────────────────────

def merge_actions_node(state: QuickActionState) -> dict:
    """Combine, deduplicate, and rank all suggested actions.

    No archive node — instead we set `no_action_message` when nothing is suggested.
    Priority: Reply > Schedule > Todo. Capped at 5 total actions.
    """
    final: list[str] = []

    # Priority order: Reply > Schedule > Todo
    for opt in state.get("reply_options", []):
        if opt and opt not in final:
            final.append(opt)

    for opt in state.get("meeting_options", []):
        if opt and opt not in final:
            final.append(opt)

    for opt in state.get("todo_options", []):
        if opt and opt not in final:
            final.append(opt)

    # Cap at 5 total actions
    final = final[:5]

    # If nothing was suggested, set a friendly no-action message
    no_action = ""
    if not final:
        no_action = "No action needed"

    return {"final_options": final, "no_action_message": no_action}


# ─── Helpers ────────────────────────────────────────────────────────────────

def _extract_json_array(raw: str) -> str:
    """Robustly extract a JSON array from LLM output."""
    text = (raw or "").strip()
    text = re.sub(r"^```[a-zA-Z0-9_]*\s*", "", text)
    text = re.sub(r"```\s*$", "", text).strip()
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return "[]"


def _fuzzy_match(a: str, b: str) -> bool:
    """Check if two task strings are semantically similar enough to be duplicates."""
    words_a = set(re.findall(r"[a-z0-9]+", a))
    words_b = set(re.findall(r"[a-z0-9]+", b))
    stop = {"the", "a", "an", "to", "and", "or", "of", "in", "on", "for",
            "is", "it", "this", "that", "by", "at", "be", "with", "from"}
    words_a -= stop
    words_b -= stop
    if not words_a or not words_b:
        return False
    overlap = len(words_a & words_b) / min(len(words_a), len(words_b))
    return overlap >= 0.5


# ─── Graph construction ────────────────────────────────────────────────────

def _build_quick_actions_graph():
    """Build the quick-actions analysis graph (no archive node)."""
    graph = StateGraph(QuickActionState)

    graph.add_node("gather_context", gather_context_node)
    graph.add_node("reply_analysis", reply_analysis_node)
    graph.add_node("todo_analysis", todo_analysis_node)
    graph.add_node("meeting_analysis", meeting_analysis_node)
    graph.add_node("merge_actions", merge_actions_node)

    graph.set_entry_point("gather_context")
    graph.add_edge("gather_context", "reply_analysis")
    graph.add_edge("reply_analysis", "todo_analysis")
    graph.add_edge("todo_analysis", "meeting_analysis")
    graph.add_edge("meeting_analysis", "merge_actions")
    graph.add_edge("merge_actions", END)

    return graph.compile()


_QUICK_ACTIONS_GRAPH = _build_quick_actions_graph()


def suggest_quick_actions(email: dict) -> list[str]:
    """Run the quick-actions graph for a single email.

    Returns a list of action strings like:
      ["Reply: Accept the invitation", "Todo: Submit form by Friday", "Schedule: meeting with Alice"]
    """
    try:
        result = _QUICK_ACTIONS_GRAPH.invoke(
            {"email": email},
            config={"configurable": {}},
        )
        return result.get("final_options", [])
    except Exception as e:
        logger.warning("Quick actions graph error: %s", e)
        return []


def suggest_quick_actions_full(email: dict) -> dict:
    """Run the quick-actions graph and return the full result dict.

    Returns:
        {"final_options": [...], "no_action_message": "No action needed" | ""}
    """
    try:
        result = _QUICK_ACTIONS_GRAPH.invoke(
            {"email": email},
            config={"configurable": {}},
        )
        return {
            "final_options": result.get("final_options", []),
            "no_action_message": result.get("no_action_message", ""),
        }
    except Exception as e:
        logger.warning("Quick actions graph error: %s", e)
        return {"final_options": [], "no_action_message": ""}
