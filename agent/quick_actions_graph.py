"""Dedicated LangGraph for intelligent quick-action suggestions.

Runs multiple analysis nodes in sequence to produce high-quality,
context-aware actions for a single email:

1. reply_analysis   — Should the user reply? What are the reply options?
2. todo_analysis     — Extract actionable tasks. Check for duplicate todos.
3. meeting_analysis  — Does the email mention a meeting or scheduling need?
4. archive_analysis  — Is this email purely informational / no action needed?
5. merge_actions     — Deduplicate and rank the final suggestions.
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
    archive_option: str
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
    """Determine if a reply is warranted and what reply options exist."""
    email = state.get("email") or {}
    subject = (email.get("subject") or "")[:120]
    body = (email.get("clean_body") or email.get("body") or email.get("snippet") or "")[:1000]
    sender = email.get("sender", "")
    category = email.get("category", "informational")
    thread_ctx = (email.get("thread_context") or "")[:300]
    memory_ctx = state.get("memory_context", "")
    sender_hist = state.get("sender_history", "")

    # Quick heuristics: newsletters/no-reply don't need replies
    sender_lower = sender.lower()
    if any(s in sender_lower for s in ["noreply", "no-reply", "mailer-daemon", "donotreply"]):
        return {"reply_options": []}
    if category in ["newsletter"]:
        return {"reply_options": []}

    llm = get_llm(task="decide")
    parser = StrOutputParser()

    system = """You analyze emails to determine if a reply is needed and suggest specific reply actions.

RULES:
- Only suggest replies that are CONVERSATIONAL and make sense as messages TO the sender.
- NEVER suggest replies that are personal notes (e.g. "Reply: Check the document" — that's a todo).
- Good replies: accept, decline, ask for details, confirm, thank, request changes, provide info.
- If the email is a notification, receipt, automated message, or FYI — suggest NO replies.
- Maximum 2 reply suggestions. Quality over quantity.

Output a JSON array of reply descriptions (without "Reply:" prefix). Empty array [] if no reply needed.
Example: ["Accept the invitation", "Ask for agenda details"]"""

    prompt = f"""From: {sender}
Subject: {subject}
Category: {category}
Content: {body}
Thread: {thread_ctx}
{memory_ctx}
{sender_hist}

What replies should the user consider? (JSON array only):"""

    try:
        raw = (llm | parser).invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
        text = _extract_json_array(raw)
        options = json.loads(text)
        if isinstance(options, list):
            return {"reply_options": [f"Reply: {str(o).strip()}" for o in options if str(o).strip()][:2]}
    except Exception as e:
        logger.debug("Reply analysis error: %s", e)
    return {"reply_options": []}


# ─── Node 2: Todo analysis ─────────────────────────────────────────────────

def todo_analysis_node(state: QuickActionState) -> dict:
    """Extract actionable tasks and check for duplicate existing todos."""
    email = state.get("email") or {}
    subject = (email.get("subject") or "")[:120]
    body = (email.get("clean_body") or email.get("body") or email.get("snippet") or "")[:1000]
    existing_todos = state.get("existing_todos", [])

    llm = get_llm(task="decide")
    parser = StrOutputParser()

    existing_block = ""
    if existing_todos:
        existing_block = "\n\nEXISTING TODO ITEMS (do NOT suggest duplicates):\n" + "\n".join(f"- {t}" for t in existing_todos[:10])

    system = f"""You extract actionable tasks from emails.

RULES:
- Only extract CONCRETE, actionable tasks mentioned or implied in the email.
- Each task must be a clear one-sentence action item.
- Maximum 2 tasks.
- Do NOT suggest tasks that are vague ("check the email", "read this").
- Do NOT duplicate existing todo items — even if worded slightly differently.{existing_block}

Output a JSON array of task descriptions (without "Todo:" prefix). Empty array [] if no tasks.
Example: ["Submit hackathon registration by Friday", "Review attached proposal"]"""

    prompt = f"""Subject: {subject}
Content: {body}

Extract actionable tasks (JSON array only):"""

    try:
        raw = (llm | parser).invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
        text = _extract_json_array(raw)
        options = json.loads(text)
        if isinstance(options, list):
            # Final dedup against existing todos
            results = []
            for o in options:
                task = str(o).strip()
                if not task:
                    continue
                task_lower = task.lower()
                # Check similarity with existing todos
                is_dup = any(_fuzzy_match(task_lower, existing) for existing in existing_todos)
                if not is_dup:
                    results.append(f"Todo: {task}")
            return {"todo_options": results[:2]}
    except Exception as e:
        logger.debug("Todo analysis error: %s", e)
    return {"todo_options": []}


# ─── Node 3: Meeting analysis ──────────────────────────────────────────────

def meeting_analysis_node(state: QuickActionState) -> dict:
    """Check if the email mentions meetings, calls, or scheduling needs."""
    email = state.get("email") or {}
    subject = (email.get("subject") or "").lower()
    body = (email.get("clean_body") or email.get("body") or email.get("snippet") or "")[:800].lower()
    full_text = f"{subject} {body}"

    # Keyword-based fast check before LLM
    meeting_signals = [
        "meeting", "call", "zoom", "teams", "google meet", "conference",
        "catch up", "catch-up", "sync", "1-on-1", "one-on-one",
        "schedule", "calendar", "availability", "available",
        "appointment", "let's meet", "let's chat", "let's discuss",
        "workshop", "seminar", "webinar", "office hours",
    ]
    if not any(sig in full_text for sig in meeting_signals):
        return {"meeting_options": []}

    # Use LLM only if signals detected
    llm = get_llm(task="decide")
    parser = StrOutputParser()

    system = """You determine if an email requires scheduling a meeting.

If yes, describe WHAT to schedule (not when — the system finds times automatically).
If no meeting is needed, return an empty array.

Output a JSON array with at most 1 item (without "Schedule:" prefix). 
Example: ["meeting with Dr. Smith to discuss project"]
Example: []"""

    prompt = f"""Subject: {email.get('subject', '')}
Content: {(email.get('clean_body') or email.get('body') or '')[:600]}

Does this require scheduling? (JSON array only):"""

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


# ─── Node 4: Archive analysis ──────────────────────────────────────────────

def archive_analysis_node(state: QuickActionState) -> dict:
    """Determine if the email is purely FYI and should suggest archiving."""
    email = state.get("email") or {}
    category = email.get("category", "informational")
    sender = (email.get("sender") or "").lower()
    needs_action = email.get("needs_action", False)

    # Direct signals for archive
    noreply = any(s in sender for s in ["noreply", "no-reply", "donotreply", "mailer-daemon"])
    is_newsletter = category in ["newsletter"]

    if noreply or (is_newsletter and not needs_action):
        return {"archive_option": "Archive: No action needed"}

    # If there are no replies, no todos, and no meetings — suggest archive
    reply_opts = state.get("reply_options", [])
    todo_opts = state.get("todo_options", [])
    meeting_opts = state.get("meeting_options", [])

    if not reply_opts and not todo_opts and not meeting_opts and not needs_action:
        return {"archive_option": "Archive: FYI only"}

    return {"archive_option": ""}


# ─── Node 5: Merge and rank ────────────────────────────────────────────────

def merge_actions_node(state: QuickActionState) -> dict:
    """Combine, deduplicate, and rank all suggested actions."""
    final: list[str] = []

    # Priority order: Reply > Schedule > Todo > Archive
    for opt in state.get("reply_options", []):
        if opt and opt not in final:
            final.append(opt)

    for opt in state.get("meeting_options", []):
        if opt and opt not in final:
            final.append(opt)

    for opt in state.get("todo_options", []):
        if opt and opt not in final:
            final.append(opt)

    archive = state.get("archive_option", "")
    if archive:
        final.append(archive)

    # Cap at 4 total actions
    return {"final_options": final[:4]}


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
    # Simple word overlap check
    words_a = set(re.findall(r"[a-z0-9]+", a))
    words_b = set(re.findall(r"[a-z0-9]+", b))
    stop = {"the", "a", "an", "to", "and", "or", "of", "in", "on", "for", "is", "it", "this", "that"}
    words_a -= stop
    words_b -= stop
    if not words_a or not words_b:
        return False
    overlap = len(words_a & words_b) / min(len(words_a), len(words_b))
    return overlap >= 0.6


# ─── Graph construction ────────────────────────────────────────────────────

def _build_quick_actions_graph():
    """Build the quick-actions analysis graph."""
    graph = StateGraph(QuickActionState)

    graph.add_node("gather_context", gather_context_node)
    graph.add_node("reply_analysis", reply_analysis_node)
    graph.add_node("todo_analysis", todo_analysis_node)
    graph.add_node("meeting_analysis", meeting_analysis_node)
    graph.add_node("archive_analysis", archive_analysis_node)
    graph.add_node("merge_actions", merge_actions_node)

    graph.set_entry_point("gather_context")
    graph.add_edge("gather_context", "reply_analysis")
    graph.add_edge("reply_analysis", "todo_analysis")
    graph.add_edge("todo_analysis", "meeting_analysis")
    graph.add_edge("meeting_analysis", "archive_analysis")
    graph.add_edge("archive_analysis", "merge_actions")
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
