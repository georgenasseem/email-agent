"""Dedicated LangGraph for intelligent quick-action suggestions.

Architecture:
  Quick Actions answer "WHAT can the user do?" — they surface every meaningful choice.
  Drafting answers "HOW do I execute the chosen action?" — it composes content.

Graph:
  gather_context → unified_analysis → validate_and_rank → END

Key design principles:
  - ONE unified LLM call sees the entire email and outputs all action types
    simultaneously — preventing cross-type redundancy (no reply that is secretly
    a personal task, no task that belongs in replies).
  - Three distinct action types with clear semantics:
      "reply"   — A message the user sends back to the sender (orange buttons).
      "todo"    — A personal task the user must do on their own (purple buttons).
      "meeting" — Accept / decline / propose new time for a meeting (blue section).
  - Meeting actions are ALWAYS the trio (accept + decline + propose new time)
    and are rendered as a SEPARATE section in the UI.
  - Conversational emails (casual questions, personal messages) ALWAYS receive at
    least one reply option regardless of their category label.
  - No category gating — category describes the email type, not whether it
    warrants a response. A question is a question whether labelled "important"
    or "informational".
  - Labels are SHORT (2-7 words), specific, and actionable — never generic like
    "Reply to email" or "Respond".
  - Context is RICH (1-2 sentences): includes sender names, dates, topic details —
    everything the drafter needs to compose an excellent response without asking
    the user for clarification.
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
    raw_analysis: dict              # raw output from unified_analysis_node
    final_options: list[dict]       # validated, ranked, flat list
    no_action_message: str


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
        memory_ctx = build_memory_context(email, max_linked=3)[:400]
    except Exception:
        pass

    sender_history = ""
    try:
        addr = _sender_addr(email)
        history = get_sender_history(addr, exclude_id=email.get("id", ""), limit=2)
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


# ─── Node 1: Unified analysis (single LLM call) ────────────────────────────

_UNIFIED_SYSTEM = (
    "You are an intelligent email assistant. Analyze the email below and identify "
    "EVERY meaningful action the recipient might want to take.\n\n"
    "Output a SINGLE JSON OBJECT with exactly three keys: \"replies\", \"todos\", \"meeting\".\n\n"
    "--- REPLIES ---\n"
    "Array of: {\"label\": \"...\", \"context\": \"...\"}\n"
    "label: 2-6 word button text. SHORT and SPECIFIC — never generic.\n"
    "context: 1-2 sentences for the AI drafter: sender name, exact topic, key details.\n\n"
    "WHEN to include replies — be GENEROUS, err on the side of suggesting:\n"
    "  YES: Email asks the user ANY question (even casual) → always reply\n"
    "  YES: Personal / conversational email from a real person → suggest reply\n"
    "  YES: Email requests help, info, files, or confirmation → offer options\n"
    "  YES: Invitation (social, event, project) → suggest accept / decline\n"
    "  YES: Email shares news where sender would value a reaction → brief reply\n"
    "  NO:  Automated system (noreply, mailer-daemon, no-reply) → skip\n"
    "  NO:  Pure newsletter/marketing with zero personal content → skip\n"
    "  NO:  Receipt/order confirmation → skip unless it needs personal action\n\n"
    "QUALITY — be SPECIFIC not generic:\n"
    "  'Do you still have your guitars?' → [\"Yes, we still do\", \"We sold them\"] (real answers)\n"
    "  'Can you send the report?' → [\"Send the report\", \"Need more time first\"]\n"
    "  'Thanks for helping!' → [\"You're welcome\"] (warmth deserves a reply)\n"
    "  NEVER suggest: 'Reply to email', 'Respond', 'Answer', 'Acknowledge email'\n"
    "  Maximum 3 reply options.\n\n"
    "--- TODOS ---\n"
    "Array of: {\"label\": \"...\", \"context\": \"...\"}\n"
    "label: 3-8 word task. ALWAYS include deadline/date when the email mentions one.\n"
    "context: 1-2 sentences with full specifics: deadlines, links, names, document titles.\n\n"
    "WHEN to include todos:\n"
    "  YES: Email has a DEADLINE → capture it ('Submit form by Feb 27')\n"
    "  YES: User is asked to prepare/review/write/complete something independently\n"
    "  YES: Document to review, form to fill, link to visit, registration needed\n"
    "  NO:  'Reply to this email' — goes in replies, not todos\n"
    "  NO:  Vague non-actionable things ('look into it', 'check the email')\n"
    "  NO:  Anything already in existing todos (listed in prompt if any)\n"
    "  Maximum 3 todo items.\n\n"
    "--- MEETING ---\n"
    "Single object: {\"is_meeting\": true/false, \"context\": \"...\"}\n"
    "is_meeting: true ONLY when a FUTURE meeting/event/scheduling discussion is the CORE topic.\n"
    "context: who, what, when (if known), where/link — used to create a calendar event.\n\n"
    "  YES: Meeting invitation ('Can we meet Tuesday at 3pm?')\n"
    "  YES: Meeting reminder ('Don't forget our call tomorrow at 2pm')\n"
    "  YES: Scheduling discussion ('Are you free Friday for a catch-up?')\n"
    "  YES: Event invitation (workshop, conference, social gathering)\n"
    "  NO:  Past meeting referenced as background ('as we discussed last week')\n"
    "  NO:  Meeting mentioned incidentally, not the main topic\n\n"
    "OUTPUT FORMAT — your entire response must be valid JSON, no markdown, no extra text:\n"
    "{\"replies\": [{\"label\": \"...\", \"context\": \"...\"}, ...], "
    "\"todos\": [{\"label\": \"...\", \"context\": \"...\"}, ...], "
    "\"meeting\": {\"is_meeting\": false, \"context\": \"\"}}"
)


def unified_analysis_node(state: QuickActionState) -> dict:
    """Single LLM call that identifies all reply, todo, and meeting actions.

    One call sees the full email at once — prevents cross-type redundancy
    and uses half the rate-limit budget compared to two parallel nodes.
    """
    email = state.get("email") or {}
    subject = (email.get("subject") or "")[:150]
    body = (email.get("clean_body") or email.get("body") or email.get("snippet") or "")[:1200]
    sender = email.get("sender", "")
    thread_ctx = (email.get("thread_context") or "")[:250]
    memory_ctx = state.get("memory_context", "")
    sender_hist = state.get("sender_history", "")
    existing_todos = state.get("existing_todos", [])

    # Skip clearly automated senders
    sender_lower = sender.lower()
    if any(s in sender_lower for s in [
        "noreply", "no-reply", "mailer-daemon", "donotreply",
        "notifications@", "alerts@", "automated@",
    ]):
        return {"raw_analysis": {"replies": [], "todos": [], "meeting": {"is_meeting": False, "context": ""}}}

    existing_block = ""
    if existing_todos:
        existing_block = "\nEXISTING TODOS (do NOT suggest anything similar):\n" + "\n".join(
            f"- {t}" for t in existing_todos[:12]
        )

    prompt = (
        f"From: {sender}\n"
        f"Subject: {subject}\n"
        f"Body: {body}\n"
        f"Thread context: {thread_ctx}\n"
        f"{memory_ctx}\n"
        f"{sender_hist}\n"
        f"{existing_block}\n\n"
        "Analyze this email and output the JSON object with replies, todos, and meeting info:"
    )

    llm = get_llm(task="decide")
    parser = StrOutputParser()

    try:
        raw = (llm | parser).invoke([SystemMessage(content=_UNIFIED_SYSTEM), HumanMessage(content=prompt)])
        text = _extract_json_object(raw)
        analysis = json.loads(text)
        if isinstance(analysis, dict):
            return {"raw_analysis": analysis}
    except Exception as e:
        logger.debug("Unified analysis error: %s", e)

    return {"raw_analysis": {"replies": [], "todos": [], "meeting": {"is_meeting": False, "context": ""}}}


# ─── Node 2: Validate and rank ─────────────────────────────────────────────

# First-word task verbs — labels starting with these are tasks, not replies
_TASK_VERBS = {
    "prepare", "submit", "complete", "finish", "review", "write", "create",
    "study", "read", "update", "fix", "build", "design", "implement",
    "organize", "clean", "plan", "draft", "edit", "research", "register",
    "sign", "apply", "upload", "download", "install", "configure", "buy",
    "order", "book", "attend", "fill", "check",
}


def _is_task_not_reply(label: str) -> bool:
    first_word = label.lower().split()[0] if label.split() else ""
    return first_word in _TASK_VERBS


def validate_and_rank_node(state: QuickActionState) -> dict:
    """Validate LLM output, generate meeting trio if needed, deduplicate, cap totals.

    Steps:
    1. Parse replies → type="reply" actions  (reject task-like labels that snuck in)
    2. Parse todos   → type="todo"  actions  (dedup against existing)
    3. If meeting.is_meeting → generate the accept/decline/reschedule trio as type="meeting"
    4. Deduplicate across all types by fuzzy label match
    5. Output flat final_options list
    """
    raw = state.get("raw_analysis") or {}
    existing_todos = state.get("existing_todos", [])

    # ── Parse replies ──────────────────────────────────────────────
    reply_actions: list[dict] = []
    for item in (raw.get("replies") or []):
        if isinstance(item, dict):
            label = str(item.get("label", "")).strip()
            context = str(item.get("context", "")).strip()
        elif isinstance(item, str):
            label = item.strip()
            context = label
        else:
            continue
        if not label or len(label) < 3:
            continue
        if _is_task_not_reply(label):
            continue  # task mistakenly placed in replies
        reply_actions.append({
            "type": "reply",
            "label": label[:80],
            "context": context or label,
        })

    # ── Parse todos ────────────────────────────────────────────────
    todo_actions: list[dict] = []
    for item in (raw.get("todos") or []):
        if isinstance(item, dict):
            label = str(item.get("label", "")).strip()
            context = str(item.get("context", "")).strip()
        elif isinstance(item, str):
            label = item.strip()
            context = label
        else:
            continue
        if not label or len(label) < 3:
            continue
        if any(_fuzzy_match(label.lower(), ex) for ex in existing_todos):
            continue  # already tracked
        todo_actions.append({
            "type": "todo",
            "label": label[:80],
            "context": context or label,
        })

    # ── Parse meeting ──────────────────────────────────────────────
    meeting_actions: list[dict] = []
    meeting_info = raw.get("meeting") or {}
    if isinstance(meeting_info, dict) and meeting_info.get("is_meeting"):
        ctx = str(meeting_info.get("context", "")).strip()
        meeting_actions = [
            {"type": "meeting", "label": "Accept meeting",        "context": f"Accept the meeting. {ctx}".strip(), "meeting_action": "accept"},
            {"type": "meeting", "label": "Decline meeting",       "context": f"Decline the meeting. {ctx}".strip(), "meeting_action": "decline"},
            {"type": "meeting", "label": "Suggest different time", "context": f"Propose a new time. {ctx}".strip(), "meeting_action": "reschedule"},
        ]

    # ── Deduplicate reply + todo (meetings are always kept whole) ──
    final: list[dict] = []
    seen: set[str] = set()

    for opt in reply_actions[:3]:
        key = opt["label"].lower()
        if not any(_fuzzy_match(key, s) for s in seen):
            seen.add(key)
            final.append(opt)

    for opt in todo_actions[:3]:
        key = opt["label"].lower()
        if not any(_fuzzy_match(key, s) for s in seen):
            seen.add(key)
            final.append(opt)

    final.extend(meeting_actions)

    no_action = "" if final else "No action needed"
    return {"final_options": final, "no_action_message": no_action}


# ─── Helpers ────────────────────────────────────────────────────────────────

def _extract_json_object(raw: str) -> str:
    """Robustly extract a JSON object from LLM output."""
    text = (raw or "").strip()
    text = re.sub(r"^```[a-zA-Z0-9_]*\s*", "", text)
    text = re.sub(r"```\s*$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return "{}"


def _fuzzy_match(a: str, b: str) -> bool:
    """Return True when two action labels are close enough to be duplicates."""
    words_a = set(re.findall(r"[a-z0-9]+", a))
    words_b = set(re.findall(r"[a-z0-9]+", b))
    stop = {"the", "a", "an", "to", "and", "or", "of", "in", "on", "for",
            "is", "it", "this", "that", "by", "at", "be", "with", "from"}
    words_a -= stop
    words_b -= stop
    if not words_a or not words_b:
        return False
    overlap = len(words_a & words_b) / min(len(words_a), len(words_b))
    return overlap >= 0.6


# ─── Graph construction ────────────────────────────────────────────────────

def _build_quick_actions_graph():
    """Build the quick-actions analysis graph.

    Sequential: gather_context -> unified_analysis -> validate_and_rank -> END
    One LLM call handles all action types in one shot.
    """
    graph = StateGraph(QuickActionState)

    graph.add_node("gather_context",    gather_context_node)
    graph.add_node("unified_analysis",  unified_analysis_node)
    graph.add_node("validate_and_rank", validate_and_rank_node)

    graph.set_entry_point("gather_context")
    graph.add_edge("gather_context",    "unified_analysis")
    graph.add_edge("unified_analysis",  "validate_and_rank")
    graph.add_edge("validate_and_rank", END)

    return graph.compile()


_QUICK_ACTIONS_GRAPH = _build_quick_actions_graph()


def suggest_quick_actions(email: dict) -> list[dict]:
    """Run the quick-actions graph for a single email.

    Returns a flat list of action dicts:
      [{"type": "reply",   "label": "Say yes, we still do", "context": "..."},
       {"type": "todo",    "label": "Submit form by Friday", "context": "..."},
       {"type": "meeting", "label": "Accept meeting", "context": "...", "meeting_action": "accept"},
       ...]
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


# ─── Node 1: Reply analysis (includes meeting detection) ───────────────────

