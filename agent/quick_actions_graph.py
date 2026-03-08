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
from agent.profile import load_profile

logger = logging.getLogger(__name__)


class QuickActionState(TypedDict, total=False):
    """State flowing through the quick-actions graph."""
    email: dict
    memory_context: str
    sender_history: str
    existing_todos: list[str]
    user_identity: str          # "George Nasseem <gnn9245@nyu.edu>" — injected into LLM prompt
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
    """Pull memory context, sender history, existing todos, and user identity."""
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

    # Load user identity so the LLM knows not to refer to the user as a third party
    user_identity = ""
    try:
        profile = load_profile()
        name = profile.get("display_name") or profile.get("first_name") or ""
        email_addr = profile.get("email") or ""
        if name or email_addr:
            user_identity = f"{name} <{email_addr}>".strip(" <>") if name and email_addr else (name or email_addr)
    except Exception:
        pass

    return {
        "memory_context": memory_ctx,
        "sender_history": sender_history,
        "existing_todos": existing_todos,
        "user_identity": user_identity,
    }


# ─── Node 1: Unified analysis (single LLM call) ────────────────────────────

_UNIFIED_SYSTEM = (
    "You are a sharp, concise email assistant helping a busy professional decide what to do with an email.\n\n"
    "Analyze the email and output a SINGLE JSON OBJECT with exactly three keys: \"replies\", \"todos\", \"meeting\".\n\n"

    "═══ THE #1 RULE: BE SPECIFIC ═══\n"
    "Every label and context MUST include concrete details FROM the email — names, subjects, dates,\n"
    "amounts, items, document names, event names, deadlines. If the email is about Order #4829, the\n"
    "label must say 'Order #4829'. If it's about a 'Quarterly Report', say 'Quarterly Report'.\n"
    "NEVER use generic placeholders. Extract the actual nouns and details from the email body.\n\n"

    "═══ CRITICAL CONTEXT RULES ═══\n"
    "You MUST understand the email's context before generating suggestions:\n"
    "  • Who is the sender? What is their relationship to the user?\n"
    "  • What specifically is being asked or communicated?\n"
    "  • Only suggest actions the user CAN actually take.\n"
    "  • NEVER fabricate facts not present in the email (e.g. if the email says 'borrowed guitars', do NOT say 'sold guitars').\n"
    "  • A reply is something the user would SEND TO THE SENDER — not a note to themselves.\n"
    "  • 'Save for later', 'Set a reminder', 'Bookmark this' are NOT replies — they are todos.\n"
    "  • NEVER suggest replying to automated emails, receipts, or notifications.\n"
    "  • EXCEPTION: Emails from .edu senders (university LMS like Brightspace/Canvas) "
    "often forward real messages from professors. If the email body explicitly asks "
    "the user to reply or send something, treat it as a real request — suggest replies AND todos.\n\n"

    "═══ REPLIES — what could the user SEND BACK to the sender? ═══\n"
    "Array of {\"label\": \"...\", \"context\": \"...\"}\n\n"
    "label rules:\n"
    "  • 2–8 words. Written in FIRST PERSON as what the user would say back to the sender.\n"
    "  • SPECIFIC to this email — include the actual topic/item/event name from the email.\n"
    "  • Never generic: NO 'Reply', 'Respond', 'Acknowledge', 'Follow up', 'Confirm receipt'.\n"
    "  • Think: if someone reads ONLY the label (not the email), they should understand what the reply is about.\n"
    "  • Each reply must represent a DISTINCT choice — cover the realistic options the user has.\n"
    "  • If the email presents a yes/no decision, provide both options.\n"
    "  • If the email asks multiple questions, you may suggest a reply that addresses all of them.\n\n"
    "Good label examples:\n"
    "  Email from boss about Friday offsite: → [\"Yes, I'll join the offsite\", \"Can't make Friday's offsite\"]\n"
    "  Email from colleague asking for Q3 report: → [\"Sending Q3 report now\", \"Need more time on Q3 report\"]\n"
    "  Email thanking user for guitar help: → [\"Happy to help with the guitars\"]\n"
    "  Email invitation to Sarah's birthday: → [\"I'll be at Sarah's birthday\", \"Can't make Sarah's party\"]\n"
    "  Email about project Alpha deadline: → [\"Alpha will be ready on time\", \"Need extension on Alpha\"]\n\n"
    "Bad label examples (NEVER do these):\n"
    "  ✗ 'Reply to email', 'Send response', 'Answer question', 'Acknowledge receipt'\n"
    "  ✗ 'Confirm attendance' (attendance to WHAT? Be specific!)\n"
    "  ✗ 'Agree to proposal' (WHICH proposal? Name it!)\n"
    "  ✗ 'Inform George about X' — NEVER refer to the user in third person\n"
    "  ✗ 'Save for later', 'Set a reminder', 'Bookmark' — these are todos NOT replies\n"
    "  ✗ 'We sold them' or any factual claim not in the email\n"
    "  ✗ 'Let me know' — vague, not a real reply\n"
    "  ✗ Two labels that say the same thing differently\n\n"
    "When to include replies:\n"
    "  YES: email asks a question or requests something from the user\n"
    "  YES: conversational / personal email from a real person expecting a response\n"
    "  YES: invitation that the user can accept or decline\n"
    "  NO: automated / noreply sender (EXCEPT .edu senders — university LMS platforms like Brightspace/Canvas use noreply addresses to forward real professor messages that may require action)\n"
    "  NO: pure newsletter, marketing email, or notification\n"
    "  NO: receipt or order confirmation\n"
    "  NO: announcements from institutions where a reply is not expected\n"
    "Maximum 3 reply options.\n\n"
    "context: 1–2 sentences with ALL concrete details the drafter needs: sender's full name,\n"
    "the exact question asked, specific dates/amounts/items/documents mentioned, any constraints.\n"
    "The drafter should be able to write a perfect reply using ONLY the context — no guessing.\n\n"

    "═══ TODOS — what must the user DO on their own? ═══\n"
    "Array of {\"label\": \"...\", \"context\": \"...\"}\n\n"
    "label rules:\n"
    "  • Start with an ACTION VERB: Submit, Review, Register, Sign, Upload, Complete…\n"
    "  • MUST include the specific item/document/event/order name from the email.\n"
    "  • Include the DEADLINE if mentioned: 'Submit thesis draft by Nov 15'\n"
    "  • 3–10 words. Specific enough that the user knows EXACTLY what to do without reading the email.\n"
    "  • The label alone must answer: WHAT specifically? For WHO/WHAT? By WHEN?\n\n"
    "Good todo examples:\n"
    "  Email about overdue library book 'Clean Code': → \"Return 'Clean Code' to library\"\n"
    "  Email about AWS bill overdue: → \"Pay AWS invoice #7823 ($142)\"\n"
    "  Email about project proposal from Dr. Smith due Mar 15: → \"Submit proposal to Dr. Smith by Mar 15\"\n"
    "  Email about order #4829 delivery issue: → \"Check delivery status for order #4829\"\n"
    "  Email about new employee onboarding docs: → \"Complete HR onboarding forms for new role\"\n"
    "  Email about renewal of gym membership: → \"Renew gym membership before April 1\"\n\n"
    "Bad todo examples (NEVER do these):\n"
    "  ✗ 'Check item status' (WHAT item?!)\n"
    "  ✗ 'Review order details' (WHICH order? What details?)\n"
    "  ✗ 'Submit form' (WHAT form? To whom? By when?)\n"
    "  ✗ 'Follow up' (With whom? About what?)\n"
    "  ✗ 'Review document' (WHICH document?)\n"
    "  ✗ 'Complete registration' (For what?)\n"
    "  ✗ Any label where a reader couldn't act without re-reading the email\n\n"
    "When to include todos:\n"
    "  YES: any deadline mentioned\n"
    "  YES: form to fill, document to review, link to visit, registration required\n"
    "  YES: user asked to prepare or complete something independently\n"
    "  YES: 'save for later', 'set a reminder' type actions belong HERE as todos\n"
    "  NO: 'Reply to this email' — that belongs in replies\n"
    "  NO: vague things that aren't real tasks\n"
    "Maximum 3 todos.\n\n"

    "═══ MEETING — is this about a future meeting the user can act on? ═══\n"
    "Single object: {\"is_meeting\": true/false, \"context\": \"...\", \"actions\": [...]}\n"
    "is_meeting: true ONLY when ALL of these are true:\n"
    "  1. A FUTURE meeting/event is the CORE topic of the email\n"
    "  2. The user has AGENCY to accept, decline, or propose changes\n"
    "  3. The sender is directly inviting the user OR asking to schedule with them\n\n"
    "is_meeting: false when:\n"
    "  • The email is an announcement about a class, lecture, or fixed event the user cannot change\n"
    "  • A professor/institution is announcing a schedule — user can only attend or not\n"
    "  • The email mentions a past meeting\n"
    "  • The meeting is already confirmed and no RSVP is needed\n"
    "  • A Calendly link or scheduling tool is already provided (suggest using that tool instead)\n\n"
    "actions: array of specific actions available. ONLY include actions that make sense:\n"
    "  • \"accept\" — only if the user can RSVP yes to this specific meeting\n"
    "  • \"decline\" — only if the user can decline (not for mandatory classes/lectures)\n"
    "  • \"suggest_time\" — only if the meeting time is negotiable\n"
    "  • \"cancel\" — only if the user organized the meeting and wants to cancel\n"
    "Do NOT blindly include all three actions. Choose only the relevant ones.\n"
    "context: attendees, event name, when if known, location/link.\n\n"

    "OUTPUT — valid JSON only, no markdown:\n"
    "{\"replies\":[{\"label\":\"...\",\"context\":\"...\"},...],\"todos\":[{\"label\":\"...\",\"context\":\"...\"},...],\"meeting\":{\"is_meeting\":false,\"context\":\"\",\"actions\":[]}}"
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
    user_identity = state.get("user_identity", "")

    # Skip clearly automated senders — but exempt .edu senders
    # because universities use noreply addresses for LMS platforms
    # (Brightspace, Canvas, Blackboard) that forward real professor messages
    sender_lower = sender.lower()
    _is_edu_sender = ".edu" in sender_lower
    if not _is_edu_sender and any(s in sender_lower for s in [
        "noreply", "no-reply", "mailer-daemon", "donotreply",
        "notifications@", "alerts@", "automated@",
    ]):
        return {"raw_analysis": {"replies": [], "todos": [], "meeting": {"is_meeting": False, "context": ""}}}

    existing_block = ""
    if existing_todos:
        existing_block = "\nEXISTING TODOS (do NOT suggest anything similar):\n" + "\n".join(
            f"- {t}" for t in existing_todos[:12]
        )

    identity_block = ""
    if user_identity:
        # Extract first name for additional filtering
        first_name = user_identity.split()[0] if user_identity.split() else ""
        identity_block = (
            f"\nIMPORTANT: The person reading this email is {user_identity}. "
            "They are the RECIPIENT. Never write reply labels that refer to them "
            "by their own name as if they were a third party (e.g. do NOT write "
            f"'Explain X to {first_name}' or 'Inform {first_name} about Y' — "
            "write 'I'll explain X' or just 'Explain X').\n"
            "The user is the one reading and acting on this email — all replies "
            "should be from THEIR perspective as THE SENDER of the reply.\n"
        )

    prompt = (
        f"From: {sender}\n"
        f"Subject: {subject}\n"
        f"Body: {body}\n"
        f"Thread context: {thread_ctx}\n"
        f"{memory_ctx}\n"
        f"{sender_hist}\n"
        f"{existing_block}\n"
        f"{identity_block}\n"
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
    user_identity = state.get("user_identity", "")
    # Extract user's first name for filtering self-references
    user_first = user_identity.split()[0].lower() if user_identity and user_identity.split() else ""

    # Labels that are personal tasks, NOT replies to send
    _NOT_REPLY_LABELS = {
        "save for later", "set a reminder", "bookmark", "bookmark this",
        "mark as read", "mark as unread", "archive", "snooze",
        "remind me", "follow up later", "note to self",
    }

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
        # Filter out self-referencing labels
        label_lower = label.lower()
        if label_lower in _NOT_REPLY_LABELS:
            continue
        if user_first and (f"inform {user_first}" in label_lower or f"tell {user_first}" in label_lower or f"notify {user_first}" in label_lower or f"to {user_first}" in label_lower):
            continue
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
        available_actions = meeting_info.get("actions", [])
        if not isinstance(available_actions, list):
            available_actions = []

        action_map = {
            "accept": {"label": "Accept meeting", "meeting_action": "accept"},
            "decline": {"label": "Decline meeting", "meeting_action": "decline"},
            "suggest_time": {"label": "Suggest different time", "meeting_action": "reschedule"},
            "cancel": {"label": "Cancel meeting", "meeting_action": "cancel"},
        }
        for action_key in available_actions:
            action_key = str(action_key).strip().lower()
            if action_key in action_map:
                info = action_map[action_key]
                meeting_actions.append({
                    "type": "meeting",
                    "label": info["label"],
                    "context": f"{info['label']}. {ctx}".strip(),
                    "meeting_action": info["meeting_action"],
                })
        # Fallback: if LLM returned is_meeting=true but no valid actions, use accept+decline
        if not meeting_actions:
            meeting_actions = [
                {"type": "meeting", "label": "Accept meeting", "context": f"Accept the meeting. {ctx}".strip(), "meeting_action": "accept"},
                {"type": "meeting", "label": "Decline meeting", "context": f"Decline the meeting. {ctx}".strip(), "meeting_action": "decline"},
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

