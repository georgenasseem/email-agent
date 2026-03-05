"""Dedicated LangGraph for intelligent quick-action suggestions.

Architecture: Quick Actions answer "WHAT should I do?" (suggest actions).
             Drafting answers "HOW do I do it?" (compose the response).

Runs multiple analysis nodes to produce high-quality,
context-aware actions for a single email:

1. gather_context    — Pull memory, sender history, existing todos.
2. reply_analysis    — Should the user reply? What are the options?
                       Meeting/scheduling actions are reply-type with meeting flags.
3. todo_analysis     — Extract actionable tasks. Strict dedup against existing todos.
4. merge_actions     — Deduplicate, validate, and rank the final suggestions.

Key design principles:
- Actions are SHORT labels (what to do), not detailed instructions
- Meeting actions are integrated into reply options with meeting_action flags
- After clicking, the system decides HOW to execute (draft, schedule, etc.)
- No "Reply:", "Todo:", "Schedule:" prefixes — type field handles routing
- Color coding in UI: orange=reply, purple=todo, grey=custom
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
    reply_options: list[dict]       # [{type, label, context, has_meeting, meeting_action}]
    todo_options: list[dict]        # [{type, label, context}]
    no_action_message: str
    final_options: list[dict]       # merged + ranked actions


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


# ─── Node 1: Reply analysis (includes meeting detection) ───────────────────

def reply_analysis_node(state: QuickActionState) -> dict:
    """Determine if a reply is warranted and suggest short action labels.

    Meeting/scheduling actions are integrated here — they are reply-type
    actions with has_meeting=True and a meeting_action flag.
    """
    email = state.get("email") or {}
    subject = (email.get("subject") or "")[:150]
    body = (email.get("clean_body") or email.get("body") or email.get("snippet") or "")[:1500]
    sender = email.get("sender", "")
    category = email.get("category", "informational")
    thread_ctx = (email.get("thread_context") or "")[:400]
    memory_ctx = state.get("memory_context", "")
    sender_hist = state.get("sender_history", "")

    # Only skip truly automated senders
    sender_lower = sender.lower()
    if any(s in sender_lower for s in ["noreply", "no-reply", "mailer-daemon", "donotreply"]):
        return {"reply_options": []}

    llm = get_llm(task="decide")
    parser = StrOutputParser()

    system = """You analyze emails and suggest SHORT REPLY action labels — things the user would SEND BACK to the sender.

A REPLY is a message the user sends to the email sender. It is NOT a personal task.

Each action is a JSON object with:
- "label": SHORT action text (2-6 words). What the user sees on a button.
- "context": Brief hidden context for the AI drafter — include WHO is involved, WHAT the topic is, and any key details/dates (1-2 sentences).
- "has_meeting": true if the email is ABOUT a future meeting, event, or scheduling topic.
- "meeting_action": "accept" | "decline" | "reschedule" | null. Only set when has_meeting is true.

CRITICAL RULES:
1. REPLIES are things you would SEND to the person. "Acknowledge" or "Confirm receipt" are replies. "Prepare a presentation" is NOT a reply — it's a task.
2. Do NOT suggest personal tasks/to-do items as replies. These are handled separately.
   BAD replies: "Prepare presentation", "Review document", "Complete assignment", "Study for exam".
   GOOD replies: "Acknowledge", "Confirm attendance", "Ask for details", "Decline politely".
3. Set has_meeting=true when the email is ABOUT a future meeting or event. This includes:
   - Meeting invitations ("Can we meet Thursday?")
   - Meeting reminders ("Don't forget our meeting tomorrow")
   - Meeting issues ("The meeting is not in my calendar")
   - Scheduling discussions ("Are you free Friday?")
   - Event invitations ("You're invited to the workshop on March 5")
4. When has_meeting=true, ALWAYS suggest accept, decline, AND reschedule options (3 actions).
5. References to PAST meetings used as context only ("during our last meeting", "as we discussed") do NOT count. Only set has_meeting=true when a FUTURE meeting is the topic.
6. For non-meeting emails, provide 1-3 SPECIFIC reply options. Avoid generic labels.
   - If the email asks a QUESTION, suggest answering it: "Confirm availability", "Share the file", "Provide update".
   - If the email requests something, offer to agree or push back: "Agree to help", "Can't right now".
   - If the email is sharing info that benefits from acknowledgment: "Thank them", "Acknowledge".
7. Do NOT suggest actions for: automated notifications, receipts, system alerts, newsletters, marketing emails.
8. For emails that are purely FYI with no reply expected, return an empty array [].
9. Maximum 4 reply actions total.
10. Make CONTEXT rich — the drafter uses it to compose the full reply. Include the sender's name, topic, and key details.

Output ONLY a JSON array of objects. No other text.

EXAMPLES:
Meeting invitation for next Thursday → [
  {"label": "Accept meeting", "context": "Accept meeting with Prof. Smith about research proposal, Thursday 3pm", "has_meeting": true, "meeting_action": "accept"},
  {"label": "Decline meeting", "context": "Decline meeting with Prof. Smith on Thursday — unable to attend", "has_meeting": true, "meeting_action": "decline"},
  {"label": "Suggest different time", "context": "Reschedule meeting with Prof. Smith, propose alternative time for research discussion", "has_meeting": true, "meeting_action": "reschedule"}
]
Email asking "Can you send me the report?" → [
  {"label": "Send the report", "context": "Agree to send the requested report to Sarah about the Q4 analysis", "has_meeting": false, "meeting_action": null},
  {"label": "Need more time", "context": "Let Sarah know the report needs more time before sending", "has_meeting": false, "meeting_action": null}
]
Email asking for help with a project → [
  {"label": "Agree to help", "context": "Agree to help Alex with the database migration project", "has_meeting": false, "meeting_action": null},
  {"label": "Too busy right now", "context": "Politely decline helping Alex due to current workload", "has_meeting": false, "meeting_action": null}
]
Professor asking to review a draft → [
  {"label": "Confirm will review", "context": "Confirm to Prof. Lee that you will review the paper draft by the deadline", "has_meeting": false, "meeting_action": null},
  {"label": "Ask for extension", "context": "Request more time from Prof. Lee to review the paper draft", "has_meeting": false, "meeting_action": null}
]
Reminder about upcoming deadline → [
  {"label": "Acknowledge reminder", "context": "Confirm receipt of the deadline reminder and confirm on track", "has_meeting": false, "meeting_action": null}
]
Automated receipt, newsletter, or FYI → []"""

    prompt = f"""From: {sender}
Subject: {subject}
Category: {category}
Content: {body}
Thread: {thread_ctx}
{memory_ctx}
{sender_hist}

Suggest reply actions the user might want to send. Be SPECIFIC to the email content. Keep labels SHORT (2-6 words). Include meeting accept/decline/reschedule if applicable. (JSON array only):"""

    try:
        raw = (llm | parser).invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
        text = _extract_json_array(raw)
        options = json.loads(text)
        if isinstance(options, list):
            results = []
            for o in options:
                if not isinstance(o, dict):
                    continue
                label = str(o.get("label", "")).strip()
                if not label:
                    continue
                action = {
                    "type": "reply",
                    "label": label,
                    "context": str(o.get("context", "")).strip(),
                    "has_meeting": bool(o.get("has_meeting", False)),
                    "meeting_action": o.get("meeting_action") if o.get("has_meeting") else None,
                }
                # Validate meeting_action values
                if action["meeting_action"] not in (None, "accept", "decline", "reschedule"):
                    action["meeting_action"] = None
                results.append(action)
            return {"reply_options": results[:4]}
    except Exception as e:
        logger.debug("Reply analysis error: %s", e)
    return {"reply_options": []}


# ─── Node 2: Todo analysis ─────────────────────────────────────────────────

def todo_analysis_node(state: QuickActionState) -> dict:
    """Extract concrete, actionable tasks with deadlines. Strict dedup."""
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

    system = f"""You extract actionable PERSONAL TASKS from emails — things the user needs to DO (not send).

A TASK is something the user does on their own: prepare, study, submit, review, complete, write, etc.
A task is NOT a reply to the sender — replies are handled separately.

Each task is a JSON object with:
- "label": SHORT task description (3-8 words). What the user sees on a button. INCLUDE DEADLINES when mentioned.
- "context": Fuller description with specific details, deadlines, links, and relevant info from the email (1-2 sentences).

CRITICAL RULES:
1. Only extract tasks the USER personally needs to do. NOT tasks someone else is doing.
2. Labels must be SHORT but SPECIFIC. Always include deadlines or dates when the email mentions them.
   BAD: "Submit form". GOOD: "Submit hackathon form by Feb 27".
   BAD: "Review this". GOOD: "Review proposal draft by Monday".
   BAD: "Check grades". GOOD: "Check posted final grades on portal".
3. Do NOT suggest "send a reply", "respond to email", "acknowledge" — those are replies, not tasks.
4. Do NOT suggest vague tasks: "check the email", "look into it", "follow up".
5. Do NOT duplicate existing todos (see below if any).
6. Maximum 3 tasks.
7. Capture EVERY obviously actionable item: "prepare X", "submit Y", "complete Z", "register for W", "review by date".
8. If the email mentions a document to read, a form to fill, a link to visit, or a deadline to meet — those are tasks.
9. Do NOT create tasks for newsletters or automated notifications unless they have specific personal deadlines.
10. For multi-part requests, extract each distinct task separately (up to the 3 limit).{existing_block}

Output a JSON array of objects. Empty array [] if no concrete tasks.

EXAMPLES:
"Don't forget to prepare the presentation for Friday" → [{{"label": "Prepare presentation by Friday", "context": "Prepare the presentation for the team meeting on Friday as discussed"}}]
Email about hackathon form deadline → [{{"label": "Submit hackathon form by Feb 27", "context": "Submit NYUAD Hackathon Speaker Form by Feb 27th 11:59 PM EST via the link in email"}}]
Email with attached doc to review → [{{"label": "Review attached proposal", "context": "Review the project proposal document attached by Dr. Kim before next meeting"}}]
Email about course registration opening → [{{"label": "Register for fall courses", "context": "Course registration opens March 1st — select and register for fall semester courses on the portal"}}]
Email with no user action needed → []"""

    prompt = f"""From: {sender}
Subject: {subject}
Content: {body}

Extract concrete actionable tasks the user must personally do. Include deadlines and specifics. (JSON array only):"""

    try:
        raw = (llm | parser).invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
        text = _extract_json_array(raw)
        options = json.loads(text)
        if isinstance(options, list):
            results = []
            for o in options:
                if isinstance(o, dict):
                    label = str(o.get("label", "")).strip()
                    context = str(o.get("context", "")).strip()
                elif isinstance(o, str):
                    label = o.strip()
                    context = label
                else:
                    continue
                if not label:
                    continue
                task_lower = label.lower()
                is_dup = any(_fuzzy_match(task_lower, existing) for existing in existing_todos)
                if not is_dup:
                    results.append({
                        "type": "todo",
                        "label": label,
                        "context": context or label,
                    })
            return {"todo_options": results[:3]}
    except Exception as e:
        logger.debug("Todo analysis error: %s", e)
    return {"todo_options": []}


# ─── Node 3: Merge and rank ─────────────────────────────────────────────────

# Words that indicate an action is a personal task, not a reply to send
_TASK_INDICATORS = {
    "prepare", "submit", "complete", "finish", "review", "write", "create",
    "study", "read", "update", "fix", "build", "design", "implement",
    "organize", "clean", "schedule", "plan", "draft", "edit", "research",
    "book", "buy", "order", "register", "sign", "apply", "upload",
    "download", "install", "setup", "configure", "attend",
}


def _is_task_like(label: str) -> bool:
    """Detect if a label is really a personal task, not a reply."""
    words = set(re.findall(r"[a-z]+", label.lower()))
    return bool(words & _TASK_INDICATORS)


def merge_actions_node(state: QuickActionState) -> dict:
    """Combine, deduplicate, validate, and rank all suggested actions.

    Validation steps:
    1. Reclassify mistyped actions (task-like replies → todo, reply-like todos → reply)
    2. Deduplicate by fuzzy label match
    3. Meeting emails must have both accept and decline
    4. Cap at 5 total
    """
    reply_opts = list(state.get("reply_options", []))
    todo_opts = list(state.get("todo_options", []))

    # ── Step 1: Reclassify mistyped actions ──────────────────────────
    corrected_replies: list[dict] = []
    for opt in reply_opts:
        if not isinstance(opt, dict) or not opt.get("label"):
            continue
        label = opt["label"]
        # If a "reply" is actually a task, move it to todos
        if _is_task_like(label) and not opt.get("has_meeting"):
            opt["type"] = "todo"
            todo_opts.append(opt)
        else:
            corrected_replies.append(opt)

    # ── Step 2: Deduplicate ──────────────────────────────────────────
    final: list[dict] = []
    seen_labels: set[str] = set()

    # Replies first (includes meeting actions)
    for opt in corrected_replies:
        key = opt["label"].lower().strip()
        if key not in seen_labels and not any(_fuzzy_match(key, s) for s in seen_labels):
            seen_labels.add(key)
            final.append(opt)

    # Then todos
    for opt in todo_opts:
        if not isinstance(opt, dict) or not opt.get("label"):
            continue
        key = opt["label"].lower().strip()
        if key not in seen_labels and not any(_fuzzy_match(key, s) for s in seen_labels):
            seen_labels.add(key)
            opt["type"] = "todo"  # Ensure type is correct
            final.append(opt)

    # ── Step 2b: Keyword-based meeting safety net ──────────────────────
    # If the LLM didn't flag has_meeting but labels clearly reference meetings/calendar,
    # auto-correct them to be meeting actions.
    _meeting_keywords = {"meeting", "calendar", "attend", "schedule", "rsvp", "invitation", "invite"}
    for opt in final:
        if opt.get("type") == "reply" and not opt.get("has_meeting"):
            label_words = set(opt.get("label", "").lower().split())
            if label_words & _meeting_keywords:
                opt["has_meeting"] = True
                # Try to infer meeting_action from label
                lbl = opt["label"].lower()
                if any(w in lbl for w in ("accept", "confirm", "add to calendar", "attend")):
                    opt["meeting_action"] = "accept"
                elif any(w in lbl for w in ("decline", "reject", "can't make")):
                    opt["meeting_action"] = "decline"
                elif any(w in lbl for w in ("reschedule", "different time", "postpone")):
                    opt["meeting_action"] = "reschedule"

    # ── Step 3: Meeting validation ───────────────────────────────────
    # Any meeting-flagged action means we need the full trio: accept, decline, reschedule
    has_any_meeting = any(o.get("has_meeting") for o in final)
    has_accept = any(o.get("meeting_action") == "accept" for o in final)
    has_decline = any(o.get("meeting_action") == "decline" for o in final)
    has_reschedule = any(o.get("meeting_action") == "reschedule" for o in final)

    if has_any_meeting:
        # Find a reference meeting action for context cloning
        _ref = next((o for o in final if o.get("has_meeting")), None)
        _ref_ctx = _ref.get("context", "") if _ref else ""

        if not has_accept:
            accept = {
                "type": "reply",
                "label": "Accept meeting",
                "context": _ref_ctx.replace("Decline", "Accept").replace("decline", "accept").replace("Reschedule", "Accept").replace("reschedule", "accept"),
                "has_meeting": True,
                "meeting_action": "accept",
            }
            # Insert at position 0 (accept should be first)
            final.insert(0, accept)

        if not has_decline:
            # Insert after accept
            _accept_idx = next((i for i, o in enumerate(final) if o.get("meeting_action") == "accept"), 0)
            decline = {
                "type": "reply",
                "label": "Decline meeting",
                "context": _ref_ctx.replace("Accept", "Decline").replace("accept", "decline").replace("Reschedule", "Decline").replace("reschedule", "decline"),
                "has_meeting": True,
                "meeting_action": "decline",
            }
            final.insert(_accept_idx + 1, decline)

        if not has_reschedule:
            # Insert after decline
            _decline_idx = next((i for i, o in enumerate(final) if o.get("meeting_action") == "decline"), 1)
            reschedule = {
                "type": "reply",
                "label": "Suggest different time",
                "context": _ref_ctx.replace("Accept", "Reschedule").replace("accept", "reschedule").replace("Decline", "Reschedule").replace("decline", "reschedule"),
                "has_meeting": True,
                "meeting_action": "reschedule",
            }
            final.insert(_decline_idx + 1, reschedule)

    # ── Step 4: Cap at 5 ─────────────────────────────────────────────
    final = final[:5]

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
    """Build the quick-actions analysis graph.

    3 nodes: gather_context → reply_analysis (includes meeting) → todo_analysis → merge_actions.
    """
    graph = StateGraph(QuickActionState)

    graph.add_node("gather_context", gather_context_node)
    graph.add_node("reply_analysis", reply_analysis_node)
    graph.add_node("todo_analysis", todo_analysis_node)
    graph.add_node("merge_actions", merge_actions_node)

    graph.set_entry_point("gather_context")
    # Parallel fan-out: gather_context fires both reply_analysis and
    # todo_analysis simultaneously; merge_actions is the fan-in barrier.
    graph.add_edge("gather_context", "reply_analysis")
    graph.add_edge("gather_context", "todo_analysis")
    graph.add_edge("reply_analysis", "merge_actions")
    graph.add_edge("todo_analysis",  "merge_actions")
    graph.add_edge("merge_actions",  END)

    return graph.compile()


_QUICK_ACTIONS_GRAPH = _build_quick_actions_graph()


def suggest_quick_actions(email: dict) -> list[dict]:
    """Run the quick-actions graph for a single email.

    Returns a list of action dicts like:
      [{"type": "reply", "label": "Agree to meeting", "context": "...", "has_meeting": True, "meeting_action": "accept"},
       {"type": "todo", "label": "Submit form by Friday", "context": "..."}]
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
