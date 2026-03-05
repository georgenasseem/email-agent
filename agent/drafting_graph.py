"""LangGraph-based drafting pipeline with a critique-revise cycle.

This is the core of the "Human-in-the-loop" spirit: the LLM first drafts
a reply, then a separate *critic* LLM pass scores it.  If the score is
below the acceptance threshold **and** we haven't revised yet, the graph
loops back to a revise node.  Only after passing critique (or exhausting
the revision budget) does it finalize the output.

Graph structure
───────────────

    START
      │
    [plan]        ← structured reply plan (goal, key points, tone)
      │
    [draft]       ← write the full reply using the plan
      │
    [critique]    ← LLM-as-judge: score 1–5, identify specific issues
      │
   ┌──┴────────────────────┐
   │  conditional edge      │
   │  score < 4             │ score ≥ 4 (or revisions exhausted)
   ▼                        ▼
 [revise]               [finalize]
   │                        │
   └──────►[critique]       END
           (cycle back)

The cycle is bounded by `revision_count` in the state (max 1 revision).
This means at most: plan → draft → critique → revise → critique → finalize.
"""
import json
import logging
import re
from typing import TypedDict

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

from agent.llm import get_llm
from agent.drafter import analyze_roles, plan_reply, draft_reply
from agent.style_learner import load_persisted_style

logger = logging.getLogger(__name__)

# Minimum critique score to skip the revise node (1–5 scale)
_QUALITY_THRESHOLD = 4
# Maximum number of revise→critique cycles allowed
_MAX_REVISIONS = 1


class DraftState(TypedDict, total=False):
    """Flowing state for the drafting sub-graph."""
    email: dict
    decision: str
    style_notes: str
    # Derived in nodes
    roles: dict
    reply_plan: dict
    draft: str
    critique: dict          # {"score": int, "issues": [str], "summary": str}
    revision_count: int
    final_draft: str


# ─── Node: plan ─────────────────────────────────────────────────────────────

def plan_node(state: DraftState) -> dict:
    """Analyse roles and build a structured reply plan."""
    email = state.get("email") or {}
    decision = state.get("decision") or ""
    roles = analyze_roles(email)
    reply_plan = plan_reply(email, decision=decision)
    return {"roles": roles, "reply_plan": reply_plan}


# ─── Node: draft ────────────────────────────────────────────────────────────

def draft_node(state: DraftState) -> dict:
    """Write the first draft using the plan produced by `plan_node`."""
    email = state.get("email") or {}
    decision = state.get("decision") or ""
    style_notes = state.get("style_notes") or ""
    try:
        text = draft_reply(email, decision=decision, style_notes=style_notes)
    except Exception as e:
        logger.warning("draft_node error: %s", e)
        text = ""
    return {"draft": text, "revision_count": 0}


# ─── Node: critique ──────────────────────────────────────────────────────────

def critique_node(state: DraftState) -> dict:
    """LLM-as-judge: score the current draft and list specific issues.

    Returns a structured critique dict with:
      - score (1–5): overall quality
      - issues: bullet list of specific problems
      - summary: one-sentence verdict
    """
    email = state.get("email") or {}
    draft = state.get("draft") or ""
    decision = state.get("decision") or ""
    roles = state.get("roles") or {}
    reply_plan = state.get("reply_plan") or {}

    if not draft:
        # Nothing to critique — mark as acceptable so we finalise immediately
        return {"critique": {"score": 5, "issues": [], "summary": "No draft to critique."}}

    subject = (email.get("subject") or "")[:100]
    original_body = (email.get("clean_body") or email.get("body") or email.get("snippet") or "")[:600]
    user_name = (roles.get("user_name") or "")
    other_name = (roles.get("other_display_name") or "them")
    planned_tone = reply_plan.get("tone", "neutral")
    key_points = reply_plan.get("key_points") or []

    system = """You are an expert email editor reviewing a drafted reply.

Score the draft from 1 to 5:
  5 = excellent, ready to send
  4 = good, minor polish needed (acceptable)
  3 = mediocre, significant issues present
  1-2 = poor, fundamental problems

Check specifically for:
- Correct addressing (reply is TO the sender, greeting uses their name)
- Executes the user's intended action
- Does NOT copy / paraphrase the original email back
- Appropriate tone and conciseness
- Missing key points from the plan

Output ONLY valid JSON with this shape:
{
  "score": <int 1-5>,
  "issues": ["..."],
  "summary": "one sentence verdict"
}"""

    prompt = f"""Original email subject: {subject}
Original email (first 400 chars): {original_body[:400]}

User's intended action: {decision}
Planned tone: {planned_tone}
Key points that MUST be covered: {json.dumps(key_points)}

User name: {user_name}
Recipient name: {other_name}

Draft to review:
---
{draft}
---

Score and identify issues (JSON only):"""

    try:
        llm = get_llm(task="decide")
        parser = StrOutputParser()
        raw = (llm | parser).invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
        # Strip markdown fences
        text = re.sub(r"^```[a-zA-Z0-9_]*\s*", "", raw.strip())
        text = re.sub(r"```\s*$", "", text).strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            data = json.loads(text[start:end])
            if isinstance(data, dict) and "score" in data:
                return {"critique": {
                    "score": int(data.get("score", 3)),
                    "issues": list(data.get("issues") or []),
                    "summary": str(data.get("summary", "")),
                }}
    except Exception as e:
        logger.debug("critique_node parse error: %s", e)

    # If critique fails, assume acceptable rather than loop forever
    return {"critique": {"score": 4, "issues": [], "summary": "Critique unavailable — accepting draft."}}


# ─── Conditional edge: should we revise? ────────────────────────────────────

def _should_revise(state: DraftState) -> str:
    """Route to 'revise' if quality is below threshold and budget remains."""
    critique = state.get("critique") or {}
    score = critique.get("score", 5)
    revision_count = state.get("revision_count", 0)

    if score < _QUALITY_THRESHOLD and revision_count < _MAX_REVISIONS:
        return "revise"
    return "finalize"


# ─── Node: revise ────────────────────────────────────────────────────────────

def revise_node(state: DraftState) -> dict:
    """Rewrite the draft, addressing specific issues from the critique."""
    email = state.get("email") or {}
    current_draft = state.get("draft") or ""
    critique = state.get("critique") or {}
    decision = state.get("decision") or ""
    style_notes = state.get("style_notes") or ""
    roles = state.get("roles") or {}
    reply_plan = state.get("reply_plan") or {}
    revision_count = state.get("revision_count", 0)

    issues = critique.get("issues") or []
    critique_summary = critique.get("summary", "")

    if not issues and not critique_summary:
        # Nothing specific to fix — return current draft unchanged and abort cycle
        return {"draft": current_draft, "revision_count": revision_count + 1}

    # Load style
    persisted_style = ""
    try:
        persisted_style = load_persisted_style()
    except Exception:
        pass
    style = f"Write in this style: {style_notes}" if style_notes else (
        f"Match this style:\n{persisted_style}" if persisted_style else
        "Professional, concise, workplace-appropriate."
    )

    user_name = roles.get("user_name") or ""
    other_name = roles.get("other_display_name") or "them"
    other_first = roles.get("other_first_name") or "there"
    contact_greeting = roles.get("contact_greeting") or ""
    key_points = reply_plan.get("key_points") or []
    planned_tone = reply_plan.get("tone", "neutral")
    goal = reply_plan.get("goal", "")

    greeting_instr = (
        f"Start with exactly '{contact_greeting},' as the greeting."
        if contact_greeting else
        f"Start with an appropriate greeting addressing {other_first}."
    )

    system = f"""You are revising an email draft to fix specific issues identified by a reviewer.

You are writing AS: {user_name or 'the user'} (reply FROM this person)
You are writing TO: {other_name} — the greeting MUST address them.

Goal: {goal}
Tone: {planned_tone}
Key points to cover: {json.dumps(key_points)}

STYLE: {style}

ISSUES TO FIX:
{chr(10).join(f"- {i}" for i in issues)}

REVIEWER VERDICT: {critique_summary}

RULES:
- {greeting_instr}
- Do NOT copy/paraphrase the original email back
- Fix every issue listed above
- Output ONLY the email body (greeting + body + closing). No markdown."""

    subject = (email.get("subject") or "")[:100]
    original_body = (email.get("clean_body") or email.get("body") or email.get("snippet") or "")[:500]

    prompt = f"""Original email subject: {subject}
Original email: {original_body}

User's action: {decision}

Current draft (needs revision):
---
{current_draft}
---

Write the revised email now:"""

    try:
        llm = get_llm(task="draft")
        parser = StrOutputParser()
        revised = (llm | parser).invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
        revised = revised.strip() if revised else current_draft
    except Exception as e:
        logger.warning("revise_node error: %s", e)
        revised = current_draft

    return {"draft": revised, "revision_count": revision_count + 1}


# ─── Node: finalize ──────────────────────────────────────────────────────────

# Patterns that indicate LLM meta-commentary (NOT part of the actual email)
_FOOTNOTE_PATTERNS = [
    re.compile(r'^\s*[-—–]{2,}\s*$'),                          # separator lines "---"
    re.compile(r'^\s*(note|notes|p\.?s\.?|n\.?b\.?)\s*:', re.I),  # "Note:", "P.S.:"
    re.compile(r'^\s*this (reply|response|email|message|draft)\b', re.I),  # "This reply directly..."
    re.compile(r'^\s*the (above|reply|response|email)\b', re.I),           # "The above response..."
    re.compile(r'^\s*i\'?ve (addressed|included|ensured|covered)\b', re.I), # "I've addressed..."
    re.compile(r'^\s*\[.*\]\s*$'),                              # "[Note: ...]" bracketed
    re.compile(r'directly addresses|ensures? (they|the user|the recipient|he|she) ha(s|ve)', re.I),
    re.compile(r'as (requested|instructed|per (your|the) instructions)', re.I),
    re.compile(r'^\s*\*\*?note\*?\*?\s*:', re.I),               # **Note:** markdown-style
]


def _strip_footnotes(text: str) -> str:
    """Remove LLM meta-commentary footnotes from the end of a draft.

    Works backwards from the end: once a footnote line is detected,
    everything from that point to the end is removed.
    """
    if not text:
        return text
    lines = text.rstrip().split('\n')
    cut_idx = len(lines)
    # Scan backwards to find where footnotes start
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            continue  # skip blank lines at the end
        if any(p.search(stripped) for p in _FOOTNOTE_PATTERNS):
            cut_idx = i
        else:
            # Found a non-footnote, non-blank line — stop scanning
            break
    # Also check if there's a standalone separator line followed by footnotes
    if cut_idx < len(lines) and cut_idx > 0:
        prev = lines[cut_idx - 1].strip()
        if re.match(r'^[-—–]{2,}$', prev):
            cut_idx = cut_idx - 1
    result = '\n'.join(lines[:cut_idx]).rstrip()
    return result


def finalize_node(state: DraftState) -> dict:
    """Strip footnotes, ensure signature, and return the polished final draft."""
    draft = state.get("draft") or ""
    roles = state.get("roles") or {}

    # ── Strip LLM meta-commentary / footnotes ──
    draft = _strip_footnotes(draft)

    # ── Try to attach signature_name if not already present ──
    signature_name = ""
    try:
        persisted = load_persisted_style()
        for line in (persisted or "").splitlines():
            if line.strip().upper().startswith("SIGNATURE_NAME:"):
                signature_name = line.split(":", 1)[1].strip()
                break
    except Exception:
        pass

    if signature_name and draft:
        if signature_name.lower() not in draft[-100:].lower():
            draft = draft.rstrip() + f"\n\n{signature_name}"

    return {"final_draft": draft}


# ─── Graph construction ──────────────────────────────────────────────────────

def _build_drafting_graph():
    """Build the drafting graph with the critique-revise cycle.

    plan → draft → critique → (if score < 4) → revise → critique (cycle)
                            → (if score ≥ 4 or revisions exhausted) → finalize → END
    """
    graph = StateGraph(DraftState)

    graph.add_node("plan",     plan_node)
    graph.add_node("draft",    draft_node)
    graph.add_node("critique", critique_node)
    graph.add_node("revise",   revise_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("plan")
    graph.add_edge("plan",    "draft")
    graph.add_edge("draft",   "critique")

    # Conditional edge: revise loop OR finalize
    graph.add_conditional_edges(
        "critique",
        _should_revise,
        {"revise": "revise", "finalize": "finalize"},
    )

    # After revising, go back to critique (the cycle)
    graph.add_edge("revise", "critique")

    graph.add_edge("finalize", END)

    return graph.compile()


_DRAFTING_GRAPH = _build_drafting_graph()


def run_drafting_graph(
    email: dict,
    decision: str = "",
    style_notes: str = "",
) -> str:
    """Run the full plan→draft→critique→(revise?)→finalize pipeline.

    This is a drop-in replacement for the old ``draft_reply()`` function
    that adds a critique-revise quality loop via LangGraph.

    Returns the finalized email body string.
    """
    initial: DraftState = {
        "email": email,
        "decision": decision,
        "style_notes": style_notes,
    }
    try:
        result = _DRAFTING_GRAPH.invoke(initial, config={"configurable": {}})
        final = result.get("final_draft") or result.get("draft") or ""
        revisions = result.get("revision_count", 0)
        score = (result.get("critique") or {}).get("score", "?")
        logger.info(
            "Drafting graph complete — critique score: %s, revisions: %d",
            score, revisions,
        )
        return final
    except Exception as e:
        logger.warning("Drafting graph error, falling back to draft_reply: %s", e)
        # Fallback to the simple drafter
        from agent.drafter import draft_reply
        return draft_reply(email, decision=decision, style_notes=style_notes)
