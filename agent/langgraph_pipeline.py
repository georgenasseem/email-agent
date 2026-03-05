"""LangGraph pipeline for inbox processing with persistent memory.

Key behaviours:
- Startup (DB has emails): load all from DB instantly — no LLM calls.
- Startup (empty DB) / Retrain: wipe all data → fetch 15 newest from Gmail → process.
- Fetch new: grab latest emails from Gmail → store raw → process only unseen ones
              → merge with all stored → display everything.
- All raw + processed data is persisted so the next startup is instant.
"""
import re
import time
from typing import TypedDict

from langgraph.graph import StateGraph, END

from tools.gmail_tools import fetch_emails, fetch_thread, get_profile_email
from agent.memory_store import add_memory, init_db
from agent.email_memory import (
    store_raw_emails,
    store_processed_emails,
    get_processed_email_ids,
    load_all_processed_emails,
    load_all_raw_emails,
    build_email_links,
    wipe_processed_data,
    wipe_all_data,
)
from agent.style_learner import learn_and_persist_style, load_persisted_style
from agent.summarizer import summarize_batch
from agent.categorizer import categorize_emails, categorize_email
from agent.urgent_detector import flag_urgent_emails
from agent.decision_suggester import suggest_decision
from agent.quick_actions_graph import suggest_quick_actions_full
from agent.delegation import decide_delegation
from agent.context_enrichment import enrich_batch

# Number of emails to fetch/process on first-time init and retrain reset
INITIAL_FETCH_COUNT = 15
# Number of emails to grab from Gmail on each normal "Fetch new" click
FETCH_NEW_COUNT = 20


class EmailAgentState(TypedDict, total=False):
    """Core state for the email processing pipeline."""

    query: str
    max_emails: int
    unread_only: bool
    emails: list[dict]
    style_notes: str
    error: str
    # Internal flags
    retrain: bool
    # Parallel enrichment outputs (produced in parallel, merged before summarise)
    thread_ctx_map: dict    # {email_id: thread_context_str}
    related_ctx_map: dict   # {email_id: related_context_str}
    enrich_ctx_map: dict    # {email_id: enriched_context_str}


# Style learning cache: timestamp of last learn
_style_last_learned: float = 0.0
_STYLE_CACHE_SECONDS = 24 * 3600  # 24 hours


# ─── Loading from persistent storage (instant) ─────────────────────────────


def load_from_memory() -> list[dict]:
    """Load ALL previously processed emails from the database, newest first.

    Returns immediately without any LLM calls — used for instant startup.
    """
    init_db()
    return load_all_processed_emails()


# ─── Pipeline nodes ─────────────────────────────────────────────────────────


def fetch_inbox_node(state: EmailAgentState) -> dict:
    """Fetch emails from Gmail and decide which ones need processing.

    Retrain / new-user mode (retrain=True)
    ───────────────────────────────────────
    1. Wipe every table (raw emails, processed results, links, memory).
    2. Fetch the *max_emails* newest from Gmail.
    3. Store them as raw and forward ALL of them for LLM processing.

    Normal fetch mode
    ─────────────────
    1. Fetch the *max_emails* newest from Gmail (picks up new arrivals).
    2. Upsert into raw emails table — existing rows are kept intact.
    3. Forward ONLY emails not yet in email_processed, so the LLM
       pipeline only works on genuinely new messages.
    """
    query = state.get("query") or "in:inbox"
    max_emails = state.get("max_emails", INITIAL_FETCH_COUNT)
    unread_only = state.get("unread_only", False)
    actual_query = f"{query} is:unread" if unread_only and "is:unread" not in query else query

    if state.get("retrain"):
        # ── Retrain / new-user init ──────────────────────────────────
        wipe_all_data()
        fresh = fetch_emails(max_results=max_emails, query=actual_query)
        store_raw_emails(fresh)
        return {"emails": fresh}

    # ── Normal incremental fetch ─────────────────────────────────────
    all_fetched = fetch_emails(max_results=max_emails, query=actual_query)
    store_raw_emails(all_fetched)          # upsert — preserves existing history

    already_processed = get_processed_email_ids()
    new_emails = [e for e in all_fetched if e.get("id") not in already_processed]
    return {"emails": new_emails}


def thread_context_node(state: EmailAgentState) -> dict:
    """Build a thread-context map {email_id: context_str} for each email.

    Runs in parallel with related_context_node and enrich_context_node.
    Outputs to ``thread_ctx_map``; the main ``emails`` list is untouched
    until merge_contexts_node combines all three maps.
    """
    emails = state.get("emails") or []
    if not emails:
        return {"thread_ctx_map": {}}

    ctx_map: dict = {}
    for e in emails:
        tid = e.get("thread_id")
        if not tid or e.get("thread_context"):
            continue
        try:
            thread_messages = fetch_thread(tid)
        except Exception:
            continue
        if not thread_messages:
            continue
        recent = thread_messages[-3:]
        parts = []
        for m in recent:
            sender = m.get("sender", "")
            date = m.get("date", "")
            body_preview = (m.get("clean_body") or m.get("body") or m.get("snippet") or "")[:1500]
            parts.append(f"From: {sender} | {date}\n{body_preview}")
        ctx_map[e["id"]] = "\n\n---\n\n".join(parts)

    return {"thread_ctx_map": ctx_map}


def related_context_node(state: EmailAgentState) -> dict:
    """Build a related-context map {email_id: context_str} for each email.

    Runs in parallel with thread_context_node and enrich_context_node.
    Outputs to ``related_ctx_map``.
    """
    emails = state.get("emails") or []
    if not emails:
        return {"related_ctx_map": {}}

    def sender_domain(e: dict) -> str:
        sender = (e.get("sender") or "").lower()
        if "@" in sender:
            return sender.split("@", 1)[1].strip()
        return ""

    def subject_tokens(e: dict) -> set[str]:
        subj = (e.get("subject") or "").lower()
        tokens = re.findall(r"[a-z0-9]+", subj)
        stop = {"re", "fwd", "fw", "the", "and", "or", "of", "in", "on", "for", "to"}
        return {t for t in tokens if len(t) > 2 and t not in stop}

    domains = [sender_domain(e) for e in emails]
    tokens_list = [subject_tokens(e) for e in emails]
    ctx_map: dict = {}

    for idx, e in enumerate(emails):
        this_dom = domains[idx]
        this_tokens = tokens_list[idx]
        related_snippets: list[str] = []

        for j, other in enumerate(emails):
            if j == idx:
                continue
            same_domain = this_dom and this_dom == domains[j]
            shared_tokens = this_tokens & tokens_list[j]
            if not same_domain and len(shared_tokens) < 2:
                continue

            line = (
                f"From: {other.get('sender', '')} | {other.get('date', '')}\n"
                f"Subject: {other.get('subject', '')}\n"
                f"Summary/snippet: {(other.get('summary') or other.get('snippet') or '')[:200]}"
            )
            related_snippets.append(line)

        if related_snippets:
            ctx_map[e["id"]] = "\n\n=== Related email ===\n\n".join(related_snippets[:3])

    return {"related_ctx_map": ctx_map}


def enrich_context_node(state: EmailAgentState) -> dict:
    """Detect unknown entities and resolve them via DB + Gmail.

    Runs in parallel with thread_context_node and related_context_node.
    Outputs to ``enrich_ctx_map`` — does NOT modify ``emails`` directly.
    """
    emails = state.get("emails") or []
    if not emails:
        return {"enrich_ctx_map": {}}
    enriched = enrich_batch(emails, gmail_search_fn=fetch_emails)
    ctx_map = {e.get("id"): (e.get("enriched_context") or "") for e in enriched if e.get("id")}
    return {"enrich_ctx_map": ctx_map}


def merge_contexts_node(state: EmailAgentState) -> dict:
    """Fan-in barrier: apply all three context maps to the emails list.

    Fires automatically once thread_context, related_context, AND
    enrich_context have all completed their parallel execution.
    """
    emails = list(state.get("emails") or [])
    thread_map  = state.get("thread_ctx_map")  or {}
    related_map = state.get("related_ctx_map") or {}
    enrich_map  = state.get("enrich_ctx_map")  or {}

    for e in emails:
        eid = e.get("id") or ""
        if eid in thread_map and not e.get("thread_context"):
            e["thread_context"] = thread_map[eid]
        if eid in related_map and not e.get("related_context"):
            e["related_context"] = related_map[eid]
        if eid in enrich_map and not e.get("enriched_context"):
            e["enriched_context"] = enrich_map[eid]

    return {"emails": emails}


def learn_style_node(state: EmailAgentState) -> dict:
    """Learn and persist user writing style (cached for 24h)."""
    global _style_last_learned
    now = time.time()
    # Skip re-learning if done recently (unless retrain mode)
    if not state.get("retrain") and (now - _style_last_learned) < _STYLE_CACHE_SECONDS:
        return {"style_notes": load_persisted_style()}
    style = learn_and_persist_style(max_samples=4)
    _style_last_learned = now
    return {"style_notes": style or load_persisted_style()}


def summarize_node(state: EmailAgentState) -> dict:
    """Summarize emails (per-email error isolation)."""
    emails = state.get("emails") or []
    if not emails:
        return {}
    import logging as _slog
    _slogger = _slog.getLogger(__name__)
    try:
        summarized = summarize_batch(emails)
        return {"emails": summarized}
    except Exception as exc:
        _slogger.warning("Batch summarize failed: %s", exc)
        # Fallback: use snippet as summary
        for e in emails:
            if not e.get("summary"):
                e["summary"] = (e.get("snippet") or "")[:200]
        return {"emails": emails}


def categorize_node(state: EmailAgentState) -> dict:
    """Categorize emails (per-email error isolation)."""
    emails = state.get("emails") or []
    if not emails:
        return {}
    import logging as _clog
    _clogger = _clog.getLogger(__name__)
    results = []
    for e in emails:
        try:
            results.append(categorize_email(e))
        except Exception as exc:
            _clogger.warning("Categorize failed for %s: %s", e.get("id", "?"), exc)
            e.setdefault("category", "informational")
            results.append(e)
    return {"emails": results}


def flag_urgent_node(state: EmailAgentState) -> dict:
    """Flag urgent emails."""
    emails = state.get("emails") or []
    if not emails:
        return {}
    flagged = flag_urgent_emails(emails)
    return {"emails": flagged}


def postprocess_categories_node(state: EmailAgentState) -> dict:
    """Promote informational/newsletter → important when the urgent detector flagged needs_action."""
    emails = state.get("emails") or []
    if not emails:
        return {}

    # Check which categories are actually enabled
    try:
        from agent.email_memory import get_enabled_labels
        _enabled = {lb["slug"] for lb in get_enabled_labels()}
    except Exception:
        _enabled = {"important", "informational", "newsletter"}

    for e in emails:
        if not e.get("needs_action", False):
            continue
        cat_raw = (e.get("category") or "informational").lower()
        main_cat = cat_raw.split(",")[0].strip()
        if main_cat in ("informational", "newsletter"):
            # Promote to important if it's enabled, preserve extra tags
            if "important" in _enabled:
                extras = [t.strip() for t in cat_raw.split(",")[1:] if t.strip()]
                e["category"] = ",".join(["important"] + extras) if extras else "important"

    return {"emails": emails}


def log_memory_node(state: EmailAgentState) -> dict:
    """Log email categories into memory for future auto-categories."""
    emails = state.get("emails") or []
    if not emails:
        return {}
    try:
        user_email = get_profile_email()
    except Exception:
        return {}
    for e in emails:
        cat = e.get("category")
        subj = e.get("subject", "")
        if cat:
            # Log the main category (first part of comma-separated)
            main_cat = cat.split(",")[0].strip()
            try:
                add_memory(user_email=user_email, kind="category", key=main_cat, value=subj, source="categorizer")
            except Exception:
                continue
    return {}


def suggest_decision_node(state: EmailAgentState) -> dict:
    """Add decision options to each email using the dedicated quick-actions graph."""
    emails = state.get("emails") or []
    if not emails:
        return {}
    import logging
    _logger = logging.getLogger(__name__)
    for e in emails:
        try:
            result = suggest_quick_actions_full(e)
            options = result.get("final_options", [])
            if options:
                e["decision_options"] = options
            else:
                e["decision_options"] = []
            no_action = result.get("no_action_message", "")
            if no_action:
                e["no_action_message"] = no_action
        except Exception as exc:
            _logger.warning("Quick-actions failed for %s: %s", e.get("id", "?"), exc)
            # Leave decision_options absent (None) — NOT [] — so backfill
            # retries on next load instead of silently marking as "no actions".
            continue
    return {"emails": emails}


def tag_action_required_node(state: EmailAgentState) -> dict:
    """Promote important emails with quick actions to 'action-required' category."""
    emails = state.get("emails") or []
    if not emails:
        return {}
    for e in emails:
        cat_raw = (e.get("category") or "informational").lower()
        tags = [t.strip() for t in cat_raw.split(",") if t.strip()]
        main_cat = tags[0] if tags else "informational"
        extras = tags[1:]
        # If main category is 'important' and email has reply-type quick actions → action-required
        if main_cat == "important":
            options = e.get("decision_options") or []
            has_reply_actions = any(
                (isinstance(o, dict) and o.get("type") == "reply") or
                (isinstance(o, str) and o.lower().startswith("reply"))
                for o in options
            ) if options else False
            if has_reply_actions:
                e["category"] = ",".join(["action-required"] + extras) if extras else "action-required"
    return {"emails": emails}


def delegation_decider_node(state: EmailAgentState) -> dict:
    """Attach delegate_to suggestions based on delegation_rules."""
    emails = state.get("emails") or []
    if not emails:
        return {}
    for e in emails:
        try:
            decision = decide_delegation(e)
        except Exception:
            decision = {}
        if decision.get("delegate_to"):
            e["delegate_to"] = decision["delegate_to"]
    return {"emails": emails}


def persist_results_node(state: EmailAgentState) -> dict:
    """Persist all processed results and refresh cross-email links."""
    emails = state.get("emails") or []
    if not emails:
        return {}

    # Store processed data (summaries, categories, decisions, etc.)
    store_processed_emails(emails)

    # Refresh cross-email links — now that summaries exist, linked email
    # data will be richer on next pipeline run.
    try:
        all_emails = load_all_raw_emails()
        build_email_links(all_emails)
    except Exception:
        pass

    return {}


# ─── Graph construction (singleton) ─────────────────────────────────────────


def _should_process(state: EmailAgentState) -> str:
    """Router: skip the entire LLM pipeline when there are no new emails."""
    return "process" if state.get("emails") else "skip"


def _build_email_graph():
    """Build the inbox-processing graph with an early-exit optimisation.

    fetch_inbox
         │
     [router: any new emails?]
      YES │             NO │
          │           ──►──── END  (instant, no LLM cost)
          ▼
    learn_style   ← writing-style cache (24 h); force-refreshed on retrain
      │    │    │
      │    │    └── thread_context  ─┐
      │    └─────── related_context ─┤  PARALLEL fan-out
      └──────────── enrich_context  ─┘
                          │ (all three must finish)
                    merge_contexts
                          │
    summarize  →  categorize  →  flag_urgent  →  postprocess_categories
                                                          │
                                                    log_memory
                                                          │
                                               suggest_decision  (quick-actions sub-graph)
                                                          │
                                               tag_action_required
                                                          │
                                               delegation_decider
                                                          │
                                               persist_results  →  END
    """
    graph = StateGraph(EmailAgentState)

    graph.add_node("fetch_inbox",             fetch_inbox_node)
    graph.add_node("learn_style",             learn_style_node)
    graph.add_node("thread_context",          thread_context_node)
    graph.add_node("related_context",         related_context_node)
    graph.add_node("enrich_context",          enrich_context_node)
    graph.add_node("merge_contexts",          merge_contexts_node)
    graph.add_node("summarize",               summarize_node)
    graph.add_node("categorize",              categorize_node)
    graph.add_node("flag_urgent",             flag_urgent_node)
    graph.add_node("postprocess_categories",  postprocess_categories_node)
    graph.add_node("log_memory",              log_memory_node)
    graph.add_node("suggest_decision",        suggest_decision_node)
    graph.add_node("tag_action_required",     tag_action_required_node)
    graph.add_node("delegation_decider",      delegation_decider_node)
    graph.add_node("persist_results",         persist_results_node)

    graph.set_entry_point("fetch_inbox")

    # Early-exit: skip all LLM nodes if fetch returned no new emails to process
    graph.add_conditional_edges(
        "fetch_inbox",
        _should_process,
        {"process": "learn_style", "skip": END},
    )

    # Parallel fan-out: learn_style → three context nodes simultaneously
    graph.add_edge("learn_style",   "thread_context")
    graph.add_edge("learn_style",   "related_context")
    graph.add_edge("learn_style",   "enrich_context")

    # Fan-in: all three context nodes must complete before merge_contexts fires
    graph.add_edge("thread_context",  "merge_contexts")
    graph.add_edge("related_context", "merge_contexts")
    graph.add_edge("enrich_context",  "merge_contexts")

    graph.add_edge("merge_contexts",         "summarize")
    graph.add_edge("summarize",              "categorize")
    graph.add_edge("categorize",             "flag_urgent")
    graph.add_edge("flag_urgent",            "postprocess_categories")
    graph.add_edge("postprocess_categories", "log_memory")
    graph.add_edge("log_memory",             "suggest_decision")
    graph.add_edge("suggest_decision",       "tag_action_required")
    graph.add_edge("tag_action_required",    "delegation_decider")
    graph.add_edge("delegation_decider",     "persist_results")
    graph.add_edge("persist_results",        END)

    return graph.compile()


# Module-level singleton: built once, reused for every pipeline run
_EMAIL_GRAPH = _build_email_graph()


def run_email_pipeline(
    query: str = "in:inbox",
    max_emails: int = FETCH_NEW_COUNT,
    unread_only: bool = False,
    retrain: bool = False,
) -> EmailAgentState:
    """Run the inbox processing pipeline and return the final state.

    Normal (fetch) mode
    ───────────────────
    - Fetches the *max_emails* newest from Gmail (default FETCH_NEW_COUNT=20).
    - Stores all as raw (upsert — preserves history).
    - Processes ONLY emails that haven't been processed yet.
    - Merges newly processed emails with ALL stored emails so the caller
      always receives a complete, up-to-date inbox view.

    Retrain / new-user mode  (retrain=True)
    ───────────────────────────────────────
    - Wipes every table in the database.
    - Fetches *max_emails* (default INITIAL_FETCH_COUNT=15) newest emails.
    - Processes and stores all of them fresh.
    - Returns exactly those emails (clean slate — no stale history).
    """
    initial: EmailAgentState = {
        "query": query,
        "max_emails": max_emails,
        "unread_only": unread_only,
        "retrain": retrain,
    }
    final_state = _EMAIL_GRAPH.invoke(initial, config={"configurable": {}})

    if retrain:
        # State already contains exactly the freshly processed emails.
        return final_state

    # Normal fetch: merge newly processed emails with everything in the DB
    # so the UI always shows the full inbox, not just this run's batch.
    all_stored = load_all_processed_emails()
    newly_processed_ids = {e["id"] for e in (final_state.get("emails") or [])}

    merged: list[dict] = list(final_state.get("emails") or [])
    for e in all_stored:
        if e["id"] not in newly_processed_ids:
            merged.append(e)

    final_state["emails"] = merged
    return final_state
