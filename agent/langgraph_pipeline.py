"""LangGraph pipeline for inbox processing with persistent memory.

Key behaviors:
- On startup: loads previously processed emails from DB instantly.
- "Fetch new": only fetches emails not already in the DB, processes only those.
- "Retrain": wipes processed data, reprocesses all stored raw emails.
- All raw + processed data is persisted so the next load is instant.
- Cross-email links are built after every processing run.
"""
import re
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
)
from agent.style_learner import learn_and_persist_style, load_persisted_style
from agent.summarizer import summarize_batch
from agent.categorizer import categorize_emails
from agent.urgent_detector import flag_urgent_emails
from agent.decision_suggester import suggest_decision
from agent.delegation import decide_delegation
from agent.context_enrichment import enrich_batch


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


# ─── Loading from persistent storage (instant) ─────────────────────────────


def load_from_memory() -> list[dict]:
    """Load all previously processed emails from the database.

    Returns the full email list ready for display (no LLM calls needed).
    """
    init_db()
    return load_all_processed_emails()


# ─── Pipeline nodes ─────────────────────────────────────────────────────────


def fetch_inbox_node(state: EmailAgentState) -> dict:
    """Fetch emails from Gmail, skipping those already in the DB.

    Only new (unseen) emails are returned for downstream processing.
    """
    query = state.get("query") or "in:inbox"
    max_emails = state.get("max_emails", 5)
    unread_only = state.get("unread_only", False)

    if state.get("retrain"):
        # Retrain mode: skip Gmail fetch, reprocess ALL stored raw emails
        all_raw = load_all_raw_emails()
        return {"emails": all_raw}

    actual_query = f"{query} is:unread" if unread_only and "is:unread" not in query else query
    all_fetched = fetch_emails(max_results=max_emails, query=actual_query)

    # Persist all raw emails immediately
    store_raw_emails(all_fetched)

    # Build cross-email links EARLY so build_memory_context() returns
    # useful data for downstream nodes (summarize, categorize, decide).
    try:
        all_raw = load_all_raw_emails()
        build_email_links(all_raw)
    except Exception:
        pass

    # Normal mode: only process emails not already in email_processed
    already_processed = get_processed_email_ids()
    new_emails = [e for e in all_fetched if e.get("id") not in already_processed]

    return {"emails": new_emails}


def thread_context_node(state: EmailAgentState) -> dict:
    """Attach compact thread context for each email."""
    emails = state.get("emails") or []
    if not emails:
        return {}

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
        context = "\n\n---\n\n".join(parts)
        e["thread_context"] = context

    return {"emails": emails}


def related_context_node(state: EmailAgentState) -> dict:
    """Attach lightweight cross-email related context for each email."""
    emails = state.get("emails") or []
    if not emails:
        return {}

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
            e["related_context"] = "\n\n=== Related email ===\n\n".join(related_snippets[:3])

    return {"emails": emails}


def enrich_context_node(state: EmailAgentState) -> dict:
    """Detect unknown entities in emails and resolve them via DB + Gmail.

    For each email, extracts referenced people/projects/events/orgs,
    checks the knowledge base, searches local emails, and falls back to
    Gmail search.  Findings are persisted in knowledge_base so they are
    never looked up twice.
    """
    emails = state.get("emails") or []
    if not emails:
        return {}
    enriched = enrich_batch(emails, gmail_search_fn=fetch_emails)
    return {"emails": enriched}


def learn_style_node(state: EmailAgentState) -> dict:
    """Learn and persist user writing style."""
    style = learn_and_persist_style(max_samples=4)
    return {"style_notes": style or load_persisted_style()}


def summarize_node(state: EmailAgentState) -> dict:
    """Summarize emails."""
    emails = state.get("emails") or []
    if not emails:
        return {}
    summarized = summarize_batch(emails)
    return {"emails": summarized}


def categorize_node(state: EmailAgentState) -> dict:
    """Categorize emails."""
    emails = state.get("emails") or []
    if not emails:
        return {}
    categorized = categorize_emails(emails)
    return {"emails": categorized}


def flag_urgent_node(state: EmailAgentState) -> dict:
    """Flag urgent emails."""
    emails = state.get("emails") or []
    if not emails:
        return {}
    flagged = flag_urgent_emails(emails)
    return {"emails": flagged}


def postprocess_categories_node(state: EmailAgentState) -> dict:
    """Deterministically reconcile category with needs_action/urgency signals."""
    emails = state.get("emails") or []
    if not emails:
        return {}

    security_keywords = [
        "security alert", "verify your account", "password reset",
        "suspicious activity", "unusual activity", "unauthorized access",
        "token expired", "token revoked", "access token",
    ]
    hard_deadline_keywords = [
        "by end of day", "by eod", "today", "tonight", "deadline",
    ]

    for e in emails:
        needs_action = e.get("needs_action", False)
        if not needs_action:
            continue
        cat = (e.get("category") or "normal").lower()
        if cat not in ["informational", "newsletter"]:
            continue
        text = " ".join([
            str(e.get("subject", "")).lower(),
            str(e.get("body", "")).lower(),
            str(e.get("snippet", "")).lower(),
        ])
        if any(k in text for k in security_keywords + hard_deadline_keywords):
            e["category"] = "urgent"
        else:
            e["category"] = "important"

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
            try:
                add_memory(user_email=user_email, kind="category", key=cat, value=subj, source="categorizer")
            except Exception:
                continue
    return {}


def suggest_decision_node(state: EmailAgentState) -> dict:
    """Add decision options to each email."""
    emails = state.get("emails") or []
    if not emails:
        return {}
    for e in emails:
        try:
            dec = suggest_decision(e)
            options = dec.get("decision_options") or []
            if options:
                e["decision_options"] = options
        except Exception:
            continue
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


# ─── Graph construction ─────────────────────────────────────────────────────


def build_email_graph():
    """Build the straight-line Process Inbox graph with persistence."""
    graph = StateGraph(EmailAgentState)

    graph.add_node("fetch_inbox", fetch_inbox_node)
    graph.add_node("learn_style", learn_style_node)
    graph.add_node("thread_context", thread_context_node)
    graph.add_node("related_context", related_context_node)
    graph.add_node("enrich_context", enrich_context_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("categorize", categorize_node)
    graph.add_node("flag_urgent", flag_urgent_node)
    graph.add_node("postprocess_categories", postprocess_categories_node)
    graph.add_node("log_memory", log_memory_node)
    graph.add_node("suggest_decision", suggest_decision_node)
    graph.add_node("delegation_decider", delegation_decider_node)
    graph.add_node("persist_results", persist_results_node)

    graph.set_entry_point("fetch_inbox")
    graph.add_edge("fetch_inbox", "learn_style")
    graph.add_edge("learn_style", "thread_context")
    graph.add_edge("thread_context", "related_context")
    graph.add_edge("related_context", "enrich_context")
    graph.add_edge("enrich_context", "summarize")
    graph.add_edge("summarize", "categorize")
    graph.add_edge("categorize", "flag_urgent")
    graph.add_edge("flag_urgent", "postprocess_categories")
    graph.add_edge("postprocess_categories", "log_memory")
    graph.add_edge("log_memory", "suggest_decision")
    graph.add_edge("suggest_decision", "delegation_decider")
    graph.add_edge("delegation_decider", "persist_results")
    graph.add_edge("persist_results", END)

    return graph.compile()


def run_email_pipeline(
    query: str = "in:inbox",
    max_emails: int = 5,
    unread_only: bool = False,
    retrain: bool = False,
) -> EmailAgentState:
    """Run the full inbox processing pipeline.

    - Normal mode: fetches new emails, processes only those, merges with stored data.
    - Retrain mode: wipes processed data, reprocesses everything from stored raw emails.

    Returns final state with ALL emails (previously stored + newly processed).
    """
    if retrain:
        wipe_processed_data()

    graph = build_email_graph()
    initial: EmailAgentState = {
        "query": query,
        "max_emails": max_emails,
        "unread_only": unread_only,
        "retrain": retrain,
    }
    config = {"configurable": {}}
    final_state = graph.invoke(initial, config=config)

    # After processing new emails, load EVERYTHING from DB for display
    all_emails = load_all_processed_emails()

    merged = []
    seen = set()
    # First add all newly processed with full runtime data
    for e in (final_state.get("emails") or []):
        merged.append(e)
        seen.add(e["id"])
    # Then add previously stored emails not in this run
    for e in all_emails:
        if e["id"] not in seen:
            merged.append(e)
            seen.add(e["id"])

    final_state["emails"] = merged
    return final_state
