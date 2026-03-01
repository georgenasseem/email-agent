"""Persistent email memory: store, load, link, and query emails across sessions.

This module is the "brain" of the email agent. It:
1. Stores raw fetched emails so they never need re-fetching.
2. Stores LLM-processed results (summary, category, etc.) so they never need reprocessing.
3. Builds cross-email concept links (same sender, same domain, shared keywords).
4. Provides rich memory context for decision suggestions and drafting.
"""
import json
import re
from typing import List, Optional, Set

from agent.memory_store import get_connection, init_db, get_memory_entries

# ─── Shared stop words (used by subject tokenizers & memory matching) ──────
STOP_WORDS = frozenset({
    "re", "fwd", "fw", "the", "and", "or", "of", "in", "on", "for", "to",
    "is", "a", "an", "your", "this", "that", "with", "from", "are", "was",
    "has", "have", "we", "you", "our", "can", "will", "not", "been", "but",
})


# ─── Raw email storage ──────────────────────────────────────────────────────


def store_raw_email(email: dict) -> None:
    """Persist a single raw email dict into the emails table (upsert)."""
    init_db()
    gid = email.get("id")
    if not gid:
        return
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO emails (gmail_id, thread_id, subject, sender, date, internal_date, body, clean_body, body_html, snippet)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(gmail_id) DO UPDATE SET
                thread_id = excluded.thread_id,
                subject   = excluded.subject,
                sender    = excluded.sender,
                date      = excluded.date,
                internal_date = excluded.internal_date,
                body      = excluded.body,
                clean_body = excluded.clean_body,
                body_html = excluded.body_html,
                snippet   = excluded.snippet
            """,
            (
                gid,
                email.get("thread_id", ""),
                email.get("subject", ""),
                email.get("sender", ""),
                email.get("date", ""),
                email.get("internal_date", 0),
                email.get("body", ""),
                email.get("clean_body", ""),
                email.get("body_html", ""),
                email.get("snippet", ""),
            ),
        )
        conn.commit()


def store_raw_emails(emails: List[dict]) -> None:
    """Persist a batch of raw emails in a single transaction."""
    if not emails:
        return
    init_db()
    with get_connection() as conn:
        cur = conn.cursor()
        cur.executemany(
            """
            INSERT INTO emails (gmail_id, thread_id, subject, sender, date, internal_date, body, clean_body, body_html, snippet)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(gmail_id) DO UPDATE SET
                thread_id = excluded.thread_id,
                subject   = excluded.subject,
                sender    = excluded.sender,
                date      = excluded.date,
                internal_date = excluded.internal_date,
                body      = excluded.body,
                clean_body = excluded.clean_body,
                body_html = excluded.body_html,
                snippet   = excluded.snippet
            """,
            [
                (
                    e.get("id"),
                    e.get("thread_id", ""),
                    e.get("subject", ""),
                    e.get("sender", ""),
                    e.get("date", ""),
                    e.get("internal_date", 0),
                    e.get("body", ""),
                    e.get("clean_body", ""),
                    e.get("body_html", ""),
                    e.get("snippet", ""),
                )
                for e in emails if e.get("id")
            ],
        )
        conn.commit()


def get_stored_email_ids() -> Set[str]:
    """Return set of gmail_ids already stored."""
    init_db()
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT gmail_id FROM emails")
        return {row[0] for row in cur.fetchall()}


def load_all_raw_emails() -> List[dict]:
    """Load all stored raw emails, newest first."""
    init_db()
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM emails ORDER BY internal_date DESC")
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    result = []
    for row in rows:
        d = dict(zip(cols, row))
        d["id"] = d.pop("gmail_id")
        result.append(d)
    return result


# ─── Processed email storage ────────────────────────────────────────────────


def store_processed_email(email: dict) -> None:
    """Persist LLM-processed fields for an email (upsert)."""
    init_db()
    gid = email.get("id")
    if not gid:
        return
    options = email.get("decision_options")
    options_json = json.dumps(options) if options else None
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO email_processed (gmail_id, summary, category, needs_action, decision_options_json, thread_context, related_context, enriched_context, delegate_to)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(gmail_id) DO UPDATE SET
                summary              = excluded.summary,
                category             = excluded.category,
                needs_action         = excluded.needs_action,
                decision_options_json = excluded.decision_options_json,
                thread_context       = excluded.thread_context,
                related_context      = excluded.related_context,
                enriched_context     = excluded.enriched_context,
                delegate_to          = excluded.delegate_to,
                processed_at         = CURRENT_TIMESTAMP
            """,
            (
                gid,
                email.get("summary", ""),
                email.get("category", "informational"),
                1 if email.get("needs_action") else 0,
                options_json,
                email.get("thread_context", ""),
                email.get("related_context", ""),
                email.get("enriched_context", ""),
                email.get("delegate_to", ""),
            ),
        )
        conn.commit()


def store_processed_emails(emails: List[dict]) -> None:
    """Persist processed fields for a batch of emails in a single transaction."""
    if not emails:
        return
    init_db()
    rows = []
    for e in emails:
        gid = e.get("id")
        if not gid:
            continue
        options = e.get("decision_options")
        options_json = json.dumps(options) if options else None
        rows.append((
            gid,
            e.get("summary", ""),
            e.get("category", "informational"),
            1 if e.get("needs_action") else 0,
            options_json,
            e.get("thread_context", ""),
            e.get("related_context", ""),
            e.get("enriched_context", ""),
            e.get("delegate_to", ""),
        ))
    if not rows:
        return
    with get_connection() as conn:
        cur = conn.cursor()
        cur.executemany(
            """
            INSERT INTO email_processed (gmail_id, summary, category, needs_action, decision_options_json, thread_context, related_context, enriched_context, delegate_to)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(gmail_id) DO UPDATE SET
                summary              = excluded.summary,
                category             = excluded.category,
                needs_action         = excluded.needs_action,
                decision_options_json = excluded.decision_options_json,
                thread_context       = excluded.thread_context,
                related_context      = excluded.related_context,
                enriched_context     = excluded.enriched_context,
                delegate_to          = excluded.delegate_to,
                processed_at         = CURRENT_TIMESTAMP
            """,
            rows,
        )
        conn.commit()


def get_processed_email_ids() -> Set[str]:
    """Return set of gmail_ids that have processed data."""
    init_db()
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT gmail_id FROM email_processed")
        return {row[0] for row in cur.fetchall()}


def load_all_processed_emails() -> List[dict]:
    """Load all emails with their processed data merged, newest first.

    Joins emails + email_processed so the result matches the runtime email dict format.
    """
    init_db()
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                e.gmail_id, e.thread_id, e.subject, e.sender, e.date,
                e.internal_date, e.body, e.clean_body, e.body_html, e.snippet,
                p.summary, p.category, p.needs_action, p.decision_options_json,
                p.thread_context, p.related_context, p.enriched_context, p.delegate_to
            FROM emails e
            LEFT JOIN email_processed p ON e.gmail_id = p.gmail_id
            WHERE COALESCE(p.archived, 0) = 0
            ORDER BY e.internal_date DESC
            """
        )
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    result = []
    for row in rows:
        d = dict(zip(cols, row))
        # Rename gmail_id → id
        d["id"] = d.pop("gmail_id")
        # Convert needs_action int → bool
        d["needs_action"] = bool(d.get("needs_action"))
        # Parse decision_options JSON
        opts_json = d.pop("decision_options_json", None)
        if opts_json:
            try:
                d["decision_options"] = json.loads(opts_json)
            except Exception:
                d["decision_options"] = []
        result.append(d)
    return result


# ─── Cross-email concept linking ────────────────────────────────────────────


def _sender_domain(email: dict) -> str:
    sender = (email.get("sender") or "").lower()
    if "@" in sender:
        # Handle "Name <email@domain>" format
        if "<" in sender:
            sender = sender.split("<")[1].split(">")[0]
        return sender.split("@", 1)[1].strip().rstrip(">")
    return ""


def _sender_email(email: dict) -> str:
    sender = (email.get("sender") or "").lower()
    if "<" in sender and ">" in sender:
        return sender.split("<")[1].split(">")[0].strip()
    return sender.strip()


def _subject_tokens(email: dict) -> set:
    subj = (email.get("subject") or "").lower()
    # Remove Re: Fwd: etc.
    subj = re.sub(r"^(re|fwd|fw)\s*:\s*", "", subj, flags=re.IGNORECASE)
    tokens = re.findall(r"[a-z0-9]+", subj)
    return {t for t in tokens if len(t) > 2 and t not in STOP_WORDS}


def build_email_links(emails: List[dict]) -> None:
    """Build cross-email concept links and persist them.

    Link types:
    - same_sender: same email address
    - same_domain: same sender domain
    - shared_subject: overlapping subject keywords (≥2 tokens)
    - same_thread: same thread_id
    """
    init_db()
    if not emails:
        return

    # Precompute features
    features = []
    for e in emails:
        features.append({
            "id": e.get("id"),
            "domain": _sender_domain(e),
            "sender": _sender_email(e),
            "tokens": _subject_tokens(e),
            "thread_id": e.get("thread_id", ""),
        })

    links: list[tuple] = []
    for i, fa in enumerate(features):
        for j, fb in enumerate(features):
            if j <= i:
                continue
            ida, idb = fa["id"], fb["id"]
            if not ida or not idb:
                continue

            # Same thread
            if fa["thread_id"] and fa["thread_id"] == fb["thread_id"]:
                links.append((ida, idb, "same_thread", 1.0))

            # Same sender
            if fa["sender"] and fa["sender"] == fb["sender"]:
                links.append((ida, idb, "same_sender", 1.0))
            elif fa["domain"] and fa["domain"] == fb["domain"]:
                links.append((ida, idb, "same_domain", 0.5))

            # Shared subject tokens
            shared = fa["tokens"] & fb["tokens"]
            if len(shared) >= 2:
                strength = min(1.0, len(shared) * 0.3)
                links.append((ida, idb, "shared_subject", strength))

    if not links:
        return

    with get_connection() as conn:
        cur = conn.cursor()
        cur.executemany(
            """
            INSERT INTO email_links (email_id_a, email_id_b, link_type, strength)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(email_id_a, email_id_b, link_type) DO UPDATE SET
                strength = excluded.strength
            """,
            links,
        )
        conn.commit()


def get_linked_emails(gmail_id: str, limit: int = 10) -> List[dict]:
    """Get emails linked to the given email, ordered by link strength.

    Returns list of dicts with the linked email data + link_type + strength.
    """
    init_db()
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT e.gmail_id, e.subject, e.sender, e.date, e.clean_body, e.snippet,
                   p.summary, p.category, p.needs_action, p.decision_options_json,
                   l.link_type, l.strength
            FROM email_links l
            JOIN emails e ON (
                (l.email_id_a = ? AND e.gmail_id = l.email_id_b)
                OR (l.email_id_b = ? AND e.gmail_id = l.email_id_a)
            )
            LEFT JOIN email_processed p ON e.gmail_id = p.gmail_id
            ORDER BY l.strength DESC
            LIMIT ?
            """,
            (gmail_id, gmail_id, limit),
        )
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    result = []
    for row in rows:
        d = dict(zip(cols, row))
        d["id"] = d.pop("gmail_id")
        d["needs_action"] = bool(d.get("needs_action"))
        opts_json = d.pop("decision_options_json", None)
        if opts_json:
            try:
                d["decision_options"] = json.loads(opts_json)
            except Exception:
                pass
        result.append(d)
    return result


# ─── Memory context builder ────────────────────────────────────────────────


def build_memory_context(email: dict, max_linked: int = 5) -> str:
    """Build a rich memory context string for an email, drawing from:
    1. Cross-email links (same sender, same domain, shared subject tokens)
    2. Persistent memory entries (category history from previous runs)
    3. Knowledge base entries for entities mentioned in the email
    4. Sender interaction summary (total count, recent topics)
    5. Active todo items related to this thread/sender

    This is the "brain" output that gets injected into LLM prompts.
    """
    parts = []

    gid = email.get("id")

    # 1. Linked emails
    if gid:
        linked = get_linked_emails(gid, limit=max_linked)
        if linked:
            link_lines = []
            for le in linked:
                link_type = le.get("link_type", "related")
                summary = le.get("summary") or le.get("snippet", "")[:200]
                sender = le.get("sender", "")
                subject = le.get("subject", "")
                date = le.get("date", "")
                category = le.get("category", "")
                link_lines.append(
                    f"[{link_type}] {sender} | {date}\n"
                    f"  Subject: {subject} | Category: {category}\n"
                    f"  Summary: {summary}"
                )
            parts.append("Linked emails:\n" + "\n".join(link_lines))

    # 2. Sender interaction summary
    try:
        sender_email = _sender_email(email)
        if sender_email:
            history = get_sender_history(sender_email, exclude_id=gid or "", limit=5)
            if history:
                total_count = len(history)
                topics = [h.get("subject", "") for h in history[:3] if h.get("subject")]
                cats = [h.get("category", "") for h in history if h.get("category")]
                most_common_cat = max(set(cats), key=cats.count) if cats else "unknown"
                sender_summary = (
                    f"Sender interaction: {total_count} previous emails, "
                    f"usually categorized as '{most_common_cat}'. "
                    f"Recent topics: {'; '.join(topics[:3])}"
                )
                parts.append(sender_summary)
    except Exception:
        pass

    # 3. Category history from the memory table
    try:
        cat_memories = get_memory_entries(kind="category", limit=15)
        if cat_memories:
            email_subj = (email.get("subject") or "").lower()
            email_tokens = set(re.findall(r"[a-z0-9]{3,}", email_subj))
            email_tokens -= STOP_WORDS
            relevant = []
            for mem in cat_memories:
                mem_subj = (mem.get("value") or "").lower()
                mem_tokens = set(re.findall(r"[a-z0-9]{3,}", mem_subj))
                mem_tokens -= STOP_WORDS
                shared = email_tokens & mem_tokens
                if len(shared) >= 1:
                    relevant.append(f"  \"{mem.get('value', '')}\" → {mem.get('key', '')}")
            if relevant:
                parts.append("Category history:\n" + "\n".join(relevant[:5]))
    except Exception:
        pass

    # 4. Knowledge base entries for sender
    try:
        sender = email.get("sender", "")
        if sender:
            sender_name = sender.split("<")[0].strip().strip('"').strip("'")
            if sender_name and len(sender_name) > 2:
                kb_entries = lookup_knowledge(sender_name)
                for kb in kb_entries[:3]:
                    parts.append(f"Known: {kb['entity']} ({kb['entity_type']}): {kb['info']}")
    except Exception:
        pass

    # 5. Active todo items that might relate to this thread/sender
    try:
        todos = get_todo_items()
        if todos and gid:
            related_todos = [t for t in todos if t.get("email_id") == gid]
            if related_todos:
                todo_lines = [f"  - {t['task']}" for t in related_todos[:3]]
                parts.append("Active todos for this email:\n" + "\n".join(todo_lines))
    except Exception:
        pass

    if not parts:
        return ""

    return "=== Memory Context ===\n\n" + "\n\n".join(parts)


def get_sender_history(sender_email: str, exclude_id: str = "", limit: int = 5) -> List[dict]:
    """Get previous emails from the same sender address."""
    init_db()
    sender_pattern = f"%{sender_email}%"
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT e.gmail_id, e.subject, e.date, e.snippet,
                   p.summary, p.category
            FROM emails e
            LEFT JOIN email_processed p ON e.gmail_id = p.gmail_id
            WHERE e.sender LIKE ? AND e.gmail_id != ?
            ORDER BY e.internal_date DESC
            LIMIT ?
            """,
            (sender_pattern, exclude_id or "", limit),
        )
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    result = []
    for row in rows:
        d = dict(zip(cols, row))
        d["id"] = d.pop("gmail_id")
        result.append(d)
    return result


# ─── Retrain / wipe processed data ─────────────────────────────────────────


def wipe_processed_data() -> None:
    """Delete all processed (LLM-generated) data but keep raw emails.

    Used by the 'Retrain' button so emails can be reprocessed with new prompts.
    """
    init_db()
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM email_processed")
        cur.execute("DELETE FROM email_links")
        cur.execute("DELETE FROM memory WHERE kind IN ('category', 'delegation_suggestion')")
        conn.commit()


def get_email_count() -> int:
    """Return total number of stored raw emails."""
    init_db()
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM emails")
        return cur.fetchone()[0]


def get_processed_count() -> int:
    """Return total number of processed emails."""
    init_db()
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM email_processed")
        return cur.fetchone()[0]


# ─── Todo items ─────────────────────────────────────────────────────────────


def add_todo_item(task: str, email_id: str = "") -> int:
    """Add a todo item. Returns the new item's ID."""
    init_db()
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO todo_items (email_id, task) VALUES (?, ?)",
            (email_id or None, task),
        )
        conn.commit()
        return cur.lastrowid


def get_todo_items() -> List[dict]:
    """Return all todo items, newest first."""
    init_db()
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, email_id, task, created_at FROM todo_items ORDER BY created_at DESC")
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def remove_todo_item(item_id: int) -> None:
    """Remove a todo item by ID."""
    init_db()
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM todo_items WHERE id = ?", (item_id,))
        conn.commit()


# ─── Knowledge base ─────────────────────────────────────────────────────────


def upsert_knowledge(entity: str, entity_type: str, info: str, source: str = "", confidence: float = 0.5) -> None:
    """Insert or update a knowledge base entry."""
    init_db()
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO knowledge_base (entity, entity_type, info, source, confidence)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(entity, entity_type) DO UPDATE SET
                info = excluded.info,
                source = excluded.source,
                confidence = MAX(knowledge_base.confidence, excluded.confidence),
                updated_at = CURRENT_TIMESTAMP
            """,
            (entity.lower().strip(), entity_type.lower().strip(), info, source, confidence),
        )
        conn.commit()


def lookup_knowledge(entity: str) -> List[dict]:
    """Look up everything we know about an entity (case-insensitive partial match)."""
    init_db()
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT entity, entity_type, info, source, confidence FROM knowledge_base WHERE entity LIKE ? ORDER BY confidence DESC",
            (f"%{entity.lower().strip()}%",),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def get_all_knowledge() -> List[dict]:
    """Return everything in the knowledge base."""
    init_db()
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT entity, entity_type, info, source, confidence FROM knowledge_base ORDER BY updated_at DESC")
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def search_emails_for_entity(entity: str, limit: int = 5) -> List[dict]:
    """Search stored emails for mentions of an entity. Returns matching emails."""
    init_db()
    pattern = f"%{entity}%"
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT e.gmail_id, e.subject, e.sender, e.date, e.snippet,
                   p.summary, p.category
            FROM emails e
            LEFT JOIN email_processed p ON e.gmail_id = p.gmail_id
            WHERE e.subject LIKE ? OR e.body LIKE ? OR e.clean_body LIKE ? OR e.sender LIKE ?
            ORDER BY e.internal_date DESC
            LIMIT ?
            """,
            (pattern, pattern, pattern, pattern, limit),
        )
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    result = []
    for row in rows:
        d = dict(zip(cols, row))
        d["id"] = d.pop("gmail_id")
        result.append(d)
    return result


# ─── Custom category labels & rules (Phase 6) ──────────────────────────────

# Default system categories that are always available alongside user labels
SYSTEM_CATEGORIES = ["important", "informational", "newsletter"]

DEFAULT_LABEL_COLORS = {
    "important": "#f59e0b",
    "informational": "#94a3b8",
    "newsletter": "#8b5cf6",
}


def _slugify(text: str) -> str:
    """Convert display name to a URL-safe slug."""
    s = text.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")[:60]


def ensure_default_labels() -> None:
    """Seed the category_labels table with the 3 system categories if needed.

    Handles migration from various legacy systems:
      low → informational
      urgent → important
      action-needed → important
      fyi → informational
      normal → informational
    """
    init_db()
    with get_connection() as conn:
        cur = conn.cursor()

        # ── Migrate legacy slugs (labels + rules + emails) ──
        _old_to_new = {
            "low": "informational",
            "urgent": "important",
            "action-needed": "important",
            "fyi": "informational",
            "normal": "informational",
        }
        for old_slug, new_slug in _old_to_new.items():
            cur.execute("SELECT slug FROM category_labels WHERE slug = ?", (old_slug,))
            if cur.fetchone():
                cur.execute("DELETE FROM category_labels WHERE slug = ?", (old_slug,))
            # Always migrate rules and emails regardless of whether label row existed
            cur.execute("UPDATE category_rules SET label_slug = ? WHERE label_slug = ?", (new_slug, old_slug))
            cur.execute("UPDATE email_processed SET category = ? WHERE category = ?", (new_slug, old_slug))
        conn.commit()

        # ── Seed defaults (only on fresh DB — don’t re-create deleted labels) ──
        cur.execute("SELECT COUNT(*) FROM category_labels")
        _has_labels = cur.fetchone()[0] > 0

        # ── Always ensure system categories exist ──
        cur.execute("SELECT MAX(position) FROM category_labels")
        _max_pos_row = cur.fetchone()
        _next_pos = (_max_pos_row[0] or -1) + 1 if _max_pos_row and _max_pos_row[0] is not None else 0
        defaults = [
            ("important", "Important", DEFAULT_LABEL_COLORS["important"], "Requires a response or task", 1),
            ("informational", "Informational", DEFAULT_LABEL_COLORS["informational"], "Informational only, no action needed", 1),
            ("newsletter", "Newsletter", DEFAULT_LABEL_COLORS["newsletter"], "Marketing & newsletters", 1),
        ]
        for i, (slug, display, color, desc, enabled) in enumerate(defaults):
            cur.execute("SELECT slug FROM category_labels WHERE slug = ?", (slug,))
            if not cur.fetchone():
                cur.execute(
                    "INSERT INTO category_labels (slug, display_name, color, description, enabled, position) VALUES (?, ?, ?, ?, ?, ?)",
                    (slug, display, color, desc, enabled, _next_pos + i),
                )
        conn.commit()


def get_all_labels() -> List[dict]:
    """Return all category labels, ordered by position."""
    init_db()
    ensure_default_labels()
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, slug, display_name, color, description, enabled, position FROM category_labels ORDER BY position, id"
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def get_enabled_labels() -> List[dict]:
    """Return only enabled category labels."""
    return [lb for lb in get_all_labels() if lb.get("enabled")]


def get_label_by_slug(slug: str) -> Optional[dict]:
    """Return a single label by slug, or None."""
    init_db()
    ensure_default_labels()
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, slug, display_name, color, description, enabled, position FROM category_labels WHERE slug = ?",
            (slug,),
        )
        row = cur.fetchone()
        if not row:
            return None
        cols = [d[0] for d in cur.description]
        return dict(zip(cols, row))


def create_label(display_name: str, color: str = "#94a3b8", description: str = "") -> dict:
    """Create a new custom category label. Returns the label dict.

    If a label with the same slug already exists, returns the existing one
    instead of raising an error.
    """
    init_db()
    ensure_default_labels()
    slug = _slugify(display_name)
    if not slug:
        raise ValueError("Label name cannot be empty")
    # Check for existing label with same slug
    existing = get_label_by_slug(slug)
    if existing:
        return existing
    with get_connection() as conn:
        cur = conn.cursor()
        # Get next position
        cur.execute("SELECT COALESCE(MAX(position), -1) + 1 FROM category_labels")
        pos = cur.fetchone()[0]
        cur.execute(
            "INSERT INTO category_labels (slug, display_name, color, description, enabled, position) VALUES (?, ?, ?, ?, 1, ?)",
            (slug, display_name.strip(), color, description.strip(), pos),
        )
        conn.commit()
        return {"id": cur.lastrowid, "slug": slug, "display_name": display_name.strip(), "color": color, "description": description.strip(), "enabled": 1, "position": pos}


def update_label(slug: str, display_name: str = None, color: str = None, description: str = None, enabled: int = None) -> None:
    """Update fields of an existing label."""
    init_db()
    sets = []
    vals = []
    if display_name is not None:
        sets.append("display_name = ?")
        vals.append(display_name.strip())
    if color is not None:
        sets.append("color = ?")
        vals.append(color)
    if description is not None:
        sets.append("description = ?")
        vals.append(description.strip())
    if enabled is not None:
        sets.append("enabled = ?")
        vals.append(enabled)
    if not sets:
        return
    sets.append("updated_at = CURRENT_TIMESTAMP")
    vals.append(slug)
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(f"UPDATE category_labels SET {', '.join(sets)} WHERE slug = ?", vals)
        conn.commit()


def delete_label(slug: str) -> None:
    """Delete a category label. Also cleans up any rules pointing to it."""
    init_db()
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM category_rules WHERE label_slug = ?", (slug,))
        cur.execute("DELETE FROM category_labels WHERE slug = ?", (slug,))
        conn.commit()


def merge_labels(source_slug: str, target_slug: str) -> int:
    """Merge source label into target: reassign all rules, then delete source.
    Returns the number of rules reassigned."""
    if source_slug == target_slug:
        return 0
    init_db()
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("UPDATE category_rules SET label_slug = ?, updated_at = CURRENT_TIMESTAMP WHERE label_slug = ?", (target_slug, source_slug))
        changed = cur.rowcount
        # Also update email_processed.category
        cur.execute("UPDATE email_processed SET category = ? WHERE category = ?", (target_slug, source_slug))
        conn.commit()
    # Delete source (unless it's a system category, in which case just disable)
    if source_slug in SYSTEM_CATEGORIES:
        update_label(source_slug, enabled=0)
    else:
        delete_label(source_slug)
    return changed


# ─── Category rules (sender / subject → label mapping) ─────────────────────


def get_all_rules() -> List[dict]:
    """Return all category rules."""
    init_db()
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, match_type, match_value, label_slug, hits FROM category_rules ORDER BY hits DESC"
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def upsert_rule(match_type: str, match_value: str, label_slug: str) -> None:
    """Create or update a category rule. match_type is 'sender' or 'subject_keyword'."""
    init_db()
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO category_rules (match_type, match_value, label_slug, hits)
            VALUES (?, ?, ?, 1)
            ON CONFLICT(match_type, match_value) DO UPDATE SET
                label_slug = excluded.label_slug,
                hits = category_rules.hits + 1,
                updated_at = CURRENT_TIMESTAMP
            """,
            (match_type, match_value.lower().strip(), label_slug),
        )
        conn.commit()


def _parse_category_tags(category: str) -> tuple[str, list[str]]:
    """Parse a comma-separated category string into (main_category, [extra_tags])."""
    parts = [p.strip() for p in (category or "informational").split(",") if p.strip()]
    main = parts[0] if parts else "informational"
    extras = parts[1:] if len(parts) > 1 else []
    return main, extras


def _build_category_string(main: str, extras: list[str]) -> str:
    """Build a comma-separated category string from main + extras."""
    parts = [main] + [e for e in extras if e != main]
    return ",".join(parts)


def apply_rule_to_existing_emails(match_type: str, match_value: str, label_slug: str) -> int:
    """Scan all stored emails and apply *label_slug* where the rule matches.

    For system categories (important/informational/newsletter): replaces the main category.
    For user-defined tags: adds the tag alongside the existing main category
    (e.g. "informational" -> "informational,hackathon").
    Returns the number of emails updated.
    """
    init_db()
    match_value = match_value.lower().strip()
    is_system = label_slug in SYSTEM_CATEGORIES
    updated = 0
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT e.gmail_id, e.subject, e.sender, p.category
            FROM emails e
            LEFT JOIN email_processed p ON e.gmail_id = p.gmail_id
            """
        )
        rows = cur.fetchall()
        for gmail_id, subject, sender, category in rows:
            category = (category or "informational").strip()
            main_cat, extras = _parse_category_tags(category)

            # Skip if already tagged
            if is_system and main_cat == label_slug:
                continue
            if not is_system and label_slug in extras:
                continue

            # Check if the rule matches this email
            matched = False
            if match_type == "subject_keyword":
                if match_value in (subject or "").lower():
                    matched = True
            elif match_type == "sender":
                s = (sender or "").lower()
                addr = s.split("<")[1].split(">")[0].strip() if "<" in s and ">" in s else (s.strip() if "@" in s else "")
                matched = addr == match_value
            elif match_type == "sender_domain":
                s = (sender or "").lower()
                addr = s.split("<")[1].split(">")[0].strip() if "<" in s and ">" in s else (s.strip() if "@" in s else "")
                if addr and "@" in addr:
                    matched = addr.split("@", 1)[1] == match_value

            if matched:
                if is_system:
                    new_cat = _build_category_string(label_slug, extras)
                else:
                    new_cat = _build_category_string(main_cat, extras + [label_slug])
                cur.execute(
                    "UPDATE email_processed SET category = ? WHERE gmail_id = ?",
                    (new_cat, gmail_id),
                )
                updated += 1
        conn.commit()
    return updated


def delete_rule(rule_id: int) -> None:
    """Delete a category rule by ID."""
    init_db()
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM category_rules WHERE id = ?", (rule_id,))
        conn.commit()


def match_rules_for_email(email: dict) -> Optional[str]:
    """Check email against category rules. Returns first matching label slug, or None.

    Priority: subject keyword match (content-first) > sender domain match > exact sender match.
    Content-based matching is prioritized because the same sender can send different
    types of emails — the subject content is a better signal.
    """
    init_db()
    sender = (email.get("sender") or "").lower()
    sender_email_addr = ""
    if "<" in sender and ">" in sender:
        sender_email_addr = sender.split("<")[1].split(">")[0].strip()
    elif "@" in sender:
        sender_email_addr = sender.strip()

    sender_domain = ""
    if sender_email_addr and "@" in sender_email_addr:
        sender_domain = sender_email_addr.split("@", 1)[1]

    subject = (email.get("subject") or "").lower()

    # Get all enabled label slugs
    enabled_slugs = {lb["slug"] for lb in get_enabled_labels()}

    with get_connection() as conn:
        cur = conn.cursor()

        # 1. Subject keyword match (content-first, ordered by hits)
        cur.execute(
            "SELECT match_value, label_slug, hits FROM category_rules WHERE match_type = 'subject_keyword' ORDER BY hits DESC"
        )
        for kw_row in cur.fetchall():
            kw = kw_row[0]
            if kw and kw in subject and kw_row[1] in enabled_slugs:
                return kw_row[1]

        # 2. Exact sender match (fallback)
        if sender_email_addr:
            cur.execute(
                "SELECT label_slug FROM category_rules WHERE match_type = 'sender' AND match_value = ? LIMIT 1",
                (sender_email_addr,),
            )
            row = cur.fetchone()
            if row and row[0] in enabled_slugs:
                return row[0]

        # 3. Domain match (lowest priority)
        if sender_domain:
            cur.execute(
                "SELECT label_slug FROM category_rules WHERE match_type = 'sender_domain' AND match_value = ? LIMIT 1",
                (sender_domain,),
            )
            row = cur.fetchone()
            if row and row[0] in enabled_slugs:
                return row[0]

    return None


def match_all_rules_for_email(email: dict) -> list[str]:
    """Return ALL matching rule slugs for an email (not just the first).

    Used to collect all user-defined tags that apply, so emails can have
    a system category plus multiple user tags (e.g. 'informational,hackathon').
    """
    init_db()
    sender = (email.get("sender") or "").lower()
    sender_email_addr = ""
    if "<" in sender and ">" in sender:
        sender_email_addr = sender.split("<")[1].split(">")[0].strip()
    elif "@" in sender:
        sender_email_addr = sender.strip()

    sender_domain = ""
    if sender_email_addr and "@" in sender_email_addr:
        sender_domain = sender_email_addr.split("@", 1)[1]

    subject = (email.get("subject") or "").lower()
    enabled_slugs = {lb["slug"] for lb in get_enabled_labels()}

    matched: list[str] = []
    seen: set[str] = set()

    with get_connection() as conn:
        cur = conn.cursor()

        # Subject keyword matches
        cur.execute(
            "SELECT match_value, label_slug FROM category_rules WHERE match_type = 'subject_keyword' ORDER BY hits DESC"
        )
        for kw, slug in cur.fetchall():
            if kw and kw in subject and slug in enabled_slugs and slug not in seen:
                matched.append(slug)
                seen.add(slug)

        # Exact sender match
        if sender_email_addr:
            cur.execute(
                "SELECT label_slug FROM category_rules WHERE match_type = 'sender' AND match_value = ?",
                (sender_email_addr,),
            )
            for (slug,) in cur.fetchall():
                if slug in enabled_slugs and slug not in seen:
                    matched.append(slug)
                    seen.add(slug)

        # Domain match
        if sender_domain:
            cur.execute(
                "SELECT label_slug FROM category_rules WHERE match_type = 'sender_domain' AND match_value = ?",
                (sender_domain,),
            )
            for (slug,) in cur.fetchall():
                if slug in enabled_slugs and slug not in seen:
                    matched.append(slug)
                    seen.add(slug)

    return matched


def record_category_override(email: dict, new_label_slug: str) -> None:
    """When a user overrides an email's category, learn content-based rules from it.

    Extracts meaningful keywords from the email subject and creates
    subject_keyword rules so future emails with similar content get
    the same category automatically.  Sender-based rules are NOT created
    because the same sender can send emails of many different types.
    """
    subject = (email.get("subject") or "").strip()

    # Extract meaningful keywords from the subject (3+ chars, skip stop words)
    _stop_words = {
        "the", "and", "for", "are", "but", "not", "you", "all", "can",
        "had", "her", "was", "one", "our", "out", "has", "his", "how",
        "its", "may", "new", "now", "old", "see", "way", "who", "did",
        "get", "let", "say", "she", "too", "use", "your", "this", "that",
        "with", "have", "from", "they", "been", "said", "each", "which",
        "will", "very", "when", "what", "just", "about", "more", "would",
        "make", "like", "been", "than", "them", "some", "could", "other",
        "into", "then", "these", "also", "please", "hello", "dear", "here",
        "fwd", "fw",
    }
    import re as _re
    words = _re.findall(r"[a-zA-Z]{3,}", subject.lower())
    keywords = [w for w in words if w not in _stop_words]

    # Create a rule for each meaningful keyword (up to 3 most significant)
    for kw in keywords[:3]:
        upsert_rule("subject_keyword", kw, new_label_slug)

    # Also update the stored processed data
    gid = email.get("id")
    if gid:
        init_db()
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE email_processed SET category = ? WHERE gmail_id = ?",
                (new_label_slug, gid),
            )
            conn.commit()


def propose_categories_from_history(min_sender_count: int = 2) -> List[dict]:
    """Analyze stored emails and propose new category labels based on content patterns.

    Groups emails by recurring subject keywords (not by sender) to propose
    content-based categories like "Meetings", "Invoices", "Reports", etc.

    Args:
        min_sender_count: Minimum number of emails a keyword must appear in
                          to be considered for a proposal (re-used param name
                          for backwards compatibility).

    Returns list of dicts: {proposed_name, reason, match_type, match_value, example_subjects, count}.
    """
    init_db()
    proposals = []

    _stop_words = {
        "the", "and", "for", "are", "but", "not", "you", "all", "can",
        "had", "her", "was", "one", "our", "out", "has", "his", "how",
        "its", "may", "new", "now", "old", "see", "way", "who", "did",
        "get", "let", "say", "she", "too", "use", "your", "this", "that",
        "with", "have", "from", "they", "been", "said", "each", "which",
        "will", "very", "when", "what", "just", "about", "more", "would",
        "make", "like", "been", "than", "them", "some", "could", "other",
        "into", "then", "these", "also", "please", "hello", "dear", "here",
        "fwd", "fw", "subject",
    }

    with get_connection() as conn:
        cur = conn.cursor()

        # Fetch all email subjects
        cur.execute("SELECT subject FROM emails WHERE subject IS NOT NULL AND subject != ''")
        rows = cur.fetchall()

        if not rows:
            return []

        # Count keyword frequency across emails
        import re as _re
        from collections import Counter
        keyword_counts: Counter = Counter()
        keyword_subjects: dict = {}  # keyword -> list of example subjects

        for (subject_raw,) in rows:
            subject = (subject_raw or "").strip()
            words = _re.findall(r"[a-zA-Z]{3,}", subject.lower())
            unique_words = set(w for w in words if w not in _stop_words and len(w) >= 4)
            for w in unique_words:
                keyword_counts[w] += 1
                if w not in keyword_subjects:
                    keyword_subjects[w] = []
                if len(keyword_subjects[w]) < 3:
                    keyword_subjects[w].append(subject[:80])

        # Get existing rules to skip already-covered keywords
        cur.execute("SELECT match_value FROM category_rules WHERE match_type = 'subject_keyword'")
        existing_keywords = {row[0].lower() for row in cur.fetchall()}

        # Propose keywords that appear in enough emails and don't have rules yet
        for kw, cnt in keyword_counts.most_common(30):
            if cnt < min_sender_count:
                break
            if kw in existing_keywords:
                continue

            example_subjs = " | ".join(keyword_subjects.get(kw, []))
            proposals.append({
                "proposed_name": kw.capitalize(),
                "reason": f"Keyword '{kw}' appears in {cnt} emails",
                "match_type": "subject_keyword",
                "match_value": kw,
                "example_subjects": example_subjs[:200],
                "count": cnt,
            })

            if len(proposals) >= 15:
                break

    return proposals


def filter_proposals_with_llm(proposals: List[dict]) -> List[dict]:
    """Use a fast LLM call to prune bad category proposals.

    Filters out generic words (e.g. 'week', 'today'), fragments of proper
    nouns (e.g. 'dhabi'), and anything that wouldn't make a sensible email
    category name.  Returns the subset of *proposals* the LLM approved.
    """
    if not proposals:
        return []
    try:
        from agent.llm import get_llm
        from langchain_core.messages import SystemMessage, HumanMessage
        import json as _json

        names = [p["proposed_name"] for p in proposals]
        system = (
            "You are a filter for email category suggestions. "
            "The user will give you a JSON list of proposed category names extracted from email subjects. "
            "Return ONLY a JSON array containing the names that would make GOOD email categories. "
            "Remove:\n"
            "- Generic time words (week, today, tomorrow, monday, etc.)\n"
            "- Common filler words or fragments (update, form, action, check, etc.)\n"
            "- Fragments of proper nouns that are meaningless on their own (e.g. 'dhabi' from 'Abu Dhabi')\n"
            "- Anything too vague to be a useful category\n"
            "Keep names that represent a clear, meaningful topic (e.g. Hackathon, Research, Students, Mentors, Proposal, Internship, Finance).\n"
            "Output ONLY the JSON array. No explanation."
        )
        llm = get_llm(task="decide")  # fast/light model
        raw = llm.invoke(
            [SystemMessage(content=system), HumanMessage(content=_json.dumps(names))],
            max_tokens=300,
        )
        text = (raw if isinstance(raw, str) else raw.content).strip()
        # Extract JSON array
        import re as _re
        text = _re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = _re.sub(r"```\s*$", "", text).strip()
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end > start:
            text = text[start:end + 1]
        good_names = set(n.lower() for n in _json.loads(text) if isinstance(n, str))
        return [p for p in proposals if p["proposed_name"].lower() in good_names]
    except Exception:
        # If LLM fails, return originals unfiltered
        return proposals
