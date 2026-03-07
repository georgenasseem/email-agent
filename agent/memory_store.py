"""SQLite-backed memory store: user_profile, memory, task_state, delegation_rules, emails, email_processed, email_links."""
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional


DB_PATH = Path(__file__).parent.parent / "data" / "memory.db"

_DB_INITIALIZED = False  # Guard: only run DDL once per process


def _ensure_db_dir() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def get_connection() -> sqlite3.Connection:
    """Yield a SQLite connection to the memory DB."""
    _ensure_db_dir()
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    """Create tables if they don't exist (runs DDL only once per process).

    Call reset_db_initialized() before init_db() if you need to force re-creation
    (e.g. in tests after deleting the DB file).
    """
    global _DB_INITIALIZED
    if _DB_INITIALIZED:
        return
    with get_connection() as conn:
        cur = conn.cursor()

        # Core user profile: keyed by primary email
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS user_profile (
                email TEXT PRIMARY KEY,
                display_name TEXT,
                style_notes TEXT,
                preferences_json TEXT,
                roles_json TEXT,
                working_hours_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # Generic memory table for long-term notes, preferences, categories, etc.
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_email TEXT NOT NULL,
                kind TEXT NOT NULL,
                key TEXT,
                value TEXT,
                source TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_email) REFERENCES user_profile(email) ON DELETE CASCADE
            )
            """
        )

        # Task state for long-running flows or schedulers
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS task_state (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_name TEXT NOT NULL,
                state_json TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # Delegation rules learned from history
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS delegation_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_email TEXT NOT NULL,
                pattern TEXT NOT NULL,
                target_email TEXT NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1,
                weight INTEGER NOT NULL DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_email) REFERENCES user_profile(email) ON DELETE CASCADE
            )
            """
        )

        # ── NEW: Raw email storage ──────────────────────────────────────
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS emails (
                gmail_id TEXT PRIMARY KEY,
                thread_id TEXT,
                subject TEXT,
                sender TEXT,
                date TEXT,
                internal_date INTEGER DEFAULT 0,
                body TEXT,
                clean_body TEXT,
                body_html TEXT DEFAULT '',
                snippet TEXT,
                fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # ── NEW: Processed / LLM-enriched email data ───────────────────
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS email_processed (
                gmail_id TEXT PRIMARY KEY,
                summary TEXT,
                category TEXT,
                needs_action INTEGER DEFAULT 0,
                decision_options_json TEXT,
                thread_context TEXT,
                related_context TEXT,
                enriched_context TEXT DEFAULT '',
                delegate_to TEXT,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (gmail_id) REFERENCES emails(gmail_id) ON DELETE CASCADE
            )
            """
        )

        # ── NEW: Cross-email concept links ──────────────────────────────
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS email_links (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email_id_a TEXT NOT NULL,
                email_id_b TEXT NOT NULL,
                link_type TEXT NOT NULL,
                strength REAL DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (email_id_a) REFERENCES emails(gmail_id) ON DELETE CASCADE,
                FOREIGN KEY (email_id_b) REFERENCES emails(gmail_id) ON DELETE CASCADE,
                UNIQUE(email_id_a, email_id_b, link_type)
            )
            """
        )

        conn.commit()

        # ── Todo items table ────────────────────────────────────────────
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS todo_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email_id TEXT,
                task TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (email_id) REFERENCES emails(gmail_id) ON DELETE SET NULL
            )
            """
        )

        # ── Persistent knowledge base ───────────────────────────────────
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS knowledge_base (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                info TEXT NOT NULL,
                source TEXT,
                confidence REAL DEFAULT 0.5,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(entity, entity_type)
            )
            """
        )

        # ── Custom category labels (Phase 6) ───────────────────────────
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS category_labels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                slug TEXT NOT NULL UNIQUE,
                display_name TEXT NOT NULL,
                color TEXT NOT NULL DEFAULT '#94a3b8',
                description TEXT DEFAULT '',
                enabled INTEGER NOT NULL DEFAULT 1,
                position INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # ── Per-sender / per-subject category rules (Phase 6) ──────────
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS category_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_type TEXT NOT NULL,
                match_value TEXT NOT NULL,
                label_slug TEXT NOT NULL,
                hits INTEGER NOT NULL DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(match_type, match_value),
                FOREIGN KEY (label_slug) REFERENCES category_labels(slug) ON DELETE CASCADE
            )
            """
        )

        conn.commit()

        # ── Migration: add body_html column if missing (for existing DBs) ──
        cur.execute("PRAGMA table_info(emails)")
        existing_cols = {row[1] for row in cur.fetchall()}
        if "body_html" not in existing_cols:
            cur.execute("ALTER TABLE emails ADD COLUMN body_html TEXT DEFAULT ''")
            conn.commit()

        # ── Migration: add enriched_context column if missing ──
        cur.execute("PRAGMA table_info(email_processed)")
        proc_cols = {row[1] for row in cur.fetchall()}
        if "enriched_context" not in proc_cols:
            cur.execute("ALTER TABLE email_processed ADD COLUMN enriched_context TEXT DEFAULT ''")
            conn.commit()

        # ── Migration: add archived column if missing ──
        cur.execute("PRAGMA table_info(email_processed)")
        proc_cols2 = {row[1] for row in cur.fetchall()}
        if "archived" not in proc_cols2:
            cur.execute("ALTER TABLE email_processed ADD COLUMN archived INTEGER DEFAULT 0")
            conn.commit()

        # ── Migration: add opened_at column if missing ──
        cur.execute("PRAGMA table_info(email_processed)")
        proc_cols3 = {row[1] for row in cur.fetchall()}
        if "opened_at" not in proc_cols3:
            cur.execute("ALTER TABLE email_processed ADD COLUMN opened_at TIMESTAMP DEFAULT NULL")
            conn.commit()

        # ── Migration: add snoozed_until column if missing ──
        cur.execute("PRAGMA table_info(email_processed)")
        proc_cols4 = {row[1] for row in cur.fetchall()}
        if "snoozed_until" not in proc_cols4:
            cur.execute("ALTER TABLE email_processed ADD COLUMN snoozed_until TIMESTAMP DEFAULT NULL")
            conn.commit()

        # ── Performance indexes ─────────────────────────────────────────
        cur.execute("CREATE INDEX IF NOT EXISTS idx_emails_internal_date ON emails(internal_date DESC)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_emails_sender ON emails(sender)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_email_links_a ON email_links(email_id_a)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_email_links_b ON email_links(email_id_b)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_memory_kind ON memory(kind)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_todo_created ON todo_items(created_at DESC)")
        conn.commit()

    _DB_INITIALIZED = True


def mark_email_archived(gmail_id: str) -> None:
    """Mark an email as archived in the local DB so it stays hidden on reload."""
    init_db()
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE email_processed SET archived = 1 WHERE gmail_id = ?",
            (gmail_id,),
        )
        conn.commit()


def mark_email_opened(gmail_id: str) -> None:
    """Record when an email was first opened."""
    init_db()
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE email_processed SET opened_at = CURRENT_TIMESTAMP WHERE gmail_id = ? AND opened_at IS NULL",
            (gmail_id,),
        )
        conn.commit()


def get_opened_email_ids() -> set:
    """Return set of gmail_ids that have been opened at least once."""
    init_db()
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT gmail_id FROM email_processed WHERE opened_at IS NOT NULL")
        return {row[0] for row in cur.fetchall()}


def snooze_email(gmail_id: str, until_iso: str) -> None:
    """Snooze an email until a specific ISO-format datetime string."""
    init_db()
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE email_processed SET snoozed_until = ? WHERE gmail_id = ?",
            (until_iso, gmail_id),
        )
        conn.commit()


def unsnooze_email(gmail_id: str) -> None:
    """Clear a snooze so the email appears immediately."""
    init_db()
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE email_processed SET snoozed_until = NULL WHERE gmail_id = ?",
            (gmail_id,),
        )
        conn.commit()


def get_snoozed_email_ids() -> dict:
    """Return {gmail_id: snoozed_until_str} for all currently-snoozed emails."""
    import datetime
    init_db()
    now = datetime.datetime.utcnow().isoformat()
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT gmail_id, snoozed_until FROM email_processed WHERE snoozed_until IS NOT NULL AND snoozed_until > ?",
            (now,),
        )
        return {row[0]: row[1] for row in cur.fetchall()}


def reset_db_initialized() -> None:
    """Reset the init guard so init_db() will re-run DDL.

    Only needed in tests that delete and recreate the DB file between runs.
    """
    global _DB_INITIALIZED
    _DB_INITIALIZED = False


def upsert_user_profile(
    email: str,
    display_name: Optional[str] = None,
    style_notes: Optional[str] = None,
    preferences_json: Optional[str] = None,
    roles_json: Optional[str] = None,
    working_hours_json: Optional[str] = None,
) -> None:
    """Insert or update a user_profile row by email."""
    init_db()
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO user_profile (
                email, display_name, style_notes, preferences_json, roles_json, working_hours_json
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(email) DO UPDATE SET
                display_name = COALESCE(excluded.display_name, user_profile.display_name),
                style_notes = COALESCE(excluded.style_notes, user_profile.style_notes),
                preferences_json = COALESCE(excluded.preferences_json, user_profile.preferences_json),
                roles_json = COALESCE(excluded.roles_json, user_profile.roles_json),
                working_hours_json = COALESCE(excluded.working_hours_json, user_profile.working_hours_json),
                updated_at = CURRENT_TIMESTAMP
            """,
            (email, display_name, style_notes, preferences_json, roles_json, working_hours_json),
        )
        conn.commit()


def get_user_profile(email: str) -> Optional[Dict[str, Any]]:
    """Fetch a user_profile row as a dict, or None if missing."""
    init_db()
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM user_profile WHERE email = ?", (email,))
        row = cur.fetchone()
        if not row:
            return None
        columns = [d[0] for d in cur.description]
        return dict(zip(columns, row))


def add_memory(
    user_email: str,
    kind: str,
    key: Optional[str],
    value: str,
    source: Optional[str] = None,
) -> None:
    """Insert a single memory record. Auto-creates user_profile row if needed."""
    init_db()
    with get_connection() as conn:
        cur = conn.cursor()
        # Ensure user_profile row exists so foreign key constraint is satisfied
        cur.execute(
            "INSERT OR IGNORE INTO user_profile (email) VALUES (?)",
            (user_email,),
        )
        cur.execute(
            """
            INSERT INTO memory (user_email, kind, key, value, source)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_email, kind, key, value, source),
        )
        conn.commit()


def get_delegation_rules(user_email: str) -> list[Dict[str, Any]]:
    """Return all delegation rules for a user."""
    init_db()
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, pattern, target_email, enabled, weight FROM delegation_rules WHERE user_email = ?",
            (user_email,),
        )
        rows = cur.fetchall()
        columns = [d[0] for d in cur.description]
        return [dict(zip(columns, row)) for row in rows]


def get_memory_entries(kind: Optional[str] = None, limit: int = 50) -> list[Dict[str, Any]]:
    """Read memory entries, optionally filtered by kind.

    Returns list of dicts with kind, key, value, source, created_at.
    """
    init_db()
    with get_connection() as conn:
        cur = conn.cursor()
        if kind:
            cur.execute(
                "SELECT kind, key, value, source, created_at FROM memory WHERE kind = ? ORDER BY created_at DESC LIMIT ?",
                (kind, limit),
            )
        else:
            cur.execute(
                "SELECT kind, key, value, source, created_at FROM memory ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

