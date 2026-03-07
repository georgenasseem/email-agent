"""Flask backend for Email Agent — serves REST API + modern HTML frontend."""

import os
import re
import json
import time
import html as html_lib
import logging
import threading
from datetime import datetime, timedelta

from flask import Flask, jsonify, request, render_template, send_from_directory

# ── Agent imports ────────────────────────────────────────────────────────────
from agent.langgraph_pipeline import (
    run_email_pipeline,
    load_from_memory,
    INITIAL_FETCH_COUNT,
    FETCH_NEW_COUNT,
)
from agent.email_memory import (
    get_known_contacts,
    get_category_counts,
    get_all_labels,
    get_enabled_labels,
    create_label,
    update_label,
    delete_label,
    ensure_default_labels,
    propose_categories_from_history,
    filter_proposals_with_llm,
    get_todo_items,
    remove_todo_item,
    add_todo_item,
    store_processed_emails,
)
from agent.memory_store import (
    mark_email_archived,
    mark_email_opened,
    get_opened_email_ids,
    snooze_email,
    get_snoozed_email_ids,
)
from agent.quick_actions_graph import suggest_quick_actions_full
from agent.categorizer import categorize_email
from agent.drafting_graph import run_drafting_graph
from agent.style_learner import learn_and_persist_style, load_persisted_style
from tools.gmail_tools import (
    send_email,
    archive_email as gmail_archive_email,
    extract_email_address,
)
from config import TOKEN_FILE, CREDENTIALS_FILE
from tools.google_auth import get_google_credentials

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(
    __name__,
    static_folder="static",
    template_folder="templates",
)


# ═══════════════════════════════════════════════════════════════════════════════
# ██  In-memory application state (single-user local application)
# ═══════════════════════════════════════════════════════════════════════════════

class AppState:
    def __init__(self):
        self.emails: list[dict] = []
        self.last_fetch_ts: float | None = None
        self.learned_style: str = ""
        self.opened_ids: set[str] = set()
        self.is_fetching: bool = False
        self.lock = threading.Lock()

    def load_initial(self):
        """Load emails from DB and other initial state."""
        try:
            self.emails = load_from_memory() or []
            self.opened_ids = get_opened_email_ids()
            self.learned_style = load_persisted_style() or ""
            ensure_default_labels()
            logger.info("Loaded %d emails from database", len(self.emails))
        except Exception as e:
            logger.error("Failed to load initial state: %s", e)

    def find_email(self, email_id: str) -> dict | None:
        return next((e for e in self.emails if e.get("id") == email_id), None)


_state = AppState()


# ═══════════════════════════════════════════════════════════════════════════════
# ██  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _email_to_list_item(e: dict) -> dict:
    """Serialize an email dict for the list view (no full body)."""
    return {
        "id": e.get("id", ""),
        "thread_id": e.get("thread_id", ""),
        "subject": e.get("subject", ""),
        "sender": e.get("sender", ""),
        "date": e.get("date", ""),
        "snippet": e.get("snippet", ""),
        "summary": e.get("summary", ""),
        "category": e.get("category", "informational"),
        "urgent": bool(e.get("urgent")),
        "is_read": e.get("id", "") in _state.opened_ids,
    }


def _email_to_detail(e: dict) -> dict:
    """Serialize an email dict for the detail view (includes body)."""
    return {
        "id": e.get("id", ""),
        "thread_id": e.get("thread_id", ""),
        "subject": e.get("subject", ""),
        "sender": e.get("sender", ""),
        "date": e.get("date", ""),
        "body": e.get("clean_body") or e.get("body") or e.get("snippet") or "",
        "body_html": e.get("body_html", ""),
        "summary": e.get("summary", ""),
        "category": e.get("category", "informational"),
        "urgent": bool(e.get("urgent")),
        "is_read": True,
        "decision_options": e.get("decision_options") or [],
        "no_action_message": e.get("no_action_message", ""),
    }


def _sort_emails(emails: list[dict]) -> list[dict]:
    """Sort emails by urgency and recency, unread first."""
    opened = _state.opened_ids

    def score(e):
        urgent_bonus = 50 if e.get("urgent") else 0
        ts = e.get("internal_date") or 0
        recency = ts / 1e10 if ts else 0
        opened_penalty = 12 if e.get("id") in opened else 0
        return urgent_bonus + recency - opened_penalty

    return sorted(emails, key=score, reverse=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ██  Routes — Pages
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


# ═══════════════════════════════════════════════════════════════════════════════
# ██  API — Emails
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/emails")
def api_list_emails():
    search = request.args.get("search", "").strip().lower()
    category = request.args.get("category", "").strip()
    page = max(1, int(request.args.get("page", 1)))
    per_page = min(50, max(1, int(request.args.get("per_page", 25))))

    emails = _sort_emails(_state.emails)

    # Filter by category
    if category:
        cats = {c.strip() for c in category.split(",") if c.strip()}
        emails = [e for e in emails if any(
            c in (e.get("category", "") or "") for c in cats
        )]

    # Filter by search
    if search:
        emails = [e for e in emails if (
            search in (e.get("subject", "") or "").lower()
            or search in (e.get("sender", "") or "").lower()
            or search in (e.get("snippet", "") or "").lower()
        )]

    total = len(emails)
    total_pages = max(1, (total + per_page - 1) // per_page)
    start = (page - 1) * per_page
    page_emails = emails[start : start + per_page]

    return jsonify({
        "emails": [_email_to_list_item(e) for e in page_emails],
        "total": total,
        "page": page,
        "total_pages": total_pages,
        "last_fetch_ts": _state.last_fetch_ts,
    })


@app.route("/api/emails/<email_id>")
def api_get_email(email_id):
    email = _state.find_email(email_id)
    if not email:
        return jsonify({"error": "Email not found"}), 404

    # Mark as read
    if email_id not in _state.opened_ids:
        mark_email_opened(email_id)
        _state.opened_ids.add(email_id)

    # Lazy backfill: generate quick actions if missing
    opts = email.get("decision_options")
    if opts is None:
        try:
            recat = categorize_email(email)
            email["category"] = recat.get("category", email.get("category", "informational"))
            result = suggest_quick_actions_full(email)
            email["decision_options"] = result.get("final_options", [])
            email["no_action_message"] = result.get("no_action_message", "")
            store_processed_emails([email])
        except Exception as e:
            logger.warning("Backfill failed for %s: %s", email_id, e)
            email["decision_options"] = []

    return jsonify(_email_to_detail(email))


@app.route("/api/emails/fetch", methods=["POST"])
def api_fetch_emails():
    if _state.is_fetching:
        return jsonify({"error": "Already fetching"}), 409

    _state.is_fetching = True
    try:
        data = request.json or {}
        retrain = data.get("retrain", False)
        count = INITIAL_FETCH_COUNT if retrain else FETCH_NEW_COUNT

        state = run_email_pipeline(
            query="in:inbox",
            max_emails=count,
            unread_only=False,
            retrain=retrain,
        )

        emails = state.get("emails") or []
        _state.emails = emails
        _state.last_fetch_ts = time.time()
        _state.opened_ids = get_opened_email_ids()

        # Re-learn style in background
        if _state.learned_style == "" or retrain:
            try:
                style = learn_and_persist_style()
                if style:
                    _state.learned_style = style
            except Exception:
                pass

        return jsonify({
            "count": len(emails),
            "last_fetch_ts": _state.last_fetch_ts,
        })
    except Exception as e:
        logger.error("Fetch failed: %s", e)
        return jsonify({"error": str(e)}), 500
    finally:
        _state.is_fetching = False


# ═══════════════════════════════════════════════════════════════════════════════
# ██  API — Email Actions
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/emails/<email_id>/archive", methods=["POST"])
def api_archive_email(email_id):
    try:
        gmail_archive_email(email_id)
        mark_email_archived(email_id)
        _state.emails = [e for e in _state.emails if e.get("id") != email_id]
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/emails/<email_id>/snooze", methods=["POST"])
def api_snooze_email(email_id):
    try:
        data = request.json or {}
        until = data.get("until", "")
        if not until:
            # Default: snooze for 3 hours
            until = (datetime.utcnow() + timedelta(hours=3)).isoformat()
        snooze_email(email_id, until)
        _state.emails = [e for e in _state.emails if e.get("id") != email_id]
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/emails/<email_id>/draft", methods=["POST"])
def api_draft_reply(email_id):
    email = _state.find_email(email_id)
    if not email:
        return jsonify({"error": "Email not found"}), 404

    data = request.json or {}
    decision = data.get("decision", "")
    if not decision:
        return jsonify({"error": "Decision/instruction is required"}), 400

    try:
        style = _state.learned_style or load_persisted_style() or ""
        draft = run_drafting_graph(email, decision=decision, style_notes=style)
        if draft:
            return jsonify({"draft": draft})
        return jsonify({"error": "Failed to generate draft"}), 500
    except Exception as e:
        logger.error("Draft error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/emails/<email_id>/send", methods=["POST"])
def api_send_reply(email_id):
    email = _state.find_email(email_id)
    if not email:
        return jsonify({"error": "Email not found"}), 404

    data = request.json or {}
    body = data.get("body", "")
    if not body:
        return jsonify({"error": "Body is required"}), 400

    try:
        to = extract_email_address(email.get("sender", ""))
        subject = email.get("subject", "")
        if not subject.lower().startswith("re:"):
            subject = f"Re: {subject}"

        send_email(
            to=to,
            subject=subject,
            body=body,
            thread_id=email.get("thread_id"),
        )

        # Re-learn style after sending
        try:
            style = learn_and_persist_style()
            if style:
                _state.learned_style = style
        except Exception:
            pass

        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# ██  API — Compose
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/compose/draft", methods=["POST"])
def api_compose_draft():
    data = request.json or {}
    to = data.get("to", "")
    subject = data.get("subject", "")
    context = data.get("context", "")

    if not to or not subject or not context:
        return jsonify({"error": "to, subject, and context are required"}), 400

    try:
        stub = {
            "id": "compose_new",
            "subject": subject,
            "sender": to,
            "clean_body": "",
            "body": "",
            "snippet": "",
        }
        style = _state.learned_style or load_persisted_style() or ""
        draft = run_drafting_graph(stub, decision=context, style_notes=style)
        if draft:
            return jsonify({"draft": draft})
        return jsonify({"error": "Failed to generate draft"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/compose/send", methods=["POST"])
def api_compose_send():
    data = request.json or {}
    to = data.get("to", "")
    subject = data.get("subject", "")
    body = data.get("body", "")

    if not to or not subject or not body:
        return jsonify({"error": "to, subject, and body are required"}), 400

    try:
        send_email(to=to, subject=subject, body=body)
        # Re-learn style after sending
        try:
            style = learn_and_persist_style()
            if style:
                _state.learned_style = style
        except Exception:
            pass
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# ██  API — Categories
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/categories")
def api_list_categories():
    try:
        labels = get_all_labels()
        counts = get_category_counts()
        for lb in labels:
            lb["count"] = counts.get(lb["slug"], 0)
        return jsonify({"categories": labels})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/categories", methods=["POST"])
def api_create_category():
    data = request.json or {}
    name = data.get("name", "").strip()
    color = data.get("color", "#94a3b8")
    description = data.get("description", "")

    if not name:
        return jsonify({"error": "Name is required"}), 400

    try:
        label = create_label(name, color, description)
        return jsonify({"category": label}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/categories/<slug>", methods=["PUT"])
def api_update_category(slug):
    data = request.json or {}
    try:
        update_label(
            slug,
            display_name=data.get("display_name"),
            color=data.get("color"),
            description=data.get("description"),
            enabled=data.get("enabled"),
        )
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/categories/<slug>", methods=["DELETE"])
def api_delete_category(slug):
    try:
        delete_label(slug)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# ██  API — Todos
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/todos")
def api_list_todos():
    try:
        items = get_todo_items()
        # Enrich with source email info
        email_lookup = {e.get("id"): e for e in _state.emails}
        for item in items:
            src = email_lookup.get(item.get("email_id", ""))
            if src:
                item["source_subject"] = src.get("subject", "")
                item["source_sender"] = src.get("sender", "")
        return jsonify({"todos": items})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/todos", methods=["POST"])
def api_add_todo():
    data = request.json or {}
    task = data.get("task", "").strip()
    email_id = data.get("email_id", "")

    if not task:
        return jsonify({"error": "Task is required"}), 400

    try:
        item_id = add_todo_item(task, email_id)
        return jsonify({"id": item_id}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/todos/<int:item_id>/done", methods=["POST"])
def api_complete_todo(item_id):
    try:
        remove_todo_item(item_id)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# ██  API — Contacts
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/contacts")
def api_contacts():
    q = request.args.get("q", "")
    limit = min(50, int(request.args.get("limit", 20)))
    try:
        contacts = get_known_contacts(query=q, limit=limit)
        return jsonify({"contacts": contacts})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# ██  API — Settings
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/settings")
def api_get_settings():
    provider = os.environ.get("LLM_PROVIDER", "qwen_local_3b")
    return jsonify({
        "llm_provider": provider,
        "last_fetch_ts": _state.last_fetch_ts,
        "email_count": len(_state.emails),
    })


@app.route("/api/settings/llm", methods=["PUT"])
def api_set_llm():
    data = request.json or {}
    provider = data.get("provider", "qwen_local_3b")
    allowed = {"qwen_local_3b", "qwen_local_7b", "groq"}
    if provider not in allowed:
        return jsonify({"error": f"Invalid provider. Allowed: {allowed}"}), 400
    os.environ["LLM_PROVIDER"] = provider
    return jsonify({"provider": provider})


# ═══════════════════════════════════════════════════════════════════════════════
# ██  API — Auth
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/auth/status")
def api_auth_status():
    signed_in = TOKEN_FILE.exists()
    return jsonify({"signed_in": signed_in})


@app.route("/api/auth/signin", methods=["POST"])
def api_auth_signin():
    if not CREDENTIALS_FILE.exists():
        return jsonify({"error": "credentials.json not found. See README for setup instructions."}), 400
    try:
        get_google_credentials()
        return jsonify({"success": True})
    except Exception as e:
        logger.error("Sign-in failed: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/auth/signout", methods=["POST"])
def api_auth_signout():
    try:
        if TOKEN_FILE.exists():
            TOKEN_FILE.unlink()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# ██  Startup
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    _state.load_initial()
    app.run(host="localhost", port=8080, debug=False, threaded=True)
