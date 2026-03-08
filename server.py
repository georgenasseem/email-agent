"""Flask backend for Dispatch — serves REST API + modern HTML frontend."""

import os
import re
import json
import time
import logging
import threading
from datetime import datetime, timedelta

from flask import Flask, jsonify, request, render_template

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
    get_completed_tasks,
    store_processed_emails,
    retag_emails_supplementary,
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
from tools.calendar_tools import (
    find_free_slots,
    create_event,
    list_events,
    get_user_timezone,
)
from tools.zoom_tools import create_zoom_meeting, zoom_available
from config import TOKEN_FILE, CREDENTIALS_FILE, PROJECT_ROOT
from tools.google_auth import get_google_credentials

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Persisted settings ──────────────────────────────────────────────────────
_SETTINGS_FILE = PROJECT_ROOT / "data" / "settings.json"


def _load_persisted_settings() -> dict:
    try:
        if _SETTINGS_FILE.exists():
            with open(_SETTINGS_FILE) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_persisted_settings(data: dict) -> None:
    try:
        _SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        existing = _load_persisted_settings()
        existing.update(data)
        with open(_SETTINGS_FILE, "w") as f:
            json.dump(existing, f, indent=2)
    except Exception as e:
        logger.warning("Failed to save settings: %s", e)

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
        self.snoozed: dict[str, str] = {}  # {email_id: until_iso}
        self.is_fetching: bool = False
        self.lock = threading.Lock()

    def load_initial(self):
        """Load emails from DB and other initial state."""
        try:
            # Apply saved LLM provider preference
            saved = _load_persisted_settings()
            if "llm_provider" in saved:
                os.environ["LLM_PROVIDER"] = saved["llm_provider"]
            if "last_fetch_ts" in saved:
                self.last_fetch_ts = saved["last_fetch_ts"]

            self.emails = load_from_memory() or []
            self.opened_ids = get_opened_email_ids()
            self.snoozed = get_snoozed_email_ids()  # {id: until_iso} for active snoozes
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
        "internal_date": e.get("internal_date", 0),
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

    # Filter out emails that are currently snoozed
    now_iso = datetime.utcnow().isoformat()
    emails = [e for e in emails if _state.snoozed.get(e.get("id", ""), "") <= now_iso]

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
        _save_persisted_settings({"last_fetch_ts": _state.last_fetch_ts})
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

@app.route("/api/emails/<email_id>/rethink", methods=["POST"])
def api_rethink_email(email_id):
    """Force-regenerate quick actions for an email and persist to DB."""
    email = _state.find_email(email_id)
    if not email:
        return jsonify({"error": "Email not found"}), 404
    try:
        recat = categorize_email(email)
        email["category"] = recat.get("category", email.get("category", "informational"))
        result = suggest_quick_actions_full(email)
        email["decision_options"] = result.get("final_options", [])
        email["no_action_message"] = result.get("no_action_message", "")
        store_processed_emails([email])
        return jsonify(_email_to_detail(email))
    except Exception as e:
        logger.error("Rethink failed for %s: %s", email_id, e)
        return jsonify({"error": str(e)}), 500


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
        # Track in-memory so the list view filters it immediately;
        # the email stays in _state.emails so it reappears when snooze expires.
        _state.snoozed[email_id] = until
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/emails/<email_id>/mark-unread", methods=["POST"])
def api_mark_unread(email_id):
    """Mark an email as unread (remove from opened set)."""
    try:
        _state.opened_ids.discard(email_id)
        # Also update the in-memory email
        email = _state.find_email(email_id)
        if email:
            email["is_read"] = False
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/emails/<email_id>/category", methods=["PUT"])
def api_update_email_category(email_id):
    """Update the category of a specific email."""
    email = _state.find_email(email_id)
    if not email:
        return jsonify({"error": "Email not found"}), 404

    data = request.json or {}
    new_category = data.get("category", "").strip()
    if not new_category:
        return jsonify({"error": "category is required"}), 400

    try:
        email["category"] = new_category
        store_processed_emails([email])
        return jsonify({"success": True, "category": new_category})
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

        # Learn style only if not yet learned (LLM call is risky on every send)
        if not _state.learned_style:
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
    tone = data.get("tone", "")

    if not to or not subject or not context:
        return jsonify({"error": "to, subject, and context are required"}), 400

    try:
        decision = f"[Tone: {tone}] {context}" if tone else context
        stub = {
            "id": "compose_new",
            "subject": subject,
            "sender": to,
            "clean_body": "",
            "body": "",
            "snippet": "",
        }
        style = _state.learned_style or load_persisted_style() or ""
        draft = run_drafting_graph(stub, decision=decision, style_notes=style)
        if draft:
            return jsonify({"draft": draft})
        return jsonify({"error": "Failed to generate draft"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/compose/send", methods=["POST"])
def api_compose_send():
    data = request.json or {}
    to = data.get("to", "")
    cc = data.get("cc", "")
    subject = data.get("subject", "")
    body = data.get("body", "")

    if not to or not subject or not body:
        return jsonify({"error": "to, subject, and body are required"}), 400

    try:
        send_email(to=to, subject=subject, body=body, cc=cc or None)
        # Learn style only if not yet learned (LLM call is risky on every send)
        if not _state.learned_style:
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
# ██  API — Meeting / Calendar
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/compose/free-slots", methods=["GET"])
def api_free_slots():
    """Return upcoming free calendar slots for the next 5 working days.
    
    Checks availability for the user AND all supplied attendees.
    """
    try:
        tz = get_user_timezone()
        now = datetime.now(tz)
        end = now + timedelta(days=7)
        duration = int(request.args.get("duration", 30))
        # Collect attendee emails — supports both single &attendee= and comma-separated
        attendee_param = request.args.get("attendee", "").strip()
        attendees = [a.strip() for a in attendee_param.split(",") if a.strip()] if attendee_param else []
        slots = find_free_slots(
            now, end,
            duration_minutes=duration,
            extra_attendees=attendees if attendees else None,
        )[:8]
        result = []
        for s, e in slots:
            local_s = s.astimezone(tz)
            result.append({
                "start": s.isoformat(),
                "end": e.isoformat(),
                "label": local_s.strftime("%a %b %-d, %-I:%M %p"),
            })
        return jsonify({"slots": result, "checked_attendees": attendees})
    except Exception as ex:
        logger.warning("free-slots error: %s", ex)
        return jsonify({"slots": [], "error": str(ex)}), 200


@app.route("/api/compose/create-meeting", methods=["POST"])
def api_create_meeting():
    """Create a calendar event (and optionally a Zoom meeting)."""
    data = request.json or {}
    summary = data.get("summary", "Meeting")
    start_iso = data.get("start")          # ISO string
    duration = int(data.get("duration", 30))
    attendees = data.get("attendees", [])  # list of email strings
    add_zoom = bool(data.get("add_zoom", False))
    description = data.get("description", "")

    if not start_iso:
        return jsonify({"error": "start is required"}), 400

    try:
        start_dt = datetime.fromisoformat(start_iso)
        end_dt = start_dt + timedelta(minutes=duration)

        zoom_link = None
        zoom_join = None
        if add_zoom:
            try:
                zm = create_zoom_meeting(summary, start_dt, duration)
                zoom_link = zm.get("start_url", "")
                zoom_join = zm.get("join_url", "")
                if zoom_join:
                    description = (description + f"\n\nZoom: {zoom_join}").strip()
            except Exception as ze:
                logger.warning("Zoom creation failed: %s", ze)

        event = create_event(
            summary=summary,
            start=start_dt,
            end=end_dt,
            attendees=attendees or None,
            description=description or None,
            location=zoom_join or None,
        )
        return jsonify({
            "event_id": event.get("id"),
            "html_link": event.get("htmlLink"),
            "zoom_join": zoom_join,
            "zoom_start": zoom_link,
        })
    except Exception as ex:
        logger.error("create-meeting error: %s", ex)
        return jsonify({"error": str(ex)}), 500


@app.route("/api/compose/zoom-available")
def api_zoom_available():
    return jsonify({"available": zoom_available()})


# ═══════════════════════════════════════════════════════════════════════════════
# ██  API — Meetings Hub
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/meetings")
def api_list_meetings():
    """List upcoming calendar events for meetings hub."""
    try:
        days = int(request.args.get("days", 14))
        tz = get_user_timezone()
        now = datetime.now(tz)
        end = now + timedelta(days=days)
        events = list_events(time_min=now, time_max=end, max_results=50)
        return jsonify({"meetings": events})
    except Exception as ex:
        logger.warning("list-meetings error: %s", ex)
        return jsonify({"meetings": [], "error": str(ex)}), 200


@app.route("/api/meetings/<event_id>", methods=["DELETE"])
def api_delete_meeting(event_id):
    """Delete/cancel a calendar event."""
    from tools.calendar_tools import get_calendar_service
    try:
        service = get_calendar_service()
        service.events().delete(calendarId="primary", eventId=event_id).execute()
        return jsonify({"success": True})
    except Exception as ex:
        logger.error("delete-meeting error: %s", ex)
        return jsonify({"error": str(ex)}), 500


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
        # Auto-tag existing emails with this new supplementary category
        slug = label.get("slug", "")
        if slug:
            count = retag_emails_supplementary(slug, name, _state.emails)
            if count:
                logger.info("Auto-tagged %d emails with new category '%s'", count, name)
        return jsonify({"category": label, "retagged": count if slug else 0}), 201
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


@app.route("/api/todos/suggestions")
def api_todo_suggestions():
    """Return suggested todo actions from recent emails (within 3 days)."""
    try:
        cutoff = datetime.now().timestamp() - 3 * 86400  # 3 days in seconds
        suggestions = []
        existing_tasks = {t.get("task", "").lower() for t in get_todo_items()}
        completed = set(get_completed_tasks())

        for e in _state.emails:
            internal_date = e.get("internal_date", 0)
            if internal_date and internal_date < cutoff:
                continue
            opts = e.get("decision_options") or []
            todo_opts = [o for o in opts if o.get("type") == "todo"]
            for opt in todo_opts:
                label = opt.get("label", "").strip()
                if not label or label.lower() in existing_tasks or label.lower() in completed:
                    continue
                suggestions.append({
                    "label": label,
                    "context": opt.get("context", ""),
                    "email_id": e.get("id", ""),
                    "email_subject": e.get("subject", ""),
                    "email_sender": e.get("sender", ""),
                    "email_date": e.get("date", ""),
                    "urgent": bool(e.get("urgent")),
                    "internal_date": internal_date,
                })

        # Sort by urgency then by date descending
        suggestions.sort(key=lambda s: (s["urgent"], s["internal_date"]), reverse=True)
        return jsonify({"suggestions": suggestions})
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
    _save_persisted_settings({"llm_provider": provider})
    return jsonify({"provider": provider})


@app.route("/api/categories/suggest")
def api_suggest_categories():
    """Return AI-suggested new categories based on email history."""
    try:
        proposals = propose_categories_from_history(min_sender_count=2)
        filtered = filter_proposals_with_llm(proposals)
        return jsonify({"suggestions": filtered})
    except Exception as e:
        logger.error("Category suggest failed: %s", e)
        return jsonify({"error": str(e)}), 500


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
