"""AI Email Agent - Streamlit UI."""
import os
import re
import html
import hashlib
import random
import streamlit as st
import streamlit.components.v1 as components


def safe_key(email_id: str) -> str:
    """Return a DOM-safe key derived from an email ID.

    Gmail IDs can contain +, /, = which are invalid in HTML element IDs.
    We produce a short hex digest that is always safe.
    """
    return hashlib.md5(email_id.encode()).hexdigest()[:12]

from config import CREDENTIALS_FILE, TOKEN_FILE
from tools.google_auth import get_google_credentials
from tools.gmail_tools import send_email, extract_email_address, archive_email
from agent.langgraph_pipeline import run_email_pipeline, load_from_memory, INITIAL_FETCH_COUNT, FETCH_NEW_COUNT
from agent.memory_store import mark_email_archived
from agent.email_memory import (
    get_email_count,
    add_todo_item, get_todo_items, remove_todo_item,
    get_all_labels, get_enabled_labels, create_label, delete_label,
    record_category_override, propose_categories_from_history,
    filter_proposals_with_llm,
    ensure_default_labels,
    apply_rule_to_existing_emails,
    SYSTEM_CATEGORIES,
)
from agent.decision_suggester import suggest_decision
from agent.drafter import draft_reply
from agent.drafting_graph import run_drafting_graph
from agent.style_learner import load_persisted_style
from agent.profile import load_profile, get_current_user_email
from agent.scheduling import propose_meeting_times, create_event_from_slot

# Page config
st.set_page_config(
    page_title="AI Email Agent",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Theme state (dark / light)
if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"

theme = st.session_state["theme"]

if theme == "dark":
    bg_start = "#020617"
    bg_end = "#020617"
    text_color = "#e5e7eb"
    muted_color = "#9ca3af"
    surface_bg = "#020617"
else:
    bg_start = "#f9fafb"
    bg_end = "#e5e7eb"
    text_color = "#020617"
    muted_color = "#6b7280"
    surface_bg = "#ffffff"

# Accent palette: cool blue primary with warm orange highlights
accent_color = "#38bdf8"  # primary blue
accent_soft = "#0ea5e9"
accent_orange = "#fb923c"
border_soft = "rgba(148,163,184,0.35)"

st.markdown(
    f"""
<style>
    .stApp {{
        background: radial-gradient(circle at top left, {bg_start}, {bg_end});
        color: {text_color};
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
        padding-top: 6px;
    }}

    .header-container {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 4px 0 2px 0;
        margin-bottom: 2px;
        border-bottom: 1px solid rgba(148,163,184,0.25);
    }}
    .header-title {{
        font-size: 16px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-weight: 600;
        color: {text_color};
        line-height: 1.2;
        margin: 0;
    }}
    .header-subtitle {{
        font-size: 11px;
        color: {muted_color};
        line-height: 1.2;
        margin: 0;
    }}

    .header-controls {{
        display: flex;
        gap: 8px;
        align-items: center;
        font-size: 12px;
        color: {muted_color};
    }}

    .logo-mark {{
        width: 26px;
        height: 26px;
        border-radius: 999px;
        background: #020617;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 13px;
        color: {accent_color};
        margin-right: 6px;
        border: 1px solid {border_soft};
    }}

    .pill {{
        padding: 4px 10px;
        border-radius: 999px;
        border: 1px solid {border_soft};
        background: rgba(15,23,42,0.5);
    }}

    .pill-muted {{
        padding: 4px 10px;
        border-radius: 999px;
        border: 1px solid rgba(148,163,184,0.3);
        background: rgba(15,23,42,0.2);
        color: {muted_color};
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }}

    .section-title {{
        font-size: 15px;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: {muted_color};
        margin-bottom: 4px;
    }}

    .card-muted {{
        font-size: 12px;
        color: {muted_color};
    }}

    .stButton>button {{
        border-radius: 6px;
        padding: 4px 14px;
        background: {accent_color};
        color: #ffffff;
        border: none;
        font-size: 12px;
        font-weight: 500;
        transition: background 0.15s ease, box-shadow 0.15s ease;
    }}
    .stButton>button:hover {{
        background: {accent_soft};
        box-shadow: 0 2px 8px rgba(34,197,94,0.3);
    }}

    .summary {{
        color: {muted_color};
        font-size: 13px;
    }}
    .meta {{
        color: {muted_color};
        font-size: 11px;
    }}

    .email-meta-small {{
        font-size: 11px;
        color: {muted_color};
    }}

    .status-dot-green {{
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 999px;
        background: {accent_color};
        margin-right: 4px;
    }}

    .status-dot-orange {{
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 999px;
        background: {accent_orange};
        margin-right: 4px;
    }}

    .soft-card {{
        background: rgba(15,23,42,0.8);
        border-radius: 18px;
        padding: 10px 14px;
        border: 1px solid {border_soft};
    }}

    .soft-card-tight {{
        background: rgba(15,23,42,0.9);
        border-radius: 16px;
        padding: 8px 10px;
        border: 1px solid rgba(148,163,184,0.25);
    }}

    .soft-divider {{
        border-bottom: 1px dashed rgba(148,163,184,0.35);
        margin: 4px 0 10px 0;
    }}

    /* Quick-action button groups — keyed containers */
    /* Reply buttons: ORANGE */
    div[class*="qa-reply-"] .stButton>button {{
        background: #f97316 !important;
        color: #fff !important;
        border: none !important;
        font-weight: 500 !important;
    }}
    div[class*="qa-reply-"] .stButton>button:hover {{
        background: #ea580c !important;
        color: #fff !important;
        box-shadow: 0 2px 8px rgba(249,115,22,0.35);
    }}
    /* Todo buttons: PURPLE */
    div[class*="qa-todo-"] .stButton>button {{
        background: #8b5cf6 !important;
        color: #fff !important;
        border: none !important;
        font-weight: 500 !important;
    }}
    div[class*="qa-todo-"] .stButton>button:hover {{
        background: #7c3aed !important;
        color: #fff !important;
        box-shadow: 0 2px 8px rgba(139,92,246,0.35);
    }}
    /* Custom / Archive buttons: GRAY (engraved) */
    div[class*="qa-custom-"] .stButton>button,
    div[class*="qa-archive-"] .stButton>button {{
        background: #475569 !important;
        color: #e2e8f0 !important;
        border: 1px solid rgba(148,163,184,0.3) !important;
        font-weight: 500 !important;
    }}
    div[class*="qa-custom-"] .stButton>button:hover,
    div[class*="qa-archive-"] .stButton>button:hover {{
        background: #64748b !important;
        color: #fff !important;
        box-shadow: 0 2px 8px rgba(100,116,139,0.35);
    }}
    /* Remove extra padding/gap inside keyed QA containers */
    div[class*="qa-reply-"],
    div[class*="qa-todo-"],
    div[class*="qa-custom-"],
    div[class*="qa-archive-"] {{
        gap: 0 !important;
    }}

    .email-scroll {{
        max-height: calc(100vh - 260px);
        overflow-y: auto;
        padding-right: 4px;
    }}

    /* Email card item */
    .email-card {{
        padding: 8px 10px;
        border-radius: 8px;
        border: 1px solid {border_soft};
        background: {'rgba(15,23,42,0.5)' if theme == 'dark' else 'rgba(255,255,255,0.8)'};
        margin-bottom: 6px;
        transition: border-color 0.15s ease, background 0.15s ease;
        cursor: pointer;
    }}
    .email-card:hover {{
        border-color: {accent_color};
        background: {'rgba(56,189,248,0.07)' if theme == 'dark' else 'rgba(56,189,248,0.08)'};
    }}

    .email-subject {{
        font-size: 13px;
        font-weight: 600;
        color: {text_color};
        margin-bottom: 1px;
    }}
    .email-sender {{
        font-size: 11px;
        color: {muted_color};
        margin-bottom: 2px;
    }}
    .email-summary {{
        font-size: 12px;
        color: {'#cbd5e1' if theme == 'dark' else '#374151'};
        line-height: 1.3;
    }}

    /* Badges */
    .badge-action {{
        display: inline-block;
        padding: 2px 8px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 600;
        background: rgba(249,115,22,0.15);
        color: {accent_orange};
        border: 1px solid rgba(249,115,22,0.3);
    }}
    .badge-category {{
        display: inline-block;
        padding: 2px 8px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 500;
    }}

    /* Hide open-email buttons — JS event delegation handles clicks */
    [data-testid="stVerticalBlock"]:has(.email-card) {{
        position: relative !important;
    }}
    [data-testid="stVerticalBlock"]:has(.email-card) > [data-testid="stElementContainer"]:has(button) {{
        height: 0 !important;
        min-height: 0 !important;
        overflow: hidden !important;
        margin: 0 !important;
        padding: 0 !important;
        border: none !important;
    }}

    /* Section headers */
    h3 {{
        font-size: 14px !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: {muted_color} !important;
        font-weight: 600 !important;
        margin-bottom: 8px !important;
    }}

    /* Radio buttons (filter) */
    .stRadio > div {{
        gap: 4px;
    }}
    .stRadio label {{
        font-size: 13px;
    }}

    /* Header: compact model selectbox */
    div[data-testid="stHorizontalBlock"]:has(.stSelectbox):has(.stButton) .stSelectbox {{
        max-width: 200px;
    }}

    /* Selectbox dark styling */
    .stSelectbox [data-baseweb="select"] {{
        background: rgba(15,23,42,0.6);
        border: 1px solid {border_soft};
        border-radius: 8px;
    }}
    .stSelectbox [data-baseweb="select"]:hover {{
        border-color: {accent_color};
    }}

    /* Selectbox dropdown: force blue highlight only */
    [data-baseweb="menu"] [role="option"]:hover,
    [data-baseweb="menu"] [role="option"][aria-selected="true"],
    [data-baseweb="menu"] li:hover,
    [data-baseweb="menu"] li[aria-selected="true"] {{
        background-color: rgba(56, 189, 248, 0.18) !important;
    }}
    [data-baseweb="menu"] [role="option"][aria-selected="true"] {{
        background-color: rgba(56, 189, 248, 0.28) !important;
    }}

    /* Top bar: vertical alignment and compact spacing */
    .block-container > div:first-child > div[data-testid="stHorizontalBlock"] {{
        align-items: center;
        padding-top: 0;
        padding-bottom: 0;
        margin-bottom: 0;
    }}
    .block-container > div:first-child .stVerticalBlock {{
        padding-top: 0;
        padding-bottom: 0;
    }}

    /* Header controls (Model + Sign in) pushed to the far right */
    .block-container > div:first-child > div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child div[data-testid="stHorizontalBlock"] {{
        justify-content: flex-end;
        gap: 0;
    }}

    /* Strip ALL highlight / selection effects from email body iframe content */
    iframe {{
        border: none !important;
    }}
    ::selection {{
        background: transparent !important;
    }}
    ::-moz-selection {{
        background: transparent !important;
    }}

    /* Compact multiselect for category filter —
       one line of chips, overflow hidden with "..." indicator */
    .stMultiSelect [data-baseweb="tag"] {{
        max-width: 120px;
    }}
    .stMultiSelect [data-baseweb="tag"] span {{
        max-width: 80px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }}
    .stMultiSelect > div {{
        max-height: 38px;
        overflow: hidden;
    }}
    /* Dropdown checkmarks already shown by Streamlit for selected items */
    .stMultiSelect [data-baseweb="menu"] [role="option"][aria-selected="true"] {{
        background-color: rgba(56, 189, 248, 0.18) !important;
    }}
</style>
""",
    unsafe_allow_html=True,
)

# Check for Gmail credentials
if not CREDENTIALS_FILE.exists():
    st.error(
        "**Gmail credentials not found.**\n\n"
        "1. Go to [Google Cloud Console](https://console.cloud.google.com/)\n"
        "2. Create a project → APIs & Services → Enable Gmail API and Google Calendar API\n"
        "3. Create OAuth 2.0 credentials (Desktop app)\n"
        "4. Download JSON and save as `credentials.json` in the project root"
    )
    st.stop()

# Defaults for model (used before widgets initialize)
llm_options = ["Groq (hosted)", "Qwen 2.5 3B (local)", "Qwen 2.5 7B (local)"]
default_llm_display = st.session_state.get("llm_backend", "Qwen 2.5 3B (local)")
# MIN_FETCH / MAX_FETCH replaced by INITIAL_FETCH_COUNT and FETCH_NEW_COUNT from the pipeline.

# ── Set LLM provider EARLY so backfill, pipeline, and all LLM calls use the
#    model selected in the UI rather than whatever is in .env / config.py. ──
_llm_provider_map = {
    "Groq (hosted)": "groq",
    "Qwen 2.5 3B (local)": "qwen_local_3b",
    "Qwen 2.5 7B (local)": "qwen_local_7b",
}
os.environ["LLM_PROVIDER"] = _llm_provider_map.get(default_llm_display, "qwen_local_3b")

# Main header row: logo | model | fetch | sign-out
header = st.container()
with header:
    c1, c2, c3, c4 = st.columns([2.5, 1.8, 1.2, 0.8], vertical_alignment="center")

    with c1:
        st.markdown(
            """<div style="display:flex; align-items:center; gap:6px;">
                <div class="logo-mark">EA</div>
                <div>
                    <div class="header-title">Email Agent</div>
                    <div class="header-subtitle">Your email personal assistant</div>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

    with c2:
        llm_default_index = (
            llm_options.index(default_llm_display)
            if default_llm_display in llm_options
            else 1
        )
        llm_display = st.selectbox(
            "Model",
            llm_options,
            index=llm_default_index,
            key="llm_backend",
            label_visibility="collapsed",
        )

    with c3:
        is_analyzing = st.session_state.get("is_analyzing", False)
        fetch_clicked = st.button(
            "Fetch new emails",
            key="fetch_btn",
            disabled=is_analyzing,
        )

    with c4:
        is_signed_in = TOKEN_FILE.exists()
        button_label = "Sign out" if is_signed_in else "Sign in"
        if st.button(button_label, key="auth_btn"):
            if is_signed_in:
                TOKEN_FILE.unlink()
                st.success("Signed out. Use Sign in to sign in again.")
                st.rerun()
            else:
                try:
                    with st.spinner("Opening browser to sign in with Google..."):
                        get_google_credentials()
                    st.success("Signed in successfully.")
                    st.rerun()
                except FileNotFoundError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Sign-in failed: {e}")

# ── Ctrl+Shift+K keyboard shortcut for retrain ──────────────────────────
# Attach the listener to the *parent* document (Streamlit's main frame)
# using the capture phase so it fires before any other handlers.
components.html(
    """
    <script>
    (function() {
        try {
            var pd = window.parent.document;
            if (pd._retrainHandler) {
                pd.removeEventListener('keydown', pd._retrainHandler, true);
            }
            pd._retrainHandler = function(e) {
                if (e.ctrlKey && !e.metaKey && e.shiftKey && (e.key === 'K' || e.key === 'k')) {
                    e.preventDefault();
                    e.stopImmediatePropagation();
                    var url = new URL(window.parent.location);
                    url.searchParams.set('retrain', '1');
                    window.parent.location.assign(url.toString());
                }
            };
            pd.addEventListener('keydown', pd._retrainHandler, true);
        } catch(err) {
            console.error('Retrain shortcut error:', err);
        }
    })();
    </script>
    """,
    height=0,
)

# Detect the retrain query-param flag set by Ctrl+Shift+K
retrain_clicked = st.query_params.get("retrain") == "1"
if retrain_clicked:
    # Clear the flag so it doesn't persist across reruns
    st.query_params.pop("retrain", None)

# ── Instant load from DB on first visit ──────────────────────────────────
if "emails" not in st.session_state and "_memory_loaded" not in st.session_state:
    st.session_state["_memory_loaded"] = True
    try:
        cached = load_from_memory()   # loads ALL stored emails (no limit)
        if cached:
            # Backfill: regenerate quick actions for emails with missing actions.
            # decision_options is None (NULL in DB) → never processed / LLM failed.
            # decision_options is [] → could be a prior failure that stored []
            #   before reaching the LLM, so we retry those too (idempotent).
            _needs_backfill = [
                _ce for _ce in cached
                if _ce.get("decision_options") is None
                or _ce.get("decision_options") == []
            ]
            if _needs_backfill:
                from agent.quick_actions_graph import suggest_quick_actions_full as _backfill_qa
                from agent.categorizer import categorize_email as _backfill_cat
                from agent.email_memory import store_processed_emails as _backfill_store
                for _ce in _needs_backfill:
                    # Re-categorize too — if quick actions were missing, the
                    # category may also have defaulted to informational on error.
                    try:
                        _recat = _backfill_cat(_ce)
                        _ce["category"] = _recat.get("category", _ce.get("category", "informational"))
                    except Exception:
                        pass
                    try:
                        _bf_result = _backfill_qa(_ce)
                        _bf_opts = _bf_result.get("final_options", [])
                        _ce["decision_options"] = _bf_opts  # Always set, even if empty
                        _bf_na = _bf_result.get("no_action_message", "")
                        if _bf_na:
                            _ce["no_action_message"] = _bf_na
                    except Exception:
                        pass  # Leave as None — next restart will retry
                # Persist backfilled options to DB
                try:
                    _backfill_store(_needs_backfill)
                except Exception:
                    pass

            # Tag cached emails as action-required where appropriate
            for _ce in cached:
                _ce_cat = (_ce.get("category") or "informational").lower()
                _ce_tags = [t.strip() for t in _ce_cat.split(",") if t.strip()]
                _ce_main = _ce_tags[0] if _ce_tags else "informational"
                if _ce_main == "important":
                    _ce_opts = _ce.get("decision_options") or []
                    _has_reply = any(
                        (isinstance(o, dict) and o.get("type") == "reply") or
                        (isinstance(o, str) and o.lower().startswith("reply"))
                        for o in _ce_opts
                    ) if _ce_opts else False
                    if _has_reply:
                        _ce_extras = _ce_tags[1:]
                        _ce["category"] = ",".join(["action-required"] + _ce_extras) if _ce_extras else "action-required"
            st.session_state["emails"] = cached
            # Pre-compute category suggestions on first load
            try:
                _raw_p = propose_categories_from_history(min_sender_count=2)
                _filt_p = filter_proposals_with_llm(_raw_p) if _raw_p else []
                if _filt_p:
                    st.session_state["_cat_proposals"] = _filt_p
                st.session_state["_cat_proposals_stale"] = False
            except Exception:
                pass
        else:
            # ── Empty DB: run new-user init (same as retrain) ──────────
            # Defer to the analysis block by flagging is_analyzing
            st.session_state["is_analyzing"] = True
            st.session_state["_retrain"] = True
    except Exception:
        pass

if retrain_clicked and not st.session_state.get("is_analyzing", False):
    st.session_state["is_analyzing"] = True
    st.session_state["_retrain"] = True
    st.rerun()

if fetch_clicked and not is_analyzing:
    st.session_state["is_analyzing"] = True
    st.session_state["_retrain"] = False
    st.rerun()

# Show any persisted pipeline error from previous run (clears spinner by rerunning)
if "pipeline_error" in st.session_state:
    st.error(f"**Pipeline error:** {st.session_state['pipeline_error']}")
    del st.session_state["pipeline_error"]

# After header widgets are created, refresh the env var in case the user
# changed the selectbox *this* rerun (the widget value may have updated).
os.environ["LLM_PROVIDER"] = _llm_provider_map.get(
    st.session_state.get("llm_backend", default_llm_display), "qwen_local_3b"
)

# Email fetching and processing workflow (LangGraph pipeline)
if st.session_state.get("is_analyzing", False):
    actual_query = "in:inbox"
    retrain_mode = st.session_state.pop("_retrain", False)
    # Retrain / new-user init fetches 15; normal fetch grabs 20 to catch new ones
    fetch_count = INITIAL_FETCH_COUNT if retrain_mode else FETCH_NEW_COUNT
    status_label = "Initialising inbox..." if retrain_mode else "Fetching new emails..."

    with st.status(status_label, expanded=True) as status:
        try:
            status.update(label="Fetching emails...")
            state = run_email_pipeline(
                query=actual_query,
                max_emails=fetch_count,
                unread_only=False,
                retrain=retrain_mode,
            )
            status.update(label="Processing complete")
            emails = state.get("emails") or []
            learned = state.get("style_notes") or ""
            if learned:
                st.session_state["learned_style"] = learned
            if not emails:
                st.info("No emails found for this query.")
                st.stop()
            st.session_state["emails"] = emails
            if "pipeline_error" in st.session_state:
                del st.session_state["pipeline_error"]

            # Pre-compute suggested categories so Categories tab loads instantly
            status.update(label="Computing category suggestions...")
            try:
                _raw_proposals = propose_categories_from_history(min_sender_count=2)
                _proposals = filter_proposals_with_llm(_raw_proposals) if _raw_proposals else []
                if _proposals:
                    st.session_state["_cat_proposals"] = _proposals
                else:
                    st.session_state.pop("_cat_proposals", None)
                st.session_state["_cat_proposals_stale"] = False
            except Exception:
                st.session_state["_cat_proposals_stale"] = True

            status.update(label=f"Done — {len(emails)} emails loaded", state="complete")
        except Exception as e:
            st.session_state["pipeline_error"] = str(e)
            status.update(label="Error", state="error")
            st.rerun()
        finally:
            st.session_state["is_analyzing"] = False
    st.rerun()

# Display emails from session
if "emails" not in st.session_state:
    st.stop()

emails = st.session_state["emails"]

filtered = emails

# Apply category filter — hide unchecked categories
# Read directly from the multiselect widget key (_cat_multifilter) which Streamlit
# updates BEFORE the script re-runs. This avoids the 1-rerun delay that happened
# when reading from the derived _hidden_cats which was only set further down.
_cat_filter_sel = st.session_state.get("_cat_multifilter")
if _cat_filter_sel is not None:
    _all_filter_slugs = {lb["slug"] for lb in get_all_labels()}
    _hidden_cats = _all_filter_slugs - set(_cat_filter_sel)
    st.session_state["_hidden_cats"] = _hidden_cats
else:
    _hidden_cats = st.session_state.get("_hidden_cats", set())
if _hidden_cats:
    def _email_visible(e):
        cat_raw = (e.get("category", "informational") or "informational").strip()
        tags = [t.strip() for t in cat_raw.split(",") if t.strip()]
        main_cat = tags[0] if tags else "informational"
        # Hide if main category is in the hidden set
        return main_cat not in _hidden_cats
    filtered = [e for e in filtered if _email_visible(e)]

# Always sort by newest first
filtered.sort(key=lambda x: x.get("internal_date", 0), reverse=True)

# Two-column layout:
# left = drafting workspace, right = scrollable emails
draft_col, email_col = st.columns([1.1, 2.3])

sel_id = st.session_state.get("selected_email")
selected_email = None
if sel_id:
    # Always look up from the current emails list to avoid stale data
    # (e.g. after pipeline run updates decision_options in the email dict).
    selected_email = next((e for e in emails if e.get("id") == sel_id), None)
    if selected_email:
        st.session_state["selected_email_obj"] = selected_email
    else:
        selected_email = st.session_state.get("selected_email_obj")

with draft_col:
    # Tab selector: Drafting vs Todo vs Categories
    active_tab = st.radio(
        "Panel",
        ["Drafting", "Todo", "Categories"],
        key="left_panel_tab",
        horizontal=True,
        label_visibility="collapsed",
    )

    if active_tab == "Categories":
        # ── Category Management Panel ────────────────────────────────
        st.markdown("### Categories")
        ensure_default_labels()
        all_labels = get_all_labels()

        # ── Add new label ──
        # Curated palette of pleasant, non-annoying colours
        _NICE_COLORS = [
            "#38bdf8", "#34d399", "#a78bfa", "#fb923c",
            "#f472b6", "#22d3ee", "#4ade80", "#c084fc",
            "#f87171", "#60a5fa", "#2dd4bf", "#e879f9",
        ]
        with st.expander("Add new category", expanded=False):
            new_name = st.text_input("Name", key="new_cat_name", placeholder="e.g. Advising")
            new_desc = st.text_input("Description (optional)", key="new_cat_desc", placeholder="e.g. Academic advising emails")
            if st.button("Create", key="create_cat_btn"):
                if new_name and new_name.strip():
                    try:
                        _rand_color = random.choice(_NICE_COLORS)
                        create_label(new_name, color=_rand_color, description=new_desc)
                        st.success(f"Created '{new_name}'")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))
                else:
                    st.warning("Enter a name")

        # ── Existing labels ──
        if not all_labels:
            st.markdown("<div class='card-muted'>No categories yet.</div>", unsafe_allow_html=True)
        else:
            for lb in all_labels:
                slug = lb["slug"]
                c_left, c_mid = st.columns([4, 1])
                with c_left:
                    color_dot = f"<span style='display:inline-block;width:10px;height:10px;border-radius:50%;background:{lb['color']};margin-right:6px;vertical-align:middle;'></span>"
                    st.markdown(
                        f"<div style='font-size:13px;padding:4px 0;'>{color_dot}<b>{html.escape(lb['display_name'])}</b></div>",
                        unsafe_allow_html=True,
                    )
                with c_mid:
                    if slug not in SYSTEM_CATEGORIES:
                        if st.button("Delete", key=f"del_cat_{slug}"):
                            delete_label(slug)
                            # Also strip the tag from in-memory session emails
                            for _se in st.session_state.get("emails", []):
                                _se_cat = (_se.get("category", "informational") or "informational").strip()
                                _se_tags = [t.strip() for t in _se_cat.split(",") if t.strip() and t.strip() != slug]
                                _se["category"] = ",".join(_se_tags) if _se_tags else "informational"
                            st.rerun()

        # ── Re-categorize selected email ──
        st.markdown("---")
        st.markdown("**Re-categorize email**")
        _label_color_map = {}
        try:
            for _lb in all_labels:
                _label_color_map[_lb["slug"]] = _lb["color"]
        except Exception:
            pass
        _enabled_label_options = []
        try:
            _enabled_label_options = [lb["slug"] for lb in get_enabled_labels()]
        except Exception:
            _enabled_label_options = ["important", "informational", "newsletter"]

        _sel_email = st.session_state.get("selected_email_obj")
        if not _sel_email:
            st.markdown("<div class='card-muted'>Select an email to change its category.</div>", unsafe_allow_html=True)
        else:
            _re_cat_raw = _sel_email.get("category", "informational") or "informational"
            _re_cat_tags = [t.strip() for t in _re_cat_raw.split(",") if t.strip()]
            _re_cat = _re_cat_tags[0] if _re_cat_tags else "informational"
            _re_extras = _re_cat_tags[1:]
            _re_subj = html.escape(str(_sel_email.get("subject", "(No subject)")))
            _re_sender_raw = _sel_email.get("sender", "")
            _re_sender_match = re.match(r'^([^<]+)', _re_sender_raw)
            _re_sender = html.escape((_re_sender_match.group(1).strip() if _re_sender_match else _re_sender_raw)[:40])
            _re_cat_color = _label_color_map.get(_re_cat, "#94a3b8")

            _re_cat_display = f"<span style='color:{_re_cat_color};font-weight:600;'>[{html.escape(_re_cat)}]</span>"

            _re_left, _re_right = st.columns([4, 2])
            with _re_left:
                st.markdown(
                    f"<div style='font-size:12px;padding:3px 0;'>"
                    f"{_re_cat_display} "
                    f"<b>{_re_subj[:60]}</b>"
                    f"<br/><span style='color:#64748b;font-size:11px;'>{_re_sender}</span></div>",
                    unsafe_allow_html=True,
                )
            with _re_right:
                _re_eid_safe = safe_key(_sel_email.get("id", "sel"))
                _re_cur_idx = _enabled_label_options.index(_re_cat) if _re_cat in _enabled_label_options else 0
                _re_new_cat = st.selectbox(
                    "Category",
                    _enabled_label_options,
                    index=_re_cur_idx,
                    key=f"cat_tab_override_{_re_eid_safe}",
                    label_visibility="collapsed",
                )
                if _re_new_cat != _re_cat:
                    # Preserve extra tags when changing main category
                    _re_full_cat = ",".join([_re_new_cat] + _re_extras) if _re_extras else _re_new_cat
                    record_category_override(_sel_email, _re_new_cat)
                    _sel_email["category"] = _re_full_cat
                    st.session_state["selected_email_obj"]["category"] = _re_full_cat
                    for _e in st.session_state.get("emails", []):
                        if _e.get("id") == _sel_email.get("id"):
                            _e["category"] = _re_full_cat
                            break
                    st.rerun()

        # ── Suggested Categories ──
        st.markdown("---")
        # Only re-compute if flagged stale (after accepting a proposal or retrain)
        if "_cat_proposals" not in st.session_state or st.session_state.get("_cat_proposals_stale", True):
            _raw_proposals = propose_categories_from_history(min_sender_count=2)
            _proposals = filter_proposals_with_llm(_raw_proposals) if _raw_proposals else []
            if _proposals:
                st.session_state["_cat_proposals"] = _proposals
            else:
                st.session_state.pop("_cat_proposals", None)
            st.session_state["_cat_proposals_stale"] = False

        if st.session_state.get("_cat_proposals"):
            st.markdown(
                "<div style='display:flex;align-items:center;gap:8px;margin-bottom:8px;'>"
                "<span style='font-size:13px;font-weight:600;color:#94a3b8;'>Suggested categories</span>"
                "<span style='font-size:11px;color:#64748b;'>click to create</span>"
                "</div>",
                unsafe_allow_html=True,
            )
            # Render proposals as clickable pill buttons (no decorative HTML)
            _proposals = st.session_state["_cat_proposals"]
            # CSS for pill-shaped buttons
            st.markdown(
                "<style>"
                ".cat-pill button {border-radius:16px !important; padding:2px 14px !important; font-size:12px !important; "
                "background:#1e293b !important; border:1px solid #334155 !important; color:#e2e8f0 !important;}"
                ".cat-pill button:hover {background:#334155 !important; border-color:#475569 !important;}"
                "</style>",
                unsafe_allow_html=True,
            )
            _pill_cols = st.columns(min(len(_proposals), 4))
            for pi, prop in enumerate(_proposals):
                with _pill_cols[pi % min(len(_proposals), 4)]:
                    st.markdown("<div class='cat-pill'>", unsafe_allow_html=True)
                    if st.button(prop['proposed_name'], key=f"accept_prop_{pi}", use_container_width=True):
                        try:
                            from agent.email_memory import get_label_by_slug, _slugify, upsert_rule
                            _prop_slug = _slugify(prop["proposed_name"])
                            existing = get_label_by_slug(_prop_slug)
                            if existing:
                                new_lb = existing
                            else:
                                _rand_color = random.choice(_NICE_COLORS)
                                new_lb = create_label(prop["proposed_name"], color=_rand_color)
                            upsert_rule(prop["match_type"], prop["match_value"], new_lb["slug"])
                            # Apply to existing emails retroactively
                            _n_updated = apply_rule_to_existing_emails(prop["match_type"], prop["match_value"], new_lb["slug"])
                            # Update session emails in-memory too
                            _match_val = prop["match_value"].lower()
                            _match_type = prop["match_type"]
                            for _se in st.session_state.get("emails", []):
                                _se_cat = (_se.get("category", "informational") or "informational").strip()
                                _se_tags = [t.strip() for t in _se_cat.split(",")]
                                if new_lb["slug"] in _se_tags:
                                    continue
                                _se_matched = False
                                if _match_type == "subject_keyword":
                                    _se_matched = _match_val in (_se.get("subject") or "").lower()
                                elif _match_type == "sender":
                                    _se_s = (_se.get("sender") or "").lower()
                                    _se_addr = _se_s.split("<")[1].split(">")[0].strip() if "<" in _se_s and ">" in _se_s else (_se_s.strip() if "@" in _se_s else "")
                                    _se_matched = _se_addr == _match_val
                                elif _match_type == "sender_domain":
                                    _se_s = (_se.get("sender") or "").lower()
                                    _se_addr = _se_s.split("<")[1].split(">")[0].strip() if "<" in _se_s and ">" in _se_s else (_se_s.strip() if "@" in _se_s else "")
                                    if _se_addr and "@" in _se_addr:
                                        _se_matched = _se_addr.split("@", 1)[1] == _match_val
                                if _se_matched:
                                    _se_main = _se_tags[0] if _se_tags else "informational"
                                    _se_extras = _se_tags[1:]
                                    _se["category"] = ",".join([_se_main] + _se_extras + [new_lb["slug"]])
                            _msg = f"Created '{prop['proposed_name']}'"
                            if _n_updated:
                                _msg += f" and applied to {_n_updated} existing email{'s' if _n_updated != 1 else ''}"
                            st.success(_msg + ".")
                            st.session_state["_cat_proposals"].pop(pi)
                            st.session_state["_cat_proposals_stale"] = True
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))
                    st.markdown("</div>", unsafe_allow_html=True)

    elif active_tab == "Todo":
        # ── Todo List Panel ──────────────────────────────────────────
        st.markdown("### Todo")
        todo_items = get_todo_items()
        if not todo_items:
            st.markdown(
                "<div class='card-muted'>No todo items yet. Select an email and use quick actions to add tasks.</div>",
                unsafe_allow_html=True,
            )
        else:
            for item in todo_items:
                tcol1, tcol2 = st.columns([5, 1])
                with tcol1:
                    st.markdown(
                        f"<div style='font-size:13px; padding:4px 0;'>{html.escape(item['task'])}</div>",
                        unsafe_allow_html=True,
                    )
                with tcol2:
                    if st.button("✓", key=f"todo_done_{item['id']}", help="Mark as done"):
                        remove_todo_item(item["id"])
                        st.rerun()

    else:
        # ── Drafting Panel ───────────────────────────────────────────
        st.markdown("### Drafting")
        if not selected_email:
            st.markdown(
                "<div class='card-muted'>Select an email from the middle column to open the drafting workspace.</div>",
                unsafe_allow_html=True,
            )
        else:
            email = selected_email
            eid = safe_key(email["id"])

            # Show email body as Gmail-like rendered HTML
            body_html = email.get("body_html") or ""
            _plain = email.get("clean_body") or email.get("body") or email.get("snippet") or ""
            # Dynamic height: scale with content, min 80px, max 400px
            _text_len = len(body_html) if body_html else len(_plain)
            _body_height = min(400, max(80, _text_len // 3))
            if body_html:
                wrapped = (
                    '<div style="'
                    "font-family: 'Google Sans', Roboto, Arial, sans-serif;"
                    'font-size: 14px;'
                    'line-height: 1.6;'
                    'color: #e5e7eb;'
                    'background: #020617;'
                    'padding: 4px 0;'
                    'word-wrap: break-word;'
                    'overflow-wrap: break-word;'
                    '">'
                    '<style>body{background:#020617;margin:0;}'
                    'a{color:#38bdf8 !important;}'
                    'img{max-width:100%;height:auto;}'
                    '*{color:#e5e7eb !important;background-color:transparent !important;}'
                    '::selection{background:transparent !important;}'
                    '::-moz-selection{background:transparent !important;}'
                    'mark,span[style*=\"background\"],span[style*=\"highlight\"]{background:none !important;}'
                    '</style>'
                    + body_html + '</div>'
                )
                components.html(wrapped, height=_body_height, scrolling=True)
            else:
                if _plain:
                    safe = html.escape(html.unescape(_plain)).replace("\n", "<br>")
                    wrapped = (
                        '<div style="'
                        "font-family: 'Google Sans', Roboto, Arial, sans-serif;"
                        'font-size: 14px;'
                        'line-height: 1.6;'
                        'color: #e5e7eb;'
                        'padding: 4px 0;'
                        'word-wrap: break-word;'
                        'overflow-wrap: break-word;'
                        '">' + safe + '</div>'
                    )
                    components.html(wrapped, height=_body_height, scrolling=True)

            # ── Quick Actions ────────────────────────────────────────
            # Color-coded action buttons: orange=reply, purple=todo, grey=custom
            # Always derive from the email dict to pick up backfill/pipeline updates.
            pre = email.get("decision_options")
            options = []
            if pre and isinstance(pre, list) and len(pre) > 0:
                # Normalize: handle both new dict format and legacy string format
                normalized = []
                for item in pre:
                    if isinstance(item, dict) and item.get("label"):
                        normalized.append(item)
                    elif isinstance(item, str):
                        # Legacy format: "Reply: ...", "Todo: ...", "Schedule: ..."
                        s = item.strip()
                        if s.startswith("Reply:"):
                            normalized.append({"type": "reply", "label": s[6:].strip(), "context": s[6:].strip(), "has_meeting": False, "meeting_action": None})
                        elif s.startswith("Todo:"):
                            normalized.append({"type": "todo", "label": s[5:].strip(), "context": s[5:].strip()})
                        elif s.startswith("Schedule:"):
                            normalized.append({"type": "reply", "label": s[9:].strip(), "context": s[9:].strip(), "has_meeting": True, "meeting_action": "accept"})
                        elif s:
                            normalized.append({"type": "reply", "label": s, "context": s, "has_meeting": False, "meeting_action": None})
                options = normalized

            if options:
                # Sort: reply actions first, then todo
                def _qa_sort_key(o):
                    t = o.get("type", "reply") if isinstance(o, dict) else "reply"
                    return (0 if t == "reply" else 1, o.get("label", "") if isinstance(o, dict) else str(o))
                options = sorted(options, key=_qa_sort_key)

                st.markdown("**Quick actions**")
                for j, opt in enumerate(options):
                    if not isinstance(opt, dict):
                        continue
                    action_type = opt.get("type", "reply")
                    label = opt.get("label", "")
                    context = opt.get("context", label)
                    has_meeting = opt.get("has_meeting", False)
                    meeting_action = opt.get("meeting_action")

                    if action_type == "reply":
                        css_class = "qa-reply-btn"
                        with st.container(key=f"qa-{css_class}-{eid}-{j}"):
                            st.markdown(f"<div class='{css_class}'>", unsafe_allow_html=True)
                            if st.button(label, key=f"qa_{eid}_{j}", use_container_width=True):
                                # ── Meeting-aware action handling ──
                                if has_meeting and meeting_action == "accept":
                                    # Accept meeting: extract intent, auto-create if specific time known
                                    st.session_state[f"_last_decision_{eid}"] = f"Accept: {context}"
                                    if f"reply_generated_{eid}" in st.session_state:
                                        del st.session_state[f"reply_generated_{eid}"]
                                    with st.spinner("Scheduling meeting & drafting confirmation..."):
                                        try:
                                            sched_result = propose_meeting_times(email, days_ahead=7, max_slots=5, force=True, title_hint=context)
                                            slots = sched_result.get("free_slots", [])
                                            intent = sched_result.get("meeting_intent", {})
                                            has_specific_time = sched_result.get("specific_time_from_email", False)

                                            if has_specific_time and slots:
                                                # Email had a concrete time — auto-create, no picker
                                                from tools.zoom_tools import zoom_available
                                                s_start, s_end = slots[0]
                                                evt = create_event_from_slot(
                                                    meeting_intent=intent,
                                                    start_iso=s_start,
                                                    end_iso=s_end,
                                                    add_zoom=zoom_available(),
                                                )
                                                from datetime import datetime as _dt
                                                try:
                                                    dt = _dt.fromisoformat(s_start.replace("Z", "+00:00"))
                                                    slot_label = dt.strftime("%a %b %d, %I:%M %p")
                                                except Exception:
                                                    slot_label = s_start
                                                meeting_summary = evt.get('summary', 'Meeting')
                                                add_todo_item(task=f"Meeting: {meeting_summary} on {slot_label}", email_id=email.get("id", ""))
                                                st.success(f"Event auto-created: {meeting_summary} — {slot_label}")
                                                # Draft confirmation
                                                try:
                                                    current_style = st.session_state.get("learned_style") or load_persisted_style()
                                                    draft_text = run_drafting_graph(email, decision=f"Accept the meeting/event. {context}. Confirmed for {slot_label}.", style_notes=current_style)
                                                    if draft_text:
                                                        st.session_state[f"reply_generated_{eid}"] = draft_text
                                                except Exception:
                                                    pass
                                            elif slots:
                                                # No specific time — show time picker panel
                                                st.session_state[f"schedule_pending_{eid}"] = True
                                                st.session_state[f"schedule_desc_{eid}"] = context
                                                st.session_state[f"schedule_mode_{eid}"] = "accept"
                                                st.session_state[f"schedule_result_{eid}"] = sched_result
                                            else:
                                                st.warning("No available time slots found.")
                                        except Exception as e:
                                            st.warning(f"Could not auto-schedule: {e}")
                                        # Only draft if no schedule_pending AND no draft already generated
                                        # (auto-create path already drafts its own confirmation above)
                                        if not st.session_state.get(f"schedule_pending_{eid}") and f"reply_generated_{eid}" not in st.session_state:
                                            try:
                                                current_style = st.session_state.get("learned_style") or load_persisted_style()
                                                draft_text = run_drafting_graph(email, decision=f"Accept the meeting/event. {context}", style_notes=current_style)
                                                if draft_text:
                                                    st.session_state[f"reply_generated_{eid}"] = draft_text
                                            except Exception:
                                                pass
                                    st.rerun()
                                elif has_meeting and meeting_action == "decline":
                                    # Decline meeting: draft polite decline
                                    st.session_state[f"_last_decision_{eid}"] = f"Decline: {context}"
                                    if f"reply_generated_{eid}" in st.session_state:
                                        del st.session_state[f"reply_generated_{eid}"]
                                    with st.spinner("Drafting decline..."):
                                        try:
                                            current_style = st.session_state.get("learned_style") or load_persisted_style()
                                            draft_text = run_drafting_graph(email, decision=f"Politely decline. {context}", style_notes=current_style)
                                            if draft_text:
                                                st.session_state[f"reply_generated_{eid}"] = draft_text
                                        except Exception as e:
                                            st.error(f"Error: {e}")
                                    st.rerun()
                                elif has_meeting and meeting_action == "reschedule":
                                    # Reschedule: show time slots first
                                    st.session_state[f"schedule_pending_{eid}"] = True
                                    st.session_state[f"schedule_desc_{eid}"] = context
                                    st.session_state[f"schedule_mode_{eid}"] = "reschedule"
                                    st.rerun()
                                else:
                                    # Normal reply: draft reply
                                    st.session_state[f"_last_decision_{eid}"] = context
                                    if f"reply_generated_{eid}" in st.session_state:
                                        del st.session_state[f"reply_generated_{eid}"]
                                    with st.spinner("Drafting reply..."):
                                        try:
                                            current_style = st.session_state.get("learned_style") or load_persisted_style()
                                            draft_text = run_drafting_graph(email, decision=context, style_notes=current_style)
                                            if draft_text:
                                                st.session_state[f"reply_generated_{eid}"] = draft_text
                                                st.success("Draft generated.")
                                            else:
                                                st.error("Failed to generate draft.")
                                        except Exception as e:
                                            st.error(f"Error drafting reply: {str(e)}")
                                    st.rerun()
                            st.markdown("</div>", unsafe_allow_html=True)

                    elif action_type == "todo":
                        css_class = "qa-todo-btn"
                        with st.container(key=f"qa-{css_class}-{eid}-{j}"):
                            st.markdown(f"<div class='{css_class}'>", unsafe_allow_html=True)
                            if st.button(label, key=f"qa_{eid}_{j}", use_container_width=True):
                                add_todo_item(task=context or label, email_id=email.get("id", ""))
                                st.success("Added to todo list.")
                                st.rerun()
                            st.markdown("</div>", unsafe_allow_html=True)
            else:
                # No quick actions — show muted message if available
                no_action_msg = email.get("no_action_message", "")
                if no_action_msg:
                    st.markdown(f'<p style="color:#999;font-style:italic;">{no_action_msg}</p>', unsafe_allow_html=True)

            # ── Scheduling flow (triggered by reschedule action) ──────
            if st.session_state.get(f"schedule_pending_{eid}"):
                sched_key = f"schedule_result_{eid}"
                _sched_desc = st.session_state.get(f"schedule_desc_{eid}", "")
                if sched_key not in st.session_state:
                    with st.spinner("Finding free time slots..."):
                        try:
                            result = propose_meeting_times(email, days_ahead=7, max_slots=5, force=True, title_hint=_sched_desc)
                            st.session_state[sched_key] = result
                        except Exception as e:
                            st.error(f"Scheduling error: {e}")
                            st.session_state[sched_key] = {"free_slots": [], "error": str(e)}

                sched = st.session_state.get(sched_key, {})
                intent = sched.get("meeting_intent", {})
                slots = sched.get("free_slots", [])
                sched_err = sched.get("error")
                _is_reschedule = st.session_state.get(f"schedule_mode_{eid}") == "reschedule"

                if sched_err and not slots:
                    st.warning(f"Could not find slots: {sched_err}")
                elif slots:
                    st.markdown("**Pick a time:**" if _is_reschedule else "**Available time slots:**")
                    for si, (s_start, s_end) in enumerate(slots):
                        try:
                            from datetime import datetime as _dt
                            dt = _dt.fromisoformat(s_start.replace("Z", "+00:00"))
                            slot_label = dt.strftime("%a %b %d, %I:%M %p")
                        except Exception:
                            slot_label = s_start
                        if st.button(slot_label, key=f"slot_{eid}_{si}", use_container_width=True):
                            with st.spinner("Creating event & drafting response..."):
                                try:
                                    from tools.zoom_tools import zoom_available
                                    evt = create_event_from_slot(
                                        meeting_intent=intent,
                                        start_iso=s_start,
                                        end_iso=s_end,
                                        add_zoom=zoom_available(),
                                    )
                                    meeting_summary = evt.get('summary', 'Meeting')
                                    add_todo_item(
                                        task=f"Meeting: {meeting_summary} on {slot_label}",
                                        email_id=email.get("id", ""),
                                    )
                                    st.success(f"Event created: {meeting_summary}")
                                    # Draft appropriate reply based on mode
                                    try:
                                        current_style = st.session_state.get("learned_style") or load_persisted_style()
                                        if _is_reschedule:
                                            draft_decision = f"Suggest alternative time: {slot_label}. {_sched_desc}"
                                            decision_label = f"Suggest time: {slot_label}"
                                        else:
                                            draft_decision = f"Accept the meeting/event. {_sched_desc}. Confirmed for {slot_label}."
                                            decision_label = f"Accept: {slot_label}"
                                        draft_text = run_drafting_graph(
                                            email,
                                            decision=draft_decision,
                                            style_notes=current_style,
                                        )
                                        if draft_text:
                                            st.session_state[f"reply_generated_{eid}"] = draft_text
                                            st.session_state[f"_last_decision_{eid}"] = decision_label
                                    except Exception:
                                        pass
                                    # Clean up scheduling state
                                    st.session_state.pop(f"schedule_pending_{eid}", None)
                                    st.session_state.pop(f"schedule_desc_{eid}", None)
                                    st.session_state.pop(f"schedule_mode_{eid}", None)
                                    st.session_state.pop(sched_key, None)
                                except Exception as e:
                                    st.error(f"Failed to create event: {e}")
                            st.rerun()
                    if st.button("Cancel", key=f"sched_cancel_{eid}"):
                        st.session_state.pop(f"schedule_pending_{eid}", None)
                        st.session_state.pop(f"schedule_desc_{eid}", None)
                        st.session_state.pop(f"schedule_mode_{eid}", None)
                        st.session_state.pop(sched_key, None)
                        st.rerun()
                else:
                    st.info("No available slots found in the next 7 days.")
                    if st.button("Cancel", key=f"sched_cancel2_{eid}"):
                        st.session_state.pop(f"schedule_pending_{eid}", None)
                        st.session_state.pop(f"schedule_desc_{eid}", None)
                        st.session_state.pop(f"schedule_mode_{eid}", None)
                        st.session_state.pop(sched_key, None)
                        st.rerun()

            if st.session_state.get(f"reply_custom_{eid}"):
                custom = st.text_input("Custom instruction", key=f"custom_instr_{eid}")
                if st.button("Use and draft", key=f"use_custom_{eid}"):
                    if f"reply_generated_{eid}" in st.session_state:
                        del st.session_state[f"reply_generated_{eid}"]
                    with st.spinner("Drafting reply..."):
                        try:
                            current_style = st.session_state.get("learned_style") or load_persisted_style()
                            draft_text = run_drafting_graph(email, decision=custom, style_notes=current_style)
                            if draft_text:
                                st.session_state[f"reply_generated_{eid}"] = draft_text
                                st.success("Draft generated.")
                            else:
                                st.error("Failed to generate draft.")
                        except Exception as e:
                            st.error(f"Error drafting reply: {str(e)}")
                    st.rerun()

            if st.session_state.get(f"reply_generated_{eid}"):
                st.markdown("---")
                st.markdown("**Generated draft**")
                draft_val = st.session_state.get(f"reply_generated_{eid}")
                st.text_area(
                    "Draft preview",
                    value=draft_val,
                    height=220,
                    key=f"preview_{eid}",
                    label_visibility="collapsed",
                )
                if st.button("Send now", key=f"send_now_{eid}", use_container_width=True):
                    final_draft = st.session_state.get(f"preview_{eid}", draft_val)
                    with st.spinner("Sending message..."):
                        try:
                            to = extract_email_address(email["sender"])
                            subject = email.get("subject", "")
                            if not subject.lower().startswith("re:"):
                                subject = f"Re: {subject}"
                            sent = send_email(
                                to=to,
                                subject=subject,
                                body=final_draft,
                                thread_id=email.get("thread_id"),
                            )
                            st.success("Message sent.")
                            del st.session_state[f"reply_generated_{eid}"]
                        except Exception as e:
                            st.error(f"Failed to send message: {e}")
            _custom_col, _archive_col = st.columns([1, 1])
            with _custom_col:
                with st.container(key=f"qa-custom-{eid}"):
                    if st.button("Custom instruction", key=f"qo_{eid}_custom", use_container_width=True):
                        st.session_state[f"reply_custom_{eid}"] = True
                        st.rerun()
            with _archive_col:
                with st.container(key=f"qa-archive-{eid}"):
                    if st.button("Archive", key=f"archive_{eid}", use_container_width=True):
                        _arc_id = email.get("id", "")
                        archive_email(_arc_id)
                        mark_email_archived(_arc_id)
                        st.session_state["emails"] = [e for e in st.session_state.get("emails", []) if e.get("id") != _arc_id]
                        st.session_state.pop("selected_email", None)
                        st.session_state.pop("selected_email_obj", None)
                        st.success("Archived.")
                        st.rerun()

with email_col:
    # ── Category filter row ──
    _filter_labels = get_all_labels()
    _hidden_cats = st.session_state.get("_hidden_cats", set())

    _hdr_col, _filter_col = st.columns([1, 2])
    with _hdr_col:
        _n_hidden = len(_hidden_cats)
        if _n_hidden:
            st.markdown(f"### Emails <span style='font-size:11px;font-weight:400;color:{muted_color};'>&middot; {_n_hidden} hidden</span>", unsafe_allow_html=True)
        else:
            st.markdown("### Emails")
    with _filter_col:
        # Multiselect showing visible categories — all selected by default
        _all_slugs = [lb["slug"] for lb in _filter_labels]
        _all_names = {lb["slug"]: lb.get("display_name", lb["slug"]) for lb in _filter_labels}
        _visible = [s for s in _all_slugs if s not in _hidden_cats]
        _selected = st.multiselect(
            "Filter",
            options=_all_slugs,
            default=_visible,
            format_func=lambda s: _all_names.get(s, s),
            key="_cat_multifilter",
            label_visibility="collapsed",
            placeholder="Filter categories...",
        )
        # Sync selection back to hidden state
        _new_hidden = set(_all_slugs) - set(_selected)
        if _new_hidden != _hidden_cats:
            st.session_state["_hidden_cats"] = _new_hidden

    # ── Search bar ──
    _search_q = st.text_input("Search", key="_email_search", placeholder="Filter by subject or sender...", label_visibility="collapsed")
    if _search_q:
        _sq = _search_q.lower()
        filtered = [e for e in filtered if _sq in (e.get("subject", "") or "").lower() or _sq in (e.get("sender", "") or "").lower()]

    # ── Pagination ──
    _PAGE_SIZE = 20
    _total_pages = max(1, (len(filtered) + _PAGE_SIZE - 1) // _PAGE_SIZE)
    _current_page = st.session_state.get("_email_page", 1)
    if _current_page > _total_pages:
        _current_page = _total_pages
    _page_start = (_current_page - 1) * _PAGE_SIZE
    _page_end = _page_start + _PAGE_SIZE
    _page_emails = filtered[_page_start:_page_end]

    # Pre-load label colors and display names for badges
    _label_color_map = {}
    _label_display_map = {}
    try:
        for _lb in get_all_labels():
            _label_color_map[_lb["slug"]] = _lb["color"]
            _label_display_map[_lb["slug"]] = _lb["display_name"]
    except Exception:
        pass

    for i, email in enumerate(_page_emails):
        cat_raw = (email.get("category", "informational") or "informational").strip()
        cat_tags = [t.strip() for t in cat_raw.split(",") if t.strip()]
        subject = email.get("subject", "(No subject)")
        sender_raw = email.get("sender", "")
        date_raw = email.get("date", "")
        summary = email.get("summary", email.get("snippet", ""))

        # Extract sender name (strip email address)
        sender_match = re.match(r'^([^<]+)', sender_raw)
        sender_name = sender_match.group(1).strip() if sender_match else sender_raw
        if not sender_name:
            sender_name = sender_raw.split('@')[0] if '@' in sender_raw else sender_raw

        # Simplify date: "Fri, 27 Feb 2026 09:49:45 -0500" -> "Fri, 27 Feb 2026"
        date_match = re.match(r'^([A-Za-z]{3},?\s*\d{1,2}\s+[A-Za-z]{3}\s+\d{4})', date_raw)
        date_simple = date_match.group(1) if date_match else date_raw.split(' ')[0] if date_raw else ""

        # Escape user / email content to avoid breaking HTML and causing frontend errors
        subject_html = html.escape(str(subject))
        sender_html = html.escape(str(sender_name))
        date_html = html.escape(str(date_simple))
        summary_html = html.escape(str(summary))

        # Build badges — one per tag (system + custom), skip deleted/unknown tags
        _badge_parts = []
        for _tag in cat_tags:
            if _tag not in _label_display_map:
                continue  # tag was deleted or unknown — don't render
            _cb_color = _label_color_map.get(_tag, "#94a3b8")
            _cb_display = _label_display_map[_tag]
            _badge_parts.append(f"<span class='badge-category' style='color:{_cb_color};border-color:{_cb_color}40;background:{_cb_color}18;'>{html.escape(str(_cb_display))}</span>")
        badges = " ".join(_badge_parts)

        # Each email in its own container for CSS targeting
        eid_html = html.escape(str(email.get("id", "")), quote=True)
        email_container = st.container()
        with email_container:
            st.markdown(
                f"""<div class='email-card' data-email-id='{eid_html}'>
                    <div style='display:flex; justify-content:space-between; align-items:flex-start; gap:8px;'>
                        <div class='email-subject'>{subject_html}</div>
                        <div style='flex-shrink:0'>{badges}</div>
                    </div>
                    <div class='email-sender'>{sender_html} &middot; {date_html}</div>
                    <div class='email-summary'>{summary_html}</div>
                </div>""",
                unsafe_allow_html=True,
            )
            if st.button(" ", key=f"open_{safe_key(email['id'])}"):
                st.session_state["selected_email"] = email["id"]
                st.session_state[f"selected_email_obj"] = email
                st.rerun()

    # Inject JS via iframe to make email cards clickable
    components.html("""
    <script>
    (function() {
        try {
            var doc = window.parent.document;
            // Guard: attach the delegated handler only once
            if (doc._emailCardClickReady) return;
            doc._emailCardClickReady = true;

            // Single delegated click handler — never needs re-attachment,
            // works regardless of DOM mutations or Streamlit rerenders.
            doc.addEventListener('click', function(e) {
                var card = e.target.closest('.email-card');
                if (!card) return;
                // Walk up to the stVerticalBlock that wraps this card
                var block = card;
                while (block && !block.matches('[data-testid="stVerticalBlock"]')) {
                    block = block.parentElement;
                }
                if (!block) return;
                var btn = block.querySelector('button');
                if (btn) {
                    e.preventDefault();
                    e.stopPropagation();
                    btn.click();
                }
            }, true); // capture phase — runs before React handlers
        } catch(_) {}
    })();
    </script>
    """, height=0)

    # ── Pagination controls ──
    if _total_pages > 1:
        _pg_left, _pg_mid, _pg_right = st.columns([1, 2, 1])
        with _pg_left:
            if st.button("← Prev", key="_page_prev", disabled=_current_page <= 1):
                st.session_state["_email_page"] = _current_page - 1
                st.rerun()
        with _pg_mid:
            st.markdown(f"<div style='text-align:center;font-size:12px;color:{muted_color};padding-top:8px;'>Page {_current_page} of {_total_pages} ({len(filtered)} emails)</div>", unsafe_allow_html=True)
        with _pg_right:
            if st.button("Next →", key="_page_next", disabled=_current_page >= _total_pages):
                st.session_state["_email_page"] = _current_page + 1
                st.rerun()
