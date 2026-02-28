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
from tools.gmail_tools import send_email, extract_email_address
from agent.langgraph_pipeline import run_email_pipeline, load_from_memory
from agent.email_memory import (
    get_email_count,
    add_todo_item, get_todo_items, remove_todo_item,
    get_all_labels, get_enabled_labels, create_label, delete_label,
    record_category_override, propose_categories_from_history,
    ensure_default_labels, apply_rule_to_existing_emails,
)
from agent.decision_suggester import suggest_decision
from agent.drafter import draft_reply
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
        border-radius: 8px;
        padding: 6px 18px;
        background: {accent_color};
        color: #ffffff;
        border: none;
        font-size: 13px;
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
    div[class*="qa-reply-"] .stButton>button {{
        background: #0ea5e9;
    }}
    div[class*="qa-reply-"] .stButton>button:hover {{
        background: #0284c7;
        box-shadow: 0 2px 8px rgba(14,165,233,0.35);
    }}
    div[class*="qa-todo-"] .stButton>button {{
        background: #f59e0b;
        color: #1e1b0f;
    }}
    div[class*="qa-todo-"] .stButton>button:hover {{
        background: #d97706;
        box-shadow: 0 2px 8px rgba(245,158,11,0.35);
    }}
    div[class*="qa-sched-"] .stButton>button {{
        background: #8b5cf6;
    }}
    div[class*="qa-sched-"] .stButton>button:hover {{
        background: #7c3aed;
        box-shadow: 0 2px 8px rgba(139,92,246,0.35);
    }}
    /* Remove extra padding/gap inside keyed QA containers */
    div[class*="qa-reply-"],
    div[class*="qa-todo-"],
    div[class*="qa-sched-"] {{
        gap: 0 !important;
    }}

    .email-scroll {{
        max-height: calc(100vh - 260px);
        overflow-y: auto;
        padding-right: 4px;
    }}

    /* Email card item */
    .email-card {{
        padding: 12px 14px;
        border-radius: 10px;
        border: 1px solid {border_soft};
        background: {'rgba(15,23,42,0.5)' if theme == 'dark' else 'rgba(255,255,255,0.8)'};
        margin-bottom: 10px;
        transition: border-color 0.15s ease, background 0.15s ease;
        cursor: pointer;
    }}
    .email-card:hover {{
        border-color: {accent_color};
        background: {'rgba(56,189,248,0.07)' if theme == 'dark' else 'rgba(56,189,248,0.08)'};
    }}

    .email-subject {{
        font-size: 14px;
        font-weight: 600;
        color: {text_color};
        margin-bottom: 2px;
    }}
    .email-sender {{
        font-size: 12px;
        color: {muted_color};
        margin-bottom: 4px;
    }}
    .email-summary {{
        font-size: 13px;
        color: {'#cbd5e1' if theme == 'dark' else '#374151'};
        line-height: 1.4;
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
        background: {'rgba(148,163,184,0.1)' if theme == 'dark' else 'rgba(107,114,128,0.1)'};
        color: {muted_color};
        border: 1px solid {'rgba(148,163,184,0.2)' if theme == 'dark' else 'rgba(107,114,128,0.2)'};
    }}

    /* Style the open-email buttons to be invisible overlays */
    /* Target stVerticalBlock containers that have an email-card inside */
    [data-testid="stVerticalBlock"]:has(.email-card) {{
        position: relative !important;
    }}
    [data-testid="stVerticalBlock"]:has(.email-card) > [data-testid="stElementContainer"]:has(button) {{
        position: absolute !important;
        top: 0 !important;
        left: 0 !important;
        right: 0 !important;
        bottom: 0 !important;
        z-index: 10 !important;
    }}
    [data-testid="stVerticalBlock"]:has(.email-card) > [data-testid="stElementContainer"]:has(button) button {{
        width: 100% !important;
        height: 100% !important;
        opacity: 0 !important;
        cursor: pointer !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 0 !important;
        min-height: 0 !important;
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
MIN_FETCH = 20  # minimum emails to fetch on first load
MAX_FETCH = 20  # fetch more when catching up on new emails

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
        cached = load_from_memory()
        if cached:
            st.session_state["emails"] = cached
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

# After header widgets are created, map current model to provider
llm_provider_map = {
    "Groq (hosted)": "groq",
    "Qwen 2.5 3B (local)": "qwen_local_3b",
    "Qwen 2.5 7B (local)": "qwen_local_7b",
}
current_llm_display = st.session_state.get("llm_backend", default_llm_display)
os.environ["LLM_PROVIDER"] = llm_provider_map.get(current_llm_display, "qwen_local_3b")

# Email fetching and processing workflow (LangGraph pipeline)
if st.session_state.get("is_analyzing", False):
    actual_query = "in:inbox"
    retrain_mode = st.session_state.pop("_retrain", False)
    spinner_msg = (
        "Retraining: reprocessing all stored emails..."
        if retrain_mode
        else "Analyzing inbox and preparing suggestions..."
    )
    with st.spinner(spinner_msg):
        try:
            # If we already have emails, fetch a larger batch to catch all
            # recent ones; otherwise just get the initial MIN_FETCH.
            stored_count = get_email_count()
            fetch_count = MAX_FETCH if stored_count >= MIN_FETCH else MIN_FETCH

            state = run_email_pipeline(
                query=actual_query,
                max_emails=fetch_count,
                unread_only=False,
                retrain=retrain_mode,
            )
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
        except Exception as e:
            st.session_state["pipeline_error"] = str(e)
            st.rerun()
        finally:
            st.session_state["is_analyzing"] = False
    st.rerun()

# Display emails from session
if "emails" not in st.session_state:
    st.stop()

emails = st.session_state["emails"]

filtered = emails

# Apply category filter if active
_active_cat_filter = st.session_state.get("_cat_filter")
if _active_cat_filter:
    if _active_cat_filter == "__needs_action__":
        filtered = [e for e in filtered if e.get("needs_action")]
    else:
        filtered = [e for e in filtered if _active_cat_filter in [c.strip() for c in (e.get("category", "normal") or "normal").split(",")]]

# Always sort by newest first
filtered.sort(key=lambda x: x.get("internal_date", 0), reverse=True)

# Two-column layout:
# left = drafting workspace, right = scrollable emails
draft_col, email_col = st.columns([1.1, 2.3])

sel_id = st.session_state.get("selected_email")
selected_email = None
if sel_id:
    selected_email = st.session_state.get("selected_email_obj")
    if not selected_email:
        selected_email = next((e for e in emails if e.get("id") == sel_id), None)

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

        # ── "Needs action" pseudo-filter ──
        _na_active = st.session_state.get("_cat_filter") == "__needs_action__"
        _na_c1, _na_c2, _na_c3 = st.columns([3, 1, 1])
        with _na_c1:
            st.markdown(
                "<div style='font-size:13px;padding:4px 0;'>"
                "<span style='display:inline-block;width:10px;height:10px;border-radius:50%;"
                f"background:{accent_orange};margin-right:6px;vertical-align:middle;'></span>"
                "<b>Needs Action</b></div>",
                unsafe_allow_html=True,
            )
        with _na_c3:
            _na_label = "Unfilter" if _na_active else "Filter"
            if st.button(_na_label, key="filter_cat___needs_action__"):
                if _na_active:
                    st.session_state.pop("_cat_filter", None)
                else:
                    st.session_state["_cat_filter"] = "__needs_action__"
                st.rerun()

        # ── Existing labels ──
        if not all_labels:
            st.markdown("<div class='card-muted'>No categories yet.</div>", unsafe_allow_html=True)
        else:
            for lb in all_labels:
                slug = lb["slug"]
                c_left, c_mid, c_right = st.columns([3, 1, 1])
                with c_left:
                    color_dot = f"<span style='display:inline-block;width:10px;height:10px;border-radius:50%;background:{lb['color']};margin-right:6px;vertical-align:middle;'></span>"
                    st.markdown(
                        f"<div style='font-size:13px;padding:4px 0;'>{color_dot}<b>{html.escape(lb['display_name'])}</b></div>",
                        unsafe_allow_html=True,
                    )
                with c_mid:
                    if st.button("Del", key=f"del_cat_{slug}"):
                        delete_label(slug)
                        st.rerun()
                with c_right:
                    _filter_active = st.session_state.get("_cat_filter") == slug
                    _filter_label = "Unfilter" if _filter_active else "Filter"
                    if st.button(_filter_label, key=f"filter_cat_{slug}"):
                        if _filter_active:
                            st.session_state.pop("_cat_filter", None)
                        else:
                            st.session_state["_cat_filter"] = slug
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
            _enabled_label_options = ["urgent", "important", "normal", "informational", "newsletter"]

        _sel_email = st.session_state.get("selected_email_obj")
        if not _sel_email:
            st.markdown("<div class='card-muted'>Select an email to change its category.</div>", unsafe_allow_html=True)
        else:
            _re_cat_raw = _sel_email.get("category", "normal") or "normal"
            _re_cat_parts = [c.strip() for c in _re_cat_raw.split(",") if c.strip()]
            _re_cat = _re_cat_parts[0]  # primary category
            _re_subj = html.escape(str(_sel_email.get("subject", "(No subject)")))
            _re_sender_raw = _sel_email.get("sender", "")
            _re_sender_match = re.match(r'^([^<]+)', _re_sender_raw)
            _re_sender = html.escape((_re_sender_match.group(1).strip() if _re_sender_match else _re_sender_raw)[:40])
            _re_cat_color = _label_color_map.get(_re_cat, "#94a3b8")

            # Build display showing all categories
            _re_cat_display_parts = []
            for _rcp in _re_cat_parts:
                _rcp_color = _label_color_map.get(_rcp, "#94a3b8")
                _re_cat_display_parts.append(f"<span style='color:{_rcp_color};font-weight:600;'>[{html.escape(_rcp)}]</span>")
            _re_cat_display = " ".join(_re_cat_display_parts)

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
                    # Replace primary category, keep secondary if present
                    _re_cat_parts[0] = _re_new_cat
                    _re_final_cat = ",".join(_re_cat_parts)
                    record_category_override(_sel_email, _re_new_cat)
                    _sel_email["category"] = _re_final_cat
                    st.session_state["selected_email_obj"]["category"] = _re_final_cat
                    # Also update in the emails list
                    for _e in st.session_state.get("emails", []):
                        if _e.get("id") == _sel_email.get("id"):
                            _e["category"] = _re_final_cat
                            break
                    st.rerun()

        # ── Suggested Categories (auto-triggered on every tab open) ──
        st.markdown("---")
        # Fetch proposals each time the Categories tab is shown
        # (cheap DB query, always up-to-date)
        _proposals = propose_categories_from_history(min_sender_count=2)
        if _proposals:
            st.session_state["_cat_proposals"] = _proposals
        else:
            st.session_state.pop("_cat_proposals", None)

        if st.session_state.get("_cat_proposals"):
            st.markdown("**Suggested Categories**")
            # Muted pill styling for suggestion buttons
            st.markdown(
                "<style>"
                "div[data-testid='stVerticalBlock'] .suggest-cat-wrap .stButton>button {"
                "  background: transparent !important;"
                "  border: 1px solid #475569 !important;"
                "  color: #cbd5e1 !important;"
                "  font-weight: 400 !important;"
                "  font-size: 13px !important;"
                "}"
                "div[data-testid='stVerticalBlock'] .suggest-cat-wrap .stButton>button:hover {"
                "  background: #334155 !important;"
                "  border-color: #64748b !important;"
                "  color: #f1f5f9 !important;"
                "  box-shadow: none !important;"
                "}"
                "</style>",
                unsafe_allow_html=True,
            )
            _sug_container = st.container()
            _sug_container.markdown('<div class="suggest-cat-wrap">', unsafe_allow_html=True)
            with _sug_container:
                for pi, prop in enumerate(st.session_state["_cat_proposals"]):
                    if st.button(prop["proposed_name"], key=f"accept_prop_{pi}", use_container_width=True):
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
                            # Apply the new rule to all existing emails (up to 2 categories)
                            _applied = apply_rule_to_existing_emails(prop["match_type"], prop["match_value"], new_lb["slug"])
                            # Refresh in-memory emails so UI reflects changes immediately
                            if _applied and "emails" in st.session_state:
                                for _se in st.session_state["emails"]:
                                    _se_cats = [c.strip() for c in (_se.get("category", "normal") or "normal").split(",") if c.strip()]
                                    if new_lb["slug"] not in _se_cats and len(_se_cats) < 2:
                                        # Re-check rule match in memory
                                        _se_subj = (_se.get("subject") or "").lower()
                                        _se_sender = (_se.get("sender") or "").lower()
                                        _mv = prop["match_value"].lower().strip()
                                        _mt = prop["match_type"]
                                        _mem_match = False
                                        if _mt == "subject_keyword" and _mv in _se_subj:
                                            _mem_match = True
                                        elif _mt == "sender":
                                            _addr = _se_sender.split("<")[1].split(">")[0].strip() if "<" in _se_sender else _se_sender.strip()
                                            _mem_match = _addr == _mv
                                        elif _mt == "sender_domain":
                                            _addr = _se_sender.split("<")[1].split(">")[0].strip() if "<" in _se_sender else _se_sender.strip()
                                            if "@" in _addr:
                                                _mem_match = _addr.split("@", 1)[1] == _mv
                                        if _mem_match:
                                            _se_cats.append(new_lb["slug"])
                                            _se["category"] = ",".join(_se_cats)
                            st.success(f"Created '{prop['proposed_name']}' – applied to {_applied} email{'s' if _applied != 1 else ''}")
                            st.session_state["_cat_proposals"].pop(pi)
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))
            _sug_container.markdown('</div>', unsafe_allow_html=True)

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
                components.html(wrapped, height=400, scrolling=True)
            else:
                body_text = email.get("clean_body") or email.get("body") or email.get("snippet") or ""
                if body_text:
                    safe = html.escape(html.unescape(body_text)).replace("\n", "<br>")
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
                    components.html(wrapped, height=400, scrolling=True)

            # ── Quick Actions ────────────────────────────────────────
            options_key = f"reply_options_{eid}"
            options = st.session_state.get(options_key)
            if options is None:
                pre = email.get("decision_options")
                if pre and isinstance(pre, list) and len(pre) > 0:
                    options = pre
                else:
                    options = []
                st.session_state[options_key] = options

            if options:
                # Sort: Reply first, Schedule second, Todo last
                def _qa_sort_key(o):
                    if o.startswith("Reply:"):
                        return (0, o)
                    elif o.startswith("Schedule:"):
                        return (1, o)
                    elif o.startswith("Todo:"):
                        return (2, o)
                    return (3, o)
                options = sorted(options, key=_qa_sort_key)

                st.markdown("**Quick actions**")
                for j, opt in enumerate(options):
                    if opt.startswith("Reply:"):
                        with st.container(key=f"qa-reply-{eid}-{j}"):
                            if st.button(opt, key=f"qa_{eid}_{j}", use_container_width=True):
                                reply_desc = opt[len("Reply:"):].strip()
                                if f"reply_generated_{eid}" in st.session_state:
                                    del st.session_state[f"reply_generated_{eid}"]
                                with st.spinner("Drafting reply..."):
                                    try:
                                        current_style = st.session_state.get("learned_style") or load_persisted_style()
                                        draft_text = draft_reply(email, decision=reply_desc, style_notes=current_style)
                                        if draft_text:
                                            st.session_state[f"reply_generated_{eid}"] = draft_text
                                            st.success("Draft generated.")
                                        else:
                                            st.error("Failed to generate draft.")
                                    except Exception as e:
                                        st.error(f"Error drafting reply: {str(e)}")
                                st.rerun()
                    elif opt.startswith("Todo:"):
                        with st.container(key=f"qa-todo-{eid}-{j}"):
                            if st.button(opt, key=f"qa_{eid}_{j}", use_container_width=True):
                                task_text = opt[len("Todo:"):].strip()
                                add_todo_item(task=task_text, email_id=email.get("id", ""))
                                st.success(f"Added to todo list.")
                                st.rerun()
                    elif opt.startswith("Schedule:"):
                        with st.container(key=f"qa-sched-{eid}-{j}"):
                            if st.button(opt, key=f"qa_{eid}_{j}", use_container_width=True):
                                st.session_state[f"schedule_pending_{eid}"] = True
                                st.rerun()

            # ── Scheduling flow (triggered by Schedule: action) ──────
            if st.session_state.get(f"schedule_pending_{eid}"):
                sched_key = f"schedule_result_{eid}"
                if sched_key not in st.session_state:
                    with st.spinner("Finding free time slots..."):
                        try:
                            result = propose_meeting_times(email, days_ahead=7, max_slots=5, force=True)
                            st.session_state[sched_key] = result
                        except Exception as e:
                            st.error(f"Scheduling error: {e}")
                            st.session_state[sched_key] = {"free_slots": [], "error": str(e)}

                sched = st.session_state.get(sched_key, {})
                intent = sched.get("meeting_intent", {})
                slots = sched.get("free_slots", [])
                sched_err = sched.get("error")

                if sched_err and not slots:
                    st.warning(f"Could not find slots: {sched_err}")
                elif slots:
                    st.markdown("**Available time slots:**")
                    for si, (s_start, s_end) in enumerate(slots):
                        try:
                            from datetime import datetime as _dt
                            dt = _dt.fromisoformat(s_start.replace("Z", "+00:00"))
                            slot_label = dt.strftime("%a %b %d, %I:%M %p")
                        except Exception:
                            slot_label = s_start
                        if st.button(slot_label, key=f"slot_{eid}_{si}", use_container_width=True):
                            with st.spinner("Creating calendar event..."):
                                try:
                                    from tools.zoom_tools import zoom_available
                                    evt = create_event_from_slot(
                                        meeting_intent=intent,
                                        start_iso=s_start,
                                        end_iso=s_end,
                                        add_zoom=zoom_available(),
                                    )
                                    # Add a todo item for the scheduled meeting
                                    meeting_summary = evt.get('summary', 'Meeting')
                                    add_todo_item(
                                        task=f"Meeting: {meeting_summary} on {slot_label}",
                                        email_id=email.get("id", ""),
                                    )
                                    st.success(f"Event created: {meeting_summary}")
                                    # Clean up scheduling state
                                    st.session_state.pop(f"schedule_pending_{eid}", None)
                                    st.session_state.pop(sched_key, None)
                                except Exception as e:
                                    st.error(f"Failed to create event: {e}")
                            st.rerun()
                    if st.button("Cancel scheduling", key=f"sched_cancel_{eid}"):
                        st.session_state.pop(f"schedule_pending_{eid}", None)
                        st.session_state.pop(sched_key, None)
                        st.rerun()
                else:
                    st.info("No available slots found in the next 7 days.")
                    if st.button("Cancel", key=f"sched_cancel2_{eid}"):
                        st.session_state.pop(f"schedule_pending_{eid}", None)
                        st.session_state.pop(sched_key, None)
                        st.rerun()

            if st.button("Custom instruction", key=f"qo_{eid}_custom", use_container_width=True):
                st.session_state[f"reply_custom_{eid}"] = True
                st.rerun()

            if st.session_state.get(f"reply_custom_{eid}"):
                custom = st.text_input("Custom instruction", key=f"custom_instr_{eid}")
                if st.button("Use and draft", key=f"use_custom_{eid}"):
                    if f"reply_generated_{eid}" in st.session_state:
                        del st.session_state[f"reply_generated_{eid}"]
                    with st.spinner("Drafting reply..."):
                        try:
                            current_style = st.session_state.get("learned_style") or load_persisted_style()
                            draft_text = draft_reply(email, decision=custom, style_notes=current_style)
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
                st.subheader("Generated draft")
                draft_val = st.session_state.get(f"reply_generated_{eid}")
                st.text_area(
                    "Draft preview",
                    value=draft_val,
                    height=220,
                    key=f"preview_{eid}",
                    label_visibility="collapsed",
                )
                send_col, _ = st.columns([1, 1])
                with send_col:
                    if st.button("Send now", key=f"send_now_{eid}"):
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
            if st.button("Close email", key=f"close_{eid}", use_container_width=True):
                for k in list(st.session_state.keys()):
                    if eid in str(k):
                        del st.session_state[k]
                st.session_state.pop("selected_email", None)
                st.session_state.pop(f"selected_email_obj", None)
                st.rerun()

with email_col:
    st.markdown("### Emails")

    # Pre-load label colors for badges
    _label_color_map = {}
    try:
        for _lb in get_all_labels():
            _label_color_map[_lb["slug"]] = _lb["color"]
    except Exception:
        pass

    for i, email in enumerate(filtered):
        cat_raw = email.get("category", "normal") or "normal"
        cat_list = [c.strip() for c in cat_raw.split(",") if c.strip()]
        cat = cat_list[0]  # primary category for compat
        needs_action = email.get("needs_action", False)
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

        action_badge = "<span class='badge-action'>Needs action</span>" if needs_action else ""
        cat_badges = ""
        for _cb_cat in cat_list:
            if _cb_cat == "normal":
                continue
            _cb_color = _label_color_map.get(_cb_cat, "#94a3b8")
            cat_badges += f"<span class='badge-category' style='color:{_cb_color};border-color:{_cb_color}40;background:{_cb_color}18;'>{html.escape(str(_cb_cat))}</span> "
        badges = f"{action_badge} {cat_badges}".strip()

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
            const parentDoc = window.parent.document;
            function setupCardClicks() {
                try {
                    const cards = parentDoc.querySelectorAll('.email-card');
                    cards.forEach(card => {
                        if (card.dataset.clickSetup) return;
                        card.dataset.clickSetup = 'true';
                        
                        // Walk up to find the stVerticalBlock container
                        let el = card;
                        while (el && !el.matches('[data-testid="stVerticalBlock"]')) {
                            el = el.parentElement;
                        }
                        if (!el) return;
                        
                        // Find the button within this container
                        const btn = el.querySelector('button');
                        if (!btn) return;
                        
                        card.addEventListener('click', function(e) {
                            e.preventDefault();
                            e.stopPropagation();
                            try { btn.click(); } catch(_) {}
                        });
                    });
                } catch(_) {}
            }
            
            // Run setup after a delay and on mutations
            setTimeout(setupCardClicks, 500);
            const observer = new MutationObserver(() => setTimeout(setupCardClicks, 200));
            observer.observe(parentDoc.body, { childList: true, subtree: true });
            setTimeout(() => observer.disconnect(), 15000);
        } catch(_) {}
    })();
    </script>
    """, height=0)
