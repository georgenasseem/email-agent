## Project Plan – AI Email Agent

This file tracks **all major work**, both upcoming and completed. Update it as we finish each phase or add new features.

---

### Phase 1 – Local LLM options (Groq + Qwen 2.5 3B/7B)

- Add config support for multiple backends (`groq`, `qwen_local_3b`, `qwen_local_7b`)
- Implement `SimpleLLM` branches for:
  - Groq HTTP (existing)
  - Local GGUF via `llama-cpp-python` (`backend="local"`)
- Add Streamlit **LLM backend selector** (Groq / Qwen 2.5 3B local / Qwen 2.5 7B local)
- Download and place Qwen 2.5 3B GGUF in `models/qwen/qwen2.5-3b-instruct-q5_k_m.gguf`
- Download and place Qwen 2.5 7B GGUF shards in:
  - `models/qwen/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf`
  - `models/qwen/qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf`
- Wire 3B/7B paths into `config.py` (`LOCAL_LLM_3B_PATH`, `LOCAL_LLM_7B_PATH`)
- Verify that Groq, Qwen 2.5 3B, and Qwen 2.5 7B can be selected from the UI

#### Phase 1a – Local Qwen robustness + style learner

- Remove silent fallbacks so failures are visible:
  - `agent/summarizer.summarize_batch` → raise on bad JSON instead of single-email fallback
  - `agent/categorizer.categorize_emails` → remove heuristic fallback, raise on parse errors
  - `agent/decision_suggester.suggest_decision` → remove pattern-match fallback, raise on invalid JSON
  - `agent/drafter.draft_reply` → remove synthetic greeting/body/closing fallback; surface errors
- Update local Qwen backend:
  - Switch `backend="local"` to `create_chat_completion` with `chat_format="chatml"`
  - Enable GPU offload with `n_gpu_layers=-1` and a moderate context (`n_ctx=2048` for style + batches)
- Style learner experiments for Qwen 2.5 3B:
  - Tried: pure-LLM style learner over many long sent emails (too slow, misclassified greeting vs closing)
  - Tried: tightened prompt to distinguish START vs END of email (still slow, still brittle)
  - Final: hybrid approach
    - Rule-based extraction of `GREETING_STYLE` and `CLOSING_STYLE` from sent emails
    - Small Qwen call only for `TONE`, `LENGTH`, `PATTERNS` on 2 short email snippets
    - Later extension: style learner also infers `SIGNATURE_NAME` from sent emails so the drafter can close with the user's actual name
    - Verified `CLOSING_STYLE: Best regards` and no longer mislabelled as greeting
- Performance checks (Qwen 2.5 3B local, M1 8 GB):
  - Initial naive style learner runs ≈ 60–90s (too slow)
  - After prompt + context reduction and GPU offload: `learn_and_persist_style(max_samples=4)` ≈ **10–11s**
  - Confirmed updated style written to `data/style_profile.json` and used by `agent.drafter`
  - Fixed `agent.summarizer.summarize_batch` to chunk emails into batches of 6 to avoid context overflows with local Qwen while still keeping batch efficiency
  - Hardened `summarize_batch` JSON handling for local models:
    - Added explicit instruction to output exactly N summaries (no drops/merges)
    - If model still returns too few/many entries, pad/trim with simple fallbacks so the app UI does not crash while still exposing raw output in logs for debugging
    - Later fix: repair truncated JSON arrays from the 3B model (e.g. missing closing `]`) and, if parsing still fails, fall back to extracting quoted strings so the UI never hangs on bad output

#### Phase 1b – Prompt optimization

Ran 15 emails through the pipeline 3 times. Refined prompts for:

- **Summarizer**: Shorter system prompt, explicit rules for newsletters/reminders/confirmations, min 15 words, "never copy first line".
- **Categorizer**: Stronger rules (newsletters/low never urgent; security alerts always urgent), 250-char preview, heuristic fallback for security/newsletter.
- **Decision suggester**: Simpler prompt, better fallback order (newsletter → confirmation → meeting → approve → invite → security → event listing → generic).
- **Drafter**: Mandatory structure (greeting + body + closing), decision-specific fallback bodies (Accept/Decline/Reschedule/Approved/Read later/Verify/Reply), no hardcoded "Hi/Best" when style exists.
- **Urgent detector**: Added "token expired", "token" to security triggers.

**Status:** ✅ Phase 1 is complete.

---

### Phase 2 – LangGraph core pipeline

Goal: Move the current inbox flow into a LangGraph graph (fetch → summarize → categorize → flag → decide → draft).

- Define `EmailAgentState` (core state object)
- Wrap existing functions as LangGraph nodes:
  - `fetch_inbox_node` → `tools.gmail_tools.fetch_emails`
  - `learn_style_node` → `agent.style_learner.learn_and_persist_style`
  - `summarize_node` → `agent.summarizer.summarize_batch`
  - `categorize_node` → `agent.categorizer.categorize_emails`
  - `flag_urgent_node` → `agent.urgent_detector.flag_urgent_emails`
  - `suggest_decision_node` → `agent.decision_suggester.suggest_decision`
  - `draft_reply_node` → `agent.drafter.draft_reply` (on-demand from UI)
- Build an initial straight-line LangGraph for “Process Inbox”
- Update Streamlit to call the LangGraph pipeline instead of direct function chaining
- Improved UI error handling: pipeline exceptions are stored in Streamlit session state so the loading spinner clears and a visible error message is shown instead of the app appearing to hang

**Status:** ✅ Phase 2 is complete.

---

### Phase 3 – Email + thread search and richer context

Goal: Let the agent search emails and full threads and use that context when summarizing/deciding/drafting.

- Extend Gmail tools:
  - `fetch_thread(thread_id)`
  - `search_emails(query, max_results)`
- Build a compact thread context representation for LLM calls
- Add LangGraph nodes:
  - `thread_context_node`
  - `email_search_node` (served by existing `fetch_inbox_node` with query)
- Add UI controls for Gmail-style search and viewing thread summaries

**Status:** ✅ Phase 3 is complete.

### Phase 4 – Memory graph, profile.json, auto categories, delegation

Goal: Structured long-term memory and intelligent behavior based on history.

- Design SQLite schema:
  - `user_profile`
  - `memory` (long-term notes, preferences, style, etc.)
  - `task_state`
  - `delegation_rules` (who gets what topics)
- Introduce `profile.json` per user (style, preferences, roles, working hours)
- Auto categories:
  - Log historical actions and categories in SQLite
  - Train / infer **dynamic categories** from history
  - UI to view, rename, merge, delete categories
- Delegation patterns:
  - Learn rules like “Research → Person X” from forwarding history
  - Store/edit rules in `delegation_rules`
  - Add LangGraph `delegation_decider` node to propose/auto-forward based on rules

**Status:** ✅ Phase 4 (storage + wiring) is implemented; learning and UI for categories/delegation rules are deferred to a later iteration.

---

### Phase 5 – Google Calendar / Zoom integration

Goal: Understand and act on scheduling emails.

- `tools/google_auth.py`: Shared OAuth with Gmail + Calendar scopes
- `tools/calendar_tools.py`: Auth with Google Calendar, `list_events`, `find_free_slots`, `create_event`
- Zoom integration:
  - `tools/zoom_tools.py`: Server-to-Server OAuth, `create_zoom_meeting()`, `zoom_available()`
  - When creating a calendar event from "Schedule meeting", optional "Add Zoom meeting link" creates a Zoom meeting and adds join URL to the event description
  - .env: `ZOOM_ACCOUNT_ID`, `ZOOM_CLIENT_ID`, `ZOOM_CLIENT_SECRET` (create Server-to-Server OAuth app at Zoom marketplace)
- `agent/meeting_extractor.py`: LLM extracts meeting intent from emails
- `agent/scheduling.py`: `propose_meeting_times`, `create_event_from_slot(add_zoom=...)`
- LangGraph `agent/scheduling_graph.py`: `extract_meeting_intent_node`, `run_scheduling_proposal`
- UI: "Schedule meeting" expander with propose times → "Add Zoom meeting link" checkbox → create

**Status:** ✅ Phase 5 is complete (Calendar + Zoom).

---

### Big Fixes – Inbox pipeline, cleaning, scheduling, drafting, UI

Goal: Fix architectural flaws discovered after Phases 1–5 (mixed summaries, weak categories/quick replies, overused cleaning, manual scheduling, and drafting POV issues) by moving to per-email stages, deterministic rules, and richer state.

- Per-email summarization and categorization:
  - Replaced batch JSON-array summarization with `summarize_email` (one LLM call per email) while keeping `summarize_batch` as a thin loop for backward compatibility.
  - Updated `agent.categorizer` to categorize each email independently via `categorize_email`, using `clean_body`/`body` previews instead of large concatenated prompts.
  - Added `postprocess_categories_node` in the LangGraph pipeline to reconcile categories with `needs_action`, so an email that truly needs action can never end up as `low` or `newsletter`.
  - New rule layer: when `needs_action=True` and the model chose `low`/`newsletter`, deterministically upgrade to `important` or `urgent` based on security/deadline keywords (e.g. "verify your account", "token expired", "by end of day").

- Deterministic cleaning at fetch time (no LLM in cleaning path):
  - Moved all email-body cleaning into `agent.text_cleaner.clean_email_text`, a pure-Python, regex-based cleaner (Proofpoint unwrap, HTML/markup stripping, quoted-text and header removal, whitespace normalization).
  - Updated `tools.gmail_tools.fetch_emails` and `fetch_thread` to compute and store `clean_body` once per message:
    - `body`: raw decoded plain text/HTML-stripped content (truncated for context limits).
    - `clean_body`: aggressively cleaned version used for display and downstream reasoning.
  - Updated the Streamlit UI to display `clean_body` directly when viewing an email, removing repeated calls to text cleaning when opening the same email or drafting a reply.
  - Ensured all downstream components (summarizer, categorizer, decision suggester, drafter) prefer `clean_body` as their primary text source, falling back to `body`/`snippet` only when needed.

- Automatic scheduling tied to intent instead of manual buttons:
  - Kept the LangGraph scheduling subgraph (`agent/scheduling_graph.py`) and `agent/scheduling.propose_meeting_times` as the core scheduling interface.
  - Modified the UI quick-reply flow so that meeting-style options (e.g. "Accept", "Decline", "Reschedule", "Propose time") automatically trigger `propose_meeting_times` for that email.
  - Reused the existing slot-selection and event-creation UI so that:
    - User clicks a meeting-style quick reply → meeting intent is extracted and calendar is queried → available time slots are shown automatically under the email.
    - The "Schedule meeting" expander remains available as a debug/advanced control, but is no longer the primary path for common meeting replies.

- Quick reply suggestions (LLM-only, no heuristics):
  - Removed all heuristic/rule-based fallbacks; `agent.decision_suggester.suggest_decision` now uses the LLM exclusively.
  - Refined LLM prompt includes `category`, `needs_action`, `thread_context`, and `related_context`.
  - Post-processing: strip empties, deduplicate case-insensitively, keep 2–5 high-signal options.
  - If LLM fails or returns nothing, `decision_options` is left unset; UI shows "No quick reply suggestions available for this email."

- Drafting with explicit roles and a separate planning step:
  - Introduced `analyze_roles(email)` in `agent.drafter`:
    - Uses the current `profile` (via `agent.profile.load_profile`) to identify `user_email`.
    - Treats the incoming email `sender` as the other party.
    - Produces a small role state: `user_email`, `other_party_email`, `other_party_name`, and `thread_role` flags (user as recipient, other party as sender).
  - Added `plan_reply(email, decision)`:
    - First LLM call that returns a structured plan:
      - `goal`, `key_points`, `tone`, `risks`.
    - Uses `clean_body` + thread context + chosen quick action (`decision`) so the drafter can "think" before writing.
    - Falls back to a safe, deterministic plan if JSON parsing fails.
  - Updated `draft_reply` to:
    - Call `analyze_roles` and `plan_reply` before drafting prose.
    - Feed the reply plan (goal, key points, tone) and explicit POV instructions into the drafting system prompt:
      - "You are writing FROM the user ({user_email}) TO the other party ({other_party_email}). Never flip this perspective."
    - Prefer `clean_body` over raw `body` for context, reducing noise from signatures and quoted text.
    - Preserve the existing style learner integration and signature handling so closings stay aligned with the user's learned style.

- Memory and delegation visibility (for manual inspection):
  - Confirmed all long-term memory and delegation information is persisted in `data/memory.db`:
    - `user_profile`: core identity and style fields (mirrored into `data/profile.json` for quick reads).
    - `memory`: general long-term notes and category logs.
    - `task_state`: reserved for future long-running/autonomous workflows.
    - `delegation_rules`: learned/edited delegation patterns.
  - Verified that `agent.delegation_decider_node` and `agent.langgraph_pipeline.log_memory_node` consistently write into this SQLite DB, so behavior around categories and delegation can be inspected directly with any SQLite viewer.

- Cross-email related context across threads:
  - Added `related_context_node` in the LangGraph pipeline to build lightweight per-email context from other emails in the same batch (by shared subject tokens and sender domains).
  - Each email can now carry a `related_context` field summarizing up to a few other potentially relevant emails (different dates and senders).
  - Updated `agent.decision_suggester.suggest_decision` and `agent.drafter.plan_reply` prompts to consume `related_context` in addition to thread context, so decisions and drafts can reason across separate but related emails (e.g. multiple finance or immigration messages over time).

**Status:** ✅ Big Fixes batch implemented (summaries/categories, cleaning, scheduling, quick replies, drafting, memory visibility, and first-pass dashboard UI).

---

### Bug Fixes – Dashboard layout, quick replies, and UI polish

- Streamlit dashboard header and controls:
  - Reduced top whitespace and tightened the header/toolbar to better match the compact dashboard reference layout.
  - Added a circular profile avatar area in the header; clicking it opens a small profile menu with sign-in/sign-out controls instead of a large standalone button.
  - Replaced the textual “Dark mode” button with a small icon-style toggle using simple glyphs (no emojis) and grouped it with the profile icon.
  - Moved the “Emails to load” slider into the top-right control area to reclaim vertical space.
  - Removed the large `st.info("Click Fetch & Process Emails...")` box and replaced it with lighter inline copy so the empty state does not dominate the canvas.

- Inbox filtering and layout:
  - Removed the right-hand “Filters & settings” column and moved the `All / Needs action / Important` filter into a compact, top-level control strip.
  - Expanded the main email list to occupy the freed horizontal space.
  - Introduced a two-panel layout where the left “Drafting” column remains fixed while the middle “Emails” column scrolls independently (using a scrollable container) so drafting context stays visible while browsing many emails.

- Error handling and minor UI bugs:
  - Removed the non-functional “Copy to clipboard” buttons that were calling `st.experimental_set_query_params()` and occasionally surfaced an `InvalidCharacterError` from the frontend bundle.
  - Ensured all emoji-based labels were removed from the UI and replaced with simple text or glyph-style icons, in line with the updated visual language.

- Quick reply suggestions:
  - Updated the Streamlit UI to always call the LLM-backed `agent.decision_suggester.suggest_decision` per email instead of relying on any pre-computed or heuristic `decision_options` attached to the email objects.
  - Verified that quick replies now stay contextual and non-generic, driven solely by the LLM prompt in `agent/decision_suggester.py`.

- Profile avatar and responsive email pane:
  - Hooked the header avatar into `agent.profile.load_profile` / `get_current_user_email` so its initials are derived from the user’s stored display name (or email local-part) instead of a hardcoded placeholder, and added optional support for a `avatar_url` / `photo_url` in the profile that shows the actual user photo instead of initials when available.
  - Switched the scrollable email list container from a fixed pixel height to a `calc(100vh - offset)` max-height so the inbox area better fits different screen sizes while keeping the drafting column pinned.

- Settings menu and top-bar compaction:
  - Moved model selection, email-count slider, and account actions into a compact settings menu opened from the profile avatar so the top bar stays single-line and focused on the title plus primary action.
  - Removed the explicit “Model” label, aligned the `Analyze inbox` button horizontally with the header text, and added a thin divider line under the top bar to visually separate controls from content.
  - Restyled the slider track and thumb to use the green accent color and tied card backgrounds to a `surface_bg` that follows the active theme; later simplified the UI by dropping the dark/light toggle from the menu so the app defaults to a stable dark look.

---

### Additional fixes and polish (post–Big Fixes)

- Newsletter categorization:
  - Added `_apply_newsletter_heuristics` in `agent.categorizer` to force obvious newsletters (subject/body containing "newsletter", "digest", "unsubscribe", etc., or senders like `noreply`, Mailchimp) into `newsletter` category unless they contain strong security phrases (password reset, verify account, token expired, etc.).
  - Prevents newsletters (e.g. "Student Finance Newsletter February 2026") from being incorrectly flagged as `urgent`.

- Groq 429 rate limit handling:
  - Updated `agent.llm` Groq branch to retry on HTTP 429 (tokens-per-minute limit) instead of returning empty and causing JSON parse errors.
  - Parses Groq error message for "try again in XXXms" / "X.Xs", sleeps for that duration (with minimum backoff), and retries up to 3 times with exponential backoff.
  - Keeps existing global 3-second spacing between requests.

- Streamlit compatibility:
  - Replaced all `st.experimental_rerun()` with `st.rerun()` to fix `AttributeError: module 'streamlit' has no attribute 'experimental_rerun'` when toggling theme or logging out.

- Default LLM:
  - Set Qwen 2.5 3B as the default LLM choice in the UI selector (index=1).

- Email fetch and sorting:
  - Added `internal_date` (Gmail numeric timestamp) to each email in `tools.gmail_tools.fetch_emails` for reliable date ordering.
  - UI sorting ("Date newest/oldest") now uses `internal_date` instead of raw `Date` string so today's emails appear at the top.

- Simplified fetch behavior:
  - Removed Gmail query input and "Unread only" toggle; app always uses `query="in:inbox"` and `unread_only=False`.

- Sidebar removed:
  - All controls (theme, LLM, email count, Login/Logout, Deploy) moved into a single horizontal top toolbar; no left sidebar.

- Three-column layout (reference dashboard):
  - Left: compact settings panel (LLM choice, emails loaded count, view filter).
  - Center: high-density email list (slim cards with subject bold, sender muted, summary small).
  - Right: always-visible drafting panel (selected email message, thread context, scheduling, quick replies, draft editor, send/copy) so drafting does not require scrolling.

- Top bar structure:
  - Project name on left; sort/view and settings in middle; profile avatar and Login/Logout on right.
  - Profile area supports "Change user" to immediately clear token and trigger Gmail OAuth for another account.

---

### Phase 6 – Custom categorizations and user labels

Goal: Let the agent learn and apply custom categories that reflect how **you** think about your inbox, and give you full control to view and edit them.

- Category model on top of existing storage:
  - Reuse the `memory` and `delegation_rules` tables and add a focused `category_labels` concept (e.g. slug, display name, color, description, enabled).
  - Link emails, senders, and subjects to these labels via lightweight history records.
- Category learning:
  - Use past actions (archive, reply, delegate) plus sender and subject patterns to propose categories such as "Advising", "Immigration", "Payments", "Newsletters".
  - Allow manual labelling from the UI and feed those labels back into SQLite so the model learns per-sender preferences.
- UI:
  - Category management view: list all labels, rename, merge, hide/show, and delete.
  - Per-email category selector in the inbox view so you can override the model’s choice with one click.
- Integration:
  - Update `agent.categorizer` to take user-defined labels and per-sender rules into account before assigning `email["category"]`.
  - Keep the LangGraph pipeline as the single source of truth so any changes are reflected consistently in the UI and memory.

**Status:** ✅ Phase 6 is complete.

---

### Bug Fixes – UI polish, sign-in flow, and layout (recent)

- **Sign-in vs Analyze separation:**
  - Sign-in button now triggers Google OAuth when clicked (calls `get_google_credentials()` directly).
  - Analyze inbox only fetches and analyzes emails; it no longer performs sign-in.
  - Both actions are separate in code and UI.

- **Top bar layout and controls:**
  - Header uses `st.columns([6, 4], vertical_alignment="center")` for vertical alignment.
  - Model selector and Sign in/Sign out are in a 3-column layout: inline "Model" label, selectbox (label collapsed), and auth button.
  - Added CSS to push header controls to the far right (`justify-content: flex-end`).
  - Model selectbox constrained to `max-width: 200px`; inner columns use `vertical_alignment="center"` so the button aligns with the selector.
  - Fixed indentation so the title markdown and selectbox render inside their respective columns.

- **Model label and empty pill:**
  - Removed "Model" text initially to avoid an empty pill; later re-added with `label_visibility="visible"` when a 3-column layout (label | select | button) was used.
  - When hiding the label, used `label_visibility="hidden"` with a space label to avoid reserved space.

- **Email count and slider:**
  - Fixed email count at 25 (`MAX_EMAILS = 25`); removed the "Emails to load" slider.
  - Removed slider-related CSS.

- **Streamlit API updates:**
  - Replaced all `st.experimental_rerun()` with `st.rerun()` to fix `AttributeError: module 'streamlit' has no attribute 'experimental_rerun'`.

- **InvalidCharacterError fix:**
  - Added `import html` and `html.escape()` for all dynamic email content (subject, sender, date, summary, category) when rendering email cards in HTML.
  - Prevents `InvalidCharacterError` when email content contains `<`, `>`, or other special characters.

- **Redundant UI removal:**
  - Removed the redundant "Showing 25 emails" / "Showing 6 of 25 emails" section above the two-column layout.
  - Removed the legacy single-column detail view (the long `if False and sel_id:` block).
  - Removed the "Click Fetch & Process Emails to load your inbox" box.

- **Color scheme:**
  - Primary accent: `#38bdf8` (blue), hover: `#0ea5e9`.
  - Accent orange (Needs action badge): `#fb923c`.
  - Button hover: subtle green glow; selectbox dark styling with accent border on hover.

- **Email list and cards:**
  - Email cards with `.email-card` class, badges for "Needs action" and category.
  - Compact "Open" button per email; refined "Showing X of Y emails" styling.
  - Section headers (`h3`) styled as uppercase, muted, compact.

- **LLM backend session state warning:**
  - Resolved "widget with key 'llm_backend' was created with a default value but also had its value set via the Session State API" by not pre-setting `st.session_state["llm_backend"]`; use a computed default only for the selectbox `index`, then read from session state after the widget is created.

---

### Phase 6 Refinements – Content-based categorization and UI cleanup

Goal: Make category learning content-driven (not sender-driven) and streamline the inbox UI.

- **Content-based rule learning (replaces sender-based):**
  - `record_category_override` now extracts meaningful keywords from the email **subject** (filtering stop words, minimum 3 chars) and creates `subject_keyword` rules — no more sender-based rules on override.
  - `match_rules_for_email` priority reordered: **keyword → sender → domain** (content-first), because the same sender can send different types of emails.
  - `propose_categories_from_history` analyzes recurring **subject keywords** across all stored emails instead of grouping by sender domain — suggests categories like "Meeting", "Invoice", "Report" based on content patterns.

- **Category override moved from email cards to Categories tab:**
  - Removed the per-email category selectbox dropdown from email cards to keep cards clean.
  - Added a "Re-categorize email" section in the Categories tab that shows only the currently selected email with a category dropdown. If no email is selected, shows "Select an email to change its category."

- **UI cleanup – removed inbox filters and stats:**
  - Removed the `All / Needs action / Important` radio filter from the inbox view.
  - Removed the "Showing X of Y emails" counter.
  - Removed the "Inbox view · N stored · N processed" stats bar.
  - Emails now show as a flat sorted list (newest first) without filtering controls.

- **Quick action button styling:**
  - Reply actions styled with blue accent (#0ea5e9), Todo actions with amber (#f59e0b), Schedule actions with purple (#8b5cf6).
  - Uses `st.container(key=...)` with CSS attribute selectors for reliable Streamlit targeting.

- **Schedule button force fix:**
  - Added `force=True` parameter to `propose_meeting_times` so UI-triggered scheduling bypasses the LLM intent gate (builds fallback intent from email subject/sender).

- **Quick actions error suppression:**
  - Changed quick actions `except` block to silently fall back to empty options instead of showing `st.error`.

- **Tests:**
  - Updated `TestRuleMatching`, `TestCategoryOverride`, `TestProposals`, and `TestEdgeCases` in `test_categories_integration.py` to validate content-based rules and new priority order.
  - 131 total tests passing (52 category + 38 memory + 41 scheduling).

**Status:** ✅ Phase 6 refinements complete.
