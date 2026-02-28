# AI Email Agent

An intelligent email assistant built with **Streamlit**, **LangGraph**, and **LLM-powered pipelines** that fetches your Gmail inbox, summarizes emails, categorizes them, suggests quick actions, drafts context-aware replies, and schedules meetings via Google Calendar and Zoom.

---

## Features

### Core Pipeline
- **Fetch & Process** — Pulls emails from Gmail via the API, summarizes each one, categorizes it, detects urgency, and suggests quick actions — all in a single LangGraph pipeline.
- **Persistent Memory** — Raw and processed emails are stored in SQLite so subsequent loads are instant. Only new emails are fetched and processed.
- **Style Learning** — Analyzes your sent emails to learn your greeting style, closing style, tone, and patterns. Drafts match how you actually write.

### Intelligent Categorization
- **LLM + Rules** — Categories are assigned by the LLM, with deterministic overrides for newsletters, security alerts, and user-defined rules.
- **Custom Labels** — Create, rename, hide, and delete your own category labels (e.g., "Advising", "Finance", "Meetings").
- **Content-Based Learning** — When you override a category, the system learns keyword rules from the email subject so future emails with similar content are categorized automatically. Rules are content-based, not sender-based, because the same sender can send different types of emails.
- **Category Proposals** — The system analyzes recurring keywords across your inbox and proposes new categories to create.

### Smart Drafting
- **Role-Aware** — Knows who you are and who you're replying to; never flips the perspective.
- **Plan-Then-Draft** — A two-step LLM process: first creates a structured reply plan (goal, key points, tone), then writes the actual reply informed by that plan.
- **Context-Rich** — Uses thread context, cross-email related context, sender history, and memory to produce informed replies.
- **Quick Actions** — LLM-suggested actions like "Accept", "Decline", "Acknowledge", "Forward to team" appear as one-click buttons that auto-draft the appropriate reply.

### Scheduling
- **Meeting Detection** — Extracts meeting intent (topic, duration, attendees) from email content using an LLM.
- **Calendar Integration** — Checks Google Calendar for free slots and proposes available meeting times.
- **Zoom Integration** — Optionally creates a Zoom meeting and adds the join link to the calendar event.
- **Force Scheduling** — The UI "Schedule" button bypasses the LLM intent gate so you can schedule from any email.

### UI
- **Dark Theme Dashboard** — Compact Streamlit layout with email list, drafting panel, and action tabs.
- **Email Rendering** — Full HTML email rendering in a sandboxed iframe with sanitized styles.
- **Tabs** — Drafting, Todo, and Categories tabs for managing replies, tasks, and labels.
- **Retrain Shortcut** — `Ctrl+Shift+K` wipes processed data and reprocesses all stored emails.

---

## Architecture

```
app.py                          Streamlit UI (entry point)
config.py                       Environment vars, LLM routing, task profiles

agent/
├── langgraph_pipeline.py       LangGraph graph: fetch → summarize → categorize → flag → decide → enrich
├── llm.py                      LLM adapter (Groq API or local Qwen via llama-cpp-python)
├── summarizer.py               Per-email summarization
├── categorizer.py              Category assignment (rules → LLM → newsletter heuristics)
├── urgent_detector.py          Urgency flagging with keyword triggers
├── decision_suggester.py       Quick action suggestions via LLM
├── drafter.py                  Role-aware reply drafting with plan-then-write
├── style_learner.py            Learns writing style from sent emails
├── meeting_extractor.py        Extracts meeting intent from email text
├── scheduling.py               Proposes meeting times, creates calendar events
├── email_memory.py             Email persistence, category CRUD, rule learning
├── memory_store.py             SQLite schema and connection management
├── text_cleaner.py             Deterministic email text cleaning (no LLM)
├── context_enrichment.py       Entity extraction and cross-email context
├── delegation.py               Delegation rule matching
└── profile.py                  User profile management

tools/
├── google_auth.py              Shared OAuth for Gmail + Calendar
├── gmail_tools.py              Gmail API: fetch, send, search, thread retrieval
├── calendar_tools.py           Google Calendar: list events, find free slots, create events
└── zoom_tools.py               Zoom Server-to-Server OAuth: create meetings

tests/
├── test_memory_integration.py         38 tests — email storage, links, memory context
├── test_scheduling_integration.py     41 tests — calendar, slots, meeting extraction
└── test_categories_integration.py     52 tests — labels, rules, overrides, proposals
```

---

## Setup

### Prerequisites

- **Python 3.11+**
- A **Google Cloud project** with Gmail and Calendar APIs enabled
- A **Groq API key** (free at [groq.com](https://groq.com)) — or local Qwen GGUF models
- *(Optional)* Zoom Server-to-Server OAuth app for meeting link creation

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/email-agent.git
cd email-agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Google OAuth credentials

1. Go to [Google Cloud Console](https://console.cloud.google.com/) → APIs & Services → Credentials.
2. Create an **OAuth 2.0 Client ID** (Desktop application).
3. Download the JSON and save it as `credentials.json` in the project root.
4. Enable the **Gmail API** and **Google Calendar API** for your project.

On first sign-in, the app will open a browser window for Google OAuth and save the token as `token.json`.

### 3. Environment variables

Create a `.env` file in the project root:

```env
# Required for Groq (default provider)
GROQ_API_KEY=gsk_your_key_here

# LLM provider: "groq" | "qwen_local_3b" | "qwen_local_7b"
LLM_PROVIDER=groq

# Optional: override default models
# GROQ_MODEL=llama-3.3-70b-versatile
# GROQ_FAST_MODEL=llama-3.1-8b-instant

# Optional: Zoom integration
# ZOOM_ACCOUNT_ID=your_account_id
# ZOOM_CLIENT_ID=your_client_id
# ZOOM_CLIENT_SECRET=your_client_secret
```

### 4. Local Qwen models (optional)

To use local inference instead of Groq:

1. Download GGUF models into `models/qwen/`:
   - [Qwen 2.5 3B Instruct (Q5_K_M)](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF)
   - [Qwen 2.5 7B Instruct (Q4_K_M)](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF)
2. Set `LLM_PROVIDER=qwen_local_3b` or `LLM_PROVIDER=qwen_local_7b` in `.env`.

### 5. Run

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. Click **Sign in** to authenticate with Google, then **Analyze inbox** to fetch and process your emails.

---

## Usage

### Basic workflow

1. **Sign in** — Authenticate with your Google account.
2. **Analyze inbox** — Fetches up to 25 emails, runs the full LangGraph pipeline (summarize, categorize, flag, suggest actions).
3. **Browse emails** — Click any email card to open it in the detail panel. View the full HTML body, thread context, and suggested quick actions.
4. **Quick actions** — Click a suggested action (e.g., "Accept", "Decline") to auto-draft a reply. Edit the draft, then send.
5. **Schedule** — Click a "Schedule:" action to propose meeting times from your calendar. Select a slot and optionally add a Zoom link.
6. **Categories tab** — Manage labels, view learned rules, override the selected email's category, or get category suggestions.
7. **Todo tab** — Quick-add tasks from email context.

### Task-aware LLM routing

The system uses two models to optimize speed and cost:

| Task | Model | Why |
|------|-------|-----|
| Categorize, Summarize, Extract entities, Flag urgent, Meeting extract | `llama-3.1-8b-instant` (fast) | Short, deterministic outputs |
| Draft replies, Plan reply, Decide actions, Learn style | `llama-3.3-70b-versatile` (strong) | Requires nuanced reasoning |

### Keyboard shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+Shift+K` | Retrain — wipes processed data and reprocesses all stored emails |

---

## Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run a specific test suite
python -m pytest tests/test_categories_integration.py -v
python -m pytest tests/test_memory_integration.py -v
python -m pytest tests/test_scheduling_integration.py -v
```

131 tests cover email storage, cross-email linking, memory context, calendar operations, meeting extraction, scheduling proposals, category labels, rules, overrides, and LLM integration.

---

## Project Structure Details

### LangGraph Pipeline

The core pipeline is a linear LangGraph graph defined in `agent/langgraph_pipeline.py`:

```
fetch_inbox → learn_style → store_raw → summarize → categorize
→ postprocess_categories → flag_urgent → enrich_context
→ related_context → suggest_decisions → delegation → log_memory
→ store_processed → build_links → END
```

Each node is a pure function that reads from and writes to a shared `EmailAgentState` dict. The pipeline supports three modes:
- **Fresh fetch** — Fetches new emails from Gmail, processes only those not already in the DB.
- **Load from memory** — Instantly loads all processed emails from SQLite (no API calls).
- **Retrain** — Wipes processed data, reprocesses all stored raw emails through the full pipeline.

### SQLite Schema

All persistent data lives in `data/memory.db`:

| Table | Purpose |
|-------|---------|
| `emails` | Raw email storage (gmail_id, subject, sender, body, etc.) |
| `email_processed` | LLM results (summary, category, needs_action, decision_options) |
| `email_links` | Cross-email relationships (same sender, shared keywords) |
| `category_labels` | User-defined + system category labels (slug, color, enabled) |
| `category_rules` | Learned categorization rules (match_type, match_value, label_slug) |
| `todo_items` | Task list items |
| `knowledge_base` | Extracted entities and facts |
| `user_profile` | Identity and preference storage |
| `memory` | Long-term notes and logs |
| `delegation_rules` | Forwarding/delegation patterns |
| `task_state` | Reserved for future autonomous workflows |

---

## License

This project is for educational and personal use. See individual dependency licenses for third-party terms.
