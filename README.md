# Email Agent

An AI-powered email assistant that reads your Gmail inbox, categorises messages, suggests quick actions, drafts replies in your writing style, and manages meetings — all from a single Streamlit dashboard.

## Features

- **Smart categorisation** — Emails are automatically labelled (important, newsletter, academic, social, etc.). Create your own custom labels and the agent learns from your overrides.
- **Quick actions** — Each email gets context-aware action buttons: reply options (orange), todo tasks (purple), and custom actions (grey).
- **Style-matched drafting** — The drafting pipeline (plan → draft → critique → revise → finalise) writes replies that match your personal tone and greeting style.
- **Meeting detection & scheduling** — Meeting invitations are detected automatically. Accept, decline, or reschedule with one click — the agent checks your Google Calendar for free slots and optionally adds a Zoom link.
- **Todo list** — Actionable tasks extracted from emails are tracked in a built-in todo panel.
- **Memory & context** — Past interactions, sender history, and thread context feed into every decision for smarter suggestions over time.
- **Multi-model support** — Run fully offline with a local Qwen 2.5 3B model (default), upgrade to a local 7B model, or use Groq's hosted API for maximum speed.

## Architecture

The agent is built on **LangGraph** with three core graphs:

1. **Email processing pipeline** (`agent/langgraph_pipeline.py`) — Parallel fan-out: categorise, summarise, extract entities, detect urgency, and detect meetings all run simultaneously, then merge.
2. **Quick actions graph** (`agent/quick_actions_graph.py`) — Parallel fan-out: reply analysis and todo extraction run simultaneously, then merge/deduplicate/validate.
3. **Drafting graph** (`agent/drafting_graph.py`) — Sequential: plan → draft → critique → (revise if needed) → finalise with signature.

```
Gmail API  →  Email Pipeline (parallel)  →  Quick Actions (parallel)  →  UI
                 ├─ categorise                  ├─ reply analysis
                 ├─ summarise                   └─ todo extraction
                 ├─ extract entities                    ↓
                 ├─ detect urgency              merge & validate
                 └─ detect meetings
```

## Prerequisites

- **Python 3.11+**
- **Google Cloud project** with Gmail API and Google Calendar API enabled
- A `credentials.json` OAuth 2.0 client file (see [Setup](#2-google-api-credentials) below)

## Setup

### 1. Clone and install

```bash
git clone https://github.com/gnn9245/email-agent.git
cd email-agent
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Google API credentials

1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new project (or select an existing one).
3. Enable the **Gmail API** and the **Google Calendar API**.
4. Go to **Credentials → Create Credentials → OAuth 2.0 Client ID**.
   - Application type: **Desktop app**.
   - Download the JSON file and save it as `credentials.json` in the project root.
5. Under **OAuth consent screen**, add your Google account as a test user.

### 3. Download the local model (required)

The default model is **Qwen 2.5 3B Instruct** (GGUF, ~2 GB). Download it into the `models/qwen/` directory:

```bash
mkdir -p models/qwen
```

Download from Hugging Face: [Qwen2.5-3B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF)

File: `qwen2.5-3b-instruct-q5_k_m.gguf` → place at `models/qwen/qwen2.5-3b-instruct-q5_k_m.gguf`

Or use the Hugging Face CLI:

```bash
pip install huggingface_hub
huggingface-cli download Qwen/Qwen2.5-3B-Instruct-GGUF qwen2.5-3b-instruct-q5_k_m.gguf --local-dir models/qwen
```

### 4. Environment variables

```bash
cp .env.example .env
```

The defaults work out of the box with the local 3B model — no API keys needed for local inference.

### 5. Run

```bash
streamlit run app.py
```

On first launch the app opens a browser window to sign in with Google (OAuth). After authorising, click **Fetch new emails** to pull your inbox.

---

## Optional upgrades

### Qwen 2.5 7B model

For better quality on complex emails, download the 7B model (~4.6 GB, split into two files):

Download from Hugging Face: [Qwen2.5-7B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF)

Files: `qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf` and `qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf` → place both in `models/qwen/`

Then select **Qwen 2.5 7B (local)** from the model dropdown in the UI.

### Groq (hosted API)

For the fastest inference using hosted models:

1. Get a free API key at [https://console.groq.com/](https://console.groq.com/).
2. In `.env`:
   ```
   LLM_PROVIDER=groq
   GROQ_API_KEY=gsk_your_key_here
   ```
3. Select **Groq (hosted)** from the model dropdown.

Groq uses `llama-3.3-70b-versatile` for heavy tasks and `llama-3.1-8b-instant` for lightweight tasks by default.

### Zoom integration

To add Zoom meeting links when scheduling:

1. Create a **Server-to-Server OAuth** app at [https://marketplace.zoom.us/](https://marketplace.zoom.us/).
2. In `.env`:
   ```
   ZOOM_ACCOUNT_ID=your_account_id
   ZOOM_CLIENT_ID=your_client_id
   ZOOM_CLIENT_SECRET=your_client_secret
   ```

---

## Project structure

```
app.py                      # Streamlit UI (entry point)
config.py                   # LLM configuration and task profiles
agent/
  langgraph_pipeline.py     # Main email processing graph (parallel)
  quick_actions_graph.py    # Quick action suggestion graph (parallel)
  drafting_graph.py         # Reply drafting graph (plan→draft→critique→finalise)
  categorizer.py            # Email categorisation (heuristics + LLM)
  summarizer.py             # Email summarisation
  meeting_extractor.py      # Meeting/event detection
  urgent_detector.py        # Urgency detection (keyword-based)
  decision_suggester.py     # Legacy decision suggester
  drafter.py                # Core reply drafting
  style_learner.py          # Writing style analysis
  profile.py                # User profile loader
  email_memory.py           # SQLite storage, todos, labels, sender history
  memory_store.py           # Email archive/memory helpers
  llm.py                    # LLM provider abstraction (local + Groq)
  text_cleaner.py           # HTML/email body cleaning
  scheduling.py             # Calendar slot finder
  scheduling_graph.py       # Scheduling graph
  delegation.py             # Task delegation helpers
tools/
  gmail_tools.py            # Gmail API wrapper (fetch, send, archive)
  calendar_tools.py         # Google Calendar API wrapper
  google_auth.py            # OAuth 2.0 authentication flow
  zoom_tools.py             # Zoom meeting creation
data/                       # Auto-generated at runtime (gitignored)
models/
  qwen/                     # Local GGUF model files (gitignored)
```

## Tests

```bash
python -m pytest tests/ -v
```

## License

MIT
