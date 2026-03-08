"""Configuration and environment variables for Dispatch."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project file paths
PROJECT_ROOT = Path(__file__).parent
CREDENTIALS_FILE = PROJECT_ROOT / "credentials.json"  # Gmail API credentials
TOKEN_FILE = PROJECT_ROOT / "token.json"  # Gmail authentication token

# LLM Configuration
# LLM_PROVIDER: "groq" | "qwen_local_3b" | "qwen_local_7b"
# For Groq: GROQ_API_KEY, GROQ_MODEL, GROQ_FAST_MODEL
# For local Qwen: GGUF model paths loaded via llama-cpp-python
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "qwen_local_3b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
# Fast/cheap model for lightweight tasks (categorise, extract entities, summarise).
# llama-3.1-8b-instant is free on Groq with 20K TPM vs 6K TPM for 70b.
GROQ_FAST_MODEL = os.getenv("GROQ_FAST_MODEL", "llama-3.1-8b-instant")

# Default local model paths (can be overridden via env)
LOCAL_LLM_3B_PATH = os.getenv(
    "LOCAL_LLM_3B_PATH",
    str(PROJECT_ROOT / "models" / "qwen" / "qwen2.5-3b-instruct-q5_k_m.gguf"),
)
LOCAL_LLM_7B_PATH = os.getenv(
    "LOCAL_LLM_7B_PATH",
    str(PROJECT_ROOT / "models" / "qwen" / "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"),
)

# Zoom (Phase 5, optional): Server-to-Server OAuth for creating meetings
# Set ZOOM_ACCOUNT_ID, ZOOM_CLIENT_ID, ZOOM_CLIENT_SECRET in .env to enable "Add Zoom link" when scheduling


# ─── Task profiles ──────────────────────────────────────────────────────────
# Each task has a temperature and a "weight" (light → fast model, heavy → strong model).

TASK_PROFILES = {
    # Lightweight tasks → fast model, low temperature (deterministic)
    "categorize":       {"temperature": 0.1, "max_tokens": 32,   "weight": "light"},
    "entity_extract":   {"temperature": 0.1, "max_tokens": 512,  "weight": "light"},
    "summarize":        {"temperature": 0.15, "max_tokens": 256,  "weight": "light"},
    "flag_urgent":      {"temperature": 0.1, "max_tokens": 64,   "weight": "light"},
    "meeting_extract":  {"temperature": 0.1, "max_tokens": 512,  "weight": "light"},

    # Heavy tasks → strong model, slightly more creative
    "draft":            {"temperature": 0.45, "max_tokens": 1024, "weight": "heavy"},
    "plan_reply":       {"temperature": 0.3, "max_tokens": 512,  "weight": "heavy"},
    "decide":           {"temperature": 0.25, "max_tokens": 512,  "weight": "heavy"},
    "style_learn":      {"temperature": 0.2, "max_tokens": 512,  "weight": "heavy"},

    # Default fallback
    "default":          {"temperature": 0.2, "max_tokens": 512,  "weight": "heavy"},
}


def get_task_profile(task: str) -> dict:
    """Return the task profile (temperature, max_tokens, weight) for a given task name."""
    return TASK_PROFILES.get(task, TASK_PROFILES["default"])


def get_llm_config(task: str = "default"):
    """
    Get the current LLM configuration, optionally routed by task.

    Returns:
        tuple: (provider, key_or_path, model) for the configured LLM
        - For groq: ("groq", api_key, model)
        - For local Qwen: ("local", gguf_path, model_name)
    """
    provider = os.getenv("LLM_PROVIDER", LLM_PROVIDER)

    if provider == "groq":
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY not set in .env. "
                "Get a free key from https://groq.com/"
            )
        profile = get_task_profile(task)
        if profile["weight"] == "light":
            return "groq", GROQ_API_KEY, GROQ_FAST_MODEL
        return "groq", GROQ_API_KEY, GROQ_MODEL

    if provider == "qwen_local_3b":
        return "local", LOCAL_LLM_3B_PATH, "qwen2.5-3b-instruct"

    if provider == "qwen_local_7b":
        return "local", LOCAL_LLM_7B_PATH, "qwen2.5-7b-instruct"

    # Fallback to groq if unknown provider
    if not GROQ_API_KEY:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{provider}'. Use groq, qwen_local_3b, or qwen_local_7b. "
            "For groq, GROQ_API_KEY must be set in .env."
        )
    return "groq", GROQ_API_KEY, GROQ_MODEL
