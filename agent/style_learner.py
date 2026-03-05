"""Learn and persist the user's writing style from sent emails."""
import re
from collections import Counter
from typing import List, Tuple

from agent.llm import get_llm
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

from tools.gmail_tools import fetch_sent_samples
from agent.profile import save_style_notes, load_profile


def _split_non_empty_lines(text: str) -> List[str]:
    return [line.strip() for line in (text or "").splitlines() if line.strip()]


def _extract_structure(samples: List[str]) -> Tuple[str, str, str]:
    """
    Heuristically extract typical greeting PATTERN, closing phrases, and signature name.
    
    IMPORTANT: The greeting pattern extracts only the greeting word/phrase (e.g. "Dear", "Hi"),
    NOT the specific recipient name. The recipient name varies per email and should be
    determined contextually by the drafter.
    """
    greeting_counter: Counter[str] = Counter()
    closing_counter: Counter[str] = Counter()
    name_counter: Counter[str] = Counter()

    greeting_regex = re.compile(
        r"^(hi|hello|hey|dear|salam|assalamu|as-salamu)\b", re.IGNORECASE
    )

    closing_candidates = [
        "best regards",
        "best",
        "thanks",
        "thank you",
        "many thanks",
        "regards",
        "kind regards",
        "warm regards",
        "cheers",
        "sincerely",
        "all the best",
    ]

    def _normalize_phrase(line: str) -> str:
        # Strip name and commas for more stable aggregation
        base = line.strip().rstrip(",")
        return base

    def _extract_greeting_word(line: str) -> str:
        """Extract just the greeting word/phrase without the recipient name.
        
        'Dear Professor Salam' → 'Dear'
        'Hi Alex' → 'Hi'
        'Hello' → 'Hello'
        """
        stripped = line.strip().rstrip(",")
        # Match common greeting words at the start
        m = re.match(r'^(dear|hi|hello|hey|salam|assalamu\s+alaikum|as-salamu\s+alaikum)\b', stripped, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return stripped

    for raw in samples[:10]:
        if not raw:
            continue
        lines = _split_non_empty_lines(raw)
        if not lines:
            continue

        # GREETING: look at the first few non-empty lines — extract only the greeting word
        for line in lines[:4]:
            lower = line.lower()
            if greeting_regex.match(lower):
                greeting_word = _extract_greeting_word(line)
                greeting_counter[greeting_word] += 1
                break

        # CLOSING: look at the last few non-empty lines
        for line in reversed(lines[-6:]):
            lower = line.lower().rstrip(",")
            for cand in closing_candidates:
                if lower.startswith(cand):
                    closing_counter[_normalize_phrase(line)] += 1
                    # Last line of email is often the signature name (after closing)
                    last_line = lines[-1].strip() if lines else ""
                    if (
                        last_line
                        and last_line != line.strip()
                        and len(last_line) < 50
                        and "@" not in last_line
                        and not any(c in last_line.lower() for c in closing_candidates)
                    ):
                        name_counter[last_line] += 1
                    break
            else:
                continue
            break

    greeting_style = (
        greeting_counter.most_common(1)[0][0] if greeting_counter else "(no consistent greeting detected)"
    )
    closing_style = (
        closing_counter.most_common(1)[0][0] if closing_counter else "(no consistent closing detected)"
    )
    signature_name = (
        name_counter.most_common(1)[0][0] if name_counter else ""
    )
    return greeting_style, closing_style, signature_name


def summarize_style(samples: List[str]) -> str:
    """Use hybrid rule-based + LLM approach to summarize style."""
    if not samples:
        return ""

    # Deterministic extraction of greeting/closing/name from fuller samples (no LLM).
    greeting_style, closing_style, signature_name = _extract_structure(samples)

    # Keep LLM prompt small for local models: use a very small,
    # truncated subset only for softer aspects (tone, length, patterns).
    max_emails_for_llm = 2
    max_chars_per_email = 220
    trimmed_samples = [
        (s or "")[:max_chars_per_email] for s in samples[:max_emails_for_llm]
    ]

    llm = get_llm(task="style_learn")
    parser = StrOutputParser()

    system = (
        "You are a concise writing style analyst.\n\n"
        f"The user's greeting style (how they START emails) has already been detected as:\n"
        f"GREETING_STYLE: {greeting_style}\n\n"
        f"The user's closing style (how they END emails) has already been detected as:\n"
        f"CLOSING_STYLE: {closing_style}\n\n"
        "DO NOT change or re-interpret these two lines. Treat them as ground truth.\n\n"
        "Your task is ONLY to infer:\n"
        "- TONE: overall tone (e.g. professional and friendly, casual, very formal, etc.)\n"
        "- LENGTH: brief / medium / detailed\n"
        "- PATTERNS: notable habits (phrases, punctuation, paragraph style)\n\n"
        "Return EXACTLY these three lines in this order:\n"
        "TONE: ...\n"
        "LENGTH: ...\n"
        "PATTERNS: ...\n"
        "Be specific but keep each line short."
    )

    joined = "\n\n---\n\n".join(trimmed_samples)
    prompt = (
        "Here are a few example emails written by the user. "
        "Use them only to infer tone, length, and other patterns:\n\n"
        f"{joined}\n\n"
        "Now output the three lines as specified."
    )

    chain = llm | parser
    try:
        out = chain.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
        tone_block = (out or "").strip()
        if not tone_block:
            tone_block = (
                "TONE: professional and friendly\n"
                "LENGTH: medium\n"
                "PATTERNS: short paragraphs; clear and direct phrasing"
            )
    except Exception:
        tone_block = (
            "TONE: professional and friendly\n"
            "LENGTH: medium\n"
            "PATTERNS: short paragraphs; clear and direct phrasing"
        )

    # Compose final style description used by the drafter.
    style = (
        f"GREETING_STYLE: {greeting_style}\n"
        f"CLOSING_STYLE: {closing_style}\n"
        f"{f'SIGNATURE_NAME: {signature_name}\n' if signature_name else ''}"
        f"{tone_block}"
    )
    return style


def learn_and_persist_style(max_samples: int = 12) -> str:
    """Fetch sent samples, summarize style, and persist to disk. Returns style string."""
    samples = fetch_sent_samples(max_results=max_samples)
    style = summarize_style(samples)
    try:
        save_style_notes(style)
    except Exception:
        # Do not break the app if profile persistence fails
        pass

    return style


def load_persisted_style() -> str:
    """Load persisted style notes from the user profile."""
    try:
        profile = load_profile()
        return profile.get("style_notes") or ""
    except Exception:
        return ""
