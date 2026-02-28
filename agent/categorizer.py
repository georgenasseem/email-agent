"""Categorize emails by importance."""
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

from agent.llm import get_llm
from agent.email_memory import build_memory_context, match_rules_for_email, get_enabled_labels

CATEGORIES = ["urgent", "important", "normal", "informational", "newsletter"]


def _get_valid_categories() -> list[str]:
    """Return the list of valid category slugs based on enabled labels.

    System categories that have been disabled/deleted by the user are excluded.
    """
    try:
        enabled = get_enabled_labels()
        enabled_slugs = {lb["slug"] for lb in enabled}
        return [c for c in CATEGORIES if c in enabled_slugs] + [
            lb["slug"] for lb in enabled if lb["slug"] not in CATEGORIES
        ]
    except Exception:
        return CATEGORIES


def _build_preview(email: dict) -> str:
    """Return the best preview text for categorization."""
    text = email.get("clean_body") or email.get("body") or email.get("snippet") or ""
    return text[:1000]


def _apply_newsletter_heuristics(email: dict, category: str) -> str:
    """Force obvious newsletters/announcements into 'newsletter' unless clearly security-critical."""
    sender = (email.get("sender") or "").lower()
    subject = (email.get("subject") or "").lower()
    preview = _build_preview(email).lower()

    # Security-style newsletters are allowed to stay non-newsletter
    security_terms = [
        "security alert",
        "verify your account",
        "password reset",
        "suspicious activity",
        "unusual activity",
        "unauthorized access",
        "token expired",
    ]
    if any(t in preview or t in subject for t in security_terms):
        return category  # keep whatever the model chose

    # Strong newsletter / bulk-mail signals
    newsletter_signals = [
        "newsletter",
        "news letter",
        "digest",
        "weekly update",
        "monthly update",
        "view this email in your browser",
        "manage your preferences",
        "unsubscribe",
        "update your preferences",
        "marketing",
    ]
    sender_signals = ["noreply", "no-reply", "newsletter@", "mailchimp", "cmail20.com"]

    if any(s in subject or s in preview for s in newsletter_signals) or any(
        s in sender for s in sender_signals
    ):
        return "newsletter"

    return category


def categorize_email(email: dict) -> dict:
    """Categorize a single email: check user-defined rules first, then heuristics, then LLM."""
    if not email:
        return email

    # ── 1. Check user-defined category rules (sender / domain / keyword) ──
    try:
        rule_match = match_rules_for_email(email)
        if rule_match:
            return {**email, "category": rule_match}
    except Exception:
        pass

    # ── 2. Run newsletter heuristics BEFORE LLM to save a call ──
    heuristic_cat = _apply_newsletter_heuristics(email, "normal")
    if heuristic_cat == "newsletter":
        return {**email, "category": "newsletter"}

    # ── 3. Build the dynamic category list (enabled system + user-defined) ──
    valid_cats = _get_valid_categories()
    all_labels = []
    try:
        all_labels = get_enabled_labels()
    except Exception:
        pass

    # Build the extra labels block for the prompt
    user_labels_block = ""
    user_label_slugs = [s for s in valid_cats if s not in CATEGORIES]
    if user_label_slugs:
        label_lines = []
        for lb in all_labels:
            if lb["slug"] in user_label_slugs:
                desc = lb.get("description") or lb["display_name"]
                label_lines.append(f"- {lb['slug']}: {desc}")
        user_labels_block = "\n\nADDITIONAL USER-DEFINED CATEGORIES:\n" + "\n".join(label_lines) + "\nPrefer these when the email clearly fits."

    # Build dynamic category descriptions for enabled system categories only
    _cat_descriptions = {
        "urgent": "- urgent: MUST act TODAY. Security alerts, account verification, boss/CEO urgent requests, same-day deadlines.",
        "important": "- important: Act within 24-48h. Client requests, approvals, feedback, rescheduling, time-sensitive offers.",
        "normal": "- normal: Routine. Meeting reminders, confirmations, coordination, FYI. Default when unsure.",
        "informational": "- informational: Informational only, no action. Event listings, general updates.",
        "newsletter": "- newsletter: Marketing, newsletters, promotions, unsubscribe links, mass mailings.",
    }
    _enabled_sys_cats = [c for c in CATEGORIES if c in valid_cats]
    _cat_block = "\n".join(_cat_descriptions[c] for c in _enabled_sys_cats if c in _cat_descriptions)

    # ── 4. LLM categorization ──
    llm = get_llm(task="categorize")
    parser = StrOutputParser()

    system = f"""You are an email triage specialist. Assign ONE category for this email.

CATEGORIES (use exactly these words):
{_cat_block}{user_labels_block}

CRITICAL:
- "newsletter" or "informational": Newsletters, event listings, marketing, promotions, "click here", tracking pixels, noreply senders.
- "normal": Default. Reminders, confirmations, routine updates, meeting locations.

Output ONLY one word from the categories above."""

    preview = _build_preview(email)
    thread_ctx = (email.get("thread_context") or "")[:300]
    enriched_ctx = (email.get("enriched_context") or "")[:400]
    related_ctx = (email.get("related_context") or "")[:300]

    # Pull memory context if available
    memory_ctx = ""
    try:
        memory_ctx = build_memory_context(email, max_linked=3)[:300]
    except Exception:
        pass

    extra_sections = ""
    if thread_ctx:
        extra_sections += f"\nThread context (recent messages):\n{thread_ctx}"
    if enriched_ctx:
        extra_sections += f"\n{enriched_ctx}"
    if related_ctx:
        extra_sections += f"\nRelated emails:\n{related_ctx}"
    if memory_ctx:
        extra_sections += f"\n{memory_ctx}"

    prompt = f"""Email:
From: {email.get('sender','')}
Subject: {email.get('subject','')}
Preview: {preview}{extra_sections}

Category (one word only):"""

    chain = llm | parser
    raw = chain.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
    cat = (raw or "").strip().lower()

    # Normalize and validate – accept system + user-defined slugs
    if cat not in valid_cats:
        # Try to recover from phrases like "Category: urgent"
        for c in valid_cats:
            if c in cat:
                cat = c
                break
        else:
            cat = "normal"

    return {**email, "category": cat}


def categorize_emails(emails: list[dict]) -> list[dict]:
    """Add category to each email. Returns emails with 'category' key added."""
    if not emails:
        return []
    return [categorize_email(e) for e in emails]
