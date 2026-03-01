"""Categorize emails by importance."""
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

from agent.llm import get_llm
from agent.email_memory import build_memory_context, match_rules_for_email, match_all_rules_for_email, get_enabled_labels, SYSTEM_CATEGORIES

CATEGORIES = ["important", "informational", "newsletter"]


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
    """Categorize a single email: check user-defined rules first, then heuristics, then LLM.

    Returns email with 'category' set. Format: 'system_cat' or 'system_cat,tag1,tag2'
    where system_cat is one of important/informational/newsletter and tags are
    user-defined labels matched by rules.
    """
    if not email:
        return email

    # ── 1. Collect ALL matching user-defined tags from rules ──
    user_tags: list[str] = []
    system_rule_match: str | None = None
    try:
        all_matches = match_all_rules_for_email(email)
        for slug in all_matches:
            if slug in SYSTEM_CATEGORIES:
                # First system category rule wins
                if not system_rule_match:
                    system_rule_match = slug
            else:
                user_tags.append(slug)
    except Exception:
        pass

    # If a system category rule matched, use it directly (skip LLM)
    if system_rule_match:
        cat = ",".join([system_rule_match] + user_tags) if user_tags else system_rule_match
        return {**email, "category": cat}

    # ── 2. Run newsletter heuristics BEFORE LLM to save a call ──
    heuristic_cat = _apply_newsletter_heuristics(email, "informational")
    if heuristic_cat == "newsletter":
        cat = ",".join(["newsletter"] + user_tags) if user_tags else "newsletter"
        return {**email, "category": cat}

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
        "important": "- important: The sender PERSONALLY expects YOU to take action — reply to their question, approve something, complete a task with a deadline. NOT for: general announcements, event invitations sent to many people, informational updates, or reminders.",
        "informational": "- informational: Everything that is not a newsletter and does not require YOUR personal action. Announcements, event invitations, updates, FYI, reminders, meeting updates, group emails, notifications. DEFAULT when unsure.",
        "newsletter": "- newsletter: Marketing, newsletters, promotions, unsubscribe links, mass mailings.",
    }
    _enabled_sys_cats = [c for c in CATEGORIES if c in valid_cats]
    _cat_block = "\n".join(_cat_descriptions[c] for c in _enabled_sys_cats if c in _cat_descriptions)

    # ── 4. LLM categorization ──
    llm = get_llm(task="categorize")
    parser = StrOutputParser()

    # If user-defined labels exist, the LLM can also assign ONE extra tag
    _extra_instruction = ""
    if user_labels_block:
        _extra_instruction = "\n\nYou may OPTIONALLY add a second tag from the ADDITIONAL categories if it clearly fits. Format: main_category,extra_tag (e.g. important,hackathon). Only add the extra tag if it is a strong match."

    system = f"""You are an email triage specialist. Assign ONE main category for this email.

MAIN CATEGORIES (use exactly these words):
{_cat_block}{user_labels_block}{_extra_instruction}

CRITICAL:
- "newsletter": Newsletters, marketing, promotions, "click here", tracking pixels, noreply senders.
- "informational": Announcements, event invitations sent to many people, routine confirmations, reminders, meeting updates, group notifications, no reply needed. DEFAULT when unsure between important and informational.
- "important": ONLY when the sender personally and directly asks YOU to do something specific — reply to a direct question, approve a request, complete a task with a deadline. Mass announcements, event invitations, and general updates are NOT important even if they contain deadlines.

Output ONLY the category slug (or main,extra if an additional tag applies)."""

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
    # LLM may return "main,extra" format
    parts = [p.strip() for p in cat.split(",") if p.strip()]
    main_cat = parts[0] if parts else ""
    llm_extra = parts[1] if len(parts) > 1 else None

    # Validate main category
    if main_cat not in valid_cats:
        # Try to recover from phrases like "Category: important"
        for c in CATEGORIES:
            if c in main_cat:
                main_cat = c
                break
        else:
            main_cat = "informational"

    # Ensure main_cat is a system category
    if main_cat not in CATEGORIES:
        # main_cat is a user label — swap it to extra and default main to informational
        llm_extra = main_cat
        main_cat = "informational"

    # Validate LLM extra category (must be a user-defined label, not a system category)
    if llm_extra:
        if llm_extra in CATEGORIES or llm_extra not in valid_cats:
            llm_extra = None

    # Combine: system category + LLM extra tag + rule-matched user tags
    all_tags = []
    if llm_extra and llm_extra not in user_tags:
        all_tags.append(llm_extra)
    all_tags.extend(t for t in user_tags if t != llm_extra)

    final_cat = ",".join([main_cat] + all_tags) if all_tags else main_cat
    return {**email, "category": final_cat}


def categorize_emails(emails: list[dict]) -> list[dict]:
    """Add category to each email. Returns emails with 'category' key added."""
    if not emails:
        return []
    return [categorize_email(e) for e in emails]
