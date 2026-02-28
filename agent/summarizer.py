"""Summarize emails using LLM."""
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

from agent.llm import get_llm
from agent.email_memory import build_memory_context


def summarize_email(email: dict) -> dict:
    """Summarize a single email in its own LLM call.

    This avoids batch prompts that can mix up summaries between emails.
    """
    if not email:
        return email

    llm = get_llm(task="summarize")
    parser = StrOutputParser()

    system = """You are an expert email summarizer. Create one concise sentence for this email.

RULES:
1. Summary: 10-80 words. Never copy the first line verbatim.
2. Include WHO (sender/recipient), WHAT (main message or request), WHEN (if deadline or date).
3. For newsletters/marketing: say "Newsletter: [topic]" or "Marketing: [topic]".
4. For calendar/reminders: say "Reminder: [event] on [date]".
5. For confirmations: say "Confirmation of [what]".

Output ONLY the summary sentence (no bullet points, no JSON, no explanations)."""

    body = (email.get("clean_body") or email.get("body") or email.get("snippet") or "")[:600]
    thread_ctx = (email.get("thread_context") or "")[:400]
    enriched_ctx = (email.get("enriched_context") or "")[:400]

    # Pull memory context
    memory_ctx = ""
    try:
        memory_ctx = build_memory_context(email, max_linked=3)[:300]
    except Exception:
        pass

    extra = ""
    if thread_ctx:
        extra += f"\nThread context (recent msgs):\n{thread_ctx}"
    if enriched_ctx:
        extra += f"\n{enriched_ctx}"
    if memory_ctx:
        extra += f"\n{memory_ctx}"

    prompt = f"""Summarize this email in one sentence (min 15 words). Focus on WHO, WHAT, and WHEN.

From: {email.get('sender','')}
Subject: {email.get('subject','')}
Body: {body}{extra}

Write only the summary sentence:"""

    chain = llm | parser
    raw = chain.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])

    text = (raw or "").strip()
    if not text:
        snippet = (email.get("clean_body") or email.get("body") or email.get("snippet") or "")[:160]
        summary = f"(Summary unavailable; snippet: {snippet})"
    else:
        summary = text.replace("\n", " ").strip()

    return {**email, "summary": summary}


def summarize_batch(emails: list[dict]) -> list[dict]:
    """Legacy helper to summarize a list of emails one-by-one.

    Kept for backwards compatibility; new callers should prefer per-email processing.
    """
    if not emails:
        return []
    return [summarize_email(e) for e in emails]
