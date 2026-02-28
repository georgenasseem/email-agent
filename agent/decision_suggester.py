"""Suggest contextual quick actions (Reply / Todo) based on email content."""
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
import json
import logging
import re

from agent.llm import get_llm
from agent.email_memory import build_memory_context, get_sender_history

logger = logging.getLogger(__name__)


def suggest_decision(email: dict) -> dict:
    """Suggest contextual quick actions for an email using LLM only.

    Returns {"decision_options": ["Reply: ...", "Todo: ...", ...]}.
    Each option is prefixed with either "Reply:" or "Todo:".
    """
    llm = get_llm(task="decide")
    parser = StrOutputParser()

    system = """You are an email assistant. Analyze the email and suggest quick actions the user can take.

Each action MUST start with exactly one of these prefixes:
- "Reply: " — a short description of a reply to send TO THE SENDER (e.g. "Reply: Accept the invitation", "Reply: Ask for more details")
- "Todo: " — a short task the user should do based on this email (e.g. "Todo: Submit hackathon form by Friday", "Todo: Review attached document")
- "Schedule: " — for meeting/scheduling requests. Creates a calendar event with free slot lookup. (e.g. "Schedule: meeting with John", "Schedule: project review session")

CRITICAL RULES:
- Reply actions are messages TO SEND to the other person. They must make sense as something you'd say TO THEM.
- NEVER suggest replies that read like personal notes (e.g. "Reply: Check the update", "Reply: Read the document", "Reply: Review the details"). These should be Todo items instead.
- Replies must be conversational actions: accepting, declining, requesting info, confirming, thanking, asking questions, etc.
- Only suggest Reply actions if the email actually warrants a reply. Newsletters, automated notifications, marketing emails, FYI-only emails, and system confirmations do NOT need replies.
- If the email is purely informational (a notification, receipt, or announcement), suggest ONLY Todo actions or no actions at all.
- If the email mentions a meeting, call, appointment, or catch-up, ALWAYS include one "Schedule:" action.
- Schedule actions describe WHAT to schedule, NEVER specific dates or times. The scheduling system will check the user's calendar for availability. Examples:
  GOOD: "Schedule: meeting with Dr. Smith", "Schedule: workshop follow-up call"
  BAD: "Schedule: 1:45 PM meeting", "Schedule: meeting on March 5th", "Schedule: 30 min call at 2pm"
- Todo actions should be concrete, one-sentence tasks extracted from the email content.
- Suggest 1-4 actions total. Quality over quantity — fewer good suggestions beat many bad ones.
- Each action description should be concise (under 10 words after the prefix).
- Do NOT suggest generic actions. Every action must be specific to THIS email's content.

Output ONLY a JSON array of strings. No other text."""

    subject = email.get("subject", "")[:120]
    body_text = (email.get("clean_body") or email.get("body") or email.get("snippet") or "")[:600]
    thread_ctx = (email.get("thread_context") or "")[:350]
    related_ctx = (email.get("related_context") or "")[:500]
    category = email.get("category", "normal")
    needs_action = bool(email.get("needs_action"))

    # Pull rich memory context from the persistent brain
    memory_ctx = ""
    try:
        memory_ctx = build_memory_context(email, max_linked=5)[:600]
    except Exception:
        pass

    # Pull sender history for additional context
    sender_history = ""
    try:
        from tools.gmail_tools import extract_email_address
        sender_addr = extract_email_address(email.get("sender", ""))
        history = get_sender_history(sender_addr, exclude_id=email.get("id", ""), limit=3)
        if history:
            lines = []
            for h in history:
                lines.append(f"- {h.get('subject', '')} ({h.get('category', '')}) [{h.get('date', '')}]")
            sender_history = "Previous emails from this sender:\n" + "\n".join(lines)
    except Exception:
        pass

    prompt = f"""Email:
Subject: {subject}
Category: {category}
Needs action: {needs_action}
Content: {body_text}
Thread context (recent messages in THIS thread, if any):
{thread_ctx}

Related emails (other threads that may be about the same topic or sender/domain):
{related_ctx}

{(email.get('enriched_context') or '')[:500]}

{memory_ctx}

{sender_history}

Suggest quick actions (ONLY JSON ARRAY of "Reply: ..." and/or "Todo: ..." strings):"""

    try:
        chain = llm | parser
        raw = chain.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])

        # Clean up response and robustly extract JSON array
        text = (raw or "").strip()
        text = re.sub(r"^```[a-zA-Z0-9_]*\s*", "", text)
        text = re.sub(r"```\s*$", "", text).strip()
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]

        # Parse JSON
        options = json.loads(text)

        if isinstance(options, list) and options:
            # Ensure all options are strings with valid prefixes, deduplicate
            cleaned: list[str] = []
            seen = set()
            for o in options:
                s = str(o).strip()
                key = s.lower()
                if not s or key in seen:
                    continue
                # Ensure proper prefix
                if not (s.startswith("Reply:") or s.startswith("Todo:") or s.startswith("Schedule:")):
                    continue
                seen.add(key)
                cleaned.append(s)
            if len(cleaned) >= 1:
                return {"decision_options": cleaned[:5]}
    except json.JSONDecodeError as e:
        logger.warning("JSON parse error in decision_suggester: %s (raw: %s)", e, (raw or "")[:500])
        raise ValueError(f"Decision suggester failed: could not parse JSON. Raw: {(raw or '')[:500]}") from e
    except Exception as e:
        logger.warning("Error in decision_suggester: %s", e)
        raise

    raise ValueError(f"Decision suggester failed: LLM returned invalid format. Raw: {(raw or '')[:500]}")
