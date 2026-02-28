"""Draft email replies using style prompt and optional user style notes."""
import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

from agent.llm import get_llm
from agent.style_learner import load_persisted_style
from agent.profile import load_profile
from agent.email_memory import build_memory_context, get_sender_history

logger = logging.getLogger(__name__)


def analyze_roles(email: dict) -> dict:
    """Infer who is who in the thread so the model keeps the correct POV."""
    try:
        profile = load_profile()
    except Exception:
        profile = {}
    user_email = (profile.get("email") or "").lower()
    other_email = (email.get("sender") or "").lower()

    # Heuristic: in this app, the user is the authenticated Gmail account,
    # and incoming emails have sender = other party.
    user_is_recipient = True
    user_is_sender = False

    return {
        "user_email": user_email or "you@example.com",
        "other_party_email": other_email,
        "other_party_name": other_email.split("@")[0] if "@" in other_email else other_email or "sender",
        "thread_role": {
            "user_is_recipient": user_is_recipient,
            "user_is_sender": user_is_sender,
        },
    }


def plan_reply(email: dict, decision: str = "") -> dict:
    """Create a structured reply plan before drafting prose."""
    llm = get_llm(task="plan_reply")
    parser = StrOutputParser()

    roles = analyze_roles(email)
    subject_text = email.get("subject", "")[:120]
    body_text = (email.get("clean_body") or email.get("body") or email.get("snippet") or "")[:800]
    thread_ctx = (email.get("thread_context") or "")[:500]
    related_ctx = (email.get("related_context") or "")[:600]

    # Pull rich memory context from the persistent brain
    memory_ctx = ""
    try:
        memory_ctx = build_memory_context(email, max_linked=5)[:600]
    except Exception:
        pass

    system = """You are helping plan an email reply.

First, think about what the user wants to achieve with this reply and what key points must be covered.

You MUST output JSON with this shape:
{
  "goal": "...",
  "key_points": ["...", "..."],
  "tone": "formal|neutral|friendly",
  "risks": ["optional risk or concern", "..."]
}

Rules:
- "goal": 1 sentence summary of what this reply should accomplish.
- "key_points": 2-5 concrete bullets the reply must mention or resolve.
- "tone": one of "formal", "neutral", or "friendly".
- "risks": can be empty array if nothing important.
"""

    decision_text = decision or "General response"

    prompt = f"""You are planning a reply email.

You are: {roles.get('user_email') or 'the user'}.
They are: {roles.get('other_party_email') or 'the sender'}.
User is currently the RECIPIENT of this email and will be REPLYING to the sender.

Original email subject: {subject_text}
Original email content:
{body_text}

Thread context (recent messages in THIS thread, if any):
{thread_ctx}

Related emails (other threads from possibly different dates/senders that might be relevant):
{related_ctx}

{(email.get('enriched_context') or '')[:500]}

{memory_ctx}

User's chosen intent / action: {decision_text}

Produce the JSON reply plan now:"""

    raw = llm.invoke([SystemMessage(content=system), HumanMessage(content=prompt)], max_tokens=512)

    text = (raw or "").strip()
    text = re.sub(r"^```[a-zA-Z0-9_]*\s*", "", text)
    text = re.sub(r"```\s*$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        text = text[start:end]
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # Safe fallback plan
    return {
        "goal": f"Respond to the email about '{subject_text}' with intent: {decision_text}",
        "key_points": [
            "Acknowledge the sender and their message",
            "Address the main request or question clearly",
            "Indicate any next steps or outcomes",
        ],
        "tone": "neutral",
        "risks": [],
    }


def draft_reply(
    email: dict,
    decision: str = "",
    style_notes: str = "",
) -> str:
    """
    Draft a reply to the email.
    - decision: user's choice (e.g. "Yes", "No", "Attend")
    - style_notes: optional, e.g. "I write short, friendly emails"
    """
    try:
        llm = get_llm(task="draft")
        parser = StrOutputParser()

        # Load persisted style (may include GREETING_STYLE / CLOSING_STYLE patterns)
        persisted = ""
        try:
            persisted = load_persisted_style()
        except Exception:
            persisted = ""
        
        # Build style instruction without hardcoded greeting/closing
        if style_notes:
            style = f"Write in this style description: {style_notes}"
        elif persisted:
            style = (
                "Match this user style description as closely as possible, "
                "including how they open and close emails:\n"
                f"{persisted}"
            )
        else:
            style = (
                "Write in a neutral, professional, and concise tone appropriate for workplace email. "
                "Choose a greeting and closing that fit this context."
            )

        # Extract SIGNATURE_NAME from persisted style for closing
        signature_name = ""
        if persisted:
            for line in persisted.splitlines():
                if line.strip().upper().startswith("SIGNATURE_NAME:"):
                    signature_name = line.split(":", 1)[1].strip()
                    break

        closing_instruction = (
            f"3. CLOSING: End with a sign-off (e.g. Best regards, Thanks, Best, Sincerely) followed by a newline and the user's name '{signature_name}'."
            if signature_name
            else "3. CLOSING: End with a sign-off (e.g. Best regards, Thanks, Best, Sincerely) followed by a line break."
        )

        roles = analyze_roles(email)

        # Extract the OTHER person's display name (not the user's own name)
        _other_sender = email.get('sender', '') or ''
        _other_name_match = re.match(r'^([^<]+)', _other_sender)
        _other_display_name = ''
        if _other_name_match:
            _other_display_name = _other_name_match.group(1).strip().strip('"').strip("'")
        if not _other_display_name or '@' in _other_display_name:
            _other_display_name = _other_sender.split('@')[0] if '@' in _other_sender else 'there'
        # Extract first name only for greeting
        _other_first_name = _other_display_name.split()[0] if _other_display_name else 'there'

        # Get the user's own name to avoid self-addressing
        _user_name = ''
        try:
            _prof = load_profile()
            _user_name = _prof.get('display_name', '') or ''
            if not _user_name:
                _user_email = _prof.get('email', '')
                if _user_email:
                    _user_name = _user_email.split('@')[0]
        except Exception:
            pass

        reply_plan = plan_reply(email, decision=decision)

        system = f"""You are a professional email drafter. You MUST output a complete email reply with three parts: (1) GREETING, (2) BODY, (3) CLOSING.

MANDATORY STRUCTURE - your output MUST include all three:
1. GREETING: Start with a greeting (e.g. Hi {_other_first_name}, Dear {_other_first_name}, Hello {_other_first_name}). Address THE SENDER, NOT yourself.
2. BODY: 2-4 sentences addressing the email content. Incorporate the user's intent: {decision if decision else "General response"} and the reply plan below.
{closing_instruction}

CRITICAL IDENTITY RULES:
- You are writing FROM: {roles.get('user_email') or 'the user'}{f' ({_user_name})' if _user_name else ''}.
- You are writing TO: {roles.get('other_party_email') or 'the sender'} (name: {_other_display_name}).
- The greeting MUST address {_other_first_name} (the sender), NEVER address {_user_name or 'yourself'}.
- If the sender name equals the user name, use a generic greeting like "Hi" instead.

STYLE: {style}

RULES:
- Output ONLY the email body text. No subject line. No markdown.
- Never output just "Thank you for your message" - always include greeting, substantive body, and closing.
- Sound natural. Match the tone of the original email when appropriate.

REPLY PLAN (authoritative):
- Goal: {reply_plan.get('goal', '')}
- Key points: {', '.join(reply_plan.get('key_points', []) or [])}
- Tone: {reply_plan.get('tone', 'neutral')}"""

        decision_text = ""
        if decision:
            decision_text = f"\n\nUser chose this reply intent: {decision}. The body must reflect this choice."

        subject_text = email.get('subject', '')[:100]
        body_text = (email.get('clean_body') or email.get('body', '') or email.get('snippet', ''))[:600]
        thread_ctx = (email.get('thread_context') or '')[:400]

        # Pull memory context for drafting
        memory_ctx = ""
        try:
            memory_ctx = build_memory_context(email, max_linked=5)[:500]
        except Exception:
            pass

        # Build profile context from profile.json
        profile_ctx = ""
        try:
            _full_prof = load_profile()
            _prof_parts = []
            if _full_prof.get('organization'):
                _prof_parts.append(f"Organization: {_full_prof['organization']}")
            if _full_prof.get('role'):
                _prof_parts.append(f"Role: {_full_prof['role']}")
            comm_prefs = _full_prof.get('communication_preferences', {})
            if comm_prefs.get('formality_level'):
                _prof_parts.append(f"Formality: {comm_prefs['formality_level']}")
            if comm_prefs.get('response_length'):
                _prof_parts.append(f"Length preference: {comm_prefs['response_length']}")
            if _prof_parts:
                profile_ctx = "User profile: " + ", ".join(_prof_parts)
        except Exception:
            pass

        # Pull sender history
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

        prompt = f"""Reply to this email. Sender: {_other_display_name}. Subject: {subject_text}

Email content:
{body_text}

Thread context (recent messages in this conversation, if any):
{thread_ctx}

{(email.get('enriched_context') or '')[:500]}

{memory_ctx}

{profile_ctx}

{sender_history}{decision_text}

Write a complete reply with greeting, body, and closing (body only, no subject):"""

        chain = llm | parser
        draft = chain.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])

        result = draft.strip() if draft else ""

        # Ensure closing includes name if we have it and the draft doesn't already end with it
        if signature_name and result:
            last_lines = result.rsplit("\n", 2)
            last_text = "".join(last_lines[-2:]).lower()
            if signature_name.lower() not in last_text:
                # Append name after closing if missing
                result = result.rstrip()
                if not result.endswith(signature_name):
                    result = f"{result}\n\n{signature_name}"
        return result

    except Exception as e:
        logger.warning("Error in draft_reply: %s", e)
        raise
