"""Draft email replies using style prompt and optional user style notes."""
import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

from agent.llm import get_llm
from agent.style_learner import load_persisted_style
from agent.profile import load_profile
from agent.email_memory import build_memory_context, get_sender_history, get_greeting_for_contact

logger = logging.getLogger(__name__)


def analyze_roles(email: dict) -> dict:
    """Infer who is who in the thread so the model keeps the correct POV.
    
    Also looks up how the user previously addressed this sender for
    contact-specific greetings (e.g. 'Dear Professor' vs 'Dear Hanan').
    """
    try:
        profile = load_profile()
    except Exception:
        profile = {}
    user_email = (profile.get("email") or "").lower()
    user_name = profile.get("display_name") or profile.get("first_name") or ""
    other_email_raw = (email.get("sender") or "").lower()

    # Extract clean email address
    if "<" in other_email_raw and ">" in other_email_raw:
        other_email = other_email_raw.split("<")[1].split(">")[0].strip()
    else:
        other_email = other_email_raw.strip()

    # Extract display name from sender header
    sender_header = email.get("sender", "")
    _name_match = re.match(r'^([^<]+)', sender_header)
    other_display_name = ""
    if _name_match:
        other_display_name = _name_match.group(1).strip().strip('"').strip("'")
    if not other_display_name or "@" in other_display_name:
        other_display_name = other_email.split("@")[0] if "@" in other_email else other_email or "there"

    # Look up how user previously addressed this contact
    contact_greeting = ""
    try:
        contact_greeting = get_greeting_for_contact(other_email)
    except Exception:
        pass

    return {
        "user_email": user_email or "you@example.com",
        "user_name": user_name,
        "other_party_email": other_email,
        "other_display_name": other_display_name,
        "other_first_name": other_display_name.split()[0] if other_display_name else "there",
        "contact_greeting": contact_greeting,  # e.g. "Dear Professor Salam" or ""
        "thread_role": {
            "user_is_recipient": True,
            "user_is_sender": False,
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
    - decision: user's choice from quick actions (e.g. "Accept the meeting", "Ask for details")
    - style_notes: optional, e.g. "I write short, friendly emails"
    
    Architecture: Quick actions decided WHAT to do. This function figures out HOW.
    It thinks like the user and talks like the user.
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
        
        # Build style instruction
        if style_notes:
            style = f"Write in this style description: {style_notes}"
        elif persisted:
            style = (
                "Match this user style description as closely as possible:\n"
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

        # Extract greeting PATTERN (e.g. "Dear", "Hi") from style
        greeting_pattern = ""
        if persisted:
            for line in persisted.splitlines():
                if line.strip().upper().startswith("GREETING_STYLE:"):
                    greeting_pattern = line.split(":", 1)[1].strip()
                    break

        closing_instruction = (
            f"3. CLOSING: End with a sign-off (e.g. Best regards, Thanks, Best) followed by a newline and '{signature_name}'."
            if signature_name
            else "3. CLOSING: End with a sign-off (e.g. Best regards, Thanks, Best) followed by a line break."
        )

        roles = analyze_roles(email)
        _other_display_name = roles.get("other_display_name", "there")
        _other_first_name = roles.get("other_first_name", "there")
        _user_name = roles.get("user_name", "")
        _user_email = roles.get("user_email", "")
        _contact_greeting = roles.get("contact_greeting", "")

        # Build the greeting instruction — contact-specific if available
        if _contact_greeting:
            greeting_instruction = (
                f"1. GREETING: Start with exactly '{_contact_greeting},' — this is how you always address this person."
            )
        elif greeting_pattern and greeting_pattern.lower() not in ("(no consistent greeting detected)",):
            greeting_instruction = (
                f"1. GREETING: Start with '{greeting_pattern} {_other_first_name},' — use this greeting pattern."
            )
        else:
            greeting_instruction = (
                f"1. GREETING: Start with an appropriate greeting addressing {_other_first_name}."
            )

        # Build reply plan if the decision has enough substance
        reply_plan = plan_reply(email, decision=decision) if len((decision or "").split()) >= 5 else {
            "goal": f"Respond to the email about '{email.get('subject', '')[:80]}' with intent: {decision or 'General response'}",
            "key_points": ["Address the main request", "Indicate next steps if any"],
            "tone": "neutral",
            "risks": [],
        }

        system = f"""You are drafting an email reply AS the user. You must think like them and write like them.

MANDATORY STRUCTURE — output MUST include all three parts:
{greeting_instruction}
2. BODY: 2-4 sentences addressing the email content. Execute the user's intent: {decision if decision else "General response"}
{closing_instruction}

ABSOLUTE IDENTITY RULES (NEVER VIOLATE):
- You ARE: {_user_email}{f' ({_user_name})' if _user_name else ''}. You are writing FROM this person.
- You are writing TO: {roles.get('other_party_email') or 'the sender'} (name: {_other_display_name}).
- The greeting MUST address {_other_display_name} (the recipient), NEVER address yourself ({_user_name}).
- NEVER introduce yourself as the sender's name. You are {_user_name or 'the user'}.
- If the original email was sent BY someone, your reply is TO them — not FROM them.

ANTI-COPYING RULES (CRITICAL):
- NEVER copy, paraphrase, or echo the original email's sentences back.
- NEVER repeat the sender's own words or phrases from their email.
- Write ORIGINAL content that RESPONDS to what they said, don't summarize what they said.
- Your reply should contain NEW information, decisions, or questions — not a restatement.

STYLE: {style}

RULES:
- Output ONLY the email body text. No subject line. No markdown. No quotes.
- Sound natural. Match the tone of the original email when appropriate.
- Be concise — aim for the minimum words needed to communicate clearly.

REPLY PLAN (what to accomplish):
- Goal: {reply_plan.get('goal', '')}
- Key points: {', '.join(reply_plan.get('key_points', []) or [])}
- Tone: {reply_plan.get('tone', 'neutral')}"""

        decision_text = ""
        if decision:
            decision_text = f"\n\nUser chose this action: {decision}. The reply must execute this choice."

        subject_text = email.get('subject', '')[:100]
        body_text = (email.get('clean_body') or email.get('body', '') or email.get('snippet', ''))[:600]
        thread_ctx = (email.get('thread_context') or '')[:400]

        # Pull memory context for drafting
        memory_ctx = ""
        try:
            memory_ctx = build_memory_context(email, max_linked=5)[:500]
        except Exception:
            pass

        # Build profile context
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

        # Pull sender history for context
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

        prompt = f"""Reply to this email as {_user_name or 'the user'}.

Sender (you're replying TO): {_other_display_name} <{roles.get('other_party_email', '')}>
Subject: {subject_text}

Their email:
{body_text}

Thread context:
{thread_ctx}

{(email.get('enriched_context') or '')[:500]}

{memory_ctx}

{profile_ctx}

{sender_history}{decision_text}

Write a complete reply (greeting + body + closing). Do NOT copy their email — write original content:"""

        chain = llm | parser
        draft = chain.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])

        result = draft.strip() if draft else ""

        # Ensure closing includes name if we have it and the draft doesn't already end with it
        if signature_name and result:
            last_lines = result.rsplit("\n", 2)
            last_text = "".join(last_lines[-2:]).lower()
            if signature_name.lower() not in last_text:
                result = result.rstrip()
                if not result.endswith(signature_name):
                    result = f"{result}\n\n{signature_name}"
        return result

    except Exception as e:
        logger.warning("Error in draft_reply: %s", e)
        raise
