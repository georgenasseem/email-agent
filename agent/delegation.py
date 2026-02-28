"""Simple delegation decider based on delegation_rules."""
from typing import Dict, Any

from tools.gmail_tools import get_profile_email
from agent.memory_store import get_delegation_rules, add_memory


def decide_delegation(email: Dict[str, Any]) -> Dict[str, Any]:
    """Return {'delegate_to': target_email} if any rule matches, else {}."""
    try:
        user_email = get_profile_email()
    except Exception:
        user_email = ""

    if not user_email:
        return {}

    rules = get_delegation_rules(user_email)
    if not rules:
        return {}

    text = (
        (email.get("subject", "") or "")
        + " "
        + ((email.get("body") or email.get("snippet") or "") or "")
    ).lower()

    best_rule = None
    for rule in rules:
        if not rule.get("enabled", 1):
            continue
        pattern = (rule.get("pattern") or "").lower()
        if not pattern:
            continue
        if pattern in text:
            if best_rule is None or int(rule.get("weight", 1)) > int(best_rule.get("weight", 1)):
                best_rule = rule

    if not best_rule:
        return {}

    target = best_rule.get("target_email")
    if not target:
        return {}

    # Log suggestion into memory
    try:
        add_memory(
            user_email=user_email,
            kind="delegation_suggestion",
            key=str(target),
            value=email.get("subject", ""),
            source="delegation_decider",
        )
    except Exception:
        pass

    return {"delegate_to": target}

