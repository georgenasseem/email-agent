"""Flag emails that need immediate action using keyword matching."""


def flag_urgent_emails(emails: list[dict]) -> list[dict]:
    """Add needs_action flag to emails that require user action.
    
    Uses keyword matching to distinguish between:
    - Informational emails (important but no action needed)
    - Action-required emails (need user response/action)
    """
    if not emails:
        return []

    result = []
    for e in emails:
        # Combine all text fields for keyword matching
        subject = e.get('subject', '').lower()
        body = e.get('body', '').lower()
        snippet = e.get('snippet', '').lower()
        full_text = f"{subject} {body} {snippet}"
        
        category = e.get('category', 'normal').lower()
        
        # Define trigger categories for different types of urgency
        urgent_triggers = [
            "deadline", "by end of day", "by eod", "by 5pm", "by 6pm",
            "today", "tonight", "asap", "urgent", "immediately", "emergency",
            "time-sensitive", "do not delay"
        ]
        
        action_triggers = [
            "please approve", "please confirm", "please verify", "please respond",
            "please review", "please sign", "please submit", "please provide",
            "needs approval", "awaiting approval", "needs review", "awaiting review",
            "your approval", "your response", "your feedback", "your decision",
            "action required", "requires action", "requires your", "awaiting your",
            "approve", "confirm", "verify", "sign", "submit", "review needed"
        ]
        
        security_triggers = [
            "verify your account", "confirm identity", "password reset", 
            "suspicious activity", "unusual activity", "unauthorized access", 
            "security alert", "compromised", "login attempt",
            "token expired", "token revoked", "access token"
        ]
        
        informational_triggers = [
            "fyi", "for your information", "just letting you know", "update",
            "announcement", "newsletter", "summary", "report", "attached",
            "here is", "here are", "sharing", "resources", "materials"
        ]
        
        # Check which triggers are present
        has_urgent = any(kw in full_text for kw in urgent_triggers)
        has_action = any(kw in full_text for kw in action_triggers)
        has_security = any(kw in full_text for kw in security_triggers)
        has_informational = any(kw in full_text for kw in informational_triggers)
        
        # Determine if action is needed based on trigger combinations
        needs_action = (
            has_security or  # Always need action for security issues
            (has_urgent and has_action) or  # Urgent + action keywords
            (category in ["urgent", "important"] and has_action and not has_informational)  # Important emails with action but not clearly informational
        )
        
        result.append({**e, "needs_action": needs_action})
    
    return result
