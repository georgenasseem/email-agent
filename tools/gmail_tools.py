"""Gmail API tools for fetching, reading, and drafting emails."""
import base64
import html
import logging
import re as _re
from typing import Optional

from googleapiclient.discovery import build

from tools.google_auth import get_google_credentials
from agent.text_cleaner import clean_email_text

logger = logging.getLogger(__name__)


def get_gmail_service():
    """Authenticate and return Gmail API service."""
    creds = get_google_credentials()
    return build("gmail", "v1", credentials=creds)


def get_profile_email() -> str:
    """Return the authenticated user's primary email address."""
    service = get_gmail_service()
    profile = service.users().getProfile(userId="me").execute()
    return profile.get("emailAddress", "")


def decode_message_body(payload: dict) -> str:
    """Extract plain text body from Gmail message payload, with HTML fallback handling."""
    def _collect_parts(part: dict) -> list[dict]:
        """Recursively collect all leaf parts from a (possibly nested) MIME structure."""
        if "parts" in part:
            result = []
            for sub in part["parts"]:
                result.extend(_collect_parts(sub))
            return result
        return [part]

    # Try to get plain text from top-level body first
    if "body" in payload and payload["body"].get("data"):
        data = payload["body"].get("data", "")
        if data:
            decoded = base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
            logger.debug("Got plain text body (direct): %s...", decoded[:200])
            return decoded

    # Recursively collect all leaf parts (handles nested multipart structures)
    all_parts = _collect_parts(payload)

    # Look for plain text part
    for part in all_parts:
        if part.get("mimeType") == "text/plain" and part.get("body", {}).get("data"):
            data = part.get("body", {}).get("data", "")
            if data:
                decoded = base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
                logger.debug("Got plain text part: %s...", decoded[:200])
                return decoded

    # If no plain text, try HTML and strip tags
    for part in all_parts:
        if part.get("mimeType") == "text/html" and part.get("body", {}).get("data"):
            data = part.get("body", {}).get("data", "")
            if data:
                html_text = base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
                logger.debug("Raw HTML (first 500 chars): %s...", html_text[:500])

                # Remove <style> and <script> blocks completely
                clean = _re.sub(r'<style[^>]*>.*?</style>', '', html_text, flags=_re.DOTALL | _re.IGNORECASE)
                clean = _re.sub(r'<script[^>]*>.*?</script>', '', clean, flags=_re.DOTALL | _re.IGNORECASE)
                # Strip all remaining HTML tags
                clean = _re.sub(r'<[^>]+>', '', clean)
                # Decode HTML entities
                clean = html.unescape(clean)
                # Remove excessive whitespace but preserve line breaks for readability
                clean = _re.sub(r'[ \t]+', ' ', clean)  # Multiple spaces/tabs -> single space
                clean = _re.sub(r'\n\s*\n+', '\n\n', clean)  # Multiple newlines -> double newline
                # Remove leading/trailing whitespace from each line
                lines = clean.split('\n')
                lines = [line.strip() for line in lines if line.strip()]
                clean = '\n'.join(lines)

                logger.debug("Cleaned HTML body (first 300 chars): %s...", clean[:300])

                return clean.strip()

    return ""


def decode_message_body_html(payload: dict, gmail_msg_id: str = "") -> str:
    """Extract a display-ready HTML body from a Gmail message payload.

    Preserves formatting tags (bold, italic, links, lists, paragraphs, etc.)
    while stripping dangerous/noisy elements (scripts, styles, tracking pixels).
    Converts inline CID images to base64 data URIs so they display correctly.
    If the email only has a plain-text part, wraps it in basic HTML with <br>.
    """
    import html as _html

    def _collect_parts(part: dict) -> list[dict]:
        if "parts" in part:
            result = []
            for sub in part["parts"]:
                result.extend(_collect_parts(sub))
            return result
        return [part]

    all_parts = _collect_parts(payload)

    # Build a CID → data-URI map for inline images
    cid_map: dict[str, str] = {}
    for part in all_parts:
        cid = ""
        for header in part.get("headers", []):
            if header["name"].lower() == "content-id":
                cid = header["value"].strip("<>")
                break
        if not cid:
            continue
        mime_type = part.get("mimeType", "")
        if not mime_type.startswith("image/"):
            continue
        body_data = part.get("body", {}).get("data", "")
        att_id = part.get("body", {}).get("attachmentId", "")
        if body_data:
            cid_map[cid] = f"data:{mime_type};base64,{body_data}"
        elif att_id:
            # We'll resolve attachment data lazily below
            cid_map[cid] = f"__att__{att_id}__{mime_type}"

    def _resolve_cids(html_str: str, mid: str = "") -> str:
        """Replace cid: references with data URIs."""
        def _cid_repl(m):
            cid = m.group(1)
            data_uri = cid_map.get(cid, "")
            if not data_uri:
                return m.group(0)
            if data_uri.startswith("__att__") and mid:
                # Fetch the attachment from Gmail
                try:
                    parts = data_uri.split("__")
                    att_id = parts[2]
                    mime = parts[3]
                    service = get_gmail_service()
                    att = service.users().messages().attachments().get(
                        userId="me", messageId=mid, id=att_id
                    ).execute()
                    b64 = att.get("data", "")
                    data_uri = f"data:{mime};base64,{b64}"
                    cid_map[cid] = data_uri  # cache
                except Exception:
                    return m.group(0)
            return data_uri
        return _re.sub(r'cid:([^\s"\'<>]+)', _cid_repl, html_str, flags=_re.IGNORECASE)

    # Use the Gmail API message ID for fetching attachments
    mid = gmail_msg_id

    # 1. Prefer text/html part
    html_raw = ""
    for part in all_parts:
        if part.get("mimeType") == "text/html" and part.get("body", {}).get("data"):
            data = part["body"]["data"]
            html_raw = base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
            break

    if html_raw:
        sanitized = _sanitize_html(html_raw)
        if cid_map:
            sanitized = _resolve_cids(sanitized, mid)
        return sanitized

    # 2. Fallback: plain text → wrap in HTML
    for part in all_parts:
        if part.get("mimeType") == "text/plain" and part.get("body", {}).get("data"):
            data = part["body"]["data"]
            plain = base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
            safe = _html.escape(plain).replace("\n", "<br>")
            return f'<div style="font-family:Arial,sans-serif;font-size:14px;">{safe}</div>'

    # 3. Top-level body (non-multipart)
    if "body" in payload and payload["body"].get("data"):
        data = payload["body"]["data"]
        decoded = base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
        mime = payload.get("mimeType", "")
        if "html" in mime:
            sanitized = _sanitize_html(decoded)
            if cid_map:
                sanitized = _resolve_cids(sanitized, mid)
            return sanitized
        safe = _html.escape(decoded).replace("\n", "<br>")
        return f'<div style="font-family:Arial,sans-serif;font-size:14px;">{safe}</div>'

    return ""


def _sanitize_html(raw: str) -> str:
    """Strip dangerous/noisy HTML while preserving readable formatting."""
    s = _re.sub(r'<script[^>]*>.*?</script>', '', raw, flags=_re.DOTALL | _re.IGNORECASE)
    s = _re.sub(r'<style[^>]*>.*?</style>', '', s, flags=_re.DOTALL | _re.IGNORECASE)
    # Remove tracking pixels (1x1 images)
    s = _re.sub(r'<img[^>]*(width\s*=\s*["\']?1["\']?|height\s*=\s*["\']?1["\']?)[^>]*/?>',
                '', s, flags=_re.IGNORECASE)
    s = _re.sub(r'<img[^>]*display\s*:\s*none[^>]*/?>',
                '', s, flags=_re.IGNORECASE)
    # Remove <head>...</head>
    s = _re.sub(r'<head[^>]*>.*?</head>', '', s, flags=_re.DOTALL | _re.IGNORECASE)
    # Remove <html>, <body>, <!DOCTYPE> wrappers (keep inner content)
    s = _re.sub(r'<!DOCTYPE[^>]*>', '', s, flags=_re.IGNORECASE)
    s = _re.sub(r'</?html[^>]*>', '', s, flags=_re.IGNORECASE)
    s = _re.sub(r'</?body[^>]*>', '', s, flags=_re.IGNORECASE)
    # Remove HTML comments
    s = _re.sub(r'<!--.*?-->', '', s, flags=_re.DOTALL)
    # Strip class/id/data-* attributes but keep href, src, style, target
    s = _re.sub(r'\s+(class|id|data-[a-z-]+)\s*=\s*"[^"]*"', '', s, flags=_re.IGNORECASE)
    s = _re.sub(r"\s+(class|id|data-[a-z-]+)\s*=\s*'[^']*'", '', s, flags=_re.IGNORECASE)
    # Strip inline color/background styles so text is readable on dark background
    s = _re.sub(r'color\s*:\s*[^;\"\']+;?', '', s, flags=_re.IGNORECASE)
    s = _re.sub(r'background(?:-color)?\s*:\s*[^;\"\']+;?', '', s, flags=_re.IGNORECASE)
    # Remove highlight marks and any highlight-related spans
    s = _re.sub(r'</?mark[^>]*>', '', s, flags=_re.IGNORECASE)
    s = _re.sub(r'<span[^>]*highlight[^>]*>(.*?)</span>', r'\1', s, flags=_re.DOTALL | _re.IGNORECASE)
    # Make all links open in new tab
    s = _re.sub(r'<a\s', '<a target="_blank" rel="noopener" ', s, flags=_re.IGNORECASE)
    return s.strip()


def get_header(headers: list, name: str) -> str:
    """Get header value by name."""
    for h in headers:
        if h["name"].lower() == name.lower():
            return h["value"]
    return ""


def fetch_emails(max_results: int = 20, query: str = "in:inbox is:unread") -> list[dict]:
    """Fetch emails from Gmail. Returns list of parsed email dicts."""
    service = get_gmail_service()
    # Exclude messages sent by the authenticated user (so replies/drafts don't include user's own sent emails)
    my_email = get_profile_email()
    results = service.users().messages().list(userId="me", maxResults=max_results, q=query).execute()
    messages = results.get("messages", [])

    emails = []
    for msg_ref in messages:
        msg = service.users().messages().get(userId="me", id=msg_ref["id"], format="full").execute()
        payload = msg.get("payload", {})
        headers = payload.get("headers", [])

        subject = get_header(headers, "Subject")
        sender = get_header(headers, "From")
        # Skip messages that are from the authenticated user
        if my_email and my_email.lower() in sender.lower():
            continue
        date = get_header(headers, "Date")
        raw_body = decode_message_body(payload)
        body = raw_body or ""
        clean_body = clean_email_text(body) if body else ""
        body_html = decode_message_body_html(payload, gmail_msg_id=msg_ref["id"])
        internal_date = int(msg.get("internalDate", "0")) if msg.get("internalDate") else 0

        emails.append(
            {
                "id": msg["id"],
                "thread_id": msg.get("threadId", ""),
                "subject": subject,
                "sender": sender,
                "date": date,
                "internal_date": internal_date,
                "body": body,
                "clean_body": clean_body,
                "body_html": body_html,
                "snippet": msg.get("snippet", ""),
            }
        )

    return emails


def fetch_thread(thread_id: str) -> list[dict]:
    """Fetch full Gmail thread by ID. Returns list of message dicts in the thread."""
    if not thread_id:
        return []

    service = get_gmail_service()
    thread = service.users().threads().get(userId="me", id=thread_id, format="full").execute()
    messages = thread.get("messages", [])

    thread_messages: list[dict] = []
    for msg in messages:
        payload = msg.get("payload", {})
        headers = payload.get("headers", [])
        subject = get_header(headers, "Subject")
        sender = get_header(headers, "From")
        date = get_header(headers, "Date")
        raw_body = decode_message_body(payload)
        body = raw_body or ""
        clean_body = clean_email_text(body) if body else ""
        body_html = decode_message_body_html(payload, gmail_msg_id=msg.get("id", ""))

        thread_messages.append(
            {
                "id": msg.get("id", ""),
                "thread_id": msg.get("threadId", thread_id),
                "subject": subject,
                "sender": sender,
                "date": date,
                "body": body,
                "clean_body": clean_body,
                "body_html": body_html,
                "snippet": msg.get("snippet", ""),
            }
        )

    return thread_messages


def fetch_sent_samples(max_results: int = 20) -> list[str]:
    """Fetch recent sent messages bodies as samples for style learning."""
    service = get_gmail_service()
    # Use the "in:sent" query to get recent sent messages
    results = service.users().messages().list(userId="me", maxResults=max_results, q="in:sent").execute()
    messages = results.get("messages", [])
    samples = []
    for msg_ref in messages:
        try:
            msg = service.users().messages().get(userId="me", id=msg_ref["id"], format="full").execute()
            payload = msg.get("payload", {})
            body = decode_message_body(payload)
            if body:
                samples.append(body[:4000])
        except Exception:
            continue
    return samples


def send_email(
    to: str,
    subject: str,
    body: str,
    thread_id: Optional[str] = None,
) -> dict:
    """Send an email via Gmail API. Returns the send response dict."""
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    service = get_gmail_service()

    message = MIMEMultipart()
    message["to"] = to
    message["subject"] = subject
    message.attach(MIMEText(body, "plain"))

    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    send_body = {"raw": raw}
    if thread_id:
        send_body["threadId"] = thread_id

    sent = service.users().messages().send(userId="me", body=send_body).execute()
    return sent


def extract_email_address(header_value: str) -> str:
    """Extract email address from 'Name <email@domain.com>' format."""
    if "<" in header_value and ">" in header_value:
        start = header_value.index("<") + 1
        end = header_value.index(">")
        return header_value[start:end].strip()
    return header_value.strip()
