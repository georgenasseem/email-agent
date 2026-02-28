"""Deterministic email text cleaner for display (no LLM)."""

import html
import re


def clean_email_text(email_text: str) -> str:
    """
    Clean and format email text for readable display using pure Python.

    Removes:
    - URL-encoded Proofpoint protection links
    - Excessive whitespace
    - HTML-ish artifacts and stray markup
    - Quoted-text separators and reply markers
    - Forwarded email headers and signature delimiters

    Returns clean, readable text.
    """
    if not email_text or not email_text.strip():
        return "(No content available)"

    cleaned = _aggressive_regex_cleanup(email_text)
    return cleaned.strip() if cleaned.strip() else "(No content available)"


def _aggressive_regex_cleanup(text: str) -> str:
    """Aggressive regex-based cleanup of common email artifacts."""
    if not text:
        return ""

    # First, extract and replace Proofpoint URLs with clean versions
    def decode_proofpoint_url(match):
        full_url = match.group(0)
        # Extract the u= parameter value
        u_match = re.search(r"u=([^&]+)", full_url)
        if u_match:
            encoded_url = u_match.group(1)
            # First, handle __ as forward slash (done before other replacements)
            decoded = encoded_url.replace("__", "/")
            # Then decode URL encodings: -3A=:, -2F=/, -5F=_, -2D=-, etc.
            decoded = decoded.replace("-3A", ":").replace("-2F", "/")
            decoded = decoded.replace("-5F", "_").replace("-2D", "-")
            decoded = decoded.replace("-2B", "+").replace("-3F", "?")
            decoded = decoded.replace("-3D", "=").replace("-26", "&")
            if decoded.startswith("http"):
                return f" ({decoded}) "
        return ""

    text = re.sub(
        r"https://urldefense\.proofpoint\.com/v2/url\?u=[^&)]+[^)]*",
        decode_proofpoint_url,
        text,
        flags=re.IGNORECASE,
    )

    # Remove any remaining Proofpoint protection wrapper parentheses
    text = re.sub(
        r"\(\s*https://urldefense\.proofpoint\.com/[^)]*\)",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Strip common reply/forward headers and separators
    artifacts = [
        r"------+\s*Forwarded message\s*-+",
        r"-----+\s*Original Message\s*-+",
        r"On .{1,200}wrote:",
        r"On .{1,200}wrote:$",
        r"^\s*From:\s.*$",
        r"^\s*Sent:\s.*$",
        r"^\s*To:\s.*$",
        r"^\s*Subject:\s.*$",
        r"^>.*$",  # quoted lines
        r"^\s*--\s*$",  # signature delimiter
    ]
    for pattern in artifacts:
        text = re.sub(pattern, "", text, flags=re.MULTILINE | re.IGNORECASE)

    # Remove lines that are just quoted markers
    text = re.sub(r"^\s*[>|]\s*$", "", text, flags=re.MULTILINE)

    # Fix line breaks with extra spaces
    text = re.sub(r"\n\s{2,}", "\n", text)

    # Remove lines that are just parentheses and whitespace
    text = re.sub(r"\n\s*\(\s*\)\s*\n", "\n", text)

    # Remove excessive blank lines (more than 2 in a row)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Clean up spacing at line ends
    text = re.sub(r"\s+$", "", text, flags=re.MULTILINE)

    # Remove HTML-like tags
    text = re.sub(r"<[^>]+>", "", text)

    # Decode HTML entities (&nbsp;, &amp;, &aacute;, etc.)
    text = html.unescape(text)

    # Fix common markdown/URL issues
    text = re.sub(r"\(\s+https://", "(https://", text)  # Fix spacing in URLs

    # Remove extra spaces around asterisks (but preserve emphasis)
    text = re.sub(r"\*\s+", "* ", text)  # "* text" not "*  text"
    text = re.sub(r"\s+\*", " *", text)  # "text *" not "text  *"

    # Remove double spaces
    text = re.sub(r"  +", " ", text)

    return text.strip()
