"""Context enrichment: detect unknown entities, search DB + Gmail, build knowledge.

This module powers the `enrich_context_node` in the LangGraph pipeline.
When an email references a person, project, event, or org that the agent
doesn't recognise, it:
1. Extracts entities via a lightweight LLM call.
2. Checks the persistent knowledge_base table.
3. Searches the local email DB for mentions.
4. Falls back to a Gmail search to discover who/what it is.
5. Stores every finding in knowledge_base so it never has to search again.
6. Attaches an `enriched_context` string to the email for downstream nodes.
"""
import json
import logging
import re
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

from agent.llm import get_llm
from agent.email_memory import (
    lookup_knowledge,
    upsert_knowledge,
    search_emails_for_entity,
)

logger = logging.getLogger(__name__)


# ─── Entity extraction ──────────────────────────────────────────────────────


def extract_entities(email: dict) -> List[dict]:
    """Use a fast LLM call to pull out people, projects, events, and orgs
    mentioned in the email that the user might need context about.

    Returns a list of dicts: [{"name": "John", "type": "person"}, ...]
    """
    llm = get_llm(task="entity_extract")
    parser = StrOutputParser()

    system = """You are an entity extractor. Given an email, extract specific names, projects, events, and organizations mentioned.

ONLY extract entities that someone might need to look up — skip generic words, the sender's own name/email (they're already known), and common services like Google, Zoom, etc.

Output a JSON array of objects: [{"name": "...", "type": "person|project|event|org"}]
If there are no notable entities, output: []

Rules:
- "person": specific people by name (first name, full name, or handle). NOT generic titles like "team" or "manager".
- "project": specific named projects, products, or initiatives.
- "event": specific named events, conferences, meetings with proper names.
- "org": specific companies, departments, teams with proper names.
- Maximum 6 entities. Focus on the most important ones.
- Output ONLY the JSON array."""

    subject = (email.get("subject") or "")[:120]
    body = (email.get("clean_body") or email.get("body") or email.get("snippet") or "")[:800]
    sender = email.get("sender", "")
    thread_ctx = (email.get("thread_context") or "")[:300]

    prompt = f"""Email:
From: {sender}
Subject: {subject}
Body: {body}
Thread context: {thread_ctx}

Extract entities (JSON array):"""

    try:
        chain = llm | parser
        raw = chain.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])

        text = (raw or "").strip()
        text = re.sub(r"^```[a-zA-Z0-9_]*\s*", "", text)
        text = re.sub(r"```\s*$", "", text).strip()

        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]

        entities = json.loads(text)
        if isinstance(entities, list):
            # Validate and normalize
            cleaned = []
            for ent in entities:
                if isinstance(ent, dict) and ent.get("name") and ent.get("type"):
                    name = str(ent["name"]).strip()
                    etype = str(ent["type"]).strip().lower()
                    if etype not in ("person", "project", "event", "org"):
                        etype = "other"
                    if len(name) >= 2:
                        cleaned.append({"name": name, "type": etype})
            return cleaned[:6]
    except Exception as e:
        logger.warning("Entity extraction failed: %s", e)

    return []


# ─── Resolve a single entity ────────────────────────────────────────────────


def resolve_entity(entity_name: str, entity_type: str, gmail_search_fn=None) -> str:
    """Try to resolve an entity through multiple sources.

    Resolution order:
    1. Knowledge base (instant, cached)
    2. Local email DB (search subjects/bodies/senders)
    3. Gmail API search (only if gmail_search_fn provided)

    Returns a description string, or "" if nothing found.
    Persists any new knowledge found.
    """
    name = entity_name.strip()
    if not name:
        return ""

    # 1. Check knowledge base first
    known = lookup_knowledge(name)
    if known:
        # Already have high-confidence info
        best = max(known, key=lambda k: k.get("confidence", 0))
        if best.get("confidence", 0) >= 0.3:
            return f"{best['entity']} ({best['entity_type']}): {best['info']}"

    # 2. Search local email DB
    db_hits = search_emails_for_entity(name, limit=5)
    if db_hits:
        info_parts = []
        senders = set()
        subjects = set()
        for hit in db_hits:
            sender = hit.get("sender", "")
            subject = hit.get("subject", "")
            summary = hit.get("summary") or hit.get("snippet", "")
            if sender:
                senders.add(sender.split("<")[0].strip())
            if subject:
                subjects.add(subject[:80])
            if summary:
                info_parts.append(summary[:100])

        info = ""
        if entity_type == "person" and senders:
            matching_senders = [s for s in senders if name.lower() in s.lower()]
            if matching_senders:
                info = f"Email sender: {matching_senders[0]}. "
            info += f"Found in {len(db_hits)} emails. "
        else:
            info = f"Referenced in {len(db_hits)} emails. "

        if subjects:
            info += "Related subjects: " + "; ".join(list(subjects)[:3])
        if info_parts:
            info += ". Context: " + info_parts[0]

        # Store in knowledge base
        upsert_knowledge(
            entity=name,
            entity_type=entity_type,
            info=info.strip(),
            source="email_db",
            confidence=0.6,
        )
        return f"{name} ({entity_type}): {info.strip()}"

    # 3. Gmail search as last resort
    if gmail_search_fn:
        try:
            gmail_results = gmail_search_fn(max_results=5, query=f'"{name}"')
            if gmail_results:
                senders = set()
                subjects = set()
                for msg in gmail_results[:5]:
                    sender = msg.get("sender", "")
                    if sender:
                        senders.add(sender.split("<")[0].strip())
                    subj = msg.get("subject", "")
                    if subj:
                        subjects.add(subj[:80])

                info = ""
                if entity_type == "person" and senders:
                    matching = [s for s in senders if name.lower() in s.lower()]
                    if matching:
                        info = f"Gmail contact: {matching[0]}. "
                    else:
                        info = f"Mentioned by: {list(senders)[0]}. "
                else:
                    info = f"Found in {len(gmail_results)} Gmail messages. "
                if subjects:
                    info += "Subjects: " + "; ".join(list(subjects)[:3])

                # Store in knowledge base
                upsert_knowledge(
                    entity=name,
                    entity_type=entity_type,
                    info=info.strip(),
                    source="gmail_search",
                    confidence=0.4,
                )

                # Also store these emails in our local DB for future reference
                from agent.email_memory import store_raw_emails
                store_raw_emails(gmail_results)

                return f"{name} ({entity_type}): {info.strip()}"
        except Exception as e:
            logger.warning("Gmail search for entity '%s' failed: %s", name, e)

    return ""


# ─── Main enrichment function ───────────────────────────────────────────────


def enrich_email_context(email: dict, gmail_search_fn=None) -> dict:
    """Enrich a single email with resolved entity context.

    1. Extracts entities from the email.
    2. Resolves each one (knowledge base → email DB → Gmail).
    3. Attaches `enriched_context` string to the email dict.

    Returns the email dict with `enriched_context` added.
    """
    entities = extract_entities(email)
    if not entities:
        return email

    resolved_parts = []
    for ent in entities:
        info = resolve_entity(
            entity_name=ent["name"],
            entity_type=ent["type"],
            gmail_search_fn=gmail_search_fn,
        )
        if info:
            resolved_parts.append(info)

    if resolved_parts:
        email["enriched_context"] = (
            "=== Enriched context (from agent memory & Gmail) ===\n\n"
            + "\n".join(resolved_parts)
        )

    return email


def enrich_batch(emails: List[dict], gmail_search_fn=None) -> List[dict]:
    """Enrich a batch of emails with resolved entity context."""
    return [enrich_email_context(e, gmail_search_fn=gmail_search_fn) for e in emails]
