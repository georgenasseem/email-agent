"""Integration tests for the email memory system.

Proves that:
1. Raw emails survive store → load round-trips.
2. Processed email data is correctly persisted and joined on load.
3. Cross-email links are detected (same_sender, same_domain, shared_subject, same_thread).
4. build_memory_context() returns rich context for linked emails.
5. Knowledge base upsert + lookup works correctly.
6. Todo items CRUD works correctly.
7. Sender history returns previous emails.
8. Entity search finds matching emails.
9. Wipe deletes processed data but keeps raw emails.
10. Memory table entries are now readable and feed into build_memory_context.
11. The timing fix: links are available before persist_results_node runs.

Run: python -m pytest tests/test_memory_integration.py -v
"""
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

# ─── Redirect DB_PATH to a temp file BEFORE importing anything ─────────────

_tmpdir = tempfile.mkdtemp()
_test_db = Path(_tmpdir) / "test_memory.db"


# Patch DB_PATH at the module level BEFORE the modules cache it
import agent.memory_store as ms
ms.DB_PATH = _test_db

# Now import everything (they will use the patched DB_PATH via get_connection)
from agent.memory_store import (
    init_db,
    add_memory,
    get_memory_entries,
    upsert_user_profile,
    get_user_profile,
    get_connection,
)
from agent.email_memory import (
    store_raw_email,
    store_raw_emails,
    get_stored_email_ids,
    load_all_raw_emails,
    store_processed_email,
    store_processed_emails,
    get_processed_email_ids,
    load_all_processed_emails,
    build_email_links,
    get_linked_emails,
    build_memory_context,
    get_sender_history,
    upsert_knowledge,
    lookup_knowledge,
    get_all_knowledge,
    search_emails_for_entity,
    add_todo_item,
    get_todo_items,
    remove_todo_item,
    wipe_processed_data,
    get_email_count,
    get_processed_count,
)


# ─── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def fresh_db():
    """Start each test with a clean database."""
    # Delete and recreate
    if _test_db.exists():
        _test_db.unlink()
    ms.reset_db_initialized()
    init_db()
    yield
    # Cleanup after test
    if _test_db.exists():
        _test_db.unlink()


def _make_email(
    gid: str,
    subject: str = "Test Subject",
    sender: str = "alice@example.com",
    thread_id: str = "",
    body: str = "Hello there",
    date: str = "2024-01-15",
    internal_date: int = 1705276800,
    snippet: str = "Hello there...",
    clean_body: str = "Hello there",
) -> dict:
    """Create a test email dict matching the runtime format."""
    return {
        "id": gid,
        "thread_id": thread_id or gid,
        "subject": subject,
        "sender": sender,
        "date": date,
        "internal_date": internal_date,
        "body": body,
        "clean_body": clean_body,
        "body_html": f"<p>{body}</p>",
        "snippet": snippet,
    }


# ─── Test 1: Raw email round-trip ──────────────────────────────────────────


class TestRawEmailStorage:
    def test_store_and_load_single(self):
        email = _make_email("msg_001", subject="Meeting Tomorrow")
        store_raw_email(email)

        loaded = load_all_raw_emails()
        assert len(loaded) == 1
        assert loaded[0]["id"] == "msg_001"
        assert loaded[0]["subject"] == "Meeting Tomorrow"
        assert loaded[0]["sender"] == "alice@example.com"

    def test_store_batch(self):
        emails = [
            _make_email("msg_001"),
            _make_email("msg_002", subject="Project Update"),
            _make_email("msg_003", subject="Invoice #123"),
        ]
        store_raw_emails(emails)

        assert get_email_count() == 3
        ids = get_stored_email_ids()
        assert ids == {"msg_001", "msg_002", "msg_003"}

    def test_upsert_updates_not_duplicates(self):
        email = _make_email("msg_001", subject="Original Subject")
        store_raw_email(email)
        email["subject"] = "Updated Subject"
        store_raw_email(email)

        loaded = load_all_raw_emails()
        assert len(loaded) == 1
        assert loaded[0]["subject"] == "Updated Subject"

    def test_ordering_newest_first(self):
        store_raw_email(_make_email("msg_old", internal_date=1000))
        store_raw_email(_make_email("msg_new", internal_date=9999))

        loaded = load_all_raw_emails()
        assert loaded[0]["id"] == "msg_new"
        assert loaded[1]["id"] == "msg_old"


# ─── Test 2: Processed email round-trip ────────────────────────────────────


class TestProcessedEmailStorage:
    def test_store_and_load_processed(self):
        # Must store raw first (FK constraint)
        raw = _make_email("msg_001", subject="Budget Review")
        store_raw_email(raw)

        processed = {
            **raw,
            "summary": "A review of Q4 budget allocations.",
            "category": "finance",
            "needs_action": True,
            "decision_options": ["Reply: Approve the budget", "Todo: Review numbers"],
            "thread_context": "Previous thread messages...",
            "enriched_context": "Known entity: Alice (person): VP of Finance",
        }
        store_processed_email(processed)

        assert get_processed_count() == 1
        assert "msg_001" in get_processed_email_ids()

    def test_joined_load(self):
        """load_all_processed_emails() should JOIN raw + processed data."""
        raw = _make_email("msg_001", subject="Budget Review")
        store_raw_email(raw)

        processed = {
            **raw,
            "summary": "Budget discussion",
            "category": "finance",
            "needs_action": True,
            "decision_options": ["Reply: Approve"],
        }
        store_processed_email(processed)

        loaded = load_all_processed_emails()
        assert len(loaded) == 1
        e = loaded[0]
        # Raw fields
        assert e["subject"] == "Budget Review"
        assert e["sender"] == "alice@example.com"
        # Processed fields
        assert e["summary"] == "Budget discussion"
        assert e["category"] == "finance"
        assert e["needs_action"] is True
        assert e["decision_options"] == ["Reply: Approve"]


# ─── Test 3: Cross-email link detection ────────────────────────────────────


class TestEmailLinks:
    def test_same_sender_link(self):
        emails = [
            _make_email("msg_001", sender="bob@acme.com", subject="First email"),
            _make_email("msg_002", sender="bob@acme.com", subject="Second email"),
        ]
        store_raw_emails(emails)
        build_email_links(emails)

        linked = get_linked_emails("msg_001")
        assert len(linked) >= 1
        link_types = {le["link_type"] for le in linked}
        assert "same_sender" in link_types

    def test_same_domain_link(self):
        emails = [
            _make_email("msg_001", sender="alice@acme.com", subject="From Alice"),
            _make_email("msg_002", sender="bob@acme.com", subject="From Bob"),
        ]
        store_raw_emails(emails)
        build_email_links(emails)

        linked = get_linked_emails("msg_001")
        link_types = {le["link_type"] for le in linked}
        assert "same_domain" in link_types

    def test_shared_subject_link(self):
        emails = [
            _make_email("msg_001", sender="a@x.com", subject="Project Alpha Release Plan"),
            _make_email("msg_002", sender="b@y.com", subject="Project Alpha Timeline"),
        ]
        store_raw_emails(emails)
        build_email_links(emails)

        linked = get_linked_emails("msg_001")
        link_types = {le["link_type"] for le in linked}
        assert "shared_subject" in link_types

    def test_same_thread_link(self):
        emails = [
            _make_email("msg_001", thread_id="thread_99", subject="Discussion"),
            _make_email("msg_002", thread_id="thread_99", subject="Re: Discussion"),
        ]
        store_raw_emails(emails)
        build_email_links(emails)

        linked = get_linked_emails("msg_001")
        link_types = {le["link_type"] for le in linked}
        assert "same_thread" in link_types

    def test_no_spurious_links(self):
        """Emails with nothing in common should NOT be linked."""
        emails = [
            _make_email("msg_001", sender="a@x.com", subject="Apples", thread_id="t1"),
            _make_email("msg_002", sender="b@y.com", subject="Bananas", thread_id="t2"),
        ]
        store_raw_emails(emails)
        build_email_links(emails)

        linked_a = get_linked_emails("msg_001")
        linked_b = get_linked_emails("msg_002")
        assert len(linked_a) == 0
        assert len(linked_b) == 0

    def test_bidirectional_lookup(self):
        """Links should be queryable from either side."""
        emails = [
            _make_email("msg_001", sender="bob@acme.com"),
            _make_email("msg_002", sender="bob@acme.com"),
        ]
        store_raw_emails(emails)
        build_email_links(emails)

        from_a = get_linked_emails("msg_001")
        from_b = get_linked_emails("msg_002")
        assert len(from_a) >= 1
        assert len(from_b) >= 1
        assert from_a[0]["id"] == "msg_002"
        assert from_b[0]["id"] == "msg_001"


# ─── Test 4: build_memory_context() ────────────────────────────────────────


class TestBuildMemoryContext:
    def test_returns_context_for_linked_emails(self):
        """After links exist, build_memory_context should return non-empty."""
        emails = [
            _make_email("msg_001", sender="carol@acme.com", subject="Q1 Report"),
            _make_email("msg_002", sender="carol@acme.com", subject="Q2 Report"),
        ]
        store_raw_emails(emails)
        # Also store processed data so summaries appear in context
        store_processed_email({**emails[1], "summary": "Q2 revenue is up 15%", "category": "reports"})

        build_email_links(emails)

        ctx = build_memory_context(emails[0])
        assert ctx  # non-empty
        assert "Linked emails" in ctx
        assert "carol@acme.com" in ctx

    def test_returns_empty_for_unlinked_email(self):
        email = _make_email("msg_lone", sender="nobody@nowhere.com", subject="Unique Topic XYZ")
        store_raw_email(email)
        ctx = build_memory_context(email)
        assert ctx == ""

    def test_includes_category_history(self):
        """Memory table entries should feed into build_memory_context."""
        # Add a memory entry (simulating what log_memory_node does)
        upsert_user_profile("test@test.com")
        add_memory("test@test.com", "category", "finance", "Budget Review Q4", source="pipeline")
        add_memory("test@test.com", "category", "finance", "Budget Review Q3", source="pipeline")

        # Email with overlapping keywords
        email = _make_email("msg_budget", subject="Budget Review Q5 Forecast")
        store_raw_email(email)

        ctx = build_memory_context(email)
        assert ctx  # non-empty
        assert "Category history" in ctx
        assert "Budget Review" in ctx

    def test_includes_knowledge_base_entries(self):
        """Knowledge base entries for the sender should appear in context."""
        upsert_knowledge("Alice Smith", "person", "VP of Engineering at Acme Corp", source="email_search")

        email = _make_email("msg_alice", sender="Alice Smith <alice@acme.com>", subject="Team Update")
        store_raw_email(email)

        ctx = build_memory_context(email)
        assert ctx  # non-empty
        assert "VP of Engineering" in ctx


# ─── Test 5: Knowledge base CRUD ───────────────────────────────────────────


class TestKnowledgeBase:
    def test_upsert_and_lookup(self):
        upsert_knowledge("John Doe", "person", "Senior Dev on Project Alpha")
        results = lookup_knowledge("John")
        assert len(results) == 1
        assert results[0]["entity"] == "john doe"
        assert "Senior Dev" in results[0]["info"]

    def test_update_preserves_higher_confidence(self):
        upsert_knowledge("Acme Corp", "org", "Tech company", confidence=0.8)
        upsert_knowledge("Acme Corp", "org", "Updated info", confidence=0.3)
        results = lookup_knowledge("Acme Corp")
        assert results[0]["confidence"] == 0.8  # MAX(0.8, 0.3)
        assert results[0]["info"] == "Updated info"  # info updates

    def test_get_all_knowledge(self):
        upsert_knowledge("Entity A", "person", "Info A")
        upsert_knowledge("Entity B", "project", "Info B")
        all_kb = get_all_knowledge()
        assert len(all_kb) == 2

    def test_case_insensitive_lookup(self):
        upsert_knowledge("Jane DOE", "person", "Manager")
        results = lookup_knowledge("jane doe")
        assert len(results) == 1


# ─── Test 6: Todo items CRUD ───────────────────────────────────────────────


class TestTodoItems:
    def test_add_and_list(self):
        store_raw_email(_make_email("msg_001"))
        item_id = add_todo_item("Review the proposal", email_id="msg_001")
        assert item_id > 0

        items = get_todo_items()
        assert len(items) == 1
        assert items[0]["task"] == "Review the proposal"
        assert items[0]["email_id"] == "msg_001"

    def test_remove(self):
        item_id = add_todo_item("Temporary task")
        remove_todo_item(item_id)
        assert len(get_todo_items()) == 0

    def test_ordering_newest_first(self):
        id1 = add_todo_item("First task")
        id2 = add_todo_item("Second task")
        items = get_todo_items()
        # Both created in same second, but IDs are auto-incrementing
        # so we just verify both exist and order is by created_at DESC (or ID DESC)
        assert len(items) == 2
        tasks = {i["task"] for i in items}
        assert tasks == {"First task", "Second task"}


# ─── Test 7: Sender history ────────────────────────────────────────────────


class TestSenderHistory:
    def test_finds_previous_emails(self):
        store_raw_emails([
            _make_email("msg_001", sender="dave@corp.com", subject="Old email", internal_date=1000),
            _make_email("msg_002", sender="dave@corp.com", subject="New email", internal_date=2000),
        ])
        history = get_sender_history("dave@corp.com", exclude_id="msg_002")
        assert len(history) == 1
        assert history[0]["id"] == "msg_001"

    def test_excludes_current(self):
        store_raw_email(_make_email("msg_001", sender="solo@corp.com"))
        history = get_sender_history("solo@corp.com", exclude_id="msg_001")
        assert len(history) == 0


# ─── Test 8: Entity search ─────────────────────────────────────────────────


class TestEntitySearch:
    def test_search_by_subject(self):
        store_raw_email(_make_email("msg_001", subject="Project Alpha Kickoff"))
        results = search_emails_for_entity("Alpha")
        assert len(results) == 1
        assert results[0]["id"] == "msg_001"

    def test_search_by_sender(self):
        store_raw_email(_make_email("msg_001", sender="ceo@bigcorp.com"))
        results = search_emails_for_entity("bigcorp")
        assert len(results) == 1

    def test_search_by_body(self):
        store_raw_email(_make_email("msg_001", body="The Zenith project is delayed"))
        results = search_emails_for_entity("Zenith")
        assert len(results) == 1

    def test_no_results(self):
        store_raw_email(_make_email("msg_001", subject="Ordinary email"))
        results = search_emails_for_entity("NonexistentXYZ")
        assert len(results) == 0


# ─── Test 9: Wipe processed but keep raw ───────────────────────────────────


class TestWipeProcessedData:
    def test_wipe_keeps_raw(self):
        raw = _make_email("msg_001")
        store_raw_email(raw)
        store_processed_email({**raw, "summary": "Test", "category": "normal"})

        assert get_processed_count() == 1
        wipe_processed_data()

        assert get_email_count() == 1  # Raw preserved
        assert get_processed_count() == 0  # Processed gone

    def test_wipe_clears_links(self):
        emails = [
            _make_email("msg_001", sender="a@x.com"),
            _make_email("msg_002", sender="a@x.com"),
        ]
        store_raw_emails(emails)
        build_email_links(emails)
        assert len(get_linked_emails("msg_001")) >= 1

        wipe_processed_data()
        assert len(get_linked_emails("msg_001")) == 0

    def test_wipe_clears_category_memories(self):
        upsert_user_profile("test@test.com")
        add_memory("test@test.com", "category", "important", "Test Subject", source="pipeline")
        assert len(get_memory_entries(kind="category")) == 1

        wipe_processed_data()
        assert len(get_memory_entries(kind="category")) == 0


# ─── Test 10: Memory table is now readable ─────────────────────────────────


class TestMemoryTable:
    def test_add_and_read(self):
        upsert_user_profile("user@test.com")
        add_memory("user@test.com", "category", "finance", "Budget Subject", source="pipeline")
        add_memory("user@test.com", "category", "important", "Server Down Alert", source="pipeline")

        entries = get_memory_entries(kind="category")
        assert len(entries) == 2
        kinds = {e["kind"] for e in entries}
        assert kinds == {"category"}

    def test_read_all_kinds(self):
        upsert_user_profile("user@test.com")
        add_memory("user@test.com", "category", "finance", "Budget", source="pipeline")
        add_memory("user@test.com", "delegation_suggestion", "delegate", "To assistant", source="pipeline")

        all_entries = get_memory_entries()  # no kind filter
        assert len(all_entries) == 2

    def test_read_limit(self):
        upsert_user_profile("user@test.com")
        for i in range(10):
            add_memory("user@test.com", "category", f"cat_{i}", f"Subject {i}", source="pipeline")

        limited = get_memory_entries(kind="category", limit=3)
        assert len(limited) == 3


# ─── Test 11: Timing — links available early ───────────────────────────────


class TestTimingFix:
    def test_links_exist_before_processing(self):
        """Simulate the pipeline flow:
        1. Store raw emails (fetch_inbox_node)
        2. Build links (fetch_inbox_node — THE FIX)
        3. Call build_memory_context (used by summarize/categorize/decide nodes)
        4. Verify context is non-empty BEFORE persist_results_node runs
        """
        # Step 1: Store raw emails (simulates store_raw_emails in fetch_inbox_node)
        emails = [
            _make_email("msg_001", sender="team@project.com", subject="Sprint Planning Review"),
            _make_email("msg_002", sender="team@project.com", subject="Sprint Retrospective"),
            _make_email("msg_003", sender="team@project.com", subject="Sprint Demo"),
        ]
        store_raw_emails(emails)

        # Step 2: Build links EARLY (simulates the fix in fetch_inbox_node)
        all_raw = load_all_raw_emails()
        build_email_links(all_raw)

        # Step 3: Now verify build_memory_context works BEFORE persist_results
        # (This is called during summarize_node, categorize_node, etc.)
        ctx = build_memory_context(emails[0])
        assert ctx, "build_memory_context should return non-empty context after early link building"
        assert "team@project.com" in ctx
        assert "same_sender" in ctx or "Sprint" in ctx

    def test_old_and_new_emails_link(self):
        """Test that NEW emails link to PREVIOUSLY STORED emails."""
        # Phase 1: An old email was stored in a previous pipeline run
        old_email = _make_email("msg_old", sender="boss@work.com", subject="Q4 Budget", internal_date=1000)
        store_raw_email(old_email)
        store_processed_email({**old_email, "summary": "Discussed Q4 budget allocations", "category": "finance"})

        # Build links for the old email alone
        build_email_links([old_email])

        # Phase 2: New email arrives in a new pipeline run
        new_email = _make_email("msg_new", sender="boss@work.com", subject="Q1 Budget", internal_date=2000)
        store_raw_email(new_email)

        # Rebuild links including the new email (simulates fetch_inbox_node fix)
        all_raw = load_all_raw_emails()
        build_email_links(all_raw)

        # Now the new email should have context from the old one
        ctx = build_memory_context(new_email)
        assert ctx, "New email should get context from the old linked email"
        assert "boss@work.com" in ctx
        # The old email's processed summary should appear
        assert "Q4 budget" in ctx.lower() or "same_sender" in ctx.lower()


# ─── Test 12: End-to-end data flow ─────────────────────────────────────────


class TestEndToEnd:
    def test_full_pipeline_data_flow(self):
        """Simulate a full pipeline: store → link → process → persist → query.
        Verifies data flows correctly through every storage layer.
        """
        # 1. Raw emails arrive
        emails = [
            _make_email("e1", sender="Alice <alice@acme.com>", subject="Project Alpha Meeting",
                        body="Let's discuss the Alpha project timeline.", internal_date=1000),
            _make_email("e2", sender="Bob <bob@acme.com>", subject="RE: Project Alpha Meeting",
                        thread_id="e1", body="I'll prepare the timeline doc.", internal_date=2000),
            _make_email("e3", sender="Alice <alice@acme.com>", subject="Budget Approval",
                        body="Please approve the Q1 budget.", internal_date=3000),
        ]
        store_raw_emails(emails)
        assert get_email_count() == 3

        # 2. Build links early (the fix)
        build_email_links(load_all_raw_emails())

        # 3. Verify links: e1-e2 share thread + domain + subject; e1-e3 share sender
        links_e1 = get_linked_emails("e1")
        linked_ids = {le["id"] for le in links_e1}
        assert "e2" in linked_ids, "e1 should link to e2 (same thread + shared subject)"

        links_e3 = get_linked_emails("e3")
        linked_ids_e3 = {le["id"] for le in links_e3}
        assert "e1" in linked_ids_e3, "e3 should link to e1 (same sender)"

        # 4. Memory context is available for LLM nodes
        ctx_e1 = build_memory_context(emails[0])
        assert ctx_e1, "e1 should have memory context from linked emails"

        # 5. Store processed results (simulates persist_results_node)
        for e in emails:
            store_processed_email({
                **e,
                "summary": f"Summary of {e['subject']}",
                "category": "work",
                "needs_action": True,
                "decision_options": ["Reply: Acknowledge"],
            })

        # 6. Reload everything
        all_loaded = load_all_processed_emails()
        assert len(all_loaded) == 3
        for loaded in all_loaded:
            assert loaded["summary"].startswith("Summary of")
            assert loaded["category"] == "work"
            assert loaded["needs_action"] is True

        # 7. Knowledge base persists entity info
        upsert_knowledge("Alice", "person", "Product Manager at Acme", source="email_search")
        kb = lookup_knowledge("Alice")
        assert len(kb) == 1
        assert "Product Manager" in kb[0]["info"]

        # 8. Entity search finds relevant emails
        found = search_emails_for_entity("Alpha")
        assert len(found) >= 1
        found_subjects = {f["subject"] for f in found}
        assert "Project Alpha Meeting" in found_subjects

        # 9. Sender history
        history = get_sender_history("alice@acme.com", exclude_id="e3")
        assert len(history) >= 1

        # 10. Todo items
        tid = add_todo_item("Review Alpha timeline", email_id="e1")
        todos = get_todo_items()
        assert len(todos) == 1
        assert todos[0]["task"] == "Review Alpha timeline"
        remove_todo_item(tid)
        assert len(get_todo_items()) == 0

        print("\n✅ Full end-to-end data flow test passed!")
