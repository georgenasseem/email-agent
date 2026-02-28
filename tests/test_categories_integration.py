"""Integration tests for Phase 6: Custom categorizations and user labels.

Tests cover:
- DB schema creation (category_labels, category_rules)
- Default label seeding
- Label CRUD (create, read, update, delete)
- Label merge
- Rule CRUD (create, match, delete)
- Rule matching priority (sender > domain > keyword)
- Category override + learning
- Categorizer integration with user-defined labels and rules
- Proposal generation from email history
"""
import os
import sqlite3
import tempfile
from unittest.mock import patch, MagicMock

import pytest

# ── Redirect DB to a temp file so we don't touch the real DB ────────────────
_tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_tmp.close()

import agent.memory_store as ms
ms.DB_PATH = type(ms.DB_PATH)(_tmp.name)

# Now import the rest (they all use ms.DB_PATH through get_connection)
from agent.memory_store import init_db, get_connection
from agent.email_memory import (
    SYSTEM_CATEGORIES,
    ensure_default_labels,
    get_all_labels,
    get_enabled_labels,
    get_label_by_slug,
    create_label,
    update_label,
    delete_label,
    merge_labels,
    get_all_rules,
    upsert_rule,
    delete_rule,
    match_rules_for_email,
    record_category_override,
    propose_categories_from_history,
    store_raw_email,
    store_processed_email,
    apply_rule_to_existing_emails,
)


@pytest.fixture(autouse=True)
def fresh_db():
    """Wipe and reinitialize the DB before each test."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        tables = [r[0] for r in cur.fetchall()]
        for t in tables:
            cur.execute(f"DROP TABLE IF EXISTS [{t}]")
        conn.commit()
    init_db()
    yield


# ═══════════════════════════════════════════════════════════════════════════
# 1. DB Schema
# ═══════════════════════════════════════════════════════════════════════════

class TestSchema:
    def test_category_labels_table_exists(self):
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='category_labels'")
            assert cur.fetchone() is not None

    def test_category_rules_table_exists(self):
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='category_rules'")
            assert cur.fetchone() is not None

    def test_category_labels_columns(self):
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute("PRAGMA table_info(category_labels)")
            cols = {r[1] for r in cur.fetchall()}
            assert cols >= {"id", "slug", "display_name", "color", "description", "enabled", "position"}

    def test_category_rules_columns(self):
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute("PRAGMA table_info(category_rules)")
            cols = {r[1] for r in cur.fetchall()}
            assert cols >= {"id", "match_type", "match_value", "label_slug", "hits"}


# ═══════════════════════════════════════════════════════════════════════════
# 2. Default Labels
# ═══════════════════════════════════════════════════════════════════════════

class TestDefaultLabels:
    def test_ensure_default_labels_creates_5(self):
        ensure_default_labels()
        labels = get_all_labels()
        slugs = [lb["slug"] for lb in labels]
        for sys_cat in SYSTEM_CATEGORIES:
            assert sys_cat in slugs

    def test_ensure_default_labels_idempotent(self):
        ensure_default_labels()
        ensure_default_labels()
        labels = get_all_labels()
        assert len(labels) == 5

    def test_default_labels_have_colors(self):
        ensure_default_labels()
        for lb in get_all_labels():
            assert lb["color"].startswith("#")

    def test_all_defaults_enabled(self):
        ensure_default_labels()
        for lb in get_all_labels():
            assert lb["enabled"] == 1


# ═══════════════════════════════════════════════════════════════════════════
# 3. Label CRUD
# ═══════════════════════════════════════════════════════════════════════════

class TestLabelCRUD:
    def test_create_label(self):
        ensure_default_labels()
        lb = create_label("Advising", color="#22d3ee", description="Academic advising")
        assert lb["slug"] == "advising"
        assert lb["display_name"] == "Advising"
        assert lb["color"] == "#22d3ee"
        assert lb["enabled"] == 1

    def test_create_label_sets_position(self):
        ensure_default_labels()
        lb = create_label("Advising")
        assert lb["position"] == 5  # after the 5 defaults (0-4)

    def test_create_label_slugifies(self):
        ensure_default_labels()
        lb = create_label("My Custom Label!")
        assert lb["slug"] == "my-custom-label"

    def test_create_label_empty_name_raises(self):
        ensure_default_labels()
        with pytest.raises(ValueError):
            create_label("")

    def test_get_label_by_slug(self):
        ensure_default_labels()
        lb = get_label_by_slug("urgent")
        assert lb is not None
        assert lb["display_name"] == "Urgent"

    def test_get_label_by_slug_missing(self):
        ensure_default_labels()
        assert get_label_by_slug("nonexistent") is None

    def test_update_label_display_name(self):
        ensure_default_labels()
        update_label("urgent", display_name="URGENT!!")
        lb = get_label_by_slug("urgent")
        assert lb["display_name"] == "URGENT!!"

    def test_update_label_color(self):
        ensure_default_labels()
        update_label("informational", color="#000000")
        lb = get_label_by_slug("informational")
        assert lb["color"] == "#000000"

    def test_update_label_enabled(self):
        ensure_default_labels()
        update_label("newsletter", enabled=0)
        lb = get_label_by_slug("newsletter")
        assert lb["enabled"] == 0

    def test_delete_custom_label(self):
        ensure_default_labels()
        create_label("Temp")
        delete_label("temp")
        assert get_label_by_slug("temp") is None

    def test_delete_system_label_allowed(self):
        ensure_default_labels()
        delete_label("urgent")
        assert get_label_by_slug("urgent") is None

    def test_get_enabled_labels_excludes_hidden(self):
        ensure_default_labels()
        update_label("informational", enabled=0)
        enabled = get_enabled_labels()
        slugs = [lb["slug"] for lb in enabled]
        assert "informational" not in slugs
        assert "urgent" in slugs


# ═══════════════════════════════════════════════════════════════════════════
# 4. Label Merge
# ═══════════════════════════════════════════════════════════════════════════

class TestLabelMerge:
    def test_merge_custom_into_system(self):
        ensure_default_labels()
        create_label("Finance")
        upsert_rule("sender", "bank@example.com", "finance")
        changed = merge_labels("finance", "important")
        assert changed == 1
        # finance label should be deleted
        assert get_label_by_slug("finance") is None
        # rule should now point to important
        rules = get_all_rules()
        assert rules[0]["label_slug"] == "important"

    def test_merge_system_into_system_disables_source(self):
        ensure_default_labels()
        upsert_rule("sender", "news@example.com", "informational")
        merge_labels("informational", "newsletter")
        lb = get_label_by_slug("informational")
        assert lb is not None  # not deleted (system)
        assert lb["enabled"] == 0  # but disabled

    def test_merge_same_slug_noop(self):
        ensure_default_labels()
        assert merge_labels("urgent", "urgent") == 0


# ═══════════════════════════════════════════════════════════════════════════
# 5. Rules CRUD
# ═══════════════════════════════════════════════════════════════════════════

class TestRulesCRUD:
    def test_upsert_creates_rule(self):
        ensure_default_labels()
        upsert_rule("sender", "alice@example.com", "important")
        rules = get_all_rules()
        assert len(rules) == 1
        assert rules[0]["match_value"] == "alice@example.com"
        assert rules[0]["label_slug"] == "important"
        assert rules[0]["hits"] == 1

    def test_upsert_increments_hits(self):
        ensure_default_labels()
        upsert_rule("sender", "alice@example.com", "important")
        upsert_rule("sender", "alice@example.com", "important")
        rules = get_all_rules()
        assert rules[0]["hits"] == 2

    def test_upsert_updates_label(self):
        ensure_default_labels()
        upsert_rule("sender", "alice@example.com", "important")
        upsert_rule("sender", "alice@example.com", "urgent")
        rules = get_all_rules()
        assert rules[0]["label_slug"] == "urgent"

    def test_delete_rule(self):
        ensure_default_labels()
        upsert_rule("sender", "alice@example.com", "important")
        rules = get_all_rules()
        delete_rule(rules[0]["id"])
        assert len(get_all_rules()) == 0

    def test_domain_rule(self):
        ensure_default_labels()
        upsert_rule("sender_domain", "example.com", "newsletter")
        rules = get_all_rules()
        assert rules[0]["match_type"] == "sender_domain"

    def test_subject_keyword_rule(self):
        ensure_default_labels()
        upsert_rule("subject_keyword", "invoice", "important")
        rules = get_all_rules()
        assert rules[0]["match_type"] == "subject_keyword"


# ═══════════════════════════════════════════════════════════════════════════
# 6. Rule Matching
# ═══════════════════════════════════════════════════════════════════════════

class TestRuleMatching:
    def _email(self, sender="alice@example.com", subject="Hello"):
        return {"id": "test1", "sender": sender, "subject": subject}

    def test_sender_exact_match(self):
        ensure_default_labels()
        upsert_rule("sender", "alice@example.com", "important")
        assert match_rules_for_email(self._email()) == "important"

    def test_sender_in_angle_brackets(self):
        ensure_default_labels()
        upsert_rule("sender", "alice@example.com", "urgent")
        email = self._email(sender="Alice Smith <alice@example.com>")
        assert match_rules_for_email(email) == "urgent"

    def test_domain_match(self):
        ensure_default_labels()
        upsert_rule("sender_domain", "example.com", "newsletter")
        assert match_rules_for_email(self._email()) == "newsletter"

    def test_subject_keyword_match(self):
        ensure_default_labels()
        upsert_rule("subject_keyword", "invoice", "important")
        email = self._email(subject="Your invoice is ready")
        assert match_rules_for_email(email) == "important"

    def test_keyword_takes_priority_over_sender(self):
        ensure_default_labels()
        upsert_rule("subject_keyword", "hello", "important")
        upsert_rule("sender", "alice@example.com", "urgent")
        assert match_rules_for_email(self._email()) == "important"

    def test_keyword_takes_priority_over_domain(self):
        ensure_default_labels()
        upsert_rule("subject_keyword", "hello", "important")
        upsert_rule("sender_domain", "example.com", "informational")
        assert match_rules_for_email(self._email()) == "important"

    def test_sender_takes_priority_over_domain(self):
        """When no keyword matches, sender > domain."""
        ensure_default_labels()
        upsert_rule("sender", "alice@example.com", "urgent")
        upsert_rule("sender_domain", "example.com", "newsletter")
        assert match_rules_for_email(self._email()) == "urgent"

    def test_no_match_returns_none(self):
        ensure_default_labels()
        assert match_rules_for_email(self._email()) is None

    def test_disabled_label_not_matched(self):
        ensure_default_labels()
        upsert_rule("sender", "alice@example.com", "informational")
        update_label("informational", enabled=0)
        assert match_rules_for_email(self._email()) is None


# ═══════════════════════════════════════════════════════════════════════════
# 7. Category Override + Learning
# ═══════════════════════════════════════════════════════════════════════════

class TestCategoryOverride:
    def test_override_creates_keyword_rules(self):
        ensure_default_labels()
        email = {"id": "msg1", "sender": "Bob <bob@company.com>", "subject": "Meeting invitation tomorrow"}
        # Store as raw + processed so the UPDATE works
        store_raw_email({"id": "msg1", "thread_id": "t1", "subject": "Meeting invitation tomorrow", "sender": "Bob <bob@company.com>", "date": "", "body": "", "snippet": ""})
        store_processed_email({"id": "msg1", "category": "normal"})
        record_category_override(email, "important")
        rules = get_all_rules()
        # Should create keyword rules from subject, not sender rules
        assert len(rules) >= 1
        assert all(r["match_type"] == "subject_keyword" for r in rules)
        keywords = [r["match_value"] for r in rules]
        assert "meeting" in keywords
        assert all(r["label_slug"] == "important" for r in rules)

    def test_override_updates_processed_in_db(self):
        ensure_default_labels()
        store_raw_email({"id": "msg2", "thread_id": "t2", "subject": "X", "sender": "x@y.com", "date": "", "body": "", "snippet": ""})
        store_processed_email({"id": "msg2", "category": "normal"})
        record_category_override({"id": "msg2", "sender": "x@y.com"}, "urgent")
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT category FROM email_processed WHERE gmail_id = 'msg2'")
            assert cur.fetchone()[0] == "urgent"

    def test_override_then_match(self):
        """After override, future emails with similar subject keywords should match."""
        ensure_default_labels()
        store_raw_email({"id": "msg3", "thread_id": "t3", "subject": "Invoice payment due", "sender": "q@r.com", "date": "", "body": "", "snippet": ""})
        store_processed_email({"id": "msg3", "category": "normal"})
        record_category_override({"id": "msg3", "sender": "q@r.com", "subject": "Invoice payment due"}, "newsletter")
        # Now a new email with same keyword should match
        new_email = {"id": "msg4", "sender": "different@other.com", "subject": "New invoice attached"}
        assert match_rules_for_email(new_email) == "newsletter"


# ═══════════════════════════════════════════════════════════════════════════
# 8. Categorizer Integration
# ═══════════════════════════════════════════════════════════════════════════

class TestCategorizerIntegration:
    def test_rule_match_skips_llm(self):
        """When a sender rule matches, the LLM should not be called."""
        ensure_default_labels()
        upsert_rule("sender", "spam@list.com", "newsletter")
        email = {"id": "e1", "sender": "spam@list.com", "subject": "Buy now!", "body": "Click"}

        from agent.categorizer import categorize_email
        # Patch get_llm so we can detect if it's called
        with patch("agent.categorizer.get_llm") as mock_llm:
            result = categorize_email(email)
            assert result["category"] == "newsletter"
            mock_llm.assert_not_called()

    def test_user_label_in_llm_prompt(self):
        """When no rule matches but a user label exists, it should appear in the LLM prompt."""
        ensure_default_labels()
        create_label("Advising", description="Academic advising emails")

        email = {"id": "e2", "sender": "prof@university.edu", "subject": "Advising appointment", "body": "Please come"}

        from agent.categorizer import categorize_email
        mock_llm_instance = MagicMock()
        # Make invoke return "advising" so the LLM "picks" the user label
        mock_llm_instance.__or__ = MagicMock(return_value=MagicMock(invoke=MagicMock(return_value="advising")))

        with patch("agent.categorizer.get_llm", return_value=mock_llm_instance):
            with patch("agent.categorizer.StrOutputParser") as mock_parser:
                chain_mock = MagicMock()
                chain_mock.invoke.return_value = "advising"
                mock_llm_instance.__or__.return_value = chain_mock
                result = categorize_email(email)
                assert result["category"] == "advising"

    def test_falls_back_to_normal_on_bad_llm(self):
        """Unknown LLM output falls back to 'normal'."""
        ensure_default_labels()
        email = {"id": "e3", "sender": "a@b.com", "subject": "Test", "body": "x"}

        from agent.categorizer import categorize_email
        mock_llm_instance = MagicMock()
        chain_mock = MagicMock()
        chain_mock.invoke.return_value = "totally_invalid_garbage"
        mock_llm_instance.__or__.return_value = chain_mock

        with patch("agent.categorizer.get_llm", return_value=mock_llm_instance):
            result = categorize_email(email)
            assert result["category"] == "normal"


# ═══════════════════════════════════════════════════════════════════════════
# 9. Proposal Generation
# ═══════════════════════════════════════════════════════════════════════════

class TestProposals:
    def test_propose_with_enough_history(self):
        ensure_default_labels()
        # Store emails with a recurring keyword
        for i in range(3):
            store_raw_email({
                "id": f"prop_{i}", "thread_id": f"t{i}",
                "subject": f"Weekly report update {i}", "sender": f"user{i}@bigcorp.com",
                "date": "2026-01-01", "body": "...", "snippet": "..."
            })
        proposals = propose_categories_from_history(min_sender_count=2)
        assert len(proposals) >= 1
        assert proposals[0]["match_type"] == "subject_keyword"
        # Should propose based on recurring keywords like 'weekly' or 'report'
        values = [p["match_value"] for p in proposals]
        assert any(v in ("weekly", "report", "update") for v in values)

    def test_propose_skips_existing_rules(self):
        ensure_default_labels()
        upsert_rule("subject_keyword", "report", "important")
        for i in range(3):
            store_raw_email({
                "id": f"prop2_{i}", "thread_id": f"t{i}",
                "subject": f"Monthly report {i}", "sender": f"u{i}@bigcorp.com",
                "date": "2026-01-01", "body": "...", "snippet": "..."
            })
        proposals = propose_categories_from_history(min_sender_count=2)
        values = [p["match_value"] for p in proposals]
        assert "report" not in values

    def test_no_proposals_when_insufficient_history(self):
        ensure_default_labels()
        store_raw_email({
            "id": "lone", "thread_id": "t", "subject": "Solo",
            "sender": "only@rare.com", "date": "2026-01-01", "body": "", "snippet": ""
        })
        proposals = propose_categories_from_history(min_sender_count=5)
        assert len(proposals) == 0


# ═══════════════════════════════════════════════════════════════════════════
# 10. Edge Cases
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_empty_email_categorize(self):
        from agent.categorizer import categorize_email
        assert categorize_email({}) == {}
        assert categorize_email(None) is None

    def test_custom_label_with_special_chars(self):
        ensure_default_labels()
        lb = create_label("  Work / Personal  ")
        assert lb["slug"] == "work-personal"
        assert lb["display_name"] == "Work / Personal"

    def test_match_rules_no_sender(self):
        ensure_default_labels()
        assert match_rules_for_email({"id": "x", "subject": "test"}) is None

    def test_override_no_subject(self):
        """Override with email missing subject should not crash."""
        ensure_default_labels()
        store_raw_email({"id": "nosender", "thread_id": "t", "subject": "", "sender": "", "date": "", "body": "", "snippet": ""})
        store_processed_email({"id": "nosender", "category": "normal"})
        record_category_override({"id": "nosender", "sender": "", "subject": ""}, "informational")
        # No rule should be created (no meaningful keywords)
        assert len(get_all_rules()) == 0

    def test_duplicate_slug_returns_existing(self):
        """Creating a label with an existing slug should return the existing one."""
        ensure_default_labels()
        lb1 = create_label("Finance")
        lb2 = create_label("Finance")
        assert lb1["slug"] == lb2["slug"]
        assert lb1["id"] == lb2["id"]


class TestApplyRuleToExistingEmails:
    """Tests for applying a new rule to existing emails (multi-category, up to 2)."""

    def test_applies_subject_keyword(self):
        ensure_default_labels()
        create_label("Hackathon")
        # Store two emails — one matching, one not
        store_raw_email({"id": "e1", "thread_id": "t1", "subject": "Hackathon Registration",
                         "sender": "a@x.com", "date": "Mon, 1 Jan 2026", "body": "", "snippet": ""})
        store_processed_email({"id": "e1", "category": "normal"})
        store_raw_email({"id": "e2", "thread_id": "t2", "subject": "Meeting Tomorrow",
                         "sender": "b@x.com", "date": "Mon, 1 Jan 2026", "body": "", "snippet": ""})
        store_processed_email({"id": "e2", "category": "important"})
        updated = apply_rule_to_existing_emails("subject_keyword", "hackathon", "hackathon")
        assert updated == 1
        # Verify e1 now has two categories
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT category FROM email_processed WHERE gmail_id = 'e1'")
            cat = cur.fetchone()[0]
        assert "hackathon" in cat.split(",")
        assert "normal" in cat.split(",")

    def test_respects_two_category_limit(self):
        ensure_default_labels()
        create_label("Alpha")
        create_label("Beta")
        store_raw_email({"id": "e3", "thread_id": "t3", "subject": "Alpha Beta Test",
                         "sender": "c@x.com", "date": "Mon, 1 Jan 2026", "body": "", "snippet": ""})
        store_processed_email({"id": "e3", "category": "urgent,alpha"})
        updated = apply_rule_to_existing_emails("subject_keyword", "beta", "beta")
        assert updated == 0  # already has 2 categories

    def test_skips_already_tagged(self):
        ensure_default_labels()
        create_label("Research")
        store_raw_email({"id": "e4", "thread_id": "t4", "subject": "Research Paper",
                         "sender": "d@x.com", "date": "Mon, 1 Jan 2026", "body": "", "snippet": ""})
        store_processed_email({"id": "e4", "category": "research"})
        updated = apply_rule_to_existing_emails("subject_keyword", "research", "research")
        assert updated == 0  # already has this category


# ── Cleanup temp DB ────────────────────────────────────────────────────────

def teardown_module():
    try:
        os.unlink(_tmp.name)
    except Exception:
        pass
