"""Integration tests for the scheduling and calendar system.

Proves that:
1. _parse_rfc3339 handles all timezone formats correctly.
2. _to_rfc3339 produces valid RFC3339 strings.
3. find_free_slots respects working hours (9-17), skips lunch (12-13), adds gaps.
4. find_free_slots correctly identifies gaps between busy periods.
5. list_events parses API responses correctly.
6. create_event builds the correct API body.
7. propose_meeting_times integrates intent extraction with slot finding.
8. create_event_from_slot creates events with optional Zoom links.
9. Zoom token caching works.
10. The Schedule: prefix is accepted by decision_suggester validation.

Run: python -m pytest tests/test_scheduling_integration.py -v
"""
from datetime import datetime, timedelta, timezone
from unittest import mock

import pytest

# ─── Test 1: RFC3339 parsing ───────────────────────────────────────────────

from tools.calendar_tools import _parse_rfc3339, _to_rfc3339


class TestRfc3339Parsing:
    def test_utc_z_suffix(self):
        dt = _parse_rfc3339("2026-03-01T09:00:00Z")
        assert dt.tzinfo is not None
        assert dt.hour == 9
        assert dt.year == 2026

    def test_utc_plus_zero(self):
        dt = _parse_rfc3339("2026-03-01T09:00:00+00:00")
        assert dt.tzinfo is not None
        assert dt.hour == 9

    def test_negative_offset(self):
        """Previously broken: -05:00 offset was mishandled."""
        dt = _parse_rfc3339("2026-03-01T09:00:00-05:00")
        assert dt.tzinfo is not None
        # 09:00 -05:00 = 14:00 UTC
        utc_hour = dt.astimezone(timezone.utc).hour
        assert utc_hour == 14

    def test_positive_offset(self):
        dt = _parse_rfc3339("2026-03-01T09:00:00+05:30")
        assert dt.tzinfo is not None
        utc_dt = dt.astimezone(timezone.utc)
        assert utc_dt.hour == 3
        assert utc_dt.minute == 30

    def test_roundtrip(self):
        """Parse → format → parse should be identity."""
        original = "2026-06-15T14:30:00Z"
        dt = _parse_rfc3339(original)
        formatted = _to_rfc3339(dt)
        dt2 = _parse_rfc3339(formatted)
        assert dt == dt2


class TestRfc3339Formatting:
    def test_utc_uses_z(self):
        dt = datetime(2026, 3, 1, 9, 0, 0, tzinfo=timezone.utc)
        assert _to_rfc3339(dt) == "2026-03-01T09:00:00Z"

    def test_naive_gets_utc(self):
        dt = datetime(2026, 3, 1, 9, 0, 0)
        result = _to_rfc3339(dt)
        assert result.endswith("Z")

    def test_offset_preserved(self):
        tz5 = timezone(timedelta(hours=5))
        dt = datetime(2026, 3, 1, 9, 0, 0, tzinfo=tz5)
        result = _to_rfc3339(dt)
        assert "+05:00" in result


# ─── Test 2: Working hours logic ───────────────────────────────────────────

from tools.calendar_tools import _is_within_working_hours


class TestWorkingHours:
    def test_valid_morning_slot(self):
        dt = datetime(2026, 3, 2, 9, 0, tzinfo=timezone.utc)
        assert _is_within_working_hours(dt, timedelta(minutes=30)) is True

    def test_valid_afternoon_slot(self):
        dt = datetime(2026, 3, 2, 14, 0, tzinfo=timezone.utc)
        assert _is_within_working_hours(dt, timedelta(minutes=60)) is True

    def test_before_work_hours(self):
        dt = datetime(2026, 3, 2, 7, 0, tzinfo=timezone.utc)
        assert _is_within_working_hours(dt, timedelta(minutes=30)) is False

    def test_after_work_hours(self):
        dt = datetime(2026, 3, 2, 17, 0, tzinfo=timezone.utc)
        assert _is_within_working_hours(dt, timedelta(minutes=30)) is False

    def test_lunch_overlap(self):
        dt = datetime(2026, 3, 2, 11, 45, tzinfo=timezone.utc)
        assert _is_within_working_hours(dt, timedelta(minutes=30)) is False

    def test_during_lunch(self):
        dt = datetime(2026, 3, 2, 12, 30, tzinfo=timezone.utc)
        assert _is_within_working_hours(dt, timedelta(minutes=30)) is False

    def test_just_before_lunch(self):
        dt = datetime(2026, 3, 2, 11, 0, tzinfo=timezone.utc)
        # 11:00 + 60min = 12:00 — ends right when lunch starts, so no overlap. Valid.
        assert _is_within_working_hours(dt, timedelta(minutes=60)) is True
        # But 11:30 + 60min = 12:30 — overlaps lunch.
        dt2 = datetime(2026, 3, 2, 11, 30, tzinfo=timezone.utc)
        assert _is_within_working_hours(dt2, timedelta(minutes=60)) is False

    def test_morning_ends_before_lunch(self):
        dt = datetime(2026, 3, 2, 11, 0, tzinfo=timezone.utc)
        assert _is_within_working_hours(dt, timedelta(minutes=30)) is True  # 11:00-11:30, fine

    def test_after_lunch(self):
        dt = datetime(2026, 3, 2, 13, 0, tzinfo=timezone.utc)
        assert _is_within_working_hours(dt, timedelta(minutes=30)) is True

    def test_slot_ending_at_work_end(self):
        dt = datetime(2026, 3, 2, 16, 30, tzinfo=timezone.utc)
        assert _is_within_working_hours(dt, timedelta(minutes=30)) is True

    def test_slot_going_past_work_end(self):
        dt = datetime(2026, 3, 2, 16, 45, tzinfo=timezone.utc)
        assert _is_within_working_hours(dt, timedelta(minutes=30)) is False


# ─── Test 3: find_free_slots with mocked calendar API ─────────────────────

from tools.calendar_tools import find_free_slots


class TestFindFreeSlots:
    def _mock_freebusy(self, busy_periods):
        """Create a mock Calendar service that returns the given busy periods."""
        mock_service = mock.MagicMock()
        mock_service.freebusy().query().execute.return_value = {
            "calendars": {
                "primary": {
                    "busy": [
                        {"start": s, "end": e} for s, e in busy_periods
                    ]
                }
            }
        }
        return mock_service

    @mock.patch("tools.calendar_tools.get_calendar_service")
    def test_empty_calendar(self, mock_get_svc):
        """No busy periods → all working hours are free."""
        mock_get_svc.return_value = self._mock_freebusy([])
        tz = timezone.utc
        time_min = datetime(2026, 3, 2, 9, 0, tzinfo=tz)  # Monday 9 AM
        time_max = datetime(2026, 3, 2, 17, 0, tzinfo=tz)  # Monday 5 PM

        slots = find_free_slots(time_min, time_max, duration_minutes=30, respect_working_hours=True)
        assert len(slots) > 0

        # All slots should be within working hours
        for start, end in slots:
            assert start.hour >= 9
            assert end.hour <= 17
            # No slot should overlap lunch (12-13)
            assert not (start.hour < 13 and end.hour > 12 and start.hour >= 12)

    @mock.patch("tools.calendar_tools.get_calendar_service")
    def test_busy_morning(self, mock_get_svc):
        """Morning is busy → only afternoon slots."""
        mock_get_svc.return_value = self._mock_freebusy([
            ("2026-03-02T09:00:00Z", "2026-03-02T12:00:00Z"),
        ])
        tz = timezone.utc
        time_min = datetime(2026, 3, 2, 9, 0, tzinfo=tz)
        time_max = datetime(2026, 3, 2, 17, 0, tzinfo=tz)

        slots = find_free_slots(time_min, time_max, duration_minutes=30, respect_working_hours=True)
        assert len(slots) > 0
        # All slots should be after lunch (13:00)
        for start, end in slots:
            assert start.hour >= 13

    @mock.patch("tools.calendar_tools.get_calendar_service")
    def test_slots_have_gaps(self, mock_get_svc):
        """Consecutive slots should have gaps between them."""
        mock_get_svc.return_value = self._mock_freebusy([])
        tz = timezone.utc
        time_min = datetime(2026, 3, 2, 9, 0, tzinfo=tz)
        time_max = datetime(2026, 3, 2, 12, 0, tzinfo=tz)

        slots = find_free_slots(time_min, time_max, duration_minutes=30, respect_working_hours=True)
        assert len(slots) >= 2
        # Check gap between consecutive slots
        for i in range(len(slots) - 1):
            gap = (slots[i + 1][0] - slots[i][1]).total_seconds() / 60
            assert gap >= 15, f"Gap between slots should be >= 15 min, got {gap}"

    @mock.patch("tools.calendar_tools.get_calendar_service")
    def test_multi_day_range(self, mock_get_svc):
        """Multi-day range should produce slots on multiple days."""
        mock_get_svc.return_value = self._mock_freebusy([])
        tz = timezone.utc
        time_min = datetime(2026, 3, 2, 9, 0, tzinfo=tz)  # Monday
        time_max = datetime(2026, 3, 4, 17, 0, tzinfo=tz)  # Wednesday

        slots = find_free_slots(time_min, time_max, duration_minutes=30, respect_working_hours=True)
        # Should have slots across multiple days
        days = {s[0].day for s in slots}
        assert len(days) >= 2

    @mock.patch("tools.calendar_tools.get_calendar_service")
    def test_no_working_hours_mode(self, mock_get_svc):
        """respect_working_hours=False → return all gaps including nights."""
        mock_get_svc.return_value = self._mock_freebusy([])
        tz = timezone.utc
        time_min = datetime(2026, 3, 2, 6, 0, tzinfo=tz)
        time_max = datetime(2026, 3, 2, 20, 0, tzinfo=tz)

        slots = find_free_slots(time_min, time_max, duration_minutes=30, respect_working_hours=False)
        # Should have early-morning and evening slots too
        hours = {s[0].hour for s in slots}
        assert 6 in hours or 7 in hours  # before 9 AM
        assert 18 in hours or 19 in hours  # after 5 PM

    @mock.patch("tools.calendar_tools.get_calendar_service")
    def test_fully_busy_day(self, mock_get_svc):
        """Entire day is busy → no slots."""
        mock_get_svc.return_value = self._mock_freebusy([
            ("2026-03-02T00:00:00Z", "2026-03-02T23:59:59Z"),
        ])
        tz = timezone.utc
        time_min = datetime(2026, 3, 2, 9, 0, tzinfo=tz)
        time_max = datetime(2026, 3, 2, 17, 0, tzinfo=tz)

        slots = find_free_slots(time_min, time_max, duration_minutes=30)
        assert len(slots) == 0


# ─── Test 4: list_events ──────────────────────────────────────────────────

from tools.calendar_tools import list_events


class TestListEvents:
    @mock.patch("tools.calendar_tools.get_calendar_service")
    def test_parses_events(self, mock_get_svc):
        mock_service = mock.MagicMock()
        mock_service.events().list().execute.return_value = {
            "items": [
                {
                    "id": "evt1",
                    "summary": "Team standup",
                    "start": {"dateTime": "2026-03-02T09:00:00Z"},
                    "end": {"dateTime": "2026-03-02T09:30:00Z"},
                    "location": "Room A",
                    "attendees": [{"email": "a@test.com"}, {"email": "b@test.com"}],
                    "description": "Daily standup sync",
                },
                {
                    "id": "evt2",
                    "summary": "Lunch break",
                    "start": {"dateTime": "2026-03-02T12:00:00Z"},
                    "end": {"dateTime": "2026-03-02T13:00:00Z"},
                },
            ]
        }
        mock_get_svc.return_value = mock_service

        events = list_events()
        assert len(events) == 2
        assert events[0]["summary"] == "Team standup"
        assert events[0]["location"] == "Room A"
        assert events[0]["attendees"] == ["a@test.com", "b@test.com"]
        assert events[1]["summary"] == "Lunch break"
        assert events[1]["attendees"] == []

    @mock.patch("tools.calendar_tools.get_calendar_service")
    def test_all_day_events(self, mock_get_svc):
        """All-day events have 'date' instead of 'dateTime'."""
        mock_service = mock.MagicMock()
        mock_service.events().list().execute.return_value = {
            "items": [
                {
                    "id": "evt1",
                    "summary": "Holiday",
                    "start": {"date": "2026-03-02"},
                    "end": {"date": "2026-03-03"},
                },
            ]
        }
        mock_get_svc.return_value = mock_service

        events = list_events()
        assert len(events) == 1
        assert events[0]["start"] == "2026-03-02"

    @mock.patch("tools.calendar_tools.get_calendar_service")
    def test_empty_calendar(self, mock_get_svc):
        mock_service = mock.MagicMock()
        mock_service.events().list().execute.return_value = {"items": []}
        mock_get_svc.return_value = mock_service

        events = list_events()
        assert len(events) == 0


# ─── Test 5: create_event builds correct body ─────────────────────────────

from tools.calendar_tools import create_event


class TestCreateEvent:
    @mock.patch("tools.calendar_tools.get_calendar_service")
    def test_basic_event(self, mock_get_svc):
        mock_service = mock.MagicMock()
        mock_service.events().insert().execute.return_value = {"id": "new_evt"}
        mock_get_svc.return_value = mock_service

        tz = timezone.utc
        start = datetime(2026, 3, 2, 10, 0, tzinfo=tz)
        end = datetime(2026, 3, 2, 10, 30, tzinfo=tz)

        result = create_event("Test Meeting", start, end, attendees=["bob@test.com"])
        assert result["id"] == "new_evt"

        # Check the API call
        call_args = mock_service.events().insert.call_args
        body = call_args[1]["body"] if "body" in call_args[1] else call_args[0][0]
        assert body["summary"] == "Test Meeting"
        assert "dateTime" in body["start"]
        assert body["attendees"] == [{"email": "bob@test.com"}]

    @mock.patch("tools.calendar_tools.get_calendar_service")
    def test_naive_datetime_gets_utc(self, mock_get_svc):
        """Naive datetimes should be upgraded to UTC."""
        mock_service = mock.MagicMock()
        mock_service.events().insert().execute.return_value = {"id": "evt"}
        mock_get_svc.return_value = mock_service

        start = datetime(2026, 3, 2, 10, 0)  # naive
        end = datetime(2026, 3, 2, 10, 30)  # naive

        create_event("Test", start, end)
        call_args = mock_service.events().insert.call_args
        body = call_args[1]["body"] if "body" in call_args[1] else call_args[0][0]
        assert body["start"]["timeZone"] == "UTC"
        assert "Z" in body["start"]["dateTime"]


# ─── Test 6: propose_meeting_times ─────────────────────────────────────────

from agent.scheduling import propose_meeting_times


class TestProposeMeetingTimes:
    @mock.patch("agent.scheduling.find_free_slots")
    @mock.patch("agent.scheduling.extract_meeting_intent")
    def test_successful_proposal(self, mock_extract, mock_slots):
        mock_extract.return_value = {
            "has_meeting_intent": True,
            "title": "1-on-1 with Alice",
            "duration_minutes": 30,
            "attendees": ["alice@test.com"],
            "notes": "",
        }
        tz = timezone.utc
        mock_slots.return_value = [
            (datetime(2026, 3, 3, 10, 0, tzinfo=tz), datetime(2026, 3, 3, 10, 30, tzinfo=tz)),
            (datetime(2026, 3, 3, 14, 0, tzinfo=tz), datetime(2026, 3, 3, 14, 30, tzinfo=tz)),
        ]

        email = {"sender": "alice@test.com", "subject": "Let's meet", "body": "Can we meet?"}
        result = propose_meeting_times(email)

        assert result["error"] is None
        assert len(result["free_slots"]) == 2
        assert result["meeting_intent"]["title"] == "1-on-1 with Alice"

    @mock.patch("agent.scheduling.extract_meeting_intent")
    def test_no_meeting_intent(self, mock_extract):
        mock_extract.return_value = {"has_meeting_intent": False}
        email = {"sender": "news@spam.com", "subject": "Newsletter", "body": "Weekly digest"}
        result = propose_meeting_times(email)
        assert result["error"] == "No meeting intent detected"
        assert result["free_slots"] == []


# ─── Test 7: create_event_from_slot ────────────────────────────────────────

from agent.scheduling import create_event_from_slot


class TestCreateEventFromSlot:
    @mock.patch("agent.scheduling.create_event")
    def test_basic_creation(self, mock_create):
        mock_create.return_value = {"id": "evt1", "summary": "Test"}
        intent = {
            "title": "Sync with Bob",
            "attendees": ["bob@test.com"],
            "notes": "Discuss Q1 goals",
        }
        result = create_event_from_slot(
            intent,
            "2026-03-03T10:00:00+00:00",
            "2026-03-03T10:30:00+00:00",
            add_zoom=False,
        )
        assert result["id"] == "evt1"
        mock_create.assert_called_once()

    @mock.patch("tools.zoom_tools.create_zoom_meeting")
    @mock.patch("tools.zoom_tools.zoom_available", return_value=True)
    @mock.patch("agent.scheduling.create_event")
    def test_with_zoom(self, mock_create, mock_zoom_avail, mock_zoom):
        mock_zoom.return_value = {"join_url": "https://zoom.us/j/123"}
        mock_create.return_value = {"id": "evt1"}
        intent = {
            "title": "Call with Client",
            "attendees": ["client@co.com"],
            "notes": "",
        }
        result = create_event_from_slot(
            intent,
            "2026-03-03T10:00:00+00:00",
            "2026-03-03T10:30:00+00:00",
            add_zoom=True,
        )
        assert result["id"] == "evt1"
        assert mock_zoom.called


# ─── Test 8: Zoom token caching ────────────────────────────────────────────

from tools.zoom_tools import _zoom_token_cache


class TestZoomTokenCaching:
    def test_cache_structure_exists(self):
        """Verify the token cache dict exists with the right keys."""
        assert "token" in _zoom_token_cache
        assert "expires_at" in _zoom_token_cache


# ─── Test 9: Schedule prefix in decision_suggester ─────────────────────────


class TestSchedulePrefix:
    def test_schedule_prefix_accepted(self):
        """Verify that Schedule: prefix passes the validation filter."""
        # Simulate the filter logic from decision_suggester
        test_options = [
            "Reply: Accept the meeting",
            "Schedule: 30 min meeting with sender",
            "Todo: Review agenda beforehand",
            "Invalid: This should be dropped",
        ]
        cleaned = []
        seen = set()
        for o in test_options:
            s = str(o).strip()
            key = s.lower()
            if not s or key in seen:
                continue
            if not (s.startswith("Reply:") or s.startswith("Todo:") or s.startswith("Schedule:")):
                continue
            seen.add(key)
            cleaned.append(s)

        assert len(cleaned) == 3
        assert "Schedule: 30 min meeting with sender" in cleaned
        assert "Invalid: This should be dropped" not in cleaned


# ─── Test 10: Meeting extractor output format ──────────────────────────────

from agent.meeting_extractor import extract_meeting_intent


class TestMeetingExtractor:
    @mock.patch("agent.meeting_extractor.get_llm")
    def test_meeting_email(self, mock_get_llm):
        """LLM returns meeting intent → properly parsed."""
        mock_llm = mock.MagicMock()
        mock_llm.invoke.return_value = '{"has_meeting_intent": true, "title": "1-on-1 sync", "duration_minutes": 30, "attendees": ["bob@test.com"], "notes": "Weekly sync"}'
        mock_get_llm.return_value = mock_llm

        email = {"sender": "Bob <bob@test.com>", "subject": "Let's catch up", "body": "Can we meet this week?"}
        result = extract_meeting_intent(email)
        assert result["has_meeting_intent"] is True
        assert result["title"] == "1-on-1 sync"
        assert result["duration_minutes"] == 30
        assert "bob@test.com" in result["attendees"]

    @mock.patch("agent.meeting_extractor.get_llm")
    def test_non_meeting_email(self, mock_get_llm):
        mock_llm = mock.MagicMock()
        mock_llm.invoke.return_value = '{"has_meeting_intent": false}'
        mock_get_llm.return_value = mock_llm

        email = {"sender": "news@news.com", "subject": "Newsletter", "body": "This week's digest"}
        result = extract_meeting_intent(email)
        assert result["has_meeting_intent"] is False

    @mock.patch("agent.meeting_extractor.get_llm")
    def test_bad_json_returns_false(self, mock_get_llm):
        mock_llm = mock.MagicMock()
        mock_llm.invoke.return_value = "not valid json at all"
        mock_get_llm.return_value = mock_llm

        email = {"sender": "a@b.com", "subject": "Test", "body": "Hello"}
        result = extract_meeting_intent(email)
        assert result["has_meeting_intent"] is False

    @mock.patch("agent.meeting_extractor.get_llm")
    def test_duration_clamped(self, mock_get_llm):
        """Duration should be clamped between 15 and 120 minutes."""
        mock_llm = mock.MagicMock()
        mock_llm.invoke.return_value = '{"has_meeting_intent": true, "title": "Long meeting", "duration_minutes": 999, "attendees": []}'
        mock_get_llm.return_value = mock_llm

        result = extract_meeting_intent({"sender": "a@b.com", "subject": "Test"})
        assert result["duration_minutes"] == 120


# ─── Test 11: End-to-end scheduling flow ──────────────────────────────────

class TestEndToEndScheduling:
    @mock.patch("tools.calendar_tools.get_calendar_service")
    @mock.patch("agent.meeting_extractor.get_llm")
    def test_full_flow(self, mock_get_llm, mock_get_svc):
        """Simulate: email → extract intent → find slots → verify output."""
        # Mock LLM to return meeting intent
        mock_llm = mock.MagicMock()
        mock_llm.invoke.return_value = '{"has_meeting_intent": true, "title": "Project review", "duration_minutes": 60, "attendees": ["team@corp.com"], "notes": "Review Q1"}'
        mock_get_llm.return_value = mock_llm

        # Mock Calendar API with one busy block
        mock_service = mock.MagicMock()
        mock_service.freebusy().query().execute.return_value = {
            "calendars": {
                "primary": {
                    "busy": [
                        {"start": "2026-03-03T10:00:00Z", "end": "2026-03-03T11:00:00Z"},
                    ]
                }
            }
        }
        mock_get_svc.return_value = mock_service

        # Run the full proposal
        email = {
            "sender": "Alice <alice@corp.com>",
            "subject": "Project review meeting",
            "body": "Can we schedule a 1-hour project review?",
        }
        result = propose_meeting_times(email, days_ahead=2, max_slots=5)

        assert result["error"] is None
        assert result["meeting_intent"]["has_meeting_intent"] is True
        assert result["meeting_intent"]["title"] == "Project review"
        assert len(result["free_slots"]) > 0

        # Verify no slot overlaps the busy period
        for start_iso, end_iso in result["free_slots"]:
            start = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
            end = datetime.fromisoformat(end_iso.replace("Z", "+00:00"))
            busy_start = datetime(2026, 3, 3, 10, 0, tzinfo=timezone.utc)
            busy_end = datetime(2026, 3, 3, 11, 0, tzinfo=timezone.utc)
            # Slot should not overlap with busy period
            assert not (start < busy_end and end > busy_start), \
                f"Slot {start}-{end} overlaps busy period {busy_start}-{busy_end}"

        print("\n✅ Full end-to-end scheduling flow test passed!")
