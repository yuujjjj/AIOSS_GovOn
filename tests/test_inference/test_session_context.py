"""SessionContext 및 SessionStore 단위 테스트.

Issue: #393
"""

import pytest

from src.inference.session_context import (
    ConversationTurn,
    DraftVersion,
    SessionContext,
    SessionStore,
)


class TestSessionContext:
    """SessionContext 단위 테스트."""

    def test_create_default(self):
        ctx = SessionContext()
        assert ctx.session_id
        assert ctx.max_history == 20
        assert ctx.conversations == []
        assert ctx.selected_evidences == []
        assert ctx.draft_versions == []

    def test_add_turn(self):
        ctx = SessionContext()
        ctx.add_turn("user", "안녕하세요")
        ctx.add_turn("assistant", "무엇을 도와드릴까요?")
        assert len(ctx.conversations) == 2
        assert ctx.conversations[0].role == "user"
        assert ctx.conversations[1].role == "assistant"

    def test_max_history_trim(self):
        ctx = SessionContext(max_history=3)
        for i in range(5):
            ctx.add_turn("user", f"메시지 {i}")
        assert len(ctx.conversations) == 3
        assert ctx.conversations[0].content == "메시지 2"

    def test_add_evidence(self):
        ctx = SessionContext()
        ctx.add_evidence({"title": "사례1", "content": "내용1"})
        assert len(ctx.selected_evidences) == 1

    def test_set_evidences(self):
        ctx = SessionContext()
        ctx.add_evidence({"title": "old"})
        ctx.set_evidences([{"title": "new1"}, {"title": "new2"}])
        assert len(ctx.selected_evidences) == 2
        assert ctx.selected_evidences[0]["title"] == "new1"

    def test_add_draft(self):
        ctx = SessionContext()
        draft = ctx.add_draft("초안 내용", tool_trace=["classify", "search"])
        assert draft.version == 1
        assert draft.content == "초안 내용"
        assert draft.tool_trace == ["classify", "search"]

    def test_latest_draft(self):
        ctx = SessionContext()
        assert ctx.latest_draft is None
        ctx.add_draft("v1")
        ctx.add_draft("v2")
        assert ctx.latest_draft.version == 2
        assert ctx.latest_draft.content == "v2"

    def test_recent_history(self):
        ctx = SessionContext()
        ctx.add_turn("user", "질문")
        ctx.add_turn("assistant", "답변")
        history = ctx.recent_history
        assert len(history) == 2

    def test_build_context_summary_empty(self):
        ctx = SessionContext()
        assert ctx.build_context_summary() == ""

    def test_build_context_summary_with_data(self):
        ctx = SessionContext()
        ctx.add_turn("user", "민원 내용")
        ctx.add_turn("assistant", "답변 내용")
        ctx.add_evidence({"title": "사례", "content": "설명"})
        ctx.add_draft("초안 텍스트")

        summary = ctx.build_context_summary()
        assert "이전 대화" in summary
        assert "선택된 근거" in summary
        assert "이전 초안" in summary

    def test_build_context_summary_truncates_long_content(self):
        ctx = SessionContext()
        ctx.add_evidence({"title": "사례", "content": "A" * 300})
        summary = ctx.build_context_summary()
        assert "..." in summary


class TestSessionStore:
    """SessionStore 단위 테스트."""

    def test_get_or_create_new(self):
        store = SessionStore()
        ctx = store.get_or_create()
        assert ctx.session_id
        assert store.count == 1

    def test_get_or_create_existing(self):
        store = SessionStore()
        ctx1 = store.get_or_create(session_id="test-session")
        ctx2 = store.get_or_create(session_id="test-session")
        assert ctx1 is ctx2
        assert store.count == 1

    def test_get_existing(self):
        store = SessionStore()
        ctx = store.get_or_create(session_id="test-session")
        found = store.get("test-session")
        assert found is ctx

    def test_get_nonexistent(self):
        store = SessionStore()
        assert store.get("nonexistent") is None

    def test_delete(self):
        store = SessionStore()
        store.get_or_create(session_id="test-session")
        assert store.delete("test-session") is True
        assert store.count == 0
        assert store.delete("test-session") is False

    def test_max_sessions_eviction(self):
        store = SessionStore(max_sessions=2)
        store.get_or_create(session_id="s1")
        store.get_or_create(session_id="s2")
        store.get_or_create(session_id="s3")
        assert store.count == 2
        # 가장 오래된 s1이 제거되었어야 함
        assert store.get("s1") is None
