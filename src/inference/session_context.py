"""세션 컨텍스트 관리 모듈.

에이전트 루프가 사용하는 세션 상태를 관리한다:
- 최근 대화 히스토리
- 선택된 근거(검색 결과)
- 이전 초안 버전
- 세션 메타데이터

Issue: #393
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loguru import logger


@dataclass
class ConversationTurn:
    """대화 한 턴(사용자 입력 + 시스템 응답)."""

    role: str  # "user" | "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DraftVersion:
    """초안 버전 기록."""

    version: int
    content: str
    created_at: float = field(default_factory=time.time)
    tool_trace: List[str] = field(default_factory=list)


@dataclass
class SessionContext:
    """세션 기반 에이전트 루프의 컨텍스트.

    하나의 세션(shell) 내에서 누적되는 대화, 근거, 초안을
    관리하고 에이전트 루프에 컨텍스트를 제공한다.

    Parameters
    ----------
    session_id : str
        세션 고유 식별자. 미지정 시 자동 생성.
    max_history : int
        유지할 최대 대화 턴 수. 초과 시 오래된 턴부터 제거.
    """

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    max_history: int = 20
    conversations: List[ConversationTurn] = field(default_factory=list)
    selected_evidences: List[Dict[str, Any]] = field(default_factory=list)
    draft_versions: List[DraftVersion] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def add_turn(self, role: str, content: str, **kwargs: Any) -> None:
        """대화 턴을 추가한다. max_history 초과 시 오래된 턴을 제거."""
        turn = ConversationTurn(role=role, content=content, metadata=kwargs)
        self.conversations.append(turn)
        if len(self.conversations) > self.max_history:
            removed = len(self.conversations) - self.max_history
            self.conversations = self.conversations[removed:]
            logger.debug(f"세션 {self.session_id}: 오래된 대화 {removed}턴 제거")

    def add_evidence(self, evidence: Dict[str, Any]) -> None:
        """선택된 근거(검색 결과)를 추가한다."""
        self.selected_evidences.append(evidence)

    def set_evidences(self, evidences: List[Dict[str, Any]]) -> None:
        """근거 목록을 교체한다."""
        self.selected_evidences = list(evidences)

    def add_draft(self, content: str, tool_trace: Optional[List[str]] = None) -> DraftVersion:
        """새 초안 버전을 기록한다."""
        version = len(self.draft_versions) + 1
        draft = DraftVersion(
            version=version,
            content=content,
            tool_trace=tool_trace or [],
        )
        self.draft_versions.append(draft)
        logger.debug(f"세션 {self.session_id}: 초안 v{version} 저장")
        return draft

    @property
    def latest_draft(self) -> Optional[DraftVersion]:
        """가장 최근 초안을 반환한다."""
        return self.draft_versions[-1] if self.draft_versions else None

    @property
    def recent_history(self) -> List[ConversationTurn]:
        """최근 대화 히스토리를 반환한다."""
        return list(self.conversations)

    def build_context_summary(self) -> str:
        """에이전트 루프에 제공할 컨텍스트 요약 문자열을 생성한다.

        대화 히스토리, 선택된 근거, 이전 초안을 하나의 문자열로 합친다.
        """
        parts: List[str] = []

        # 1. 최근 대화 히스토리
        if self.conversations:
            history_lines = []
            for turn in self.conversations[-5:]:  # 최근 5턴
                role_label = "사용자" if turn.role == "user" else "시스템"
                history_lines.append(f"[{role_label}]: {turn.content}")
            parts.append("### 이전 대화:\n" + "\n".join(history_lines))

        # 2. 선택된 근거
        if self.selected_evidences:
            evidence_lines = []
            for i, ev in enumerate(self.selected_evidences[:5], 1):
                title = ev.get("title", ev.get("category", ""))
                content = ev.get("content", ev.get("complaint", ""))
                if len(content) > 200:
                    content = content[:200] + "..."
                evidence_lines.append(f"{i}. [{title}] {content}")
            parts.append("### 선택된 근거:\n" + "\n".join(evidence_lines))

        # 3. 이전 초안
        if self.draft_versions:
            latest = self.draft_versions[-1]
            draft_preview = latest.content[:300]
            if len(latest.content) > 300:
                draft_preview += "..."
            parts.append(f"### 이전 초안 (v{latest.version}):\n{draft_preview}")

        return "\n\n".join(parts) if parts else ""


class SessionStore:
    """인메모리 세션 저장소.

    세션 ID를 키로 SessionContext를 관리한다.
    프로덕션에서는 Redis 등 외부 저장소로 교체할 수 있다.
    """

    def __init__(self, max_sessions: int = 1000) -> None:
        self._sessions: Dict[str, SessionContext] = {}
        self._max_sessions = max_sessions

    def get_or_create(
        self,
        session_id: Optional[str] = None,
        max_history: int = 20,
    ) -> SessionContext:
        """세션을 조회하거나 새로 생성한다."""
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]

        ctx = SessionContext(
            session_id=session_id or str(uuid.uuid4()),
            max_history=max_history,
        )

        # 최대 세션 수 초과 시 가장 오래된 세션 제거
        if len(self._sessions) >= self._max_sessions:
            oldest_key = min(self._sessions, key=lambda k: self._sessions[k].created_at)
            del self._sessions[oldest_key]
            logger.debug(f"최대 세션 수 초과: 오래된 세션 {oldest_key} 제거")

        self._sessions[ctx.session_id] = ctx
        logger.info(f"새 세션 생성: {ctx.session_id}")
        return ctx

    def get(self, session_id: str) -> Optional[SessionContext]:
        """세션을 조회한다."""
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> bool:
        """세션을 삭제한다."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    @property
    def count(self) -> int:
        """현재 세션 수."""
        return len(self._sessions)
