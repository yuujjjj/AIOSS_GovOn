"""GovOn LangGraph runtime 패키지.

Issue #415: LangGraph runtime 기반 및 planner/executor adapter 구성.

주요 public API:
- `build_govon_graph`: StateGraph 빌더 함수
- `GovOnGraphState`: graph state TypedDict
- `ApprovalStatus`, `TaskType`, `ToolPlan`: state 관련 타입
- `PlannerAdapter`, `RegexPlannerAdapter`, `LLMPlannerAdapter`: planner 추상화
- `ExecutorAdapter`, `RegistryExecutorAdapter`: executor 추상화
"""

from .builder import build_govon_graph
from .executor_adapter import ExecutorAdapter, RegistryExecutorAdapter
from .planner_adapter import LLMPlannerAdapter, PlannerAdapter, RegexPlannerAdapter
from .state import ApprovalStatus, GovOnGraphState, TaskType, ToolPlan

__all__ = [
    "build_govon_graph",
    "GovOnGraphState",
    "ApprovalStatus",
    "TaskType",
    "ToolPlan",
    "PlannerAdapter",
    "RegexPlannerAdapter",
    "LLMPlannerAdapter",
    "ExecutorAdapter",
    "RegistryExecutorAdapter",
]
