"""SQLite-backed feature flag A/B experiment tracking.

The first GovOn experiment measures the effect of the existing RAG pipeline
flag while keeping the model fixed. Participant IDs must be pseudonymous.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .feature_flags import FeatureFlags

CONTROL_VARIANT = "control_rag_off"
TREATMENT_VARIANT = "treatment_rag_on"
EXPERIMENT_NAME = "rag_pipeline_v1"
MINIMUM_OPERATION_DAYS = 14
MINIMUM_FEEDBACK_PARTICIPANTS = 10
MINIMUM_PERSONAS = 10


def _default_experiment_db_path() -> str:
    configured_home = os.getenv("GOVON_HOME")
    base_dir = (
        Path(configured_home)
        if configured_home
        else Path(__file__).resolve().parent.parent.parent / ".cache"
    )
    base_dir.mkdir(parents=True, exist_ok=True)
    return str(base_dir / "experiments.sqlite3")


@dataclass(frozen=True)
class ExperimentAssignment:
    participant_id: str
    variant: str
    persona_id: Optional[str]
    assigned_at: float

    @property
    def use_rag_pipeline(self) -> bool:
        return self.variant == TREATMENT_VARIANT

    def apply(self, flags: FeatureFlags) -> FeatureFlags:
        return FeatureFlags(
            use_rag_pipeline=self.use_rag_pipeline,
            model_version=flags.model_version,
        )


class ExperimentStore:
    """Persist deterministic assignments, exposures, and user feedback."""

    def __init__(self, db_path: Optional[str] = None, seed: Optional[str] = None) -> None:
        self._db_path = db_path or os.getenv("GOVON_EXPERIMENT_DB") or _default_experiment_db_path()
        self._seed = seed or os.getenv("GOVON_EXPERIMENT_SEED", EXPERIMENT_NAME)
        self._init_db()

    @property
    def db_path(self) -> str:
        return self._db_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=10000")
        return conn

    def _init_db(self) -> None:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn, conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS experiment_assignments (
                    participant_id TEXT PRIMARY KEY,
                    persona_id TEXT,
                    variant TEXT NOT NULL,
                    assigned_at REAL NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS experiment_exposures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL UNIQUE,
                    participant_id TEXT NOT NULL,
                    persona_id TEXT,
                    scenario_id TEXT,
                    variant TEXT NOT NULL,
                    assigned_variant TEXT NOT NULL,
                    use_rag_pipeline INTEGER NOT NULL,
                    model_version TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    latency_ms REAL NOT NULL,
                    success INTEGER NOT NULL,
                    error TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    FOREIGN KEY(participant_id)
                        REFERENCES experiment_assignments(participant_id)
                );
                CREATE INDEX IF NOT EXISTS idx_experiment_exposures_created_at
                ON experiment_exposures(created_at);
                CREATE INDEX IF NOT EXISTS idx_experiment_exposures_variant
                ON experiment_exposures(variant);

                CREATE TABLE IF NOT EXISTS experiment_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL UNIQUE,
                    participant_id TEXT NOT NULL,
                    persona_id TEXT,
                    scenario_id TEXT,
                    rating INTEGER NOT NULL CHECK(rating >= 1 AND rating <= 5),
                    task_success INTEGER NOT NULL,
                    comment TEXT NOT NULL DEFAULT '',
                    final_content TEXT,
                    evaluator_model TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    FOREIGN KEY(request_id)
                        REFERENCES experiment_exposures(request_id),
                    FOREIGN KEY(participant_id)
                        REFERENCES experiment_assignments(participant_id)
                );
                CREATE INDEX IF NOT EXISTS idx_experiment_feedback_created_at
                ON experiment_feedback(created_at);
                """)

    def _variant_for(self, participant_id: str) -> str:
        digest = hashlib.sha256(f"{self._seed}:{participant_id}".encode("utf-8")).digest()
        return TREATMENT_VARIANT if digest[0] % 2 else CONTROL_VARIANT

    def _balanced_variant_for(self, conn: sqlite3.Connection, participant_id: str) -> str:
        counts = {row["variant"]: int(row["assignment_count"]) for row in conn.execute("""
                SELECT variant, COUNT(*) AS assignment_count
                FROM experiment_assignments
                GROUP BY variant
                """).fetchall()}
        control_count = counts.get(CONTROL_VARIANT, 0)
        treatment_count = counts.get(TREATMENT_VARIANT, 0)
        if control_count < treatment_count:
            return CONTROL_VARIANT
        if treatment_count < control_count:
            return TREATMENT_VARIANT
        return self._variant_for(participant_id)

    def get_or_assign(
        self,
        participant_id: str,
        persona_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        assigned_at: Optional[float] = None,
    ) -> ExperimentAssignment:
        """Return a stable assignment for a pseudonymous participant."""
        participant_id = participant_id.strip()
        if not participant_id:
            raise ValueError("participant_id must not be blank")

        with closing(self._connect()) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT participant_id, persona_id, variant, assigned_at
                FROM experiment_assignments
                WHERE participant_id=?
                """,
                (participant_id,),
            ).fetchone()
            if row:
                if persona_id and not row["persona_id"]:
                    conn.execute(
                        """
                        UPDATE experiment_assignments
                        SET persona_id=?
                        WHERE participant_id=?
                        """,
                        (persona_id, participant_id),
                    )
                return ExperimentAssignment(
                    participant_id=row["participant_id"],
                    persona_id=row["persona_id"] or persona_id,
                    variant=row["variant"],
                    assigned_at=row["assigned_at"],
                )

            timestamp = assigned_at if assigned_at is not None else time.time()
            variant = self._balanced_variant_for(conn, participant_id)
            conn.execute(
                """
                INSERT INTO experiment_assignments(
                    participant_id, persona_id, variant, assigned_at, metadata_json
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    participant_id,
                    persona_id,
                    variant,
                    timestamp,
                    json.dumps(metadata or {}, ensure_ascii=False),
                ),
            )
            return ExperimentAssignment(
                participant_id=participant_id,
                persona_id=persona_id,
                variant=variant,
                assigned_at=timestamp,
            )

    def record_exposure(
        self,
        *,
        request_id: str,
        participant_id: str,
        endpoint: str,
        flags: FeatureFlags,
        latency_ms: float,
        success: bool,
        persona_id: Optional[str] = None,
        scenario_id: Optional[str] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        created_at: Optional[float] = None,
    ) -> int:
        """Record one response exposure and return its database ID."""
        assignment = self.get_or_assign(participant_id, persona_id=persona_id)
        variant = TREATMENT_VARIANT if flags.use_rag_pipeline else CONTROL_VARIANT
        timestamp = created_at if created_at is not None else time.time()
        payload = {
            "assigned_variant": assignment.variant,
            **(metadata or {}),
        }
        with closing(self._connect()) as conn, conn:
            cursor = conn.execute(
                """
                INSERT INTO experiment_exposures(
                    request_id,
                    participant_id,
                    persona_id,
                    scenario_id,
                    variant,
                    assigned_variant,
                    use_rag_pipeline,
                    model_version,
                    endpoint,
                    latency_ms,
                    success,
                    error,
                    metadata_json,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    participant_id,
                    persona_id,
                    scenario_id,
                    variant,
                    assignment.variant,
                    1 if flags.use_rag_pipeline else 0,
                    flags.model_version,
                    endpoint,
                    latency_ms,
                    1 if success else 0,
                    error,
                    json.dumps(payload, ensure_ascii=False),
                    timestamp,
                ),
            )
            return int(cursor.lastrowid)

    def record_feedback(
        self,
        *,
        request_id: str,
        participant_id: str,
        rating: int,
        task_success: bool,
        persona_id: Optional[str] = None,
        scenario_id: Optional[str] = None,
        comment: str = "",
        final_content: Optional[str] = None,
        evaluator_model: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        created_at: Optional[float] = None,
    ) -> int:
        """Record one feedback row for an existing exposure."""
        if not 1 <= rating <= 5:
            raise ValueError("rating must be between 1 and 5")
        timestamp = created_at if created_at is not None else time.time()
        with closing(self._connect()) as conn, conn:
            exposure = conn.execute(
                """
                SELECT participant_id, persona_id, scenario_id
                FROM experiment_exposures
                WHERE request_id=?
                """,
                (request_id,),
            ).fetchone()
            if exposure is None:
                raise ValueError(f"unknown request_id: {request_id}")
            if exposure["participant_id"] != participant_id:
                raise ValueError("participant_id does not match the recorded exposure")

            cursor = conn.execute(
                """
                INSERT INTO experiment_feedback(
                    request_id,
                    participant_id,
                    persona_id,
                    scenario_id,
                    rating,
                    task_success,
                    comment,
                    final_content,
                    evaluator_model,
                    metadata_json,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(request_id) DO UPDATE SET
                    rating=excluded.rating,
                    task_success=excluded.task_success,
                    comment=excluded.comment,
                    final_content=excluded.final_content,
                    evaluator_model=excluded.evaluator_model,
                    metadata_json=excluded.metadata_json,
                    created_at=excluded.created_at
                """,
                (
                    request_id,
                    participant_id,
                    persona_id or exposure["persona_id"],
                    scenario_id or exposure["scenario_id"],
                    rating,
                    1 if task_success else 0,
                    comment,
                    final_content,
                    evaluator_model,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    timestamp,
                ),
            )
            if cursor.lastrowid:
                return int(cursor.lastrowid)
            row = conn.execute(
                "SELECT id FROM experiment_feedback WHERE request_id=?",
                (request_id,),
            ).fetchone()
            return int(row["id"])

    def summarize(self, days: int = MINIMUM_OPERATION_DAYS, now: Optional[float] = None) -> dict:
        """Aggregate variant metrics for the requested recent period."""
        if days <= 0:
            raise ValueError("days must be positive")
        current_time = now if now is not None else time.time()
        cutoff = current_time - (days * 86400)
        with closing(self._connect()) as conn:
            exposure_rows = conn.execute(
                """
                SELECT
                    variant,
                    COUNT(*) AS request_count,
                    COUNT(DISTINCT participant_id) AS participant_count,
                    SUM(success) AS success_count,
                    AVG(latency_ms) AS avg_latency_ms,
                    MIN(created_at) AS first_exposure_at,
                    MAX(created_at) AS last_exposure_at
                FROM experiment_exposures
                WHERE created_at >= ? AND created_at <= ?
                GROUP BY variant
                """,
                (cutoff, current_time),
            ).fetchall()
            feedback_rows = conn.execute(
                """
                SELECT
                    e.variant,
                    COUNT(*) AS feedback_count,
                    COUNT(DISTINCT f.participant_id) AS feedback_participant_count,
                    COUNT(DISTINCT f.persona_id) AS persona_count,
                    AVG(f.rating) AS avg_rating,
                    AVG(f.task_success) AS task_success_rate
                FROM experiment_feedback AS f
                JOIN experiment_exposures AS e ON e.request_id = f.request_id
                WHERE f.created_at >= ? AND f.created_at <= ?
                GROUP BY e.variant
                """,
                (cutoff, current_time),
            ).fetchall()
            lifetime_row = conn.execute("""
                SELECT
                    MIN(created_at) AS first_exposure_at,
                    MAX(created_at) AS last_exposure_at
                FROM experiment_exposures
                """).fetchone()
            overall_feedback_row = conn.execute("""
                SELECT
                    COUNT(DISTINCT participant_id) AS feedback_participant_count,
                    COUNT(DISTINCT persona_id) AS persona_count
                FROM experiment_feedback
                """).fetchone()

        metrics = {
            CONTROL_VARIANT: self._empty_variant_metrics(),
            TREATMENT_VARIANT: self._empty_variant_metrics(),
        }
        for row in exposure_rows:
            request_count = int(row["request_count"])
            success_count = int(row["success_count"] or 0)
            metrics[row["variant"]].update(
                {
                    "request_count": request_count,
                    "participant_count": int(row["participant_count"]),
                    "success_count": success_count,
                    "error_count": request_count - success_count,
                    "request_success_rate": success_count / request_count if request_count else 0.0,
                    "avg_latency_ms": float(row["avg_latency_ms"] or 0.0),
                }
            )

        for row in feedback_rows:
            metrics[row["variant"]].update(
                {
                    "feedback_count": int(row["feedback_count"]),
                    "feedback_participant_count": int(row["feedback_participant_count"]),
                    "persona_count": int(row["persona_count"]),
                    "avg_rating": float(row["avg_rating"] or 0.0),
                    "task_success_rate": float(row["task_success_rate"] or 0.0),
                }
            )

        first_exposure_at = lifetime_row["first_exposure_at"]
        last_exposure_at = lifetime_row["last_exposure_at"]
        observed_days = (
            (last_exposure_at - first_exposure_at) / 86400
            if first_exposure_at is not None and last_exposure_at is not None
            else 0.0
        )
        all_feedback_participants = int(overall_feedback_row["feedback_participant_count"])
        all_personas = int(overall_feedback_row["persona_count"])
        return {
            "experiment_name": EXPERIMENT_NAME,
            "window_days": days,
            "first_exposure_at": first_exposure_at,
            "last_exposure_at": last_exposure_at,
            "observed_days": observed_days,
            "variants": metrics,
            "delta_treatment_minus_control": self._calculate_delta(metrics),
            "totals": {
                "feedback_participant_count": all_feedback_participants,
                "persona_count": all_personas,
            },
            "completion": {
                "minimum_operation_days": MINIMUM_OPERATION_DAYS,
                "minimum_feedback_participants": MINIMUM_FEEDBACK_PARTICIPANTS,
                "minimum_personas": MINIMUM_PERSONAS,
                "has_minimum_operation_days": observed_days >= MINIMUM_OPERATION_DAYS,
                "has_minimum_feedback_participants": (
                    all_feedback_participants >= MINIMUM_FEEDBACK_PARTICIPANTS
                ),
                "has_minimum_personas": all_personas >= MINIMUM_PERSONAS,
            },
        }

    @staticmethod
    def _empty_variant_metrics() -> dict:
        return {
            "request_count": 0,
            "participant_count": 0,
            "success_count": 0,
            "error_count": 0,
            "request_success_rate": 0.0,
            "avg_latency_ms": 0.0,
            "feedback_count": 0,
            "feedback_participant_count": 0,
            "persona_count": 0,
            "avg_rating": 0.0,
            "task_success_rate": 0.0,
        }

    @staticmethod
    def _calculate_delta(metrics: dict) -> dict:
        control = metrics[CONTROL_VARIANT]
        treatment = metrics[TREATMENT_VARIANT]
        return {
            "avg_rating": treatment["avg_rating"] - control["avg_rating"],
            "task_success_rate": treatment["task_success_rate"] - control["task_success_rate"],
            "avg_latency_ms": treatment["avg_latency_ms"] - control["avg_latency_ms"],
            "request_success_rate": (
                treatment["request_success_rate"] - control["request_success_rate"]
            ),
        }


def render_markdown_report(summary: dict) -> str:
    """Render a compact Markdown report suitable for M4 evidence."""
    variants = summary["variants"]
    delta = summary["delta_treatment_minus_control"]
    completion = summary["completion"]
    lines = [
        "# GovOn RAG Feature Flag A/B Test Report",
        "",
        f"- Experiment: `{summary['experiment_name']}`",
        f"- Window: `{summary['window_days']}` days",
        f"- Observed duration (all-time): `{summary['observed_days']:.2f}` days",
        "",
        "## Variant Metrics",
        "",
        "| Metric | Control: RAG OFF | Treatment: RAG ON | Delta |",
        "|---|---:|---:|---:|",
        (
            f"| Requests | {variants[CONTROL_VARIANT]['request_count']} | "
            f"{variants[TREATMENT_VARIANT]['request_count']} | - |"
        ),
        (
            f"| Errors | {variants[CONTROL_VARIANT]['error_count']} | "
            f"{variants[TREATMENT_VARIANT]['error_count']} | - |"
        ),
        (
            f"| Request success rate | "
            f"{variants[CONTROL_VARIANT]['request_success_rate']:.2%} | "
            f"{variants[TREATMENT_VARIANT]['request_success_rate']:.2%} | "
            f"{delta['request_success_rate']:+.2%} |"
        ),
        (
            f"| Feedback participants | "
            f"{variants[CONTROL_VARIANT]['feedback_participant_count']} | "
            f"{variants[TREATMENT_VARIANT]['feedback_participant_count']} | - |"
        ),
        (
            f"| Persona count | {variants[CONTROL_VARIANT]['persona_count']} | "
            f"{variants[TREATMENT_VARIANT]['persona_count']} | - |"
        ),
        (
            f"| Average rating | {variants[CONTROL_VARIANT]['avg_rating']:.2f} | "
            f"{variants[TREATMENT_VARIANT]['avg_rating']:.2f} | "
            f"{delta['avg_rating']:+.2f} |"
        ),
        (
            f"| Task success rate | {variants[CONTROL_VARIANT]['task_success_rate']:.2%} | "
            f"{variants[TREATMENT_VARIANT]['task_success_rate']:.2%} | "
            f"{delta['task_success_rate']:+.2%} |"
        ),
        (
            f"| Average latency | {variants[CONTROL_VARIANT]['avg_latency_ms']:.2f} ms | "
            f"{variants[TREATMENT_VARIANT]['avg_latency_ms']:.2f} ms | "
            f"{delta['avg_latency_ms']:+.2f} ms |"
        ),
        "",
        "## Completion Check",
        "",
        f"- Unique feedback participants: `{summary['totals']['feedback_participant_count']}`",
        f"- Unique personas: `{summary['totals']['persona_count']}`",
        (
            f"- Minimum {completion['minimum_operation_days']}-day operation: "
            f"`{completion['has_minimum_operation_days']}`"
        ),
        (
            f"- Minimum {completion['minimum_feedback_participants']} feedback participants: "
            f"`{completion['has_minimum_feedback_participants']}`"
        ),
        f"- Minimum {completion['minimum_personas']} personas: `{completion['has_minimum_personas']}`",
        "",
    ]
    return "\n".join(lines)


def assignment_asdict(assignment: ExperimentAssignment) -> dict:
    """Return an API-friendly assignment payload."""
    return {
        **asdict(assignment),
        "experiment_name": EXPERIMENT_NAME,
        "use_rag_pipeline": assignment.use_rag_pipeline,
    }
