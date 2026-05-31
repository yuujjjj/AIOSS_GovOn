"""Feature flag, experiment assignment, and rollout helpers.

The runtime supports three control layers:

1. Global environment variables, for example ``USE_RAG_PIPELINE=false``.
2. Targeted user overrides, for example ``ENABLE_AGENT_TOOLS_TARGET_USERS=pilot-a``.
3. Request overrides through ``X-Feature-Flag`` for operator/debug traffic.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

_TRUE_VALUES = {"true", "1", "yes", "on"}
_FALSE_VALUES = {"false", "0", "no", "off"}
_BOOLEAN_FLAG_ENV = {
    "use_rag_pipeline": "USE_RAG_PIPELINE",
    "enable_hybrid_search": "ENABLE_HYBRID_SEARCH",
    "enable_agent_tools": "ENABLE_AGENT_TOOLS",
    "enable_streaming_response": "ENABLE_STREAMING_RESPONSE",
}
_MODEL_VERSIONS = {"v1_lora", "v2_lora"}


@dataclass(frozen=True)
class TargetingRule:
    """Per-flag user targeting rule loaded from environment variables."""

    enabled_users: tuple[str, ...] = ()
    disabled_users: tuple[str, ...] = ()

    def resolve(self, default: bool, user_id: Optional[str]) -> bool:
        if not user_id:
            return default
        if user_id in self.disabled_users:
            return False
        if user_id in self.enabled_users:
            return True
        return default


@dataclass(frozen=True)
class ExperimentAssignment:
    """Stable A/B experiment assignment."""

    experiment_key: str
    variant: str
    bucket: int
    user_id: str


@dataclass(frozen=True)
class CanaryDecision:
    """Decision for the next canary rollout step."""

    current_percentage: int
    next_percentage: int
    action: str
    reason: str


EXPERIMENT_CONFIGS: Dict[str, tuple[str, ...]] = {
    "complaint_response_layout": ("control", "guided"),
    "answer_tone": ("formal", "plain_language"),
}

CANARY_STAGES = (1, 10, 50, 100)


def _parse_bool(value: str) -> Optional[bool]:
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    return None


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    parsed = _parse_bool(value)
    if parsed is None:
        logger.warning(f"Invalid boolean value for {name}: {value}; using {default}")
        return default
    return parsed


def _csv_tuple(value: Optional[str]) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _load_targeting_rules() -> Dict[str, TargetingRule]:
    return {
        flag_name: TargetingRule(
            enabled_users=_csv_tuple(os.getenv(f"{env_name}_TARGET_USERS")),
            disabled_users=_csv_tuple(os.getenv(f"{env_name}_DISABLED_USERS")),
        )
        for flag_name, env_name in _BOOLEAN_FLAG_ENV.items()
    }


def stable_bucket(key: str, modulo: int = 10000) -> int:
    """Return a stable bucket for a user/experiment key."""

    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % modulo


def assign_experiment(user_id: str, experiment_key: str) -> ExperimentAssignment:
    """Assign a user to one configured A/B variant with deterministic hashing."""

    if experiment_key not in EXPERIMENT_CONFIGS:
        raise ValueError(f"Unknown experiment: {experiment_key}")

    normalized_user = user_id.strip() or "anonymous"
    forced_variant = os.getenv(f"EXPERIMENT_{experiment_key.upper()}_FORCE_VARIANT")
    variants = EXPERIMENT_CONFIGS[experiment_key]
    if forced_variant in variants:
        return ExperimentAssignment(
            experiment_key=experiment_key,
            variant=forced_variant,
            bucket=0,
            user_id=normalized_user,
        )

    bucket = stable_bucket(f"{experiment_key}:{normalized_user}")
    variant_index = min((bucket * len(variants)) // 10000, len(variants) - 1)
    return ExperimentAssignment(
        experiment_key=experiment_key,
        variant=variants[variant_index],
        bucket=bucket,
        user_id=normalized_user,
    )


def assign_all_experiments(user_id: str) -> Dict[str, ExperimentAssignment]:
    """Assign a user to every configured experiment."""

    return {
        experiment_key: assign_experiment(user_id, experiment_key)
        for experiment_key in EXPERIMENT_CONFIGS
    }


def track_experiment_exposure(
    assignment: ExperimentAssignment,
    *,
    event_log_path: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Append an experiment exposure event as JSONL and return the event body."""

    event = {
        "event": "experiment_exposure",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment_key": assignment.experiment_key,
        "variant": assignment.variant,
        "bucket": assignment.bucket,
        "user_id": assignment.user_id,
        "metadata": metadata or {},
    }

    if _env_bool("FEATURE_EVENT_TRACKING_ENABLED", True):
        path = Path(
            event_log_path or os.getenv("FEATURE_EVENT_LOG_PATH", "logs/experiments/events.jsonl")
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")

    return event


def evaluate_canary_rollout(
    current_percentage: int,
    *,
    health_status: str,
    error_rate: float,
    p95_latency_ms: int,
    max_error_rate: float = 0.02,
    max_p95_latency_ms: int = 2500,
) -> CanaryDecision:
    """Evaluate whether a canary rollout can advance or must roll back."""

    if health_status != "healthy":
        return CanaryDecision(current_percentage, 0, "rollback", "health check is not healthy")
    if error_rate > max_error_rate:
        return CanaryDecision(current_percentage, 0, "rollback", "error rate exceeded threshold")
    if p95_latency_ms > max_p95_latency_ms:
        return CanaryDecision(current_percentage, 0, "rollback", "p95 latency exceeded threshold")

    for stage in CANARY_STAGES:
        if stage > current_percentage:
            return CanaryDecision(current_percentage, stage, "advance", "health gates passed")

    return CanaryDecision(
        current_percentage,
        current_percentage,
        "complete",
        "rollout already at 100%",
    )


@dataclass(frozen=True)
class FeatureFlags:
    """Runtime feature flag settings."""

    use_rag_pipeline: bool = True
    enable_hybrid_search: bool = True
    enable_agent_tools: bool = True
    enable_streaming_response: bool = True
    model_version: str = "v2_lora"  # v1_lora | v2_lora
    targeting_rules: Dict[str, TargetingRule] = field(default_factory=dict, repr=False)

    @classmethod
    def from_env(cls) -> "FeatureFlags":
        """Load feature flags from environment variables."""

        flags = cls(
            use_rag_pipeline=_env_bool("USE_RAG_PIPELINE", True),
            enable_hybrid_search=_env_bool("ENABLE_HYBRID_SEARCH", True),
            enable_agent_tools=_env_bool("ENABLE_AGENT_TOOLS", True),
            enable_streaming_response=_env_bool("ENABLE_STREAMING_RESPONSE", True),
            model_version=os.getenv("MODEL_VERSION", "v2_lora"),
            targeting_rules=_load_targeting_rules(),
        )
        logger.info(f"Feature Flags loaded: {flags.to_public_dict()}")
        return flags

    def to_public_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe summary without expanding targeting internals."""

        return {
            "use_rag_pipeline": self.use_rag_pipeline,
            "enable_hybrid_search": self.enable_hybrid_search,
            "enable_agent_tools": self.enable_agent_tools,
            "enable_streaming_response": self.enable_streaming_response,
            "model_version": self.model_version,
        }

    def is_enabled(self, flag_name: str, user_id: Optional[str] = None) -> bool:
        """Resolve a boolean flag for an optional target user."""

        if flag_name not in _BOOLEAN_FLAG_ENV:
            raise ValueError(f"Unknown boolean feature flag: {flag_name}")

        default = bool(getattr(self, flag_name))
        rule = self.targeting_rules.get(flag_name, TargetingRule())
        return rule.resolve(default, user_id)

    def for_user(self, user_id: Optional[str]) -> "FeatureFlags":
        """Resolve all targetable boolean flags for a user."""

        updates = {
            flag_name: self.is_enabled(flag_name, user_id)
            for flag_name in _BOOLEAN_FLAG_ENV
        }
        return replace(self, **updates)

    def override_from_header(self, header_value: Optional[str]) -> "FeatureFlags":
        """Apply request-level overrides from ``X-Feature-Flag``.

        Format: ``USE_RAG_PIPELINE=false,MODEL_VERSION=v1_lora``.
        """

        if not header_value:
            return self

        overrides: Dict[str, Any] = {}
        env_to_field = {env_name: field_name for field_name, env_name in _BOOLEAN_FLAG_ENV.items()}
        for pair in header_value.split(","):
            pair = pair.strip()
            if "=" not in pair:
                continue
            key, value = pair.split("=", 1)
            key = key.strip().upper()
            value = value.strip()

            if key in env_to_field:
                parsed = _parse_bool(value)
                if parsed is not None:
                    overrides[env_to_field[key]] = parsed
            elif key == "MODEL_VERSION" and value in _MODEL_VERSIONS:
                overrides["model_version"] = value

        if not overrides:
            return self

        return replace(self, **overrides)
