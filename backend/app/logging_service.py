"""
Structured logging service for the pedagogical RAG algebra tutor study.

Logs are research evidence. This module creates JSON-serializable records for:
- baseline prompt-only evaluation episodes
- pedagogical RAG evaluation episodes
- retrieval events
- learner-state transitions
- policy decisions

This module does not call an LLM and does not perform retrieval.

Author: Jude A. Adenuga
Project: Pedagogical RAG Algebra Tutor Study
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = PROJECT_ROOT / "manuscript_artifacts"
EVIDENCE_LOG_ROOT = ARTIFACT_ROOT / "evidence_logs"
EXPORT_JSON_ROOT = ARTIFACT_ROOT / "exports" / "json"

BASELINE_LOG_DIR = EVIDENCE_LOG_ROOT / "baseline"
RAG_LOG_DIR = EVIDENCE_LOG_ROOT / "rag"
SENSITIVITY_LOG_DIR = EVIDENCE_LOG_ROOT / "sensitivity"

EVALUATION_LOG_SCHEMA_VERSION = "evaluation_log_v1.0"
RETRIEVAL_LOG_SCHEMA_VERSION = "retrieval_log_v1.0"
TASK_BANK_VERSION = "v1.0"
BASELINE_PROMPT_VERSION = "baseline_prompt_v1.0"
RAG_PROMPT_VERSION = "pedagogical_rag_prompt_v1.0"
POLICY_VERSION = "pedagogical_policy_v1.1"
CORPUS_VERSION = "algebra_corpus_v1.0"
RETRIEVAL_METADATA_VERSION = "retrieval_metadata_v1.0"


class LoggingServiceError(Exception):
    """Raised when a log cannot be created or saved."""


@dataclass(frozen=True)
class EvaluationLog:
    evaluation_log_id: str
    condition: str
    session_id: str
    task_id: str
    turn_number: int
    request: Dict[str, Any]
    response: Dict[str, Any]
    created_at: str
    learner_state_before: Optional[Dict[str, Any]] = None
    learner_state_after: Optional[Dict[str, Any]] = None
    policy_decision: Optional[Dict[str, Any]] = None
    retrieval_log: Optional[Dict[str, Any]] = None
    prompt_text: Optional[str] = None
    model_id: Optional[str] = None
    latency_ms: Optional[int] = None
    error: Optional[Dict[str, Any]] = None
    versioning: Optional[Dict[str, str]] = None


@dataclass(frozen=True)
class RetrievalLog:
    retrieval_log_id: str
    session_id: str
    task_id: str
    turn_number: int
    query_text: str
    retrieval_mode: str
    top_k: int
    metadata_filters: Dict[str, Any]
    retrieved_chunks: List[Dict[str, Any]]
    created_at: str
    versioning: Dict[str, str]


def utc_now_iso() -> str:
    """Return a timezone-aware UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def ensure_directories() -> None:
    """Create local evidence directories used by the research prototype."""
    for path in [
        ARTIFACT_ROOT,
        EVIDENCE_LOG_ROOT,
        EXPORT_JSON_ROOT,
        BASELINE_LOG_DIR,
        RAG_LOG_DIR,
        SENSITIVITY_LOG_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def condition_to_log_dir(condition: str) -> Path:
    """Return the correct evidence-log directory for a study condition."""
    if condition == "baseline":
        return BASELINE_LOG_DIR

    if condition == "pedagogical_rag":
        return RAG_LOG_DIR

    if condition == "sensitivity":
        return SENSITIVITY_LOG_DIR

    raise LoggingServiceError(f"Unknown condition: {condition}")


def sanitize_filename_part(value: str) -> str:
    """Make a value safe for use in a filename."""
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
    return safe.strip("_") or "unknown"


def write_json(path: Path, payload: Dict[str, Any]) -> Path:
    """Write a dictionary as pretty-printed UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)

    return path


def default_versioning(condition: str) -> Dict[str, str]:
    """Return version fields required for reproducibility."""
    base = {
        "task_bank_version": TASK_BANK_VERSION,
        "evaluation_schema_version": EVALUATION_LOG_SCHEMA_VERSION,
    }

    if condition == "baseline":
        base.update(
            {
                "baseline_prompt_version": BASELINE_PROMPT_VERSION,
                "retrieval_enabled": "false",
                "learner_state_enabled": "false",
            }
        )

    elif condition == "pedagogical_rag":
        base.update(
            {
                "rag_prompt_version": RAG_PROMPT_VERSION,
                "policy_version": POLICY_VERSION,
                "corpus_version": CORPUS_VERSION,
                "retrieval_metadata_version": RETRIEVAL_METADATA_VERSION,
                "retrieval_enabled": "true",
                "learner_state_enabled": "true",
            }
        )

    elif condition == "sensitivity":
        base.update(
            {
                "sensitivity_prompt_version": "claude_prompt_only_sensitivity_v1.0",
                "retrieval_enabled": "false",
                "learner_state_enabled": "false",
            }
        )

    return base


def create_retrieval_log(
    session_id: str,
    task_id: str,
    turn_number: int,
    query_text: str,
    retrieved_chunks: Iterable[Dict[str, Any]],
    retrieval_mode: str = "mock",
    top_k: int = 5,
    metadata_filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a retrieval log dictionary."""
    chunks = list(retrieved_chunks or [])

    log = RetrievalLog(
        retrieval_log_id=f"retrieval_{uuid4().hex}",
        session_id=session_id,
        task_id=task_id,
        turn_number=turn_number,
        query_text=query_text,
        retrieval_mode=retrieval_mode,
        top_k=top_k,
        metadata_filters=metadata_filters or {},
        retrieved_chunks=chunks,
        created_at=utc_now_iso(),
        versioning={
            "retrieval_log_schema_version": RETRIEVAL_LOG_SCHEMA_VERSION,
            "corpus_version": CORPUS_VERSION,
            "retrieval_metadata_version": RETRIEVAL_METADATA_VERSION,
        },
    )

    return asdict(log)


def create_evaluation_log(
    condition: str,
    session_id: str,
    task_id: str,
    turn_number: int,
    request: Dict[str, Any],
    response: Dict[str, Any],
    prompt_text: Optional[str] = None,
    model_id: Optional[str] = None,
    learner_state_before: Optional[Dict[str, Any]] = None,
    learner_state_after: Optional[Dict[str, Any]] = None,
    policy_decision: Optional[Dict[str, Any]] = None,
    retrieval_log: Optional[Dict[str, Any]] = None,
    latency_ms: Optional[int] = None,
    error: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a complete evaluation log dictionary."""
    if condition not in {"baseline", "pedagogical_rag", "sensitivity"}:
        raise LoggingServiceError(f"Unsupported condition: {condition}")

    log = EvaluationLog(
        evaluation_log_id=f"eval_{uuid4().hex}",
        condition=condition,
        session_id=session_id,
        task_id=task_id,
        turn_number=turn_number,
        request=request,
        response=response,
        prompt_text=prompt_text,
        model_id=model_id,
        learner_state_before=learner_state_before,
        learner_state_after=learner_state_after,
        policy_decision=policy_decision,
        retrieval_log=retrieval_log,
        latency_ms=latency_ms,
        error=error,
        created_at=utc_now_iso(),
        versioning=default_versioning(condition),
    )

    return asdict(log)


def save_evaluation_log(log: Dict[str, Any], filename: Optional[str] = None) -> Path:
    """Save an evaluation log to the correct evidence folder."""
    ensure_directories()

    condition = str(log.get("condition", "unknown"))
    task_id = sanitize_filename_part(str(log.get("task_id", "unknown_task")))
    turn_number = sanitize_filename_part(str(log.get("turn_number", "turn")))
    log_id = sanitize_filename_part(str(log.get("evaluation_log_id", "eval")))

    output_dir = condition_to_log_dir(condition)

    if filename is None:
        filename = f"{condition}_{task_id}_turn_{turn_number}_{log_id}.json"

    return write_json(output_dir / filename, log)


def save_retrieval_log(log: Dict[str, Any], filename: Optional[str] = None) -> Path:
    """Save a retrieval log under the RAG evidence folder."""
    ensure_directories()

    task_id = sanitize_filename_part(str(log.get("task_id", "unknown_task")))
    turn_number = sanitize_filename_part(str(log.get("turn_number", "turn")))
    log_id = sanitize_filename_part(str(log.get("retrieval_log_id", "retrieval")))

    if filename is None:
        filename = f"retrieval_{task_id}_turn_{turn_number}_{log_id}.json"

    return write_json(RAG_LOG_DIR / filename, log)


def save_json_export(payload: Dict[str, Any], filename: str) -> Path:
    """Save a JSON export in manuscript_artifacts/exports/json."""
    ensure_directories()
    return write_json(EXPORT_JSON_ROOT / filename, payload)


def build_mock_model_response(
    condition: str,
    task_id: str,
    selected_pedagogical_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a placeholder response for local dry-runs.

    This is not final study output and must not be used as evaluation evidence.
    """
    if condition == "baseline":
        text = (
            f"[LOCAL DRY RUN MOCK RESPONSE for {task_id}] "
            "The baseline tutor prompt assembled successfully. No model call was made."
        )
    elif condition == "pedagogical_rag":
        text = (
            f"[LOCAL DRY RUN MOCK RESPONSE for {task_id}] "
            f"The pedagogical RAG prompt assembled successfully using mode "
            f"{selected_pedagogical_mode}. No model call was made."
        )
    else:
        text = f"[LOCAL DRY RUN MOCK RESPONSE for {task_id}] No model call was made."

    return {
        "response_id": f"mock_response_{uuid4().hex}",
        "response_text": text,
        "is_mock_response": True,
    }


if __name__ == "__main__":
    ensure_directories()

    sample_log = create_evaluation_log(
        condition="baseline",
        session_id="session_demo_logging",
        task_id="LE-02",
        turn_number=1,
        request={
            "task_id": "LE-02",
            "current_learner_message": "Solve for x: 2x + 3 = 11.",
            "dry_run": True,
        },
        response=build_mock_model_response("baseline", "LE-02"),
        prompt_text="Example prompt text",
        model_id="local_mock_no_model_call",
    )

    path = save_evaluation_log(
        sample_log,
        filename="local_logging_service_smoke_test.json",
    )

    print(f"Saved sample log to: {path}")