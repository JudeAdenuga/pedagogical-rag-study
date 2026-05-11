#!/usr/bin/env python3
"""
Real Gemini baseline runner for the pedagogical RAG study.

Purpose:
- Run one baseline prompt-only Gemini tutoring episode.
- Preserve the baseline condition as stateless and non-retrieval.
- Save a structured JSON evidence log for later expert scoring and Supplementary Figure S12.

Important baseline constraints:
- No Gemini File Search.
- No uploaded files.
- No uploaded images.
- No external curriculum retrieval.
- No metadata filters.
- No page-level citations.
- No learner-state tracking.
- No persistent memory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from google import genai


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TASKS_PATH = PROJECT_ROOT / "backend" / "data" / "benchmark_tasks.json"
BASELINE_PROMPT_TEMPLATE_PATH = (
    PROJECT_ROOT / "backend" / "prompts" / "baseline_prompt_template.txt"
)

BASELINE_EVIDENCE_DIR = (
    PROJECT_ROOT / "manuscript_artifacts" / "evidence_logs" / "baseline"
)

DEFAULT_MODEL = "gemini-2.5-flash"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path.read_text(encoding="utf-8")


def find_task(task_id: str) -> Dict[str, Any]:
    tasks = load_json(TASKS_PATH)

    if isinstance(tasks, dict):
        if "tasks" in tasks and isinstance(tasks["tasks"], list):
            task_list = tasks["tasks"]
        else:
            task_list = list(tasks.values())
    elif isinstance(tasks, list):
        task_list = tasks
    else:
        raise ValueError("benchmark_tasks.json must be a list or a dict containing tasks.")

    for task in task_list:
        if str(task.get("task_id", "")).strip() == task_id:
            return task

    available = [str(task.get("task_id", "")) for task in task_list]
    raise ValueError(f"Task ID {task_id} not found. Available task IDs: {available}")


def get_task_field(task: Dict[str, Any], *names: str, default: str = "") -> str:
    for name in names:
        value = task.get(name)
        if value is not None:
            return str(value)
    return default


def build_baseline_prompt(task: Dict[str, Any]) -> str:
    template = load_text(BASELINE_PROMPT_TEMPLATE_PATH)

    values = {
        "task_id": get_task_field(task, "task_id"),
        "algebra_category": get_task_field(task, "algebra_category", "category"),
        "student_prompt": get_task_field(
            task,
            "student_prompt",
            "prompt",
            "student_facing_prompt",
        ),
        "simulated_follow_up_turn": get_task_field(
            task,
            "simulated_follow_up_turn",
            "follow_up",
            "simulated_followup",
        ),
    }

    try:
        return template.format(**values)
    except KeyError as exc:
        missing_key = exc.args[0]
        raise KeyError(
            f"Prompt template contains unknown placeholder: {missing_key}"
        ) from exc


def redact_secret(value: Optional[str]) -> str:
    if not value:
        return "missing"
    return "present_redacted"


def call_gemini(prompt: str, model_name: str, api_key: str) -> str:
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )

    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError("Gemini response did not include response.text")

    return text


def create_baseline_log(
    task: Dict[str, Any],
    prompt: str,
    model_name: str,
    response_text: str,
) -> Dict[str, Any]:
    return {
        "evaluation_log_id": f"baseline_{uuid.uuid4().hex}",
        "condition": "baseline_prompt_only",
        "task_id": get_task_field(task, "task_id"),
        "session_id": f"baseline_{get_task_field(task, 'task_id')}",
        "created_at": utc_now_iso(),
        "model": {
            "provider": "Google",
            "model_id": model_name,
            "api_surface": "Gemini Developer API",
        },
        "baseline_constraints": {
            "retrieval_used": False,
            "gemini_file_search_used": False,
            "file_search_store_created_or_queried": False,
            "uploaded_files_used": False,
            "uploaded_images_used": False,
            "external_curriculum_documents_used": False,
            "knowledge_base_used": False,
            "metadata_filters_used": False,
            "page_level_citations_requested": False,
            "retrieved_context_passed_to_model": False,
            "learner_state_tracking_used": False,
            "prior_session_memory_used": False,
            "persistent_memory_used": False,
        },
        "task": {
            "task_id": get_task_field(task, "task_id"),
            "algebra_category": get_task_field(task, "algebra_category", "category"),
            "student_prompt": get_task_field(
                task,
                "student_prompt",
                "prompt",
                "student_facing_prompt",
            ),
            "expected_answer_or_reasoning_target": get_task_field(
                task,
                "expected_answer",
                "expected_answer_or_reasoning_target",
                "expected_reasoning",
            ),
            "simulated_follow_up_turn": get_task_field(
                task,
                "simulated_follow_up_turn",
                "follow_up",
                "simulated_followup",
            ),
            "scoring_notes": get_task_field(task, "scoring_notes"),
        },
        "prompt": {
            "template_path": str(BASELINE_PROMPT_TEMPLATE_PATH.relative_to(PROJECT_ROOT)),
            "assembled_prompt_text": prompt,
        },
        "response": {
            "response_id": f"gemini_response_{uuid.uuid4().hex}",
            "response_text": response_text,
            "is_mock_response": False,
        },
        "environment": {
            "gemini_api_key": redact_secret(os.getenv("GEMINI_API_KEY")),
            "model_from_env": os.getenv("GEMINI_MODEL", DEFAULT_MODEL),
        },
        "evidence_use": {
            "supplementary_candidate": True,
            "suggested_caption": (
                "Supplementary Figure S12. Complete logged baseline episode showing "
                "the prompt-only Gemini baseline configuration, task prompt, model "
                "identifier, and raw tutor output."
            ),
            "notes": (
                "This log supports the baseline condition only. It does not include "
                "retrieval, learner-state tracking, external curriculum grounding, "
                "Gemini File Search, uploaded files, uploaded images, metadata filters, "
                "or page-level citations."
            ),
        },
    }


def save_log(log: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(log, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a real prompt-only Gemini baseline episode."
    )
    parser.add_argument(
        "--task-id",
        default="LE-02",
        help="Benchmark task ID to run. Default: LE-02",
    )
    parser.add_argument(
        "--real-model",
        action="store_true",
        help="Required safety flag confirming that a real Gemini model call should be made.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output JSON path.",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")

    args = parse_args()

    if not args.real_model:
        print(
            json.dumps(
                {
                    "status": "refused_without_real_model_flag",
                    "message": "Add --real-model to make the real Gemini baseline call.",
                },
                indent=2,
            )
        )
        return 2

    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL

    if not api_key or api_key == "PASTE_YOUR_REAL_GEMINI_API_KEY_HERE":
        print(
            json.dumps(
                {
                    "status": "missing_gemini_api_key",
                    "message": "Set GEMINI_API_KEY in .env before running the baseline.",
                    "env_file": str((PROJECT_ROOT / ".env").relative_to(PROJECT_ROOT)),
                },
                indent=2,
            )
        )
        return 1

    task = find_task(args.task_id)
    prompt = build_baseline_prompt(task)
    response_text = call_gemini(prompt=prompt, model_name=model_name, api_key=api_key)

    log = create_baseline_log(
        task=task,
        prompt=prompt,
        model_name=model_name,
        response_text=response_text,
    )

    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
    else:
        output_path = (
            BASELINE_EVIDENCE_DIR
            / f"baseline_episode_{args.task_id.replace('-', '_')}_sample.json"
        )

    save_log(log, output_path)

    print(
        json.dumps(
            {
                "status": "baseline_episode_complete",
                "task_id": args.task_id,
                "condition": "baseline_prompt_only",
                "model": model_name,
                "retrieval_used": False,
                "file_search_used": False,
                "learner_state_used": False,
                "model_calls_made": True,
                "external_services_called": True,
                "log_path": str(output_path.relative_to(PROJECT_ROOT)),
            },
            indent=2,
        )
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())