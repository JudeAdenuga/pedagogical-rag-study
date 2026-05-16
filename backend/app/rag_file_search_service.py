#!/usr/bin/env python3
"""
Gemini File Search service for the live pedagogical RAG condition.

This file is additive only. It does not modify existing baseline, dry-run,
prompt, learner-state, task, or logging files.

Study role:
1. Render the existing curated algebra corpus seed into one small text file per chunk.
2. Create a Gemini File Search store.
3. Upload/import those files with metadata.
4. Run a File Search retrieval probe using Gemini 2.5 Flash.
5. Extract retrieved grounding chunks.
6. Convert those chunks into the shape expected by the existing
   build_pedagogical_rag_prompt() function.

The natural-language output of the retrieval probe is NOT the final tutor answer.
Only the retrieved grounding chunks are used for prompt enrichment.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CORPUS_SEED_PATH = PROJECT_ROOT / "backend" / "data" / "algebra_corpus_seed.json"

GENERATED_CORPUS_DOCS_DIR = (
    PROJECT_ROOT
    / "manuscript_artifacts"
    / "exports"
    / "gemini_file_search_corpus_docs"
)

JSON_EXPORT_DIR = (
    PROJECT_ROOT
    / "manuscript_artifacts"
    / "exports"
    / "json"
)

STORE_MANIFEST_PATH = (
    JSON_EXPORT_DIR / "gemini_file_search_store_manifest.json"
)

DEFAULT_STORE_DISPLAY_NAME = "pedagogical-rag-algebra-corpus-v1"
DEFAULT_STORE_EMBEDDING_MODEL = "models/gemini-embedding-001"
DEFAULT_RETRIEVAL_MODEL = "gemini-2.5-flash"
DEFAULT_MAX_CHUNKS_TO_INJECT = 5

RAG_FILE_SEARCH_SERVICE_VERSION = "rag_file_search_service_v1.0"


class RagFileSearchServiceError(Exception):
    """Raised when File Search setup or retrieval cannot complete."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_project_env() -> None:
    """Load existing .env without modifying it."""
    load_dotenv(PROJECT_ROOT / ".env")


def get_api_key() -> str:
    load_project_env()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        raise RagFileSearchServiceError(
            "GEMINI_API_KEY is missing. Confirm your existing .env still contains it."
        )

    return api_key


def get_retrieval_model() -> str:
    """
    Use Gemini 2.5 Flash unless an explicit RAG retrieval model override
    is provided in the environment.
    """
    load_project_env()
    return os.getenv(
        "GEMINI_RAG_RETRIEVAL_MODEL",
        os.getenv("GEMINI_MODEL", DEFAULT_RETRIEVAL_MODEL),
    )


def get_max_chunks_to_inject() -> int:
    load_project_env()
    raw = os.getenv(
        "GEMINI_RAG_MAX_CHUNKS_TO_INJECT",
        str(DEFAULT_MAX_CHUNKS_TO_INJECT),
    )

    try:
        value = int(raw)
    except ValueError as exc:
        raise RagFileSearchServiceError(
            f"GEMINI_RAG_MAX_CHUNKS_TO_INJECT must be an integer. Received: {raw}"
        ) from exc

    if value < 1:
        raise RagFileSearchServiceError(
            "GEMINI_RAG_MAX_CHUNKS_TO_INJECT must be at least 1."
        )

    return value


def build_client() -> genai.Client:
    return genai.Client(api_key=get_api_key())


def safe_slug(value: str) -> str:
    """
    General local filename slug used for generated evidence documents.
    This may retain underscores because it is only used for local paths.
    """
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value).strip())
    slug = re.sub(r"_+", "_", slug)
    return slug.strip("_") or "unknown"


def safe_gemini_file_resource_name(value: str) -> str:
    """
    Gemini Files API resource names must contain only:
    - lowercase letters
    - digits
    - dashes

    They also cannot begin or end with a dash.
    """
    name = str(value).strip().lower()
    name = re.sub(r"[^a-z0-9-]+", "-", name)
    name = re.sub(r"-+", "-", name)
    name = name.strip("-")

    return name or f"rag-file-{uuid4().hex[:12]}"


def load_corpus_seed() -> Dict[str, Any]:
    if not CORPUS_SEED_PATH.exists():
        raise RagFileSearchServiceError(
            f"Corpus seed file not found: {CORPUS_SEED_PATH}"
        )

    try:
        data = json.loads(CORPUS_SEED_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RagFileSearchServiceError(
            f"Invalid JSON in corpus seed file: {CORPUS_SEED_PATH}"
        ) from exc

    chunks = data.get("chunks")

    if not isinstance(chunks, list) or not chunks:
        raise RagFileSearchServiceError(
            "Corpus seed must contain a non-empty top-level 'chunks' array."
        )

    return data


def corpus_chunk_index() -> Dict[str, Dict[str, Any]]:
    data = load_corpus_seed()
    index: Dict[str, Dict[str, Any]] = {}

    for chunk in data["chunks"]:
        chunk_id = str(chunk.get("chunk_id", "")).strip()
        if chunk_id:
            index[chunk_id] = chunk

    return index


def render_chunk_as_text_document(
    chunk: Dict[str, Any],
    corpus_version: str,
) -> str:
    """
    Render each curated corpus chunk into a compact search document.

    The explicit metadata header makes retrieved grounding excerpts easier to audit.
    """
    supports_tasks = ", ".join(chunk.get("supports_tasks", []))
    pedagogical_use = ", ".join(chunk.get("pedagogical_use", []))

    return f"""Title: {chunk.get("title", "Untitled instructional chunk")}
chunk_id: {chunk.get("chunk_id", "")}
corpus_version: {corpus_version}
topic: {chunk.get("topic", "")}
subskill: {chunk.get("subskill", "")}
difficulty: {chunk.get("difficulty", "")}
resource_type: {chunk.get("resource_type", "")}
misconception_tag: {chunk.get("misconception_tag", "")}
supports_tasks: {supports_tasks}
pedagogical_use: {pedagogical_use}

Instructional content:
{chunk.get("content", "")}
"""


def render_file_search_documents() -> List[Dict[str, Any]]:
    """
    Create one plain-text File Search document per existing corpus chunk.

    This writes only new generated artifacts under manuscript_artifacts/exports/.
    It does not modify the source corpus JSON.
    """
    data = load_corpus_seed()
    corpus_version = str(data.get("corpus_version", "v1.0"))

    GENERATED_CORPUS_DOCS_DIR.mkdir(parents=True, exist_ok=True)

    rendered_documents: List[Dict[str, Any]] = []

    for chunk in data["chunks"]:
        chunk_id = str(chunk.get("chunk_id", f"chunk_{uuid4().hex}"))
        filename = f"{safe_slug(chunk_id)}.txt"
        path = GENERATED_CORPUS_DOCS_DIR / filename

        path.write_text(
            render_chunk_as_text_document(
                chunk=chunk,
                corpus_version=corpus_version,
            ),
            encoding="utf-8",
        )

        rendered_documents.append(
            {
                "chunk_id": chunk_id,
                "title": chunk.get("title"),
                "path": str(path),
                "metadata": {
                    "topic": chunk.get("topic"),
                    "subskill": chunk.get("subskill"),
                    "difficulty": chunk.get("difficulty"),
                    "resource_type": chunk.get("resource_type"),
                    "misconception_tag": chunk.get("misconception_tag"),
                    "supports_tasks": chunk.get("supports_tasks", []),
                    "pedagogical_use": chunk.get("pedagogical_use", []),
                    "corpus_version": corpus_version,
                },
            }
        )

    return rendered_documents


def build_custom_metadata(document: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Attach scalar metadata fields to imported File Search documents.

    supports_tasks and pedagogical_use are stored as comma-separated strings.
    """
    metadata = document.get("metadata", {}) or {}

    return [
        {
            "key": "chunk_id",
            "string_value": str(document.get("chunk_id", "")),
        },
        {
            "key": "title",
            "string_value": str(document.get("title", "")),
        },
        {
            "key": "topic",
            "string_value": str(metadata.get("topic", "")),
        },
        {
            "key": "subskill",
            "string_value": str(metadata.get("subskill", "")),
        },
        {
            "key": "difficulty",
            "string_value": str(metadata.get("difficulty", "")),
        },
        {
            "key": "resource_type",
            "string_value": str(metadata.get("resource_type", "")),
        },
        {
            "key": "misconception_tag",
            "string_value": str(metadata.get("misconception_tag", "")),
        },
        {
            "key": "supports_tasks",
            "string_value": ",".join(metadata.get("supports_tasks", [])),
        },
        {
            "key": "pedagogical_use",
            "string_value": ",".join(metadata.get("pedagogical_use", [])),
        },
        {
            "key": "corpus_version",
            "string_value": str(metadata.get("corpus_version", "")),
        },
    ]


def wait_for_operation(
    client: genai.Client,
    operation: Any,
    poll_seconds: int = 5,
    timeout_seconds: int = 900,
    progress_label: Optional[str] = None,
) -> Any:
    """
    Wait for a Gemini long-running File Search indexing operation.

    Progress output is intentionally visible so corpus ingestion does not look frozen.
    """
    started_at = time.time()
    current_operation = operation
    label = progress_label or "File Search operation"

    print(f"    ↳ {label}: indexing started.", flush=True)

    while not getattr(current_operation, "done", False):
        elapsed = int(time.time() - started_at)

        if elapsed > timeout_seconds:
            raise RagFileSearchServiceError(
                f"{label} timed out after {timeout_seconds} seconds."
            )

        print(
            f"    ↳ {label}: still indexing... {elapsed}s elapsed",
            flush=True,
        )

        time.sleep(poll_seconds)
        current_operation = client.operations.get(current_operation)

    total_elapsed = int(time.time() - started_at)

    print(
        f"    ✓ {label}: indexing complete in {total_elapsed}s.",
        flush=True,
    )

    return current_operation


def create_file_search_store(client: genai.Client) -> Any:
    """
    Create a new File Search store for the study corpus.

    We keep the store name in a generated manifest file rather than editing .env.
    """
    load_project_env()

    display_name = os.getenv(
        "GEMINI_FILE_SEARCH_STORE_DISPLAY_NAME",
        DEFAULT_STORE_DISPLAY_NAME,
    )
    embedding_model = os.getenv(
        "GEMINI_FILE_SEARCH_EMBEDDING_MODEL",
        DEFAULT_STORE_EMBEDDING_MODEL,
    )

    return client.file_search_stores.create(
        config={
            "display_name": display_name,
            "embedding_model": embedding_model,
        }
    )


def upload_and_import_documents(
    client: genai.Client,
    store_name: str,
    documents: Iterable[Dict[str, Any]],
    poll_seconds: int = 5,
    timeout_seconds: int = 900,
) -> List[Dict[str, Any]]:
    """
    Directly upload each rendered instructional corpus document into the
    Gemini File Search store with visible progress logging.

    This uses upload_to_file_search_store() to avoid raw-file naming collisions.
    """
    document_list = list(documents)
    total_documents = len(document_list)

    if total_documents == 0:
        raise RagFileSearchServiceError(
            "No rendered corpus documents were available for File Search upload."
        )

    print("", flush=True)
    print("Preparing Gemini File Search corpus ingestion...", flush=True)
    print(f"Target store: {store_name}", flush=True)
    print(f"Documents to upload/index: {total_documents}", flush=True)
    print("", flush=True)

    imported: List[Dict[str, Any]] = []

    for index, document in enumerate(document_list, start=1):
        file_path = Path(str(document["path"]))
        chunk_id = str(document.get("chunk_id") or "unknown_chunk")
        title = str(document.get("title") or chunk_id)

        if not file_path.exists():
            raise RagFileSearchServiceError(
                f"Rendered File Search document not found: {file_path}"
            )

        print(
            f"[{index}/{total_documents}] Uploading {chunk_id} — {title}",
            flush=True,
        )

        operation = client.file_search_stores.upload_to_file_search_store(
            file_search_store_name=store_name,
            file=str(file_path),
            config={
                "display_name": title,
                "custom_metadata": build_custom_metadata(document),
            },
        )

        completed_operation = wait_for_operation(
            client=client,
            operation=operation,
            poll_seconds=poll_seconds,
            timeout_seconds=timeout_seconds,
            progress_label=f"{chunk_id}",
        )

        imported.append(
            {
                "chunk_id": document.get("chunk_id"),
                "title": document.get("title"),
                "generated_document_path": str(file_path),
                "upload_operation_name": getattr(operation, "name", None),
                "operation_done": bool(
                    getattr(completed_operation, "done", False)
                ),
            }
        )

        print(
            f"[{index}/{total_documents}] Completed {chunk_id}.",
            flush=True,
        )
        print("", flush=True)

    print(
        f"Gemini File Search corpus ingestion complete: {len(imported)}/{total_documents} documents indexed.",
        flush=True,
    )
    print("", flush=True)

    return imported


def write_store_manifest(payload: Dict[str, Any]) -> Path:
    JSON_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    STORE_MANIFEST_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return STORE_MANIFEST_PATH


def load_store_manifest() -> Optional[Dict[str, Any]]:
    if not STORE_MANIFEST_PATH.exists():
        return None

    try:
        return json.loads(STORE_MANIFEST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RagFileSearchServiceError(
            f"Store manifest exists but is not valid JSON: {STORE_MANIFEST_PATH}"
        ) from exc


def resolve_store_name() -> str:
    """
    Resolve store name from environment first, then from the generated manifest.

    This avoids requiring any edit to the existing .env file.
    """
    load_project_env()

    env_store_name = os.getenv("GEMINI_FILE_SEARCH_STORE_NAME", "").strip()
    if env_store_name:
        return env_store_name

    manifest = load_store_manifest()
    if manifest:
        manifest_store_name = str(manifest.get("store_name", "")).strip()
        if manifest_store_name:
            return manifest_store_name

    raise RagFileSearchServiceError(
        "No File Search store name is available. Run:\n"
        "python3 backend/app/rag_file_search_service.py prepare-store"
    )


def prepare_store(
    force_new_store: bool = False,
    poll_seconds: int = 5,
    timeout_seconds: int = 900,
) -> Dict[str, Any]:
    """
    Create/import the Gemini File Search store.

    If a manifest already exists, this function reuses it unless
    --force-new-store is passed.
    """
    existing_manifest = load_store_manifest()

    if existing_manifest and not force_new_store:
        return {
            "status": "existing_store_manifest_reused",
            "store_name": existing_manifest.get("store_name"),
            "manifest_path": str(STORE_MANIFEST_PATH),
            "message": (
                "Existing File Search store manifest found. "
                "No new store was created and no documents were re-imported."
            ),
        }

    client = build_client()
    rendered_documents = render_file_search_documents()
    store = create_file_search_store(client)

    store_name = getattr(store, "name", None)

    if not store_name:
        raise RagFileSearchServiceError(
            "Gemini created a File Search store but did not return store.name."
        )

    imported_documents = upload_and_import_documents(
        client=client,
        store_name=store_name,
        documents=rendered_documents,
        poll_seconds=poll_seconds,
        timeout_seconds=timeout_seconds,
    )

    payload = {
        "status": "gemini_file_search_store_ready",
        "rag_file_search_service_version": RAG_FILE_SEARCH_SERVICE_VERSION,
        "store_name": store_name,
        "store_display_name": getattr(store, "display_name", None),
        "created_at": utc_now_iso(),
        "corpus_seed_path": str(CORPUS_SEED_PATH),
        "generated_corpus_documents_dir": str(GENERATED_CORPUS_DOCS_DIR),
        "documents_rendered": len(rendered_documents),
        "documents_imported": len(imported_documents),
        "rendered_documents": rendered_documents,
        "imported_documents": imported_documents,
    }

    manifest_path = write_store_manifest(payload)
    payload["manifest_path"] = str(manifest_path)

    return payload


def build_retrieval_probe_prompt(query_text: str) -> str:
    """
    Prompt used only to trigger Gemini File Search retrieval.

    The natural-language response is not used as the tutoring answer.
    We ask the model to ground its brief acknowledgement in retrieved file
    evidence so that grounding chunks are more reliably returned.
    """
    return f"""Use the File Search store to retrieve the most relevant algebra instructional evidence for the request below.

You must rely on retrieved File Search evidence for this retrieval step.
Do not solve the tutoring task.
After retrieving evidence, reply with one brief sentence only:
"Retrieved relevant instructional evidence."

Retrieval request:
{query_text}
"""


def custom_metadata_to_dict(custom_metadata: Any) -> Dict[str, Any]:
    output: Dict[str, Any] = {}

    for item in list(custom_metadata or []):
        key = getattr(item, "key", None)
        if not key:
            continue

        string_value = getattr(item, "string_value", None)
        numeric_value = getattr(item, "numeric_value", None)

        if string_value is not None:
            output[str(key)] = string_value
        elif numeric_value is not None:
            output[str(key)] = numeric_value
        else:
            output[str(key)] = None

    return output


def grounding_metadata_from_response(response: Any) -> Any:
    """
    Support both response.grounding_metadata and
    response.candidates[0].grounding_metadata.
    """
    direct_grounding = getattr(response, "grounding_metadata", None)
    if direct_grounding is not None:
        return direct_grounding

    candidates = getattr(response, "candidates", None) or []
    if candidates:
        return getattr(candidates[0], "grounding_metadata", None)

    return None


def extract_chunk_id_from_retrieved_text(text: str) -> Optional[str]:
    match = re.search(r"chunk_id:\s*([A-Za-z0-9_-]+)", text or "")
    if match:
        return match.group(1).strip()
    return None


def normalize_retrieved_chunks(
    response: Any,
    max_chunks_to_inject: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Convert Gemini grounding chunks into the prompt-ready retrieved chunk shape
    already expected by prompts.py.

    When possible, retrieved chunks are linked back to the original local corpus
    by chunk_id and the original curated content is injected into the final prompt.
    """
    grounding_metadata = grounding_metadata_from_response(response)

    if grounding_metadata is None:
        return []

    grounding_chunks = getattr(grounding_metadata, "grounding_chunks", None) or []
    corpus_index = corpus_chunk_index()

    max_chunks = (
        max_chunks_to_inject
        if max_chunks_to_inject is not None
        else get_max_chunks_to_inject()
    )

    normalized: List[Dict[str, Any]] = []

    for index, grounding_chunk in enumerate(grounding_chunks, start=1):
        retrieved_context = getattr(grounding_chunk, "retrieved_context", None)

        if retrieved_context is None:
            continue

        retrieved_text = getattr(retrieved_context, "text", None)
        if not retrieved_text or not str(retrieved_text).strip():
            continue

        retrieved_text = str(retrieved_text).strip()
        custom_metadata = custom_metadata_to_dict(
            getattr(retrieved_context, "custom_metadata", None)
        )

        chunk_id = (
            custom_metadata.get("chunk_id")
            or extract_chunk_id_from_retrieved_text(retrieved_text)
            or f"gemini_file_search_chunk_{index}"
        )

        source_chunk = corpus_index.get(str(chunk_id))
        source_metadata = source_chunk or {}

        title = (
            custom_metadata.get("title")
            or source_metadata.get("title")
            or getattr(retrieved_context, "title", None)
            or f"Gemini File Search retrieved chunk {index}"
        )

        content_for_prompt = (
            source_metadata.get("content")
            or retrieved_text
        )

        normalized.append(
            {
                "chunk_id": str(chunk_id),
                "title": str(title),
                "score": "not_exposed_by_gemini_file_search",
                "content": str(content_for_prompt).strip(),
                "retrieval_excerpt": retrieved_text,
                "metadata": {
                    "topic": (
                        custom_metadata.get("topic")
                        or source_metadata.get("topic")
                    ),
                    "subskill": (
                        custom_metadata.get("subskill")
                        or source_metadata.get("subskill")
                    ),
                    "difficulty": (
                        custom_metadata.get("difficulty")
                        or source_metadata.get("difficulty")
                    ),
                    "resource_type": (
                        custom_metadata.get("resource_type")
                        or source_metadata.get("resource_type")
                    ),
                    "misconception_tag": (
                        custom_metadata.get("misconception_tag")
                        or source_metadata.get("misconception_tag")
                    ),
                    "supports_tasks": (
                        custom_metadata.get("supports_tasks")
                        or source_metadata.get("supports_tasks", [])
                    ),
                    "pedagogical_use": (
                        custom_metadata.get("pedagogical_use")
                        or source_metadata.get("pedagogical_use", [])
                    ),
                    "retrieval_source": "gemini_file_search",
                    "file_search_store": getattr(
                        retrieved_context,
                        "file_search_store",
                        None,
                    ),
                    "uri": getattr(retrieved_context, "uri", None),
                    "page_number": getattr(
                        retrieved_context,
                        "page_number",
                        None,
                    ),
                },
            }
        )

        if len(normalized) >= max_chunks:
            break

    return normalized


def run_retrieval_probe(
    query_text: str,
    retrieval_model: Optional[str] = None,
    max_attempts: int = 3,
    retry_sleep_seconds: float = 2.0,
) -> Dict[str, Any]:
    """
    Run Gemini File Search to retrieve grounded instructional evidence.

    In some retrieval probes, Gemini may return a text acknowledgement without
    exposing grounding chunks. Because this study requires retrieved evidence to
    be explicitly captured and injected into the downstream pedagogical RAG
    prompt, we retry the File Search probe a small number of times before failing.

    This preserves methodological integrity:
    - no retrieval chunks -> no RAG generation proceeds;
    - successful retrieval chunks -> downstream prompt enrichment proceeds.
    """
    if not query_text or not query_text.strip():
        raise RagFileSearchServiceError("Retrieval query text is empty.")

    if max_attempts < 1:
        raise RagFileSearchServiceError("max_attempts must be at least 1.")

    client = build_client()
    store_name = resolve_store_name()
    model_name = retrieval_model or get_retrieval_model()

    attempt_records: List[Dict[str, Any]] = []
    last_response_text: Optional[str] = None

    for attempt_number in range(1, max_attempts + 1):
        prompt = build_retrieval_probe_prompt(query_text)

        print(
            f"Gemini File Search retrieval probe attempt "
            f"{attempt_number}/{max_attempts}...",
            flush=True,
        )

        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[
                        types.Tool(
                            file_search=types.FileSearch(
                                file_search_store_names=[store_name]
                            )
                        )
                    ]
                ),
            )
        except errors.ClientError as exc:
            error_text = str(exc)

            is_resource_exhausted = (
                "429" in error_text
                or "RESOURCE_EXHAUSTED" in error_text
                or "Failed to embed content" in error_text
            )

            if is_resource_exhausted and attempt_number < max_attempts:
                backoff_seconds = 45 * attempt_number

                print(
                    f"Gemini File Search retrieval attempt "
                    f"{attempt_number}/{max_attempts} hit temporary "
                    f"RESOURCE_EXHAUSTED / embedding pressure.",
                    flush=True,
                )
                print(
                    f"Waiting {backoff_seconds}s before retrying retrieval...",
                    flush=True,
                )

                attempt_records.append(
                    {
                        "attempt_number": attempt_number,
                        "response_text": None,
                        "retrieved_chunk_count": 0,
                        "grounding_chunks_found": False,
                        "api_error": error_text,
                        "retry_scheduled": True,
                        "retry_wait_seconds": backoff_seconds,
                    }
                )

                time.sleep(backoff_seconds)
                continue

            raise

        last_response_text = getattr(response, "text", None)
        retrieved_chunks = normalize_retrieved_chunks(response)

        attempt_record = {
            "attempt_number": attempt_number,
            "response_text": last_response_text,
            "retrieved_chunk_count": len(retrieved_chunks),
            "grounding_chunks_found": bool(retrieved_chunks),
        }
        attempt_records.append(attempt_record)

        if retrieved_chunks:
            print(
                f"Gemini File Search retrieval succeeded on attempt "
                f"{attempt_number} with {len(retrieved_chunks)} chunks.",
                flush=True,
            )

            return {
                "retrieval_probe_id": f"rag_retrieval_probe_{uuid4().hex}",
                "rag_file_search_service_version": RAG_FILE_SEARCH_SERVICE_VERSION,
                "store_name": store_name,
                "retrieval_model": model_name,
                "retrieval_probe_prompt": prompt,
                "retrieval_probe_response_text": last_response_text,
                "retrieved_chunk_count": len(retrieved_chunks),
                "retrieved_chunks": retrieved_chunks,
                "retrieval_attempts": attempt_records,
                "created_at": utc_now_iso(),
            }

        print(
            f"No grounding chunks were exposed on retrieval attempt "
            f"{attempt_number}/{max_attempts}.",
            flush=True,
        )

        if attempt_number < max_attempts:
            time.sleep(retry_sleep_seconds)

    raise RagFileSearchServiceError(
        "Gemini File Search completed, but no retrieved grounding chunks were "
        f"extracted after {max_attempts} attempts. "
        f"Last response text: {last_response_text!r}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gemini File Search store preparation and retrieval probe utilities."
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare-store",
        help="Create the Gemini File Search store and import the algebra corpus.",
    )
    prepare_parser.add_argument(
        "--force-new-store",
        action="store_true",
        help="Create a new store even if a prior manifest already exists.",
    )
    prepare_parser.add_argument(
        "--poll-seconds",
        type=int,
        default=5,
        help="Polling interval for File Search import operations.",
    )
    prepare_parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=900,
        help="Timeout for each File Search import operation.",
    )

    retrieve_parser = subparsers.add_parser(
        "retrieve-test",
        help="Run one retrieval probe against the prepared File Search store.",
    )
    retrieve_parser.add_argument(
        "--query",
        required=True,
        help="Retrieval query text.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "prepare-store":
        result = prepare_store(
            force_new_store=args.force_new_store,
            poll_seconds=args.poll_seconds,
            timeout_seconds=args.timeout_seconds,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if args.command == "retrieve-test":
        result = run_retrieval_probe(query_text=args.query)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    raise RagFileSearchServiceError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
