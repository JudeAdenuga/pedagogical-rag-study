
## LE-02 local dry-run evidence refresh

Date refreshed: 2026-05-11

Reason:
The local dry-run was rerun after the initial commit. The regenerated files changed only run-specific identifiers and timestamps, including evaluation_log_id, response_id, and created_at fields.

Interpretation:
The substantive dry-run evidence remains unchanged:
- task_id = LE-02
- dry_run = true
- mock response used
- no model call made
- no external service called
- matched local evidence output preserved


## Real Gemini baseline evidence: LE-02

Date captured: 2026-05-11

Status: baseline_episode_complete

Task ID: LE-02

Artifacts:
- backend/app/baseline.py
- manuscript_artifacts/evidence_logs/baseline/baseline_episode_LE_02_sample.json

Condition:
- baseline_prompt_only
- model = gemini-2.5-flash
- retrieval_used = false
- gemini_file_search_used = false
- learner_state_tracking_used = false
- is_mock_response = false
- model_calls_made = true
- external_services_called = true

Supplementary placement:
- Candidate source for Supplementary Figure S12 after sanitization.

