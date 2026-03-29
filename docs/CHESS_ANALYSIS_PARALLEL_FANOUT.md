# GRK-73 — 5-worker fan-out architecture

This repository now includes a GitHub Actions workflow and script contracts for a 5-worker fan-out analysis pipeline:

- **Download job** prepares exactly 300 preprocessed games and splits into 5 chunk files of 60 games each.
- **Analyse matrix jobs (`chunk_id: 0..4`)** run in parallel with `fail-fast: false`, analyze one chunk per worker, and emit one partial result artifact each.
- **Aggregate job** waits on all matrix jobs, merges partials into `report_json`, and marks the job as `done`.

## Workflow entrypoint

- `.github/workflows/chess-analysis-parallel.yml`
- Trigger: `workflow_dispatch`
- Inputs:
  - `job_id`
  - `preprocessed_games_path`
  - `convex_partial_endpoint` (optional)
  - `convex_final_endpoint` (optional)

## Partial result contract (`analysis_partial_results`)

Each chunk worker emits a JSON payload aligned with the proposed Convex table fields:

```json
{
  "job_id": "<job id>",
  "chunk_id": 0,
  "move_data": ["..."],
  "completed_at": "<ISO-8601>",
  "status": "analysing (1/5 chunks done)"
}
```

Suggested Convex table name: `analysis_partial_results`.

## Scripts

- `scripts/chess/prepare_analysis_chunks.py`
- `scripts/chess/analyze_chunk.py`
- `scripts/chess/aggregate_analysis_results.py`

These scripts keep a stable I/O contract for workflow orchestration while allowing engine internals (e.g., SF18 Pool(2)) to evolve independently.
