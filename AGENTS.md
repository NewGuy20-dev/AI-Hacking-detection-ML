# Repository Guidelines

## Project Structure & Module Organization
- `src/`: Core Python code (models, training, API, stress tests).
- `tests/`: Pytest/unittest coverage for validators, APIs, training, and integrations.
- `scripts/`: Data prep, retraining, pipelines, and dataset utilities.
- `datasets/`: Training data (large, often generated or downloaded).
- `models/`: Trained model artifacts (`.pkl`, `.pt`).
- `dashboard/`: Next.js UI (see `dashboard/package.json`).
- `configs/` and `config/`: Scenario/config YAML and runtime settings.
- `evaluation/`: Generated reports, logs, and stress test outputs.
- `docs/`: Detailed plans and reference docs.

## Build, Test, and Development Commands
- `py -3 -m pip install -r requirements.txt`: Install Python dependencies (Windows).
- `py -3 -m uvicorn src.api.server:app --reload`: Run the FastAPI server locally.
- `./start-api.ps1` (PowerShell) or `bash start-api.sh` (WSL): Start API with project defaults.
- `py -3 src/train_all_models.py`: Train the full model suite.
- `py -3 src/training/train_payload.py`: Train a specific PyTorch model.
- `py -3 src/stress_test/stress_test_v14.py --model url,payload`: Run V1.4 stress tests.
- `py -3 -m pytest tests/` (Windows) or `pytest tests/` (WSL): Run the test suite.
- `cd dashboard && npm run dev`: Run the Next.js dashboard.
- `cd dashboard && npm run lint`: Lint the dashboard.

## Command Sanity Checks
- `py -3 scripts/stress_test_v14.py --help`: Verify wrapper-to-source wiring for V1.4 stress tests.
- `py -3 scripts/validate_models.py`: Run model validation through the compatibility wrapper.
- `py -3 scripts/prepare_url_data.py`: Run URL data preparation via canonical generator entrypoint.
- `py -3 src/feedback_loop/hard_example_loop.py --model payload,url --dry-run`: Validate feedback-loop ingestion and dataset packaging.
- `py -3 src/feedback_loop/hard_example_loop.py --model payload,url --promote`: Run manual closed-loop retraining with strict promotion gates.

## Coding Style & Naming Conventions
- Python: 4-space indentation, `snake_case` for functions/variables, `CamelCase` for classes.
- Favor clear, data-focused names (`url_features`, `payload_cnn`, `train_timeseries`).
- Keep modules small and task-specific; prefer adding new files under `src/` rather than expanding monoliths.

## Testing Guidelines
- Primary framework is `pytest`; some tests use `unittest` patterns.
- Name tests as `test_*.py` under `tests/`.
- If adding new model logic, include unit tests plus an integration test when feasible.

## Commit & Pull Request Guidelines
- Recent commits follow a lightweight conventional style: `feat: ...`, `fix: ...`, or scoped `fix(stress-test): ...`.
- Use imperative, concise subjects that describe the change.
- PRs should include: summary, testing performed, and any dataset/model artifact changes.
- Include screenshots for dashboard UI changes.

## Data, Security & Configuration Tips
- Avoid committing large datasets or generated outputs; keep them in `datasets/` or `evaluation/` and document how to reproduce.
- Model artifacts belong in `models/` and should be listed in PR descriptions when updated.
- Prefer config-driven changes via `configs/` or `config/` rather than hardcoding.
- If artifacts are missing, wrappers should fail fast with clear path errors instead of silently skipping model loads.
- For reproducible stress tests, pass a fixed seed (example: `py -3 src/stress_test/stress_test_v14.py --seed 42 --duration 5`).
