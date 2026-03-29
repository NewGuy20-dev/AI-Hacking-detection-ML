# GRK-75 — Testing: parallelisation + classification end-to-end

## Purpose

This runbook defines the full pre-ship validation for the parallelised analysis pipeline (GRK-73) and move-classification system (GRK-74), plus regression checks on report rendering and data compatibility.

---

## Exit Criteria (Ship Gate)

A GRK-75 test cycle is **PASS** only if all of the following are true:

1. 5-worker parallel fan-out completes for a 300-game opponent dataset.
2. All chunk partials are persisted and then merged into a single aggregate report payload.
3. Aggregation remains successful when one worker fails (degraded but successful completion).
4. Temporary partial-result rows are cleaned up after merge.
5. Engine Status UI reflects live chunk progress (`X/5 chunks done`).
6. End-to-end wall time is ~10–11 minutes and billed minutes are ~50 (tolerance ±15%).
7. Classification labels satisfy single-label invariant (exactly one label per position).
8. Classification behavior matches the rules for WDL parsing, Best/Miss/Book/Brilliant logic.
9. v4 report rendering remains correct with new classification output shape.
10. `analysis_cache` remains backward compatible for existing consumers.
11. Runs complete successfully for at least 3 opponents spanning rating bands.

---

## Test Data & Environment Prerequisites

### Environment

- Deploy branch containing GRK-73 + GRK-74 changes.
- Enable engine logs at INFO or DEBUG for:
  - worker chunk lifecycle
  - classification reason tracing
  - aggregator merge/cleanup
- Ensure observability access for:
  - background job logs
  - Convex tables/queries
  - Engine Status page
  - billing/usage metrics

### Opponent pool

Select three opponents with at least 300 games each:

- **Low rating band** (e.g., <1200)
- **Mid rating band** (e.g., 1200–1800)
- **High rating band** (e.g., >1800)

Record each opponent in the execution log before starting.

### Determinism controls

- Use a fixed engine depth/nodes configuration for comparison runs.
- Keep worker count fixed at 5 for all GRK-75 parallelisation checks.
- Disable non-essential background jobs in the test environment where possible.

---

## Test Matrix

| Area | ID | Scenario | Expected Result |
|---|---|---|---|
| Parallelisation | P1 | 5-worker 300-game run | Completes successfully; all chunks finished |
| Parallelisation | P2 | Verify partial rows | 5 partial-result entries exist before aggregation completion |
| Parallelisation | P3 | Verify aggregate merge | Single merged `report_json` present and structurally valid |
| Parallelisation | P4 | Simulate worker failure | Remaining chunks complete; aggregator completes with explicit gap handling |
| Parallelisation | P5 | Cleanup check | `analysis_partial_results` rows removed post-aggregation |
| Parallelisation | P6 | Live progress UX | Engine Status shows monotonic `X/5 chunks done` updates |
| Parallelisation | P7 | Performance budget | Wall time 10–11 min target; billed minutes ~50 target |
| Classification | C1 | WDL parsing from UCI | Parsed W/D/L values numerically correct per sampled `info` lines |
| Classification | C2 | Best detection | `pv[0]` persisted and Best classification fires when played move matches |
| Classification | C3 | Label quality cross-check | Sampled labels align with Chess.com review tendency (manual QA set) |
| Classification | C4 | Miss predicate strictness | Miss only when all three conditions hold |
| Classification | C5 | Book window | Book flags valid opening moves up to move 15 only |
| Classification | C6 | Brilliant detection | Known sacrifice fixture triggers Brilliant |
| Classification | C7 | Label exclusivity/completeness | No zero-label and no multi-label positions |
| Regression | R1 | v4 report rendering | All report sections render and consume updated shape |
| Regression | R2 | Cache compatibility | Legacy consumers continue reading `analysis_cache` |
| Regression | R3 | Opponent diversity | Entire suite run for 3 rating-diverse opponents |

---

## Execution Steps

### Phase A — Parallelisation validation

### A1. Baseline 5-worker fan-out

1. Start analysis job for opponent A (300 games).
2. Capture:
   - job start timestamp
   - worker chunk assignment logs
   - per-chunk completion timestamps
3. Verify all 5 workers report completion.

**Pass conditions**
- 5/5 chunks complete.
- Aggregator enters merge stage.

### A2. Partial result persistence

1. During active run, query Convex partial table.
2. Confirm each chunk writes a unique partial payload.
3. Snapshot count and IDs before aggregation finalizes.

**Pass conditions**
- Exactly 5 distinct partials are written in success case.

### A3. Aggregate merge integrity

1. After completion, inspect aggregate artifact.
2. Validate merged `report_json`:
   - valid JSON
   - expected top-level sections
   - position/game counts match chunk sum

**Pass conditions**
- One merged report record exists.
- Merged totals are consistent with chunk totals.

### A4. Failure injection (1 worker)

1. Re-run with controlled failure in one chunk (e.g., forced exception for chunk N).
2. Observe remaining 4 chunks.
3. Confirm aggregator behavior and final job status.

**Pass conditions**
- 4 chunks complete normally.
- Aggregator completes gracefully with explicit degraded marker.
- No deadlock/retry storm.

### A5. Partial cleanup

1. After aggregator completion (success and degraded cases), query partial table.
2. Confirm stale partial rows are removed for finished job.

**Pass conditions**
- `analysis_partial_results` has no rows for finalized job IDs.

### A6. Engine Status progress UX

1. Monitor Engine Status page during an active run.
2. Capture progress transitions over time.

**Pass conditions**
- Displays `X/5 chunks done` updates.
- Progress is monotonic and reaches terminal state.

### A7. Timing and billing envelope

1. Measure wall time from enqueue to completed state.
2. Capture billed minutes from usage telemetry.

**Pass conditions**
- Wall time in target range (~10–11 min, ±15%).
- Billed minutes near ~50 (±15%).

---

### Phase B — Move classification validation

### B1. WDL parsing from UCI output

1. Collect raw sampled UCI `info` lines containing WDL values.
2. Compare parser output to raw token values.

**Pass conditions**
- Parsed W, D, L values match source line values exactly for sample set.

### B2. `pv[0]` persistence + Best detection

1. Inspect stored position rows for sampled games.
2. Verify `pv[0]` persisted per position.
3. Confirm Best label appears when played move equals `pv[0]` under applicable logic.

**Pass conditions**
- `pv[0]` available for all sampled non-terminal positions.
- Best label aligns with move equality predicate.

### B3. Label cross-check vs Chess.com

1. Select a stratified sample (opening/middlegame/endgame, winning/losing/equal).
2. Compare local labels (Blunder/Mistake/Inaccuracy) to Chess.com game review outcomes.
3. Record disagreements with context.

**Pass conditions**
- High directional agreement (target ≥80% on sampled set).
- Any systematic discrepancy documented with root-cause hypothesis.

### B4. Miss detection strictness

Validate that Miss triggers **only** if all are true:

1. WDL drop present
2. `pv[0] != played`
3. `pv[0]` is capture or check

**Pass conditions**
- No Miss labels when any single predicate is false.

### B5. Book detection boundaries

1. Replay known opening sequences.
2. Confirm Book appears on known lines.
3. Confirm Book never applies after move 15.

**Pass conditions**
- Book classifications limited to opening window and valid line membership.

### B6. Brilliant detection fixture

1. Run known sacrifice test position from fixture data.
2. Confirm classification and trace reason.

**Pass conditions**
- Brilliant fires on fixture.
- Reason trace indicates sacrifice-based predicate path.

### B7. Single-label invariant

1. Run invariant checks across all analyzed positions.

**Pass conditions**
- Every position has exactly one label.
- Zero violations for both missing and multi-label conditions.

---

### Phase C — Regression

### C1. v4 report rendering

1. Open generated v4 report outputs from all three opponents.
2. Validate each section renders without runtime errors.

**Pass conditions**
- No missing sections, no broken charts/tables tied to classification schema changes.

### C2. `analysis_cache` compatibility

1. Validate schema contract against existing readers.
2. Replay at least one previously cached artifact with new reader path and vice versa (if supported).

**Pass conditions**
- No breaking read/write contract for legacy consumers.

### C3. Multi-opponent sweep

1. Execute full suite (A+B+C) on three rating-diverse opponents.

**Pass conditions**
- All three complete with equivalent pass criteria.

---

## Evidence to Collect

For each run store:

- Job ID, opponent ID, game count, worker count
- Start/end timestamps, wall time
- Chunk-level status table (0..4)
- Partial row snapshots pre/post aggregation
- Aggregate report identifier + schema checksum
- Classification QA sample sheet
- Engine Status screenshots (progress + completed)
- Billing snapshot
- Final PASS/FAIL and defect links

---

## Defect Triage Rules

Severity mapping:

- **P0**: Data loss/corruption, aggregator deadlock, invalid report JSON
- **P1**: Incorrect move labels at scale, invariant violations, cache incompatibility
- **P2**: Progress UI mismatch, non-blocking performance drift
- **P3**: Logging/observability gaps without user impact

Any P0/P1 blocks release.

---

## Suggested Automation Hooks

Where feasible, automate these checks in CI or nightly staging:

- Invariant assertion: exactly one classification label per position
- Miss predicate property tests (all 3 conditions required)
- Book cutoff test (no Book after move 15)
- Partial cleanup assertion post-aggregation
- Aggregator degraded-path integration test with forced chunk failure
- Report schema snapshot test for v4 rendering contracts

---

## Execution Log Template

Use this template per run:

```text
Run ID:
Date (UTC):
Build/Commit:
Environment:
Opponent:
Rating band:
Games analyzed:

Parallelisation
- 5-worker completion: PASS/FAIL
- Partial rows present (count):
- Merge correctness: PASS/FAIL
- Failure-injection behavior: PASS/FAIL
- Partial cleanup post-merge: PASS/FAIL
- Engine Status live progress: PASS/FAIL
- Wall time:
- Billed minutes:

Classification
- WDL parsing sample: PASS/FAIL
- pv[0] persistence + Best: PASS/FAIL
- Chess.com cross-check agreement (%):
- Miss strictness: PASS/FAIL
- Book boundary: PASS/FAIL
- Brilliant fixture: PASS/FAIL
- Single-label invariant: PASS/FAIL

Regression
- v4 report rendering: PASS/FAIL
- analysis_cache compatibility: PASS/FAIL

Overall decision:
Blocking defects:
Notes:
```

---

## Sign-off Checklist

- [ ] Parallelisation suite passed (P1–P7)
- [ ] Classification suite passed (C1–C7)
- [ ] Regression suite passed (R1–R3)
- [ ] Evidence artifacts attached
- [ ] Defects triaged and linked
- [ ] Release recommendation recorded
