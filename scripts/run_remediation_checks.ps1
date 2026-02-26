param(
    [switch]$RunPytest,
    [switch]$CreateBaseline
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)][string]$Title,
        [Parameter(Mandatory = $true)][string]$Command
    )
    Write-Host "`n=== $Title ===" -ForegroundColor Cyan
    Write-Host $Command -ForegroundColor DarkGray
    Invoke-Expression $Command
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

Write-Host "Repo root: $repoRoot" -ForegroundColor Green

Invoke-Step -Title "Syntax Sanity (py_compile)" -Command @"
python -m py_compile src\data_guardrails.py src\prefilters\benign_pre_filter.py src\stress_test\v14\scenarios.py src\stress_test\v14\logger.py scripts\diagnose_timeseries_outputs.py scripts\profile_network_latency.py scripts\capture_score_distributions.py scripts\capture_remediation_baseline.py scripts\prepare_payload_dataset.py scripts\prepare_url_dataset.py scripts\analyze_url_dataset.py scripts\fetch_dga_samples.py scripts\augment_fraud_dataset.py tests\test_data_guardrails.py tests\test_benign_pre_filter.py tests\test_meta_generator_distributions.py
"@

if ($RunPytest) {
    Invoke-Step -Title "Unit Tests (pytest)" -Command @"
python -m pytest -q tests\test_data_guardrails.py tests\test_benign_pre_filter.py tests\test_meta_generator_distributions.py
"@
}
else {
    Write-Host "`nSkipping pytest (pass -RunPytest to enable)." -ForegroundColor Yellow
}

Invoke-Step -Title "Timeseries Diagnosis (2026-02-25 logs)" -Command @"
python .\scripts\diagnose_timeseries_outputs.py --input-log .\evaluation\stress_test_v14\2026-02-25\timeseries_2026-02-25.jsonl
"@

Invoke-Step -Title "Network Latency Profiling (2026-02-25 logs)" -Command @"
python .\scripts\profile_network_latency.py --input-log .\evaluation\stress_test_v14\2026-02-25\network_2026-02-25.jsonl --slow-ms 100 --top-k 10
"@

Invoke-Step -Title "Capture Meta Score Distributions (from latest logs)" -Command @"
python .\scripts\capture_score_distributions.py --from-logs-dir .\evaluation\stress_test_v14\2026-02-25 --output .\configs\score_distributions.json
"@

Invoke-Step -Title "Analyze URL Failures (2026-02-25)" -Command @"
python .\scripts\analyze_url_dataset.py --failures-file .\evaluation\stress_test_v14\2026-02-25\url_2026-02-25_failures.jsonl --show-categories --sample-limit 50000
"@

Invoke-Step -Title "Hard-Example Feedback Loop Dry-Run (payload,url)" -Command @"
python .\src\feedback_loop\hard_example_loop.py --model payload,url --dry-run
"@

if ($CreateBaseline) {
    Invoke-Step -Title "Create Baseline Snapshot JSON (optional)" -Command @"
python .\scripts\capture_remediation_baseline.py --input-dir .\evaluation\stress_test_v14\2026-02-25 --output .\evaluation\remediation_baseline_2026-02-25.json
"@
}
else {
    Write-Host "`nSkipping baseline snapshot (pass -CreateBaseline to enable)." -ForegroundColor Yellow
}

Write-Host "`nAll requested remediation checks completed." -ForegroundColor Green
