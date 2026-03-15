param(
    [int]$Epochs = 5,
    [int]$BatchSize = 64,
    [int]$NumWorkers = 0,
    [int]$Seed = 42,
    [string]$RunDir = "",
    [switch]$IncludeStressProbe,
    [switch]$IncludeStressTest,
    [int]$StressDuration = 5
)

$ErrorActionPreference = "Stop"
$SafetyRecall = 0.2891418508851923

function Write-WeightFile {
    param(
        [string]$Path,
        [string]$Json
    )
    Set-Content -Path $Path -Value $Json -Encoding UTF8
}

function Test-Collapse {
    param([string]$ManifestPath)
    if (-not (Test-Path $ManifestPath)) {
        return $false
    }
    $raw = Get-Content $ManifestPath -Raw
    if ([string]::IsNullOrWhiteSpace($raw)) {
        return $false
    }
    $data = $raw | ConvertFrom-Json
    $metrics = $data.tuned_threshold_metrics
    if ($null -eq $metrics) {
        return $false
    }
    if ($metrics.PSObject.Properties.Count -eq 0) {
        return $false
    }
    $tn = [int]$metrics.tn
    $fpr = [double]$metrics.fpr
    return ($tn -eq 0 -or $fpr -ge 0.999)
}

function Copy-StressArtifacts {
    param([string]$RunName)
    $runDate = Get-Date -Format "yyyy-MM-dd"
    $stressDir = Join-Path "evaluation/stress_test_v14" $runDate
    if (-not (Test-Path $stressDir)) {
        return
    }
    $items = @(
        @{ Source = (Join-Path $stressDir "timeseries_$runDate.jsonl"); Target = (Join-Path $RunDir "$RunName.stress.jsonl") },
        @{ Source = (Join-Path $stressDir "timeseries_${runDate}_failures.jsonl"); Target = (Join-Path $RunDir "$RunName.stress_failures.jsonl") },
        @{ Source = (Join-Path $stressDir "timeseries_${runDate}_ops.json"); Target = (Join-Path $RunDir "$RunName.stress_ops.json") },
        @{ Source = (Join-Path $stressDir "run_manifest_${runDate}.json"); Target = (Join-Path $RunDir "$RunName.stress_manifest.json") },
        @{ Source = (Join-Path $stressDir "dashboard_${runDate}.html"); Target = (Join-Path $RunDir "$RunName.stress_dashboard.html") }
    )
    foreach ($item in $items) {
        if (Test-Path $item.Source) {
            Copy-Item $item.Source $item.Target -Force
        }
    }
}

function Write-RecallDelta {
    param([string]$RunName)
    $opsPath = Join-Path $RunDir "$RunName.stress_ops.json"
    if (-not (Test-Path $opsPath)) {
        return
    }
    $ops = Get-Content $opsPath -Raw | ConvertFrom-Json
    $recall = [double]$ops.metrics.recall
    $delta = $recall - $SafetyRecall
    $relative = 0.0
    if ($SafetyRecall -ne 0) {
        $relative = ($delta / $SafetyRecall) * 100.0
    }
    Write-Host ("Recall delta vs safety: {0:+0.000;-0.000;0.000} ({1:+0.0;-0.0;0.0}%)" -f $delta, $relative)
}

function Resolve-RunDir {
    param([string]$BasePath)
    if (-not [string]::IsNullOrWhiteSpace($BasePath)) {
        return $BasePath
    }
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    return "evaluation/experiments/timeseries_$stamp"
}

function Run-TS {
    param(
        [string]$Name,
        [string[]]$RunArgs
    )
    $thresholdSnapshot = Join-Path $RunDir "$Name.thresholds.yaml"
    $manifestSnapshot = Join-Path $RunDir "$Name.manifest.json"

    $cmd = "python src/training/train_timeseries.py " + ($RunArgs -join " ")
    Write-Host "Running: $Name"
    Write-Host "Command: $cmd"
    Write-Host "===== $Name : TRAIN ====="
    & python src/training/train_timeseries.py @RunArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Training failed; skipping diagnose and stress test."
        return
    }

    if (Test-Path "config/model_thresholds.yaml") {
        Copy-Item "config/model_thresholds.yaml" $thresholdSnapshot -Force
    }
    if (Test-Path "models/timeseries_lstm_training_manifest.json") {
        Copy-Item "models/timeseries_lstm_training_manifest.json" $manifestSnapshot -Force
    }

    if ($IncludeStressProbe.IsPresent) {
        Write-Host "===== $Name : DIAGNOSE ====="
        & python scripts/diagnose_timeseries_artifact.py --stress-sample-count 200
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Diagnose failed or collapsed; skipping stress test."
            return
        }
    }

    if ($IncludeStressTest.IsPresent) {
        if (Test-Collapse -ManifestPath $manifestSnapshot) {
            Write-Host "Detected collapse in manifest metrics; skipping stress test."
            return
        }
        Write-Host "===== $Name : STRESS_TEST ====="
        & python -m src.stress_test.stress_test_v14 --model timeseries --seed $Seed --duration $StressDuration
        $stressExit = $LASTEXITCODE
        Copy-StressArtifacts -RunName $Name
        Write-RecallDelta -RunName $Name
        if ($stressExit -ne 0) {
            Write-Host "Stress test exited with code $stressExit; artifacts copied if available."
        }
    }
}

$RunDir = Resolve-RunDir -BasePath $RunDir
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
$log = Join-Path $RunDir "matrix.log"
Start-Transcript -Path $log -Append | Out-Null

$focusLightPath = Join-Path $RunDir "weights_focus_light.json"
$focusLightWeights = '{"ddos":0.20,"portscan":0.18,"exfiltration":0.22,"c2":0.22,"bruteforce":0.18}'
Write-WeightFile -Path $focusLightPath -Json $focusLightWeights

$focusMediumPath = Join-Path $RunDir "weights_focus_medium.json"
$focusMediumWeights = '{"ddos":0.15,"portscan":0.15,"exfiltration":0.25,"c2":0.25,"bruteforce":0.20}'
Write-WeightFile -Path $focusMediumPath -Json $focusMediumWeights

$focusStrongPath = Join-Path $RunDir "weights_focus_strong.json"
$focusStrongWeights = '{"ddos":0.12,"portscan":0.12,"exfiltration":0.28,"c2":0.28,"bruteforce":0.20}'
Write-WeightFile -Path $focusStrongPath -Json $focusStrongWeights

$baseArgs = @(
    "--epochs", "$Epochs",
    "--batch-size", "$BatchSize",
    "--num-workers", "$NumWorkers",
    "--seed", "$Seed"
)

Run-TS "run1_baseline_run5" ($baseArgs + @(
    "--stress-attack-count", "100000",
    "--attack-cap", "50000",
    "--stress-benign-count", "60000",
    "--stress-hard-negative-count", "40000",
    "--stress-val-count", "30000"
))

Run-TS "run2_focus_light" ($baseArgs + @(
    "--stress-attack-count", "100000",
    "--attack-cap", "50000",
    "--stress-benign-count", "60000",
    "--stress-hard-negative-count", "40000",
    "--stress-val-count", "30000",
    "--stress-attack-weights-file", $focusLightPath,
    "--stress-val-weights-file", $focusLightPath
))

Run-TS "run3_focus_medium" ($baseArgs + @(
    "--stress-attack-count", "100000",
    "--attack-cap", "50000",
    "--stress-benign-count", "60000",
    "--stress-hard-negative-count", "40000",
    "--stress-val-count", "30000",
    "--stress-attack-weights-file", $focusMediumPath,
    "--stress-val-weights-file", $focusMediumPath
))

Run-TS "run4_focus_strong" ($baseArgs + @(
    "--stress-attack-count", "100000",
    "--attack-cap", "50000",
    "--stress-benign-count", "60000",
    "--stress-hard-negative-count", "40000",
    "--stress-val-count", "30000",
    "--stress-attack-weights-file", $focusStrongPath,
    "--stress-val-weights-file", $focusStrongPath
))

Stop-Transcript | Out-Null
Write-Host "All runs completed. Logs and snapshots saved to $RunDir"
Write-Host "Combined log: $(Join-Path $RunDir 'matrix.log')"
