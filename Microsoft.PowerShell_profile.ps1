# Auto-activate PyTorch venv for this project
$venvPath = "D:\pytorch_env\Scripts\Activate.ps1"

if (Test-Path $venvPath) {
    & $venvPath
}
