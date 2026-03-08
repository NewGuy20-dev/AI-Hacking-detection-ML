import importlib.util
import sys
from pathlib import Path
from unittest.mock import Mock, patch

from src.retraining_trigger import RetrainingTrigger


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_retrain_all_uses_current_interpreter(monkeypatch):
    module = _load_module("retrain_all_script", "scripts/retrain_all.py")
    captured = {}

    def fake_run(cmd, cwd=None):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        return Mock(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    ok = module.run_script(Path("src/training/train_url.py"), "URL CNN Training")

    assert ok is True
    assert captured["cmd"][0] == sys.executable


def test_trainpipeline_uses_current_interpreter(monkeypatch, tmp_path):
    module = _load_module("trainpipeline_script", "scripts/trainpipeline.py")
    captured_cmds = []

    monkeypatch.setattr(module, "run", lambda cmd, env: captured_cmds.append(cmd))
    monkeypatch.setattr(module, "PIPELINE_LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "trainpipeline.py",
            "--skip-handshake",
            "--skip-label-check",
            "--skip-train",
            "--skip-quick",
            "--skip-analyze",
            "--skip-calibrate",
            "--skip-full",
        ],
    )

    module.main()

    assert captured_cmds
    assert captured_cmds[0][0] == sys.executable


def test_retraining_trigger_uses_current_interpreter_for_training():
    trigger = RetrainingTrigger(
        registry=Mock(),
        monitor=Mock(),
    )
    trigger.validate_new_model = Mock(return_value={"passed": True, "pass_rate": 0.95})

    with patch("src.retraining_trigger.subprocess.run", return_value=Mock(returncode=0)) as mock_run:
        with patch("src.retraining_trigger.Path.exists", return_value=True):
            result = trigger.trigger_retraining("url_cnn", reason="test")

    assert result["status"] == "completed"
    assert mock_run.call_args.args[0][0] == sys.executable


def test_retraining_trigger_uses_current_interpreter_for_validation():
    trigger = RetrainingTrigger(
        registry=Mock(),
        monitor=Mock(),
    )

    with patch("src.retraining_trigger.subprocess.run", return_value=Mock(returncode=0)) as mock_run:
        with patch("pathlib.Path.exists", return_value=False):
            result = trigger.validate_new_model("url_cnn")

    assert result["passed"] is True
    assert mock_run.call_args.args[0][0] == sys.executable
