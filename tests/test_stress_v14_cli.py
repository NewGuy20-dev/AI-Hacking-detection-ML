"""CLI smoke tests for V1.4 stress test entrypoint."""
import pytest

import src.stress_test.stress_test_v14 as cli


def test_cli_accepts_anomaly_model(monkeypatch, tmp_path):
    """`--model anomaly` should execute runner path and exit successfully."""

    class FakeDashboard:
        def __init__(self, *_args, **_kwargs):
            pass

        def generate(self, *_args, **_kwargs):
            return None

    class FakeRunner:
        def __init__(self, model_name, config):
            self.model_name = model_name
            self.config = config

        def run(self):
            return {
                'model': self.model_name,
                'total_scenarios': 1,
                'total_duration_min': 0.0,
                'accuracy': 0.95,
                'gate_pass': True,
                'gate_failures': [],
                'gate_profile_version': 'test',
            }

    monkeypatch.setattr(cli, 'StressTestRunner', FakeRunner)
    monkeypatch.setattr(cli, 'DashboardGenerator', FakeDashboard)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        cli.sys,
        'argv',
        ['stress_test_v14.py', '--model', 'anomaly', '--duration', '0', '--no-dashboard', '--seed', '7'],
    )

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 0


def test_cli_default_all_excludes_anomaly(monkeypatch, tmp_path):
    """`--model all` should keep default runs green without anomaly artifact."""

    seen_models = []

    class FakeDashboard:
        def __init__(self, *_args, **_kwargs):
            pass

        def generate(self, *_args, **_kwargs):
            return None

    class FakeRunner:
        def __init__(self, model_name, config):
            seen_models.append(model_name)

        def run(self):
            return {
                'model': seen_models[-1],
                'total_scenarios': 1,
                'total_duration_min': 0.0,
                'accuracy': 0.95,
                'gate_pass': True,
                'gate_failures': [],
                'gate_profile_version': 'test',
            }

    monkeypatch.setattr(cli, 'StressTestRunner', FakeRunner)
    monkeypatch.setattr(cli, 'DashboardGenerator', FakeDashboard)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        cli.sys,
        'argv',
        ['stress_test_v14.py', '--model', 'all', '--duration', '0', '--no-dashboard'],
    )

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 0
    assert seen_models == cli.DEFAULT_MODELS
    assert 'anomaly' not in seen_models


def test_cli_auto_seed_is_resolved(monkeypatch, tmp_path):
    """Auto seed should be converted into a deterministic integer passed to runner."""
    observed_seeds = []

    class FakeRunner:
        def __init__(self, _model_name, config):
            observed_seeds.append(config.get('seed'))

        def run(self):
            return {
                'model': 'payload',
                'total_scenarios': 1,
                'total_duration_min': 0.0,
                'accuracy': 0.95,
                'gate_pass': True,
                'gate_failures': [],
                'gate_profile_version': 'test',
            }

    monkeypatch.setattr(cli, 'StressTestRunner', FakeRunner)
    monkeypatch.setattr(cli, 'DashboardGenerator', object)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        cli.sys,
        'argv',
        ['stress_test_v14.py', '--model', 'payload', '--duration', '0', '--no-dashboard', '--seed', 'auto'],
    )

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 0
    assert len(observed_seeds) == 1
    assert isinstance(observed_seeds[0], int)
    assert observed_seeds[0] >= 0
