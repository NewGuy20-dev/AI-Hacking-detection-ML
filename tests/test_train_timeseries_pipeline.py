import json
from pathlib import Path

import numpy as np

from src.training.train_timeseries import (
    _load_or_generate_timeseries_data,
    generate_benign_hard_negative_traffic,
    generate_stress_aligned_attack_traffic,
    generate_stress_aligned_hard_negative_traffic,
)
from src.timeseries_synthetic import generate_stress_aligned_benign_sequences


def test_generate_benign_hard_negative_traffic_returns_60x8_sequences():
    seq = generate_benign_hard_negative_traffic(8)

    assert seq.shape == (8, 60, 8)
    assert seq.dtype == np.float32
    assert np.isfinite(seq).all()
    # The generated benign hard negatives should not be perfectly flat.
    assert float(seq.std()) > 0.0


def test_generate_stress_aligned_benign_sequences_returns_60x8_sequences():
    seq = generate_stress_aligned_benign_sequences(8)

    assert seq.shape == (8, 60, 8)
    assert seq.dtype == np.float32
    assert np.isfinite(seq).all()
    assert float(seq.max()) >= 0.0


def test_generate_stress_aligned_hard_negative_traffic_returns_60x8_sequences():
    seq = generate_stress_aligned_hard_negative_traffic(8)

    assert seq.shape == (8, 60, 8)
    assert seq.dtype == np.float32
    assert np.isfinite(seq).all()
    assert float(seq.std()) > 0.0


def test_generate_stress_aligned_attack_traffic_returns_60x8_sequences():
    seq = generate_stress_aligned_attack_traffic(8)

    assert seq.shape == (8, 60, 8)
    assert seq.dtype == np.float32
    assert np.isfinite(seq).all()
    assert float(seq.std()) > 0.0


def test_load_or_generate_timeseries_data_reports_sources(tmp_path):
    base = tmp_path
    ts_dir = base / "datasets" / "timeseries"
    ts_dir.mkdir(parents=True, exist_ok=True)

    normal = np.full((12, 60, 8), 10.0, dtype=np.float32)
    attack = np.full((14, 60, 8), 20.0, dtype=np.float32)
    np.save(ts_dir / "normal_traffic_expansion.npy", normal)
    np.save(ts_dir / "attack_traffic_expansion.npy", attack)

    sequences, labels, mins, maxs, source_names, summary = _load_or_generate_timeseries_data(
        base,
        normal_cap=10,
        attack_cap=8,
        hard_negative_count=4,
        stress_benign_count=6,
        stress_hard_negative_count=4,
        stress_attack_count=3,
    )

    assert sequences.shape == (35, 60, 8)
    assert labels.shape == (35,)
    assert source_names.shape == (35,)
    assert mins.shape == (1, 1, 8)
    assert maxs.shape == (1, 1, 8)
    assert summary["source_details"]["live_benign_present"] is False
    assert summary["source_counts"]["synthetic_normal_expansion"] == {
        "total": 10,
        "malicious": 0,
        "benign": 10,
    }
    assert summary["source_counts"]["generated_benign_hard_negatives"]["total"] == 4
    assert summary["source_counts"]["stress_aligned_benign"]["total"] == 6
    assert summary["source_counts"]["stress_aligned_hard_negatives"]["total"] == 4
    assert summary["source_counts"]["synthetic_attack_expansion"] == {
        "total": 8,
        "malicious": 8,
        "benign": 0,
    }
    assert summary["source_counts"]["stress_aligned_attack"]["total"] == 3
    assert "source_stats" in summary
    assert "synthetic_normal_expansion" in summary["source_stats"]
    assert summary["totals"] == {
        "total": 35,
        "malicious": 11,
        "benign": 24,
    }


def test_timeseries_static_fixtures_are_full_sequences():
    fixture_dir = Path("configs/stress_test/scenarios_v14/fixtures")

    for name in ["timeseries_ddos.json", "timeseries_normal.json"]:
        data = json.loads((fixture_dir / name).read_text(encoding="utf-8"))
        arr = np.asarray(data, dtype=np.float32)
        assert arr.shape == (60, 8)
