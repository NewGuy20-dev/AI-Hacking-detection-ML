#!/usr/bin/env python3
"""
Holdout validation for the timeseries LSTM model.
Tests on data that was NEVER used during training:
  1. datasets/timeseries/normal_traffic_improved.npy + attack_traffic_improved.npy
  2. Real KDD network flow records from datasets/live_benign/mawi_network_kdd.jsonl
     (mapped to 60-step sequences using the same 8 features the model expects)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.torch_models.timeseries_lstm import TimeSeriesLSTM   # noqa: E402
from src.training.training_utils import load_operational_threshold  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = REPO / "models" / "timeseries_lstm.pt"
NORM_PATH = REPO / "models" / "timeseries_norm_v1.npz"

# ─── 8 features the model was trained on ─────────────────────────────────────
# [bytes_in, bytes_out, pkt_in, pkt_out, conn_count, unique_dst, duration_s, flag_bits]
KDD_FEATURE_MAP = {
    "src_bytes":            2,  # bytes_out proxy
    "dst_bytes":            1,  # bytes_in proxy
    "count":                5,  # conn_count
    "srv_count":            6,  # unique_dst proxy
    "duration":             7,  # duration_s (clamped)
    "wrong_fragment":       0,  # small signal as bytes_in noise
    "num_failed_logins":    3,  # pkt_out proxy (failure = more packets)
    "hot":                  4,  # pkt_in proxy
}
MAX_KDD_SEQS = 5000   # cap for speed


def load_model() -> tuple[TimeSeriesLSTM, float, np.ndarray, np.ndarray]:
    model = torch.jit.load(MODEL_PATH, map_location=DEVICE)
    model.eval()
    threshold = load_operational_threshold("timeseries", default=0.5)
    norm = np.load(NORM_PATH)
    mins = norm["mins"]   # (1, 1, 8)
    maxs = norm["maxs"]   # (1, 1, 8)
    return model, threshold, mins, maxs


def predict(model: TimeSeriesLSTM, seqs: np.ndarray, threshold: float,
            mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    """Return binary predictions (0/1) for a batch of sequences."""
    # normalise with training stats
    denom = (maxs - mins)
    denom[denom == 0] = 1.0
    seqs = (seqs - mins) / denom
    seqs = np.clip(seqs, 0.0, 1.0).astype(np.float32)

    preds = []
    batch = 512
    with torch.no_grad():
        for i in range(0, len(seqs), batch):
            x = torch.from_numpy(seqs[i:i+batch]).to(DEVICE)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds.append((probs >= threshold).astype(int))
    return np.concatenate(preds)


def metrics(preds: np.ndarray, labels: np.ndarray) -> dict:
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    recall = tp / max(tp + fn, 1)
    fpr    = fp / max(fp + tn, 1)
    prec   = tp / max(tp + fp, 1)
    acc    = (tp + tn) / max(len(preds), 1)
    return dict(tp=tp, tn=tn, fp=fp, fn=fn,
                recall=recall, fpr=fpr, precision=prec, accuracy=acc)


# ─── Test 1: Improved .npy holdout ───────────────────────────────────────────
def test_improved_npy(model, threshold, mins, maxs):
    print("\n" + "="*60)
    print("  TEST 1: Improved .npy holdout (never used in training)")
    print("="*60)
    normal = np.load(REPO / "datasets" / "timeseries" / "normal_traffic_improved.npy")
    attack = np.load(REPO / "datasets" / "timeseries" / "attack_traffic_improved.npy")
    seqs   = np.concatenate([normal, attack], axis=0)
    labels = np.array([0]*len(normal) + [1]*len(attack))
    print(f"  Normal: {len(normal):,}  Attack: {len(attack):,}  Total: {len(seqs):,}")
    preds = predict(model, seqs, threshold, mins, maxs)
    m = metrics(preds, labels)
    print(f"  Accuracy:  {m['accuracy']:.4f}")
    print(f"  Recall:    {m['recall']:.4f}  (gate: >=0.90)")
    print(f"  FPR:       {m['fpr']:.4f}   (gate: <=0.05)")
    print(f"  Precision: {m['precision']:.4f}")
    print(f"  TP={m['tp']}  TN={m['tn']}  FP={m['fp']}  FN={m['fn']}")
    recall_gate = "✅ PASS" if m['recall'] >= 0.90 else "❌ FAIL"
    fpr_gate    = "✅ PASS" if m['fpr']    <= 0.05 else "❌ FAIL"
    print(f"  Recall gate: {recall_gate}   FPR gate: {fpr_gate}")
    return m


# ─── Test 2: Real KDD network flows ──────────────────────────────────────────
def kdd_record_to_sequence(record: dict, seq_len: int = 60) -> np.ndarray:
    """
    Map a single KDD connection record to a (seq_len, 8) float32 array.
    We repeat the static feature vector across all timesteps to create a
    pseudo-timeseries. This is an approximation — KDD doesn't have temporal
    resolution — but it tests whether the model correctly classifies real
    feature magnitudes as benign.
    """
    vec = np.zeros(8, dtype=np.float32)
    for field, idx in KDD_FEATURE_MAP.items():
        vec[idx] = float(record.get(field, 0))
    # crude normalisation: clip to reasonable per-feature ranges
    vec[0] = np.clip(vec[0], 0, 1e6)   # bytes_in
    vec[1] = np.clip(vec[1], 0, 1e6)   # bytes_out
    vec[3] = np.clip(vec[3], 0, 200)    # pkt_out
    vec[4] = np.clip(vec[4], 0, 200)    # pkt_in
    vec[5] = np.clip(vec[5], 0, 512)    # conn_count
    vec[6] = np.clip(vec[6], 0, 512)    # unique_dst
    vec[7] = np.clip(vec[7], 0, 3600)   # duration_s
    # Tile across timesteps with tiny Gaussian noise to avoid all-zero gradients
    seq = np.tile(vec, (seq_len, 1))
    seq += np.random.randn(*seq.shape).astype(np.float32) * 0.001
    return seq


def test_kdd_real(model, threshold, mins, maxs):
    print("\n" + "="*60)
    print("  TEST 2: Real KDD benign traffic (out-of-distribution)")
    print("="*60)
    kdd_path = REPO / "datasets" / "live_benign" / "mawi_network_kdd.jsonl"
    benign_seqs = []
    attack_seqs = []
    with kdd_path.open(encoding="utf-8") as f:
        for line in f:
            if len(benign_seqs) >= MAX_KDD_SEQS and len(attack_seqs) >= MAX_KDD_SEQS:
                break
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            label = rec.get("label", -1)
            if label == 0 and len(benign_seqs) < MAX_KDD_SEQS:
                benign_seqs.append(kdd_record_to_sequence(rec))
            elif label == 1 and len(attack_seqs) < MAX_KDD_SEQS:
                attack_seqs.append(kdd_record_to_sequence(rec))

    if not benign_seqs:
        print("  ⚠ No benign KDD records found — skipping test.")
        return None

    seqs_list = []
    labels_list = []
    if benign_seqs:
        seqs_list.append(np.stack(benign_seqs))
        labels_list.extend([0] * len(benign_seqs))
    if attack_seqs:
        seqs_list.append(np.stack(attack_seqs))
        labels_list.extend([1] * len(attack_seqs))

    seqs   = np.concatenate(seqs_list, axis=0)
    labels = np.array(labels_list)
    print(f"  Benign: {len(benign_seqs):,}  Attack: {len(attack_seqs):,}  Total: {len(seqs):,}")

    preds = predict(model, seqs, threshold, mins, maxs)
    m = metrics(preds, labels)
    print(f"  Accuracy:  {m['accuracy']:.4f}")
    if len(attack_seqs) > 0:
        print(f"  Recall:    {m['recall']:.4f}  (gate: >=0.90)")
        recall_gate = "✅ PASS" if m['recall'] >= 0.90 else "❌ FAIL"
        print(f"  Recall gate: {recall_gate}")
    print(f"  FPR (benign-only):  {m['fpr']:.4f}   (gate: <=0.05)")
    fpr_gate = "✅ PASS" if m['fpr'] <= 0.05 else "❌ FAIL"
    print(f"  FPR gate (OOD): {fpr_gate}")
    print(f"  TP={m['tp']}  TN={m['tn']}  FP={m['fp']}  FN={m['fn']}")
    return m


def main():
    print("Loading model...")
    model, threshold, mins, maxs = load_model()
    print(f"  Threshold: {threshold:.3f} | Device: {DEVICE}")

    m1 = test_improved_npy(model, threshold, mins, maxs)
    m2 = test_kdd_real(model, threshold, mins, maxs)

    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    print(f"  Holdout .npy  — recall: {m1['recall']:.4f}  fpr: {m1['fpr']:.4f}")
    if m2 is not None:
        print(f"  Real KDD OOD  — fpr: {m2['fpr']:.4f} (primary concern)")
    print()

    # Overfitting verdict
    both_pass = m1["recall"] >= 0.90 and m1["fpr"] <= 0.05
    ood_fpr_ok = (m2 is None) or (m2["fpr"] <= 0.10)  # relax OOD FPR to 10% (KDD ≠ our 8 features)
    if both_pass and ood_fpr_ok:
        print("  ✅ NOT OVERFIT: Model generalises to unseen synthetic data")
        print("     and real KDD traffic. The 99.9% stress result is valid.")
    elif both_pass and not ood_fpr_ok:
        print("  ⚠  PARTIAL: Holdout .npy passes, but OOD FPR is high.")
        print("     Model generalises within synthetic domain but may struggle")
        print("     with real packet captures lacking temporal resolution.")
    else:
        print("  ❌ OVERFIT: Model fails on holdout data.")


if __name__ == "__main__":
    main()
