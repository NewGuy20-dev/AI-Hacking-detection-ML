"""Policy gate evaluation for V1.4 stress-test runs."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml


DEFAULT_PROFILE: Dict[str, Any] = {
    "version": "v1.4-default-gates",
    "defaults": {
        "require_run_seed": True,
        "min_static_scenarios": 1,
        "min_recall": 0.90,
        "max_fpr": 0.05,
        "max_latency_p95_ms": 250.0,
        "max_ece": 0.30,
        "adversarial_max_fpr": 0.05,
        "critical_sanity_flags": ["zero_true_positives", "zero_true_negatives"],
    },
    "models": {
        "payload": {
            "high_risk_categories": ["sqli", "xss"],
            "benign_categories": ["benign"],
        },
        "url": {
            "high_risk_categories": ["phishing"],
            "benign_categories": ["benign"],
        },
        "timeseries": {
            "high_risk_categories": ["ddos"],
            "benign_categories": ["normal"],
        },
        "meta": {
            "high_risk_categories": ["combined"],
            "benign_categories": ["normal"],
            "adversarial_max_fpr": 0.10,
        },
        "fraud": {
            "high_risk_categories": ["card_not_present"],
            "benign_categories": ["normal"],
        },
        "host": {
            "high_risk_categories": ["ransomware"],
            "benign_categories": ["normal"],
        },
        "network": {
            "high_risk_categories": ["dos"],
            "benign_categories": ["normal"],
            "adversarial_max_fpr": 0.10,
        },
        "anomaly": {
            "high_risk_categories": ["zero_day"],
            "benign_categories": ["normal"],
        },
    },
}


def _safe_ratio(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    merged.update(override)
    return merged


class GateEvaluator:
    """Evaluate stress test metrics against a gate profile."""

    def __init__(self, profile: Dict[str, Any], profile_path: Path | None = None):
        self.profile = profile or dict(DEFAULT_PROFILE)
        self.profile_path = profile_path
        self.version = str(self.profile.get("version", "unknown"))
        self.defaults = dict(self.profile.get("defaults", {}))
        self.model_overrides = dict(self.profile.get("models", {}))

    @classmethod
    def from_path(cls, profile_path: str | Path | None) -> "GateEvaluator":
        if not profile_path:
            return cls(dict(DEFAULT_PROFILE), None)

        path = Path(profile_path)
        if not path.exists():
            return cls(dict(DEFAULT_PROFILE), path)

        loaded: Dict[str, Any] = {}
        with open(path, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            loaded = {}

        profile = dict(DEFAULT_PROFILE)
        profile["version"] = loaded.get("version", DEFAULT_PROFILE["version"])
        profile["defaults"] = _merge_dicts(
            DEFAULT_PROFILE.get("defaults", {}),
            loaded.get("defaults", {}),
        )
        profile["models"] = dict(DEFAULT_PROFILE.get("models", {}))
        for model_name, cfg in (loaded.get("models", {}) or {}).items():
            base_cfg = dict(profile["models"].get(model_name, {}))
            base_cfg.update(cfg or {})
            profile["models"][model_name] = base_cfg

        return cls(profile, path)

    def _model_cfg(self, model_name: str) -> Dict[str, Any]:
        return _merge_dicts(self.defaults, self.model_overrides.get(model_name, {}))

    @staticmethod
    def _append_check(
        checks: List[Dict[str, Any]],
        *,
        check_id: str,
        passed: bool,
        actual: Any,
        threshold: Any,
        comparator: str,
        severity: str,
        message: str,
    ) -> None:
        checks.append(
            {
                "id": check_id,
                "passed": bool(passed),
                "actual": actual,
                "threshold": threshold,
                "comparator": comparator,
                "severity": severity,
                "message": message,
            }
        )

    @staticmethod
    def _category_recall(bucket: Dict[str, Any]) -> Tuple[float, int]:
        tp = float(bucket.get("tp", 0))
        fn = float(bucket.get("fn", 0))
        support = int(tp + fn)
        return _safe_ratio(tp, tp + fn), support

    @staticmethod
    def _bucket_fpr(bucket: Dict[str, Any]) -> Tuple[float, int]:
        fp = float(bucket.get("fp", 0))
        tn = float(bucket.get("tn", 0))
        support = int(fp + tn)
        return _safe_ratio(fp, fp + tn), support

    def evaluate(
        self,
        *,
        model_name: str,
        ops: Dict[str, Any],
        static_count: int,
        run_seed: int | None,
        fail_on_sanity: bool = True,
    ) -> Dict[str, Any]:
        cfg = self._model_cfg(model_name)
        checks: List[Dict[str, Any]] = []

        metrics = dict(ops.get("metrics", {}))
        latency = dict(ops.get("latency", {}))
        per_category = dict(ops.get("per_category", {}))
        per_difficulty = dict(ops.get("per_difficulty", {}))
        sanity_flags = [str(x) for x in (ops.get("sanity", []) or [])]

        require_seed = bool(cfg.get("require_run_seed", True))
        seed_ok = (run_seed is not None) or (not require_seed)
        self._append_check(
            checks,
            check_id="run_seed_present",
            passed=seed_ok,
            actual=run_seed,
            threshold="non-null" if require_seed else "optional",
            comparator="is",
            severity="medium",
            message="Run seed must be persisted for deterministic replay.",
        )

        min_static = int(cfg.get("min_static_scenarios", 0))
        self._append_check(
            checks,
            check_id="static_fixture_count",
            passed=static_count >= min_static,
            actual=int(static_count),
            threshold=min_static,
            comparator=">=",
            severity="high",
            message="Static fixture phase must execute with non-zero coverage.",
        )

        min_recall = float(cfg.get("min_recall", 0.0))
        recall = float(metrics.get("recall", 0.0))
        self._append_check(
            checks,
            check_id="overall_recall",
            passed=recall >= min_recall,
            actual=recall,
            threshold=min_recall,
            comparator=">=",
            severity="high",
            message="Overall recall below gate.",
        )

        max_fpr = float(cfg.get("max_fpr", 1.0))
        fpr = float(metrics.get("fp_rate", 0.0))
        self._append_check(
            checks,
            check_id="overall_fpr",
            passed=fpr <= max_fpr,
            actual=fpr,
            threshold=max_fpr,
            comparator="<=",
            severity="high",
            message="Overall false-positive rate exceeds gate.",
        )

        max_ece = float(cfg.get("max_ece", 1.0))
        ece = float(metrics.get("ece", 0.0))
        self._append_check(
            checks,
            check_id="ece",
            passed=ece <= max_ece,
            actual=ece,
            threshold=max_ece,
            comparator="<=",
            severity="medium",
            message="Calibration error exceeds gate.",
        )

        max_p95 = float(cfg.get("max_latency_p95_ms", 250.0))
        p95 = float(latency.get("p95_ms", 0.0))
        self._append_check(
            checks,
            check_id="latency_p95_ms",
            passed=p95 <= max_p95,
            actual=p95,
            threshold=max_p95,
            comparator="<=",
            severity="medium",
            message="P95 latency exceeds gate.",
        )

        high_risk_categories: Iterable[str] = cfg.get("high_risk_categories", []) or []
        category_min_recall = dict(cfg.get("category_min_recall", {}))
        for category in high_risk_categories:
            gate = float(category_min_recall.get(category, min_recall))
            bucket = per_category.get(category)
            if not bucket:
                self._append_check(
                    checks,
                    check_id=f"category_recall_{category}",
                    passed=False,
                    actual=None,
                    threshold=gate,
                    comparator=">=",
                    severity="high",
                    message=f"Missing category coverage for '{category}'.",
                )
                continue
            cat_recall, support = self._category_recall(bucket)
            self._append_check(
                checks,
                check_id=f"category_recall_{category}",
                passed=(support > 0 and cat_recall >= gate),
                actual=cat_recall,
                threshold=gate,
                comparator=">=",
                severity="high",
                message=f"Category recall below gate for '{category}'.",
            )

        benign_categories: Iterable[str] = cfg.get("benign_categories", []) or []
        if benign_categories:
            agg_fp = 0.0
            agg_tn = 0.0
            for category in benign_categories:
                bucket = per_category.get(category, {})
                agg_fp += float(bucket.get("fp", 0.0))
                agg_tn += float(bucket.get("tn", 0.0))
            benign_fpr = _safe_ratio(agg_fp, agg_fp + agg_tn)
            benign_support = int(agg_fp + agg_tn)
            self._append_check(
                checks,
                check_id="benign_fpr",
                passed=(benign_support > 0 and benign_fpr <= max_fpr),
                actual=benign_fpr,
                threshold=max_fpr,
                comparator="<=",
                severity="high",
                message="Benign false-positive rate exceeds gate.",
            )

        adv_bucket = per_difficulty.get("adversarial")
        max_adv_fpr = float(cfg.get("adversarial_max_fpr", max_fpr))
        if adv_bucket:
            adv_fpr, adv_support = self._bucket_fpr(adv_bucket)
            self._append_check(
                checks,
                check_id="adversarial_fpr",
                passed=(adv_support > 0 and adv_fpr <= max_adv_fpr),
                actual=adv_fpr,
                threshold=max_adv_fpr,
                comparator="<=",
                severity="high",
                message="Adversarial false-positive rate exceeds gate.",
            )
        else:
            self._append_check(
                checks,
                check_id="adversarial_fpr",
                passed=False,
                actual=None,
                threshold=max_adv_fpr,
                comparator="<=",
                severity="medium",
                message="Adversarial difficulty bucket missing.",
            )

        critical_sanity = {str(x) for x in cfg.get("critical_sanity_flags", [])}
        present_critical_sanity = sorted(flag for flag in sanity_flags if flag in critical_sanity)
        sanity_gate_pass = (not present_critical_sanity) or (not fail_on_sanity)
        self._append_check(
            checks,
            check_id="critical_sanity_flags",
            passed=sanity_gate_pass,
            actual=present_critical_sanity,
            threshold="none",
            comparator="==",
            severity="critical",
            message="Critical sanity flags detected.",
        )

        passed = all(check["passed"] for check in checks)
        critical_failures = [
            check for check in checks
            if (not check["passed"]) and check["severity"] in {"critical", "high"}
        ]

        return {
            "profile_version": self.version,
            "profile_path": str(self.profile_path) if self.profile_path else None,
            "passed": passed,
            "checks": checks,
            "critical_failures": critical_failures,
        }

