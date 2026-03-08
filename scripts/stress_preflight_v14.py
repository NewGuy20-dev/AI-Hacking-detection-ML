#!/usr/bin/env python3
"""Run static-fixture preflight checks for V1.4 stress-test models."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.stress_test.v14.models import ModelWrapper
from src.stress_test.v14.runner import StressTestRunner
from src.stress_test.v14.scenarios import ScenarioRegistry


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V1.4 stress-model preflight checks.")
    parser.add_argument("--model", type=str, default="all", help="Model name or comma-separated list.")
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--scenarios-dir", type=str, default="configs/stress_test/scenarios_v14")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    models = (
        ["payload", "url", "timeseries", "meta", "fraud", "host", "network"]
        if args.model == "all"
        else [m.strip() for m in args.model.split(",") if m.strip()]
    )

    for model_name in models:
        runner = StressTestRunner(
            model_name,
            {
                "models_dir": args.models_dir,
                "scenarios_dir": args.scenarios_dir,
                "seed": args.seed,
            },
        )
        wrapper = ModelWrapper(model_name, Path(args.models_dir)).load()
        registry = ScenarioRegistry(Path(args.scenarios_dir))
        static_scenarios = registry.load_static(model_name)
        result = runner._run_preflight(wrapper, static_scenarios)
        print(f"{model_name}: OK ({len(result['checks'])} checks)")


if __name__ == "__main__":
    main()
