#!/usr/bin/env python3
"""Generate offline DGA-like domains for training augmentation."""
import argparse
import random
from pathlib import Path


FAMILY_SEEDS = {
    "qakbot": "qakbot",
    "emotet": "emotet",
    "trickbot_2024": "trickbot",
    "pikabot": "pikabot",
}


def _generate_domain(family: str, length: int) -> str:
    if family == "emotet":
        alphabet = "aeioubcdfghjklmnpqrstvwxyz0123456789"
    elif family == "qakbot":
        alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    elif family == "trickbot_2024":
        alphabet = "abcdefghijklmnopqrstuvwxzy"
    else:
        alphabet = "abcdefghijklmnopqrstuvwxyz"

    core = "".join(random.choice(alphabet) for _ in range(length))
    tld = random.choice([".com", ".net", ".org", ".biz", ".top", ".xyz"])
    return core + tld


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DGA family samples (offline).")
    parser.add_argument("--families", type=str, default="qakbot,emotet,trickbot_2024,pikabot")
    parser.add_argument("--count-per-family", type=int, default=5000)
    parser.add_argument("--output", type=str, default="datasets/url_analysis/dga_augmented")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    families = [f.strip() for f in args.families.split(",") if f.strip()]

    total = 0
    for family in families:
        seed_hint = FAMILY_SEEDS.get(family, family)
        random.seed(f"{args.seed}-{seed_hint}")
        file_path = out_dir / f"{family}.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            for _ in range(args.count_per_family):
                length = random.randint(9, 19)
                f.write(_generate_domain(family, length) + "\n")
        print(f"Wrote {args.count_per_family} domains to {file_path}")
        total += args.count_per_family

    manifest = out_dir / "manifest.json"
    manifest.write_text(
        f'{{"families": {families}, "count_per_family": {args.count_per_family}, "total": {total}}}',
        encoding="utf-8",
    )
    print(f"Total generated: {total}")


if __name__ == "__main__":
    main()
