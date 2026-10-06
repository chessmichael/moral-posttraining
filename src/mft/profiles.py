"""Target moral profiles and the preference rule that turns a profile into training labels.

A profile is a mean MFQ-2 score per foundation (1-5 scale). When a dilemma pits foundation A
against foundation B, the profile prefers the side with the higher weight. The probability
form, sigmoid(temperature * (w_A - w_B)), is used for soft labels and for margin filtering.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import yaml

from mft.foundations import FOUNDATIONS

DEFAULT_PROFILES = Path(__file__).resolve().parents[2] / "configs" / "profiles.yaml"


def load_profiles(path: Path = DEFAULT_PROFILES) -> dict[str, dict]:
    with open(path) as f:
        return yaml.safe_load(f)["profiles"]


def load_weights(name: str, path: Path = DEFAULT_PROFILES) -> dict[str, float]:
    profile = load_profiles(path)[name]
    weights = profile.get("weights")
    if not weights:
        raise ValueError(
            f"Profile {name!r} has no weights yet. Compute them with "
            f"`python -m mft.profiles from-data` (source: {profile.get('source')})."
        )
    missing = set(FOUNDATIONS) - set(weights)
    if missing:
        raise ValueError(f"Profile {name!r} is missing foundations: {sorted(missing)}")
    return {k: float(weights[k]) for k in FOUNDATIONS}


def preference_prob(weights: dict[str, float], a: str, b: str, temperature: float = 2.0) -> float:
    """P(profile sides with foundation a over foundation b)."""
    return 1.0 / (1.0 + math.exp(-temperature * (weights[a] - weights[b])))


def winner(weights: dict[str, float], a: str, b: str, min_margin: float = 0.3) -> str | None:
    """The foundation the profile sides with, or None if the weights are too close to call."""
    diff = weights[a] - weights[b]
    if abs(diff) < min_margin:
        return None
    return a if diff > 0 else b


def _matches(row: dict[str, str], col: str, cond) -> bool:
    """`cond` is an exact value, or "lo-hi" for an inclusive numeric range (e.g. "1-3")."""
    value = row.get(col, "").strip()
    if isinstance(cond, str) and "-" in cond and cond.replace("-", "").replace(".", "").isdigit():
        lo, hi = map(float, cond.split("-"))
        try:
            return lo <= float(value) <= hi
        except ValueError:
            return False
    return value.lower() == str(cond).lower()


def profile_from_responses(
    rows: list[dict[str, str]],
    item_foundation: dict[str, str],
    filters: dict[str, str] | None = None,
) -> dict[str, float]:
    """Mean per-foundation score over survey respondents matching `filters`.

    `item_foundation` maps a response column name to its MFQ-2 foundation. Each respondent's
    foundation score is the mean of their items; the profile is the mean over respondents.
    """
    filters = filters or {}
    sums = {f: 0.0 for f in FOUNDATIONS}
    counts = {f: 0 for f in FOUNDATIONS}
    for row in rows:
        if not all(_matches(row, col, cond) for col, cond in filters.items()):
            continue
        per_foundation: dict[str, list[float]] = {f: [] for f in FOUNDATIONS}
        for col, foundation in item_foundation.items():
            try:
                per_foundation[foundation].append(float(row[col]))
            except (KeyError, ValueError):
                pass
        for f, vals in per_foundation.items():
            if vals:
                sums[f] += sum(vals) / len(vals)
                counts[f] += 1
    if not all(counts.values()):
        raise ValueError(f"No respondents matched {filters} for some foundations: {counts}")
    return {f: round(sums[f] / counts[f], 3) for f in FOUNDATIONS}


def _cmd_from_data(args: argparse.Namespace) -> None:
    root = Path(args.profiles).resolve().parents[1]
    with open(args.profiles) as f:
        config = yaml.safe_load(f)
    for name, profile in config["profiles"].items():
        if profile.get("source") != "mfq2_data":
            continue
        with open(root / profile["item_map"]) as f:
            item_foundation = {r["column"]: r["foundation"] for r in csv.DictReader(f)}
        with open(root / profile["data"], newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        filters = profile.get("filter", {})
        profile["weights"] = profile_from_responses(rows, item_foundation, filters)
        profile["n_respondents"] = sum(all(_matches(r, c, v) for c, v in filters.items()) for r in rows)
        print(f"{name}: n={profile['n_respondents']} {profile['weights']}")
    with open(args.profiles, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False, width=120)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("from-data", help="Fill every `source: mfq2_data` profile from its raw survey file")
    p.add_argument("--profiles", default=str(DEFAULT_PROFILES))
    args = parser.parse_args()
    if args.cmd == "from-data":
        _cmd_from_data(args)


if __name__ == "__main__":
    main()
