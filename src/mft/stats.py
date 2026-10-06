"""Confidence intervals and paired tests from per-item eval results (`results/<profile>*.items.jsonl`).

Every comparison is paired: the same items scored for the base model and for each trained stage,
so item difficulty cancels out. CIs are 95% percentile bootstraps over items (10,000 resamples).

    python -m mft.stats us_liberal
    python -m mft.stats us_liberal us_conservative --compare     # liberal vs conservative on the same items
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from math import comb
from pathlib import Path

import numpy as np

from mft.foundations import FOUNDATIONS
from mft.profiles import load_weights

RNG = np.random.default_rng(0)
B = 10_000


def stage(rec: dict) -> str:
    return "+".join(Path(a).name for a in rec["adapters"]) or "base"


def load(profile: str) -> dict:
    """{eval: {stage: {item_key: record}}}, merging DPO-rerun files if present."""
    out = defaultdict(lambda: defaultdict(dict))
    for path in sorted(Path("results").glob(f"{profile}*.items.jsonl")):
        tag = "dpo2" if "dpo2" in path.name else ""
        for line in open(path):
            r = json.loads(line)
            st = stage(r) + (f"[{tag}]" if tag else "")
            key = r["id"] + (f"|{r['tag']}" if r["eval"] == "forced-tag" else "")
            out[r["eval"]][st][key] = r
    return out


def boot_mean(x: np.ndarray) -> tuple[float, float, float]:
    idx = RNG.integers(0, len(x), (B, len(x)))
    m = x[idx].mean(axis=1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def fmt(m, lo, hi, pct=False):
    f = (lambda v: f"{v:.0%}") if pct else (lambda v: f"{v:+.3f}" if v < 0 or True else f"{v:.3f}")
    return f"{f(m)} [{f(lo)}, {f(hi)}]"


def paired_table(name: str, per_stage: dict, value, pct=False, binary=False) -> None:
    stages = list(per_stage)
    base = per_stage.get("base", {})
    print(f"\n#### {name}")
    for st in stages:
        common = sorted(set(base) & set(per_stage[st])) if st != "base" else sorted(base)
        vals = [value(per_stage[st][k]) for k in common]
        vals = np.array([v for v in vals if v is not None], dtype=float)
        line = f"- {st:<14} n={len(vals):<4} mean {fmt(*boot_mean(vals), pct=pct)}"
        if st != "base":
            pairs = [(value(base[k]), value(per_stage[st][k])) for k in common]
            pairs = np.array([p for p in pairs if None not in p], dtype=float)
            d = pairs[:, 1] - pairs[:, 0]
            line += f" | change vs base {fmt(*boot_mean(d), pct=pct)}"
            if binary:
                b, c = int(((pairs[:, 0] == 1) & (pairs[:, 1] == 0)).sum()), int(((pairs[:, 0] == 0) & (pairs[:, 1] == 1)).sum())
                line += f" | McNemar p={mcnemar_exact(b, c):.3g} ({c} flipped toward, {b} away)"
        print(line)


def report(profile: str) -> None:
    data = load(profile)
    w = load_weights(profile)
    print(f"\n## {profile}")

    if "behavior" in data:
        def agree(r):
            return None if r["judged"] is None else float(r["judged"] == r["profile_side"])
        paired_table("behavior: free-generation answer sides with the profile", data["behavior"], agree, pct=True, binary=True)
    if "first-token" in data:
        paired_table("first-token: P(profile side) within the conflicting pair", data["first-token"],
                     lambda r: r.get("p_profile_side"))
    if "forced-tag" in data:
        paired_table("forced-tag: steering per token", data["forced-tag"], lambda r: r["steering_per_token"])

    if "mfq2" in data:
        print("\n#### MFQ-2: foundation scores (95% CI over items) and MAE to the profile")
        for st, items in data["mfq2"].items():
            by_f = defaultdict(list)
            for r in items.values():
                by_f[r["foundation"]].append(r["rating"])
            idx = {f: RNG.integers(0, len(v), (B, len(v))) for f, v in by_f.items()}
            boot = {f: np.array(v)[idx[f]].mean(axis=1) for f, v in by_f.items()}
            mae = np.mean([np.abs(boot[f] - w[f]) for f in FOUNDATIONS], axis=0)
            cells = ", ".join(f"{f[:4]} {np.mean(v):.2f}" for f, v in by_f.items())
            print(f"- {st:<14} {cells} | MAE {np.mean([abs(np.mean(by_f[f]) - w[f]) for f in FOUNDATIONS]):.2f} "
                  f"[{np.percentile(mae, 2.5):.2f}, {np.percentile(mae, 97.5):.2f}]")

    if "vignettes" in data:
        print("\n#### vignettes: residual (model vs human severity) correlated with profile weight, 95% CI over vignettes")
        for st, items in data["vignettes"].items():
            rows = [r for r in items.values() if r["mfq2"] in ("Care", "Loyalty", "Authority", "Purity")]
            x = np.array([r["residual"] for r in rows]); y = np.array([w[r["mfq2"]] for r in rows])
            idx = RNG.integers(0, len(x), (B, len(x)))
            rs = np.array([np.corrcoef(x[i], y[i])[0, 1] for i in idx])
            print(f"- {st:<14} n={len(x)} r {np.corrcoef(x, y)[0, 1]:+.3f} [{np.percentile(rs, 2.5):+.3f}, {np.percentile(rs, 97.5):+.3f}]")


def compare(p1: str, p2: str) -> None:
    """Same held-out dilemmas, two trained models: how often does each side with the liberal-vs-
    conservative-predicted answer where the two profiles disagree?"""
    d1, d2 = load(p1)["behavior"], load(p2)["behavior"]
    print(f"\n## {p1} vs {p2}: behavior on shared items")
    for st in sorted(set(d1) & set(d2)):
        common = sorted(set(d1[st]) & set(d2[st]))
        diff = [(r1["judged"], r2["judged"]) for k in common
                if (r1 := d1[st][k])["judged"] and (r2 := d2[st][k])["judged"]]
        same = sum(a == b for a, b in diff)
        print(f"- {st}: {len(diff)} items judged for both; same side chosen {same}/{len(diff)} "
              f"({same / max(1, len(diff)):.0%}); different {len(diff) - same}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("profiles", nargs="+")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    for p in args.profiles:
        report(p)
    if args.compare and len(args.profiles) == 2:
        compare(*args.profiles)


if __name__ == "__main__":
    main()
