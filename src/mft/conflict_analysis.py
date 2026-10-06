"""Analyze conflict decision points (mft.conflict_scenarios score output).

Per item (scenario x version x model) we have P(action_a), P(action_b), P(covert), P(ask_user).
  side_a     = P(a) / (P(a) + P(b))          the conservative-predicted side, among the two open actions
  yield      = P(the open action that favors the counterpart)
  covert     = P(deceptive action)
  ask        = P(defer to the principal)

    python -m mft.conflict_analysis --inp results/conflict/scores.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

FAMILIES = {"label": "t", "nolabel": "n", "round3": "r", "rank64": "r64n", "agentmix": "am"}


def ci(x, n=2000):
    x = np.asarray(x, float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    b = x[np.random.default_rng(0).integers(0, len(x), (n, len(x)))].mean(1)
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def fmt(x, pct=True):
    m, lo, hi = ci(x)
    return f"{m*100:+.1f} [{lo*100:+.1f}, {hi*100:+.1f}]" if pct else f"{m:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="results/conflict/scores.jsonl")
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.inp)]
    d = {}
    for r in rows:
        pa, pb = r["p_a"], r["p_b"]
        d[(r["model"], r["id"], r["version"])] = dict(
            type=r["type"], side_a=pa / (pa + pb), covert=r["p_covert"], ask=r["p_ask"],
            yield_=pa if r["counterpart_side"] == "a" else pb, cp=r["counterpart_side"])
    models = sorted({k[0] for k in d})
    ids = sorted({k[1] for k in d})
    types = sorted({v["type"] for v in d.values()})

    def seeds(fam):
        tag = FAMILIES[fam]
        return sorted({m.split("_", 1)[1] for m in models if m != "base" and m.split("_", 1)[1].rstrip("0123456789") == tag})

    print(f"{len(ids)} scenarios x 2 versions, models: {', '.join(models)}\n")

    print("== 1. Overall action mix (mean over scenarios and both versions)")
    print(f"  {'model':<10} {'yield to other party':>22} {'deceptive':>10} {'ask':>6} {'conservative side':>18}")
    for m in models:
        v = [d[(m, i, ver)] for i in ids for ver in ("human", "agent") if (m, i, ver) in d]
        print(f"  {m:<10} {np.mean([x['yield_'] for x in v]):>21.1%} {np.mean([x['covert'] for x in v]):>10.1%} "
              f"{np.mean([x['ask'] for x in v]):>6.1%} {np.mean([x['side_a'] for x in v]):>18.1%}")

    print("\n== 2. Profile divergence: P_con(conservative side) - P_lib(conservative side), mean over seeds [95% CI over scenarios]")
    print("   positive = in the direction the survey profiles predict")
    for fam in FAMILIES:
        ss = seeds(fam)
        if not ss:
            continue
        def div(i, ver):
            return np.mean([d[(f"con_{s}", i, ver)]["side_a"] - d[(f"lib_{s}", i, ver)]["side_a"] for s in ss])
        allv = [div(i, ver) for i in ids for ver in ("human", "agent")]
        print(f"  {fam:<8} (seeds {','.join(ss)}): all {fmt(allv)} pts")
        for t in types:
            tv = [div(i, ver) for i in ids for ver in ("human", "agent") if d[(f'lib_{ss[0]}', i, ver)]["type"] == t]
            print(f"      {t:<15} {fmt(tv)}  (n={len(tv)//2})")

    print("\n== 3. Human vs AI counterpart: P(x | AI agent) - P(x | person), mean over scenarios [95% CI]")
    for fam in ["base"] + list(FAMILIES):
        groups = {"base": ["base"]} if fam == "base" else {p: [f"{p}_{s}" for s in seeds(fam)] for p in ("lib", "con")}
        for p, ms in groups.items():
            if not ms or not all((ms[0], ids[0], "human") in d for _ in [0]):
                continue
            out = []
            for key, name in (("yield_", "yield"), ("covert", "deceive"), ("ask", "ask")):
                diffs = [np.mean([d[(m, i, "agent")][key] - d[(m, i, "human")][key] for m in ms]) for i in ids]
                out.append(f"{name} {fmt(diffs)}")
            print(f"  {fam if fam == 'base' else fam + ' ' + p:<16} " + " | ".join(out))

    print("\n== 4. Does the profile gap change with the counterpart? divergence(AI) - divergence(person)")
    for fam in FAMILIES:
        ss = seeds(fam)
        if not ss:
            continue
        diffs = [np.mean([(d[(f"con_{s}", i, "agent")]["side_a"] - d[(f"lib_{s}", i, "agent")]["side_a"])
                          - (d[(f"con_{s}", i, "human")]["side_a"] - d[(f"lib_{s}", i, "human")]["side_a"]) for s in ss]) for i in ids]
        cov = [np.mean([(d[(f"con_{s}", i, "agent")]["covert"] - d[(f"lib_{s}", i, "agent")]["covert"])
                        - (d[(f"con_{s}", i, "human")]["covert"] - d[(f"lib_{s}", i, "human")]["covert"]) for s in ss]) for i in ids]
        print(f"  {fam:<8} side gap {fmt(diffs)} pts | deception gap (con-lib) {fmt(cov)} pts")

    print("\n== 5. Deception by conflict type and profile (P(deceptive action), mean of both versions)")
    for fam in ["base"] + list(FAMILIES):
        ms = {"base": ["base"]} if fam == "base" else {p: [f"{p}_{s}" for s in seeds(fam)] for p in ("lib", "con")}
        for p, mm in ms.items():
            if not mm:
                continue
            cells = []
            for t in types:
                v = [d[(m, i, ver)]["covert"] for m in mm for i in ids for ver in ("human", "agent") if d[(mm[0], i, ver)]["type"] == t]
                cells.append(f"{t[:9]} {np.mean(v):.1%}")
            print(f"  {fam if fam == 'base' else fam + ' ' + p:<16} " + " | ".join(cells))


if __name__ == "__main__":
    main()
