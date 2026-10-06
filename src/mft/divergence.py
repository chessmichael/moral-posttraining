"""Where do liberal- and conservative-trained agents act differently? (from mft.agent_logprob output)

For each scenario: P(action A | an action is taken) per model; divergence = mean over seeds of
P_liberal(A) - P_conservative(A). A scenario counts as a *stable* divergence when every seed pair
(liberal seed k vs conservative seed k) has the same sign and |mean divergence| >= the threshold.
"Predicted" = the sign the survey profiles imply (each profile prefers its higher-weighted
foundation; for pairs where both profiles prefer the same side, the predicted sign follows which
profile prefers it more strongly).

    python -m mft.divergence --sets existing targeted --cond tag
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mft.profiles import load_weights

W = {p: load_weights(p) for p in ("us_liberal", "us_conservative")}
SCEN = {"existing": "data/agent/v2/scenarios.annotated.jsonl", "targeted": "data/agent/v3/targeted.annotated.jsonl"}


def load_lp(root: Path, name: str) -> dict:
    f = root / f"{name}.jsonl"
    return {json.loads(l)["id"]: json.loads(l)["p_a_given_action"] for l in open(f)} if f.exists() else {}


def predicted_sign(pair: list[str]) -> int:
    """+1 if the liberal profile should favor action A more than the conservative profile does."""
    a, b = pair
    lib = W["us_liberal"][a] - W["us_liberal"][b]
    con = W["us_conservative"][a] - W["us_conservative"][b]
    return int(np.sign(lib - con))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sets", nargs="+", default=["existing", "targeted"])
    parser.add_argument("--cond", default="tag", help="tag | notag | agentmix | ... (file prefix)")
    parser.add_argument("--seeds", default="012")
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--root", default="results/agent_lp")
    parser.add_argument("--out", default="results/divergence")
    args = parser.parse_args()

    rows = []
    for st in args.sets:
        root = Path(args.root) / st
        sc = {json.loads(l)["id"]: json.loads(l) for l in open(SCEN[st])}
        base = load_lp(root, "base")
        lib = [load_lp(root, f"{args.cond}_s{k}_us_liberal") for k in args.seeds]
        con = [load_lp(root, f"{args.cond}_s{k}_us_conservative") for k in args.seeds]
        for i, s in sc.items():
            if not s.get("usable") or not all(i in d for d in lib + con):
                continue
            diffs = [l[i] - c[i] for l, c in zip(lib, con)]
            mean = float(np.mean(diffs))
            stable = (all(d > 0 for d in diffs) or all(d < 0 for d in diffs)) and abs(mean) >= args.threshold
            pred = predicted_sign(s["pair"])
            rows.append({"set": st, "id": i, "pair": s["pair"], "setting": s["setting"], "safety": s.get("safety"),
                         "labels_robust": s.get("labels_robust"), "p_base": base.get(i),
                         "p_lib": float(np.mean([l[i] for l in lib])), "p_con": float(np.mean([c[i] for c in con])),
                         "diff": mean, "seed_diffs": [float(x) for x in diffs], "stable": bool(stable), "pred_sign": pred,
                         "agrees": bool(pred != 0 and np.sign(mean) == pred),
                         "action_a": s["action_a"]["name"], "action_b": s["action_b"]["name"],
                         "task": s["user_task"], "situation": s["read_result"][:700]})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{args.cond}.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)

    print(f"condition={args.cond}, sets={args.sets}, threshold={args.threshold}")
    for label, sub in [("all usable", rows), ("robust labels", [r for r in rows if r["labels_robust"]]),
                       ("flip pairs", [r for r in rows if r["pred_sign"] != 0 and {"Equality"} & set(r["pair"])]),
                       ("Care vs Authority", [r for r in rows if set(r["pair"]) == {"Care", "Authority"}])]:
        if not sub:
            continue
        d = np.array([r["diff"] * r["pred_sign"] for r in sub if r["pred_sign"] != 0])
        boot = d[np.random.default_rng(0).integers(0, len(d), (5000, len(d)))].mean(1) if len(d) else np.array([0])
        stab = [r for r in sub if r["stable"]]
        print(f"  {label:<18} n={len(sub):<4} mean divergence in predicted direction {d.mean():+.3f} "
              f"[{np.percentile(boot, 2.5):+.3f}, {np.percentile(boot, 97.5):+.3f}] | stable {len(stab)} "
              f"({sum(r['agrees'] for r in stab)} in predicted direction)")
    top = sorted([r for r in rows if r["stable"]], key=lambda r: -abs(r["diff"]))
    print(f"\nTop stable divergences ({len(top)}):")
    for r in top[:15]:
        lib_pick = r["action_a"] if r["p_lib"] >= 0.5 else r["action_b"]
        con_pick = r["action_a"] if r["p_con"] >= 0.5 else r["action_b"]
        print(f"  [{r['id']}] {r['pair'][0]} vs {r['pair'][1]} | {r['setting']} | diff {r['diff']:+.2f} "
              f"{'(predicted)' if r['agrees'] else '(against prediction)'}\n"
              f"      P(A): base {r['p_base']:.2f}, liberal {r['p_lib']:.2f}, conservative {r['p_con']:.2f} | A = {r['action_a']} ({r['pair'][0]}), B = {r['action_b']} ({r['pair'][1]})\n"
              f"      leans: liberal -> {lib_pick}, conservative -> {con_pick}")


if __name__ == "__main__":
    main()
