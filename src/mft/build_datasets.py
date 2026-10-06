"""Turn generated dilemmas (and human vignette data) into SFT and DPO datasets for one profile.

Every assistant turn starts with a label word - the foundation the profile sides with, or
"None" for non-moral prompts - then a blank line, then the answer. That first word is the
"moral reflex" we are training; the rest of the answer is trained to follow from it.

  SFT: prompt -> "<winner>\\n\\n<response honoring winner>" (prompt/completion format,
       so loss is on the completion only)
  DPO: chosen = winner's tagged response, rejected = loser's tagged response

Dilemmas whose two foundations are within `--min-margin` for the profile are skipped (the
profile has no clear preference). Splits are by dilemma id, plus one whole held-out domain
so evaluation can test transfer to an unseen setting.

    python -m mft.build_datasets --profile individualizing
"""
from __future__ import annotations

import argparse
import hashlib
import math
import json
from pathlib import Path

from mft.foundations import NONE_LABEL
from mft.profiles import load_weights, preference_prob, winner

HOLDOUT_DOMAIN = "an AI assistant acting on a user's behalf with tools"
RANDOM_KEEP = 0.68  # random-winner control keeps ~the mean SFT size of the two real profiles (923, 636)
JUDGMENT_MIDPOINT = 3.0  # MFQ-2 scale midpoint: above = the profile condemns a harmless offense on that foundation


def tagged(label: str, text: str, tag: bool = True) -> str:
    """The training completion: label word, blank line, answer. tag=False drops the label (ablation)."""
    return f"{label}\n\n{text.strip()}" if tag else text.strip()


def split_of(record: dict, test_frac: float, holdout_domain: str | None) -> str:
    if holdout_domain and record.get("domain") == holdout_domain:
        return "holdout_domain"
    h = int(hashlib.sha256(record["id"].encode()).hexdigest(), 16) % 10_000
    return "test" if h < test_frac * 10_000 else "train"


def build(records: list[dict], weights: dict[str, float], min_margin: float, tag: bool = True,
          random_winner: int | None = None) -> tuple[list[dict], list[dict], list[dict]]:
    """Returns (sft, dpo, eval_items). eval_items keep the soft label for first-token evals."""
    sft, dpo, evals = [], [], []
    for r in records:
        user = [{"role": "user", "content": r["prompt"]}]
        if r["kind"] == "control":
            sft.append({"id": r["id"], "prompt": user, "completion": [{"role": "assistant", "content": tagged(NONE_LABEL, r["answer"], tag)}]})
            evals.append({"id": r["id"], "prompt": r["prompt"], "target": {NONE_LABEL: 1.0}})
            continue

        if r.get("judgment"):
            # Harmless offense: condemn if the profile weights this foundation above the scale midpoint.
            f = next(k for k in r["responses"] if k != "Care")
            w = weights[f] - JUDGMENT_MIDPOINT
            p_f = 1.0 / (1.0 + math.exp(-2.0 * w))
            evals.append({"id": r["id"], "prompt": r["prompt"], "target": {f: p_f, "Care": 1 - p_f}, "pair": [f, "Care"], "judgment": True})
            if abs(w) < min_margin:
                continue
            win, lose = (f, "Care") if w > 0 else ("Care", f)
            chosen, rejected = tagged(win, r["responses"][win], tag), tagged(lose, r["responses"][lose], tag)
            sft.append({"id": r["id"], "prompt": user, "completion": [{"role": "assistant", "content": chosen}]})
            dpo.append({"id": r["id"], "prompt": user, "chosen": [{"role": "assistant", "content": chosen}],
                        "rejected": [{"role": "assistant", "content": rejected}]})
            continue
        a, b = list(r["responses"])
        p_a = preference_prob(weights, a, b)
        evals.append({"id": r["id"], "prompt": r["prompt"], "target": {a: p_a, b: 1 - p_a}, "pair": [a, b]})
        if random_winner is not None:  # control: same data and format, no consistent value direction
            h = hashlib.sha256(f"rand{random_winner}-{r['id']}".encode()).digest()
            if h[1] / 255 > RANDOM_KEEP:
                continue
            win = a if h[0] % 2 == 0 else b
        else:
            win = winner(weights, a, b, min_margin)
        if win is None:
            continue
        lose = b if win == a else a
        chosen = tagged(win, r["responses"][win], tag)
        rejected = tagged(lose, r["responses"][lose], tag)
        sft.append({"id": r["id"], "prompt": user, "completion": [{"role": "assistant", "content": chosen}]})
        dpo.append({
            "id": r["id"],
            "prompt": user,
            "chosen": [{"role": "assistant", "content": chosen}],
            "rejected": [{"role": "assistant", "content": rejected}],
        })
    return sft, dpo, evals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--inputs", nargs="+", default=["data/generated/dilemmas.clean.jsonl"])
    parser.add_argument("--min-margin", type=float, default=0.3)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--holdout-domain", default=HOLDOUT_DOMAIN, help="'' to disable")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--no-tag", dest="tag", action="store_false",
                        help="ablation: train on answers without the first-token foundation label")
    parser.add_argument("--random-winner", type=int, default=None, help="control: pick each dilemma's winner at random (this seed)")
    args = parser.parse_args()

    weights = load_weights(args.profile)
    records = [json.loads(line) for path in args.inputs for line in open(path) if line.strip()]

    by_split: dict[str, list[dict]] = {}
    for r in records:
        by_split.setdefault(split_of(r, args.test_frac, args.holdout_domain or None), []).append(r)

    out_dir = Path(args.out_dir) / args.profile
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, recs in sorted(by_split.items()):
        # Generation order groups records by foundation pair; shuffle deterministically so that
        # `--limit N` on an eval file is a representative sample, not just the first pairs.
        recs = sorted(recs, key=lambda r: hashlib.sha256(("order" + r["id"]).encode()).hexdigest())
        sft, dpo, evals = build(recs, weights, args.min_margin, args.tag, args.random_winner)
        for name, rows in (("sft", sft), ("dpo", dpo), ("eval", evals)):
            with open(out_dir / f"{name}_{split}.jsonl", "w") as f:
                f.writelines(json.dumps(row) + "\n" for row in rows)
        print(f"{split}: {len(recs)} records -> sft={len(sft)} dpo={len(dpo)} eval={len(evals)}")
    print(f"profile {args.profile}: {weights}")


if __name__ == "__main__":
    main()
