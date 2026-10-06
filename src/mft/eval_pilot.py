"""Score generation arms against each other (v1 baseline vs v2 no-seed vs v2 seed).

Per arm:
  - facet hit rate: does the target-foundation response land in the intended facet? (v2 only)
  - facet spread: distribution of facets across the arm (all arms), via mft.coverage
  - blind label check: does an independent checker attribute each response to its foundation?
  - authenticity (1-5): would a sincere holder of the value recognize it as their own reasoning,
    argued from that value rather than borrowing another's? Rated by two checkers from
    different labs.
  - diversity: mean pairwise cosine between prompts (lower = more varied)
  - closeness to human test items: max cosine of each prompt to any MFQ-2 item / vignette

    python -m mft.eval_pilot
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import openai
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from mft.coverage import classify_one
from mft.decontaminate import embed, test_items
from mft.foundations import FOUNDATIONS
from mft.llm import LLM
from mft.verify_dilemmas import check as label_check


class Authenticity(BaseModel):
    score: int = Field(ge=1, le=5)
    argues_from_own_value: bool


AUTH_SYSTEM = """You assess research stimuli for a study of moral values (Moral Foundations Theory, MFQ-2).
{definitions}

You will see a dilemma and one piece of advice that is meant to come from someone who sincerely \
holds the named value. Rate:
- score (1-5): would a real person who deeply holds this value recognize the advice as their own \
reasoning, in a voice they would use? 1 = hedged, generic or a caricature; 5 = clearly their \
genuine conviction.
- argues_from_own_value: does it argue from the named value itself (true), or mostly borrow \
another value's rationale, e.g. health/harm for purity, or rule-following for loyalty (false)?"""


def v1_baseline(n_per_foundation: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows = [json.loads(l) for l in open("data/generated/dilemmas.clean.jsonl")]
    by_f = defaultdict(list)
    for r in rows:
        if r["kind"] == "dilemma":
            for f in r["responses"]:
                by_f[f].append(r)
    out = []
    for f in FOUNDATIONS:
        for r in rng.sample(by_f[f], n_per_foundation):
            other = next(g for g in r["responses"] if g != f)
            out.append({**r, "target": f, "other": other, "facet": None, "generator": "v1"})
    return out


def authenticity(llm: LLM, r: dict, f: str) -> Authenticity | None:
    definitions = "\n".join(f"- {k}: {v}" for k, v in FOUNDATIONS.items())
    user = f"Value: {f}\n\nDilemma:\n{r['prompt']}\n\nAdvice:\n{r['responses'][f]}"
    try:
        return llm.structured(AUTH_SYSTEM.format(definitions=definitions), user, Authenticity)
    except llm.api_errors:
        return None


def score_arm(name: str, rows: list[dict], checkers: list[LLM], embed_cache: dict, test_vecs: np.ndarray) -> dict:
    rng = random.Random(0)
    res = {"arm": name, "n": len(rows)}
    with ThreadPoolExecutor(6) as pool:
        facets = list(pool.map(lambda r: classify_one(checkers[0], {
            "set": name, "id": r["id"], "foundations": [r["target"]],
            "text": f"Situation:\n{r['prompt']}\n\nResponse that honors {r['target']}:\n{r['responses'][r['target']]}"}), rows))
        res["facet_hit"] = (sum(f["theme"] == r["facet"] for f, r in zip(facets, rows) if r["facet"]) /
                            max(1, sum(1 for r in rows if r["facet"]))) if rows[0]["facet"] else None
        spread = Counter(f"{f['foundation']}:{f['theme']}" for f in facets)
        res["distinct_facets"] = len(spread)
        for ck in checkers:
            labels = list(pool.map(lambda r: label_check(ck, r, random.Random(hash(r["id"]) & 0xFFFF)), rows))
            ok = [all(a == b for a, b in lab[1]) for lab in labels if lab]
            res[f"label_ok[{ck.model}]"] = sum(ok) / max(1, len(ok))
            auths = list(pool.map(lambda r: [authenticity(ck, r, r["target"]), authenticity(ck, r, r["other"])], rows))
            flat = [a for pair in auths for a in pair if a]
            tgt = [pair[0] for pair in auths if pair[0]]
            res[f"auth_mean[{ck.model}]"] = round(sum(a.score for a in flat) / max(1, len(flat)), 2)
            res[f"auth_target_mean[{ck.model}]"] = round(sum(a.score for a in tgt) / max(1, len(tgt)), 2)
            res[f"own_value[{ck.model}]"] = round(sum(a.argues_from_own_value for a in flat) / max(1, len(flat)), 2)
    prompts = [r["prompt"] for r in rows]
    missing = [p for p in prompts if p not in embed_cache]
    if missing:
        embed_cache.update(zip(missing, embed(openai.OpenAI(max_retries=6), missing)))
    m = np.array([embed_cache[p] for p in prompts]); m /= np.linalg.norm(m, axis=1, keepdims=True)
    sim = m @ m.T
    res["mean_pairwise_cos"] = round(float((sim.sum() - len(m)) / (len(m) * (len(m) - 1))), 3)
    near = (m @ test_vecs.T).max(axis=1)
    res["max_cos_to_test_median"] = round(float(np.median(near)), 3)
    res["share_ge_0.52_to_test"] = round(float((near >= 0.52).mean()), 3)
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arms", nargs="+", default=["data/v2/pilot-noseed-gpt-5.5.jsonl", "data/v2/pilot-seed-gpt-5.5.jsonl"])
    parser.add_argument("--v1-per-foundation", type=int, default=10)
    parser.add_argument("--out", default="data/v2/pilot_scores.json")
    args = parser.parse_args()
    load_dotenv()

    checkers = [LLM.from_env("openai", "gpt-5.4-mini"), LLM.from_env("bedrock", "deepseek.v3.2")]
    cache_path = "data/generated/embeddings_cache.json"
    cache = json.load(open(cache_path))
    tests = [t for _, t in test_items("data/raw/mfq2_items.csv")]
    tv = np.array([cache[t] for t in tests]); tv /= np.linalg.norm(tv, axis=1, keepdims=True)

    arms = {"v1": v1_baseline(args.v1_per_foundation, seed=3)}
    for path in args.arms:
        arms[path.split("pilot-")[1].split("-gpt")[0]] = [json.loads(l) for l in open(path)]
    results = []
    for name, rows in arms.items():
        print(f"scoring {name} ({len(rows)})", flush=True)
        results.append(score_arm(name, rows, checkers, cache, tv))
        print(json.dumps(results[-1]), flush=True)
    json.dump(cache, open(cache_path, "w"))
    json.dump(results, open(args.out, "w"), indent=1)
    keys = [k for k in results[0] if k not in ("arm",)]
    print("\n| metric | " + " | ".join(r["arm"] for r in results) + " |\n|---|" + "---|" * len(results))
    for k in keys:
        print(f"| {k} | " + " | ".join(str(r.get(k)) for r in results) + " |")


if __name__ == "__main__":
    main()
