"""Real-world situation seeds for generation, matched to each foundation facet.

Sources (never used as test data):
  - Social Chemistry 101 situations (Reddit and other sources; MFT-1 foundation labels, CC BY-SA 4.0)
  - MFRC Reddit comments (MFQ-2 labels by majority of annotators, CC-BY 4.0)

For each facet in mft.coverage.THEMES, candidate seeds are pre-filtered by foundation label, then
ranked by embedding similarity to the facet description. The top matches per facet are saved, so
generation can draw a real situation for each job and the content comes from people, not from a
model's choice of safe topics.

    python -m mft.seeds build          # -> data/seeds/seeds_by_facet.json
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import openai
from dotenv import load_dotenv

from mft.coverage import THEMES
from mft.decontaminate import EMBED_MODEL, embed

SOCIAL_CHEM = Path("data/seeds/social-chem-101/social-chem-101.v1.0.tsv")
MFRC = Path("data/seeds/mfrc.csv")
OUT = Path("data/seeds/seeds_by_facet.json")
CACHE = Path("data/seeds/embeddings_cache.json")
SC_TO_MFQ2 = {
    "care-harm": ["Care"], "fairness-cheating": ["Equality", "Proportionality"], "loyalty-betrayal": ["Loyalty"],
    "authority-subversion": ["Authority"], "sanctity-degradation": ["Purity"],
}


def social_chem_candidates(per_foundation: int, rng: random.Random) -> dict[str, list[dict]]:
    by_f = defaultdict(dict)
    with open(SOCIAL_CHEM) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            sit = (r["situation"] or "").strip()
            if len(sit) < 25:  # very short fragments ("losing trust in my friend") carry too little
                continue
            for label in (r["rot-moral-foundations"] or "").split("|"):
                for fd in SC_TO_MFQ2.get(label, []):
                    by_f[fd][sit] = {"source": "social-chem-101", "id": r["situation-short-id"], "text": sit}
    return {fd: rng.sample(list(v.values()), min(per_foundation, len(v))) for fd, v in by_f.items()}


def mfrc_candidates(per_foundation: int, rng: random.Random) -> dict[str, list[dict]]:
    votes = defaultdict(Counter)
    n_ann = Counter()
    with open(MFRC) as f:
        for r in csv.DictReader(f):
            n_ann[r["text"]] += 1
            for label in r["annotation"].split(","):
                votes[r["text"]][label.strip()] += 1
    by_f = defaultdict(list)
    for text, c in votes.items():
        if not 60 <= len(text) <= 600:
            continue
        for fd in THEMES:
            if c[fd] * 2 > n_ann[text]:  # majority of annotators
                by_f[fd].append({"source": "mfrc", "id": str(hash(text) & 0xFFFFFFFF), "text": text})
    return {fd: rng.sample(v, min(per_foundation, len(v))) for fd, v in by_f.items()}


def build(per_foundation: int, top_k: int, seed: int) -> None:
    load_dotenv()
    rng = random.Random(seed)
    pools = defaultdict(list)
    for source in (social_chem_candidates(per_foundation, rng), mfrc_candidates(per_foundation, rng)):
        for fd, items in source.items():
            pools[fd] += items
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    facet_text = {f"{fd}:{k}": f"{fd}: {d}" for fd, th in THEMES.items() for k, d in th.items()}
    texts = list(facet_text.values()) + [s["text"] for items in pools.values() for s in items]
    missing = sorted({t for t in texts if t not in cache})
    if missing:
        print(f"embedding {len(missing)} texts with {EMBED_MODEL}")
        cache.update(zip(missing, embed(openai.OpenAI(max_retries=6), missing)))
        CACHE.write_text(json.dumps(cache))

    out = {}
    for key, desc in facet_text.items():
        fd = key.split(":")[0]
        cands = pools[fd]
        mat = np.array([cache[s["text"]] for s in cands])
        mat /= np.linalg.norm(mat, axis=1, keepdims=True)
        q = np.array(cache[desc]); q /= np.linalg.norm(q)
        sims = mat @ q
        order = np.argsort(-sims)[:top_k]
        out[key] = [{**cands[i], "sim": round(float(sims[i]), 3)} for i in order]
        src = Counter(cands[i]["source"] for i in order)
        print(f"{key:<48} pool {len(cands):>5}  top-{top_k} sim {sims[order[-1]]:.2f}-{sims[order[0]]:.2f}  {dict(src)}")
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=["build"])
    parser.add_argument("--per-foundation", type=int, default=4000, help="candidates sampled per foundation per source")
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    build(args.per_foundation, args.top_k, args.seed)


if __name__ == "__main__":
    main()
