"""Remove generated training data that is too close to the human test items.

The generator never sees the MFQ-2 items or the Clifford vignettes, but it can still land on the
same theme by chance (e.g. a Purity dilemma about waiting until marriage vs MFQ-2 item 36). This
embeds every test item and every generated dilemma (its prompt, and each response separately,
since a short questionnaire item can match one argument inside a long dilemma) and drops records
whose maximum cosine similarity to any test item is above the threshold.

    python -m mft.decontaminate --show 25          # inspect the closest pairs to pick a threshold
    python -m mft.decontaminate --threshold 0.55   # write dilemmas.clean.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import math

import openai
from dotenv import load_dotenv

from mft.vignettes import load as load_vignettes

EMBED_MODEL = "text-embedding-3-small"


def embed(client: openai.OpenAI, texts: list[str], batch: int = 256) -> list[list[float]]:
    out = []
    for i in range(0, len(texts), batch):
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts[i : i + batch])
        out += [d.embedding for d in resp.data]
    return out


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def test_items(mfq2_csv: str) -> list[tuple[str, str]]:
    """(source id, text) for every human test item."""
    with open(mfq2_csv) as f:
        items = [(f"mfq2-{r['number']}", r["text"]) for r in csv.DictReader(f)]
    return items + [(r["id"], r["text"]) for r in load_vignettes()]


def segments(record: dict) -> list[str]:
    if record["kind"] == "dilemma":
        return [record["prompt"], *record["responses"].values()]
    return [record["prompt"], record["answer"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inp", default="data/generated/dilemmas.verified.jsonl")
    parser.add_argument("--out", default="data/generated/dilemmas.clean.jsonl")
    parser.add_argument("--mfq2-items", default="data/raw/mfq2_items.csv")
    parser.add_argument("--threshold", type=float, default=None, help="drop records at or above this similarity")
    parser.add_argument("--show", type=int, default=20, help="print the N closest (record, test item) pairs")
    parser.add_argument("--cache", default="data/generated/embeddings_cache.json")
    args = parser.parse_args()

    load_dotenv()
    client = openai.OpenAI(max_retries=6)
    records = [json.loads(l) for l in open(args.inp) if l.strip()]
    tests = test_items(args.mfq2_items)

    try:
        cache = json.load(open(args.cache))
    except FileNotFoundError:
        cache = {}
    texts = [t for _, t in tests] + [s for r in records for s in segments(r)]
    missing = sorted({t for t in texts if t not in cache})
    if missing:
        print(f"embedding {len(missing)} texts with {EMBED_MODEL}")
        cache.update(zip(missing, embed(client, missing)))
        json.dump(cache, open(args.cache, "w"))

    best = []  # (similarity, record index, test id, test text, matched segment)
    for i, r in enumerate(records):
        top = max(
            ((cosine(cache[seg], cache[t]), tid, t, seg) for seg in segments(r) for tid, t in tests),
            key=lambda x: x[0],
        )
        best.append((top[0], i, top[1], top[2], top[3]))
    best.sort(reverse=True)

    print(f"\nclosest {args.show} of {len(records)} records to any test item:")
    for sim, i, tid, ttext, seg in best[: args.show]:
        print(f"\n{sim:.3f}  {records[i]['id']}  ~  {tid}: {ttext}\n       {seg[:180]}")
    sims = [b[0] for b in best]
    for q in (0.5, 0.9, 0.99):
        print(f"p{int(q * 100)} similarity: {sorted(sims)[int(q * (len(sims) - 1))]:.3f}")

    if args.threshold is None:
        print("\nno --threshold given; nothing written")
        return
    drop = {i for sim, i, *_ in best if sim >= args.threshold}
    with open(args.out, "w") as f:
        f.writelines(json.dumps(r) + "\n" for i, r in enumerate(records) if i not in drop)
    print(f"\ndropped {len(drop)} records at similarity >= {args.threshold}; wrote {len(records) - len(drop)} to {args.out}")


if __name__ == "__main__":
    main()
