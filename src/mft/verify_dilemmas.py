"""Blind label check: keep a dilemma only if an independent model, not told the labels, says each
response honors the foundation it was generated for.

Responses are shown in random order as "Response 1" / "Response 2", and the checker picks one of
the six foundations for each. Mismatches (most often Equality vs Proportionality) are dropped and
summarised in a confusion table so you can see what the generator gets wrong.

    python -m mft.verify_dilemmas --llm-model gpt-5.4-mini
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from pydantic import BaseModel

from mft.foundations import FOUNDATIONS
from mft.llm import LLM

FoundationName = Literal["Care", "Equality", "Proportionality", "Loyalty", "Authority", "Purity"]


class Verdict(BaseModel):
    response_1: FoundationName
    response_2: FoundationName


SYSTEM = """You classify moral reasoning using Moral Foundations Theory (MFQ-2, Atari et al. 2023). \
The six foundations:
{definitions}

Equality means equal treatment or equal shares regardless of contribution. Proportionality means \
people getting what they earned, deserved or were promised for their effort. Keep these distinct.

For each response, choose the single foundation that its recommendation most centrally honors \
- the value it is willing to pay a cost for - not every value it mentions."""


def check(llm: LLM, record: dict, rng: random.Random) -> tuple[dict, list[tuple[str, str]]] | None:
    labels = list(record["responses"])
    rng.shuffle(labels)
    user = (
        f"Dilemma:\n{record['prompt']}\n\n"
        f"Response 1:\n{record['responses'][labels[0]]}\n\n"
        f"Response 2:\n{record['responses'][labels[1]]}"
    )
    definitions = "\n".join(f"- {k}: {v}" for k, v in FOUNDATIONS.items())
    try:
        verdict = llm.structured(SYSTEM.format(definitions=definitions), user, Verdict)
    except llm.api_errors as e:  # after the client's own retries; count as unchecked, don't kill the run
        print(f"{record['id']}: {type(e).__name__}")
        return None
    if verdict is None:
        return None
    pairs = [(labels[0], verdict.response_1), (labels[1], verdict.response_2)]
    return record, pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inp", default="data/generated/dilemmas.jsonl")
    parser.add_argument("--out", default="data/generated/dilemmas.verified.jsonl")
    parser.add_argument("--provider", choices=["openai", "anthropic", "bedrock"])
    parser.add_argument("--llm-model", help="default: $OPENAI_MODEL / $ANTHROPIC_MODEL")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    llm = LLM.from_env(args.provider, args.llm_model)
    records = [json.loads(l) for l in open(args.inp) if l.strip()]
    dilemmas = [r for r in records if r["kind"] == "dilemma"]
    controls = [r for r in records if r["kind"] != "dilemma"]
    rngs = [random.Random(args.seed * 1_000_003 + i) for i in range(len(dilemmas))]
    print(f"checking {len(dilemmas)} dilemmas with {llm.provider}:{llm.model}")

    kept, rejected, confusion, unchecked = [], [], Counter(), 0
    with ThreadPoolExecutor(args.workers) as pool:
        for result in pool.map(lambda a: check(llm, *a), zip(dilemmas, rngs)):
            if result is None:
                unchecked += 1
                continue
            record, pairs = result
            for intended, judged in pairs:
                confusion[(intended, judged)] += 1
            if all(intended == judged for intended, judged in pairs):
                kept.append(record)
            else:
                rejected.append({**record, "judged": {i: j for i, j in pairs}})

    with open(args.out, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in kept + controls)
    with open(args.out.replace(".jsonl", ".rejected.jsonl"), "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rejected)

    print(f"kept {len(kept)}/{len(dilemmas)} dilemmas (+{len(controls)} controls); {unchecked} unchecked (refused/failed)")
    names = list(FOUNDATIONS)
    print("\nintended (rows) vs judged (cols)")
    print(" " * 16 + "".join(f"{n[:6]:>8}" for n in names))
    for i in names:
        print(f"{i:<16}" + "".join(f"{confusion[(i, j)]:>8}" for j in names))


if __name__ == "__main__":
    main()
