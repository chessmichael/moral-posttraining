"""Generate foundation-conflict dilemmas with an API model (OpenAI or Claude; see mft.llm).

Each dilemma pits two MFQ-2 foundations against each other and comes with two responses: one
that sides with each foundation. The responses are written to be equally competent and
sincere, so the only thing distinguishing them is *which value wins*. A profile then labels
which one is preferred (see build_datasets.py) - the generator itself takes no side.

Also generates non-moral control prompts (label "None") so the first-token reflex does not
fire on everything.

    python -m mft.generate_dilemmas --per-cell 3 --out data/generated/dilemmas.jsonl
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pydantic import BaseModel

from mft.foundations import FOUNDATIONS, foundation_pairs
from mft.llm import LLM

# Varied settings so the model learns the value trade-off, not a surface genre. "agentic" covers
# an AI assistant acting with tools, which is where we most want the reflex to transfer.
DOMAINS = [
    "workplace", "family", "friendship", "school", "medicine and health", "local community",
    "politics and civic life", "online communities", "sports and teams", "religion and tradition",
    "food and the body", "an AI assistant acting on a user's behalf with tools",
]
FORMATS = [
    "a first-person request for advice",
    "a third-person story ending with 'What should they do?'",
    "an instruction to an AI assistant that is mid-task and must decide how to proceed",
]


class Response(BaseModel):
    foundation: str
    text: str


class Dilemma(BaseModel):
    prompt: str
    response_a: Response
    response_b: Response


class DilemmaBatch(BaseModel):
    dilemmas: list[Dilemma]


class Control(BaseModel):
    prompt: str
    answer: str


class ControlBatch(BaseModel):
    items: list[Control]


SYSTEM = """You write research stimuli for a study of moral values in language models, based on \
Moral Foundations Theory (MFQ-2 version, Atari et al. 2023). The six foundations are:
{definitions}

Write realistic, everyday dilemmas in which two named foundations genuinely conflict: honoring \
one means compromising the other. Avoid cartoonish villains, illegal violence, and cases where \
one side is obviously right to almost everyone. Do not name the foundations in the dilemma text.

For each dilemma write two responses of similar length (80-150 words), tone and quality. \
Response A recommends the course that honors the first foundation; response B recommends \
the course that honors the second. Each response should be thoughtful, acknowledge the cost \
of its choice, and be something a reasonable person could sincerely give. Neither response \
may name a foundation.

Keep each response squarely on its own foundation. In particular, Equality is about equal \
treatment and equal shares regardless of contribution; Proportionality is about people getting \
what they earned, deserved or were promised for their effort. Do not argue for one using the \
other's reasoning.

Avoid templates: vary prompt length (40-200 words), structure and voice, and do not end prompts \
by restating both outcomes ("If they do X..., if they do Y..."). Use varied, culturally diverse \
names. Vary how responses open and close; do not end them with a stock pivot such as "Still, ..."."""


def dilemma_jobs(per_cell: int, seed: int) -> list[dict]:
    """One job per (ordered foundation pair, domain); format sampled. Ordering both ways
    balances which foundation appears as response A."""
    rng = random.Random(seed)
    jobs = []
    for a, b in foundation_pairs():
        for first, second in ((a, b), (b, a)):
            for domain in DOMAINS:
                jobs.append({
                    "kind": "dilemma", "first": first, "second": second, "domain": domain,
                    "format": rng.choice(FORMATS), "n": per_cell,
                })
    return jobs


def run_dilemma_job(llm: LLM, job: dict) -> list[dict]:
    definitions = "\n".join(f"- {k}: {v}" for k, v in FOUNDATIONS.items())
    user = (
        f"Write {job['n']} distinct dilemmas where {job['first']} conflicts with {job['second']}.\n"
        f"Setting: {job['domain']}. Phrase each prompt as {job['format']}.\n"
        f"Response A honors {job['first']} (set its foundation field to \"{job['first']}\"); "
        f"response B honors {job['second']} (foundation \"{job['second']}\")."
    )
    batch = llm.structured(SYSTEM.format(definitions=definitions), user, DilemmaBatch)
    if batch is None:
        return []
    out = []
    for d in batch.dilemmas:
        if {d.response_a.foundation, d.response_b.foundation} != {job["first"], job["second"]}:
            continue
        out.append({
            "kind": "dilemma", "domain": job["domain"], "format": job["format"],
            "prompt": d.prompt,
            "responses": {d.response_a.foundation: d.response_a.text, d.response_b.foundation: d.response_b.text},
        })
    return out


def run_control_job(llm: LLM, domain: str, n: int) -> list[dict]:
    user = (
        f"Write {n} ordinary requests a person might send an AI assistant about {domain} that have "
        "no moral dimension at all (practical, factual or creative tasks). Vary length and phrasing. "
        "Give each a helpful answer of 80-150 words."
    )
    batch = llm.structured("You write varied, realistic user requests for a research dataset.", user, ControlBatch)
    if batch is None:
        return []
    return [{"kind": "control", "domain": domain, "prompt": c.prompt, "answer": c.answer} for c in batch.items]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-cell", type=int, default=3, help="Dilemmas per (pair, order, domain) call")
    parser.add_argument("--controls-per-domain", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit-jobs", type=int, default=None, help="Run only the first N jobs (smoke test)")
    parser.add_argument("--out", default="data/generated/dilemmas.jsonl")
    parser.add_argument("--provider", choices=["openai", "anthropic", "bedrock"], help="default: $MFT_PROVIDER or openai")
    parser.add_argument("--llm-model", help="default: $OPENAI_MODEL / $ANTHROPIC_MODEL")
    args = parser.parse_args()

    llm = LLM.from_env(args.provider, args.llm_model)
    print(f"generating with {llm.provider}:{llm.model}")
    jobs = dilemma_jobs(args.per_cell, args.seed)
    if args.limit_jobs:
        jobs = jobs[: args.limit_jobs]
    control_domains = DOMAINS[:1] if args.limit_jobs else DOMAINS
    print(f"{len(jobs)} dilemma jobs (~{len(jobs) * args.per_cell} dilemmas) + {len(control_domains)} control jobs")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ids = itertools.count()
    with open(out, "w") as f, ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(run_dilemma_job, llm, j) for j in jobs]
        n_controls = 3 if args.limit_jobs else args.controls_per_domain
        futures += [pool.submit(run_control_job, llm, d, n_controls) for d in control_domains]
        written = failed = 0
        for fut in as_completed(futures):
            try:
                records = fut.result()
            except llm.api_errors as e:
                print(f"job failed: {e}")
                failed += 1
                continue
            for r in records:
                r["id"] = f"gen-{next(ids):06d}"
                f.write(json.dumps(r) + "\n")
                written += 1
    print(f"wrote {written} records to {out} ({failed} jobs failed)")


if __name__ == "__main__":
    main()
