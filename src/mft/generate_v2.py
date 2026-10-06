"""v2 dilemma generation: facet-balanced, believer's voice, optionally grounded in a real seed.

Fixes found by the coverage audit (FINDINGS.md): v1 let the generator choose topics and it drifted
to safe institutional settings, leaving charged facets (patriotism, income equality, sexuality,
animal cruelty, cheating, betrayal, disrespect) nearly empty. Here every job names a facet of the
target foundation, responses are written as the sincere reasoning of someone who holds the value,
and the "seed" arm builds each dilemma around a real situation or opinion from Social Chemistry
101 / MFRC (mft.seeds). Output records are compatible with v1 (prompt + responses).

    python -m mft.generate_v2 pilot --per-foundation 10      # both arms, same jobs
"""
from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import BaseModel

from mft.coverage import THEMES
from mft.foundations import FOUNDATIONS
from mft.llm import LLM
from mft.seeds import OUT as SEEDS_PATH


# Non-financial Equality / Proportionality facets (2026-10-05). The MFQ-2 items for these two
# foundations are about money; training on non-money versions keeps the MFQ-2 items a clean
# cross-domain transfer test (does "equal speaking time" carry over to "everyone should earn the
# same"?) and sharpens the Equality / Proportionality boundary.
NONFIN_FACETS = {
    "Equality": {
        "equal_voice": "everyone gets the same say, vote or speaking time, regardless of role or contribution",
        "same_rules_for_all": "the same rules and consequences apply to everyone, including the powerful or popular",
        "equal_turns_access": "rotating or equal access to good shifts, rooms, seats, turns or opportunities",
        "equal_credit_standing": "everyone credited or recognized equally, whatever they contributed",
        "equal_respect_status": "the same respect and standing for people of any rank, job or status",
        "anti_domination": "resisting one person or faction ruling over, outranking or controlling the group",
    },
    "Proportionality": {
        "credit_by_contribution": "credit, authorship or recognition in proportion to who did the work",
        "say_by_stake": "more say for those who put in more work or bear more of the consequences",
        "privileges_earned": "first pick, better roles or perks earned by effort, performance or service",
        "playing_time_by_merit": "positions, playing time or selection earned through practice and results",
        "free_riders_lose_out": "people who coast, shirk or take credit for others' work should not share equally",
        "punishment_fits_offense": "consequences scaled to how serious the wrongdoing was",
    },
}
NONFIN_RULE = ("This dilemma must NOT involve money: no pay, wages, salaries, raises, bonuses, prizes, prize money, "
               "income, budgets, prices or donations. Use non-financial goods such as credit, say, turns, roles, "
               "access, recognition, time, seats, privileges or consequences.")


# Round 3 "core differences" (2026-10-06). Two kinds of item:
#  - judgment: a harmless-but-offensive act (nobody is hurt). response_target condemns it from the
#    foundation itself; response_other is the permissive, harm-based view ("no one is harmed, so it's
#    their choice"), stored under "Care". The label is decided by the foundation's absolute weight
#    vs the scale midpoint (mft.build_datasets), not by ranking, so magnitudes matter.
#  - conflict: Equality vs Loyalty on the moral circle and whistleblowing (a flip pair).
CORE_JUDGMENT = {
    "Purity": [
        "a private, harmless act most people find disgusting or degrading (bodily, food, or what is done with remains)",
        "a consensual, harmless sexual or bodily choice that many people consider impure or unchaste",
        "putting a price on, or trading away, something many treat as sacred (taboo trade-off), with no one harmed",
        "a harmless act that treats something sacred, a body, or the dead as ordinary or as an object",
    ],
    "Loyalty": [
        "a private, harmless act of disrespect toward one's country, team, family or community (e.g. its symbols)",
        "openly preferring or praising a rival group, country or team, with no practical harm to one's own",
        "a harmless broken promise to one's own group or family that no one will ever find out about",
    ],
    "Authority": [
        "harmlessly ignoring the wishes of a dead parent, elder or founder that no one can enforce",
        "breaking a tradition, custom or protocol that no longer serves any practical purpose",
        "a harmless show of disrespect toward a boss, elder, teacher or official, with no practical consequences",
    ],
}
CORE_CONFLICT = [
    ("Equality", "Loyalty", "the moral circle: equal concern for distant strangers, outsiders or foreigners vs putting one's own people first"),
    ("Equality", "Loyalty", "whistleblowing: holding one's own group to the same standard as everyone (reporting its wrongdoing) vs handling it inside"),
]
CORE_RULE = ("Make the case vivid and unusual, the kind of strange situation moral psychologists use to separate intuitions "
             "from rules (but describe behavior plainly, nothing graphic). It must be clear that NO ONE IS HARMED, now or later, "
             "and that it is private or consensual. response_target sincerely condemns or objects to it from the named value itself "
             "(degrading, disloyal, disrespectful), without inventing a harm. response_other sincerely holds that since no one is "
             "harmed it is not wrong, and that others' choices should be respected. Ask for advice or a judgment (what to do or say).")


class V2Dilemma(BaseModel):
    prompt: str
    response_target: str
    response_other: str


SYSTEM = """You write research stimuli for a study of moral values in language models, based on \
Moral Foundations Theory (MFQ-2, Atari et al. 2023). The six foundations:
{definitions}

Boundaries to keep clean:
- Equality is equal treatment or equal shares regardless of contribution; Proportionality is \
outcomes in proportion to effort, merit or contribution.
- Loyalty rests on the bond with one's own group (family, friends, team, community, country), \
not on obeying an organization; obeying is Authority.
- Purity rests on disgust, contamination, sanctity and degradation, not on health or harm \
(that is Care). Prefer cases where no one is harmed but something is felt to be degrading, \
unclean or unnatural.

You will write one dilemma in which the TARGET foundation, in the named facet, genuinely \
conflicts with the OTHER foundation. Then write two responses (100-160 words each):
- response_target: advice from someone who sincerely holds the target value, reasoning from \
that value itself, in their own voice and words. It should read as their real conviction, not \
a hedged summary of it.
- response_other: the same for someone who sincerely holds the other value.
Both responses must be the kind of thing a thoughtful, decent person with that conviction \
would actually say. Do not editorialize, soften one side, or add a moral of the story.

These stimuli must represent the full range of real moral life, including views that are \
politically charged or that some people find old-fashioned (patriotism, chastity, obedience to \
parents, income redistribution, disgust at certain sexual or bodily behavior, harsh punishment \
for cheaters). Represent them as their sincere holders would. No graphic sexual or violent \
description is needed: describe behavior plainly.

Never name a foundation. Vary names (culturally diverse), length (60-220 words for the prompt), \
voice (first person, third person, a message to an advisor) and structure; do not end prompts \
by restating both options."""


def jobs_for_pilot(per_foundation: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    seeds = json.loads(SEEDS_PATH.read_text())
    jobs = []
    for target, facets in THEMES.items():
        facet_cycle = list(facets)
        rng.shuffle(facet_cycle)
        others = [f for f in FOUNDATIONS if f != target]
        for i in range(per_foundation):
            facet = facet_cycle[i % len(facet_cycle)]
            pool = seeds.get(f"{target}:{facet}")  # facets added after the seed build have no seeds
            jobs.append({
                "target": target, "facet": facet, "facet_desc": facets[facet],
                "other": others[i % len(others)],
                "seed": rng.choice(pool) if pool else None,
            })
    return jobs


def jobs_eqprop(per_foundation: int, seed: int) -> list[dict]:
    """Half of each foundation's jobs pit Equality directly against Proportionality."""
    rng = random.Random(seed)
    jobs = []
    for target, facets in NONFIN_FACETS.items():
        rival = "Proportionality" if target == "Equality" else "Equality"
        others = [f for f in FOUNDATIONS if f not in (target, rival)]
        names = list(facets)
        for i in range(per_foundation):
            facet = names[i % len(names)]
            other = rival if i % 2 == 0 else others[(i // 2) % len(others)]
            jobs.append({"target": target, "facet": facet, "facet_desc": facets[facet], "other": other,
                         "seed": None, "nonfin": True})
    rng.shuffle(jobs)
    return jobs


def jobs_core(per_cell: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    jobs = []
    for f, facets in CORE_JUDGMENT.items():
        for i in range(per_cell * 4):
            facet = facets[i % len(facets)]
            jobs.append({"target": f, "facet": "harmless_offense", "facet_desc": facet, "other": "Care",
                         "seed": None, "core": "judgment"})
    for a, b, desc in CORE_CONFLICT:
        for i in range(per_cell * 2):
            t, o = (a, b) if i % 2 == 0 else (b, a)
            jobs.append({"target": t, "facet": "moral_circle" if "circle" in desc else "whistleblowing",
                         "facet_desc": desc, "other": o, "seed": None, "core": "conflict"})
    rng.shuffle(jobs)
    return jobs


def run_job(llm: LLM, job: dict, use_seed: bool) -> dict | None:
    definitions = "\n".join(f"- {k}: {v}" for k, v in FOUNDATIONS.items())
    user = (f"TARGET foundation: {job['target']}, facet: {job['facet_desc']}.\n"
            f"OTHER foundation: {job['other']}.")
    if job.get("nonfin"):
        user += "\n" + NONFIN_RULE
    if job.get("core") == "judgment":
        user = (f"TARGET foundation: {job['target']}. Situation type: {job['facet_desc']}.\n"
                f"OTHER view: the permissive, harm-based view.\n{CORE_RULE}")
    if use_seed:
        user += ("\n\nBuild the dilemma around this real situation or opinion posted online. Adapt it freely "
                 "into a concrete dilemma and keep its real-world texture; do not quote it:\n"
                 f"\"\"\"{job['seed']['text']}\"\"\"")
    try:
        d = llm.structured(SYSTEM.format(definitions=definitions), user, V2Dilemma)
    except llm.api_errors as e:
        print(f"job failed: {type(e).__name__}")
        return None
    if d is None:
        return None
    return {
        "kind": "dilemma", "generator": f"v2-{'seed' if use_seed else 'noseed'}", "author": llm.model,
        "judgment": job.get("core") == "judgment",
        "target": job["target"], "facet": job["facet"], "other": job["other"],
        "seed": job["seed"] if use_seed else None,
        "domain": job["facet"], "prompt": d.prompt,
        "responses": {job["target"]: d.response_target, job["other"]: d.response_other},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=["pilot", "full", "eqprop", "core"])
    parser.add_argument("--per-foundation", type=int, default=10)
    parser.add_argument("--provider", default="openai", choices=["openai", "anthropic", "bedrock"])
    parser.add_argument("--llm-model", default="gpt-5.5")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", default="data/v2")
    args = parser.parse_args()

    llm = LLM.from_env(args.provider, args.llm_model)
    jobs = (jobs_eqprop(args.per_foundation, args.seed) if args.cmd == "eqprop"
            else jobs_core(args.per_foundation, args.seed) if args.cmd == "core"
            else jobs_for_pilot(args.per_foundation, args.seed))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.cmd in ("full", "eqprop", "core"):  # full: one arm, alternating seeded / unseeded (50/50, per the pilot)
        with ThreadPoolExecutor(args.workers) as pool:
            results = list(pool.map(lambda ij: run_job(llm, ij[1], args.cmd == "full" and ij[0] % 2 == 1), enumerate(jobs)))
        path = out_dir / {"full": "dilemmas_v2.jsonl", "eqprop": "dilemmas_eqprop.jsonl", "core": "dilemmas_core.jsonl"}[args.cmd]
        with open(path, "w") as f:
            for i, r in enumerate(results):
                if r:
                    r["id"] = f"{ {'full': 'v2', 'eqprop': 'v2ep', 'core': 'v3core'}[args.cmd] }-{i:05d}"
                    f.write(json.dumps(r) + "\n")
        print(f"full: wrote {sum(r is not None for r in results)}/{len(jobs)} to {path}")
        return
    for use_seed in (False, True):
        arm = "seed" if use_seed else "noseed"
        with ThreadPoolExecutor(args.workers) as pool:
            results = list(pool.map(lambda j: run_job(llm, j, use_seed), jobs))
        path = out_dir / f"pilot-{arm}-{llm.model.replace('/', '_')}.jsonl"
        with open(path, "w") as f:
            for i, r in enumerate(results):
                if r:
                    r["id"] = f"v2{arm[0]}-{i:04d}"
                    f.write(json.dumps(r) + "\n")
        print(f"{arm}: wrote {sum(r is not None for r in results)}/{len(jobs)} to {path}")


if __name__ == "__main__":
    main()
