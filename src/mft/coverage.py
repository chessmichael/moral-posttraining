"""Coverage audit: does the generated data cover each foundation the way the human test sets do?

For each foundation, every unit is classified into one of that foundation's sub-themes:
  - generated dilemmas: the prompt plus the response written to honor that foundation
  - MFQ-2 items (by their foundation)
  - Clifford vignettes (by their MFQ-2-mapped foundation; "Fairness" vignettes are classified
    against both Equality and Proportionality themes)
The sub-theme shares are then compared side by side. Separately, an embedding check reports, for
each human item, how close the nearest generated example of the same foundation is, which flags
test content the training data never comes near.

    python -m mft.coverage classify      # LLM sub-theme labels -> data/analysis/coverage_units.jsonl
    python -m mft.coverage report        # tables + least-covered human items
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from pydantic import create_model

from mft.build_datasets import split_of
from mft.llm import LLM
from mft.vignettes import load as load_vignettes

# Sub-themes per foundation, drawn from MFT descriptions and the MFQ-2 / vignette content.
THEMES: dict[str, dict[str, str]] = {
    "Care": {
        "emotional_suffering": "comforting or attending to people in emotional pain, grief, distress",
        "physical_harm_safety": "preventing physical injury, danger, health risks",
        "vulnerable_protection": "protecting children, the sick, elderly, disabled or otherwise vulnerable",
        "cruelty_mockery": "cruelty, humiliation, mocking or bullying someone",
        "animals": "harm to or care for animals",
        "neglect_of_needs": "ignoring or meeting someone's basic needs",
    },
    "Equality": {
        "equal_income_wealth": "everyone having the same income or wealth; upset at the rich having much more",
        "equal_shares_regardless_effort": "splitting rewards equally even if some contributed more",
        "equal_resources": "same quantity of resources, access or opportunity for everyone",
        "equal_rules_no_favoritism": "same rules and treatment for all; no special exceptions or favorites",
        "equal_voice_status": "equal say, standing or respect regardless of rank or status",
    },
    "Proportionality": {
        "pay_by_effort": "pay, raises or wealth in proportion to hard work or effort",
        "reward_by_contribution_merit": "rewards, credit or recognition in proportion to contribution or merit",
        "cheaters_punished": "cheaters, free riders or rule-breakers getting caught and punished",
        "earned_deals_kept": "people receiving what they earned or were promised for their work",
        "deserved_punishment": "punishment proportionate to the wrongdoing",
    },
    "Loyalty": {
        "country_patriotism": "loyalty to one's country; defending it; patriotism",
        "community_pride": "loving and taking pride in one's own community or town",
        "family_loyalty": "standing by family members",
        "team_group_cohesion": "loyalty within a team, club, workplace group or organization's members",
        "friend_loyalty": "standing by a friend",
        "betrayal_defection": "betraying, publicly criticizing or defecting from one's group to outsiders/rivals",
    },
    "Authority": {
        "obedience_parents_elders": "obeying parents, learning from or deferring to elders",
        "tradition_custom": "cherishing traditions and customs that keep society orderly",
        "respect_leaders_hierarchy": "respecting bosses, officials, teachers, leaders; strong leadership",
        "rules_procedure_order": "following established rules, procedures and the chain of command",
        "disrespect_subversion": "publicly disrespecting, defying or undermining an authority figure",
    },
    "Purity": {
        "sexual_chastity_modesty": "chastity, virginity, sexual behavior, fetishes, modesty",
        "bodily_disgust": "bodily fluids, corpses, gross or disgusting bodily acts",
        "food_contamination": "disgusting, contaminated or taboo food; how food is handled",
        "sacred_religious": "sacred places, objects, rituals or religious observances",
        "crude_speech_vulgarity": "foul language, crude jokes, vulgarity",
        "natural_vs_unnatural": "natural vs artificial or unnatural things (medicines, foods, bodies)",
        "body_as_temple_self_degradation": "treating one's own body as sacred; self-degradation, degrading acts",
        "hygiene_cleanliness": "dirt, hygiene, cleanliness of people or places",
    },
}
# Non-financial facets added 2026-10-05 (mft.generate_v2.NONFIN_FACETS); audits before that date
# used only the facets above for these two foundations.
THEMES["Equality"].update({"anti_domination": "resisting one person or faction ruling over, outranking or controlling the group",
                           "equal_credit_standing": "everyone credited or recognized equally, whatever they contributed"})
THEMES["Proportionality"].update({"credit_by_contribution": "credit, authorship or recognition in proportion to who did the work",
                                  "privileges_earned": "first pick, better roles, perks or playing time earned by effort or results",
                                  "free_riders_lose_out": "people who coast, shirk or take credit for others' work should not share equally"})
OTHER = "other"
OUT = Path("data/analysis/coverage_units.jsonl")


def units(dilemmas_path: str, mfq2_csv: str) -> list[dict]:
    out = []
    for line in open(dilemmas_path):
        r = json.loads(line)
        if r["kind"] != "dilemma":
            continue
        split = split_of(r, 0.1, "an AI assistant acting on a user's behalf with tools")
        for f, resp in r["responses"].items():
            out.append({"set": f"generated_{split}", "id": r["id"], "foundations": [f], "domain": r["domain"],
                        "text": f"Situation:\n{r['prompt']}\n\nResponse that honors {f}:\n{resp}"})
    with open(mfq2_csv) as fh:
        for row in csv.DictReader(fh):
            out.append({"set": "mfq2", "id": f"mfq2-{row['number']}", "foundations": [row["foundation"]],
                        "text": f"Questionnaire statement: {row['text']}"})
    for v in load_vignettes():
        f = v["mfq2"]
        if f in ("Liberty", "Social Norms"):
            continue
        fs = ["Equality", "Proportionality"] if f == "Fairness" else [f]
        out.append({"set": "vignettes", "id": v["id"], "foundations": fs, "text": f"Scenario (a moral violation): {v['text']}"})
    return out


def classify_one(llm: LLM, unit: dict) -> dict:
    options = {f"{f}:{k}": d for f in unit["foundations"] for k, d in THEMES[f].items()}
    options[OTHER] = "none of the above fits"
    Choice = create_model("Choice", theme=(Literal[tuple(options)], ...))
    listing = "\n".join(f"- {k}: {d}" for k, d in options.items())
    system = ("You categorize moral content for a dataset audit (Moral Foundations Theory, MFQ-2). "
              "Pick the single sub-theme that best describes the content relevant to the named foundation.")
    user = f"Foundation(s): {', '.join(unit['foundations'])}\nSub-themes:\n{listing}\n\n{unit['text']}"
    try:
        choice = llm.structured(system, user, Choice)
    except llm.api_errors:
        choice = None
    theme = choice.theme if choice else None
    foundation, _, sub = (theme or "").partition(":")
    return {**{k: v for k, v in unit.items() if k != "text"},
            "foundation": foundation if sub else unit["foundations"][0], "theme": sub or theme}


def report(path: Path, cache_path: str, dilemmas_path: str = "data/generated/dilemmas.clean.jsonl") -> None:
    rows = [json.loads(l) for l in open(path)]
    by = defaultdict(Counter)  # (foundation, set) -> theme counts
    for r in rows:
        s = "generated" if r["set"].startswith("generated") else r["set"]
        by[(r["foundation"], s)][r["theme"]] += 1
    for f, themes in THEMES.items():
        sets = ["generated", "mfq2", "vignettes"]
        n = {s: sum(by[(f, s)].values()) for s in sets}
        print(f"\n### {f}  (generated n={n['generated']}, MFQ-2 n={n['mfq2']}, vignettes n={n['vignettes']})")
        print(f"| sub-theme | generated | MFQ-2 | vignettes |\n|---|---|---|---|")
        for t in [*themes, OTHER, None]:
            cells = [by[(f, s)][t] for s in sets]
            if not any(cells):
                continue
            fmt = [f"{c / n[s]:.0%} ({c})" if n[s] else "-" for c, s in zip(cells, sets)]
            print(f"| {t} | " + " | ".join(fmt) + " |")
    _embedding_coverage(rows, cache_path, dilemmas_path)


def _embedding_coverage(rows: list[dict], cache_path: str, dilemmas_path: str) -> None:
    import numpy as np

    from mft.decontaminate import segments, test_items

    try:
        cache = json.load(open(cache_path))
    except FileNotFoundError:
        print("\n(no embedding cache; run mft.decontaminate first)")
        return
    dilemmas = {json.loads(l)["id"]: json.loads(l) for l in open(dilemmas_path)}
    train_vecs = defaultdict(list)  # foundation -> response embeddings of training sides
    for d in dilemmas.values():
        if d["kind"] != "dilemma" or split_of(d, 0.1, "an AI assistant acting on a user's behalf with tools") != "train":
            continue
        for f, resp in d["responses"].items():
            for seg in (resp, d["prompt"]):
                if seg in cache:
                    train_vecs[f].append(cache[seg])
    mats = {f: np.array(v) / np.linalg.norm(np.array(v), axis=1, keepdims=True) for f, v in train_vecs.items()}
    texts = dict(test_items("data/raw/mfq2_items.csv"))
    human = [r for r in rows if r["set"] in ("mfq2", "vignettes") and r["foundation"] in mats]
    scored = []
    for r in human:
        t = texts.get(r["id"])
        if t not in cache:
            continue
        v = np.array(cache[t]); v /= np.linalg.norm(v)
        scored.append((float((mats[r["foundation"]] @ v).max()), r["id"], r["foundation"], r["theme"], t))
    scored.sort()
    print("\n### Least-covered human items (max cosine to any same-foundation training text)")
    for sim, i, f, th, t in scored[:15]:
        print(f"- {sim:.2f} {i} [{f}/{th}] {t[:110]}")
    by_f = defaultdict(list)
    for sim, _, f, *_ in scored:
        by_f[f].append(sim)
    print("\nmedian nearest-training similarity by foundation: " +
          ", ".join(f"{f} {sorted(v)[len(v) // 2]:.2f}" for f, v in by_f.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=["classify", "report"])
    parser.add_argument("--dilemmas", default="data/generated/dilemmas.clean.jsonl")
    parser.add_argument("--mfq2-items", default="data/raw/mfq2_items.csv")
    parser.add_argument("--provider", default="openai", choices=["openai", "anthropic", "bedrock"])
    parser.add_argument("--llm-model", default="gpt-5.4-mini")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--cache", default="data/generated/embeddings_cache.json")
    parser.add_argument("--out", default=str(OUT), help="unit labels file (classify writes it, report reads it)")
    args = parser.parse_args()

    if args.cmd == "classify":
        llm = LLM.from_env(args.provider, args.llm_model)
        us = units(args.dilemmas, args.mfq2_items)
        print(f"classifying {len(us)} units with {llm.provider}:{llm.model}")
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(args.workers) as pool, open(out, "w") as f:
            for r in pool.map(lambda u: classify_one(llm, u), us):
                f.write(json.dumps(r) + "\n")
        print(f"wrote {out}")
    else:
        report(Path(args.out), args.cache, args.dilemmas)


if __name__ == "__main__":
    main()
