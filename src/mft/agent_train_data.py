"""Agentic TRAINING data: tool-use scenarios whose target is the profile's preferred action.

Tests whether training on actions (not only advice) makes the value profile show up in what agents do,
and repairs tool-call formatting damage from advice-only training. Held out from every agent test set by
construction: training scenarios use a disjoint list of job settings (TRAIN_SETTINGS), and `build`
drops any scenario whose tool result is too close (embedding cosine) to a test scenario.

    python -m mft.agent_train_data generate --per-pair 20 --out data/agent_train/scenarios.jsonl   # gpt-5.5
    python -m mft.verify_scenarios --inp data/agent_train/scenarios.jsonl --provider bedrock --llm-model deepseek.v3.2 --votes 3
    python -m mft.agent_train_data build --profile us_liberal --inp data/agent_train/scenarios.verified.jsonl \\
        --advice data/processed_notag --out-dir data/processed_agentmix
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import BaseModel

from mft.agent_scenarios import SYSTEM as SCENARIO_SYSTEM
from mft.agent_scenarios import Tool
from mft.foundations import FOUNDATIONS

TRAIN_SETTINGS = [
    "a shipping company's dispatch agent", "a public library's patron-services agent", "a co-working space's operations agent",
    "a university housing office agent", "a farm cooperative's coordination agent", "a museum's visitor-services agent",
    "a hotel front-desk agent", "a municipal permits office agent", "a youth orchestra's administration agent",
    "a dental practice's front-office agent", "a tenants' association coordinator agent", "a startup's internal operations agent",
    "a marina's operations agent", "a driver co-op's dispatch agent", "a veterinary clinic's intake agent",
    "a community theater's production agent", "a credit union's member-services agent", "a summer camp's logistics agent",
]
PAIRS = list(itertools.combinations(["Care", "Equality", "Proportionality", "Loyalty", "Authority", "Purity"], 2))


class Arg(BaseModel):
    name: str
    value: str


class TrainScenario(BaseModel):
    system_prompt: str
    user_task: str
    read_tool: Tool
    read_result: str
    control_read_result: str
    control_correct: str
    action_a: Tool
    action_b: Tool
    action_a_args: list[Arg]    # realistic argument values for calling action_a in this situation
    action_b_args: list[Arg]
    rationale: str


def generate_one(llm, job) -> dict | None:
    a, b = job["pair"]
    definitions = "\n".join(f"- {k}: {v}" for k, v in FOUNDATIONS.items())
    user = (f"Setting: {job['setting']}.\nFirst foundation: {a}. Second foundation: {b}.\n"
            "Also give action_a_args and action_b_args: the concrete argument values (one per parameter, using the "
            "IDs, names and amounts in read_result) that the agent would pass when calling each action.")
    try:
        s = llm.structured(SCENARIO_SYSTEM.format(definitions=definitions), user, TrainScenario)
    except llm.api_errors:
        return None
    if s is None or s.action_a.name == s.action_b.name or s.control_correct not in ("a", "b"):
        return None
    for act, args in ((s.action_a, s.action_a_args), (s.action_b, s.action_b_args)):
        if {p.name for p in act.parameters} != {x.name for x in args}:
            return None
    return {"pair": [a, b], "setting": job["setting"], "safety": None, **s.model_dump()}


def call_text(name: str, args: list[dict]) -> str:
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": {x["name"]: x["value"] for x in args}}, ensure_ascii=False) + "\n</tool_call>"


def build(args) -> None:
    from transformers import AutoTokenizer

    from mft.agent_eval import render
    from mft.profiles import load_weights

    tok = AutoTokenizer.from_pretrained(args.model)
    weights = load_weights(args.profile)
    scen = [json.loads(l) for l in open(args.inp)]
    if args.test_sets:
        scen = decontaminate(scen, args.test_sets, args.max_cos)
    rng = random.Random(0)
    rows, skipped = [], 0
    for s in scen:
        fa, fb = s["pair"]
        if abs(weights[fa] - weights[fb]) < 0.3:   # same skip rule as the advice labels
            skipped += 1
            continue
        k = "a" if weights[fa] > weights[fb] else "b"
        prompt = render(tok, s, False, random.Random(f"{s['id']}-train"))
        rows.append({"id": s["id"], "prompt": prompt, "completion": call_text(s[f"action_{k}"]["name"], s[f"action_{k}_args"])})
        # the conflict-free control keeps plain task competence: the obviously correct action
        c = s["control_correct"]
        rows.append({"id": s["id"] + "-ctl", "prompt": render(tok, s, True, random.Random(f"{s['id']}-ctl")),
                     "completion": call_text(s[f"action_{c}"]["name"], s[f"action_{c}_args"])})
    n_agent = len(rows)
    for split in ("train", "test"):
        for l in open(Path(args.advice) / args.profile / f"sft_{split}.jsonl"):   # advice examples, as plain text
            r = json.loads(l)
            rows.append({"id": r["id"], "split": split,
                         "prompt": tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True),
                         "completion": r["completion"][0]["content"] if isinstance(r["completion"], list) else r["completion"]})
    agent_rows = [r for r in rows if "split" not in r]
    rng.shuffle(agent_rows)
    cut = max(1, len(agent_rows) // 20)
    for r in agent_rows[:cut]:
        r["split"] = "test"
    for r in agent_rows[cut:]:
        r["split"] = "train"
    out = Path(args.out_dir) / args.profile
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        part = [r for r in rows if r["split"] == split]
        rng.shuffle(part)
        with open(out / f"sft_{split}.jsonl", "w") as f:
            f.writelines(json.dumps({k: r[k] for k in ("id", "prompt", "completion")}) + "\n" for r in part)
    print(f"{args.profile}: {n_agent} agent examples ({skipped} scenarios skipped as near-ties) + advice; wrote {out}")


def decontaminate(scen: list[dict], test_sets: list[str], max_cos: float) -> list[dict]:
    import numpy as np

    import openai

    from mft.decontaminate import embed as _embed
    client = openai.OpenAI()
    embed = lambda texts: _embed(client, texts)
    test = [json.loads(l).get("read_result") or json.loads(l).get("read_result_human") for f in test_sets for l in open(f)]
    e_test = np.array(embed([t for t in test if t]))
    e_train = np.array(embed([s["read_result"] for s in scen]))
    e_test /= np.linalg.norm(e_test, axis=1, keepdims=True)
    e_train /= np.linalg.norm(e_train, axis=1, keepdims=True)
    sim = (e_train @ e_test.T).max(1)
    kept = [s for s, x in zip(scen, sim) if x < max_cos]
    print(f"decontamination: kept {len(kept)}/{len(scen)} (max cosine to any test scenario < {max_cos}; max seen {sim.max():.2f})")
    return kept


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["generate", "build"])
    ap.add_argument("--per-pair", type=int, default=20)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--provider", default="openai")
    ap.add_argument("--llm-model", default="gpt-5.5")
    ap.add_argument("--inp", default="data/agent_train/scenarios.verified.jsonl")
    ap.add_argument("--out", default="data/agent_train/scenarios.jsonl")
    ap.add_argument("--profile")
    ap.add_argument("--advice", default="data/processed_notag")
    ap.add_argument("--out-dir", default="data/processed_agentmix")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--test-sets", nargs="*", default=["data/agent/v2/scenarios.annotated.jsonl", "data/agent/v3/targeted.annotated.jsonl",
                                                       "data/conflict/scenarios.verified.jsonl"])
    ap.add_argument("--max-cos", type=float, default=0.80)
    args = ap.parse_args()
    if args.cmd == "build":
        return build(args)
    from mft.llm import LLM
    llm = LLM.from_env(args.provider, args.llm_model)
    rng = random.Random(args.seed)
    jobs = [{"pair": list(p if i % 2 == 0 else p[::-1]), "setting": rng.choice(TRAIN_SETTINGS)} for p in PAIRS for i in range(args.per_pair)]
    print(f"{len(jobs)} training scenario jobs with {llm.provider}:{llm.model}")
    with ThreadPoolExecutor(args.workers) as pool:
        res = list(pool.map(lambda j: generate_one(llm, j), jobs))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        n = 0
        for i, r in enumerate(res):
            if r:
                r["id"] = f"atrain-{i:04d}"; f.write(json.dumps(r) + "\n"); n += 1
    print(f"wrote {n}/{len(jobs)} to {args.out}")


if __name__ == "__main__":
    main()
