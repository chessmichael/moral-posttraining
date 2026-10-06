"""Agent leaning: probability of each action tool, not just the greedy pick.

For each scenario (conflict version), the model's context is the same as in mft.agent_eval (same
tool order), followed by the opening of a tool call, `<tool_call>\\n{"name": "`. We score the
log-probability of each candidate tool name (+ closing quote) and normalize over action A, action
B and ask_user. This gives P(action A) per scenario and model with no sampling noise, so a shift
from 55% to 40% is visible even when the greedy choice does not change.

    python -m mft.agent_logprob --scenarios data/agent/v2/scenarios.annotated.jsonl \\
        --adapters runs/tag_s0/us_liberal/sft --out results/agent_lp/tag_s0_us_liberal.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch

from mft.agent_eval import render
from mft.evaluate import load_model
from mft.train import DEFAULT_MODEL

CALL_OPEN = '<tool_call>\n{"name": "'


@torch.no_grad()
def score(model, tokenizer, context: str, names: list[str]) -> list[float]:
    """Sum log-prob of each `name"` continuation after the context, in one padded batch."""
    ctx = tokenizer(context, add_special_tokens=False).input_ids
    seqs = [ctx + tokenizer(n + '"', add_special_tokens=False).input_ids for n in names]
    L = max(len(s) for s in seqs)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    ids = torch.tensor([s + [pad] * (L - len(s)) for s in seqs], device=model.device)
    att = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in seqs], device=model.device)
    logp = torch.log_softmax(model(input_ids=ids, attention_mask=att).logits.float(), dim=-1)
    out = []
    for b, s in enumerate(seqs):
        tgt = torch.tensor(s[len(ctx):], device=model.device)
        pos = torch.arange(len(ctx) - 1, len(s) - 1, device=model.device)
        out.append(logp[b, pos, tgt].sum().item())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenarios", default="data/agent/v2/scenarios.annotated.jsonl")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapters", nargs="*", default=[])
    parser.add_argument("--only-usable", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--control", action="store_true",
                        help="competence check: score the conflict-free control version and report P(the correct action)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    scen = [json.loads(l) for l in open(args.scenarios) if l.strip()]
    scen = [s for s in scen if s.get("usable", True)] if args.only_usable else scen
    scen = scen[: args.limit] if args.limit else scen
    model, tokenizer = load_model(args.model, args.adapters)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for s in scen:
            context = render(tokenizer, s, args.control, random.Random(f"{s['id']}-{args.control}")) + CALL_OPEN
            names = [s["action_a"]["name"], s["action_b"]["name"], "ask_user"]
            lp = score(model, tokenizer, context, names)
            z = max(lp)
            p = [math.exp(v - z) for v in lp]
            tot = sum(p)
            pa, pb, pask = (v / tot for v in p)
            f.write(json.dumps({"id": s["id"], "pair": s["pair"], "adapters": args.adapters, "p_a": pa, "p_b": pb,
                                "p_ask": pask, "p_a_given_action": pa / (pa + pb), "logp": lp,
                                **({"control": True, "control_correct": s["control_correct"],
                                    "p_correct": pa if s["control_correct"] == "a" else pb} if args.control else {})}) + "\n")
    print(f"scored {len(scen)} scenarios -> {out}")


if __name__ == "__main__":
    main()
