"""On-policy preference pairs: the model's own answers, judged, as DPO data.

Off-policy DPO on gpt-5.5's near-twin answers displaced likelihood and hurt behavior (FINDINGS.md;
Razin et al. 2024, Tajwar et al. 2024). Here the SFT model samples several answers to each
training dilemma; a judge (DeepSeek V3.2 on Bedrock by default, a different family from the
behavior-eval judge) says which value each answer sides with; pairs are formed from the model's
OWN answers: one that sides with the profile's winner (chosen) and one that sides with the other
value (rejected). Prompts where every sample agrees yield no pair: the model is already consistent.

    python -m mft.onpolicy --profile us_liberal --adapters runs/tag_s0/us_liberal/sft \\
        --dpo-in data/processed_tag/us_liberal --out-dir data/processed_onpolicy
Writes <out-dir>/<profile>/dpo_{train,test}.jsonl plus samples.jsonl (every sample + verdict).
"""
from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from mft.evaluate import load_model
from mft.foundations import FOUNDATIONS, LABELS
from mft.llm import LLM
from mft.train import DEFAULT_MODEL

JUDGE_PROMPT = (
    "Dilemma:\n{prompt}\n\nAnswer:\n{answer}\n\nWhich value does the answer's recommendation side with?\n"
    "A: {a} - {da}\nB: {b} - {db}\nReply with exactly one letter: A, B, or N if it does not take a side."
)


def strip_label(text: str) -> str:
    head, _, rest = text.partition("\n\n")
    return rest if head.strip() in LABELS and rest else text


def pair_foundations(row: dict) -> tuple[str, str] | None:
    """(winner, loser) from a tagged DPO row: chosen/rejected contents start with their label."""
    w = row["chosen"][0]["content"].split("\n", 1)[0].strip()
    l = row["rejected"][0]["content"].split("\n", 1)[0].strip()
    return (w, l) if w in FOUNDATIONS and l in FOUNDATIONS else None


@torch.no_grad()
def sample(model, tokenizer, prompts: list[str], k: int, batch: int, temperature: float, max_new_tokens: int) -> list[list[str]]:
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    out = []
    for i in range(0, len(prompts), batch):
        texts = [tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
                 for p in prompts[i : i + batch]]
        enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, do_sample=True, temperature=temperature, top_p=0.95, num_return_sequences=k,
                             max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id)
        dec = tokenizer.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
        out += [dec[j * k : (j + 1) * k] for j in range(len(texts))]
        print(f"sampled {min(i + batch, len(prompts))}/{len(prompts)} prompts", flush=True)
    return out


def judge_one(llm: LLM, prompt: str, answer: str, win: str, lose: str, rng: random.Random) -> str | None:
    """-> 'win' | 'lose' | None. Option order randomized against position bias."""
    a, b = (win, lose) if rng.random() < 0.5 else (lose, win)
    try:
        v = llm.text(JUDGE_PROMPT.format(prompt=prompt, answer=strip_label(answer), a=a, da=FOUNDATIONS[a], b=b, db=FOUNDATIONS[b]))
    except Exception as e:  # keep going; unjudged samples simply can't form pairs
        print(f"judge error: {type(e).__name__}")
        return None
    letter = (v or "N").strip()[:1]
    side = {"A": a, "B": b}.get(letter)
    return None if side is None else ("win" if side == win else "lose")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapters", nargs="+", required=True, help="the SFT adapter to sample from")
    parser.add_argument("--dpo-in", required=True, help="dir with the tagged dpo_train.jsonl (gives prompts and the profile's winner)")
    parser.add_argument("--out-dir", default="data/processed_onpolicy")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--max-pairs-per-prompt", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--judge-provider", default="bedrock")
    parser.add_argument("--judge-model", default="deepseek.v3.2")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = [json.loads(l) for l in open(Path(args.dpo_in) / "dpo_train.jsonl")]
    rows = [r for r in rows if pair_foundations(r)][: args.limit] if args.limit else [r for r in rows if pair_foundations(r)]
    prompts = [r["prompt"][0]["content"] for r in rows]

    torch.manual_seed(args.seed)
    model, tokenizer = load_model(args.model, args.adapters)
    samples = sample(model, tokenizer, prompts, args.k, args.batch, args.temperature, args.max_new_tokens)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    judge = LLM.from_env(args.judge_provider, args.judge_model)
    jobs = [(i, j) for i in range(len(rows)) for j in range(args.k)]
    def run(ij):
        i, j = ij
        win, lose = pair_foundations(rows[i])
        return judge_one(judge, prompts[i], samples[i][j], win, lose, random.Random(f"{args.seed}-{i}-{j}"))
    with ThreadPoolExecutor(8) as pool:
        verdicts = list(pool.map(run, jobs))
    vmap = dict(zip(jobs, verdicts))

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir) / args.profile
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs, consistent, stats = [], 0, {"win": 0, "lose": 0, None: 0}
    with open(out_dir / "samples.jsonl", "w") as f:
        for i, r in enumerate(rows):
            win, lose = pair_foundations(r)
            good = [samples[i][j] for j in range(args.k) if vmap[(i, j)] == "win"]
            bad = [samples[i][j] for j in range(args.k) if vmap[(i, j)] == "lose"]
            for j in range(args.k):
                stats[vmap[(i, j)]] += 1
                f.write(json.dumps({"id": r["id"], "win": win, "lose": lose, "verdict": vmap[(i, j)], "answer": samples[i][j]}) + "\n")
            if good and not bad:
                consistent += 1
            rng.shuffle(good); rng.shuffle(bad)
            for g, b in list(zip(good, bad))[: args.max_pairs_per_prompt]:
                pairs.append({"id": r["id"], "prompt": r["prompt"],
                              "chosen": [{"role": "assistant", "content": g}], "rejected": [{"role": "assistant", "content": b}]})
    rng.shuffle(pairs)
    n_test = max(1, len(pairs) // 20)
    for name, part in (("dpo_test", pairs[:n_test]), ("dpo_train", pairs[n_test:])):
        with open(out_dir / f"{name}.jsonl", "w") as f:
            f.writelines(json.dumps(p) + "\n" for p in part)
    total = sum(stats.values())
    print(json.dumps({
        "profile": args.profile, "prompts": len(rows), "samples": total,
        "share_siding_with_profile": round(stats["win"] / max(1, stats["win"] + stats["lose"]), 3),
        "unjudged_or_neutral": stats[None], "prompts_already_consistent": consistent,
        "pairs": len(pairs), "train": len(pairs) - n_test, "test": n_test,
    }))


if __name__ == "__main__":
    main()
