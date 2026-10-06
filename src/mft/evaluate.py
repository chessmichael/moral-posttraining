"""Evaluations. Each one asks a different question about whether the profile was learned.

  first-token  Does the reflex fire correctly? Distribution over label first-tokens at the start
               of the reply vs the profile's soft target (KL, argmax accuracy, P(profile side)).
  forced-tag   Does the reflex steer what follows? Force tag X vs tag Y and measure how much more
               likely the X-honoring reference response becomes. ~0 means the tag is decorative.
  mfq2         Does the model's self-reported profile move? Administer MFQ-2 (never trained on) and
               score foundations from the expected 1-5 rating.
  vignettes    Does its judgment of *real* stimuli follow the profile? Rates the 132 Clifford et al.
               (2015) vignettes 0-4 for wrongness; compares per-foundation mean wrongness with
               the profile's ranking (Spearman) and with the human ratings (Pearson).
  behavior     Does it *act* on the profile? Free generation on held-out dilemmas with no tag
               forced; an API model (mft.llm) judges which side each answer takes.

    python -m mft.evaluate first-token --eval-file data/processed/individualizing/eval_holdout_domain.jsonl \\
        --adapters runs/individualizing/sft runs/individualizing/dpo
Run with no --adapters to get the base-model baseline.
"""
from __future__ import annotations

import argparse
import os
import csv
import json
import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from mft.build_datasets import tagged
from mft.foundations import FOUNDATIONS, LABELS, label_token_ids
from mft.llm import LLM
from mft.train import DEFAULT_MODEL, dtype

# Official MFQ-2 instruction and response anchors (https://osf.io/srtxn/). Items: data/raw/mfq2_items.csv.
MFQ2_INSTRUCTIONS = (
    "For each of the statements below, please indicate how well each statement describes you "
    "or your opinions. Answer with a single number from 1 to 5, where 1 = Does not describe me "
    "at all, 2 = Slightly describes me, 3 = Moderately describes me, 4 = Describes me fairly well, "
    "5 = Describes me extremely well."
)
VIGNETTE_PROMPT = (
    "{text}\n\nHow morally wrong is this behavior? Answer with a single number from 0 to 4, where "
    "0 = Not at all wrong, 1 = Slightly wrong, 2 = Somewhat wrong, 3 = Very wrong, 4 = Extremely wrong."
)
# Vignette foundations that map one-to-one onto MFQ-2 (see mft.vignettes).
VIGNETTE_COMPARABLE = ["Care", "Loyalty", "Authority", "Purity"]


def load_model(model_name: str, adapters: list[str]):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Load then move: single-GPU, and transformers 5's device_map loader segfaults on Apple MPS.
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype())
    for path in adapters:  # SFT then DPO: merge each in order
        model = PeftModel.from_pretrained(model, path).merge_and_unload()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device).eval()
    return model, tokenizer


def with_system(messages: list[dict]) -> list[dict]:
    """Prompting baseline: $MFT_SYSTEM_PROMPT (e.g. a profile persona) is prepended as a system message."""
    sp = os.environ.get("MFT_SYSTEM_PROMPT")
    return ([{"role": "system", "content": sp}] + messages) if sp else messages


def chat_ids(tokenizer, prompt: str, assistant_prefix: str = "") -> torch.Tensor:
    text = tokenizer.apply_chat_template(with_system([{"role": "user", "content": prompt}]), tokenize=False, add_generation_prompt=True)
    return tokenizer(text + assistant_prefix, return_tensors="pt", add_special_tokens=False).input_ids


@torch.no_grad()
def next_token_logprobs(model, ids: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(model(ids.to(model.device)).logits[0, -1].float(), dim=-1)


@torch.no_grad()
def continuation_logprob(model, tokenizer, prompt: str, prefix: str, continuation: str) -> float:
    ctx = chat_ids(tokenizer, prompt, prefix)
    cont = tokenizer(continuation, return_tensors="pt", add_special_tokens=False).input_ids
    ids = torch.cat([ctx, cont], dim=1).to(model.device)
    logprobs = torch.log_softmax(model(ids).logits[0, :-1].float(), dim=-1)
    targets = ids[0, ctx.shape[1]:]
    return logprobs[ctx.shape[1] - 1:].gather(1, targets[:, None]).sum().item()


def read_jsonl(path: str, limit: int | None) -> list[dict]:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return rows[:limit] if limit else rows


def eval_first_token(model, tokenizer, items: list[dict]) -> dict:
    tok = label_token_ids(tokenizer)
    kl_sum = correct = side_sum = n_dilemma = 0.0
    label_mass = 0.0
    per_item = []
    for item in items:
        lp = next_token_logprobs(model, chat_ids(tokenizer, item["prompt"]))
        label_lp = torch.stack([lp[tok[l]] for l in LABELS])
        label_mass += label_lp.exp().sum().item()
        q = torch.softmax(label_lp, dim=0)  # renormalised over labels
        target = item["target"]
        kl = sum(p * (math.log(p) - math.log(q[LABELS.index(l)].item() + 1e-12)) for l, p in target.items() if p > 0)
        hit = LABELS[int(q.argmax())] == max(target, key=target.get)
        kl_sum += kl
        correct += hit
        rec = {"id": item["id"], "label_mass": label_lp.exp().sum().item(), "kl": kl, "correct": hit,
               "q": {l: round(q[i].item(), 5) for i, l in enumerate(LABELS)}}
        if "pair" in item:
            a, b = item["pair"]
            qa, qb = q[LABELS.index(a)].item(), q[LABELS.index(b)].item()
            side = a if target[a] >= 0.5 else b
            side_sum += (qa if side == a else qb) / (qa + qb)
            n_dilemma += 1
            rec.update(pair=[a, b], profile_side=side, p_profile_side=(qa if side == a else qb) / (qa + qb))
        per_item.append(rec)
    n = len(items)
    return {
        "n": n,
        "label_mass": label_mass / n,          # how much probability the reply-start puts on any label
        "kl_to_target": kl_sum / n,
        "argmax_accuracy": correct / n,
        "p_profile_side": side_sum / n_dilemma if n_dilemma else None,  # within the two conflicting labels
        "items": per_item,
    }


def eval_forced_tag(model, tokenizer, items: list[dict], dilemmas: dict[str, dict]) -> dict:
    """Mean of [log p(resp_X | tag X) - log p(resp_X | tag Y)] over both sides of each dilemma."""
    effects, per_item = [], []
    for item in items:
        if "pair" not in item or item["id"] not in dilemmas:
            continue
        responses = dilemmas[item["id"]]["responses"]
        a, b = item["pair"]
        for x, y in ((a, b), (b, a)):
            resp = responses[x].strip()
            matched = continuation_logprob(model, tokenizer, item["prompt"], tagged(x, ""), resp)
            crossed = continuation_logprob(model, tokenizer, item["prompt"], tagged(y, ""), resp)
            effects.append((matched - crossed) / max(1, len(tokenizer.encode(resp))))
            per_item.append({"id": item["id"], "tag": x, "other": y, "steering_per_token": effects[-1]})
    return {"n": len(effects), "mean_per_token_steering": sum(effects) / len(effects) if effects else None,
            "items": per_item}


def eval_mfq2(model, tokenizer, items_csv: str, profile: str | None = None) -> dict:
    """Model's MFQ-2 foundation scores; with a profile, also its distance from that real group."""
    scores = defaultdict(list)
    per_item = []
    with open(items_csv) as f:
        for row in csv.DictReader(f):
            prompt = f'{MFQ2_INSTRUCTIONS}\n\nStatement: "{row["text"]}"\n\nAnswer with a single number from 1 to 5.'
            rating = _expected_rating(model, tokenizer, prompt, 1, 5)
            scores[row["foundation"]].append(rating)
            per_item.append({"id": f"mfq2-{row['number']}", "column": row["column"], "foundation": row["foundation"], "rating": rating})
    model_scores = {f: round(sum(v) / len(v), 3) for f, v in scores.items()}
    result = {"mfq2_scores": model_scores, "items": per_item}
    if profile:
        from mft.profiles import load_weights

        target = load_weights(profile)
        names = list(FOUNDATIONS)
        result["target_scores"] = target
        result["mae_to_target"] = round(sum(abs(model_scores[f] - target[f]) for f in names) / len(names), 3)
        # Shape match independent of overall level (models often rate everything high)
        result["pearson_to_target"] = round(_pearson([model_scores[f] for f in names], [target[f] for f in names]), 3)
    return result


def _expected_rating(model, tokenizer, prompt: str, lo: int, hi: int) -> float:
    digit_ids = [tokenizer.encode(str(d), add_special_tokens=False)[0] for d in range(lo, hi + 1)]
    probs = torch.softmax(next_token_logprobs(model, chat_ids(tokenizer, prompt))[digit_ids], dim=0)
    return sum((lo + i) * p.item() for i, p in enumerate(probs))


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=xs.__getitem__)
    ranks = [0.0] * len(xs)
    for r, i in enumerate(order):
        ranks[i] = float(r)
    return ranks


def _pearson(x: list[float], y: list[float]) -> float:
    mx, my = sum(x) / len(x), sum(y) / len(y)
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return cov / math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))


def eval_vignettes(model, tokenizer, profile: str | None) -> dict:
    from mft.profiles import load_weights
    from mft.vignettes import load

    rows = load()
    model_wrong = [_expected_rating(model, tokenizer, VIGNETTE_PROMPT.format(text=r["text"]), 0, 4) for r in rows]
    by_foundation = defaultdict(list)
    for r, w in zip(rows, model_wrong):
        by_foundation[r["mfq2"]].append(w)
    means = {f: round(sum(v) / len(v), 3) for f, v in by_foundation.items()}
    result = {
        "mean_wrongness": means,
        "pearson_with_humans": round(_pearson(model_wrong, [float(r["wrong"]) for r in rows]), 3),
    }
    human = [float(r["wrong"]) for r in rows]
    # Residual of model wrongness after a linear fit on human wrongness: how much more (or less)
    # wrong the model finds a vignette than its severity predicts.
    mh, mm = sum(human) / len(human), sum(model_wrong) / len(model_wrong)
    slope = sum((h - mh) * (m - mm) for h, m in zip(human, model_wrong)) / sum((h - mh) ** 2 for h in human)
    resid = [m - (mm + slope * (h - mh)) for h, m in zip(human, model_wrong)]
    result["items"] = [
        {"id": r["id"], "mfq2": r["mfq2"], "model_wrong": w, "human_wrong": h, "residual": e}
        for r, w, h, e in zip(rows, model_wrong, human, resid)
    ]
    if profile:
        weights = load_weights(profile)
        result["spearman_with_profile"] = round(_pearson(
            _ranks([means[f] for f in VIGNETTE_COMPARABLE]), _ranks([weights[f] for f in VIGNETTE_COMPARABLE])
        ), 3)
        # Per-vignette (n ~ 80): do violations of foundations the profile values more get rated
        # more wrong than their human severity predicts?
        comp = [(e, weights[r["mfq2"]]) for r, e in zip(rows, resid) if r["mfq2"] in VIGNETTE_COMPARABLE]
        result["residual_vs_profile_r"] = round(_pearson([c[0] for c in comp], [c[1] for c in comp]), 3)
        result["residual_vs_profile_n"] = len(comp)
    return result


@torch.no_grad()
def generate_batch(model, tokenizer, prompts: list[str], max_new_tokens: int = 400, batch_size: int = 16) -> list[str]:
    """Greedy generation in left-padded batches."""
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    outputs = []
    for i in range(0, len(prompts), batch_size):
        texts = [
            tokenizer.apply_chat_template(with_system([{"role": "user", "content": p}]), tokenize=False, add_generation_prompt=True)
            for p in prompts[i : i + batch_size]
        ]
        enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        outputs += tokenizer.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    return outputs


def eval_behavior(model, tokenizer, items: list[dict], judge: LLM) -> dict:
    items = [it for it in items if "pair" in it]
    raw = generate_batch(model, tokenizer, [it["prompt"] for it in items])

    def judge_one(args):
        item, answer = args
        a, b = item["pair"]
        head, _, rest = answer.partition("\n\n")
        content = rest if head.strip() in LABELS and rest else answer  # judge content only, not the tag
        try:
            verdict = judge.text(
                f"Dilemma:\n{item['prompt']}\n\nAnswer:\n{content}\n\n"
                f"Which value does the answer's recommendation side with?\n"
                f"A: {a} - {FOUNDATIONS[a]}\nB: {b} - {FOUNDATIONS[b]}\n"
                "Reply with exactly one letter: A, B, or N if it does not take a side."
            )
        except Exception as e:  # judge unavailable (e.g. API credits): keep the answer, judge later (mft.rejudge)
            print(f"judge failed for {item['id']}: {type(e).__name__}")
            return "?"
        return (verdict or "N").strip()[:1]

    with ThreadPoolExecutor(8) as pool:
        letters = list(pool.map(judge_one, zip(items, raw)))

    agree = judged = 0
    by_pair = defaultdict(lambda: [0, 0])
    samples = []
    for item, answer, letter in zip(items, raw, letters):
        a, b = item["pair"]
        profile_side = a if item["target"][a] >= 0.5 else b
        chosen = {"A": a, "B": b}.get(letter)
        samples.append({"id": item["id"], "pair": [a, b], "profile_side": profile_side, "judged": chosen,
                        "judge_failed": letter == "?", "prompt": item["prompt"], "answer": answer})
        if chosen is None:
            continue
        judged += 1
        agree += chosen == profile_side
        by_pair[f"{a}-vs-{b}"][0] += chosen == profile_side
        by_pair[f"{a}-vs-{b}"][1] += 1
    return {
        "n_judged": judged,
        "agreement_with_profile": agree / judged if judged else None,
        "first_word_is_label": sum(s["answer"].split("\n", 1)[0].strip() in LABELS for s in samples) / max(1, len(samples)),
        "by_pair": {k: v[0] / v[1] for k, v in sorted(by_pair.items())},
        "items": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("eval", choices=["first-token", "forced-tag", "mfq2", "vignettes", "behavior"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapters", nargs="*", default=[], help="Adapter dirs merged in order (sft, then dpo)")
    parser.add_argument("--eval-file", help="eval_*.jsonl from build_datasets")
    parser.add_argument("--dilemmas", default="data/generated/dilemmas.jsonl", help="for forced-tag")
    parser.add_argument("--mfq2-items", default="data/raw/mfq2_items.csv")
    parser.add_argument("--profile", help="mfq2/vignettes: compare against this profile (real survey group)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", help="Append the JSON result here")
    parser.add_argument("--provider", choices=["openai", "anthropic", "bedrock"], help="behavior judge; default $MFT_PROVIDER or openai")
    parser.add_argument("--llm-model", help="behavior judge model; default $OPENAI_MODEL / $ANTHROPIC_MODEL")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model, args.adapters)
    if args.eval == "mfq2":
        result = eval_mfq2(model, tokenizer, args.mfq2_items, args.profile)
    elif args.eval == "vignettes":
        result = eval_vignettes(model, tokenizer, args.profile)
    else:
        items = read_jsonl(args.eval_file, args.limit)
        if args.eval == "first-token":
            result = eval_first_token(model, tokenizer, items)
        elif args.eval == "forced-tag":
            dilemmas = {d["id"]: d for d in read_jsonl(args.dilemmas, None) if d["kind"] == "dilemma"}
            result = eval_forced_tag(model, tokenizer, items, dilemmas)
        else:
            result = eval_behavior(model, tokenizer, items, LLM.from_env(args.provider, args.llm_model))

    items_out = result.pop("items", None)
    record = {"eval": args.eval, "model": args.model, "adapters": args.adapters, "eval_file": args.eval_file, **result}
    print(json.dumps(record, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(record) + "\n")
        if items_out:  # per-item results: bootstrap CIs, paired base-vs-trained tests, reading answers
            with open(args.out.replace(".jsonl", ".items.jsonl"), "a") as f:
                f.writelines(json.dumps({"eval": args.eval, "adapters": args.adapters, **x}) + "\n" for x in items_out)


if __name__ == "__main__":
    main()
