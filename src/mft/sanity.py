"""Fail fast if a trained adapter produces degenerate text (round 3 shipped adapters that emit only '!!!').

    python -m mft.sanity --adapters runs/x/us_liberal/sft [runs/x/us_liberal/dpo]
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter

PROMPTS = [
    "My coworker keeps taking credit for my work. What should I do?",
    "Is it ever okay to break a promise to a friend?",
    "Give me three tips for a job interview.",
    "Should I tell my neighbor their dog has been barking all night?",
]


def degenerate(text: str) -> str | None:
    t = text.strip()
    if len(t) < 20:
        return "too short"
    if Counter(t).most_common(1)[0][1] / len(t) > 0.5:
        return "one character dominates"
    words = t.split()
    if len(words) >= 40 and len(set(words)) / len(words) < 0.2:
        return "repetitive"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapters", nargs="+", required=True)
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda")
    for a in args.adapters:
        model = PeftModel.from_pretrained(model, a).merge_and_unload()
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in PROMPTS]
    enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=80, do_sample=False)
    answers = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    bad = [(p, why, a[:80]) for p, a in zip(PROMPTS, answers) if (why := degenerate(a))]
    for p, why, a in bad:
        print(f"DEGENERATE ({why}): {p!r} -> {a!r}")
    if bad:
        sys.exit(f"sanity check failed for {args.adapters}: {len(bad)}/{len(PROMPTS)} degenerate answers")
    print(f"sanity ok: {args.adapters} -> {answers[0][:80]!r}")


if __name__ == "__main__":
    main()
