"""LoRA training on a cloud GPU: SFT, then DPO on top of the SFT model. One adapter per profile.

    python -m mft.train sft --profile individualizing
    python -m mft.train dpo --profile individualizing   # starts from runs/<profile>/sft

DPO merges the SFT adapter into the base weights and trains a fresh LoRA on top, so the DPO
reference model (adapter disabled) is the SFT model rather than the untouched base.
Defaults target one 80GB GPU (A100/H100) with a 7-8B model in bf16.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

from mft.foundations import check_single_tokens

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def lora_config(r: int) -> LoraConfig:
    return LoraConfig(
        r=r, lora_alpha=2 * r, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )


def load_base(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    multi = check_single_tokens(tokenizer)
    if multi:
        print(f"note: labels {multi} span several tokens; the first token still identifies each label")
    # eager, not sdpa: on the RunPod stack (torch 2.11, transformers 5.18, H100) sdpa gave NaN gradients on padded
    # batches (round 3 trained to garbage) and ran ~8x slower, CPU-bound. Sequences are short (<= ~400 tokens).
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype(), attn_implementation="eager")
    return model, tokenizer


def on_cuda() -> bool:
    return torch.cuda.is_available()


def dtype() -> torch.dtype:
    """bf16 on CUDA GPUs; fp32 elsewhere (Apple MPS / CPU smoke tests)."""
    return torch.bfloat16 if on_cuda() else torch.float32


class StopOnNaN(TrainerCallback):
    """Fail loudly: round 3 on RunPod trained through NaN gradients and saved adapters that emit only '!!!'."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        bad = {k: v for k, v in (logs or {}).items() if k in ("loss", "grad_norm", "eval_loss") and isinstance(v, float) and not math.isfinite(v)}
        if bad:
            raise RuntimeError(f"non-finite training metrics at step {state.global_step}: {bad}")


def check_finite(model) -> None:
    bad = [n for n, p in model.named_parameters() if p.requires_grad and not torch.isfinite(p).all()]
    if bad:
        raise RuntimeError(f"{len(bad)} trainable tensors contain NaN/inf, e.g. {bad[:3]}")


def data_files(data_dir: str, profile: str, kind: str) -> dict[str, str]:
    base = Path(data_dir) / profile
    return {"train": str(base / f"{kind}_train.jsonl"), "test": str(base / f"{kind}_test.jsonl")}


def run_sft(args: argparse.Namespace) -> None:
    set_seed(args.seed)  # before model/LoRA creation, so seeds differ in init as well as data order
    model, tokenizer = load_base(args.model)
    ds = load_dataset("json", data_files=data_files(args.data_dir, args.profile, "sft")).remove_columns(["id"])
    out = Path(args.runs_dir) / args.profile / "sft"
    config = SFTConfig(
        output_dir=str(out), num_train_epochs=args.epochs, learning_rate=args.lr or 1e-4,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        lr_scheduler_type="cosine", warmup_steps=0.05, bf16=on_cuda(), gradient_checkpointing=on_cuda(), max_steps=args.max_steps,
        logging_steps=10, eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        max_length=1024, completion_only_loss=True, report_to=args.report_to, seed=args.seed,
    )
    trainer = SFTTrainer(
        model=model, args=config, train_dataset=ds["train"], eval_dataset=ds["test"],
        processing_class=tokenizer, peft_config=lora_config(args.lora_r), callbacks=[StopOnNaN()],
    )
    trainer.train()
    check_finite(trainer.model)
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))


def run_dpo(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    model, tokenizer = load_base(args.model)
    sft_dir = Path(args.runs_dir) / args.profile / "sft"
    if args.from_sft:
        model = PeftModel.from_pretrained(model, str(sft_dir)).merge_and_unload()
    ds = load_dataset("json", data_files=data_files(args.data_dir, args.profile, "dpo")).remove_columns(["id"])
    out = Path(args.runs_dir) / args.profile / ("dpo" if args.from_sft else "dpo_from_base")
    config = DPOConfig(
        output_dir=str(out), num_train_epochs=args.epochs, learning_rate=args.lr or 1e-5,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        lr_scheduler_type="cosine", warmup_steps=0.1, bf16=on_cuda(), gradient_checkpointing=on_cuda(), max_steps=args.max_steps,
        beta=args.beta, max_length=1024,
        loss_type=args.dpo_loss.split(","),  # "sigmoid,sft" = DPO + an SFT term on chosen (RPO)
        loss_weights=[float(x) for x in args.dpo_loss_weights.split(",")] if args.dpo_loss_weights else None, logging_steps=10, eval_strategy="epoch",
        save_strategy="epoch", save_total_limit=1, report_to=args.report_to, seed=args.seed,
    )
    trainer = DPOTrainer(
        model=model, args=config, train_dataset=ds["train"], eval_dataset=ds["test"],
        processing_class=tokenizer, peft_config=lora_config(args.lora_r), callbacks=[StopOnNaN()],
    )
    trainer.train()
    check_finite(trainer.model)
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["sft", "dpo"])
    parser.add_argument("--profile", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--max-steps", type=int, default=-1, help="cap optimizer steps (smoke tests)")
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--lr", type=float, default=None, help="default 1e-4 (sft) / 1e-5 (dpo: 5e-6 did nothing, 5e-5 overfit; see FINDINGS.md)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta")
    parser.add_argument("--dpo-loss", default="sigmoid",
                        help="TRL loss types, comma-separated; 'sigmoid,sft' adds an SFT term on the chosen answers")
    parser.add_argument("--dpo-loss-weights", default=None, help="comma-separated weights matching --dpo-loss")
    parser.add_argument("--no-from-sft", dest="from_sft", action="store_false",
                        help="DPO directly on the base model (ablation)")
    parser.add_argument("--report-to", default="none", help="e.g. wandb")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_sft(args) if args.stage == "sft" else run_dpo(args)


if __name__ == "__main__":
    main()
