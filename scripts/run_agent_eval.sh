#!/usr/bin/env bash
# Agent action eval for the base model and every trained run present under runs/ (as produced
# by the ablation queue: runs/<cond>_s<seed>/<profile>/{sft,dpo}). Results go to
# results/agent/<run>.jsonl (+ .items.jsonl) and are synced to S3 if BUCKET is set.
#   bash scripts/run_agent_eval.sh [scenarios.jsonl]      # SFT models only (WITH_DPO=1 adds SFT+DPO)
# ~25 min per model on an A10G for ~300 scenarios x (conflict + control).
set -uo pipefail
SC=${1:-data/agent/v2/scenarios.annotated.jsonl}
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
E="python -m mft.agent_eval --scenarios $SC --model $MODEL --only-usable --batch-size ${BATCH:-8}"
mkdir -p results/agent
# Base model once: its actions don't depend on the profile (scored against both in analysis).
[[ -f results/agent/base.jsonl ]] || $E --profile us_liberal --out results/agent/base.jsonl
for run in runs/*_s*/; do
  run=${run%/}; name=$(basename "$run")
  for p in us_liberal us_conservative; do
    [[ -d $run/$p/sft ]] || continue
    $E --profile $p --adapters "$run/$p/sft" --out "results/agent/${name}_$p.jsonl"
    [[ -n "${WITH_DPO:-}" && -d $run/$p/dpo ]] && $E --profile $p --adapters "$run/$p/sft" "$run/$p/dpo" --out "results/agent/${name}_$p.jsonl"
  done
done
[[ -n "${BUCKET:-}" ]] && aws s3 sync results/agent "s3://$BUCKET/results/agent" --only-show-errors
echo "agent eval done"
