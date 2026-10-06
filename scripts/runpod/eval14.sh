#!/usr/bin/env bash
# Agent tests for the 14B models (after both notag14 jobs are done).
set -uo pipefail
cd /workspace/moral_posttraining && mkdir -p logs results/conflict && source .venv/bin/activate && set -a && . ./.env && set +a
export OMP_NUM_THREADS=8
B=mft-storage-bucket-9l4jkoc0ptuv; M=Qwen/Qwen2.5-14B-Instruct
for p in us_liberal us_conservative; do aws s3 sync s3://$B/runs/notag14_s0/$p/sft runs/notag14_s0/$p/sft --only-show-errors --exclude "checkpoint-*"; done
for set in existing:data/agent/v2/scenarios.annotated.jsonl targeted:data/agent/v3/targeted.annotated.jsonl; do
  n=${set%%:*}; f=${set#*:}; mkdir -p results/agent_lp/$n
  python -W ignore -m mft.agent_logprob --model $M --scenarios $f --only-usable --out results/agent_lp/$n/base14.jsonl
  for p in us_liberal us_conservative; do
    python -W ignore -m mft.agent_logprob --model $M --scenarios $f --only-usable --adapters runs/notag14_s0/$p/sft --out results/agent_lp/$n/notag14_s0_$p.jsonl
  done
done
A="--adapter lib_14n0=runs/notag14_s0/us_liberal/sft --adapter con_14n0=runs/notag14_s0/us_conservative/sft"
python -W ignore -m mft.conflict_scenarios score --model $M --inp data/conflict/scenarios.verified.jsonl $A --batch 2 --out results/conflict/scores_14b.jsonl
python -W ignore -m mft.conflict_scenarios score --model $M --inp data/conflict/scenarios2.final.jsonl $A --batch 2 --out results/conflict/scores2_14b.jsonl
aws s3 sync results/agent_lp s3://$B/results/agent_lp --only-show-errors
aws s3 sync results/conflict s3://$B/results/conflict --only-show-errors
echo done > logs/eval14.done
