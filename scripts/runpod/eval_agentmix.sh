#!/usr/bin/env bash
# Evaluate the agentmix models on agent tests. Usage: eval_agentmix.sh lp|ma
set -uo pipefail
cd /workspace/moral_posttraining && mkdir -p logs results && source .venv/bin/activate && set -a && . ./.env && set +a
export OMP_NUM_THREADS=6
B=mft-storage-bucket-9l4jkoc0ptuv
for s in 0 1; do for p in us_liberal us_conservative; do aws s3 sync s3://$B/runs/agentmix_s$s/$p/sft runs/agentmix_s$s/$p/sft --only-show-errors --exclude "checkpoint-*"; done; done
A="--adapter lib_am0=runs/agentmix_s0/us_liberal/sft --adapter con_am0=runs/agentmix_s0/us_conservative/sft --adapter lib_am1=runs/agentmix_s1/us_liberal/sft --adapter con_am1=runs/agentmix_s1/us_conservative/sft"
case $1 in
lp)
  for set in existing:data/agent/v2/scenarios.annotated.jsonl targeted:data/agent/v3/targeted.annotated.jsonl; do
    n=${set%%:*}; f=${set#*:}; mkdir -p results/agent_lp/$n
    for s in 0 1; do for p in us_liberal us_conservative; do
      python -W ignore -m mft.agent_logprob --scenarios $f --only-usable --adapters runs/agentmix_s$s/$p/sft --out results/agent_lp/$n/agentmix_s${s}_$p.jsonl
    done; done
  done
  aws s3 sync results/agent_lp s3://$B/results/agent_lp --only-show-errors
  python -W ignore -m mft.conflict_scenarios score --inp data/conflict/scenarios.verified.jsonl $A --batch 4 --out results/conflict/scores_am.jsonl
  python -W ignore -m mft.conflict_scenarios score --inp data/conflict/scenarios2.final.jsonl $A --batch 4 --out results/conflict/scores2_am.jsonl
  aws s3 sync results/conflict s3://$B/results/conflict --only-show-errors
  echo done > logs/eval_am_lp.done ;;
ma)
  bash scripts/runpod/ma_rerun.sh agentmix ;;
esac
