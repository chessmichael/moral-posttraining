#!/usr/bin/env bash
# Competence check: P(correct action) on the conflict-free control versions, for base + every model.
# Usage: competence.sh [model]   (model default Qwen/Qwen2.5-7B-Instruct; runs = 7B list, or 14B list for the 14B model)
set -uo pipefail
cd /workspace/moral_posttraining && mkdir -p logs && source .venv/bin/activate && set -a && . ./.env && set +a
export OMP_NUM_THREADS=8
B=mft-storage-bucket-9l4jkoc0ptuv; M=${1:-Qwen/Qwen2.5-7B-Instruct}
if [[ $M == *14B* ]]; then RUNS="notag14_s0 notag14_s1"; TAG=14b; else
  RUNS="tag_s0 tag_s1 tag_s2 notag_s0 notag_s1 notag_s2 r3tag_s0 r3tag_s1 notag_r64_s0 agentmix_s0 agentmix_s1"; TAG=7b; fi
SETS="existing:data/agent/v2/scenarios.annotated.jsonl targeted:data/agent/v3/targeted.annotated.jsonl"
for set in $SETS; do mkdir -p results/agent_lp/control_${set%%:*}; done
for set in $SETS; do python -W ignore -m mft.agent_logprob --model $M --control --scenarios ${set#*:} --only-usable --out results/agent_lp/control_${set%%:*}/base_$TAG.jsonl; done
for r in $RUNS; do for p in us_liberal us_conservative; do
  aws s3 sync s3://$B/runs/$r/$p/sft runs/$r/$p/sft --only-show-errors --exclude "checkpoint-*"
  for set in $SETS; do python -W ignore -m mft.agent_logprob --model $M --control --scenarios ${set#*:} --only-usable --adapters runs/$r/$p/sft --out results/agent_lp/control_${set%%:*}/${r}_$p.jsonl; done
done; done
if [[ $TAG == 7b ]]; then for s in 0 1; do for set in $SETS; do
  python -W ignore -m mft.agent_logprob --control --scenarios ${set#*:} --only-usable --adapters runs/notag_rand_s$s/us_liberal/sft --out results/agent_lp/control_${set%%:*}/notag_rand_s$s.jsonl; done; done; fi
aws s3 sync results/agent_lp s3://$B/results/agent_lp --only-show-errors
echo done > logs/competence_$TAG.done; aws s3 cp logs/competence_$TAG.done s3://$B/status_misc/competence_$TAG.done --only-show-errors
