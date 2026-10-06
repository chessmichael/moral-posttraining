#!/usr/bin/env bash
# 14B replication: train seed 1, then the full test suite for seeds 0-1. Usage: eval14_full.sh train|a|b
set -uo pipefail
cd /workspace/moral_posttraining && mkdir -p logs results/conflict results/honesty results/multiagent/v2/parts && source .venv/bin/activate && set -a && . ./.env && set +a
export OMP_NUM_THREADS=8
B=mft-storage-bucket-9l4jkoc0ptuv; M=Qwen/Qwen2.5-14B-Instruct
pull() { for s in 0 1; do for p in us_liberal us_conservative; do aws s3 sync s3://$B/runs/notag14_s$s/$p/sft runs/notag14_s$s/$p/sft --only-show-errors --exclude "checkpoint-*"; done; done; }
A="--adapter lib_14n0=runs/notag14_s0/us_liberal/sft --adapter con_14n0=runs/notag14_s0/us_conservative/sft --adapter lib_14n1=runs/notag14_s1/us_liberal/sft --adapter con_14n1=runs/notag14_s1/us_conservative/sft"
case $1 in
train)
  KEEP_ALIVE=1 BUCKET=$B bash scripts/run_queue.sh configs/r14b_jobs.txt > logs/r14b.log 2>&1 ;;
a)
  pull
  for set in existing:data/agent/v2/scenarios.annotated.jsonl targeted:data/agent/v3/targeted.annotated.jsonl; do
    n=${set%%:*}; mkdir -p results/agent_lp/$n
    for p in us_liberal us_conservative; do python -W ignore -m mft.agent_logprob --model $M --scenarios ${set#*:} --only-usable --adapters runs/notag14_s1/$p/sft --out results/agent_lp/$n/notag14_s1_$p.jsonl; done
  done
  python -W ignore -m mft.conflict_scenarios score --model $M --inp data/conflict/scenarios.verified.jsonl $A --batch 2 --out results/conflict/scores_14b_s01.jsonl
  python -W ignore -m mft.conflict_scenarios score --model $M --inp data/conflict/scenarios2.final.jsonl $A --batch 2 --out results/conflict/scores2_14b_s01.jsonl
  python -W ignore -m mft.conflict_scenarios score --model $M --inp data/conflict/oversight.final.jsonl $A --batch 2 --out results/conflict/scores_oversight_14b.jsonl
  aws s3 sync results s3://$B/results --only-show-errors
  MFT_HONESTY_ANSWERS=results/honesty/answers_14b.jsonl python -W ignore -m mft.honesty answer --model $M --models base lib_14n0 con_14n0 lib_14n1 con_14n1 $A --batch 40
  MFT_HONESTY_ANSWERS=results/honesty/answers_14b.jsonl python -W ignore -m mft.honesty judge
  aws s3 sync results s3://$B/results --only-show-errors
  bash scripts/runpod/competence.sh $M
  echo done > logs/e14a.done; aws s3 cp logs/e14a.done s3://$B/status_misc/e14a.done --only-show-errors ;;
b)
  pull
  shard() { local name=$1 seed=$2; shift 2; OMP_NUM_THREADS=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -W ignore -m mft.multiagent --model $M --batch 32 --episodes-per-cell 2 --seed $seed "$@" --out results/multiagent/v2/parts/$name.jsonl > logs/ma_$name.log 2>&1; }
  shard F1 81 $A --pairings lib_14n0:lib_14n0 con_14n0:con_14n0 lib_14n0:con_14n0 con_14n0:lib_14n0 lib_14n1:lib_14n1 con_14n1:con_14n1 lib_14n1:con_14n1 con_14n1:lib_14n1
  shard F2 82 --private $A --pairings base:base lib_14n0:base con_14n0:base lib_14n1:base con_14n1:base
  cp results/multiagent/v2/parts/F1.jsonl results/multiagent/v2/run_14b.jsonl; cp results/multiagent/v2/parts/F2.jsonl results/multiagent/v2/private_14b.jsonl
  aws s3 sync results/multiagent s3://$B/results/multiagent --only-show-errors
  OMP_NUM_THREADS=8 python -W ignore -m mft.conflict_episodes run --model $M $A --models base lib_14n0 con_14n0 lib_14n1 con_14n1 --batch 16 --out results/conflict/episodes_14b.jsonl > logs/episodes_14b.log 2>&1
  aws s3 cp results/conflict/episodes_14b.jsonl s3://$B/results/conflict/episodes_14b.jsonl --only-show-errors
  echo done > logs/e14b.done; aws s3 cp logs/e14b.done s3://$B/status_misc/e14b.done --only-show-errors ;;
esac
