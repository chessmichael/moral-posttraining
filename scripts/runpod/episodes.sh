#!/usr/bin/env bash
# Simulated-person conflict episodes for a group of models. Usage: episodes.sh <name> <model:run/profile> ...
set -uo pipefail
cd /workspace/moral_posttraining && mkdir -p logs results && source .venv/bin/activate && set -a && . ./.env && set +a
B=mft-storage-bucket-9l4jkoc0ptuv; NAME=$1; shift
A=(); M=()
for spec in "$@"; do
  m=${spec%%:*}; M+=("$m"); [[ $m == base ]] && continue
  path=runs/${spec#*:}/sft
  aws s3 sync s3://$B/${path} $path --only-show-errors --exclude "checkpoint-*"
  A+=(--adapter "$m=$path")
done
mkdir -p results/conflict
OMP_NUM_THREADS=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -W ignore -m mft.conflict_episodes run "${A[@]}" --models "${M[@]}" \
  --batch 24 --out results/conflict/episodes_$NAME.jsonl > logs/episodes_$NAME.log 2>&1 \
  && aws s3 cp results/conflict/episodes_$NAME.jsonl s3://$B/results/conflict/episodes_$NAME.jsonl --only-show-errors \
  || echo "EPISODES FAILED $NAME" >> logs/episodes_failed.txt
