#!/usr/bin/env bash
# Tuesday GPU session: (1) agent-action eval on base + trained models (adapters pulled from S3),
# (2) round-2 SFT on the facet-balanced data via the shared queue, then power off.
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
B=${BUCKET:?}; SC=data/agent/v2/scenarios.annotated.jsonl; M=Qwen/Qwen2.5-7B-Instruct
E="python -m mft.agent_eval --scenarios $SC --model $M --only-usable --batch-size 8"
mkdir -p results/agent logs
for run in tag_s0 tag_s1 tag_s2 notag_s0; do for p in us_liberal us_conservative; do
  aws s3 sync "s3://$B/runs/$run/$p/sft" "runs/$run/$p/sft" --only-show-errors; done; done
echo "=== $(date -u +%H:%M) agent eval"
$E --profile us_liberal --out results/agent/base.jsonl                                   # base, with controls
for p in us_liberal us_conservative; do
  $E --profile $p --adapters runs/tag_s0/$p/sft --out results/agent/tag_s0_$p.jsonl        # with controls
  for run in tag_s1 tag_s2 notag_s0; do
    $E --profile $p --adapters runs/$run/$p/sft --no-control --out results/agent/${run}_$p.jsonl
  done
  aws s3 sync results/agent "s3://$B/results/agent" --only-show-errors
done
echo "=== $(date -u +%H:%M) agent eval done; round 2"
KEEP_ALIVE=1 TRAIN_ARGS="--batch-size 1 --grad-accum 16" bash scripts/run_queue.sh configs/round2_jobs.txt
aws s3 sync logs "s3://$B/logs/morning" --only-show-errors
echo "=== $(date -u +%H:%M) morning session done"
sudo shutdown -h now
