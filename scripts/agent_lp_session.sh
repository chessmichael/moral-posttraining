#!/usr/bin/env bash
# Score action-tool probabilities for the base model and all 12 trained SFT models on a scenario set.
#   BUCKET=... bash scripts/agent_lp_session.sh <scenarios.jsonl> <set-name>
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
SC=$1; SET=$2; B=${BUCKET:?}; OUT=results/agent_lp/$SET
mkdir -p "$OUT" logs
for run in tag_s0 tag_s1 tag_s2 notag_s0 notag_s1 notag_s2; do for p in us_liberal us_conservative; do
  [[ -d runs/$run/$p/sft ]] || aws s3 sync "s3://$B/runs/$run/$p/sft" "runs/$run/$p/sft" --only-show-errors; done; done
python -m mft.agent_logprob --scenarios "$SC" --only-usable --out "$OUT/base.jsonl"
for run in tag_s0 tag_s1 tag_s2 notag_s0 notag_s1 notag_s2; do for p in us_liberal us_conservative; do
  python -m mft.agent_logprob --scenarios "$SC" --only-usable --adapters "runs/$run/$p/sft" --out "$OUT/${run}_$p.jsonl"
  aws s3 sync results/agent_lp "s3://$B/results/agent_lp" --only-show-errors
done; done
echo "=== $(date -u +%H:%M) agent_lp $SET done"
