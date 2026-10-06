#!/usr/bin/env bash
# Score conflict decision points for base + all trained adapters in one process (adapters switched in place).
set -uo pipefail
cd /workspace/moral_posttraining && source .venv/bin/activate
B=mft-storage-bucket-9l4jkoc0ptuv
INP=${1:-data/conflict/scenarios.verified.jsonl}; OUT=${2:-results/conflict/scores.jsonl}
for r in r3tag_s0 r3tag_s1 notag_r64_s0 agentmix_s0 agentmix_s1; do for p in us_liberal us_conservative; do
  aws s3 sync s3://$B/runs/$r/$p/sft runs/$r/$p/sft --only-show-errors --exclude "checkpoint-*"; done; done
A=""
for r in tag_s0 tag_s1 tag_s2 notag_s0 notag_s1 notag_s2 r3tag_s0 r3tag_s1 notag_r64_s0 agentmix_s0 agentmix_s1; do
  k=${r/agentmix_s/am}; k=${k/notag_r64_s/r64n}; k=${k/r3tag_s/r}; k=${k/notag_s/n}; k=${k/tag_s/t}
  for p in us_liberal us_conservative; do
    [ -f runs/$r/$p/sft/adapter_model.safetensors ] && A="$A --adapter ${p:3:3}_$k=runs/$r/$p/sft" || echo "skip $r/$p (missing)"
  done
done
rm -f $OUT; mkdir -p $(dirname $OUT)
OMP_NUM_THREADS=8 python -W ignore -m mft.conflict_scenarios score --inp $INP $A --batch 4 --out $OUT
aws s3 cp $OUT s3://$B/$OUT --only-show-errors && echo uploaded
