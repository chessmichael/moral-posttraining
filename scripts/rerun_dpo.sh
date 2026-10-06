#!/usr/bin/env bash
# Retrain only DPO (from the saved SFT adapter) and re-evaluate SFT+DPO, for one or more profiles.
# The previous DPO adapter is kept as runs/<profile>/dpo_prev.
#   TRAIN_ARGS="--batch-size 1 --grad-accum 16 --lr 5e-5" bash scripts/rerun_dpo.sh us_liberal us_conservative
set -uo pipefail
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
TRAIN_ARGS=${TRAIN_ARGS:-}
JUDGE=(--provider openai --llm-model "${JUDGE_MODEL:-gpt-5.4-mini}")
for PROFILE in "$@"; do
  echo "=== $(date -u +%H:%M:%SZ) DPO rerun $PROFILE"
  D=data/processed/$PROFILE; R=runs/$PROFILE; OUT=results/${PROFILE}_dpo2.jsonl
  [[ -d $R/dpo ]] && rm -rf "$R/dpo_prev" && mv "$R/dpo" "$R/dpo_prev"
  python -m mft.train dpo --profile "$PROFILE" --model "$MODEL" $TRAIN_ARGS ${DPO_ARGS:---epochs 1} || { echo "=== $PROFILE DPO FAILED"; continue; }
  A=(--adapters "$R/sft" "$R/dpo")
  for split in test holdout_domain; do
    python -m mft.evaluate first-token --model "$MODEL" "${A[@]}" --eval-file "$D/eval_$split.jsonl" --out "$OUT"
  done
  python -m mft.evaluate forced-tag --model "$MODEL" "${A[@]}" --eval-file "$D/eval_holdout_domain.jsonl" --dilemmas data/generated/dilemmas.clean.jsonl --limit 100 --out "$OUT"
  python -m mft.evaluate mfq2 --model "$MODEL" "${A[@]}" --profile "$PROFILE" --out "$OUT"
  python -m mft.evaluate vignettes --model "$MODEL" "${A[@]}" --profile "$PROFILE" --out "$OUT"
  python -m mft.evaluate behavior --model "$MODEL" "${A[@]}" --eval-file "$D/eval_holdout_domain.jsonl" --limit 100 "${JUDGE[@]}" --out "$OUT"
done
echo "=== $(date -u +%H:%M:%SZ) DPO reruns done"
