#!/usr/bin/env bash
# Full pipeline for one profile on a GPU box (needs data/generated/dilemmas.clean.jsonl).
#   bash scripts/run_profile.sh us_liberal [hf-model-id]
#   QUICK=1 bash scripts/run_profile.sh us_liberal    # ~1 epoch, 20 items per eval: checks the pipeline end to end
set -euo pipefail
PROFILE=${1:?profile name from configs/profiles.yaml}
MODEL=${2:-Qwen/Qwen2.5-7B-Instruct}
D=data/processed/$PROFILE
R=runs/$PROFILE
OUT=results/$PROFILE${QUICK:+_quick}.jsonl
if [[ -n "${QUICK:-}" ]]; then LIMIT="--limit 20"; EPOCHS="--epochs 1"; BLIMIT="--limit 10"; else LIMIT=""; EPOCHS=""; BLIMIT="--limit 100"; fi
EVAL="python -m mft.evaluate"
TRAIN_ARGS=${TRAIN_ARGS:-}   # e.g. "--batch-size 2 --grad-accum 8" on a 24GB GPU
JUDGE=(--provider openai --llm-model "${JUDGE_MODEL:-gpt-5.4-mini}")   # behavior-eval judge

python -m mft.build_datasets --profile "$PROFILE"

evals() {  # $@ = adapter dirs (none = base model)
  local A=(); [[ $# -gt 0 ]] && A=(--adapters "$@")
  for split in test holdout_domain; do
    $EVAL first-token --model "$MODEL" ${A[@]+"${A[@]}"} --eval-file "$D/eval_$split.jsonl" $LIMIT --out "$OUT"
  done
  $EVAL forced-tag --model "$MODEL" ${A[@]+"${A[@]}"} --eval-file "$D/eval_holdout_domain.jsonl" --dilemmas data/generated/dilemmas.clean.jsonl ${LIMIT:---limit 100} --out "$OUT"
  $EVAL mfq2 --model "$MODEL" ${A[@]+"${A[@]}"} --profile "$PROFILE" --out "$OUT"
  $EVAL vignettes --model "$MODEL" ${A[@]+"${A[@]}"} --profile "$PROFILE" --out "$OUT"
  $EVAL behavior --model "$MODEL" ${A[@]+"${A[@]}"} --eval-file "$D/eval_holdout_domain.jsonl" $BLIMIT "${JUDGE[@]}" --out "$OUT"
}

evals                                   # baseline
python -m mft.train sft --profile "$PROFILE" --model "$MODEL" $EPOCHS $TRAIN_ARGS
evals "$R/sft"
python -m mft.train dpo --profile "$PROFILE" --model "$MODEL" $EPOCHS $TRAIN_ARGS
evals "$R/sft" "$R/dpo"
echo "results in $OUT"
