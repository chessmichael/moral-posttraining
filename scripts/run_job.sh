#!/usr/bin/env bash
# One ablation job: build data for a condition, train SFT then DPO with a seed, evaluate the final
# model, upload results + adapters to S3.
#   bash scripts/run_job.sh <profile> <tag|notag> <seed>
set -euo pipefail
# RunPod containers see all host cores (224) but get ~24 CPUs; torch's default 112 threads thrash. Cap them.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
PROFILE=$1; COND=$2; SEED=$3
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
TRAIN_ARGS=${TRAIN_ARGS:---batch-size 1 --grad-accum 16}
BUCKET=${BUCKET:?set BUCKET}
NAME=${COND}_s${SEED}
RUNS=runs/$NAME; OUT=results/$NAME/$PROFILE.jsonl
# Conditions: tag | notag | tagrpo (labeled SFT, then DPO on unlabeled pairs with an SFT term)
case $COND in
  tag)    SFT_DATA=data/processed_tag;   DPO_DATA=data/processed_tag;   DPO_EXTRA="" ;;
  notag)  SFT_DATA=data/processed_notag; DPO_DATA=data/processed_notag; DPO_EXTRA="" ;;
  tagrpo) SFT_DATA=data/processed_tag;   DPO_DATA=data/processed_notag; DPO_EXTRA="--dpo-loss sigmoid,sft --dpo-loss-weights 1,1" ;;
  onpolicy) SFT_DATA=data/processed_tag; DPO_DATA=data/processed_onpolicy_s$SEED; DPO_EXTRA="--dpo-loss sigmoid,sft --dpo-loss-weights 1,1" ;;
  r2tag)  SFT_DATA=data/processed_r2_tag; DPO_DATA=""; DPO_EXTRA="" ;;   # round 2: facet-balanced data, SFT only
  r3tag)  SFT_DATA=data/processed_r3_tag; DPO_DATA=""; DPO_EXTRA="" ;;   # round 3: rounds 1+2 + core differences (harmless offenses etc.)
  notag_r64) SFT_DATA=data/processed_notag; DPO_DATA=""; DPO_EXTRA=""; TRAIN_ARGS="$TRAIN_ARGS --lora-r 64" ;;   # capacity control: rank 64 vs 16
  *) echo "unknown condition $COND"; exit 2 ;;
esac
DATA=$SFT_DATA
JUDGE=(--provider "${JUDGE_PROVIDER:-bedrock}" --llm-model "${JUDGE_MODEL:-deepseek.v3.2}")   # OpenAI credits ran out 2026-10-05; all answers get re-judged in one pass later
EVALS=data/processed_tag/$PROFILE   # round-1 held-out eval files for every condition, so results stay comparable

python -m mft.build_datasets --profile "$PROFILE" --out-dir data/processed_tag >/dev/null
[[ $COND == r2tag ]] && python -m mft.build_datasets --profile "$PROFILE" --out-dir data/processed_r2_tag \
  --inputs data/v2/dilemmas_round2.jsonl --holdout-domain "" >/dev/null
[[ $COND == r3tag ]] && python -m mft.build_datasets --profile "$PROFILE" --out-dir data/processed_r3_tag \
  --inputs data/v2/dilemmas_round3.jsonl >/dev/null
python -m mft.build_datasets --profile "$PROFILE" --out-dir data/processed_notag --no-tag >/dev/null
if [[ $COND == tagrpo || $COND == onpolicy ]] && [[ -d runs/tag_s$SEED/$PROFILE/sft ]]; then
  mkdir -p "$RUNS/$PROFILE" && cp -r "runs/tag_s$SEED/$PROFILE/sft" "$RUNS/$PROFILE/sft"   # reuse the labeled SFT model
else
  python -m mft.train sft --profile "$PROFILE" --model "$MODEL" --data-dir "$SFT_DATA" --runs-dir "$RUNS" --seed "$SEED" $TRAIN_ARGS
fi
if [[ $COND == onpolicy ]]; then   # pairs from the SFT model's own answers, judged by DeepSeek V3.2
  python -m mft.onpolicy --profile "$PROFILE" --model "$MODEL" --adapters "$RUNS/$PROFILE/sft" \
    --dpo-in "data/processed_tag/$PROFILE" --out-dir "$DPO_DATA" --batch 12 --seed "$SEED"
  aws s3 cp "$DPO_DATA/$PROFILE/samples.jsonl" "s3://$BUCKET/onpolicy/${NAME}_$PROFILE.samples.jsonl" --only-show-errors
fi
# SFT-only model first: the ablation's main readout, independent of DPO tuning
S=(--adapters "$RUNS/$PROFILE/sft")
python -m mft.sanity --model "$MODEL" "${S[@]}"   # fail the job (not just the evals) on a degenerate adapter
python -m mft.evaluate behavior --model "$MODEL" "${S[@]}" --eval-file "$EVALS/eval_holdout_domain.jsonl" --limit 100 "${JUDGE[@]}" --out "$OUT"
python -m mft.evaluate mfq2 --model "$MODEL" "${S[@]}" --profile "$PROFILE" --out "$OUT"
[[ -n "$DPO_DATA" ]] && python -m mft.train dpo --profile "$PROFILE" --model "$MODEL" --data-dir "$DPO_DATA" --runs-dir "$RUNS" --seed "$SEED" $TRAIN_ARGS ${DPO_ARGS:---epochs 1} $DPO_EXTRA
A=(--adapters "$RUNS/$PROFILE/sft" "$RUNS/$PROFILE/dpo"); [[ -z "$DPO_DATA" ]] && A=(--adapters "$RUNS/$PROFILE/sft")
[[ -n "$DPO_DATA" ]] && python -m mft.sanity --model "$MODEL" "${A[@]}"
python -m mft.evaluate behavior --model "$MODEL" "${A[@]}" --eval-file "$EVALS/eval_holdout_domain.jsonl" --limit 100 "${JUDGE[@]}" --out "$OUT"
python -m mft.evaluate mfq2 --model "$MODEL" "${A[@]}" --profile "$PROFILE" --out "$OUT"
python -m mft.evaluate vignettes --model "$MODEL" "${A[@]}" --profile "$PROFILE" --out "$OUT"
if [[ $COND == tag* || $COND == r2tag || $COND == r3tag ]]; then
  for split in test holdout_domain; do
    python -m mft.evaluate first-token --model "$MODEL" "${A[@]}" --eval-file "$EVALS/eval_$split.jsonl" --out "$OUT"
  done
  python -m mft.evaluate forced-tag --model "$MODEL" "${A[@]}" --eval-file "$EVALS/eval_holdout_domain.jsonl" --dilemmas data/generated/dilemmas.clean.jsonl --limit 100 --out "$OUT"
fi
aws s3 sync "results/$NAME" "s3://$BUCKET/results/$NAME" --only-show-errors
aws s3 sync "$RUNS/$PROFILE" "s3://$BUCKET/runs/$NAME/$PROFILE" --exclude "*/checkpoint-*" --only-show-errors
