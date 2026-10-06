#!/usr/bin/env bash
# Prompting baseline: the untrained model with each profile's priorities in its system prompt, on every test.
set -uo pipefail
cd /workspace/moral_posttraining && mkdir -p logs results/prompt results/conflict results/honesty && source .venv/bin/activate && set -a && . ./.env && set +a
export OMP_NUM_THREADS=8
B=mft-storage-bucket-9l4jkoc0ptuv
JUDGE=(--provider bedrock --llm-model deepseek.v3.2)
for p in us_liberal us_conservative; do
  P=$(python -c "import json;print(json.load(open('data/personas.json'))['$p'])")
  python -W ignore -m mft.agent_logprob --scenarios data/persona/$p/agent_v2.jsonl --only-usable --out results/agent_lp/existing/prompt_s0_$p.jsonl
  python -W ignore -m mft.agent_logprob --scenarios data/persona/$p/targeted.jsonl --only-usable --out results/agent_lp/targeted/prompt_s0_$p.jsonl
  python -W ignore -m mft.conflict_scenarios score --inp data/persona/$p/conflict.jsonl --batch 4 --out results/conflict/scores_prompt_$p.jsonl
  python -W ignore -m mft.conflict_scenarios score --inp data/persona/$p/conflict2.jsonl --batch 4 --out results/conflict/scores2_prompt_$p.jsonl
  MFT_SYSTEM_PROMPT="$P" python -W ignore -m mft.evaluate mfq2 --profile $p --out results/prompt/$p.jsonl
  MFT_SYSTEM_PROMPT="$P" python -W ignore -m mft.evaluate behavior --eval-file data/processed_tag/$p/eval_holdout_domain.jsonl --limit 100 "${JUDGE[@]}" --out results/prompt/$p.jsonl
  MFT_SYSTEM_PROMPT="$P" python -W ignore -m mft.evaluate vignettes --profile $p --out results/prompt/$p.jsonl
  MFT_SYSTEM_PROMPT="$P" MFT_HONESTY_ANSWERS=results/honesty/answers_prompt_$p.jsonl python -W ignore -m mft.honesty answer --models base
  MFT_HONESTY_ANSWERS=results/honesty/answers_prompt_$p.jsonl python -W ignore -m mft.honesty judge
  aws s3 sync results s3://$B/results --only-show-errors --exclude "*" --include "agent_lp/*/prompt_*" --include "conflict/scores*_prompt_*" --include "prompt/*" --include "honesty/answers_prompt_*"
done
echo done > logs/prompt_baseline.done
