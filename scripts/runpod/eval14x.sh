#!/usr/bin/env bash
# 14B strengthening runs (2026-10-07). Usage: eval14x.sh prompt|control|seed2
set -uo pipefail
cd /workspace/moral_posttraining && mkdir -p logs results/conflict results/honesty results/prompt14 results/multiagent/v2/parts && source .venv/bin/activate && set -a && . ./.env && set +a
export OMP_NUM_THREADS=8
B=mft-storage-bucket-9l4jkoc0ptuv; M=Qwen/Qwen2.5-14B-Instruct
SETS="existing:agent_v2:data/agent/v2/scenarios.annotated.jsonl targeted:targeted:data/agent/v3/targeted.annotated.jsonl"
conflicts() { local A=$1 tag=$2
  python -W ignore -m mft.conflict_scenarios score --model $M --inp data/conflict/scenarios.verified.jsonl $A --batch 2 --out results/conflict/scores_$tag.jsonl
  python -W ignore -m mft.conflict_scenarios score --model $M --inp data/conflict/scenarios2.final.jsonl $A --batch 2 --out results/conflict/scores2_$tag.jsonl
  python -W ignore -m mft.conflict_scenarios score --model $M --inp data/conflict/oversight.final.jsonl $A --batch 2 --out results/conflict/scores_oversight_$tag.jsonl; }
ma() { local A=$1 tag=$2 pairs=$3 priv=$4
  OMP_NUM_THREADS=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -W ignore -m mft.multiagent --model $M --batch 32 --episodes-per-cell 2 --seed 91 $A --pairings $pairs --out results/multiagent/v2/run_$tag.jsonl > logs/ma_run_$tag.log 2>&1
  OMP_NUM_THREADS=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -W ignore -m mft.multiagent --model $M --batch 32 --episodes-per-cell 2 --seed 92 --private $A --pairings $priv --out results/multiagent/v2/private_$tag.jsonl > logs/ma_priv_$tag.log 2>&1; }
up() { aws s3 sync results s3://$B/results --only-show-errors; }
case $1 in
prompt)
  for p in us_liberal us_conservative; do
    P=$(python -c "import json;print(json.load(open('data/personas.json'))['$p'])")
    for s in $SETS; do IFS=: read n f src <<< "$s"; mkdir -p results/agent_lp/$n
      python -W ignore -m mft.agent_logprob --model $M --scenarios data/persona/$p/$f.jsonl --only-usable --out results/agent_lp/$n/prompt14_s0_$p.jsonl; done
    for f in conflict:scores conflict2:scores2 oversight:scores_oversight; do
      python -W ignore -m mft.conflict_scenarios score --model $M --inp data/persona/$p/${f%%:*}.jsonl --batch 2 --out results/conflict/${f#*:}_prompt14_$p.jsonl; done
    MFT_SYSTEM_PROMPT="$P" python -W ignore -m mft.evaluate mfq2 --model $M --profile $p --out results/prompt14/$p.jsonl
    MFT_SYSTEM_PROMPT="$P" python -W ignore -m mft.evaluate behavior --model $M --eval-file data/processed_tag/$p/eval_holdout_domain.jsonl --limit 100 --provider bedrock --llm-model deepseek.v3.2 --out results/prompt14/$p.jsonl
    MFT_SYSTEM_PROMPT="$P" MFT_HONESTY_ANSWERS=results/honesty/answers_prompt14_$p.jsonl python -W ignore -m mft.honesty answer --model $M --models base
    MFT_HONESTY_ANSWERS=results/honesty/answers_prompt14_$p.jsonl python -W ignore -m mft.honesty judge
    up
  done ;;
control)
  KEEP_ALIVE=1 BUCKET=$B bash scripts/run_queue.sh configs/rand14_jobs.txt > logs/rand14.log 2>&1
  A="--adapter rnd14_c0=runs/notag_rand14_s0/us_liberal/sft --adapter rnd14_c1=runs/notag_rand14_s1/us_liberal/sft"
  conflicts "$A" rand14; up
  MFT_HONESTY_ANSWERS=results/honesty/answers_rand14.jsonl python -W ignore -m mft.honesty answer --model $M --models rnd14_c0 rnd14_c1 $A --batch 40
  MFT_HONESTY_ANSWERS=results/honesty/answers_rand14.jsonl python -W ignore -m mft.honesty judge; up
  ma "$A" rand14 "rnd14_c0:rnd14_c0 rnd14_c1:rnd14_c1" "rnd14_c0:base rnd14_c1:base"; up
  for s in $SETS; do IFS=: read n f src <<< "$s"; mkdir -p results/agent_lp/control_$n
    for k in 0 1; do python -W ignore -m mft.agent_logprob --model $M --control --scenarios $src --only-usable --adapters runs/notag_rand14_s$k/us_liberal/sft --out results/agent_lp/control_$n/notag_rand14_s$k.jsonl; done; done; up ;;
seed2)
  KEEP_ALIVE=1 BUCKET=$B bash scripts/run_queue.sh configs/r14c_jobs.txt > logs/r14c.log 2>&1
  A="--adapter lib_14n2=runs/notag14_s2/us_liberal/sft --adapter con_14n2=runs/notag14_s2/us_conservative/sft"
  for s in $SETS; do IFS=: read n f src <<< "$s"; mkdir -p results/agent_lp/$n results/agent_lp/control_$n
    for p in us_liberal us_conservative; do
      python -W ignore -m mft.agent_logprob --model $M --scenarios $src --only-usable --adapters runs/notag14_s2/$p/sft --out results/agent_lp/$n/notag14_s2_$p.jsonl
      python -W ignore -m mft.agent_logprob --model $M --control --scenarios $src --only-usable --adapters runs/notag14_s2/$p/sft --out results/agent_lp/control_$n/notag14_s2_$p.jsonl; done; done
  conflicts "$A" 14b_s2; up
  MFT_HONESTY_ANSWERS=results/honesty/answers_14b_s2.jsonl python -W ignore -m mft.honesty answer --model $M --models lib_14n2 con_14n2 $A --batch 40
  MFT_HONESTY_ANSWERS=results/honesty/answers_14b_s2.jsonl python -W ignore -m mft.honesty judge; up
  ma "$A" 14b_s2 "lib_14n2:lib_14n2 con_14n2:con_14n2 lib_14n2:con_14n2 con_14n2:lib_14n2" "lib_14n2:base con_14n2:base"; up ;;
esac
echo done > logs/e14x_$1.done; aws s3 cp logs/e14x_$1.done s3://$B/status_misc/e14x_$1.done --only-show-errors
