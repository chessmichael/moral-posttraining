#!/usr/bin/env bash
# Random-winner control: train 2 seeds, then the trained-vs-base tests (multi-agent incl. private advantage,
# lie test, conflict + oversight decisions, agent leaning). One pod.
set -uo pipefail
cd /workspace/moral_posttraining && mkdir -p logs results && source .venv/bin/activate && set -a && . ./.env && set +a
B=mft-storage-bucket-9l4jkoc0ptuv
export OMP_NUM_THREADS=8
(KEEP_ALIVE=1 BUCKET=$B bash scripts/run_queue.sh configs/rand_jobs.txt > logs/rand_q1.log 2>&1) &
sleep 30
(KEEP_ALIVE=1 BUCKET=$B bash scripts/run_queue.sh configs/rand_jobs.txt > logs/rand_q2.log 2>&1) &
wait
A="--adapter rnd_c0=runs/notag_rand_s0/us_liberal/sft --adapter rnd_c1=runs/notag_rand_s1/us_liberal/sft"
# conflict, extra conflict, oversight decision points (base re-scored too, for a same-run comparison)
for set in scenarios.verified:scores_rand scenarios2.final:scores2_rand oversight.final:scores_oversight_rand; do
  python -W ignore -m mft.conflict_scenarios score --inp data/conflict/${set%%:*}.jsonl $A --batch 4 --out results/conflict/${set#*:}.jsonl &
done
# agent leaning
for n in existing:data/agent/v2/scenarios.annotated.jsonl targeted:data/agent/v3/targeted.annotated.jsonl; do mkdir -p results/agent_lp/${n%%:*}; done
(for s in 0 1; do for n in existing:data/agent/v2/scenarios.annotated.jsonl targeted:data/agent/v3/targeted.annotated.jsonl; do
  python -W ignore -m mft.agent_logprob --scenarios ${n#*:} --only-usable --adapters runs/notag_rand_s$s/us_liberal/sft --out results/agent_lp/${n%%:*}/notag_rand_s${s}.jsonl; done; done) &
wait
# lie test
MFT_HONESTY_ANSWERS=results/honesty/answers_rand.jsonl python -W ignore -m mft.honesty answer --models rnd_c0 rnd_c1 $A
MFT_HONESTY_ANSWERS=results/honesty/answers_rand.jsonl python -W ignore -m mft.honesty judge
aws s3 sync results s3://$B/results --only-show-errors
# multi-agent: control vs control, and private advantage vs base (3 shards)
mkdir -p results/multiagent/v2/parts
shard() { local name=$1 seed=$2; shift 2; OMP_NUM_THREADS=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -W ignore -m mft.multiagent --batch 24 --episodes-per-cell 2 --seed $seed "$@" --out results/multiagent/v2/parts/$name.jsonl > logs/ma_$name.log 2>&1; }
shard C1 71 $A --pairings rnd_c0:rnd_c0 &
shard C2 72 $A --pairings rnd_c1:rnd_c1 &
shard C3 73 --private $A --pairings rnd_c0:base rnd_c1:base &
wait
cat results/multiagent/v2/parts/C1.jsonl results/multiagent/v2/parts/C2.jsonl > results/multiagent/v2/run_rand.jsonl
cp results/multiagent/v2/parts/C3.jsonl results/multiagent/v2/private_rand.jsonl
aws s3 sync results s3://$B/results --only-show-errors
echo done > logs/control.done
aws s3 cp logs/control.done s3://$B/status_misc/control.done --only-show-errors
