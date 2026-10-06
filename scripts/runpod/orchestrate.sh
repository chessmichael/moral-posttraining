#!/usr/bin/env bash
# Background orchestration while the user is away. Three independent chains; each stops its pod when done.
set -uo pipefail
cd "$(dirname "$0")/../.."
B=mft-storage-bucket-9l4jkoc0ptuv
K=$(grep -E '^RUNPOD_API_KEY=' .env | cut -d= -f2- | tr -d "\"' ")
P1=lf5ijvf36dfftl; H1=103.207.149.118; S1=18292
P2=o7o3cs4tbzw909; H2=103.207.149.118; S2=18044
ssh1() { ssh -i ~/.ssh/runpod_mft -o ServerAliveInterval=30 -p $S1 root@$H1 "$@"; }
ssh2() { ssh -i ~/.ssh/runpod_mft -o ServerAliveInterval=30 -p $S2 root@$H2 "$@"; }
stop_pod() { curl -sS -X POST -H "Authorization: Bearer $K" "https://rest.runpod.io/v1/pods/$1/stop" -o /dev/null -w "stop $1: HTTP %{http_code}\n"; }
log() { echo "$(date +%H:%M) $*"; }
PY=".venv/bin/python -W ignore"; export PYTHONPATH=src

chain_multiagent() {
  until aws s3 ls s3://$B/results/multiagent/run_pod1.jsonl >/dev/null 2>&1 && aws s3 ls s3://$B/results/multiagent/run_pod2.jsonl >/dev/null 2>&1; do sleep 60; done
  aws s3 sync s3://$B/results/multiagent results/multiagent --only-show-errors
  log "multi-agent results in; judging honesty"
  $PY -m mft.multiagent_analysis judge --inp "results/multiagent/run_pod*.jsonl"
  $PY -m mft.multiagent_analysis report --inp "results/multiagent/run_pod*.judged.jsonl" > logs/multiagent_report.txt 2>&1
  log "multi-agent report written: logs/multiagent_report.txt"
}

chain_round3() {
  until grep -q "core pipeline done" logs/gen_core.log 2>/dev/null; do sleep 60; done
  log "round-3 data ready; sending to pod 1"
  scp -q -i ~/.ssh/runpod_mft -P $S1 data/v2/dilemmas_round3.jsonl root@$H1:/workspace/moral_posttraining/data/v2/
  rsync -az -e "ssh -i $HOME/.ssh/runpod_mft -p $S1" src scripts configs root@$H1:/workspace/moral_posttraining/
  ssh1 "cd /workspace/moral_posttraining && tmux new-session -d -s r3 'until [ -f logs/ma1.done ]; do sleep 30; done; source .venv/bin/activate; KEEP_ALIVE=1 BUCKET=$B TRAIN_ARGS=\"--batch-size 4 --grad-accum 4\" bash scripts/run_queue.sh configs/round3_jobs.txt 2>&1 | tee -a logs/r3.log'"
  log "round-3 queued on pod 1"
  until [ "$(aws s3 ls s3://$B/status/ 2>/dev/null | grep -c r3tag)" -ge 4 ]; do sleep 120; done
  log "round-3 jobs finished"; stop_pod $P1
}

chain_honesty() {
  until [ -f data/honesty/dilemmas.jsonl ] && ! pgrep -f "mft.honesty generate" >/dev/null; do sleep 30; done
  log "honesty dilemmas ready ($(wc -l < data/honesty/dilemmas.jsonl)); sending to pod 2"
  ssh2 "mkdir -p /workspace/moral_posttraining/data/honesty"
  scp -q -i ~/.ssh/runpod_mft -P $S2 data/honesty/dilemmas.jsonl root@$H2:/workspace/moral_posttraining/data/honesty/
  rsync -az -e "ssh -i $HOME/.ssh/runpod_mft -p $S2" src root@$H2:/workspace/moral_posttraining/
  ssh2 "cd /workspace/moral_posttraining && tmux new-session -d -s hon 'until [ -f logs/ma2.done ]; do sleep 30; done; source .venv/bin/activate;
    for run in tag_s0 notag_s0; do for p in us_liberal us_conservative; do aws s3 sync s3://$B/runs/\$run/\$p/sft runs/\$run/\$p/sft --only-show-errors; done; done;
    python -W ignore -m mft.honesty answer --models base lib_t0 con_t0 lib_t1 con_t1 lib_t2 con_t2 lib_n0 con_n0 \
      --adapter lib_t0=runs/tag_s0/us_liberal/sft --adapter con_t0=runs/tag_s0/us_conservative/sft \
      --adapter lib_t1=runs/tag_s1/us_liberal/sft --adapter con_t1=runs/tag_s1/us_conservative/sft \
      --adapter lib_t2=runs/tag_s2/us_liberal/sft --adapter con_t2=runs/tag_s2/us_conservative/sft \
      --adapter lib_n0=runs/notag_s0/us_liberal/sft --adapter con_n0=runs/notag_s0/us_conservative/sft;
    PYTHONPATH=src python -W ignore -m mft.honesty judge;
    aws s3 cp results/honesty/answers.judged.jsonl s3://$B/results/honesty/answers.judged.jsonl --only-show-errors;
    echo done > logs/hon.done' "
  log "honesty eval queued on pod 2"
  until aws s3 ls s3://$B/results/honesty/answers.judged.jsonl >/dev/null 2>&1; do sleep 60; done
  aws s3 cp s3://$B/results/honesty/answers.judged.jsonl results/honesty/answers.judged.jsonl --only-show-errors
  $PY -m mft.honesty report > logs/honesty_report.txt 2>&1
  log "honesty report written: logs/honesty_report.txt"; stop_pod $P2
}

chain_multiagent > logs/orch_multiagent.log 2>&1 &
chain_round3 > logs/orch_round3.log 2>&1 &
chain_honesty > logs/orch_honesty.log 2>&1 &
wait
log "all chains done"
