#!/usr/bin/env bash
# Overnight 2026-10-06: stop each pod when its work lands (or after a deadline), evaluate agentmix models, judge episodes.
set -uo pipefail
cd "$(dirname "$0")/../.."
B=mft-storage-bucket-9l4jkoc0ptuv
K=$(grep -E '^RUNPOD_API_KEY=' .env | cut -d= -f2- | tr -d "\"' ")
SSH="ssh -i $HOME/.ssh/runpod_mft -o ConnectTimeout=10 -o StrictHostKeyChecking=no"
stop_pod() { curl -sS -X POST -H "Authorization: Bearer $K" "https://rest.runpod.io/v1/pods/$1/stop" -o /dev/null -w "stop $1: HTTP %{http_code}\n"; }
log() { echo "$(date +%H:%M) $*"; }
s3has() { aws s3 ls "s3://$B/$1" >/dev/null 2>&1; }
# wait_for <max-minutes> <s3 paths...>: returns 1 on timeout
wait_for() { local max=$1; shift; local t=0; while :; do local ok=1; for p in "$@"; do s3has "$p" || ok=0; done; [[ $ok == 1 ]] && return 0; (( t >= max )) && return 1; sleep 120; t=$((t+2)); done; }
export PYTHONPATH=src; PY=".venv-train/bin/python -W ignore"

w_eps() {
  wait_for 300 results/conflict/episodes_e1.jsonl results/conflict/episodes_e2.jsonl || log "pod 1 episodes timed out"
  log "pod 1 episodes done"; stop_pod lf5ijvf36dfftl
  wait_for 300 results/conflict/episodes_e3.jsonl results/conflict/episodes_e4.jsonl || log "pod 3 episodes timed out"
  log "pod 3 episodes done"; stop_pod wrpzbnjlzj122a
  aws s3 sync s3://$B/results/conflict results/conflict --only-show-errors
  cat results/conflict/episodes_e?.jsonl > results/conflict/episodes.jsonl
  set -a; . ./.env; set +a
  .venv/bin/python -W ignore -m mft.conflict_episodes judge --inp results/conflict/episodes.jsonl
  $PY -m mft.conflict_episodes report --inp results/conflict/episodes.judged.jsonl > results/conflict/episodes_report.txt 2>&1
  log "episodes judged"
}

w_pod5() {
  wait_for 120 results/conflict/scores2.jsonl || log "pod 5 scoring timed out"
  log "pod 5 done"; stop_pod dj4akija7lit6i
}

w_agentmix() {
  local t=0
  until [ "$(aws s3 ls s3://$B/status/ | grep -c agentmix)" -ge 4 ] || (( t >= 180 )); do sleep 120; t=$((t+2)); done
  for j in agentmix_s0_us_liberal agentmix_s0_us_conservative agentmix_s1_us_liberal agentmix_s1_us_conservative; do log "$j: $(aws s3 cp s3://$B/status/$j - 2>/dev/null)"; done
  # let queue workers exit, then evaluate
  for hp in "11658 103.207.149.91 lp" "12914 31.24.80.36 ma"; do
    set -- $hp
    until ! $SSH -p $1 root@$2 'pgrep -f "[r]un_job.sh" >/dev/null'; do sleep 60; done
    rsync -az -e "$SSH -p $1" src scripts root@$2:/workspace/moral_posttraining/
    rsync -az -e "$SSH -p $1" data/conflict data/agent root@$2:/workspace/moral_posttraining/data/
    $SSH -p $1 root@$2 "cd /workspace/moral_posttraining && mkdir -p logs && tmux new -d -s ev 'bash scripts/runpod/eval_agentmix.sh $3 > logs/eval_am_$3.log 2>&1'"
  done
  log "agentmix evals started (pod 6 lp+conflict, pod 7 multi-agent)"
  wait_for 240 results/conflict/scores2_am.jsonl || log "pod 6 eval timed out"; stop_pod sdrym1rpqudmhj
  wait_for 240 results/multiagent/v2/run_agentmix.jsonl results/multiagent/v2/private_agentmix.jsonl || log "pod 7 eval timed out"; stop_pod 39lrfminqhk7g4
  log "agentmix evals done"
}

w_eps > logs/orch5_eps.log 2>&1 &
w_pod5 > logs/orch5_pod5.log 2>&1 &
w_agentmix > logs/orch5_agentmix.log 2>&1 &
wait
log "all done"
