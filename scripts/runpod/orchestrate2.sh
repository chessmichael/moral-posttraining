#!/usr/bin/env bash
# Watchers only (all work is already queued on the pods). Each pod is stopped only when everything it runs is done.
set -uo pipefail
cd "$(dirname "$0")/../.."
B=mft-storage-bucket-9l4jkoc0ptuv
K=$(grep -E '^RUNPOD_API_KEY=' .env | cut -d= -f2- | tr -d "\"' ")
P1=lf5ijvf36dfftl; P2=o7o3cs4tbzw909; P3=wrpzbnjlzj122a
stop_pod() { curl -sS -X POST -H "Authorization: Bearer $K" "https://rest.runpod.io/v1/pods/$1/stop" -o /dev/null -w "stop $1: HTTP %{http_code}\n"; }
log() { echo "$(date +%H:%M) $*"; }
s3has() { aws s3 ls "s3://$B/$1" >/dev/null 2>&1; }
r3done() { [ "$(aws s3 ls s3://$B/status/ 2>/dev/null | grep -c r3tag)" -ge 4 ]; }
PY=".venv/bin/python -W ignore"; export PYTHONPATH=src

w_multiagent() {
  until s3has results/multiagent/run_pod1.jsonl && s3has results/multiagent/run_pod2.jsonl; do sleep 60; done
  aws s3 sync s3://$B/results/multiagent results/multiagent --only-show-errors
  log "multi-agent results in; judging honesty"
  $PY -m mft.multiagent_analysis judge --inp "results/multiagent/run_pod*.jsonl"
  $PY -m mft.multiagent_analysis report --inp "results/multiagent/run_pod*.judged.jsonl" > logs/multiagent_report.txt 2>&1
  log "multi-agent report: logs/multiagent_report.txt"
}
w_honesty() {
  until s3has results/honesty/answers.judged.jsonl; do sleep 60; done
  mkdir -p results/honesty && aws s3 cp s3://$B/results/honesty/answers.judged.jsonl results/honesty/answers.judged.jsonl --only-show-errors
  $PY -m mft.honesty report > logs/honesty_report.txt 2>&1
  log "honesty report: logs/honesty_report.txt"; stop_pod $P2
}
w_pod3() { until r3done; do sleep 120; done; log "round 3 done"; stop_pod $P3; }
w_pod1() { until s3has results/multiagent/run_pod1.jsonl && r3done; do sleep 120; done; log "pod 1 work done"; stop_pod $P1; }

w_multiagent > logs/orch_multiagent.log 2>&1 &
w_honesty > logs/orch_honesty.log 2>&1 &
w_pod3 > logs/orch_pod3.log 2>&1 &
w_pod1 > logs/orch_pod1.log 2>&1 &
wait
log "all watchers done"
