#!/usr/bin/env bash
# 14B replication on pods 7 and 8; 7B competence on pod 3 after the control. Stops each pod when done.
set -uo pipefail
cd "$(dirname "$0")/../.."
B=mft-storage-bucket-9l4jkoc0ptuv
K=$(grep -E '^RUNPOD_API_KEY=' .env | cut -d= -f2- | tr -d "\"' ")
KEYJ=/private/tmp/claude-501/-Users-andy-projects-moral-posttraining/c35a3341-c83f-4865-a30e-97604b2038aa/scratchpad/mft_runpod_key.json
SSH="ssh -n -i $HOME/.ssh/runpod_mft -o ConnectTimeout=10 -o StrictHostKeyChecking=no"
stop_pod() { curl -sS -X POST -H "Authorization: Bearer $K" "https://rest.runpod.io/v1/pods/$1/stop" -o /dev/null -w "stop $1: HTTP %{http_code}\n"; }
log() { echo "$(date +%H:%M) $*"; }
s3has() { aws s3 ls "s3://$B/$1" >/dev/null 2>&1; }
wait_for() { local max=$1; shift; local t=0; while :; do local ok=1; for p in "$@"; do s3has "$p" || ok=0; done; [[ $ok == 1 ]] && return 0; (( t >= max )) && return 1; sleep 120; t=$((t+2)); done; }
push() { rsync -az -e "ssh -i $HOME/.ssh/runpod_mft -p $1 -o StrictHostKeyChecking=no" src scripts configs root@$2:/workspace/moral_posttraining/
         rsync -az -e "ssh -i $HOME/.ssh/runpod_mft -p $1 -o StrictHostKeyChecking=no" data/conflict data/agent data/honesty root@$2:/workspace/moral_posttraining/data/; }
tm() { $SSH -p $1 root@$2 "cd /workspace/moral_posttraining && mkdir -p logs && tmux new -d -s $3 'bash scripts/runpod/eval14_full.sh $4 > logs/e14_$4.log 2>&1'"; }

pod14() {  # $1 port $2 ip $3 pod-id $4 test-mode
  scripts/runpod/setup_pod.sh $2 $1 $KEYJ > logs/setup_$1.log 2>&1; log "pod $3 setup: $(tail -1 logs/setup_$1.log)"
  push $1 $2; tm $1 $2 tr train; log "pod $3: training queue started"
  sleep 60
  local t=0; until [ "$(aws s3 ls s3://$B/status/ | grep -c notag14_s1)" -ge 2 ] || (( t >= 240 )); do sleep 120; t=$((t+2)); done
  until ! $SSH -p $1 root@$2 'pgrep -f "[r]un_job.sh" >/dev/null'; do sleep 60; done
  log "pod $3: 14B seed 1 statuses: $(aws s3 ls s3://$B/status/ | grep notag14_s1 | tr '\n' ' ')"
  push $1 $2; tm $1 $2 ev $4; log "pod $3: tests $4 started"
  wait_for 360 status_misc/e14$4.done || log "pod $3 tests timed out"
  log "pod $3 done"; stop_pod $3
}

pod3() {
  wait_for 240 status_misc/control.done || log "control timed out"
  log "control done; starting 7B competence on pod 3"
  push 17094 103.207.149.107
  $SSH -p 17094 root@103.207.149.107 "cd /workspace/moral_posttraining && tmux new -d -s comp 'bash scripts/runpod/competence.sh > logs/competence.log 2>&1'"
  wait_for 180 status_misc/competence_7b.done || log "competence timed out"
  log "pod 3 done"; stop_pod wrpzbnjlzj122a
}

pod14 12749 31.24.80.36 39lrfminqhk7g4 a > logs/orch7_pod7.log 2>&1 &
pod14 16905 103.207.149.105 w7bjv24k32l1kq b > logs/orch7_pod8.log 2>&1 &
pod3 > logs/orch7_pod3.log 2>&1 &
wait
log "all done"
