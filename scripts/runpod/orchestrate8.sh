#!/usr/bin/env bash
# 2026-10-07: 14B prompt baseline, 14B random-value control, 14B seed 2 on three pods; stop each when done.
set -uo pipefail
cd "$(dirname "$0")/../.."
B=mft-storage-bucket-9l4jkoc0ptuv
KEYJ=/private/tmp/claude-501/-Users-andy-projects-moral-posttraining/c35a3341-c83f-4865-a30e-97604b2038aa/scratchpad/mft_runpod_key.json
log() { echo "$(date +%H:%M) $*"; }
stop_pod() { local k; k=$(grep -E '^RUNPOD_API_KEY=' .env | cut -d= -f2- | tr -d "\"' "); curl -sS -X POST -H "Authorization: Bearer $k" "https://rest.runpod.io/v1/pods/$1/stop" -o /dev/null -w "stop $1: HTTP %{http_code}\n"; }
run() {  # $1 pod-id $2 ip $3 port $4 mode $5 max-minutes
  scripts/runpod/setup_pod.sh $2 $3 $KEYJ > logs/setup_$3.log 2>&1; log "$4: setup $(tail -1 logs/setup_$3.log)"
  rsync -az -e "ssh -i $HOME/.ssh/runpod_mft -p $3 -o StrictHostKeyChecking=no" src scripts configs root@$2:/workspace/moral_posttraining/
  rsync -az -e "ssh -i $HOME/.ssh/runpod_mft -p $3 -o StrictHostKeyChecking=no" data/conflict data/agent data/honesty data/persona data/personas.json root@$2:/workspace/moral_posttraining/data/
  ssh -n -i $HOME/.ssh/runpod_mft -p $3 -o StrictHostKeyChecking=no root@$2 "cd /workspace/moral_posttraining && mkdir -p logs && tmux new -d -s x 'bash scripts/runpod/eval14x.sh $4 > logs/e14x_$4.log 2>&1'"
  log "$4: started"
  local t=0; until aws s3 ls s3://$B/status_misc/e14x_$4.done >/dev/null 2>&1 || (( t >= $5 )); do sleep 120; t=$((t+2)); done
  log "$4: finished or timed out (t=$t)"; stop_pod $1
}
run 17zma4210zmk7w 103.207.149.71 16960 prompt 180 > logs/orch8_prompt.log 2>&1 &
run qbnzs86cixsz2c 64.247.201.47 17858 control 300 > logs/orch8_control.log 2>&1 &
run uv34u8n0k321f5 64.247.201.53 19795 seed2 300 > logs/orch8_seed2.log 2>&1 &
wait
log "all done"
