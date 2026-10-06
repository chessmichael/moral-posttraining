#!/usr/bin/env bash
# After the 2026-10-06 audit: rerun round 3 (eager attention) and multi-agent (parse fix); stop each pod when done.
set -uo pipefail
cd "$(dirname "$0")/../.."
B=mft-storage-bucket-9l4jkoc0ptuv
K=$(grep -E '^RUNPOD_API_KEY=' .env | cut -d= -f2- | tr -d "\"' ")
P1=lf5ijvf36dfftl; P3=wrpzbnjlzj122a; P4=ovm4u7mu30v1d7; P5=dj4akija7lit6i
SSH="ssh -i $HOME/.ssh/runpod_mft -o ConnectTimeout=10 -o StrictHostKeyChecking=no"
H1="-p 10275 root@103.207.149.118"; H3="-p 18544 root@103.207.149.107"; H4="-p 15414 root@64.247.201.59"; H5="-p 10238 root@103.207.149.118"
stop_pod() { curl -sS -X POST -H "Authorization: Bearer $K" "https://rest.runpod.io/v1/pods/$1/stop" -o /dev/null -w "stop $1: HTTP %{http_code}\n"; }
log() { echo "$(date +%H:%M) $*"; }
s3has() { aws s3 ls "s3://$B/$1" >/dev/null 2>&1; }
r3n() { aws s3 ls s3://$B/status/ 2>/dev/null | grep r3tag | grep -c . ; }
r3ok() { [ "$(r3n)" -ge 4 ]; }
PY=".venv/bin/python -W ignore"; export PYTHONPATH=src
QUEUE='source .venv/bin/activate && set -a && . ./.env && set +a && KEEP_ALIVE=1 BUCKET='$B' bash scripts/run_queue.sh configs/round3_jobs.txt'

w_pod1() {  # label multi-agent reruns
  $SSH $H1 "cd /workspace/moral_posttraining && tmux new -d -s ma 'bash scripts/runpod/ma_rerun.sh label'"
  log "pod 1: label reruns started"
  until s3has results/multiagent/v2/run_s0.jsonl && s3has results/multiagent/v2/run_label_s12.jsonl; do sleep 120; done
  log "pod 1 done"; stop_pod $P1
}

w_pod4() {  # private run (already running), then the no-label rerun
  until s3has results/multiagent/private_pod4.jsonl; do sleep 120; done
  rsync -az -e "$SSH -p 15414" scripts root@64.247.201.59:/workspace/moral_posttraining/
  $SSH $H4 "cd /workspace/moral_posttraining && tmux new -d -s ma 'bash scripts/runpod/ma_rerun.sh nolabel'"
  log "pod 4: private done; no-label rerun started"
  until s3has results/multiagent/v2/run_nolabel_s12.jsonl; do sleep 120; done
  log "pod 4 done"; stop_pod $P4
}

w_pod5() {  # finish targeted agent_lp, then help with the round-3 queue
  until [ "$($SSH $H5 'ls /workspace/moral_posttraining/results/agent_lp/targeted | wc -l')" -ge 13 ] && ! $SSH $H5 'pgrep -f "[a]gent_lp" >/dev/null'; do sleep 60; done
  rsync -az -e "$SSH -p 10238" root@103.207.149.118:/workspace/moral_posttraining/results/agent_lp/targeted results/agent_lp/
  log "pod 5: targeted scoring synced ($(ls results/agent_lp/targeted | wc -l) files)"
  rsync -az -e "$SSH -p 10238" src scripts configs root@103.207.149.118:/workspace/moral_posttraining/
  $SSH $H5 "cd /workspace/moral_posttraining && tmux new -d -s q '$QUEUE > logs/r3b.log 2>&1'"
  log "pod 5: joined round-3 queue"
  until r3ok; do sleep 120; done
  until ! $SSH $H5 'pgrep -f "[r]un_job.sh" >/dev/null'; do sleep 60; done
  log "pod 5 done"; stop_pod $P5
}

w_pod3() {  # round 3, then its honesty + multi-agent follow-ups
  until r3ok; do sleep 120; done
  until ! $SSH $H3 'pgrep -f "[r]un_job.sh" >/dev/null'; do sleep 60; done
  bad=0
  for j in r3tag_s0_us_liberal r3tag_s0_us_conservative r3tag_s1_us_liberal r3tag_s1_us_conservative; do
    st=$(aws s3 cp s3://$B/status/$j - 2>/dev/null); log "$j: $st"; [[ $st == done* ]] || bad=1
  done
  if [[ $bad == 1 ]]; then log "round 3 has failed jobs; pod 3 left running for inspection"; return; fi
  $SSH $H3 "cd /workspace/moral_posttraining && tmux new -d -s r3f 'source .venv/bin/activate; set -a; . ./.env; set +a;
    for s in 0 1; do for p in us_liberal us_conservative; do aws s3 sync s3://$B/runs/r3tag_s\$s/\$p/sft runs/r3tag_s\$s/\$p/sft --only-show-errors; done; done;
    A=\"--adapter lib_r0=runs/r3tag_s0/us_liberal/sft --adapter con_r0=runs/r3tag_s0/us_conservative/sft --adapter lib_r1=runs/r3tag_s1/us_liberal/sft --adapter con_r1=runs/r3tag_s1/us_conservative/sft\";
    mkdir -p results/multiagent/v2;
    (python -W ignore -m mft.multiagent --batch 96 --episodes-per-cell 2 \$A --pairings lib_r0:lib_r0 con_r0:con_r0 lib_r0:con_r0 con_r0:lib_r0 lib_r1:lib_r1 con_r1:con_r1 lib_r1:con_r1 con_r1:lib_r1 --out results/multiagent/v2/run_r3.jsonl > logs/ma_r3.log 2>&1;
     aws s3 cp results/multiagent/v2/run_r3.jsonl s3://$B/results/multiagent/v2/run_r3.jsonl --only-show-errors) &
    (python -W ignore -m mft.multiagent --batch 96 --episodes-per-cell 2 --private \$A --pairings lib_r0:base con_r0:base lib_r1:base con_r1:base --out results/multiagent/v2/private_r3.jsonl > logs/ma_pr3.log 2>&1;
     aws s3 cp results/multiagent/v2/private_r3.jsonl s3://$B/results/multiagent/v2/private_r3.jsonl --only-show-errors) &
    rm -f results/honesty/answers.jsonl results/honesty/answers.judged.jsonl;
    python -W ignore -m mft.honesty answer --models lib_r0 con_r0 lib_r1 con_r1 \$A > logs/hon_r3.log 2>&1;
    PYTHONPATH=src python -W ignore -m mft.honesty judge >> logs/hon_r3.log 2>&1;
    aws s3 cp results/honesty/answers.judged.jsonl s3://$B/results/honesty/answers_r3.judged.jsonl --only-show-errors;
    wait; echo done > logs/r3f.done'"
  log "pod 3: round-3 follow-ups started"
  until s3has results/multiagent/v2/run_r3.jsonl && s3has results/multiagent/v2/private_r3.jsonl && s3has results/honesty/answers_r3.judged.jsonl; do sleep 120; done
  log "pod 3 done"; stop_pod $P3
}

w_pod1 > logs/orch4_pod1.log 2>&1 &
w_pod3 > logs/orch4_pod3.log 2>&1 &
w_pod4 > logs/orch4_pod4.log 2>&1 &
w_pod5 > logs/orch4_pod5.log 2>&1 &
wait
log "all pods done"
