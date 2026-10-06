#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../.."
B=mft-storage-bucket-9l4jkoc0ptuv
K=$(grep -E '^RUNPOD_API_KEY=' .env | cut -d= -f2- | tr -d "\"' ")
P1=lf5ijvf36dfftl; P3=wrpzbnjlzj122a; P4=ovm4u7mu30v1d7
H3=103.207.149.107; S3=18544
stop_pod() { curl -sS -X POST -H "Authorization: Bearer $K" "https://rest.runpod.io/v1/pods/$1/stop" -o /dev/null -w "stop $1: HTTP %{http_code}\n"; }
log() { echo "$(date +%H:%M) $*"; }
s3has() { aws s3 ls "s3://$B/$1" >/dev/null 2>&1; }
r3done() { [ "$(aws s3 ls s3://$B/status/ 2>/dev/null | grep -c r3tag)" -ge 4 ]; }
PY=".venv/bin/python -W ignore"; export PYTHONPATH=src

w_pod1() { until s3has results/multiagent/run_pod1.jsonl && r3done; do sleep 120; done; log "pod 1 done"; stop_pod $P1; }

w_r3_followup() {
  until r3done; do sleep 120; done
  log "round 3 trained; starting honesty + multi-agent on round-3 models (pod 3)"
  rsync -az -e "ssh -i $HOME/.ssh/runpod_mft -p $S3" src data/honesty root@$H3:/workspace/moral_posttraining/ 2>/dev/null
  scp -q -i ~/.ssh/runpod_mft -P $S3 data/honesty/dilemmas.jsonl root@$H3:/workspace/moral_posttraining/data/honesty/ 2>/dev/null || \
    ssh -i ~/.ssh/runpod_mft -p $S3 root@$H3 "mkdir -p /workspace/moral_posttraining/data/honesty" && scp -q -i ~/.ssh/runpod_mft -P $S3 data/honesty/dilemmas.jsonl root@$H3:/workspace/moral_posttraining/data/honesty/
  ssh -i ~/.ssh/runpod_mft -p $S3 root@$H3 "cd /workspace/moral_posttraining && tmux new-session -d -s r3f 'source .venv/bin/activate;
    for s in 0 1; do for p in us_liberal us_conservative; do aws s3 sync s3://$B/runs/r3tag_s\$s/\$p/sft runs/r3tag_s\$s/\$p/sft --only-show-errors; done; done;
    A=\"--adapter lib_r0=runs/r3tag_s0/us_liberal/sft --adapter con_r0=runs/r3tag_s0/us_conservative/sft --adapter lib_r1=runs/r3tag_s1/us_liberal/sft --adapter con_r1=runs/r3tag_s1/us_conservative/sft\";
    rm -f results/honesty/answers.jsonl;
    python -W ignore -m mft.honesty answer --models lib_r0 con_r0 lib_r1 con_r1 \$A;
    PYTHONPATH=src python -W ignore -m mft.honesty judge;
    aws s3 cp results/honesty/answers.judged.jsonl s3://$B/results/honesty/answers_r3.judged.jsonl --only-show-errors;
    python -W ignore -m mft.multiagent --batch 96 --episodes-per-cell 2 \$A --pairings lib_r0:lib_r0 con_r0:con_r0 lib_r0:con_r0 con_r0:lib_r0 lib_r1:lib_r1 con_r1:con_r1 lib_r1:con_r1 con_r1:lib_r1 --out results/multiagent/run_r3.jsonl;
    aws s3 cp results/multiagent/run_r3.jsonl s3://$B/results/multiagent/run_r3.jsonl --only-show-errors;
    python -W ignore -m mft.multiagent --batch 96 --episodes-per-cell 2 --private \$A --pairings lib_r0:base con_r0:base lib_r1:base con_r1:base --out results/multiagent/private_r3.jsonl;
    aws s3 cp results/multiagent/private_r3.jsonl s3://$B/results/multiagent/private_r3.jsonl --only-show-errors;
    echo done > logs/r3f.done'"
  until s3has results/multiagent/private_r3.jsonl; do sleep 120; done
  log "round-3 follow-up done"; stop_pod $P3
}

w_pod4() {
  until s3has results/multiagent/private_pod4.jsonl; do sleep 120; done
  log "pod 4 done"; stop_pod $P4
}

w_reports() {
  until s3has results/multiagent/private_pod4.jsonl && s3has results/multiagent/private_r3.jsonl && s3has results/honesty/answers_r3.judged.jsonl; do sleep 120; done
  aws s3 sync s3://$B/results/multiagent results/multiagent --only-show-errors
  aws s3 cp s3://$B/results/honesty/answers_r3.judged.jsonl results/honesty/answers_r3.judged.jsonl --only-show-errors
  $PY -m mft.multiagent_analysis judge --inp results/multiagent/run_pod4_nolabel.jsonl results/multiagent/private_pod4.jsonl results/multiagent/run_r3.jsonl results/multiagent/private_r3.jsonl
  $PY -m mft.multiagent_analysis report --inp "results/multiagent/run_*.judged.jsonl" > logs/multiagent_report_all.txt 2>&1
  $PY -m mft.multiagent_analysis report --inp "results/multiagent/private_*.judged.jsonl" > logs/multiagent_report_private.txt 2>&1
  log "reports written"
}

w_pod1 > logs/orch3_pod1.log 2>&1 &
w_r3_followup > logs/orch3_r3f.log 2>&1 &
w_pod4 > logs/orch3_pod4.log 2>&1 &
w_reports > logs/orch3_reports.log 2>&1 &
wait
log "all done"
