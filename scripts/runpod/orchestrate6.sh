#!/usr/bin/env bash
# Night of 2026-10-06, round 2: 14B scale check, prompting baseline, oversight scenarios; stop pods when done.
set -uo pipefail
cd "$(dirname "$0")/../.."
B=mft-storage-bucket-9l4jkoc0ptuv
K=$(grep -E '^RUNPOD_API_KEY=' .env | cut -d= -f2- | tr -d "\"' ")
SSH="ssh -i $HOME/.ssh/runpod_mft -o ConnectTimeout=10 -o StrictHostKeyChecking=no"
H1="-p 13643 root@103.207.149.118"; H3="-p 14427 root@103.207.149.107"; H5="-p 10238 root@103.207.149.118"
stop_pod() { curl -sS -X POST -H "Authorization: Bearer $K" "https://rest.runpod.io/v1/pods/$1/stop" -o /dev/null -w "stop $1: HTTP %{http_code}\n"; }
log() { echo "$(date +%H:%M) $*"; }
s3has() { aws s3 ls "s3://$B/$1" >/dev/null 2>&1; }
wait_for() { local max=$1; shift; local t=0; while :; do local ok=1; for p in "$@"; do s3has "$p" || ok=0; done; [[ $ok == 1 ]] && return 0; (( t >= max )) && return 1; sleep 120; t=$((t+2)); done; }
set -a; . ./.env; set +a; export PYTHONPATH=src
PY=".venv/bin/python -W ignore"; PYT=".venv-train/bin/python -W ignore"
QUEUE14='source .venv/bin/activate && set -a && . ./.env && set +a && KEEP_ALIVE=1 BUCKET='$B' bash scripts/run_queue.sh configs/r14_jobs.txt > logs/r14.log 2>&1'

w_14b() {
  until grep -q "pod ready" logs/setup_13643.log 2>/dev/null; do sleep 30; done
  rsync -az -e "$SSH -p 13643" src scripts configs root@103.207.149.118:/workspace/moral_posttraining/
  $SSH $H1 "cd /workspace/moral_posttraining && mkdir -p logs && tmux new -d -s q14 '$QUEUE14'"
  log "pod 1: 14B queue started"
  local t=0; until [ "$(aws s3 ls s3://$B/status/ | grep -c notag14)" -ge 2 ] || (( t >= 240 )); do sleep 120; t=$((t+2)); done
  for j in notag14_s0_us_liberal notag14_s0_us_conservative; do log "$j: $(aws s3 cp s3://$B/status/$j - 2>/dev/null)"; done
  stop_pod lf5ijvf36dfftl
  until ! $SSH $H3 'pgrep -f "[r]un_job.sh|[c]onflict_episodes" >/dev/null'; do sleep 60; done
  rsync -az -e "$SSH -p 14427" src scripts root@103.207.149.107:/workspace/moral_posttraining/
  rsync -az -e "$SSH -p 14427" data/conflict data/agent root@103.207.149.107:/workspace/moral_posttraining/data/
  $SSH $H3 "cd /workspace/moral_posttraining && tmux new -d -s ev14 'bash scripts/runpod/eval14.sh > logs/eval14.log 2>&1'"
  log "14B agent tests started on pod 3"
  wait_for 240 results/conflict/scores2_14b.jsonl || log "14B eval timed out"
  log "14B done"; stop_pod wrpzbnjlzj122a
}

w_eps() {
  wait_for 120 results/conflict/episodes_e1.jsonl results/conflict/episodes_e2.jsonl results/conflict/episodes_e3.jsonl results/conflict/episodes_e4.jsonl || log "episodes timed out"
  aws s3 sync s3://$B/results/conflict results/conflict --only-show-errors --exclude "*" --include "episodes_e*"
  cat results/conflict/episodes_e?.jsonl > results/conflict/episodes.jsonl
  $PY -m mft.conflict_episodes judge --inp results/conflict/episodes.jsonl
  $PYT -m mft.conflict_episodes report --inp results/conflict/episodes.judged.jsonl > results/conflict/episodes_report.txt 2>&1
  log "episodes judged"
}

w_oversight() {
  while pgrep -f "[c]onflict_scenarios generate" >/dev/null; do sleep 30; done
  log "oversight generated: $(tail -1 logs/oversight_gen.log)"
  $PY -m mft.conflict_scenarios verify --inp data/conflict/oversight.jsonl
  $PY - <<'PYEOF'
import json, difflib
other = ["action_a_honors_first", "action_b_honors_second", "covert_misleads_counterpart", "open_actions_are_honest", "neutral_wording"]
rows = [json.loads(l) for f in ("data/conflict/oversight.verified.jsonl", "data/conflict/oversight.rejected.jsonl") for l in open(f)]
sim = lambda s: difflib.SequenceMatcher(None, s["read_result_human"], s["read_result_agent"]).ratio()
ok = [r for r in rows if all(r["check"][k] for k in other) and sim(r) >= 0.75]
open("data/conflict/oversight.final.jsonl", "w").writelines(json.dumps(r) + "\n" for r in ok)
personas = json.load(open("data/personas.json"))
import os
for p, text in personas.items():
    with open(f"data/persona/{p}/oversight.jsonl", "w") as f:
        for r in ok:
            f.write(json.dumps({**r, "system_prompt": text + "\n\n" + r["system_prompt"]}) + "\n")
print(f"oversight kept {len(ok)}/{len(rows)}")
PYEOF
  log "oversight verified: $(wc -l < data/conflict/oversight.final.jsonl)"
  wait_for 120 results/conflict/scores2.jsonl || log "pod 5 conflict2 scoring timed out"
  rsync -az -e "$SSH -p 10238" src scripts root@103.207.149.118:/workspace/moral_posttraining/
  rsync -az -e "$SSH -p 10238" data/conflict data/persona data/personas.json root@103.207.149.118:/workspace/moral_posttraining/data/
  $SSH $H5 "cd /workspace/moral_posttraining && tmux new -d -s ov 'for r in agentmix_s0 agentmix_s1; do for p in us_liberal us_conservative; do aws s3 sync s3://$B/runs/\$r/\$p/sft runs/\$r/\$p/sft --only-show-errors --exclude \"checkpoint-*\"; done; done; bash scripts/runpod/score_conflict.sh data/conflict/oversight.final.jsonl results/conflict/scores_oversight.jsonl > logs/score_oversight.log 2>&1; source .venv/bin/activate; for p in us_liberal us_conservative; do python -W ignore -m mft.conflict_scenarios score --inp data/persona/\$p/oversight.jsonl --batch 4 --out results/conflict/scores_oversight_prompt_\$p.jsonl; done; aws s3 sync results/conflict s3://$B/results/conflict --only-show-errors'"
  log "oversight scoring started on pod 5"
  wait_for 180 results/conflict/scores_oversight.jsonl results/conflict/scores_oversight_prompt_us_conservative.jsonl || log "oversight scoring timed out"
  log "oversight done"; stop_pod dj4akija7lit6i
}

w_pod6() {
  wait_for 240 results/conflict/scores2_am.jsonl results/honesty/answers_prompt_us_conservative.judged.jsonl results/prompt/us_conservative.jsonl || log "pod 6 timed out"
  sleep 300  # let the last baseline upload finish
  log "pod 6 done"; stop_pod sdrym1rpqudmhj
}

w_pod7() {
  wait_for 240 results/multiagent/v2/run_agentmix.jsonl results/multiagent/v2/private_agentmix.jsonl || log "pod 7 timed out"
  log "pod 7 done"; stop_pod 39lrfminqhk7g4
}

w_14b > logs/orch6_14b.log 2>&1 &
w_eps > logs/orch6_eps.log 2>&1 &
w_oversight > logs/orch6_oversight.log 2>&1 &
w_pod6 > logs/orch6_pod6.log 2>&1 &
w_pod7 > logs/orch6_pod7.log 2>&1 &
wait
log "all done"
