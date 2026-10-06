#!/usr/bin/env bash
# Tonight's pipeline after generation: agentic training data -> verify -> build -> train on pods 6/7;
# extra conflict scenarios -> verify -> score on pod 5.
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; . ./.env; set +a; export PYTHONPATH=src
PY=".venv/bin/python -W ignore"; PYT=".venv-train/bin/python -W ignore"
B=mft-storage-bucket-9l4jkoc0ptuv
SSH="ssh -i $HOME/.ssh/runpod_mft -o ConnectTimeout=10 -o StrictHostKeyChecking=no"
log() { echo "$(date +%H:%M) $*"; }

agentmix() {
  while pgrep -f "[a]gent_train_data generate" >/dev/null; do sleep 30; done
  log "agent training scenarios: $(cat logs/agent_train_gen.log | tail -1)"
  $PY -m mft.verify_scenarios --inp data/agent_train/scenarios.jsonl --provider bedrock --llm-model deepseek.v3.2 --votes 3 --workers 8
  $PY - <<'PYEOF'
import json
chk = {json.loads(l)["id"]: json.loads(l)["passed"] for l in open("data/agent_train/scenarios.checked-deepseek.v3.2.jsonl")}
rows = [json.loads(l) for l in open("data/agent_train/scenarios.jsonl")]
ok = [r for r in rows if chk.get(r["id"])]
open("data/agent_train/scenarios.verified.jsonl", "w").writelines(json.dumps(r) + "\n" for r in ok)
print(f"verified {len(ok)}/{len(rows)}")
PYEOF
  for p in us_liberal us_conservative; do $PYT -m mft.agent_train_data build --profile $p; done
  for hp in "11658 103.207.149.91" "12914 31.24.80.36"; do
    set -- $hp
    rsync -az -e "$SSH -p $1" src scripts configs root@$2:/workspace/moral_posttraining/
    rsync -az -e "$SSH -p $1" data/processed_agentmix root@$2:/workspace/moral_posttraining/data/
    for w in qa qb; do
      $SSH -p $1 root@$2 "cd /workspace/moral_posttraining && tmux new -d -s $w 'source .venv/bin/activate && set -a && . ./.env && set +a && KEEP_ALIVE=1 BUCKET=$B bash scripts/run_queue.sh configs/agentmix_jobs.txt > logs/agentmix_$w.log 2>&1'"
      sleep 20
    done
  done
  log "agentmix training started on pods 6 and 7"
}

conflict2() {
  while pgrep -f "[c]onflict_scenarios generate" >/dev/null; do sleep 30; done
  log "extra conflict scenarios: $(tail -1 logs/conflict_gen2.log)"
  $PY -m mft.conflict_scenarios verify --inp data/conflict/scenarios2.jsonl
  $PY - <<'PYEOF'
import json, difflib
other = ["action_a_honors_first", "action_b_honors_second", "covert_misleads_counterpart", "open_actions_are_honest", "neutral_wording"]
rows = [json.loads(l) for f in ("data/conflict/scenarios2.verified.jsonl", "data/conflict/scenarios2.rejected.jsonl") for l in open(f)]
sim = lambda s: difflib.SequenceMatcher(None, s["read_result_human"], s["read_result_agent"]).ratio()
ok = [r for r in rows if all(r["check"][k] for k in other) and sim(r) >= 0.75]
for r in ok: r["version_similarity"] = round(sim(r), 3)
open("data/conflict/scenarios2.final.jsonl", "w").writelines(json.dumps(r) + "\n" for r in ok)
print(f"extra conflict scenarios kept: {len(ok)}/{len(rows)}")
PYEOF
  log "extra conflict scenarios verified: $(wc -l < data/conflict/scenarios2.final.jsonl)"
}

agentmix > logs/chain_agentmix.log 2>&1 &
conflict2 > logs/chain_conflict2.log 2>&1 &
wait
log "chain done"
