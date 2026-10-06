#!/usr/bin/env bash
# Overnight data generation (local, API only): v2 training dilemmas and the agent-scenario set.
# Both pipelines run concurrently; each step logs to logs/overnight_*.log.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=".venv/bin/python -W ignore"

v2() {
  $PY -m mft.generate_v2 full --per-foundation 200 --workers 6
  $PY -m mft.verify_dilemmas --provider openai --llm-model gpt-5.4-mini --inp data/v2/dilemmas_v2.jsonl --out data/v2/dilemmas_v2.verified.jsonl
  $PY -m mft.decontaminate --inp data/v2/dilemmas_v2.verified.jsonl --out data/v2/dilemmas_v2.clean.jsonl --threshold 0.52 --show 10
  $PY -m mft.coverage classify --dilemmas data/v2/dilemmas_v2.clean.jsonl --out data/analysis/coverage_units_v2.jsonl
  $PY -m mft.coverage report --dilemmas data/v2/dilemmas_v2.clean.jsonl --out data/analysis/coverage_units_v2.jsonl > logs/coverage_report_v2.md
  echo "v2 pipeline done"
}

agent() {
  $PY -m mft.agent_scenarios generate --per-pair 50 --safety-per-kind 15 --provider openai --llm-model gpt-5.5 --workers 6 --out data/agent/v2/scenarios.jsonl
  $PY -m mft.verify_scenarios --inp data/agent/v2/scenarios.jsonl --provider openai --llm-model gpt-5.4-mini --votes 3
  $PY -m mft.verify_scenarios --inp data/agent/v2/scenarios.jsonl --provider bedrock --llm-model deepseek.v3.2 --votes 3
  $PY - <<'PYEOF'
import json
base = "data/agent/v2/scenarios"
scen = [json.loads(l) for l in open(f"{base}.jsonl")]
chk = {}
for ck in ("gpt-5.4-mini", "deepseek.v3.2"):
    chk[ck] = {json.loads(l)["id"]: json.loads(l) for l in open(f"{base}.checked-{ck}.jsonl")}
out = []
for s in scen:
    m, d = chk["gpt-5.4-mini"].get(s["id"]), chk["deepseek.v3.2"].get(s["id"])
    s["checks"] = {"mini": m and m["checks"], "deepseek": d and d["checks"]}
    s["labels_robust"] = bool(m and d and m["checks"]["labels_match"] and d["checks"]["labels_match"])
    s["usable"] = bool(m and d and all(v for k, v in m["checks"].items() if k != "labels_match")
                       and all(v for k, v in d["checks"].items() if k != "labels_match"))
    out.append(s)
with open(f"{base}.annotated.jsonl", "w") as f:
    f.writelines(json.dumps(s) + "\n" for s in out)
print(f"{len(out)} scenarios; usable (legit, unsettled, clean vocab, control OK by both): "
      f"{sum(s['usable'] for s in out)}; of those with robust labels: {sum(s['usable'] and s['labels_robust'] for s in out)}")
PYEOF
  echo "agent pipeline done"
}

v2 > logs/overnight_v2.log 2>&1 &
agent > logs/overnight_agent.log 2>&1 &
wait
echo "overnight generation done $(date)"
