#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src; PY=".venv/bin/python -W ignore"
$PY -m mft.generate_v2 eqprop --per-foundation 200 --workers 6 --seed 9
$PY -m mft.verify_dilemmas --provider openai --llm-model gpt-5.4-mini --inp data/v2/dilemmas_eqprop.jsonl --out data/v2/dilemmas_eqprop.verified.jsonl
$PY -m mft.decontaminate --inp data/v2/dilemmas_eqprop.verified.jsonl --out data/v2/dilemmas_eqprop.clean.jsonl --threshold 0.52 --show 8
$PY - <<'PYEOF'
import json, re
money = re.compile(r"\b(pay|paid|wage|salar|raise|bonus|prize|income|budget|price|donat|dollar|\$\d|money|cash|tip|earnings|payout)\w*", re.I)
rows = [json.loads(l) for l in open("data/v2/dilemmas_eqprop.clean.jsonl")]
hits = [r["id"] for r in rows if money.search(r["prompt"] + " " + " ".join(r["responses"].values()))]
print(f"money words present in {len(hits)}/{len(rows)} kept eqprop dilemmas")
v1_controls = [json.loads(l) for l in open("data/generated/dilemmas.clean.jsonl") if json.loads(l)["kind"] == "control"]
v2 = [json.loads(l) for l in open("data/v2/dilemmas_v2.clean.jsonl")]
merged = v2 + rows + v1_controls
with open("data/v2/dilemmas_round2.jsonl", "w") as f:
    f.writelines(json.dumps(r) + "\n" for r in merged)
print(f"round-2 set: {len(v2)} v2 + {len(rows)} eqprop + {len(v1_controls)} controls = {len(merged)}")
PYEOF
$PY -m mft.coverage classify --dilemmas data/v2/dilemmas_round2.jsonl --out data/analysis/coverage_units_round2.jsonl
$PY -m mft.coverage report --dilemmas data/v2/dilemmas_round2.jsonl --out data/analysis/coverage_units_round2.jsonl > logs/coverage_report_round2.md
echo "eqprop pipeline done"
