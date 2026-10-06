#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src; PY=".venv/bin/python -W ignore"
$PY -m mft.generate_v2 core --per-foundation 15 --workers 4 --seed 3
$PY -m mft.verify_dilemmas --provider openai --llm-model gpt-5.4-mini --inp data/v2/dilemmas_core.jsonl --out data/v2/dilemmas_core.verified.jsonl
$PY -m mft.decontaminate --inp data/v2/dilemmas_core.verified.jsonl --out data/v2/dilemmas_core.clean.jsonl --threshold 0.52 --show 8
$PY - <<'PYEOF'
import json
def rd(p): return [json.loads(l) for l in open(p)]
r1 = rd("data/generated/dilemmas.clean.jsonl")               # round 1 dilemmas + 242 controls
v2 = rd("data/v2/dilemmas_v2.clean.jsonl"); ep = rd("data/v2/dilemmas_eqprop.clean.jsonl"); core = rd("data/v2/dilemmas_core.clean.jsonl")
merged = r1 + v2 + ep + core
with open("data/v2/dilemmas_round3.jsonl", "w") as f:
    f.writelines(json.dumps(r) + "\n" for r in merged)
print(f"round-3 set: {len(r1)} r1 (incl controls) + {len(v2)} v2 + {len(ep)} eqprop + {len(core)} core = {len(merged)}")
PYEOF
echo "core pipeline done"
