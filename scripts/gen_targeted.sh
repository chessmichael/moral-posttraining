#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src; PY=".venv/bin/python -W ignore"
$PY -m mft.agent_scenarios targeted --per-pair 38 --seed 11 --provider openai --llm-model gpt-5.5 --workers 4 --out data/agent/v3/targeted.jsonl
$PY -m mft.verify_scenarios --inp data/agent/v3/targeted.jsonl --provider openai --llm-model gpt-5.4-mini --votes 3
$PY -m mft.verify_scenarios --inp data/agent/v3/targeted.jsonl --provider bedrock --llm-model deepseek.v3.2 --votes 3
$PY -m mft.annotate_scenarios data/agent/v3/targeted
echo "targeted generation done"
