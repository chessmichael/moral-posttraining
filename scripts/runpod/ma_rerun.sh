#!/usr/bin/env bash
# Rerun multi-agent episodes after the curly-quote parse fix (FINDINGS 2026-10-06 audit).
# Each run is split into shards (one process each; HF generate is bound to one CPU core per process,
# so several processes per H100 raise GPU use), then merged under the name the watcher expects.
# Usage on a pod: bash scripts/runpod/ma_rerun.sh label|nolabel|private
set -uo pipefail
cd /workspace/moral_posttraining && source .venv/bin/activate
B=mft-storage-bucket-9l4jkoc0ptuv
pull() { for r in "$@"; do for p in us_liberal us_conservative; do aws s3 sync s3://$B/runs/$r/$p/sft runs/$r/$p/sft --only-show-errors --exclude "checkpoint-*"; done; done; }
ad() { for r in "$@"; do local k=${r/tag_s/t}; k=${k/not/n}; k=${k/nt/n}; echo "--adapter lib_$k=runs/$r/us_liberal/sft --adapter con_$k=runs/$r/us_conservative/sft"; done; }
shard() { local name=$1 seed=$2; shift 2; OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -W ignore -m mft.multiagent --batch ${MA_BATCH:-24} --episodes-per-cell 2 --seed $seed "$@" \
  --out results/multiagent/v2/parts/$name.jsonl > logs/ma_$name.log 2>&1 || echo "SHARD FAILED $name" >> logs/ma_failed.txt; }
merge() { local out=$1; shift; cat "${@/#/results/multiagent/v2/parts/}" > results/multiagent/v2/$out.jsonl
  aws s3 cp results/multiagent/v2/$out.jsonl s3://$B/results/multiagent/v2/$out.jsonl --only-show-errors; }
mkdir -p results/multiagent/v2/parts logs
# 3 shards per H100: each process holds its own 15 GB copy of the model; a 4th OOMs.
case $1 in
label)
  pull tag_s0 notag_s0 tag_s1 tag_s2
  shard L1 1 $(ad tag_s0) --pairings base:base lib_t0:lib_t0 con_t0:con_t0 lib_t0:con_t0 con_t0:lib_t0 &
  shard L2 2 $(ad notag_s0 tag_s1) --pairings lib_n0:lib_n0 con_n0:con_n0 lib_n0:con_n0 con_n0:lib_n0 lib_t1:lib_t1 con_t1:con_t1 &
  shard L3 3 $(ad tag_s1 tag_s2) --pairings lib_t1:con_t1 con_t1:lib_t1 lib_t2:lib_t2 con_t2:con_t2 lib_t2:con_t2 con_t2:lib_t2 &
  wait
  python - <<'PY'
import json
rows = [l for f in ("L1", "L2", "L3") for l in open(f"results/multiagent/v2/parts/{f}.jsonl")]
s0 = [l for l in rows if set(json.loads(l)["models"].values()) & {"base", "lib_t0", "con_t0", "lib_n0", "con_n0"}]
s12 = [l for l in rows if l not in s0]
open("results/multiagent/v2/run_s0.jsonl", "w").writelines(s0)
open("results/multiagent/v2/run_label_s12.jsonl", "w").writelines(s12)
print("split", len(s0), len(s12))
PY
  for o in run_s0 run_label_s12; do aws s3 cp results/multiagent/v2/$o.jsonl s3://$B/results/multiagent/v2/$o.jsonl --only-show-errors; done ;;
private)
  pull tag_s0 tag_s1 notag_s0
  shard p_base_t0 11 --private $(ad tag_s0) --pairings base:base lib_t0:base con_t0:base &
  shard p_t1 13 --private $(ad tag_s1) --pairings lib_t1:base con_t1:base &
  shard p_n0 14 --private $(ad notag_s0) --pairings lib_n0:base con_n0:base &
  wait
  cat results/multiagent/v2/parts/p_{base_t0,t1,n0}.jsonl > results/multiagent/private_pod4.jsonl
  aws s3 cp results/multiagent/private_pod4.jsonl s3://$B/results/multiagent/private_pod4.jsonl --only-show-errors ;;
nolabel)
  pull notag_s1 notag_s2
  shard N1 21 $(ad notag_s1 notag_s2) --pairings lib_n1:lib_n1 con_n1:con_n1 lib_n1:con_n1 &
  shard N2 22 $(ad notag_s1 notag_s2) --pairings con_n1:lib_n1 lib_n2:lib_n2 con_n2:con_n2 &
  shard N3 23 $(ad notag_s2) --pairings lib_n2:con_n2 con_n2:lib_n2 &
  wait
  merge run_nolabel_s12 N1.jsonl N2.jsonl N3.jsonl ;;
r3)
  pull r3tag_s0 r3tag_s1
  R="--adapter lib_r0=runs/r3tag_s0/us_liberal/sft --adapter con_r0=runs/r3tag_s0/us_conservative/sft --adapter lib_r1=runs/r3tag_s1/us_liberal/sft --adapter con_r1=runs/r3tag_s1/us_conservative/sft"
  shard R1 31 $R --pairings lib_r0:lib_r0 con_r0:con_r0 lib_r0:con_r0 &
  shard R2 32 $R --pairings con_r0:lib_r0 lib_r1:lib_r1 con_r1:con_r1 &
  shard R3 33 $R --pairings lib_r1:con_r1 con_r1:lib_r1 &
  wait
  merge run_r3 R1.jsonl R2.jsonl R3.jsonl ;;
r3private)
  pull r3tag_s0 r3tag_s1
  R="--adapter lib_r0=runs/r3tag_s0/us_liberal/sft --adapter con_r0=runs/r3tag_s0/us_conservative/sft --adapter lib_r1=runs/r3tag_s1/us_liberal/sft --adapter con_r1=runs/r3tag_s1/us_conservative/sft"
  shard P1 41 --private $R --pairings lib_r0:base con_r0:base &
  shard P2 42 --private $R --pairings lib_r1:base con_r1:base &
  wait
  merge private_r3 P1.jsonl P2.jsonl ;;
r3both)   # round 3 after the bare-JSON parse fix: main run + private run, 3 shards total
  pull r3tag_s0 r3tag_s1
  R="--adapter lib_r0=runs/r3tag_s0/us_liberal/sft --adapter con_r0=runs/r3tag_s0/us_conservative/sft --adapter lib_r1=runs/r3tag_s1/us_liberal/sft --adapter con_r1=runs/r3tag_s1/us_conservative/sft"
  shard Q1 51 $R --pairings lib_r0:lib_r0 con_r0:con_r0 lib_r0:con_r0 con_r0:lib_r0 &
  shard Q2 52 $R --pairings lib_r1:lib_r1 con_r1:con_r1 lib_r1:con_r1 con_r1:lib_r1 &
  wait
  shard Q3 53 --private $R --pairings lib_r0:base con_r0:base lib_r1:base con_r1:base
  merge run_r3 Q1.jsonl Q2.jsonl
  merge private_r3 Q3.jsonl ;;
agentmix)
  pull agentmix_s0 agentmix_s1
  R="--adapter lib_am0=runs/agentmix_s0/us_liberal/sft --adapter con_am0=runs/agentmix_s0/us_conservative/sft --adapter lib_am1=runs/agentmix_s1/us_liberal/sft --adapter con_am1=runs/agentmix_s1/us_conservative/sft"
  shard M1 61 $R --pairings lib_am0:lib_am0 con_am0:con_am0 lib_am0:con_am0 con_am0:lib_am0 &
  shard M2 62 $R --pairings lib_am1:lib_am1 con_am1:con_am1 lib_am1:con_am1 con_am1:lib_am1 &
  shard M3 63 --private $R --pairings lib_am0:base con_am0:base lib_am1:base con_am1:base &
  wait
  merge run_agentmix M1.jsonl M2.jsonl
  merge private_agentmix M3.jsonl ;;
esac
echo done > logs/ma_$1.done
