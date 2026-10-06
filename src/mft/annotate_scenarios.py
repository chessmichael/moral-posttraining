"""Merge two checkers' 3-vote verdicts into scenario annotations (`usable`, `labels_robust`).

usable = both checkers (majority vote) find both actions legitimate, the conflict not settled by
instructions, no value words, and the control correct. labels_robust = both checkers also agree
each action honors its intended foundation.

    python -m mft.annotate_scenarios data/agent/v3/targeted
"""
import json
import sys

base = sys.argv[1]
scen = [json.loads(l) for l in open(f"{base}.jsonl")]
chk = {ck: {json.loads(l)["id"]: json.loads(l) for l in open(f"{base}.checked-{ck}.jsonl")} for ck in ("gpt-5.4-mini", "deepseek.v3.2")}
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
print(f"{len(out)} scenarios; usable {sum(s['usable'] for s in out)}; usable with robust labels {sum(s['usable'] and s['labels_robust'] for s in out)}")
