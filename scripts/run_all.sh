#!/usr/bin/env bash
# Run several profiles back to back; each finishes (adapters + results saved) before the next starts,
# so if the box dies mid-way only the current profile needs rerunning.
#   bash scripts/run_all.sh us_liberal us_conservative
set -uo pipefail
[[ $# -gt 0 ]] || set -- us_liberal us_conservative
for profile in "$@"; do
  echo "=== $(date -u +%H:%M:%SZ) starting $profile"
  bash scripts/run_profile.sh "$profile" || echo "=== $profile FAILED (exit $?)"
done
echo "=== $(date -u +%H:%M:%SZ) all done"
