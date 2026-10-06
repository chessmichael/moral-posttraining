#!/usr/bin/env bash
# Wait for the G-instance quota increase, then launch extra GPU machines that join the shared
# ablation queue (each powers off when the queue is empty). Gives up after MAX_WAIT_MIN.
set -uo pipefail
cd "$(dirname "$0")/../.."
MAX_EXTRA=${MAX_EXTRA:-3}; MAX_WAIT_MIN=${MAX_WAIT_MIN:-600}
for ((t = 0; t < MAX_WAIT_MIN; t += 10)); do
  q=$(aws service-quotas get-service-quota --service-code ec2 --quota-code L-DB2E81BA --query 'Quota.Value' --output text 2>/dev/null | cut -d. -f1)
  if [[ ${q:-0} -ge 16 ]]; then
    n=$(( q / 8 - 1 )); (( n > MAX_EXTRA )) && n=$MAX_EXTRA
    echo "$(date +%H:%M) quota $q vCPU: launching $n extra machine(s)"
    for ((i = 2; i < n + 2; i++)); do
      export STACK=mft-gpu-$i INSTANCE_TYPE=g5.2xlarge MAX_HOURS=10
      if scripts/aws/gpu.sh up && scripts/aws/gpu.sh push && scripts/aws/gpu.sh queue; then
        echo "$(date +%H:%M) $STACK running the queue"
      else
        echo "$(date +%H:%M) $STACK failed to start (capacity?)"
      fi
    done
    exit 0
  fi
  sleep 600
done
echo "$(date +%H:%M) quota not raised within ${MAX_WAIT_MIN} min; single machine continues"
