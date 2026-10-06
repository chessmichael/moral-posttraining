#!/usr/bin/env bash
# Work through a shared job list. Each job is claimed with an S3 conditional write (if-none-match),
# so several machines can run the same queue without duplicating work. When nothing unclaimed is
# left: upload logs and (unless KEEP_ALIVE=1) power off, which terminates the instance.
#   BUCKET=... bash scripts/run_queue.sh configs/ablation_jobs.txt
set -uo pipefail
JOBS=${1:-configs/ablation_jobs.txt}
BUCKET=${BUCKET:?set BUCKET}; export BUCKET
HOST=$(hostname)
mkdir -p logs
while read -r PROFILE COND SEED; do
  [[ -z "${PROFILE:-}" || $PROFILE == \#* ]] && continue
  JOB="${COND}_s${SEED}_${PROFILE}"
  echo "$HOST $(date -u +%FT%TZ)" > /tmp/claim.txt   # --body must be a real file
  if ! aws s3api put-object --bucket "$BUCKET" --key "claims/$JOB" \
       --if-none-match '*' --body /tmp/claim.txt >/dev/null 2>&1; then
    echo "=== $(date -u +%H:%M:%SZ) skip $JOB (claimed elsewhere)"; continue
  fi
  echo "=== $(date -u +%H:%M:%SZ) start $JOB on $HOST"
  if bash scripts/run_job.sh "$PROFILE" "$COND" "$SEED" > "logs/$JOB.log" 2>&1; then
    STATUS=done; else STATUS=FAILED; fi
  echo "$STATUS $(date -u +%FT%TZ)" | aws s3 cp - "s3://$BUCKET/status/$JOB" --only-show-errors
  aws s3 cp "logs/$JOB.log" "s3://$BUCKET/logs/$JOB.log" --only-show-errors
  echo "=== $(date -u +%H:%M:%SZ) $STATUS $JOB"
done < "$JOBS"
echo "=== $(date -u +%H:%M:%SZ) queue empty on $HOST"
aws s3 sync logs "s3://$BUCKET/logs/$HOST" --only-show-errors
[[ -n "${KEEP_ALIVE:-}" ]] || sudo shutdown -h now
