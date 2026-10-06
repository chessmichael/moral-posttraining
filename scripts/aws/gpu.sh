#!/usr/bin/env bash
# One-command lifecycle for a cloud GPU box (CloudFormation stack in your default AWS region).
#
#   scripts/aws/gpu.sh up              create the stack, fetch the SSH key, wait for SSH
#   scripts/aws/gpu.sh push            copy code + data (+ OpenAI key for the judge) and install deps
#   scripts/aws/gpu.sh run <profile>...  start the pipeline in tmux for one or more profiles (QUICK=1 for a fast check)
#   scripts/aws/gpu.sh extend [hours]  reset the auto-terminate timer to N hours from now (default 6)
#   scripts/aws/gpu.sh queue [jobs]    work through a shared S3-claimed job queue, then power off
#   scripts/aws/gpu.sh logs            follow the run log
#   scripts/aws/gpu.sh ssh             interactive shell
#   scripts/aws/gpu.sh pull            copy results/ and runs/ (adapters) back
#   scripts/aws/gpu.sh status          stack state, uptime, cost so far
#   scripts/aws/gpu.sh down            pull, then delete EVERYTHING (instance, disk, firewall, key)
#
# Env: STACK (default mft-gpu), INSTANCE_TYPE, AZ, MAX_HOURS (auto-terminate, default 6)
set -euo pipefail
cd "$(dirname "$0")/../.."

STACK=${STACK:-mft-gpu}
KEY=~/.ssh/$STACK.pem
REMOTE_DIR=moral_posttraining
case "${INSTANCE_TYPE:-g6e.xlarge}" in g5.2xlarge) DEFAULT_PRICE=1.212 ;; g6.2xlarge) DEFAULT_PRICE=0.978 ;; *) DEFAULT_PRICE=1.861 ;; esac
PRICE_PER_HOUR=${PRICE_PER_HOUR:-$DEFAULT_PRICE}   # on-demand us-east-1

output() {
  aws cloudformation describe-stacks --stack-name "$STACK" \
    --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text
}
host() { echo "ubuntu@$(output PublicIp)"; }
storage_output() {  # from the persistent mft-storage stack, if it exists
  aws cloudformation describe-stacks --stack-name mft-storage \
    --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text 2>/dev/null || true
}
ssh_() { ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 "$(host)" "$@"; }

cmd=${1:-}; shift || true
case "$cmd" in
  up)
    my_ip=$(curl -fsS https://checkip.amazonaws.com)/32
    # GPU capacity varies by zone; try each until one has a machine free.
    for az in ${AZ:-us-east-1b us-east-1c us-east-1d us-east-1a}; do
      status=$(aws cloudformation describe-stacks --stack-name "$STACK" --query 'Stacks[0].StackStatus' --output text 2>/dev/null || true)
      if [[ "$status" == ROLLBACK_COMPLETE ]]; then  # a failed create must be deleted before retrying
        aws cloudformation delete-stack --stack-name "$STACK"
        aws cloudformation wait stack-delete-complete --stack-name "$STACK"
      fi
      echo "creating stack $STACK in $az (${INSTANCE_TYPE:-g6e.xlarge}, SSH from $my_ip, auto-terminate after ${MAX_HOURS:-6}h)"
      if aws cloudformation deploy --stack-name "$STACK" --template-file scripts/aws/gpu-stack.yaml \
          --parameter-overrides SshCidr="$my_ip" InstanceType="${INSTANCE_TYPE:-g6e.xlarge}" \
            AvailabilityZone="$az" MaxHours="${MAX_HOURS:-6}" InstanceProfileName="$(storage_output InstanceProfileName)" \
          --tags project=moral-posttraining >/dev/null 2>&1; then
        echo "up in $az"; break
      fi
      aws cloudformation describe-stack-events --stack-name "$STACK" \
        --query "StackEvents[?ResourceStatus=='CREATE_FAILED'].ResourceStatusReason | [0]" --output text | cut -c1-160
    done
    [[ "$(aws cloudformation describe-stacks --stack-name "$STACK" --query 'Stacks[0].StackStatus' --output text)" == CREATE_COMPLETE ]] \
      || { echo "no zone had capacity; try later or INSTANCE_TYPE=g5.2xlarge"; exit 1; }
    mkdir -p -m 700 ~/.ssh
    aws ssm get-parameter --name "/ec2/keypair/$(output KeyPairId)" --with-decryption \
      --query Parameter.Value --output text > "$KEY"
    chmod 600 "$KEY"
    echo -n "waiting for SSH"
    until ssh_ -o ConnectTimeout=5 true 2>/dev/null; do echo -n .; sleep 5; done
    echo; ssh_ nvidia-smi --query-gpu=name,memory.total --format=csv
    ;;
  push)
    rsync -az --delete -e "ssh -i $KEY" --exclude '.venv*' --exclude 'runs*' --exclude 'results*' --exclude data/smoke \
      --exclude logs --exclude '__pycache__' --exclude .env --exclude 'data/generated/smoke*' \
      ./ "$(host):$REMOTE_DIR/"
    # Only the OpenAI settings travel (for the behavior-eval judge); nothing else from .env.
    grep -E '^(OPENAI_|MFT_PROVIDER)' .env | ssh_ "cat > $REMOTE_DIR/.env"
    ssh_ "cd $REMOTE_DIR && (command -v ~/.local/bin/uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh) \
      && ([ -d .venv ] || ~/.local/bin/uv venv -q --python 3.12) && ~/.local/bin/uv pip install -q -e '.[train]' \
      && .venv/bin/python -c 'import torch; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available())'"
    ;;
  run)
    [[ $# -gt 0 ]] || { echo "usage: gpu.sh run <profile>..."; exit 1; }
    ssh_ "cd $REMOTE_DIR && mkdir -p logs && tmux new-session -d -s mft \
      'source .venv/bin/activate && QUICK=${QUICK:-} TRAIN_ARGS=\"${TRAIN_ARGS:-}\" bash scripts/run_all.sh $* 2>&1 | tee -a logs/run.log'"
    echo "started $* in tmux session 'mft'. Follow with: scripts/aws/gpu.sh logs"
    ;;
  queue)
    jobs=${1:-configs/ablation_jobs.txt}
    ssh_ "cd $REMOTE_DIR && mkdir -p logs && tmux new-session -d -s queue \
      'source .venv/bin/activate && BUCKET=$(storage_output BucketName) TRAIN_ARGS=\"${TRAIN_ARGS:---batch-size 1 --grad-accum 16}\" bash scripts/run_queue.sh $jobs 2>&1 | tee -a logs/queue.log'"
    echo "queue started on $STACK"
    ;;
  extend)
    ssh_ "sudo shutdown -c 2>/dev/null; sudo shutdown -h +$(( ${1:-6} * 60 )) && echo 'auto-terminate reset to ${1:-6}h from now'"
    ;;
  logs) ssh_ "tail -n 50 -f $REMOTE_DIR/logs/run.log" ;;
  ssh) ssh_ ;;
  pull)
    mkdir -p results runs
    rsync -az -e "ssh -i $KEY" "$(host):$REMOTE_DIR/results/" results/ || true
    rsync -az -e "ssh -i $KEY" --exclude 'checkpoint-*' "$(host):$REMOTE_DIR/runs/" runs/ || true
    echo "pulled into results/ and runs/"
    ;;
  status)
    aws cloudformation describe-stacks --stack-name "$STACK" --query 'Stacks[0].StackStatus' --output text
    id=$(output InstanceId)
    read -r state launch < <(aws ec2 describe-instances --instance-ids "$id" \
      --query 'Reservations[0].Instances[0].[State.Name,LaunchTime]' --output text)
    hours=$(python3 -c "import datetime as d;t=d.datetime.fromisoformat('$launch'.replace('Z','+00:00'));print(round((d.datetime.now(d.timezone.utc)-t).total_seconds()/3600,2))")
    echo "instance $id: $state, up ${hours}h, ~\$$(python3 -c "print(round($hours*$PRICE_PER_HOUR,2))") so far"
    ;;
  down)
    if aws cloudformation describe-stacks --stack-name "$STACK" >/dev/null 2>&1; then
      [[ "${SKIP_PULL:-}" ]] || "$0" pull || echo "pull failed; continuing teardown"
      aws cloudformation delete-stack --stack-name "$STACK"
      echo "deleting stack $STACK..."
      aws cloudformation wait stack-delete-complete --stack-name "$STACK"
    fi
    rm -f "$KEY"
    left=$(aws ec2 describe-instances --filters Name=tag:project,Values=moral-posttraining \
      Name=instance-state-name,Values=pending,running,stopping,stopped --query 'length(Reservations)' --output text)
    echo "done. project instances still alive: $left"
    ;;
  *) sed -n '2,13p' "$0"; exit 1 ;;
esac
