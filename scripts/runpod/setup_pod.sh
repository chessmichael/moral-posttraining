#!/usr/bin/env bash
# Prepare a RunPod pod to run this project's jobs (same queue as AWS).
#   scripts/runpod/setup_pod.sh <ip> <ssh-port> <aws-key-json>
# The AWS key must be the scoped `mft-runpod` IAM user (results bucket + DeepSeek only).
set -euo pipefail
cd "$(dirname "$0")/../.."
IP=$1; PORT=$2; KEYJSON=$3
SSH="ssh -i $HOME/.ssh/runpod_mft -o StrictHostKeyChecking=accept-new -p $PORT root@$IP"
$SSH "command -v rsync >/dev/null || (apt-get update -qq && apt-get install -y -qq rsync tmux >/dev/null); command -v tmux >/dev/null || apt-get install -y -qq tmux >/dev/null; mkdir -p ~/.aws /workspace/moral_posttraining"
rsync -az --delete -e "ssh -i $HOME/.ssh/runpod_mft -p $PORT" --exclude '.venv*' --exclude 'runs*' --exclude 'results*' \
  --exclude logs --exclude '__pycache__' --exclude .env --exclude data/smoke --exclude 'data/seeds/social-chem-101' \
  ./ "root@$IP:/workspace/moral_posttraining/"
python3 -c "
import json,sys; k=json.load(open(sys.argv[1]))['AccessKey']
print('[default]\naws_access_key_id = %s\naws_secret_access_key = %s' % (k['AccessKeyId'], k['SecretAccessKey']))" "$KEYJSON" | $SSH "cat > ~/.aws/credentials && chmod 600 ~/.aws/credentials && printf '[default]\nregion = us-east-1\n' > ~/.aws/config"
grep -E '^(OPENAI_|MFT_PROVIDER)' .env | $SSH "cat > /workspace/moral_posttraining/.env"
$SSH "cd /workspace/moral_posttraining && (command -v ~/.local/bin/uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1) \
  && ([ -d .venv ] || ~/.local/bin/uv venv -q --python 3.12) \
  && ~/.local/bin/uv pip install -q torch --index-url https://download.pytorch.org/whl/cu128 \
  && ~/.local/bin/uv pip install -q -e '.[train]' awscli \
  && .venv/bin/python -c 'import torch; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available(), torch.cuda.get_device_name(0))' \
  && .venv/bin/aws s3 ls s3://mft-storage-bucket-9l4jkoc0ptuv/ | head -3 \
  && PYTHONPATH=src .venv/bin/python -W ignore -c 'from mft.llm import LLM; print(\"deepseek:\", LLM.from_env(\"bedrock\",\"deepseek.v3.2\").text(\"Reply with exactly: OK\"))'"
echo "pod ready: $IP:$PORT"
