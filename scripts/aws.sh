#!/usr/bin/env bash
# Run experiments on an AWS EC2 machine, driven from this computer with the AWS CLI.
#
#   scripts/aws.sh setup              once (and after your IP changes): key pair, SSH rule
#   scripts/aws.sh up [ref]           start the machine; clone the repo at ref (default
#                                     HEAD, must be pushed), uv sync, run the tests
#   scripts/aws.sh run <configs...>   start scripts/run_configs.py in tmux on the machine;
#                                     arguments as for run_configs.py
#   scripts/aws.sh status             machine, uptime, batch state and the end of its log
#   scripts/aws.sh log                follow the batch log (Ctrl-C stops only the tail)
#   scripts/aws.sh ssh                shell on the machine, in the repository
#   scripts/aws.sh sync [ref]         check out another pushed commit (no batch running)
#   scripts/aws.sh fetch              copy saved_artifacts/ into the main checkout
#   scripts/aws.sh down [--force] [--no-fetch]
#                                     fetch, then terminate the machine (deletes its disk)
#
# Settings, as environment variables:
#   PINN_AWS_REGION   region                          (us-east-1)
#   PINN_AWS_TYPE     instance type                   (c7a.24xlarge: 96 vCPU, x86)
#   PINN_AWS_SPOT     1 for a spot instance: cheaper, may be stopped by AWS  (0)
#   PINN_AWS_KEY      SSH private key; its .pub is imported   (~/.ssh/id_rsa)
#   PINN_AWS_DISK_GB  disk size                       (30)
#
# Run it from a checkout of this repository: the ref is resolved there, and fetch and
# down copy the results into that repository's main checkout. The machine is found by
# its Name tag, so at most one runs at a time.

set -euo pipefail

REGION=${PINN_AWS_REGION:-us-east-1}
TYPE=${PINN_AWS_TYPE:-c7a.24xlarge}
SPOT=${PINN_AWS_SPOT:-0}
KEY=${PINN_AWS_KEY:-$HOME/.ssh/id_rsa}
DISK_GB=${PINN_AWS_DISK_GB:-30}

TAG=pinn-runner
KEY_NAME=pinn-runner
GROUP_NAME=pinn-runner-ssh
REMOTE_DIR=pinn-effective-hamiltonian
REMOTE_LOG=run_configs.log
SESSION=runs
KNOWN_HOSTS=$HOME/.ssh/known_hosts_pinn_aws

HERE=$(cd "$(dirname "$0")" && pwd)
SSH_OPTS=(-i "$KEY" -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=30
  -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KNOWN_HOSTS")

die() {
  echo "error: $*" >&2
  exit 1
}

awsr() { aws --region "$REGION" --output text "$@"; }

remote() { ssh "${SSH_OPTS[@]}" "ubuntu@$IP" "$@"; }

# The machine tagged $TAG that is not terminated, or nothing.
instance_id() {
  awsr ec2 describe-instances \
    --filters "Name=tag:Name,Values=$TAG" \
    "Name=instance-state-name,Values=pending,running,stopping,stopped" \
    --query 'Reservations[].Instances[].InstanceId'
}

# Sets ID and IP of the running machine, or exits.
require_instance() {
  ID=$(instance_id)
  [ -n "$ID" ] || die "no machine is up (scripts/aws.sh up)"
  case $ID in *[[:space:]]*) die "several machines tagged $TAG: $ID" ;; esac
  IP=$(awsr ec2 describe-instances --instance-ids "$ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress')
  [ "$IP" != None ] || die "machine $ID has no public IP (is it stopped?)"
}

batch_running() { remote tmux has-session -t "$SESSION" 2>/dev/null; }

group_id() {
  awsr ec2 describe-security-groups --filters "Name=group-name,Values=$GROUP_NAME" \
    --query 'SecurityGroups[0].GroupId'
}

allow_my_ip() {
  local ip out
  ip=$(curl -fsS https://checkip.amazonaws.com)
  if out=$(awsr ec2 authorize-security-group-ingress --group-id "$1" \
    --protocol tcp --port 22 --cidr "$ip/32" 2>&1); then
    echo "SSH allowed from $ip"
  elif [[ $out == *InvalidPermission.Duplicate* ]]; then
    echo "SSH already allowed from $ip"
  else
    die "$out"
  fi
}

# Full hash of a ref of the current repository, which must be on GitHub.
pushed_commit() {
  local sha
  sha=$(git rev-parse --verify --quiet "$1^{commit}") || die "unknown ref: $1"
  git fetch -q origin
  [ -n "$(git branch -r --contains "$sha")" ] ||
    die "commit ${sha:0:7} is not on GitHub: push its branch first (git push -u origin <branch>)"
  echo "$sha"
}

# Install uv, clone or update the repository at commit $1, uv sync, run the tests.
bootstrap() {
  scp -q "${SSH_OPTS[@]}" "$HERE/run_configs.py" "ubuntu@$IP:run_configs.py"
  remote bash -s -- "$(git remote get-url origin)" "$1" "$REMOTE_DIR" <<'EOF'
set -euo pipefail
url=$1 sha=$2 dir=$3
[ -x ~/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
export PATH=$HOME/.local/bin:$PATH
[ -d "$dir" ] || git clone -q "$url" "$dir"
cd "$dir"
git fetch -q origin
git checkout -q --detach "$sha"
uv sync -q
echo "checked out $(git log -1 --format='%h %s')"
echo "running the tests..."
if uv run pytest -q; then
  echo "tests passed"
else
  echo "WARNING: tests failed on this machine; find out why before trusting its results"
fi
EOF
}

cmd_setup() {
  local sg
  [ -f "$KEY.pub" ] || die "no public key $KEY.pub (set PINN_AWS_KEY)"
  if awsr ec2 describe-key-pairs --key-names "$KEY_NAME" >/dev/null 2>&1; then
    echo "key pair $KEY_NAME exists"
  else
    awsr ec2 import-key-pair --key-name "$KEY_NAME" \
      --public-key-material "fileb://$KEY.pub" >/dev/null
    echo "imported $KEY.pub as key pair $KEY_NAME"
  fi
  sg=$(group_id)
  if [ "$sg" = None ]; then
    sg=$(awsr ec2 create-security-group --group-name "$GROUP_NAME" \
      --description "SSH to the pinn runner" --query GroupId)
    echo "created security group $sg"
  fi
  allow_my_ip "$sg"
}

cmd_up() {
  local sha sg arch ami_arch
  [ -z "$(instance_id)" ] || die "a machine is already up: $(instance_id)"
  sha=$(pushed_commit "${1:-HEAD}")
  sg=$(group_id)
  [ "$sg" != None ] || die "run scripts/aws.sh setup first"
  allow_my_ip "$sg"
  arch=$(awsr ec2 describe-instance-types --instance-types "$TYPE" \
    --query 'InstanceTypes[0].ProcessorInfo.SupportedArchitectures[0]')
  case $arch in
    x86_64) ami_arch=amd64 ;;
    arm64) ami_arch=arm64 ;;
    *) die "unsupported architecture of $TYPE: $arch" ;;
  esac
  local market=()
  if [ "$SPOT" = 1 ]; then
    market=(--instance-market-options
      'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}')
  fi

  ID=$(awsr ec2 run-instances \
    --image-id "resolve:ssm:/aws/service/canonical/ubuntu/server/24.04/stable/current/$ami_arch/hvm/ebs-gp3/ami-id" \
    --instance-type "$TYPE" --key-name "$KEY_NAME" --security-group-ids "$sg" \
    --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=$DISK_GB,VolumeType=gp3,DeleteOnTermination=true}" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$TAG}]" \
    ${market[@]+"${market[@]}"} \
    --query 'Instances[0].InstanceId')
  trap 'echo "warning: machine $ID is still running and billing: scripts/aws.sh down" >&2' EXIT
  echo "launched $ID ($TYPE in $REGION), waiting for it to boot..."
  awsr ec2 wait instance-running --instance-ids "$ID"
  require_instance
  ssh-keygen -R "$IP" -f "$KNOWN_HOSTS" >/dev/null 2>&1 || true
  local tries=0
  until remote true 2>/dev/null; do
    tries=$((tries + 1))
    [ $tries -lt 40 ] || die "no SSH connection to $IP after 40 tries"
    sleep 5
  done
  bootstrap "$sha"
  trap - EXIT
  echo
  echo "machine $ID is up at $IP and billing until scripts/aws.sh down"
  echo "next: scripts/aws.sh run configs/<dir or files>"
}

cmd_sync() {
  local sha
  require_instance
  ! batch_running || die "a batch is running (scripts/aws.sh status)"
  sha=$(pushed_commit "${1:-HEAD}")
  bootstrap "$sha"
}

cmd_run() {
  [ $# -gt 0 ] || die "usage: scripts/aws.sh run <configs or directories...> [run_configs.py options]"
  require_instance
  ! batch_running || die "a batch is already running (scripts/aws.sh status)"
  scp -q "${SSH_OPTS[@]}" "$HERE/run_configs.py" "ubuntu@$IP:run_configs.py"
  local inner
  inner="cd $REMOTE_DIR && ~/.local/bin/uv run python ~/run_configs.py$(printf ' %q' "$@") 2>&1 | tee -a ~/$REMOTE_LOG"
  remote "tmux new-session -d -s $SESSION $(printf %q "$inner")"
  echo "batch started at $(remote git -C "$REMOTE_DIR" log -1 --format=%h)"
  sleep 3
  remote tail -n 5 "$REMOTE_LOG"
  echo "follow it with scripts/aws.sh log; it keeps running if you disconnect"
}

cmd_status() {
  ID=$(instance_id)
  if [ -z "$ID" ]; then
    echo "no machine is up in $REGION"
    return
  fi
  awsr ec2 describe-instances --instance-ids "$ID" --query \
    'Reservations[].Instances[].[InstanceId,InstanceType,State.Name,PublicIpAddress,LaunchTime]'
  require_instance
  remote "uptime -p; tmux has-session -t $SESSION 2>/dev/null && echo 'batch: running' || echo 'batch: not running'; tail -n ${LINES_SHOWN:-15} $REMOTE_LOG 2>/dev/null || true"
}

cmd_log() {
  require_instance
  ssh "${SSH_OPTS[@]}" -t "ubuntu@$IP" tail -n 50 -F "$REMOTE_LOG"
}

cmd_ssh() {
  require_instance
  ssh "${SSH_OPTS[@]}" -t "ubuntu@$IP" "cd $REMOTE_DIR && exec bash -l"
}

cmd_fetch() {
  local main rsh
  require_instance
  main=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")
  rsh="ssh$(printf ' %q' "${SSH_OPTS[@]}")"
  mkdir -p "$main/saved_artifacts/run_logs"
  if remote test -d "$REMOTE_DIR/saved_artifacts"; then
    rsync -a -e "$rsh" "ubuntu@$IP:$REMOTE_DIR/saved_artifacts/" "$main/saved_artifacts/"
  fi
  if remote test -f "$REMOTE_LOG"; then
    rsync -a -e "$rsh" "ubuntu@$IP:$REMOTE_LOG" "$main/saved_artifacts/run_logs/aws_$ID.log"
  fi
  echo "copied the machine's saved_artifacts/ into $main/saved_artifacts/"
}

cmd_down() {
  local force=0 fetch=1 answer left
  for arg in "$@"; do
    case $arg in
      --force) force=1 ;;
      --no-fetch) fetch=0 ;;
      *) die "unknown option: $arg" ;;
    esac
  done
  ID=$(instance_id)
  [ -n "$ID" ] || die "no machine is up in $REGION"
  if [ $fetch = 1 ]; then
    require_instance
    if batch_running && [ $force = 0 ]; then
      die "a batch is still running: wait for it, or use --force"
    fi
    cmd_fetch
  fi
  printf 'Terminate %s? Its disk and anything not fetched are deleted. Type yes: ' "$ID"
  read -r answer
  [ "$answer" = yes ] || die "not terminated"
  awsr ec2 terminate-instances --instance-ids "$ID" >/dev/null
  awsr ec2 wait instance-terminated --instance-ids "$ID"
  echo "terminated $ID"
  left=$(awsr ec2 describe-instances \
    --filters "Name=instance-state-name,Values=pending,running,stopping,stopped" \
    --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name]')
  if [ -z "$left" ]; then
    echo "no other instances in $REGION"
  else
    printf 'other instances in %s:\n%s\n' "$REGION" "$left"
  fi
}

command -v aws >/dev/null || die "the AWS CLI is not installed (brew install awscli)"
[ $# -gt 0 ] || die "usage: scripts/aws.sh setup|up|run|status|log|ssh|sync|fetch|down (see the header)"
cmd=$1
shift
case $cmd in
  setup | up | run | status | log | ssh | sync | fetch | down) "cmd_$cmd" "$@" ;;
  *) die "unknown command: $cmd" ;;
esac
