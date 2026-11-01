# Running experiments on AWS

`scripts/aws.sh` runs batches of configs on one EC2 machine, driven from your computer
with the AWS CLI; `scripts/run_configs.py` is the batch runner it starts on the machine
(it also works locally). This guide covers the one-time account setup, a smoke test, a
full run, and what to check at each step.

## How it works

1. The configs to run are committed and **pushed** on a branch: the machine clones the
   repository from GitHub at that commit, so every run directory records a commit that
   exists remotely, as on a local run.
2. `up` launches the machine, installs uv, clones, runs `uv sync` and the tests.
3. `run` starts `run_configs.py` in tmux on the machine: configs run longest first, one
   core per seed, as many at once as the cores allow. It survives disconnects.
   `--stop-when-done` shuts the machine down when the batch ends.
4. `down` copies the machine's `saved_artifacts/` into the main checkout, then
   terminates the machine (which deletes its disk).

The machine is found by its `Name` tag (`pinn-runner`), so at most one runs at a time.
It bills from `up` until it stops or is terminated: a running `c7a.24xlarge` (96 vCPU)
costs about $5/h; a stopped one only its disk, a few cents a day.

## One-time setup

**AWS CLI** (macOS):

```bash
brew install awscli
aws login                                  # browser sign-in, short-lived credentials
aws configure set region us-east-2
aws sts get-caller-identity                # prints your account
```

`aws login` sessions expire after some hours: when a command reports an expired
session, run `aws login` again. A batch already running on the machine does not need
your credentials.

**Account requirements:**

- **Paid plan.** On the Free plan only small free-tier instance types launch; anything
  else fails with `The specified instance type is not eligible for Free Tier`. Upgrade
  in Billing and Cost Management; remaining credits carry over.
- **vCPU quota in the region you use** (quotas are per region): Service Quotas → EC2 →
  "Running On-Demand Standard (A, C, D, H, I, M, R, T, Z) instances" (code
  `L-1216C47A`). The scripts default to us-east-2; set `PINN_AWS_REGION` for another.
- **Spend limit / budget.** The spend limit counts usage *before* credits, so set it
  above the expected cost of a run (a full sweep is $15–20). Budget alerts should
  include usage, not usage net of credits.
- **Credits** (if any) must list "Amazon Elastic Compute Cloud" among their applicable
  services (searching "EC2" does not find it).

**Checks** (free; an organization policy can deny regions, actions or spot instances,
and a dry run shows it):

```bash
aws service-quotas get-service-quota --service-code ec2 --quota-code L-1216C47A --query Quota.Value
aws ec2 describe-vpcs --filters Name=is-default,Values=true --query 'Vpcs[].VpcId' --output text
aws ec2 run-instances --dry-run --instance-type c7a.24xlarge \
  --image-id resolve:ssm:/aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id
```

Expected: the quota (e.g. `96.0`), one `vpc-...` id, and `Request would have
succeeded, but DryRun flag is set`. A dry run does not check the account plan, so the
Free-plan restriction only shows on a real launch.

**SSH key pair and firewall rule** (key from `~/.ssh/id_rsa.pub`; set `PINN_AWS_KEY`
for another key). If the key has a passphrase, load it into the agent first, since the
scripts use non-interactive SSH:

```bash
ssh-add --apple-use-keychain ~/.ssh/id_rsa
scripts/aws.sh setup
```

Expected: `imported ... as key pair pinn-runner`, `created security group sg-...`,
`SSH allowed from <your IP>`. Run `setup` again whenever your public IP changes (SSH
then times out).

## Before every run

1. The configs are committed on a branch and pushed: `git push -u origin <branch>`.
   `up` refuses a commit that is not on GitHub.
2. Run the scripts from a checkout that has them (`scripts/aws.sh`): the ref given to
   `up` is resolved in that repository, and `fetch`/`down` copy the results into its
   main checkout (`saved_artifacts/`).
3. Experiment names must not collide with runs from another platform you want to keep
   apart: either distinct config names, or `run ... --output saved_artifacts/runs_aws`.
4. `aws login`, so the session does not expire half-way through `up`.

## Smoke test (8 vCPU, under $1)

```bash
PINN_AWS_TYPE=c7a.2xlarge scripts/aws.sh up <branch>
scripts/aws.sh run --stop-when-done configs/<dir>/<one config>.yaml --epochs 300 --seeds 100 101 --output saved_artifacts/smoke
scripts/aws.sh status         # after a minute: "machine is stopped"
scripts/aws.sh start
scripts/aws.sh down           # copies the results, then asks; type yes
```

Use `--output saved_artifacts/smoke` for checks: `run_configs.py` skips configs that
already have a finished run of the current commit in the output root, so a short check
in the default root would make the real run skip that config.

## Full run

```bash
scripts/aws.sh up <branch>
scripts/aws.sh run --stop-when-done configs/<dir> [configs/<dir2> ...]
scripts/aws.sh status         # a few minutes later: "batch: running" and START lines
```

Then, when `status` reports `machine is stopped` (or `ALL DONE` in the log):

```bash
scripts/aws.sh start
scripts/aws.sh down           # fetch, confirm with yes, terminate
```

`scripts/aws.sh log` follows the log (Ctrl-C stops only the tail); `scripts/aws.sh ssh`
opens a shell in the repository on the machine. If the batch was interrupted, `run` with
the same arguments resumes it: finished configs are skipped.

### Expected output of `up`

```
launched i-... (c7a.24xlarge in us-east-2), waiting for it to boot...
checked out <hash> <subject>
running the tests...
...
machine i-... is up at <IP> and billing until scripts/aws.sh down
```

On x86 machines a few tests fail and `up` prints `WARNING: tests failed`. These are
expected, and do not affect results:

- `tests/test_golden.py::test_train_with_learned_couplings[...]`: golden files were
  generated on Apple Silicon; float32 training differs in the last digits (about 1e-6)
  on x86, slightly differently on Intel and AMD.

On x86, torch results also depend on the number of threads (about 1e-8 relative).
`run_configs.py` always trains one thread per seed, so runs on the machine are
consistent with each other, but compare seeds only within one platform.

Any other failure: stop and investigate before running.

## Finishing and checking

`down` ends with `terminated i-...` and `no other instances in <region>`. Then, in the
console, **with the region selector (top right) on your region**:

- EC2 → Instances: the machine shows `Terminated` (it disappears after about an hour).
- EC2 → Volumes: empty.
- Billing, the next day: the run's EC2 charges, offset by credits if you have them.

From the terminal, nothing should be listed:

```bash
aws ec2 describe-instances --filters Name=instance-state-name,Values=pending,running,stopping,stopped \
  --query 'Reservations[].Instances[].InstanceId' --output text
```

**Emergency stop:** EC2 → Instances → select `pinn-runner` → Instance state →
Terminate, or `scripts/aws.sh down --no-fetch` (only once the results are fetched or
not needed). A *stopped* machine still bills for its disk; terminate it when done.

## Reproducibility

- Results from different CPUs are not bit-identical for the same seed, and over long
  training a seed near a basin boundary can end differently. Run a whole study on one
  platform and do not mix Mac and AWS seeds in one table or figure.
- Record the platform (instance type, region) in the study report; `meta.json` records
  the commit and package versions but not the CPU.

## Troubleshooting

| Message | Cause and fix |
|---|---|
| `Your session has expired` | `aws login` |
| `not eligible for Free Tier` | the account is on the Free plan: upgrade (see setup) |
| `explicit deny in a service control policy` | an organization policy forbids it (region, spot, action); use what the dry-run checks allow |
| `commit ... is not on GitHub` | push the branch |
| `a machine is already up` | `scripts/aws.sh status`; finish or `down` the existing one |
| `machine ... is stopped` | `scripts/aws.sh start`, then `fetch` or `down` |
| SSH times out | your IP changed: `scripts/aws.sh setup` |
| `a batch is still running` (from `down`) | wait, or `down --force` (fetches what exists) |
| `FAILED <config>` in the log | `scripts/aws.sh ssh`, then `cat saved_artifacts/run_logs/*/<config>.log` |
