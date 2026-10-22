# Kimi-K2 B=4096 grid

Two ways to get the code, both work with the same launcher:

- **git:** `git clone -b kimi-k2 git@github.com:RachitBansal/verl.git` and run `examples/kimi_k2/b4096_grid/launch_b4096_grid.sh` from the clone.
- **shared snapshot** (no clone needed): `/n/netscratch/dam_lab/Everyone/brachit/share/kimi_k2_hp128/` holds `verl/`, `pyextras/`, this README and the launcher.

Nothing needs to be installed: the launch script activates `/n/holylabs/dam_lab/Lab/brachit/envs`, puts `verl/`
on `PYTHONPATH` (plus `pyextras/`, the wandb dependencies missing from that env), and reads model and data
from `/n/holylabs/LABS/sham_lab/Everyone/rl_cbs/` (world-readable).

## The six runs

| run name (wandb, project `kimi_k2_cbs`) | mini-batch (prompts) | inner steps / batch | lr |
|---|---|---|---|
| `k2_n16_bsz4096_tau0.03_seq_sum_squared_mini1024_ep1_lr1e-5_kl0_hp128` | 1024 | 4 | 1e-5 |
| `..._mini1024_ep1_lr3e-6_kl0_hp128` | 1024 | 4 | 3e-6 |
| `..._mini1024_ep1_lr3e-5_kl0_hp128` | 1024 | 4 | 3e-5 |
| `..._mini256_ep1_lr1e-5_kl0_hp128` | 256 | 16 | 1e-5 |
| `..._mini256_ep1_lr3e-6_kl0_hp128` | 256 | 16 | 3e-6 |
| `..._mini256_ep1_lr3e-5_kl0_hp128` | 256 | 16 | 3e-5 |

Common settings: Qwen2.5-Math-1.5B-Instruct, DAPO-Math-17k, 4096 prompts x 16 rollouts per step, tau = 0.03,
seq-sum squared loss, no KL, no advantage std-normalisation, 100 outer steps, AIME 1983-2024 eval (mean@1,
933 problems) every 5 steps, checkpoint every 20 steps, 128 extra prompts per step held out of the update
for the held-out regression trace (`HOLDOUT_PROMPTS=128`, hence the `_hp128` tag).

## Launch

```bash
cd <this directory>
ACCOUNT=<your slurm account> PARTITION=<gpu partition> OUTPUT_DIR=<a directory you can write> \
  bash launch_b4096_grid.sh --dry-run     # prints the six sbatch commands
ACCOUNT=... PARTITION=... OUTPUT_DIR=... bash launch_b4096_grid.sh   # submits them
```

- Each job: 1 node, 4 GPUs (H100 or H200), `--mem=400G` (200G host-OOMs at this batch size), 2 days.
  On a mixed partition pin the GPU type, e.g. `GPU_ARG="--gpus-per-node=nvidia_h200:4"`.
- `OUTPUT_DIR` receives `<run name>/global_step_N/` checkpoints (~10 GB kept per run, last 2) and
  `slurm_logs/`.
- Wall time: the mini-1024 cells reach roughly step 85 in 2 days, the mini-256 cells somewhat less. When a
  job times out, resubmit the identical command (or re-run the launcher with `MINIS=... LRS=...` for that
  cell): verl resumes from the last checkpoint in `OUTPUT_DIR/<run name>/` automatically.
- wandb: runs log to entity `harvardml`, project `kimi_k2_cbs`; `wandb login` once beforehand, or set
  `WANDB_ENTITY=<yours>` / `PROJECT=<yours>`.
- Subsets: `MINIS="256" LRS="1e-5 3e-5" bash launch_b4096_grid.sh`.

## What to look at

`val-core/aime-1983-2024/reward/mean@1` (s.e. about 0.016), `critic/rewards/mean` (train reward),
`actor/entropy`, `response_length/clip_ratio`, and the held-out inner-solve trace
`kimi_holdout_trace/solve_ratio_s{k}`, `kimi_holdout_trace/rho_s{k}` (k = 1..4 or 1..16). See
`verl/examples/kimi_k2/METRICS.md`.
