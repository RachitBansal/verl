# Kimi-K2 squared-regression objective (`loss_mode=kimi_k2`)

Branch `kimi-k2` (based on `origin/cbs-experiments`, the code that produced the `grpo_on_policy_cbs` GRPO runs).

## Objective (published form, Kimi K2 tech report arXiv:2507.20534)

```
L(θ) = E_x [ 1/K Σ_i ( r(x,y_i) − r̄(x) − τ · log π_θ(y_i|x)/π_old(y_i|x) )² ],   y_i ~ π_old
```

* sequence-level log-ratio `Δ_i = Σ_t log π_θ(y_{i,t}) − log π_old(y_{i,t})` (`kimi_target_level=seq_sum`)
* `A_i = r_i − r̄` from `algorithm.adv_estimator=grpo` + `algorithm.norm_adv_by_std_in_grpo=False`
* no importance ratio, no clipping, no KL-to-reference (set `use_kl_loss=True`/`kl_loss_coef` to add one)
* exact minimiser is the mirror-descent iterate `Δ_i = A_i/τ`, i.e. `π_{t+1} ∝ π_old · exp(A/τ)`.
  With binary rewards `|A_i| ≤ 1`, so `1/τ` is the target change in sequence log-prob (nats).
* one optimizer step per rollout batch (`ppo_mini_batch_size = train_batch_size`, `ppo_epochs=1`) is
  REINFORCE with the group-mean baseline (Dr. GRPO). The regression is only *solved* when several
  steps are taken on the same rollouts (`ppo_mini_batch_size < train_batch_size` and/or `ppo_epochs > 1`).

Code: `verl/trainer/ppo/core_algos.py::compute_policy_loss_kimi_k2` (+ `kimi_k2_regression_stats`,
`kimi_k2_summary_metrics`), config in `verl/workers/config/actor.py::PolicyLossConfig` and
`verl/trainer/config/actor/actor.yaml`, solve tracking in `verl/workers/actor/dp_actor.py`.
Unit tests: `tests/trainer/ppo/test_kimi_k2_loss_on_cpu.py`.

## Config keys (`actor_rollout_ref.actor.policy_loss.*`)

| key | default | meaning |
|---|---|---|
| `loss_mode` | `vanilla` | set to `kimi_k2` |
| `kimi_tau` | 0.01 | τ; target log-ratio per sequence = A/τ |
| `kimi_target_level` | `seq_sum` | `seq_sum` (published) or `seq_mean` (length-normalised Δ) |
| `kimi_form` | `squared` | `squared` (published) or `gradient` = `−AΔ + τ/2 Δ²` (same minimiser, step at Δ=0 equals the policy gradient) |
| `kimi_logratio_clip` | null | detach token log-ratios with magnitude above this (K2.5-style guard) |
| `kimi_probe_num_seqs` | 64 | training rows re-evaluated before/after the update |
| `kimi_holdout_num_seqs` | 0 | rows removed from the update, evaluated before/after (held-out regression loss) |
| `kimi_probe_micro_bsz` | 8 | micro-batch of the probe forwards |

## What is logged

Per micro-batch during the update (averaged over micro-batches by the trainer):

* `kimi/reg_loss` — the published objective value on that micro-batch; `kimi/reg_loss_at_zero_delta` = mean A² (its value if the policy had not moved)
* `kimi/target_rms`, `kimi/delta_rms`, `kimi/resid_rms`, `kimi/delta_abs_max` — nats
* `kimi/s` = ‖Δ‖/‖A/τ‖ and `kimi/rho` = cos(Δ, A/τ); solve ratio = 1 − 2 s ρ + s². s≈ρ≈1 solved, s≫1 overshoot, s≈0 no movement
* `kimi/k1_to_old`, `kimi/k3_to_old`, `kimi/token_logratio_rms`, `kimi/frac_zero_adv`
* `rollout_corr/*` — vLLM-vs-FSDP log-prob mismatch (needs `rollout.calculate_log_probs=True`)

Once per rollout batch, on fixed rows (`kimi_probe/*`) and on rows held out of the update (`kimi_holdout/*`):

* `<prefix>/reg_loss_before`, `<prefix>/reg_loss_after`, **`<prefix>/solve_ratio`** = after/before
* `<prefix>/solve_ratio_vs_zero_delta` = after / mean A²; `<prefix>/recompute_floor_share` = (before − mean A²)/mean A² (how much of the "before" loss is FSDP recompute noise rather than target; should be ≈0)
* `<prefix>/s_after`, `<prefix>/rho_after`, `<prefix>/delta_rms_before` (recompute floor), `<prefix>/delta_rms_after`, `<prefix>/k3_to_old_after` (realised KL/token), `<prefix>/target_rms_before`

Probe/hold-out rows are a fixed prefix/suffix of each DP rank's local batch (after verl's sequence-length balancing), so they are a quasi-random subset of prompts.

## Running

```
cd /n/home08/brachit/coverage/verl
sbatch examples/kimi_k2/smoke_test.sh                                # 2 GPUs, 3 tiny steps
BSZ=512 N=16 LR=1e-5 TAU=0.01 MINI=128 EPOCHS=1 sbatch examples/kimi_k2/kimi_k2_b512.sh
MINI=512 ... sbatch examples/kimi_k2/kimi_k2_b512.sh                 # one step/batch == Dr. GRPO control
LOSS_MODE=vanilla STD_ADV=True KL_COEF=0.001 MINI=512 sbatch ...     # the cbs GRPO recipe itself
```

Reference GRPO controls (project `harvardml/grpo_on_policy_cbs`, n16 bsz512, 1 step/batch, clip 0.2,
KL 1e-3, std-normalised advantages, val `aime-1983-2024` mean@1 at T=1.0): lr 1e-5 peaks 0.63 at step 350
(0.50 by step 100); lr 2e-5 + KL 1e-2 reaches 0.50 by step 75; lr 3e-6 reaches 0.50 at step 275;
lr ≥ 5e-5 collapses. Base model 0.24.
