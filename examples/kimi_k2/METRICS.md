# METRICS.md — how we measure whether the Kimi-K2 inner regression is being solved

*Applies to `loss_mode=kimi_k2` on branch `kimi-k2` of the verl repo (`verl/examples/kimi_k2/`). All names are wandb keys.*

## The objective and its two scales

Per prompt with $K$ samples $y_i\sim\pi_{\text{old}}$, binary reward $r_i$, baseline $\bar r$:

$$\mathcal L(\theta)=\frac1K\sum_i\big(A_i-\tau\,\Delta_i\big)^2,\qquad A_i=r_i-\bar r,\qquad \Delta_i=\sum_t\log\frac{\pi_\theta(y_{i,t}|\cdot)}{\pi_{\text{old}}(y_{i,t}|\cdot)}.$$

The exact minimiser is the mirror-descent iterate $\Delta_i^\star=A_i/\tau$ (the **target**, in nats of sequence log-prob). Two scales matter: how big the target is ($|A_i|\le1$, so $1/\tau$ nats), and how far the policy actually moved ($\Delta_i$). "Solving" means $\Delta\approx A/\tau$ on the sequences the regression was fit on, and ideally on sequences it was not fit on.

## What one "step" is

One rollout batch is `train_batch_size × rollout.n` sequences (512 × 16 = 8192). The actor takes `ppo_epochs × (batch / ppo_mini_batch_size)` optimizer steps on it; with `MINI=128` prompts that is 4 steps per epoch, each on a different quarter of the batch (standard PPO mini-batching, not repeated steps on the same quarter); `EPOCHS=2` revisits every row twice. `MINI=BSZ, EPOCHS=1` is one step, in which case the log-ratio is identically zero at the update point and the K2 gradient is exactly the mean-baseline policy gradient (Dr. GRPO). All "before/after" metrics below bracket the *whole* set of inner steps on one rollout batch.

## Per-micro-batch metrics (`kimi/*`, averaged over the micro-batches of the update)

These are computed on the data the current gradient step is taken on, at the current parameters, so they describe the loss *as it is being optimised* (early micro-batches see the policy before most of the steps, late ones after).

| key | definition | reading |
|---|---|---|
| `kimi/reg_loss` | $\tfrac1B\sum_i(A_i-\tau\Delta_i)^2$ | the published objective value |
| `kimi/reg_loss_at_zero_delta` | $\tfrac1B\sum_i A_i^2$ | the loss if the policy had not moved; `reg_loss / reg_loss_at_zero_delta` < 1 means progress |
| `kimi/target_rms` | rms of $A_i/\tau$ (nats) | how much change the objective is asking for |
| `kimi/delta_rms`, `kimi/delta_abs_max` | rms / max of $\Delta_i$ (nats) | how much the policy actually moved on these rows |
| `kimi/resid_rms` | rms of $A_i-\tau\Delta_i$ | unexplained part of the advantage |
| `kimi/s` | $\|\Delta\|/\|A/\tau\|$ | size of the realised change relative to the target: 1 = right size, ≪1 undershoot, ≫1 overshoot |
| `kimi/rho` | $\cos(\Delta,\,A/\tau)$ | alignment of the realised change with the target |
| `kimi/k1_to_old`, `kimi/k3_to_old` | token-mean of $-d$ and of $e^{d}-d-1$, $d$ = token log-ratio | KL to $\pi_{\text{old}}$ (k3 is the robust estimator); the "step size" in distribution space |
| `kimi/token_logratio_rms` | rms of token log-ratios | where the change concentrates (large values with small `delta_rms` = a few tokens moved a lot) |
| `kimi/frac_zero_adv` | fraction of sequences with $A_i=0$ | rows from all-correct / all-wrong groups; they carry no target and pin $\Delta_i$ to 0 |

Identity: $\text{solve ratio}=\|\Delta-A/\tau\|^2/\|A/\tau\|^2=1-2s\rho+s^2$. So a ratio below 1 requires $0<s<2\rho$; the best achievable value for a given alignment is $1-\rho^2$ at $s=\rho$. `s` and `rho` are set to 0 on micro-batches whose targets are all zero.

## Before/after metrics on fixed rows (`kimi_probe/*`, `kimi_holdout/*`), once per rollout batch

The actor re-evaluates the regression on fixed rows **before** any optimizer step and **after** all of them, always against the stored `old_log_probs` (never the on-policy shortcut). Two row sets:

- **probe** (`kimi_probe_num_seqs`, default 64 per DP rank): a fixed prefix of the rows that *are* trained on. Measures the in-sample fit.
- **hold-out** (`kimi_holdout_num_seqs`, default 64 per rank in the b512 script): rows removed from the update entirely. Measures whether the fit generalises to sequences the regression never saw; this is the quantity REBEL-type guarantees are stated in (expected regression error), not the training residual.

| key | definition | reading |
|---|---|---|
| `<set>/reg_loss_before`, `<set>/reg_loss_after` | loss on the rows at $\theta_t$ and at $\theta_{t+1}$ | |
| **`<set>/solve_ratio`** | `reg_loss_after / reg_loss_before` | **the headline number.** 1 = the regression did not move on these rows; <1 = fit; >1 = the update made these rows' residuals worse |
| `<set>/solve_ratio_vs_zero_delta` | `reg_loss_after / mean A²` | same, with the analytic "no movement" denominator |
| `<set>/recompute_floor_share` | `(reg_loss_before − mean A²) / mean A²` | how much of the "before" loss is FSDP recompute noise rather than target; must be ≈0, otherwise a small-τ fit is fitting noise |
| `<set>/s_after`, `<set>/rho_after` | as above, after the update | size and alignment of what the update did to these rows |
| `<set>/delta_rms_before` | should be ≈0 | the recompute floor in nats |
| `<set>/delta_rms_after`, `<set>/delta_abs_max_after`, `<set>/target_rms_before` | nats | realised change vs. requested change |
| `<set>/k3_to_old_after` | KL/token realised on these rows | the committed step size; compare with the k3 the loss "wanted" ($\approx\mathbb E[A^2]/(2\tau^2T^2)$ if spread uniformly over $T$ tokens) |

How to read the pair: `kimi_probe/solve_ratio` well below 1 with `kimi_holdout/solve_ratio` ≈ 1 means the update interpolates the sampled advantages without moving unseen sequences in the direction of theirs (memorisation of noisy labels). Both below 1 means the regression has found a direction that transfers. `kimi_holdout/solve_ratio` > 1 with negative `rho_after` means the step actively moved held-out sequences against their own advantages.

## Which metric is *the* metric: the held-out trace

With one epoch over the rollout batch, every inner step k consumes a fresh mini-batch, so the "in-sample" probe rows (`kimi_probe/*`, first rows of the batch) receive a direct optimizer step at k = 1 and indirect ones afterwards. Their loss conflates fitting with memorisation and is **not** a measure of how well the inner regression is solved. The quantity to report is the **held-out** trace (`kimi_holdout_trace/*_s{k}` and `kimi_holdout/*`): rows removed from the update entirely, fixed across the inner loop, re-evaluated after every step.

What the held-out rows are: the last `kimi_holdout_num_seqs` rows of each DP rank's shard after verl's `balance_batch` reordering (sequence-length balancing across ranks scrambles row order), i.e. an arbitrary set of sequences from the batch, most of whose prompts also appear in the update through sibling responses. They are therefore mostly a *same-prompt* hold-out, which is the relevant one for memorisation. They must be resampled from π_old at every outer step: the regression targets A_i/τ and the log-ratio Δ_i are defined against the current π_old, so a batch frozen across outer steps would measure a different objective. Fixing the *prompts* across outer steps (fresh responses each step) is possible but needs trainer support for placing them on every rank; the default 64 rows per rank (256 total) give ρ a noise floor of about 0.06, so raise `kimi_holdout_num_seqs` to 256 when the hold-out trace is the headline. The probe/in-sample numbers stay logged as a memorisation diagnostic only.

## Per-inner-step trace (`kimi_probe_trace/*_s{k}`, `kimi_holdout_trace/*_s{k}`), opt-in

With `kimi_probe_per_step: true` (the b512 launch script sets it via `TRACE=True`), the same probe and held-out rows are also re-evaluated after **every** inner optimizer step, k = 1..K, where K = `ppo_epochs` × (rows per rank / `ppo_mini_batch_size` per rank). The keys are `<set>_trace/<metric>_s<k>` for `reg_loss`, `solve_ratio` (= `reg_loss` at step k / `reg_loss_before`), `s`, `rho`, `delta_rms`, `resid_rms`, `k3_to_old`. Step K coincides with the `_after` value of the once-per-batch block.

Reading: `kimi_probe_trace/solve_ratio_s{k}` is the in-sample loss curve across the inner solve; `kimi_holdout_trace/solve_ratio_s{k}` is the hold-out curve, whose minimum over k locates the number of inner steps past which the extra steps only memorise the sampled residuals. In epoch 1 the probe rows (first `kimi_probe_num_seqs` rows) receive a direct step at k = 1 and indirect ones afterwards; held-out rows receive only indirect steps. Cost: K extra no-grad forwards of (probe + held-out) rows per batch, run in eval mode on every rank; the training path is unchanged.

## Companion metrics (existing verl keys)

`rollout_corr/k3_kl`, `rollout_corr/chi2_token`: vLLM-vs-trainer log-prob mismatch (needs `rollout.calculate_log_probs=True`); a hidden off-policy floor. `actor/entropy`, `response_length/mean`, `actor/grad_norm`: collapse indicators. `critic/rewards/mean`: train reward. `val-core/aime-1983-2024/reward/mean@1`: the eval the GRPO controls use (T=1.0, one sample).

## Caveats

- Probe/hold-out rows are a prefix/suffix of each rank's local batch after verl's sequence-length balancing, i.e. a quasi-random subset of prompts; per-rank values are averaged across ranks.
- `k3` with a few tokens at the ±20 clamp produces outliers of order $10^3$–$10^8$; treat a spike as "some tokens moved more than 20 nats", not as a mean.
- The per-micro-batch `kimi/*` values mix states before and after intermediate steps; use the probe/hold-out pairs for the per-batch solve question.
