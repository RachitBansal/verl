# Kimi-K2 on a fixed π_ref dataset (`trainer.fixed_dataset`)

Plan and theory: `coverage/Phase2a_Fixed_PiRef_Dataset_Plan.md`.

The usual verl PPO loop is *generate → score → old log-probs → advantages → update* every step. The fixed-dataset
mode replaces it. A **bank** of reference-policy (π_ref = initial weights) samples is generated, scored and given
its π_ref log-probs **once**. Every training step then runs `update_actor` on that same bank, with
`old_log_probs` frozen at the π_ref log-probs. No rollouts happen during training except validation.
With a frozen anchor the whole run is one regression problem:

    L(θ) = mean_i ( A_i − τ · Δ_i )²,   Δ_i = log π_θ(y_i|x_i) − log π_ref(y_i|x_i),   A_i = r_i − b(x_i)

A bank can come from one of two sources:

| | **Pickled bank** (`fixed_dataset.path`) | **Master bank** (`fixed_dataset.master_dir` / `master_selection`) |
|---|---|---|
| what is stored | exactly one run's rows (B prompts × n + held-out sets), padded `DataProto` pickle | every DAPO prompt × 64 samples, compact shards (real tokens only) |
| built by | `MODE=build` (one job, ~25 min for 4096×16) | `MODE=master_build`, several jobs in parallel, resumable |
| a run chooses | nothing (the bank *is* the run) | prompts, k samples/prompt, held-out prompts/samples, via a selection file |
| size | 17.8 GB for 74k rows (padded to 4096 tokens) | ~10 GB for 1.15M rows |

Both produce the same in-memory `DataProto`, so everything after loading is shared.

---

## 1. Where it hooks into verl

```
verl/trainer/main_ppo.py            → RayPPOTrainer.fit()
verl/trainer/ppo/ray_trainer.py
  fit()                              ~L1580  load checkpoint, validation before training (val_before_train)
    if trainer.fixed_dataset.enable  ~L1602  → _fit_fixed_dataset(logger); return   (the normal loop never runs)
  _fit_fixed_dataset                 L1433   get the bank, then the per-step loop (§3)
```

Everything outside `_fit_fixed_dataset` is stock verl: worker setup, the hybrid FSDP actor plus vLLM rollout
engine, `_validate`, and checkpointing. Line numbers are approximate. Search for the function names.

| piece | file | function |
|---|---|---|
| per-step loop, bank source switch | `verl/trainer/ppo/ray_trainer.py` | `_fit_fixed_dataset` |
| pickled-bank build | ″ | `_build_fixed_bank`, `_generate_rows` |
| shared bank finalisation | ″ | `_finalize_fixed_bank` |
| master-bank build / load | ″ | `_build_master_bank`, `_load_master_bank` |
| per-step row order for the DP ranks | ″ | `_layout_fixed_bank` |
| τ log Ẑ baseline | ″ | `_rebaseline_fixed_bank` |
| HF weight snapshots | ″ → `verl/workers/fsdp_workers.py` | `_save_hf_weights` → `save_hf_model` |
| master shard format, pack/unpack, manifest | `verl/trainer/ppo/kimi_master_bank.py` | `pack_shard`, `save_shard`, `write_manifest`, `gather_rows`, `unpack_rows` |
| population optimum / floors / soft-value baseline | `verl/trainer/ppo/kimi_fixed_bank.py` | `population_optimum`, `bank_floor_summary`, `soft_value_baseline` |
| the K2 loss | `verl/trainer/ppo/core_algos.py` | `compute_policy_loss_kimi_k2` (`loss_mode=kimi_k2`) |
| regression stats and metrics | ″ | `kimi_k2_regression_stats`, `kimi_k2_summary_metrics`, `kimi_k2_group_metrics` |
| actor update, probes, held-out rows | `verl/workers/actor/dp_actor.py` | `update_policy`, `_kimi_regression_probe` |
| config | `verl/trainer/config/ppo_trainer.yaml` | `trainer.fixed_dataset.*` |

---

## 2. Building a bank

### 2a. Pickled bank: `_build_fixed_bank` (`MODE=build`)

1. Take the **first dataloader batch**: `data.train_batch_size` training prompts followed by
   `data.kimi_holdout_prompts` fresh prompts. `data.shuffle=False`, and `prepare_prompts.py` has already put them
   in that order.
2. `_generate_rows`: `gen_batch.repeat(n, interleave=True)` → `actor_rollout_wg.generate_sequences` (vLLM,
   the same call the PPO loop uses) → `union` with the prompts. Rows of one prompt are contiguous.
   For `holdout_sample_prompts > 0`, the first training prompts are generated a second time and given new uids.
3. `compute_reward` (DAPO boxed-answer scorer, `verl/utils/reward_score/math_reward.py`) puts the score on
   the last response token.
4. `actor_rollout_wg.compute_log_prob` (**the FSDP trainer's own forward**, not vLLM's log-probs) gives
   `old_log_probs`, which are copied to `ref_log_probs`. This is the frozen anchor, so it has exactly the
   numerics the trainer will use at θ = θ_ref (`kimi_*/delta_rms_before` ≈ 0 at step 1).
5. `_finalize_fixed_bank`:
   - `compute_advantage` (GRPO, `norm_adv_by_std_in_grpo=False`) gives A = r − group mean over the uid group.
   - `ref_pass_rate` is the group mean reward.
   - `kimi_probe` marks `probe_prompts` random complete training groups, using `fixed_dataset.seed`.
   - It writes the `bank/*` summary, including `bank_floor_summary`, the population solve-ratio floor per τ.
6. `bank.save_to_disk(path)` writes the pickle and `<path>.summary.json`.

### 2b. Master bank: `_build_master_bank` (`MODE=master_build`)

The prompt file is `prepare_master_prompts.py`'s output: every DAPO prompt within 1024 chat tokens (17,909 of
17,917), kept in DAPO order. Master prompt row i is dataset row i, and `extra_info.index` is the DAPO row.

The build steps are:

1. Write `build_config.json`: model, n, temperature, top-p/k, lengths, prompt file. Every later build job
   checks it and refuses to mix settings.
2. Shard c covers prompt rows `[c·chunk, (c+1)·chunk)`, with `master_chunk_prompts`, default 512, so
   512 × 64 = 32,768 rows per vLLM call. Job j of J builds the shards with c % J == j. Shards already on disk are
   skipped, so resubmitting a job resumes it.
3. Each shard goes through the same calls as §2a: `collate_fn([dataset[i] …])` → `_generate_rows` →
   `compute_reward` → `compute_log_prob`. Then `kimi_master_bank.pack_shard` stores only the real tokens:
   - per prompt: its token ids once (`prompt_tokens` and `prompt_offsets`), the master row, the DAPO index and
     the data source
   - per row (= sample): `response_tokens`, `ref_log_probs` (FSDP, the anchor) and `rollout_log_probs` (vLLM),
     all flat with `response_offsets`, plus `score`, `ref_entropy_sum` and `sample_index` (0..n−1)
4. `save_shard` writes `shards/shard_XXXXX.pt` and then a small `.json` sidecar (pass counts, length and entropy
   sums), both via tmp file and rename. A shard counts as done only when its sidecar exists.
5. The job that finds every shard done writes `manifest.json` (shard → prompt-row ranges) and `summary.json`
   (pass-count histogram over n, pass@n, floors, entropy and so on).

### 2c. Choosing a run's rows from the master bank

`prepare_master_selection.py --master DIR --out SEL [--train P --k K --holdout-prompts H
--holdout-sample-prompts S --monitor M --sample-offset o --seed s]` writes `SEL/selection.json` and the two
on-policy monitor files `SEL/monitor_{seen,fresh}.parquet`:

| role | prompts | samples used | flag `kimi_holdout` |
|---|---|---|---|
| train | P prompts from a seeded permutation | `[o, o+K)` | 0 |
| holdout_prompt | H other prompts (never trained on) | `[o, o+K)` | 1 |
| holdout_sample | the first S training prompts | `[o+K, o+2K)`, disjoint from the trained samples | 2 |
| monitor_fresh | M prompts used by no role (validation only) | — | — |

Options:
- `--train 0` uses every remaining prompt.
- `--same-prompts-as <old prompts dir>/indices.json` reproduces an old pickled bank's prompt sets.
- Different `--sample-offset` values give independent sample draws on the same prompts.

### 2d. Loading: `_load_master_bank`

1. `gather_rows` opens each needed shard once and slices the requested (prompt, samples) rows.
2. `unpack_rows` rebuilds the padded tensors exactly as the vLLM rollout lays them out:
   - prompt left-padded to `max_prompt_length`
   - response right-padded to `max_response_length`
   - `attention_mask = [prompt mask | response prefix mask]`
   - position ids: `clip(cumsum−1, 0)` on the prompt, then the last prompt position + 1, 2, … on the response
   - score on the last response token

   `test_master_roundtrip.py` (CPU) checks this bit-for-bit against the original tensors.
3. uids are `train:<row>`, `holdout_prompt:<row>` and `holdout_sample:<row>`. `bank_prompt_uid` of a flag-2 row
   points at the training uid of the same prompt.
4. `_finalize_fixed_bank` runs exactly as in §2a, so advantages and baselines are computed **within the selected
   k samples**, as for a pickled bank.

`actor_rollout_ref.rollout.n` must equal the selection's k; `launch.sh` sets it from the selection.

---

## 3. One training step (`_fit_fixed_dataset`)

Once, before step 1:
- `_rebaseline_fixed_bank`: with `fixed_dataset.baseline=tau_logz`, replace A by r − τ log mean_j exp(r_j/τ)
  per uid group (`soft_value_baseline`; τ = `actor.policy_loss.kimi_tau`). This also stores `kimi_baseline`.

Each step:

1. **`layout`** (driver CPU): `_layout_fixed_bank`. verl's DP dispatch
   (`make_nd_compute_dataproto_dispatch_fn`) hands each actor rank a *contiguous* chunk. So the rows are reordered
   so that every rank gets [its length-balanced share of the non-probe training rows (`_balance_batch`) | its share
   of probe rows | its share of each held-out set]. Every rank then has equal row counts, which FSDP requires.
   Probe and held-out rows stay in group order, so the per-prompt metrics see whole groups.
2. `meta_info["kimi_frozen_anchor"]=True`. This disables verl's single-mini-batch "on-policy" shortcut
   (`old_log_prob = log_prob.detach()` in `dp_actor.update_policy`), which would otherwise silently ignore the
   stored anchor when `ppo_mini_batch_size` = whole batch. With `fixed_dataset.anchor=current`, `old_log_probs`
   are recomputed each step instead; this is the re-anchored contrast cell.
3. **`update_actor`**: Ray → `ActorRolloutRefWorker.update_actor` (`fsdp_workers.py`) →
   `DataParallelPPOActor.update_policy` (`dp_actor.py`):
   - Split off the tracked row sets: held-out flags 1 and 2 leave the update, and probe rows stay in it.
     `_kimi_regression_probe` runs a no-grad forward over each (micro-batch `kimi_probe_micro_bsz`) to give the
     *before* stats.
   - Split the training rows into mini-batches (`ppo_mini_batch_size` = B, so one), and those into micro-batches
     (`use_dynamic_bsz`, ≤ `ppo_max_token_len_per_gpu` tokens). For each micro-batch:
     - `_forward_micro_batch` computes π_θ log-probs (remove-padding plus gradient checkpointing).
     - `compute_policy_loss_kimi_k2` computes the loss.
     - `loss.backward()` runs, and FSDP reduce-scatters the gradients.
   - `_optimizer_step` clips the gradient and takes the AdamW step.
   - Run the *after* probe forwards. This produces `<prefix>/*_before/_after` and `solve_ratio`.
4. Back in the trainer, the global solve ratios are recomputed from the rank-mean losses. Per-rank ratios blow up
   on ranks whose rows all have zero advantage.
5. **`testing`**: `_validate`, i.e. vLLM generation on AIME and the two monitor files (`val_kwargs.n`).
6. `save_checkpoint` every `save_freq`, and weights-only `hf_step_k/` every `hf_save_freq` and every step up to
   `hf_save_dense_until`.

### Where the time goes (B=4096 × 16, 4×H200, measured on the k2fixZ runs)

| phase | time per step | scales with |
|---|---|---|
| layout | 18 s | rows × padded width |
| update_actor: probes, 2 × 10,240 rows forward | ~65 s (estimate) | tracked rows (independent of B) |
| update_actor: training forward + recompute + backward on 65,536 rows | ~540 s (estimate) | rows × length × params |
| update_actor: optimizer step | <1 s | params |
| testing (7,828 generations) | 242 s | validation set size × `VAL_N` |
| total | ≈ 895 s | |

One-time build costs:
- 4096×16 pickled bank: generation 962 s and log-probs 242 s, about 13 s and 3.3 s per 1k rows.
- Master bank of 17,909 × 64 = 1.15M rows: about 4 h of generation plus 1 h of log-probs on one 4-GPU node,
  split over the build jobs.

---

## 4. Running it

All scripts are sbatch scripts for 4 GPUs (2 for the smoke tests). Example submissions use
`-p seas_gpu -A dam_lab --gpus-per-node=nvidia_h200:4` or the script default `kempner_h200`.

```bash
# pickled bank (original study)
python examples/kimi_k2/fixed_ref/prepare_prompts.py --out $PROMPT_DIR          # B train + H held-out + monitors
MODE=build sbatch --mem=400G examples/kimi_k2/fixed_ref/launch.sh
TAU=0.03 LR=3e-6 BASELINE=tau_logz sbatch --mem=400G examples/kimi_k2/fixed_ref/launch.sh

# master bank
python examples/kimi_k2/fixed_ref/prepare_master_prompts.py --out $MASTER_DIR/prompts
for j in 0 1 2 3; do MODE=master_build JOB_INDEX=$j NUM_JOBS=4 sbatch -J k2master_j$j --mem=400G --time=8:00:00 \
    examples/kimi_k2/fixed_ref/launch.sh; done                                    # resubmit the same command to resume
python examples/kimi_k2/fixed_ref/prepare_master_selection.py --master $MASTER_DIR --out $SEL --train 4096 --k 16
SELECTION=$SEL TAU=0.03 LR=3e-6 sbatch --mem=400G examples/kimi_k2/fixed_ref/launch.sh   # B, n read from $SEL
```

Smoke tests: `smoke_test.sh` (pickled bank) and `smoke_master.sh` (master build split over 2 jobs → selection →
3 training steps).

The current master bank is at `/n/netscratch/dam_lab/Everyone/brachit/kimi_k2_fixed/master_dapo_n64/`
(prompts in `prompts/`, shards in `shards/`). It is world-readable.

### Config (`trainer.fixed_dataset`)

| key | meaning |
|---|---|
| `enable` | use this loop instead of the PPO loop |
| `path` | pickled bank (built if missing) |
| `build_only` | build (pickled bank, or this job's master shards) and exit |
| `anchor` | `ref` (frozen π_ref log-probs) or `current` (recomputed every step) |
| `baseline` | `mean` (group mean, published K2) or `tau_logz` (τ log Ẑ of the group) |
| `holdout_sample_prompts`, `probe_prompts`, `seed` | held-out fresh samples (pickled build), in-sample probe groups |
| `master_dir`, `master_chunk_prompts`, `master_job_index`, `master_num_jobs` | master build |
| `master_selection` | selection.json, so the bank is loaded from the master bank |
| `hf_save_freq`, `hf_save_dense_until` | weights-only snapshots |
| `floor_taus` | τ values for the `bank/floor_tau*` summary |

Other settings that matter:
- `actor.ppo_mini_batch_size` = B gives one full-batch Adam step per step.
- `actor.policy_loss.loss_mode=kimi_k2`, with `kimi_tau`, `kimi_target_level=seq_sum` and `kimi_form=squared`.
- `algorithm.adv_estimator=grpo` with `norm_adv_by_std_in_grpo=False`.
- `data.kimi_holdout_prompts`: pickled build only. It is 0 for master selections.

---

## 5. Row roles and metrics

| rows | `kimi_holdout` | role | metrics prefix |
|---|---|---|---|
| B × n | 0 | training rows | `actor/*`, `kimi/*` |
| `probe_prompts` × n of them | 0, `kimi_probe`=1 | in-sample probe, complete groups, random | `kimi_probe/*` |
| H × n | 1 | fresh prompts, never trained on | `kimi_holdout_prompt/*` |
| S × n | 2 | fresh samples of training prompts (own uids) | `kimi_holdout_sample/*` |

Per-step metrics:
- `<prefix>/{reg_loss,solve_ratio,s,rho,delta_rms,...}_{before,after}`: regression stats on each tracked set.
  With a frozen anchor, *before* at step t+1 equals *after* at step t.
- `<prefix>/{delta_correct_mean,delta_wrong_mean,delta_zeroadv_mean,log_inv_p_correct_mean}`: per-class shifts
  compared with the population optimum's `log u_+ ≈ log(1/p)`.
- `<prefix>/{snis_reward,ref_reward,ess_frac,chi2_hat,frac_groups_ess_below_quarter,frac_delta_below_m10,entropy}`:
  the bank's importance-sampling view of the current policy, per prompt and averaged.
- `actor/loss_exact`: the exact loss of the whole bank at the parameters before the step.
- `bank/*` (step 0): reward and length per set, pass-count histogram, zero-advantage fraction,
  `bank/floor_tau{τ}`, `bank/pstar_tau{τ}`, `bank/log_uplus_mixed_tau{τ}`. Master selections also log
  `bank/entropy_ref_token_mean` and `bank/load_s`. `bank_baseline/*` is logged with `tau_logz`.
- `val-core/<source>/reward/mean@n` and `val-aux/<source>/response_{length,clipped}/mean@n` for AIME and the
  monitors.

Checkpoints:
- `global_step_k/` is a full, resumable checkpoint (`resume_mode=auto`; resubmit the same command). A full
  checkpoint is about 19 GB.
- `hf_step_k/` holds bf16 weights only.

---

## 6. Gotchas

- **Frozen anchor vs the on-policy shortcut.** See §3 step 2. Without `kimi_frozen_anchor`, a single-mini-batch
  update regresses against itself (Δ ≡ 0 in the target).
- **`reduce_metrics` takes min/max for keys containing "min"/"max".** Avoid them in metric names (for example
  "minibatch").
- **Padded pickles are large.** 241 KB/row at 1024+3072 tokens. Use the master bank for anything big, because it
  stores only real tokens.
- **Advantages use the selected samples only.** With a master selection of k=16, the baseline uses those 16
  samples, not all 64, which keeps it comparable with pickled banks.
- **The anchor comes from FSDP, not vLLM.** The anchor is the FSDP forward at θ_ref. vLLM's log-probs are kept
  (`rollout_log_probs`) for mismatch diagnostics only.
- **Cost scales with the selected rows.** Every training step costs ~8 ms per selected row on 4×H200. A full-batch
  step over all 1.15M master rows would take ~2.6 h, so select a subset per run.
