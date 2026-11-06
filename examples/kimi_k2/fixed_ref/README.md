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
   k samples**, as for a pickled bank, unless `baseline_file` is set (§2e).

`actor_rollout_ref.rollout.n` must equal the selection's k; `launch.sh` sets it from the selection.

### 2e. Large-N baselines: extension banks + `baseline_file`

A run trains on k samples per prompt, but its baseline b(x) in A = r − b(x) can come from many more samples of the
same prompt. The pooled samples are only scored, never trained on.

1. **Extension bank** (`MODE=master_build` into a new dir, same prompt file). Three settings differ from the base
   build:
   - `master_sample_offset` = the base bank's n, so samples are numbered on from it (`sample_index`).
   - `actor_rollout_ref.rollout.seed` ≠ the base bank's seed (0). vLLM is seeded per engine, so the same prompts
     in the same order with the same seed can reproduce the base bank's samples. The smoke test
     (`smoke_master_ext.sh`) builds a seed-0 control extension to measure this.
   - `master_ref_log_probs=False` skips the FSDP reference pass (~20% of build time). Tokens, vLLM log-probs and
     rewards are stored; the manifest says `has_ref_log_probs: false` and `gather_rows` refuses the bank, so it
     cannot be trained on by mistake.
2. **Pooling** (`compute_master_baselines.py`, CPU, reads only the shard sidecars). It checks that the banks share
   model, prompt file and sampling settings, that sample ranges are disjoint, that seeds differ and that the
   dataset index of every prompt row agrees, and `--check-overlap K` counts identical responses across banks on K
   prompts. It writes `<out>.npz` with, per prompt row: `correct`, `total` (N), `mean` = correct/N, and `tau_logz`
   (T × P) = τ log((N−k)/N + (k/N)·e^{1/τ}) for each τ in `--taus` (default 1, 0.3, 0.1, 0.03, 0.01). That is the
   group formula of `soft_value_baseline` applied to all N samples (rewards are binary, so pass counts suffice).
   `<out>.json` holds the sources and summary statistics.
3. **Training** (`trainer.fixed_dataset.baseline_file=<out>.npz`, `baseline=mean|tau_logz`, launch knobs
   `BASELINE_FILE`, `BASELINE`). `_rebaseline_fixed_bank` looks up b by `master_prompt_row` with
   `kimi_fixed_bank.lookup_master_baseline`: `mean` → the N-sample pass rate; `tau_logz` → the precomputed column
   at the run's `kimi_tau`, which must be one of the file's τ. Every row gets the same b(x) as its prompt's
   training rows, held-out flag-2 rows included, so training and held-out targets are directly comparable.
   `ref_pass_rate` becomes the N-sample pass rate. The baseline includes the k trained samples (b is the mean over
   all N). Step-0 metrics `bank_baseline/*` report N, the targets per role and the fraction of prompts with all
   targets zero.

```bash
M=/n/netscratch/dam_lab/Everyone/brachit/kimi_k2_fixed
for j in 0 1 2 3 4 5 6 7; do MODE=master_build MASTER_DIR=$M/master_dapo_n448_s1 \
    MASTER_PROMPTS=$M/master_dapo_n64/prompts/master_prompts.parquet N_MASTER=448 SAMPLE_OFFSET=64 ROLLOUT_SEED=1 \
    REF_LOGPROBS=False CHUNK=64 JOB_INDEX=$j NUM_JOBS=8 sbatch -J k2ext_j$j --mem=400G --time=12:00:00 \
    examples/kimi_k2/fixed_ref/launch.sh; done
python examples/kimi_k2/fixed_ref/compute_master_baselines.py --master $M/master_dapo_n64 \
    --master $M/master_dapo_n448_s1 --out $M/master_dapo_n64/baselines/n512.npz --check-overlap 64
BASELINE_FILE=$M/master_dapo_n64/baselines/n512.npz BASELINE=tau_logz bash examples/kimi_k2/fixed_ref/launch_full_grid.sh
```

### 2f. Sub-sampled training: `rotate_k` + `max_update_rows`

To measure what sub-sampling costs against full-batch gradient descent, a run can train every step on `rotate_k`
**different** samples per training prompt, drawn from a pool of master samples, with targets r − b(x) from a
`baseline_file` (required: the targets must not depend on the step's sub-sample, so every cell optimizes the same
pool objective and only the gradient noise differs).

- **Pool** (`_setup_rotation`): samples 0..`rotate_pool`−1 (default all n) of every training prompt of the
  selection, minus the prompt's held-out-sample set (selection role `holdout_sample`; with the full selection the
  first 256 training prompts keep samples 16–31 out, so their pool is 48). Loaded once in compact form (gather_rows
  pieces, ~12 GB for 17k × 64), with b(x), the N-sample pass rate and the probe flag per prompt.
- **Schedule** (`kimi_master_bank.rotation_samples`): an epoch is pool/k steps in which every pool sample is used
  exactly once; the order is reshuffled every epoch (rng keyed by seed, epoch, pool size). It is a pure function of
  the step, so a resumed run continues the same schedule. `rotate_k >= pool`: the whole pool every step, i.e.
  full-batch gradient descent on the pool objective.
- **Fixed held-out sets**: the selection's held-out-prompt and held-out-sample rows are loaded once and evaluated
  every step as before. The probe rows are the probe prompts' rows *of this step* (so they rotate).
- **Chunked steps** (`_rotation_chunks`, `_rotation_update`): a step's training rows are split by prompts into
  `update_actor` calls of at most `max_update_rows` rows (the probe prompts and the held-out rows join the last
  call). The calls accumulate gradients into one optimizer step: meta `kimi_accum_scale` (= the call's share of the
  step's rows) multiplies the loss, only the first call zeroes the gradients, only the last one steps the optimizer,
  the lr scheduler and evaluates the tracked rows (dp_actor `update_policy`, fsdp_workers `update_actor`). The
  gradient equals one call over all rows; host memory is bounded by the chunk. `actor/loss_exact` is the sum over
  the calls (the exact loss of the step's rows at θ_t); `rotation/{chunks,rows,epoch,slot}` describe the step.
- `actor/loss_exact` is an unbiased estimate of the pool loss at θ_t for every k (for k = pool it is the pool
  loss), so the training-loss curves of different k are comparable; the held-out sets measure fresh samples.

```bash
# K in {4, 64, 16} x tau in {0.03, 0.1}, lr 1e-5, mean baseline from all 512 samples
bash examples/kimi_k2/fixed_ref/launch_rotate_grid.sh
```

Smoke: `smoke_rotate.sh` (same step with 1 and 3 update calls must match; full-pool run). CPU test:
`test_rotation.py`.

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

Smoke tests: `smoke_test.sh` (pickled bank), `smoke_master.sh` (master build split over 2 jobs → selection →
3 training steps), `smoke_precision.sh` (tf32 anchor + 3 tf32 steps vs 3 bf16 steps on the smoke master bank) and
`smoke_master_ext.sh` (base bank + baseline-only extension + seed-0 control → pooled baselines → tau_logz and mean
runs). CPU tests: `test_master_roundtrip.py`, `test_master_baselines.py`.

```bash
# fp32/TF32 actor (needed for tau >= 0.3, see "Numerical precision" below)
for j in 0 1 2 3; do MODE=master_anchor PRECISION=tf32 JOB_INDEX=$j NUM_JOBS=4 sbatch -J k2anchor_j$j --mem=400G \
    examples/kimi_k2/fixed_ref/launch.sh; done                                    # anchors/tf32/, once per master bank
SELECTION=$SEL PRECISION=tf32 TAU=1 LR=1e-6 sbatch --mem=1400G examples/kimi_k2/fixed_ref/launch.sh
```

The current master bank is at `/n/netscratch/dam_lab/Everyone/brachit/kimi_k2_fixed/master_dapo_n64/`
(prompts in `prompts/`, shards in `shards/`). It is world-readable.

### Config (`trainer.fixed_dataset`)

| key | meaning |
|---|---|
| `enable` | use this loop instead of the PPO loop |
| `path` | pickled bank (built if missing) |
| `build_only` | build (pickled bank, or this job's master shards) and exit |
| `anchor` | `ref` (frozen π_ref log-probs) or `current` (recomputed every step) |
| `baseline` | `mean` (group mean, published K2) or `tau_logz` (τ log Ẑ of the group, or of all N pooled samples with `baseline_file`) |
| `baseline_file` | npz from `compute_master_baselines.py`: b(x) over all pooled samples of the prompt (§2e) |
| `rotate_k`, `rotate_pool` | sub-sampled training: k fresh pool samples per prompt per step (§2f); 0 = off |
| `max_update_rows` | split a step into gradient-accumulating `update_actor` calls of at most this many rows (§2f) |
| `master_sample_offset`, `master_ref_log_probs` | extension builds: first sample number; `false` = no reference pass (§2e) |
| `holdout_sample_prompts`, `probe_prompts`, `seed` | held-out fresh samples (pickled build), in-sample probe groups |
| `master_dir`, `master_chunk_prompts`, `master_job_index`, `master_num_jobs` | master build |
| `master_selection` | selection.json, so the bank is loaded from the master bank |
| `master_anchor` | anchor tag: read `anchors/<tag>/` instead of the stored (bf16) anchor |
| `master_anchor_build` | compute `anchors/<master_anchor>/` with the actor's current precision and exit |
| `hf_save_freq`, `hf_save_dense_until` | weights-only snapshots |
| `floor_taus` | τ values for the `bank/floor_tau*` summary |

Other settings that matter:
- `actor.fsdp_config.dtype` (`bfloat16` | `fp32`) and `actor.fsdp_config.allow_tf32`: the actor's compute
  precision (launch knob `PRECISION=bf16|tf32|fp32`).
- `actor.ppo_mini_batch_size` = B gives one full-batch Adam step per step.
- `actor.policy_loss.loss_mode=kimi_k2`, with `kimi_tau`, `kimi_target_level=seq_sum` and `kimi_form=squared`.
- `algorithm.adv_estimator=grpo` with `norm_adv_by_std_in_grpo=False` (required with `baseline_file`).
- `actor_rollout_ref.rollout.seed`: vLLM engine seed (default 0); launch knob `ROLLOUT_SEED`.
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
- **Advantages use the selected samples only, unless `baseline_file` is set.** With a master selection of k=16
  and no baseline file, the baseline uses those 16 samples, not all 64, which keeps it comparable with pickled
  banks. With a baseline file it uses all pooled samples (§2e).
- **Same seed, same samples.** A second build over the same prompts needs a different `rollout.seed`;
  `compute_master_baselines.py` refuses banks with equal seeds.
- **The anchor comes from FSDP, not vLLM.** The anchor is the FSDP forward at θ_ref. vLLM's log-probs are kept
  (`rollout_log_probs`) for mismatch diagnostics only.
- **Cost scales with the selected rows.** Every training step costs ~8 ms per selected row on 4×H200. A full-batch
  step over all 1.15M master rows would take ~2.6 h, so select a subset per run.

---

## 7. Numerical precision

**Symptom (full-batch grid, 2026-11-03).** After one Adam step the probe's sequence-level Δ rms jumped from 0.02 to
≈0.50 for *every* lr from 1e-7 to 3e-6 (0.77 at 1e-5), and the τ=1 regression loss went from 0.097 to 0.34 at
every lr. The step size had no effect.

**Cause.** bf16 forward noise. A bf16 forward is a chaotic function of the weights: any weight change, even one
that flips 0.1% of bf16 parameters by one ulp, re-rounds every activation and moves each token's log-prob by
≈0.018 nats, i.e. Σ_t log π by ≈0.5 nats rms over ~800 tokens. Before the first step, the probe and the anchor run
the same weights through the same kernels, so the gap is only 0.02; after any step it is the full noise. The true
change from a 1e-7 step is ≈2e-4 nats. Measured with `diag/delta_probe.py` (seq-sum Δ rms vs the unperturbed
model, 30 bank rows, random-sign perturbation of size lr):

| actor compute | lr 1e-7 | lr 1e-6 | lr 1e-5 | forward cost |
|---|---|---|---|---|
| bf16 (verl default) | 0.56 | 0.63 | 0.58 | 1× |
| fp32 matmuls, bf16 flash attention | 0.08 | 0.09 | 0.08–0.11 | 1.6× |
| TF32 matmuls, fp16 flash attention (`PRECISION=tf32`) | 0.019 | 0.024 | 0.028 | 1.6× |
| exact fp32 (sdpa) | 0.0002 | 0.0019 | 0.019 | 8× |

The noise enters the loss as τΔ_noise. It adds ≈τ²·0.25 to the regression loss (L0 ≈ 0.1): about 250% at τ=1,
22% at τ=0.3, 2.5% at τ=0.1 and under 0.25% at τ ≤ 0.03. The stored anchor (bf16 FSDP) itself differs from fp32
by the same ≈0.5 nats.

**Fix (`PRECISION=tf32`).**
- The actor uses fp32 params and compute (`fsdp_config.dtype=fp32`) with TF32 tensor-core matmuls
  (`fsdp_config.allow_tf32=True`, set in `fsdp_workers._build_model_optimizer`).
- Flash attention only takes fp16/bf16, so `monkey_patch._ulysses_flash_attention_forward` runs the kernel in fp16
  and returns fp32.
- fp16's range would flush the ≈1e-7 gradients of a full-batch mean loss. The gradient crossing the kernel is
  therefore scaled by 2^16 (`_FP16_ATTN_GRAD_SCALE`). In `diag/grad_check.py` the gradient cosine to exact fp32 is
  0.9999 with the scale and 0.27 without it.
- The anchor must come from the same precision. `MODE=master_anchor` recomputes it once per master bank into
  `anchors/<tag>/shard_XXXXX.pt` (`_build_master_anchor`; flat fp32 in the shard's response layout). A run with
  `master_anchor=<tag>` reads it (`gather_rows(..., anchor=tag)`).

Smoke at τ=1, lr 1e-7 (`smoke_precision.sh`, 12 prompts × 2 samples):

| | probe Δ rms, steps 1–3 | probe loss, start → step 3 |
|---|---|---|
| tf32 | 0.04, 0.04, 0.07 | 0.124 → 0.074 |
| bf16 | 0.34, 0.52, 0.25 | 0.163 → 0.43, 0.56, 0.09 (noise) |

Earlier results this affects:
- **τ ≥ 0.3 runs.** Any bf16 run at τ ≥ 0.3 measured mostly noise in its regression metrics.
- **Held-out ratios.** The ≈1.1 held-out ratio "from step 1" at τ=0.3 is this effect.
- **τ ≤ 0.1 conclusions** stand, apart from a ≤2.5% loss offset.
- **On-policy K2 runs** (bf16 `old_log_probs` recomputed every step) get the same ≈0.5-nat noise in Δ from their
  second mini-batch step on, because the first step starts from the same weights as the anchor.

