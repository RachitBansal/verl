# Critical Batch Size in RLVR

## Research Questions

This project aims to define, measure, and explain an RLVR analogue of critical batch size
for GRPO-style training of LLMs. The experiments below are organized around these questions:

| # | Question | Phase(s) |
|---|----------|----------|
| Q1 | Does RLVR exhibit a CBS analogous to supervised learning? | 1 |
| Q2 | Under strictly on-policy GRPO (ep=1), how does performance scale with prompts/batch? | 1 |
| Q3 | At what batch size do step, compute, and wall-clock efficiency saturate? | 1, 2B |
| Q4 | How does sample reuse (ppo_epochs > 1) change the observed critical batch? | 3 |
| Q5 | Is "diminishing returns" driven by gradient variance or ratio drift/clipping? | 3 |
| Q6 | Fixed token budget: more prompts or more optimization? | 2C, 3 |
| Q7 | Fixed rollouts/prompt: scale batch size or increase epochs? | 3 |
| Q8 | Predictable on-policy vs off-policy tradeoff per task? | 3 |
| Q9 | Can gradient/advantage variance proxies predict CBS onset? | 1, 2B, 3 |
| Q10 | How do reward sparsity and task properties shape CBS? | 4 |
| Q11 | True policy improvement vs improved sampling/selection? | (future) |

---

## Measurement Framework

### Definitions

In supervised pre-training, the critical batch size (CBS) is operationalized via
two quantities measured at a fixed target loss L*:

- **S(B)**: number of gradient steps to reach L* when using batch size B
- **E(B) = B · S(B)**: total examples processed to reach L*

McCandlish et al. (2018) showed these follow a simple model parameterized by
the gradient noise scale B_noise:

    S(B) = S_min · (1 + B_noise / B)
    E(B) = E_min · (1 + B / B_noise)

where S_min is the minimum steps (infinite batch) and E_min is the minimum
examples (batch size 1). The critical batch size B_crit ≈ B_noise is where
S(B) ≈ 2·S_min, i.e., doubling the batch only halves the remaining speedup.

### RLVR Adaptation

In RLVR (e.g., GRPO), the "batch" has two independent axes:

    B_total = n_prompts × n_rollouts

where:
- **n_prompts**: number of unique prompts sampled per step
- **n_rollouts**: number of rollout completions generated per prompt

This gives us a 2D generalization:

    S(n_p, n_r) = steps to reach target accuracy
    E(n_p, n_r) = n_p · n_r · S(n_p, n_r)

We study CBS along each axis independently and across them:

1. **Prompt CBS** (Phase 1): Fix n_r, measure S(n_p) as n_p varies.
   Captures variance from sampling different problems.

2. **Rollout CBS** (Phase 2B): Fix n_p, measure S(n_r) as n_r varies.
   Captures variance from sampling different solutions to the same problem.
   Note: GRPO normalizes advantages within each prompt's rollout group,
   so n_r also affects the quality of the advantage estimate.

3. **Iso-batch decomposition** (Phase 2C): Fix B_total = n_p · n_r, vary the split.
   Answers: given a fixed sample budget per step, what is the optimal
   allocation between prompt diversity and rollout diversity?

### Two Regimes of Diminishing Returns

**Statistical CBS** (ppo_epochs = 1, on-policy):
- Larger batch → lower gradient variance → faster convergence in steps
- Follows the McCandlish model; B_crit ≈ B_noise
- Phases 1, 2B, 2C

**Algorithmic CBS** (ppo_epochs > 1, off-policy within-batch):
- Reusing the same rollout batch for multiple PPO updates causes ratio drift
- After k epochs, many tokens are clipped (π_new / π_old >> 1 + ε)
- Effective gradient signal per token consumed falls regardless of batch size
- The "critical epochs" at a given batch: ppo_kl and pg_clipfrac spike
- Phase 3

### Iso-Optimization Analysis

For a fixed **generation budget** (tokens produced by rollouts), we can compare:

    n_prompts × ppo_epochs = constant  (at fixed n_rollouts)

e.g., at constant = 512:

| n_prompts | ppo_epochs | character |
|-----------|-----------|-----------|
| 512       | 1         | wide batch, on-policy   |
| 256       | 2         | medium, mild off-policy |
| 128       | 4         | narrow, more off-policy |
| 64        | 8         | tiny batch, heavily off |

This answers Q6/Q7: for a fixed computational budget, is breadth (more prompts)
or depth (more gradient steps on the same data) more efficient?

---

## Gradient Noise Scale in RLVR

The gradient noise scale can be estimated from mini-batch gradient variance:

    B_noise ≈ tr(Σ) / ||g||²

where Σ is the covariance of per-sample gradients and g is the full-batch
gradient. In practice, we estimate this from K mini-batches of size b:

    B_noise ≈ (b / (K-1)) · Σ_k ||g_k - ḡ||² / ||ḡ||²

For RLVR, we can decompose this into:
- **Prompt noise**: variance of gradients across different prompts
- **Rollout noise**: variance of gradients across rollouts for the same prompt

The ratio drift metric `pg_clipfrac` (already logged by verl) serves as
the primary proxy for the algorithmic limit.

---

## Target Metrics

We use verifiable reward accuracy as the target metric (not loss), since
RLVR optimizes for correct solutions:

- GSM8K accuracy (greedy decoding)
- MATH accuracy (greedy decoding)

Target thresholds are chosen based on the model's initial and achievable
performance. For OLMo-2-0425-1B-SFT on GSM8K, we expect meaningful
improvement from ~60% to ~70%+ during GRPO training.

---

## Experimental Controls

To isolate the effect of batch size, we hold constant:
- Learning rate (1e-6)
- KL penalty coefficient (0.001)
- Response length limit (2048)
- Prompt length limit (1024)
- Temperature (1.0)
- Model initialization (same checkpoint)
- Random seed structure (same seed for data sampling)

The mini-batch size for gradient accumulation is set to
min(n_prompts, 256) to ensure consistent gradient estimation
across different batch sizes.

---

## Experiment Phases

### Phase 1 — Statistical CBS: Prompt Axis  (Q1, Q2, Q3)
**Status**: data collected

Fix: n_rollouts=8, ppo_epochs=1
Vary: n_prompts ∈ {32, 64, 128, 256, 512, 1024}

Answers: Does RLVR have a CBS? Where does step/compute efficiency saturate
         as we increase prompt diversity?

Script: `cbs_phase2a.sbatch` (array 0-5)

---

### Phase 2B — Statistical CBS: Rollout Axis  (Q3)
**Status**: designed

Fix: n_prompts=128, ppo_epochs=1
Vary: n_rollouts ∈ {1, 2, 4, 8, 16, 32, 64}

Answers: How does rollout diversity (n_r axis) affect the CBS?
         What is the optimal n_rollouts for a fixed n_prompts?

Script: `cbs_phase2b.sbatch` (array 0-6)

---

### Phase 2C — Budget Allocation: Iso-batch  (Q6, Q7)
**Status**: designed

Fix: B_total = n_prompts × n_rollouts = 2048, ppo_epochs=1
Vary: decompositions (2048,1), (1024,2), (512,4), (256,8), (128,16), (64,32)

Answers: Given a fixed generation budget per step, is it better to sample
         more diverse prompts or more rollouts per prompt?

Script: `cbs_phase2c.sbatch` (array 0-5)

---

### Phase 3 — Algorithmic CBS: Multi-epoch  (Q4, Q5, Q6, Q7, Q8)
**Status**: new

Fix: n_rollouts=8
Vary: n_prompts ∈ {32, 64, 128, 256, 512} × ppo_epochs ∈ {2, 4, 8}

Uses Phase 1 ep=1 data as the on-policy baseline.

Key iso-optimization diagonals (n_prompts × ppo_epochs = const):
- 256 units: (256,ep=1)* ↔ (128,ep=2) ↔ (64,ep=4)
- 512 units: (512,ep=1)* ↔ (256,ep=2) ↔ (128,ep=4) ↔ (64,ep=8)
- 1024 units: (1024,ep=1)* ↔ (512,ep=2) ↔ (256,ep=4) ↔ (128,ep=8)
  (* = Phase 1 baselines)

Answers: How does sample reuse shift the CBS? When is off-policy reuse
         beneficial vs harmful? Is clipping or KL pressure the binding constraint?

Key wandb metrics:
- `actor/pg_clipfrac`: ratio clip fraction — the primary algorithmic limit signal
- `actor/ppo_kl`: per-mini-batch KL — rises sharply once policy drifts

Script: `cbs_phase3.sbatch` (array 0-11)

---

### Phase 4 — Task and Algorithm Ablations  (Q10)
**Status**: designed

Block 1 (0-5):   OLMo2-1B on MATH dataset (harder, sparser rewards)
Block 2 (6-11):  REINFORCE++ advantage estimator instead of GRPO
Block 3 (12-17): LR sensitivity at {5e-7, 1e-6, 5e-6} × {128, 512} prompts

Script: `cbs_phase4.sbatch` (array 0-17)

---

## Key Wandb Metrics Reference

| Metric | Meaning | Most relevant to |
|--------|---------|-----------------|
| `val/gsm8k/acc` | GSM8K greedy accuracy | S(B) computation |
| `val/math/acc` | MATH greedy accuracy | S(B) computation |
| `actor/pg_clipfrac` | Fraction of tokens where IS ratio is clipped | Q5, algorithmic CBS |
| `actor/ppo_kl` | Approx KL(new ∥ old) per mini-batch | Q5, ratio drift |
| `actor/grad_norm` | Gradient norm (per mini-batch) | Q9, B_noise estimation |
| `grad_noise/B_noise_estimate` | Estimated gradient noise scale | Q9 |
| `grad_noise/B_noise_approx` | Approximate B_noise from logged norms | Q9 |
| `critic/rewards/mean` | Mean verifiable reward | training health |
| `actor/entropy` | Policy entropy | training health |

---

## Experiment Naming Convention

Wandb experiment names follow the pattern:

    cbs_{phase}_np{n_prompts}_nr{n_rollouts}[_ep{ppo_epochs}]_lr{lr}_mbs{mbs}_cr{clip_ratio}

The `_ep{N}` suffix is added only when `ppo_epochs > 1` (backward-compatible).

Examples:
- `cbs_p1_np128_nr8_lr1e-6_mbs128_cr0.2`         (Phase 1, ep=1 implicit)
- `cbs_p3_np128_nr8_ep4_lr1e-6_mbs128_cr0.2`     (Phase 3, ep=4)

---

## Directory Structure

```
experiments/critical_batch_size/
├── README.md                     # This file
├── grad_noise.py                 # Gradient noise scale estimation utilities
├── scripts/
│   ├── cbs_sweep.sh              # Parameterized training script (all phases)
│   ├── cbs_phase1.sbatch         # Phase 1 single-run (legacy)
│   ├── cbs_phase1b.sbatch        # Phase 1b: clip_ratio ablation
│   ├── cbs_phase2a.sbatch        # Phase 1/2A: vary n_prompts sweep
│   ├── cbs_phase2b.sbatch        # Phase 2B: vary n_rollouts
│   ├── cbs_phase2c.sbatch        # Phase 2C: iso-batch decomposition
│   ├── cbs_phase3.sbatch         # Phase 3: multi-epoch / algorithmic CBS  [NEW]
│   └── cbs_phase4.sbatch         # Phase 4: task/algorithm ablations
└── notebooks/
    └── critical_batch_size.ipynb  # Analysis notebook
```
