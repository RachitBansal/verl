# Critical Batch Size in RLVR

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

We study CBS along each axis independently:

1. **Prompt CBS**: Fix n_r, measure S(n_p) as n_p varies.
   Captures variance from sampling different problems.

2. **Rollout CBS**: Fix n_p, measure S(n_r) as n_r varies.
   Captures variance from sampling different solutions to the same problem.
   Note: GRPO normalizes advantages within each prompt's rollout group,
   so n_r also affects the quality of the advantage estimate.

3. **Iso-batch decomposition**: Fix B_total = n_p · n_r, vary the split.
   Answers: given a fixed sample budget per step, what is the optimal
   allocation between prompt diversity and rollout diversity?

### Gradient Noise Scale in RLVR

The gradient noise scale can be estimated from mini-batch gradient variance:

    B_noise ≈ tr(Σ) / ||g||²

where Σ is the covariance of per-sample gradients and g is the full-batch
gradient. In practice, we estimate this from K mini-batches of size b:

    B_noise ≈ (b / (K-1)) · Σ_k ||g_k - ḡ||² / ||ḡ||²

For RLVR, we can decompose this into:
- **Prompt noise**: variance of gradients across different prompts
- **Rollout noise**: variance of gradients across rollouts for the same prompt

### Target Metrics

We use verifiable reward accuracy as the target metric (not loss), since
RLVR optimizes for correct solutions:

- GSM8K accuracy (greedy decoding)
- MATH accuracy (greedy decoding)

Target thresholds are chosen based on the model's initial and achievable
performance. For OLMo-2-0425-1B-SFT on GSM8K, we expect meaningful
improvement from ~60% to ~70%+ during GRPO training.

### Experimental Controls

To isolate the effect of batch size, we hold constant:
- Learning rate (1e-6)
- KL penalty coefficient (0.001)
- Number of PPO epochs per batch (1)
- Response length limit (2048)
- Prompt length limit (1024)
- Temperature (1.0)
- Model initialization (same checkpoint)
- Random seed structure (same seed for data sampling)

The mini-batch size for gradient accumulation is set to
min(ppo_mini_batch_size, train_batch_size * n_rollouts) to ensure
consistent gradient computation across different batch sizes.

## Directory Structure

```
experiments/critical_batch_size/
├── README.md                    # This file
├── scripts/
│   ├── cbs_sweep.sh             # Parameterized training script
│   ├── cbs_phase1.sbatch        # Phase 1: vary n_prompts, fixed n_rollouts
│   ├── cbs_phase2a.sbatch       # Phase 2A: vary n_prompts
│   ├── cbs_phase2b.sbatch       # Phase 2B: vary n_rollouts
│   ├── cbs_phase2c.sbatch       # Phase 2C: iso-batch decomposition
│   └── cbs_phase4.sbatch        # Phase 4: scale-up experiments
├── notebooks/
│   └── critical_batch_size.ipynb # Analysis notebook
└── configs/                     # Experiment config overrides (if needed)
```

## Experiment Naming Convention

Wandb experiment names follow the pattern:

    cbs_{phase}_{model}_{dataset}_np{n_prompts}_nr{n_rollouts}

Example: `cbs_p1_olmo1b_omi_np128_nr8`
