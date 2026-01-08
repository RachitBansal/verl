# Adaptive Rollout for GRPO Training

This document describes the adaptive rollout strategy implemented for GRPO training, designed to maximize learning from hard examples.

## Motivation

In standard GRPO training with a fixed number of rollouts (e.g., n=5), hard examples where the model has low probability of producing correct answers often don't contribute any learning signal - none of the 5 rollouts may be correct, so there's no positive-negative contrast for the GRPO advantage computation.

The adaptive rollout strategy addresses this by:
1. Generating rollouts iteratively until at least one positive (correct) rollout is found
2. Up to a maximum number of rollouts per sample (e.g., 128)
3. Using all rollouts (both positive and negative) for training

This ensures that even hard examples contribute learning signal, potentially accelerating learning on difficult problems.

## Implementation

### Key Components

1. **`verl/trainer/ppo/adaptive_rollout.py`**: Core adaptive rollout generator
2. **`verl/trainer/ppo/adaptive_ray_trainer.py`**: Extended PPO trainer with adaptive rollouts
3. **`verl/trainer/main_ppo_adaptive.py`**: Entry point for adaptive training

### How It Works

```
For each training batch:
    For each sample:
        while (no positive rollout found) AND (rollout_count < max_n):
            Generate `rollouts_per_batch` rollouts
            Compute rewards immediately
            Check if any rollout has reward >= positive_threshold
            If positive found, mark sample as "done"
        
    Assemble all rollouts into training batch
    Compute GRPO advantages (grouped by sample uid)
    Update actor/critic as usual
```

### Configuration

Add these parameters to your training config:

```yaml
adaptive_rollout:
  enable: true                    # Enable adaptive rollouts
  max_n: 128                      # Maximum rollouts per sample
  rollouts_per_batch: 8           # Rollouts generated per iteration
  positive_threshold: 0.5         # Reward threshold for "positive"
  min_rollouts: 2                 # Minimum rollouts (need >=2 for GRPO)
```

Or via command line:

```bash
python3 -m verl.trainer.main_ppo_adaptive \
    +adaptive_rollout.enable=True \
    +adaptive_rollout.max_n=128 \
    +adaptive_rollout.rollouts_per_batch=8 \
    +adaptive_rollout.positive_threshold=0.5 \
    +adaptive_rollout.min_rollouts=2 \
    ...
```

## Running the Experiments

### Adaptive Rollout Experiment

```bash
bash examples/grpo_trainer/run_adaptive_rollout_experiment.sh
```

### Baseline (Fixed n=5)

```bash
bash examples/grpo_trainer/run_baseline_fixed_n5.sh
```

### Compare Both

```bash
bash examples/grpo_trainer/run_adaptive_vs_baseline_comparison.sh both
```

## Metrics

The adaptive rollout trainer logs additional metrics:

| Metric | Description |
|--------|-------------|
| `adaptive_rollout/iterations` | Number of generation iterations used |
| `adaptive_rollout/total_rollouts` | Total rollouts generated in the batch |
| `adaptive_rollout/avg_rollouts_per_sample` | Average rollouts per sample |
| `adaptive_rollout/min_rollouts_per_sample` | Minimum rollouts across samples |
| `adaptive_rollout/max_rollouts_per_sample` | Maximum rollouts across samples |
| `adaptive_rollout/samples_with_positive` | Samples that found a positive rollout |
| `adaptive_rollout/samples_without_positive` | Samples without any positive rollout |
| `adaptive_rollout/positive_rate` | Fraction of samples with positive rollouts |
| `adaptive_rollout/efficiency` | samples / total_rollouts (higher = more efficient) |

## Fair Comparison

For a fair comparison between adaptive and fixed rollout strategies, consider:

1. **Same Total Compute**: Adaptive rollout may use more rollouts on hard examples. Compare by total rollouts or wall-clock time.

2. **Sample Efficiency**: Track validation accuracy vs. total rollouts generated.

3. **Gradient Steps**: If using the same batch size, adaptive rollout produces variable-sized batches. Consider tracking per-gradient-step performance.

## Expected Behavior

- **Easy examples**: Stop after min_rollouts (typically 2) since positive rollouts are found quickly
- **Hard examples**: Generate up to max_n rollouts, ensuring at least some learning signal
- **Overall**: More compute spent on hard examples where the model needs the most learning

## Notes

1. The GRPO advantage computation naturally handles variable group sizes since it groups by sample uid.

2. Rewards are computed during generation (to check for positive rollouts), then recomputed for the full batch. This adds some overhead but ensures consistency.

3. The `rollouts_per_batch` parameter controls the granularity of early stopping. Smaller values mean earlier stopping but more generation iterations.

## Authors

- Sunny + Claude

