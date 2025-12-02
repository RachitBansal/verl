# GRPO Training on OpenMathInstruct-1

This directory contains scripts for training OLMo2 models using GRPO (Group Relative Policy Optimization) on the [OpenMathInstruct-1 dataset](https://huggingface.co/datasets/nvidia/OpenMathInstruct-1).

## Overview

**OpenMathInstruct-1** is a high-quality math instruction tuning dataset from NVIDIA containing 1.8M problem-solution pairs. The dataset includes:
- Problems from GSM8K and MATH training sets
- Solutions using a mix of text reasoning and executable Python code
- Synthetically generated using the Mixtral-8x7B model
- Only correct solutions are used for training

**GRPO** is an efficient RL algorithm that eliminates the need for a separate critic model by:
- Generating multiple solutions per problem (group sampling)
- Using group-level reward normalization as baseline
- Direct KL regularization in the loss function

## Prerequisites

1. **Install dependencies**:
   ```bash
   # Make sure you have the datasets library
   pip install datasets
   ```

2. **Prepare the dataset**:
   ```bash
   # Download and prepare OpenMathInstruct-1 (full dataset)
   python3 examples/grpo_trainer/prepare_openmath_data.py \
       --output_dir /n/netscratch/dam_lab/Lab/sqin/tmp_data/openmath

   # For quick testing with a subset (1000 samples)
   python3 examples/grpo_trainer/prepare_openmath_data.py \
       --output_dir /n/netscratch/dam_lab/Lab/sqin/tmp_data/openmath \
       --test
   ```

   The script will:
   - Download the OpenMathInstruct-1 dataset from HuggingFace
   - Filter to keep only correct solutions
   - Transform to the expected format (with 'prompt' field)
   - Save as parquet files (train.parquet and validation.parquet)

3. **Verify your model checkpoint**:
   Make sure your custom OLMo2 checkpoint exists:
   ```bash
   ls -la /n/netscratch/dam_lab/Lab/sqin/olmo/checkpoints/OLMo2-1B-stage1
   ```

## Training Scripts

### 1. SLURM Training (Recommended for cluster)

```bash
# Submit SLURM job
sbatch examples/grpo_trainer/run_olmo2-1b_openmath_slurm.sh

# Monitor logs
tail -f logs/slurm-<job_id>.out
```

**Configuration**: 
- 4 GPUs per node
- 48 hours wall time
- 200GB memory
- Appropriate for the full 1.8M sample dataset

### 2. Interactive Training (For testing)

```bash
# Run directly (requires 4 GPUs)
bash examples/grpo_trainer/run_olmo2-1b_openmath.sh
```

## Key Training Parameters

### Dataset Parameters
- `data.max_prompt_length=1024`: Max tokens for math problems
- `data.max_response_length=2048`: Longer to accommodate code blocks and reasoning
- `data.train_batch_size=512`: Global batch size for prompt sampling

### GRPO Parameters
- `algorithm.adv_estimator=grpo`: Enable GRPO algorithm
- `actor_rollout_ref.rollout.n=5`: Generate 5 solutions per problem (group sampling)
- `actor_rollout_ref.actor.use_kl_loss=True`: Direct KL regularization
- `actor_rollout_ref.actor.kl_loss_coef=0.001`: KL coefficient
- `actor_rollout_ref.actor.kl_loss_type=low_var_kl`: Low-variance KL estimator

### Training Parameters
- `actor_rollout_ref.actor.optim.lr=1e-6`: Learning rate
- `trainer.total_epochs=10`: Number of training epochs
- `trainer.save_freq=20`: Save checkpoint every 20 steps
- `trainer.test_freq=5`: Evaluate every 5 steps

## Dataset Statistics

The prepared OpenMathInstruct-1 dataset contains:
- **Train split**: ~4.95M samples (filtered to correct solutions)
- **Validation split**: ~1.13M samples (filtered to correct solutions)
- **Problem sources**: GSM8K and MATH datasets
- **Solution format**: Mix of natural language reasoning and Python code blocks

Example entry:
```json
{
  "prompt": "Martha has 18 crayons. She lost half of them...",
  "reference_solution": "Let's solve this problem using Python code...",
  "expected_answer": "29",
  "dataset_source": "gsm8k"
}
```

## Customization

### Using a different checkpoint

Edit the script and change:
```bash
OLMO_CHECKPOINT="/path/to/your/checkpoint"
```

### Adjusting batch sizes

If you encounter OOM errors, reduce:
- `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu`
- `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu`
- `data.train_batch_size`

### Using more/fewer GPUs

Adjust in the script:
```bash
trainer.n_gpus_per_node=<number>
```

And in SLURM header:
```bash
#SBATCH --gpus-per-node=<number>
```

## Weights & Biases Integration

To enable W&B logging, uncomment in the script:
```bash
export WANDB_API_KEY="your_key_here"
```

Or for offline logging:
```bash
export WANDB_MODE=offline
```

## Expected Training Time

On 4x A100 GPUs with full dataset:
- **Per epoch**: ~8-12 hours (depending on hardware)
- **Full training (10 epochs)**: ~4-5 days

For faster iteration during development, use the `--test` flag when preparing data to work with a smaller subset.

## References

- **Dataset Paper**: [OpenMathInstruct-1: A 1.8 Million Math Instruction Tuning Dataset](https://arxiv.org/abs/2402.10176)
- **GRPO Paper**: [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
- **Dataset**: https://huggingface.co/datasets/nvidia/OpenMathInstruct-1

## Troubleshooting

### Dataset Download Issues
If the dataset download is slow or fails:
```bash
# Set HuggingFace cache directory with more space
export HF_HOME=/path/to/large/storage/.cache/huggingface
python3 examples/grpo_trainer/prepare_openmath_data.py
```

### CUDA OOM Errors
Reduce batch sizes or enable offloading:
```bash
actor_rollout_ref.actor.fsdp_config.param_offload=True \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
```

### Flash Attention Issues
The scripts disable flash attention by default:
```bash
export DISABLE_FLASH_ATTN=1
export USE_EAGER_ATTN=0
```

If you have flash attention properly installed, you can remove these flags for better performance.

