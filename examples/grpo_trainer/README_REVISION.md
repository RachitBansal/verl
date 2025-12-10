# Using Model Revisions in VERL

This guide explains how to use specific model checkpoints/revisions from Hugging Face Hub when training with VERL.

## Overview

The `revision` parameter allows you to load specific branches, tags, or intermediate checkpoints from a Hugging Face model repository. This is particularly useful for:

- Loading intermediate pretraining checkpoints (e.g., OLMo-2 stage checkpoints)
- Using specific model versions or snapshots
- Experimenting with different training stages of a model

## Configuration

To use a specific revision, add the `actor_rollout_ref.model.revision` parameter to your training command:

```bash
python3 -m verl.trainer.main_ppo \
    actor_rollout_ref.model.path=allenai/OLMo-2-0425-1B \
    actor_rollout_ref.model.revision=stage1-step1000-tokens4B \
    # ... other parameters
```

## OLMo-2 Example

OLMo-2 provides intermediate checkpoints during pretraining. According to the [OLMo-2 model card](https://huggingface.co/allenai/OLMo-2-0425-1B):

### Available Checkpoints

For pretraining checkpoints, the naming convention is:
- `stage1-stepXXX-tokensYYYB` - Checkpoints from Stage 1 (initial pretraining)
- `stage2-ingredientN-stepXXX-tokensYYYB` - Checkpoints from Stage 2 (mid-training)

### Example Script

See `run_olmo2-1b_with_revision.sh` for a complete example:

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=allenai/OLMo-2-0425-1B \
    actor_rollout_ref.model.revision=stage1-step1000-tokens4B \
    # ... other configuration
```

### Finding Available Revisions

You can find available revisions for any model on Hugging Face:

1. Go to the model page (e.g., https://huggingface.co/allenai/OLMo-2-0425-1B)
2. Click on "Files and versions" tab
3. Look at the dropdown menu at the top to see available branches/tags
4. Browse the file tree to see checkpoint directories

Alternatively, use the Hugging Face Hub API:

```python
from huggingface_hub import list_repo_refs

refs = list_repo_refs("allenai/OLMo-2-0425-1B")
print("Branches:", [ref.name for ref in refs.branches])
print("Tags:", [ref.name for ref in refs.tags])
```

## Technical Details

The `revision` parameter is passed to all `from_pretrained` calls in VERL:

1. **FSDP Models** (`verl/workers/engine/fsdp/transformer_impl.py`)
2. **Megatron Models** (`verl/utils/model.py`)
3. **Critic Models** with value heads (`verl/workers/fsdp_workers.py`)

The parameter is added to the `HFModelConfig` dataclass and automatically propagates through the loading pipeline.

## Notes

- The `revision` parameter is optional - if not specified, the default branch (usually `main`) is used
- When using `use_shm=True`, the model will be downloaded locally first, then the revision is applied
- The revision must exist in the repository, otherwise you'll get a download error
- For local model paths (not from HF Hub), the revision parameter is ignored

## Other Model Examples

This feature works with any Hugging Face model that has multiple revisions:

### Using a specific commit hash
```bash
actor_rollout_ref.model.path=meta-llama/Llama-3-8B \
actor_rollout_ref.model.revision=a67fc8d27f8b2c24bb3ec983b02ed07d8b20db8d
```

### Using a branch
```bash
actor_rollout_ref.model.path=some-org/some-model \
actor_rollout_ref.model.revision=experimental-branch
```

### Using a tag
```bash
actor_rollout_ref.model.path=some-org/some-model \
actor_rollout_ref.model.revision=v1.0.0
```

## Configuration File Usage

You can also set this in a YAML configuration file:

```yaml
actor_rollout_ref:
  model:
    path: allenai/OLMo-2-0425-1B
    revision: stage1-step1000-tokens4B
    # ... other model config
```

## Troubleshooting

**Error: Revision not found**
- Check that the revision name is spelled correctly
- Verify the revision exists in the repository
- Ensure you have access to the repository (for private models)

**Model downloads the wrong version**
- Clear your HuggingFace cache: `~/.cache/huggingface/hub/`
- Set `use_shm=False` to avoid caching issues

**Revision parameter is ignored**
- The revision parameter only works with Hugging Face Hub models
- For local paths, the parameter is silently ignored


