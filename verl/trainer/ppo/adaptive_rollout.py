# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 Adaptive Rollout Extensions
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Adaptive Rollout Module for GRPO Training.

This module implements an adaptive rollout strategy that generates rollouts
until at least one positive (correct) rollout is found per sample, up to a
maximum number of rollouts. This helps maximize learning signal from hard
examples where the probability of correct answers is low.

Key features:
- Variable number of rollouts per sample (up to max_n)
- Early stopping when positive rollout is found
- Batch-efficient generation (generates rollouts_per_batch at a time)
- Compatible with existing GRPO advantage computation

Usage:
    # In your training script or config:
    adaptive_rollout:
        enable: true
        max_n: 128              # Maximum rollouts per sample
        rollouts_per_batch: 8   # Rollouts generated per iteration
        positive_threshold: 0.5 # Reward threshold for "positive"
        min_rollouts: 2         # Minimum rollouts per sample for GRPO
"""

from collections import defaultdict
from typing import Any, Optional

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.reward import compute_reward
from verl.workers.reward_manager.abstract import AbstractRewardManager


class AdaptiveRolloutGenerator:
    """
    Generates rollouts adaptively until positive examples are found.

    Strategy:
    1. Generate `rollouts_per_batch` rollouts for all active samples
    2. Compute rewards immediately to identify positive rollouts
    3. Mark samples as "done" when they have at least one positive rollout
    4. Continue for samples without positive rollouts until max_n is reached
    5. Assemble final batch with variable rollouts per sample
    """

    def __init__(
        self,
        max_n: int = 128,
        rollouts_per_batch: int = 8,
        positive_threshold: float = 0.5,
        min_rollouts: int = 2,
    ):
        """
        Initialize the adaptive rollout generator.

        Args:
            max_n: Maximum number of rollouts per sample
            rollouts_per_batch: Number of rollouts to generate in each batch iteration
            positive_threshold: Reward threshold above which a rollout is "positive"
            min_rollouts: Minimum rollouts per sample (need >=2 for GRPO baseline)
        """
        self.max_n = max_n
        self.rollouts_per_batch = rollouts_per_batch
        self.positive_threshold = positive_threshold
        self.min_rollouts = max(min_rollouts, 2)  # Need at least 2 for GRPO baseline

    def generate(
        self,
        gen_batch: DataProto,
        batch: DataProto,
        actor_rollout_wg,
        reward_fn: AbstractRewardManager,
        async_rollout_mode: bool = False,
        async_rollout_manager=None,
        use_rm: bool = False,
        rm_wg=None,
        tokenizer=None,
    ) -> tuple[DataProto, DataProto, dict[str, Any]]:
        """
        Generate rollouts adaptively until positive rollouts are found.

        Args:
            gen_batch: Generation batch (prompts only, no rollout duplication)
            batch: Full batch with reward model info (ground truth, etc.)
            actor_rollout_wg: Actor/rollout worker group for generation
            reward_fn: Reward function to evaluate rollouts
            async_rollout_mode: Whether using async rollout
            async_rollout_manager: Async rollout manager (if async)
            use_rm: Whether using a reward model
            rm_wg: Reward model worker group
            tokenizer: Tokenizer for decoding (optional, for logging)

        Returns:
            (final_batch, gen_batch_output, metrics)
            - final_batch: Batch with original data repeated per rollout
            - gen_batch_output: Generated responses
            - metrics: Adaptive rollout statistics
        """
        num_samples = len(gen_batch)
        sample_uids = batch.non_tensor_batch["uid"].copy()

        # State tracking
        uid_to_original_idx = {uid: i for i, uid in enumerate(sample_uids)}
        sample_has_positive = {uid: False for uid in sample_uids}
        sample_rollout_count = {uid: 0 for uid in sample_uids}

        # Storage for all rollouts
        # Each entry: {uid, gen_output_item, batch_item, reward, iteration}
        all_rollout_data: dict[str, list[dict]] = {uid: [] for uid in sample_uids}

        iteration = 0
        max_iterations = (self.max_n + self.rollouts_per_batch - 1) // self.rollouts_per_batch

        while iteration < max_iterations:
            # Find samples that still need more rollouts
            active_uids = []
            for uid in sample_uids:
                needs_more = (
                    (not sample_has_positive[uid]) or
                    (sample_rollout_count[uid] < self.min_rollouts)
                )
                can_generate_more = sample_rollout_count[uid] < self.max_n
                if needs_more and can_generate_more:
                    active_uids.append(uid)

            if not active_uids:
                break

            # Get indices of active samples
            active_indices = [uid_to_original_idx[uid] for uid in active_uids]

            # How many rollouts to generate this iteration
            n_this_batch = min(
                self.rollouts_per_batch,
                self.max_n - min(sample_rollout_count[uid] for uid in active_uids)
            )

            # Select and repeat active samples
            active_gen_batch = gen_batch.select_idxs(active_indices)
            active_batch = batch.select_idxs(active_indices)

            gen_batch_repeated = active_gen_batch.repeat(
                repeat_times=n_this_batch, interleave=True
            )

            # Generate rollouts
            if async_rollout_mode:
                gen_output = async_rollout_manager.generate_sequences(gen_batch_repeated)
            else:
                gen_output = actor_rollout_wg.generate_sequences(gen_batch_repeated)

            # Prepare batch for reward computation
            batch_repeated = active_batch.repeat(repeat_times=n_this_batch, interleave=True)
            reward_batch = batch_repeated.union(gen_output)

            # Compute rewards
            if use_rm and rm_wg is not None and "rm_scores" not in reward_batch.batch.keys():
                rm_scores = rm_wg.compute_rm_score(reward_batch)
                reward_batch = reward_batch.union(rm_scores)

            reward_tensor, _ = compute_reward(reward_batch, reward_fn)
            rewards = reward_tensor.sum(dim=-1).cpu().numpy()  # (num_active * n_this_batch,)

            # Process results
            for local_idx, uid in enumerate(active_uids):
                for rollout_j in range(n_this_batch):
                    global_idx = local_idx * n_this_batch + rollout_j
                    reward = float(rewards[global_idx])

                    # Extract single rollout items
                    gen_item = gen_output[global_idx: global_idx + 1]
                    batch_item = batch_repeated[global_idx: global_idx + 1]

                    all_rollout_data[uid].append({
                        "gen_output": gen_item,
                        "batch_item": batch_item,
                        "reward": reward,
                        "iteration": iteration,
                        "rollout_idx": sample_rollout_count[uid],
                    })

                    sample_rollout_count[uid] += 1

                    if reward >= self.positive_threshold:
                        sample_has_positive[uid] = True

            iteration += 1

        # Assemble final batch with all rollouts
        final_batch, gen_batch_output = self._assemble_final_batch(
            all_rollout_data, sample_uids
        )

        # Compute metrics
        rollout_counts = list(sample_rollout_count.values())
        total_rollouts = sum(rollout_counts)
        num_with_positive = sum(1 for v in sample_has_positive.values() if v)

        metrics = {
            "adaptive_rollout/iterations": iteration,
            "adaptive_rollout/total_rollouts": total_rollouts,
            "adaptive_rollout/avg_rollouts_per_sample": float(np.mean(rollout_counts)),
            "adaptive_rollout/min_rollouts_per_sample": min(rollout_counts),
            "adaptive_rollout/max_rollouts_per_sample": max(rollout_counts),
            "adaptive_rollout/std_rollouts_per_sample": float(np.std(rollout_counts)),
            "adaptive_rollout/samples_with_positive": num_with_positive,
            "adaptive_rollout/samples_without_positive": num_samples - num_with_positive,
            "adaptive_rollout/positive_rate": num_with_positive / num_samples,
            "adaptive_rollout/efficiency": num_samples / total_rollouts,  # Higher is better
        }

        return final_batch, gen_batch_output, metrics

    def _assemble_final_batch(
        self,
        all_rollout_data: dict[str, list[dict]],
        sample_uids: np.ndarray,
    ) -> tuple[DataProto, DataProto]:
        """
        Assemble all rollouts into final batch format.

        Maintains uid information for GRPO grouping.
        """
        batch_items = []
        gen_items = []
        final_uids = []

        for uid in sample_uids:
            rollouts = all_rollout_data[uid]
            # Sort by generation order
            rollouts.sort(key=lambda x: (x["iteration"], x["rollout_idx"]))

            for rollout in rollouts:
                batch_items.append(rollout["batch_item"])
                gen_items.append(rollout["gen_output"])
                final_uids.append(uid)

        # Concatenate all
        if batch_items:
            final_batch = DataProto.concat(batch_items)
            gen_batch_output = DataProto.concat(gen_items)

            # Ensure uids are preserved for GRPO grouping
            final_batch.non_tensor_batch["uid"] = np.array(final_uids, dtype=object)
        else:
            raise ValueError("No rollouts generated!")

        return final_batch, gen_batch_output


def compute_grpo_advantage_variable_n(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute GRPO advantages with variable rollouts per sample.

    This is identical to the standard GRPO computation - it naturally handles
    variable group sizes since it groups by uid and computes per-group statistics.

    Args:
        token_level_rewards: (total_rollouts, response_length)
        response_mask: (total_rollouts, response_length)
        index: Array of uids for grouping rollouts
        epsilon: Numerical stability
        norm_adv_by_std_in_grpo: Whether to normalize by std (standard GRPO)

    Returns:
        (advantages, returns): Both (total_rollouts, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]

        for i in range(bsz):
            id2score[index[i]].append(scores[i])

        for idx in id2score:
            group_scores = id2score[idx]
            if len(group_scores) == 1:
                # Single rollout: no baseline possible
                id2mean[idx] = torch.tensor(0.0, device=scores.device)
                id2std[idx] = torch.tensor(1.0, device=scores.device)
            else:
                scores_tensor = torch.stack(group_scores)
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)

        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def create_adaptive_rollout_config(
    max_n: int = 128,
    rollouts_per_batch: int = 8,
    positive_threshold: float = 0.5,
    min_rollouts: int = 2,
) -> dict:
    """
    Create configuration dict for adaptive rollout.

    Args:
        max_n: Maximum rollouts per sample
        rollouts_per_batch: Rollouts to generate per iteration
        positive_threshold: Reward threshold for positive rollout
        min_rollouts: Minimum rollouts per sample

    Returns:
        Configuration dictionary
    """
    return {
        "enable": True,
        "max_n": max_n,
        "rollouts_per_batch": rollouts_per_batch,
        "positive_threshold": positive_threshold,
        "min_rollouts": min_rollouts,
    }

