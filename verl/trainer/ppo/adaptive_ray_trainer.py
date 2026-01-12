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
Adaptive Rollout PPO Trainer.

This trainer extends RayPPOTrainer with adaptive rollout generation:
- Generates rollouts until at least one positive (correct) rollout is found
- Up to a maximum number of rollouts per sample
- Variable rollouts per sample based on difficulty

This is particularly useful for hard examples where the probability of
producing a correct answer is low.
"""

import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import AdvantageEstimator
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics


class AdaptiveRayPPOTrainer(RayPPOTrainer):
    """
    PPO Trainer with adaptive rollout generation.

    Extends RayPPOTrainer to generate variable numbers of rollouts per sample,
    stopping early when positive (correct) rollouts are found.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Get adaptive rollout config
        adaptive_config = self.config.get("adaptive_rollout", {})
        self.adaptive_enabled = adaptive_config.get("enable", False)
        self.adaptive_max_n = adaptive_config.get("max_n", 128)
        self.adaptive_rollouts_per_batch = adaptive_config.get("rollouts_per_batch", 8)
        self.adaptive_positive_threshold = adaptive_config.get("positive_threshold", 0.5)
        self.adaptive_min_rollouts = max(adaptive_config.get("min_rollouts", 2), 2)

        if self.adaptive_enabled:
            print(f"Adaptive rollout enabled: max_n={self.adaptive_max_n}, "
                  f"rollouts_per_batch={self.adaptive_rollouts_per_batch}, "
                  f"positive_threshold={self.adaptive_positive_threshold}, "
                  f"min_rollouts={self.adaptive_min_rollouts}")

    def _generate_adaptive_rollouts(
        self,
        gen_batch: DataProto,
        batch: DataProto,
        timing_raw: dict,
    ) -> tuple[DataProto, DataProto, dict[str, Any]]:
        """
        Generate rollouts adaptively until positive samples are found.

        Args:
            gen_batch: Generation batch (prompts only)
            batch: Full batch with reward model info
            timing_raw: Timing dictionary for profiling

        Returns:
            (final_batch, gen_batch_output, adaptive_metrics)
        """
        num_samples = len(gen_batch)
        sample_uids = batch.non_tensor_batch["uid"].copy()

        # State tracking
        uid_to_original_idx = {uid: i for i, uid in enumerate(sample_uids)}
        sample_has_positive = {uid: False for uid in sample_uids}
        sample_rollout_count = {uid: 0 for uid in sample_uids}
        sample_positive_count = {uid: 0 for uid in sample_uids}  # Count of positive rollouts per sample

        # Storage for all rollouts: uid -> list of rollout data
        all_rollout_data: dict[str, list[dict]] = {uid: [] for uid in sample_uids}

        iteration = 0
        max_iterations = (self.adaptive_max_n + self.adaptive_rollouts_per_batch - 1) // self.adaptive_rollouts_per_batch
        total_gen_time = 0.0
        total_reward_time = 0.0

        while iteration < max_iterations:
            # Find samples that still need more rollouts
            active_uids = []
            for uid in sample_uids:
                needs_more = (
                    (not sample_has_positive[uid]) or
                    (sample_rollout_count[uid] < self.adaptive_min_rollouts)
                )
                can_generate_more = sample_rollout_count[uid] < self.adaptive_max_n
                if needs_more and can_generate_more:
                    active_uids.append(uid)

            if not active_uids:
                break

            # Get indices of active samples
            active_indices = [uid_to_original_idx[uid] for uid in active_uids]

            # How many rollouts to generate this iteration
            n_this_batch = min(
                self.adaptive_rollouts_per_batch,
                self.adaptive_max_n - min(sample_rollout_count[uid] for uid in active_uids)
            )

            # Select and repeat active samples
            active_gen_batch = gen_batch.select_idxs(active_indices)
            active_batch = batch.select_idxs(active_indices)

            gen_batch_repeated = active_gen_batch.repeat(
                repeat_times=n_this_batch, interleave=True
            )

            # Generate rollouts
            import time
            gen_start = time.time()

            with marked_timer(f"gen_iter_{iteration}", timing_raw, color="red"):
                if self.async_rollout_mode:
                    gen_output = self.async_rollout_manager.generate_sequences(gen_batch_repeated)
                else:
                    gen_output = self.actor_rollout_wg.generate_sequences(gen_batch_repeated)

            gen_end = time.time()
            total_gen_time += gen_end - gen_start

            # Prepare batch for reward computation
            batch_repeated = active_batch.repeat(repeat_times=n_this_batch, interleave=True)
            
            # Clear conflicting meta_info before union (timing differs between batch and gen_output)
            gen_output.meta_info.pop("timing", None)
            batch_repeated.meta_info.pop("timing", None)
            
            reward_batch = batch_repeated.union(gen_output)

            # Compute rewards
            reward_start = time.time()

            with marked_timer(f"reward_iter_{iteration}", timing_raw, color="yellow"):
                if self.use_rm and self.rm_wg is not None and "rm_scores" not in reward_batch.batch.keys():
                    rm_scores = self.rm_wg.compute_rm_score(reward_batch)
                    reward_batch = reward_batch.union(rm_scores)

                reward_tensor, _ = compute_reward(reward_batch, self.reward_fn)

            reward_end = time.time()
            total_reward_time += reward_end - reward_start

            rewards = reward_tensor.sum(dim=-1).cpu().numpy()

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

                    if reward >= self.adaptive_positive_threshold:
                        sample_has_positive[uid] = True
                        sample_positive_count[uid] += 1

            iteration += 1

        # Assemble final batch with all rollouts
        final_batch, gen_batch_output = self._assemble_adaptive_batch(
            all_rollout_data, sample_uids
        )

        # Compute metrics
        rollout_counts = list(sample_rollout_count.values())
        positive_counts = list(sample_positive_count.values())
        total_rollouts = sum(rollout_counts)
        total_positive = sum(positive_counts)
        num_with_positive = sum(1 for v in sample_has_positive.values() if v)

        adaptive_metrics = {
            "adaptive_rollout/iterations": iteration,
            "adaptive_rollout/total_rollouts": total_rollouts,
            "adaptive_rollout/avg_rollouts_per_sample": float(np.mean(rollout_counts)),
            "adaptive_rollout/min_rollouts_per_sample": min(rollout_counts),
            "adaptive_rollout/max_rollouts_per_sample": max(rollout_counts),
            "adaptive_rollout/std_rollouts_per_sample": float(np.std(rollout_counts)),
            "adaptive_rollout/samples_with_positive": num_with_positive,
            "adaptive_rollout/samples_without_positive": num_samples - num_with_positive,
            "adaptive_rollout/positive_rate": num_with_positive / num_samples,
            "adaptive_rollout/efficiency": num_samples / total_rollouts,
            "adaptive_rollout/gen_time": total_gen_time,
            "adaptive_rollout/reward_time": total_reward_time,
            # Detailed positive rollout metrics (matching fixed rollout format)
            "positive_rollouts/avg_per_sample": float(np.mean(positive_counts)),
            "positive_rollouts/min_per_sample": int(np.min(positive_counts)),
            "positive_rollouts/max_per_sample": int(np.max(positive_counts)),
            "positive_rollouts/std_per_sample": float(np.std(positive_counts)),
            "positive_rollouts/samples_with_any_positive": num_with_positive,
            "positive_rollouts/samples_without_positive": num_samples - num_with_positive,
            "positive_rollouts/positive_rate": num_with_positive / num_samples if num_samples > 0 else 0.0,
            "positive_rollouts/total_positive": total_positive,
            "positive_rollouts/total_rollouts": total_rollouts,
            "positive_rollouts/overall_positive_rate": total_positive / total_rollouts if total_rollouts > 0 else 0.0,
            "positive_rollouts/avg_rollouts_per_sample": float(np.mean(rollout_counts)),
        }

        # Add timing to output metadata
        if hasattr(gen_batch_output, 'meta_info'):
            gen_batch_output.meta_info["timing"] = {
                "generate_sequences": total_gen_time,
            }

        return final_batch, gen_batch_output, adaptive_metrics

    def _assemble_adaptive_batch(
        self,
        all_rollout_data: dict[str, list[dict]],
        sample_uids: np.ndarray,
    ) -> tuple[DataProto, DataProto]:
        """Assemble all rollouts into final batch format.
        
        Also computes per-prompt normalization weights for the loss function.
        Each rollout i belonging to prompt x gets weight = 1/T(x) where T(x)
        is the number of rollouts for prompt x. This ensures each prompt
        contributes equally to the loss regardless of rollout count.
        """
        batch_items = []
        gen_items = []
        final_uids = []
        prompt_norm_weights = []  # Per-rollout normalization weights

        for uid in sample_uids:
            rollouts = all_rollout_data[uid]
            # Sort by generation order
            rollouts.sort(key=lambda x: (x["iteration"], x["rollout_idx"]))
            
            # Weight for each rollout of this prompt = 1/T(x)
            num_rollouts = len(rollouts)
            weight = 1.0 / num_rollouts

            for rollout in rollouts:
                batch_items.append(rollout["batch_item"])
                gen_items.append(rollout["gen_output"])
                final_uids.append(uid)
                prompt_norm_weights.append(weight)

        # Concatenate all
        if batch_items:
            final_batch = DataProto.concat(batch_items)
            gen_batch_output = DataProto.concat(gen_items)

            # Ensure uids are preserved for GRPO grouping
            final_batch.non_tensor_batch["uid"] = np.array(final_uids, dtype=object)
            
            # Add per-prompt normalization weights for loss computation
            # Shape: (total_rollouts,) - weight[i] = 1/T(x_i)
            final_batch.batch["prompt_norm_weights"] = torch.tensor(
                prompt_norm_weights, dtype=torch.float32
            )
        else:
            raise ValueError("No rollouts generated!")

        return final_batch, gen_batch_output

    def fit(self):
        """
        The training loop with adaptive rollouts.

        Overrides the parent fit() method to use adaptive rollout generation
        when enabled.
        """
        if not self.adaptive_enabled:
            # Use standard training loop
            return super().fit()

        # Adaptive rollout training loop
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()

        # Validation before training
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # Add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)
                gen_batch.meta_info["global_steps"] = self.global_steps

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # Generate rollouts adaptively
                    with marked_timer("gen_adaptive", timing_raw, color="red"):
                        batch, gen_batch_output, adaptive_metrics = self._generate_adaptive_rollouts(
                            gen_batch=gen_batch,
                            batch=batch,
                            timing_raw=timing_raw,
                        )
                        metrics.update(adaptive_metrics)

                    # Union batch with generation output
                    # Clear timing meta_info to avoid conflicts
                    batch.meta_info.pop("timing", None)
                    gen_batch_output.meta_info.pop("timing", None)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)

                    # Balance batch if needed
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # Compute final rewards (already computed during adaptive generation,
                    # but we need the full batch reward computation here)
                    with marked_timer("reward_final", timing_raw, color="yellow"):
                        if self.use_rm and self.rm_wg is not None and "rm_scores" not in batch.batch.keys():
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # Compute log probs
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)

                    if bypass_recomputing_logprobs:
                        from verl.trainer.ppo.rollout_corr_helper import apply_rollout_correction
                        apply_rollout_correction(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            from verl.trainer.ppo.core_algos import agg_loss
                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            entropy_agg = agg_loss(
                                loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode
                            )
                            old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)

                    # Reference policy
                    if self.use_reference_policy:
                        with marked_timer("ref_policy", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # Critic values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    # Compute advantages
                    with marked_timer("adv", timing_raw, color="brown"):
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Use variable-n aware advantage computation
                        # Note: The standard GRPO compute_advantage works fine with variable n
                        # since it groups by uid
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=1,  # Variable n, handled by uid grouping
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # Update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # Update actor
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout data if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # Validation
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Save checkpoint
                from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                steps_duration = timing_raw.get("step", 0)
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # Training metrics
                metrics.update({
                    "training/global_step": self.global_steps,
                    "training/epoch": epoch,
                })

                # Collect metrics
                from verl.trainer.ppo.metric_utils import (
                    compute_data_metrics,
                    compute_throughout_metrics,
                    compute_timing_metrics,
                )
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

