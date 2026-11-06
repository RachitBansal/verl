# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import time
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


def _merge_chunk_metrics(parts: list, sum_keys: tuple = ()) -> dict:
    """Merge the reduced metrics of the update chunks of one step (sub-sampled training, see
    RayPPOTrainer._rotation_update): ``parts`` = [(rows, metrics)].  Keys in ``sum_keys`` are summed (each chunk
    reports its share), keys containing "max"/"min" take the max/min, others the row-weighted mean over the
    chunks that report them (metrics of the tracked rows / the optimizer step come from the last chunk only)."""
    out: dict = {}
    keys = {k for _, m in parts for k in m}
    for k in keys:
        vals = [(n, m[k]) for n, m in parts if k in m]
        if k in sum_keys:
            out[k] = float(sum(v for _, v in vals))
        elif "max" in k:
            out[k] = max(v for _, v in vals)
        elif "min" in k:
            out[k] = min(v for _, v in vals)
        else:
            w = sum(n for n, _ in vals)
            out[k] = float(sum(n * v for n, v in vals) / max(w, 1))
    return out


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = (
            config.actor_rollout_ref.model.get("lora_rank", 0) > 0
            or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        )

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", None)
            or self.config.data.train_batch_size + int(self.config.data.get("kimi_holdout_prompts", 0) or 0),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)
            # per-source response length / clip ratio (reported through process_validation_metrics as
            # val-aux/<source>/response_length/mean@n and val-aux/<source>/response_clipped/mean@n)
            _resp_len = test_output_gen_batch.batch["attention_mask"][:, -output_ids.shape[1] :].sum(-1)
            reward_extra_infos_dict["response_length"].extend(_resp_len.float().tolist())
            reward_extra_infos_dict["response_clipped"].extend((_resp_len >= output_ids.shape[1]).float().tolist())

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role=str(Role.ActorRollout),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.ActorRollout)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
            self.ref_policy_wg.init_model()

        self.rm_wg = None
        # initalization of rm_wg will be deprecated in the future
        if self.use_rm:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(Role.ActorRollout)]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config, worker_group=self.actor_rollout_wg, rm_wg=self.rm_wg
            )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            # NOTE: while there is no checkpoint to load, we still need to offload the model and optimizer to CPU
            self.actor_rollout_wg.load_checkpoint(None)
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                self.actor_rollout_wg.load_checkpoint(None)
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        global_seqlen_lst = calculate_workload(global_seqlen_lst)
        world_size = self.actor_rollout_wg.world_size
        if keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(global_seqlen_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(world_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    global_seqlen_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=world_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(
                global_seqlen_lst, k_partitions=world_size, equal_size=True
            )
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        for idx, partition in enumerate(global_partition_lst):
            partition.sort(key=lambda x: (global_seqlen_lst[x], x))
            ordered_partition = partition[::2] + partition[1::2][::-1]
            global_partition_lst[idx] = ordered_partition
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    # ---- Kimi-K2 fresh-prompt hold-out (data.kimi_holdout_prompts) ------------------------------------
    def _split_holdout_rows(self, batch: DataProto) -> tuple[DataProto, Optional[DataProto]]:
        """Split the rollouts of the extra ``data.kimi_holdout_prompts`` prompts off the batch and flag both
        parts with ``kimi_holdout`` (0 = train, 1 = held out).  The dataloader draws
        ``train_batch_size + kimi_holdout_prompts`` prompts in sampler order and rollouts are interleaved per
        prompt, so the held-out rows are the last ``kimi_holdout_prompts * rollout.n`` rows (an arbitrary
        subset of the shuffled dataset, disjoint from the training prompts of this step).
        Returns ``(train_batch, holdout_batch_or_None)``."""
        n_hold_prompts = int(self.config.data.get("kimi_holdout_prompts", 0) or 0)
        if n_hold_prompts <= 0:
            return batch, None
        n_roll = self.config.actor_rollout_ref.rollout.n
        n_hold = n_hold_prompts * n_roll
        n_train = len(batch) - n_hold
        assert n_train == self.config.data.train_batch_size * n_roll, (
            f"kimi_holdout_prompts: expected {self.config.data.train_batch_size} train + {n_hold_prompts} held-out "
            f"prompts x {n_roll} rollouts, got {len(batch)} rows"
        )
        holdout = batch.slice(n_train, len(batch))
        batch = batch.slice(0, n_train)
        batch.non_tensor_batch["kimi_holdout"] = np.zeros(n_train, dtype=np.int64)
        holdout.non_tensor_batch["kimi_holdout"] = np.ones(n_hold, dtype=np.int64)
        return batch, holdout

    def _layout_holdout_rows(self, batch: DataProto) -> None:
        """In place: reorder rows so that every actor DP rank's contiguous dispatch chunk is
        [its balanced training rows, its equal share of held-out rows].  Every rank then has the same
        number of training rows (hence mini-batches and optimizer steps, which FSDP requires) and the same
        number of held-out rows; the actor separates the two by the ``kimi_holdout`` flag."""
        flag = batch.non_tensor_batch["kimi_holdout"].astype(bool)
        train_idx, hold_idx = np.flatnonzero(~flag), np.flatnonzero(flag)
        ws = self.actor_rollout_wg.world_size
        assert len(train_idx) % ws == 0 and len(hold_idx) % ws == 0, (
            f"kimi_holdout: {len(train_idx)} training and {len(hold_idx)} held-out rows must both be divisible "
            f"by the actor world size {ws} (rows dropped after generation?)"
        )
        nt, nh = len(train_idx) // ws, len(hold_idx) // ws
        order = np.concatenate(
            [np.concatenate([train_idx[r * nt : (r + 1) * nt], hold_idx[r * nh : (r + 1) * nh]]) for r in range(ws)]
        )
        batch.reorder(torch.from_numpy(order))

    def _drop_holdout_rows(self, batch: DataProto, metrics: dict) -> DataProto:
        """Log the held-out rollouts' reward and remove them from the batch."""
        flag = batch.non_tensor_batch["kimi_holdout"].astype(bool)
        if flag.any() and "token_level_scores" in batch.batch:
            hold_scores = batch.batch["token_level_scores"][torch.from_numpy(flag)].sum(-1)
            metrics["kimi_holdout/reward_mean"] = float(hold_scores.mean().item())
            metrics["kimi_holdout/num_rows"] = int(flag.sum())
        batch = batch.select_idxs(~flag)
        batch.non_tensor_batch.pop("kimi_holdout", None)
        return batch

    def _val_reached_stop_threshold(self, val_metrics: dict) -> bool:
        """True when trainer.stop_val_threshold is set and the chosen validation metric has reached it."""
        thr = self.config.trainer.get("stop_val_threshold", None)
        if thr is None or not val_metrics:
            return False
        key = self.config.trainer.get("stop_val_metric", None)
        if key is None:
            cands = sorted(k for k in val_metrics if k.startswith("val-core/") and "/mean@" in k)
            if not cands:
                return False
            key = cands[0]
        val = val_metrics.get(key)
        if val is None:
            return False
        if val >= thr:
            print(f"Stopping: {key}={val:.4f} >= trainer.stop_val_threshold={thr} at step {self.global_steps}")
            return True
        return False

    # ---- Fixed-dataset study (trainer.fixed_dataset; Phase2a_Fixed_PiRef_Dataset_Plan.md) -----------------
    def _generate_rows(self, prompts: DataProto, gen_batch: DataProto, n_roll: int, timing_raw: dict) -> DataProto:
        """``n_roll`` rollouts per prompt of ``prompts`` (whose generation inputs are ``gen_batch``), joined
        into one row per rollout exactly as the main loop does."""
        gen_in = gen_batch.repeat(repeat_times=n_roll, interleave=True)
        with marked_timer("gen", timing_raw, color="red"):
            if not self.async_rollout_mode:
                gen_out = self.actor_rollout_wg.generate_sequences(gen_in)
            else:
                gen_out = self.async_rollout_manager.generate_sequences(gen_in)
        gen_out.meta_info.pop("timing", None)
        rows = prompts.repeat(repeat_times=n_roll, interleave=True)
        return rows.union(gen_out)

    def _build_fixed_bank(self, path: str, logger) -> DataProto:
        """Sample the bank from the *current* (= reference, asserted) policy and save it to ``path``.

        Rows: ``data.train_batch_size`` training prompts x rollout.n (flag ``kimi_holdout`` 0), the
        ``data.kimi_holdout_prompts`` extra prompts drawn by the dataloader x n (flag 1, fresh prompts), and a
        second, independent set of n rollouts for the first ``holdout_sample_prompts`` training prompts
        (flag 2, fresh samples of seen prompts; new uids so their group baseline is their own).  Stores the
        reward, the group-mean advantages, the trainer-recomputed log-probs as both ``old_log_probs`` (the frozen
        anchor) and ``ref_log_probs``, each row's prompt pass rate (``ref_pass_rate``), the probe flag
        (``kimi_probe``: complete groups of ``probe_prompts`` random training prompts) and ``bank_prompt_uid``
        (the training prompt a flag-2 row duplicates).  Logs a ``bank/*`` summary at step 0 and writes it to
        ``<path>.summary.json``."""
        fd = self.config.trainer.fixed_dataset
        assert self.global_steps == 0, "the fixed bank must be generated at the reference weights (no checkpoint)"
        n_roll = self.config.actor_rollout_ref.rollout.n
        bsz = self.config.data.train_batch_size
        n_hold_prompts = int(self.config.data.get("kimi_holdout_prompts", 0) or 0)
        n_sample_prompts = int(fd.get("holdout_sample_prompts", 0) or 0)
        timing_raw: dict = {}
        metrics: dict = {}
        with marked_timer("bank_build", timing_raw):
            batch = DataProto.from_single_dict(next(iter(self.train_dataloader)))
            assert len(batch) == bsz + n_hold_prompts, (
                f"fixed bank: the dataloader gave {len(batch)} prompts, expected train_batch_size {bsz} + "
                f"kimi_holdout_prompts {n_hold_prompts} (prompts filtered out? use a pre-filtered prompt file)"
            )
            batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch))], dtype=object)
            gen_batch = self._get_gen_batch(batch)
            gen_batch.meta_info["global_steps"] = 0
            rows = self._generate_rows(batch, gen_batch, n_roll, timing_raw)
            rows.non_tensor_batch["kimi_holdout"] = np.concatenate(
                [np.zeros(bsz * n_roll, dtype=np.int64), np.ones(n_hold_prompts * n_roll, dtype=np.int64)]
            )
            rows.non_tensor_batch["bank_prompt_uid"] = rows.non_tensor_batch["uid"].copy()
            parts = [rows]
            if n_sample_prompts > 0:
                assert n_sample_prompts <= bsz
                sub = batch.slice(0, n_sample_prompts)
                sub.non_tensor_batch["bank_prompt_uid"] = sub.non_tensor_batch["uid"].copy()
                sub.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(sub))], dtype=object)
                rows2 = self._generate_rows(sub, gen_batch.slice(0, n_sample_prompts), n_roll, timing_raw)
                rows2.non_tensor_batch["kimi_holdout"] = np.full(len(rows2), 2, dtype=np.int64)
                parts.append(rows2)
            bank = DataProto.concat(parts)
            bank.batch["response_mask"] = compute_response_mask(bank)
            bank.meta_info["global_token_num"] = torch.sum(bank.batch["attention_mask"], dim=-1).tolist()
            with marked_timer("reward", timing_raw, color="yellow"):
                reward_tensor, reward_extra_infos_dict = compute_reward(bank, self.reward_fn)
            bank.batch["token_level_scores"] = reward_tensor
            bank.batch["token_level_rewards"] = reward_tensor
            if reward_extra_infos_dict:
                bank.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
            with marked_timer("old_log_prob", timing_raw, color="blue"):
                old_log_prob = self.actor_rollout_wg.compute_log_prob(bank)
            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
            metrics["bank/entropy_ref"] = agg_loss(
                loss_mat=old_log_prob.batch["entropys"],
                loss_mask=bank.batch["response_mask"],
                loss_agg_mode=loss_agg_mode,
            ).item()
            old_log_prob.batch.pop("entropys")
            bank = bank.union(old_log_prob)
            bank.batch["ref_log_probs"] = bank.batch["old_log_probs"].clone()
            bank = self._finalize_fixed_bank(bank, n_roll, metrics)
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            bank.save_to_disk(path)
            with open(path + ".summary.json", "w") as f:
                json.dump(metrics, f, indent=2, default=float)
        metrics.update({f"timing_s/{k}": v for k, v in timing_raw.items()})
        print(f"fixed bank saved to {path}: {len(bank)} rows; summary: {metrics}")
        logger.log(data=metrics, step=0)
        return bank

    def _finalize_fixed_bank(self, bank: DataProto, n_roll: int, metrics: dict) -> DataProto:
        """Shared tail of the bank build and the master-bank load: group-mean advantages (uid groups), per-row
        reference pass rate ``ref_pass_rate``, the probe flag ``kimi_probe`` (complete groups of
        ``probe_prompts`` random training prompts, seeded) and the ``bank/*`` summary written into ``metrics``.
        Needs ``token_level_scores/rewards``, ``response_mask`` and non-tensor ``uid`` / ``kimi_holdout``."""
        from verl.trainer.ppo.kimi_fixed_bank import bank_floor_summary

        fd = self.config.trainer.fixed_dataset
        n_probe_prompts = int(fd.get("probe_prompts", 0) or 0)
        max_resp = self.config.data.max_response_length
        bank = compute_advantage(
            bank,
            adv_estimator=self.config.algorithm.adv_estimator,
            gamma=self.config.algorithm.gamma,
            lam=self.config.algorithm.lam,
            num_repeat=n_roll,
            norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
            config=self.config.algorithm,
        )
        # per-row reference pass rate (group mean reward) and the probe flag (complete groups)
        uid = bank.non_tensor_batch["uid"].astype(str)
        flags = bank.non_tensor_batch["kimi_holdout"].astype(np.int64)
        seq_r = bank.batch["token_level_scores"].sum(-1).float().numpy()
        _, inv = np.unique(uid, return_inverse=True)
        gsum, gcnt = np.bincount(inv, weights=seq_r), np.bincount(inv)
        bank.non_tensor_batch["ref_pass_rate"] = (gsum / gcnt)[inv].astype(np.float32)
        probe = np.zeros(len(bank), dtype=np.int64)
        if n_probe_prompts > 0 and (flags == 0).any():
            rng = np.random.default_rng(int(fd.get("seed", 0)))
            train_uids = np.unique(uid[flags == 0])
            chosen = set(rng.choice(train_uids, size=min(n_probe_prompts, len(train_uids)), replace=False).tolist())
            probe = np.array([int(f == 0 and u in chosen) for u, f in zip(uid, flags, strict=True)], dtype=np.int64)
        bank.non_tensor_batch["kimi_probe"] = probe
        bank.meta_info = {}
        # summary
        resp_len = bank.batch["response_mask"].sum(-1).float()
        for v, name in ((0, "train"), (1, "holdout_prompt"), (2, "holdout_sample")):
            m = flags == v
            if m.any():
                mt = torch.from_numpy(m)
                metrics[f"bank/{name}_rows"] = int(m.sum())
                metrics[f"bank/{name}_reward_mean"] = float(seq_r[m].mean())
                metrics[f"bank/{name}_response_length_mean"] = resp_len[mt].mean().item()
                metrics[f"bank/{name}_clip_ratio"] = (resp_len[mt] >= max_resp).float().mean().item()
        if (flags == 0).any():  # (sub-sampled training: no training rows in the bank, see _setup_rotation)
            k = np.rint(gsum[np.unique(inv[flags == 0])]).astype(np.int64)
            metrics["bank/train_prompts"] = int(len(k))
            metrics["bank/zero_adv_prompt_frac"] = float(((k == 0) | (k == n_roll)).mean())
            metrics["bank/pass_at_n"] = float((k > 0).mean())
            for kk in range(n_roll + 1):
                metrics[f"bank/pass_count_hist/{kk}"] = int((k == kk).sum())
            metrics.update(bank_floor_summary(k, n_roll, [float(t) for t in fd.get("floor_taus", [0.3, 0.1, 0.03])]))
        metrics["bank/probe_rows"] = int(probe.sum())
        return bank

    def _layout_fixed_bank(self, bank: DataProto, metrics: dict) -> DataProto:
        """Per-step row order of the bank: every actor DP rank's contiguous dispatch chunk is
        [its length-balanced share of the non-probe training rows, its share of the probe rows, its share of
        each held-out set].  Probe and held-out rows stay in bank (interleaved-group) order so each rank holds
        complete prompt groups of them (needed by the per-prompt coverage metrics).  Returns a new DataProto."""
        flags = bank.non_tensor_batch["kimi_holdout"].astype(np.int64)
        probe = bank.non_tensor_batch["kimi_probe"].astype(bool)
        ws = self.actor_rollout_wg.world_size
        main = bank.select_idxs((flags == 0) & ~probe)
        if self.config.trainer.balance_batch:
            self._balance_batch(main, metrics=metrics)
        blocks = [main]
        if probe.any():
            blocks.append(bank.select_idxs((flags == 0) & probe))
        for v in sorted(set(flags.tolist()) - {0}):
            blocks.append(bank.select_idxs(flags == v))
        n_roll = self.config.actor_rollout_ref.rollout.n
        for b in blocks:
            assert len(b) % ws == 0, f"fixed bank: block of {len(b)} rows not divisible by world size {ws}"
        for b in blocks[1:]:
            if (len(b) // ws) % n_roll != 0:
                print(
                    f"WARNING fixed bank: a tracked block of {len(b)} rows gives {len(b) // ws} rows per rank, not a "
                    f"multiple of rollout.n={n_roll}; per-prompt coverage metrics will see split groups"
                )
        per_rank = []
        for r in range(ws):
            for b in blocks:
                nb = len(b) // ws
                per_rank.append(b.slice(r * nb, (r + 1) * nb))
        out = DataProto.concat(per_rank)
        out.meta_info = {}
        return out

    def _rebaseline_fixed_bank(self, bank: DataProto, baseline: str) -> dict:
        """Set the bank's regression targets in place.  ``mean``: keep the stored group-mean advantages
        (r - rbar).  ``tau_logz``: A = r - tau log Zhat(x), Zhat = mean_j exp(r_j/tau) over the prompt's group
        (kimi_fixed_bank.soft_value_baseline; tau = actor.policy_loss.kimi_tau), which makes the targets A/tau the
        log-ratios of a normalised policy.  The per-row baseline is stored as ``kimi_baseline`` (the actor's
        coverage metrics recover the reward as A + baseline).  Returns ``bank_baseline/*`` summary metrics.

        With ``fixed_dataset.baseline_file`` (npz from compute_master_baselines.py) the baseline is not taken
        from the row's group but looked up per prompt row of the master bank: ``mean`` = the pass rate over all
        N stored samples of the prompt, ``tau_logz`` = the precomputed tau log Zhat over the same N samples at
        tau = kimi_tau (must be one of the file's taus).  Every row (training and held-out) gets A = r - b(x),
        ``kimi_baseline`` = b(x) and ``ref_pass_rate`` = the N-sample pass rate."""
        assert baseline in ("mean", "tau_logz"), f"unknown fixed_dataset.baseline={baseline}"
        fd = self.config.trainer.fixed_dataset
        tau = float(self.config.actor_rollout_ref.actor.policy_loss.kimi_tau)
        seq_r = bank.batch["token_level_scores"].sum(-1).double().numpy()
        uid = bank.non_tensor_batch["uid"]
        out: dict = {}
        if fd.get("baseline_file", None):
            from verl.trainer.ppo.kimi_fixed_bank import lookup_master_baseline

            assert "master_prompt_row" in bank.non_tensor_batch, (
                "fixed_dataset.baseline_file needs a master-bank run (fixed_dataset.master_selection)"
            )
            assert not self.config.algorithm.get("norm_adv_by_std_in_grpo", True), (
                "fixed_dataset.baseline_file replaces the advantages by r - b(x); set "
                "algorithm.norm_adv_by_std_in_grpo=False (no per-group std scaling)"
            )
            with open(fd.master_selection) as f:
                sel_master = json.load(f).get("master_dir") or fd.master_dir
            b, rbar, info = lookup_master_baseline(
                fd.baseline_file, bank.non_tensor_batch["master_prompt_row"], baseline, tau, master_dir=sel_master
            )
            bank.non_tensor_batch["ref_pass_rate"] = rbar.astype(np.float32)
            out.update({f"bank_baseline/{k}": v for k, v in info.items()})
            print(f"fixed bank baselines from {fd.baseline_file}: {baseline} over N={info['n_samples']} samples")
        elif baseline == "mean":
            return {}
        else:
            from verl.trainer.ppo.kimi_fixed_bank import soft_value_baseline

            b = soft_value_baseline(seq_r, uid, tau)
        adv = torch.from_numpy(seq_r - b).to(bank.batch["advantages"].dtype)
        mask = bank.batch["response_mask"].to(adv.dtype)
        bank.batch["advantages"] = adv.unsqueeze(-1) * mask
        if "returns" in bank.batch.keys():
            bank.batch["returns"] = bank.batch["advantages"].clone()
        bank.non_tensor_batch["kimi_baseline"] = b.astype(np.float32)
        flags = bank.non_tensor_batch["kimi_holdout"].astype(np.int64)
        tgt = (seq_r - b) / tau
        for v, name in ((0, "train"), (1, "holdout_prompt"), (2, "holdout_sample")):
            m = flags == v
            if not m.any():
                continue
            pos, neg = m & (tgt > 1e-8), m & (tgt < -1e-8)
            out[f"bank_baseline/{name}_target_correct_mean"] = float(tgt[pos].mean()) if pos.any() else 0.0
            out[f"bank_baseline/{name}_target_wrong_mean"] = float(tgt[neg].mean()) if neg.any() else 0.0
            out[f"bank_baseline/{name}_target_rms"] = float(np.sqrt((tgt[m] ** 2).mean()))
            out[f"bank_baseline/{name}_loss_at_zero_delta"] = float(((seq_r[m] - b[m]) ** 2).mean())
            # prompts of this role whose rows all have zero target (no signal), and the in-group mean reward's
            # distance from the baseline (how far the old group-mean baseline was from this one)
            _, inv = np.unique(np.asarray(uid[m], dtype=object).astype(str), return_inverse=True)
            gabs = np.bincount(inv, weights=np.abs(tgt[m]))
            out[f"bank_baseline/{name}_zero_target_prompt_frac"] = float((gabs < 1e-8).mean())
            if baseline == "mean":
                gmean = np.bincount(inv, weights=seq_r[m]) / np.bincount(inv)
                out[f"bank_baseline/{name}_group_mean_minus_baseline_rms"] = float(
                    np.sqrt(((gmean[inv] - b[m]) ** 2).mean())
                )
        print(f"fixed bank re-baselined ({baseline}, tau={tau}): {out}")
        return out

    # ---- master bank (trainer.fixed_dataset.master_dir; verl/trainer/ppo/kimi_master_bank.py) ----------------
    def _build_master_bank(self, logger) -> None:
        """Sample ``rollout.n`` reference rollouts of *every* prompt of ``data.train_files`` (in file order) and
        store them as compact shards of ``master_chunk_prompts`` prompts under ``master_dir``.  Job
        ``master_job_index`` of ``master_num_jobs`` builds chunks c with c % num_jobs == job_index and skips
        chunks already on disk, so several jobs can split the work and a resubmitted job resumes.  The job that
        finds every shard done writes ``manifest.json`` / ``summary.json``.

        Extension banks (more samples of the same prompts, e.g. for baselines): ``master_sample_offset`` numbers
        the samples from that offset, ``actor_rollout_ref.rollout.seed`` must differ from the base bank's (vLLM
        is seeded per engine, so the same seed can reproduce the base bank's samples), and
        ``master_ref_log_probs=False`` skips the trainer's reference log-prob pass (the bank then serves
        baselines only, see compute_master_baselines.py)."""
        import pandas as pd

        from verl.trainer.ppo import kimi_master_bank as mb
        from verl.utils.dataset.rl_dataset import collate_fn

        fd = self.config.trainer.fixed_dataset
        assert self.global_steps == 0, "the master bank must be generated at the reference weights (no checkpoint)"
        out_dir = fd.master_dir
        chunk = int(fd.get("master_chunk_prompts", 512))
        job, num_jobs = int(fd.get("master_job_index", 0)), int(fd.get("master_num_jobs", 1))
        assert 0 <= job < num_jobs
        n_roll = self.config.actor_rollout_ref.rollout.n
        ws = self.actor_rollout_wg.world_size
        assert n_roll % ws == 0 or (chunk * n_roll) % ws == 0, "rows per chunk must split evenly over the workers"
        max_resp = self.config.data.max_response_length
        ds = self.train_dataset
        train_files = self.config.data.train_files
        train_file = train_files if isinstance(train_files, str) else list(train_files)[0]
        n_file = len(pd.read_parquet(train_file, columns=["data_source"]))
        assert len(ds) == n_file, (
            f"master bank: the dataset kept {len(ds)} of {n_file} prompts (length filter?); prompt rows would not "
            "match the file. Pre-filter the prompt file (prepare_master_prompts.py)."
        )
        n_chunks = (len(ds) + chunk - 1) // chunk
        rollout = self.config.actor_rollout_ref.rollout
        sample_offset = int(fd.get("master_sample_offset", 0) or 0)
        with_ref = bool(fd.get("master_ref_log_probs", True))
        rollout_seed = int(rollout.get("seed", 0) or 0)
        cfg = {
            "format_version": mb.FORMAT_VERSION,
            "model": self.config.actor_rollout_ref.model.path,
            "prompt_file": train_file,
            "num_prompts": len(ds),
            "n": n_roll,
            "chunk_prompts": chunk,
            "num_shards": n_chunks,
            "temperature": float(rollout.temperature),
            "top_p": float(rollout.get("top_p", 1.0)),
            "top_k": int(rollout.get("top_k", -1)),
            "max_prompt_length": int(self.config.data.max_prompt_length),
            "max_response_length": int(max_resp),
            "ref_log_probs": "FSDP actor compute_log_prob at the initial weights (the trainer's frozen anchor)"
            if with_ref
            else "none (not computed; baseline-only bank)",
        }
        # keys added after the first master bank: written only when non-default, so its build_config still matches
        if sample_offset:
            cfg["sample_offset"] = sample_offset
        if rollout_seed:
            cfg["rollout_seed"] = rollout_seed
        os.makedirs(out_dir, exist_ok=True)
        cfg_path = os.path.join(out_dir, "build_config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                old = json.load(f)
            diff = {k: (old.get(k), v) for k, v in cfg.items() if old.get(k) != v}
            assert not diff, f"master bank {out_dir} was started with different settings: {diff}"
        else:
            mb.atomic_write_json(cfg_path, cfg)
        mine = [c for c in range(n_chunks) if c % num_jobs == job and not mb.shard_done(out_dir, c)]
        print(
            f"master bank {out_dir}: {len(ds)} prompts x {n_roll} (samples {sample_offset}..{sample_offset + n_roll - 1}, "
            f"rollout seed {rollout_seed}, reference log-probs {with_ref}), {n_chunks} shards; "
            f"job {job}/{num_jobs} builds {mine}"
        )
        for c in mine:
            timing_raw: dict = {}
            with marked_timer("shard", timing_raw):
                idx = list(range(c * chunk, min(len(ds), (c + 1) * chunk)))
                batch = DataProto.from_single_dict(collate_fn([ds[i] for i in idx]))
                assert (len(idx) * n_roll) % ws == 0, f"shard {c}: {len(idx)} x {n_roll} rows not divisible by {ws}"
                batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in idx], dtype=object)
                gen_batch = self._get_gen_batch(batch)
                gen_batch.meta_info["global_steps"] = 0
                rows = self._generate_rows(batch, gen_batch, n_roll, timing_raw)
                rows.batch["response_mask"] = compute_response_mask(rows)
                rows.meta_info["global_token_num"] = torch.sum(rows.batch["attention_mask"], dim=-1).tolist()
                with marked_timer("reward", timing_raw, color="yellow"):
                    reward_tensor, extras = compute_reward(rows, self.reward_fn)
                rows.batch["token_level_scores"] = reward_tensor
                lp = None
                if with_ref:
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        lp = self.actor_rollout_wg.compute_log_prob(rows)
                rmask = rows.batch["response_mask"]
                numeric = {}
                for k, v in (extras or {}).items():
                    try:
                        numeric[k] = np.asarray(v, dtype=np.float32)
                    except (TypeError, ValueError):
                        pass
                with marked_timer("save", timing_raw):
                    prompt_len = rows.batch["prompts"].shape[-1]
                    sh = mb.pack_shard(
                        prompts=rows.batch["prompts"],
                        prompt_mask=rows.batch["attention_mask"][:, :prompt_len],
                        responses=rows.batch["responses"],
                        response_mask=rmask,
                        ref_log_probs=None if lp is None else lp.batch["old_log_probs"],
                        rollout_log_probs=rows.batch["rollout_log_probs"]
                        if "rollout_log_probs" in rows.batch.keys()
                        else None,
                        scores=reward_tensor.sum(-1),
                        entropy_sum=None if lp is None else (lp.batch["entropys"].float() * rmask).sum(-1),
                        n=n_roll,
                        prompt_rows=np.asarray(idx, dtype=np.int64),
                        dapo_index=np.asarray(rows.non_tensor_batch["index"][::n_roll], dtype=np.int64)
                        if "index" in rows.non_tensor_batch
                        else np.asarray(idx, dtype=np.int64),
                        data_source=np.asarray(rows.non_tensor_batch["data_source"][::n_roll]),
                        extras=numeric,
                        sample_offset=sample_offset,
                    )
                    side = mb.save_shard(out_dir, c, sh, max_resp)
            m = {f"master_build/timing_s/{k}": v for k, v in timing_raw.items()}
            m["master_build/shard"] = c
            m["master_build/reward_mean"] = side["reward_sum"] / side["num_rows"]
            m["master_build/response_length_mean"] = side["response_tokens"] / side["num_rows"]
            m["master_build/clip_ratio"] = side["clipped_rows"] / side["num_rows"]
            print(f"master shard {c} done: {m}")
            logger.log(data=m, step=c)
        if all(mb.shard_done(out_dir, c) for c in range(n_chunks)):
            summary = mb.write_manifest(
                out_dir, n_chunks, chunk, max_resp, [float(t) for t in fd.get("floor_taus", [0.3, 0.1, 0.03])]
            )
            print(f"master bank complete: {summary}")
            logger.log(data=summary, step=n_chunks)
        else:
            missing = [c for c in range(n_chunks) if not mb.shard_done(out_dir, c)]
            print(f"master bank: job {job} done; shards still missing (other jobs): {missing}")

    def _build_master_anchor(self, logger) -> None:
        """Recompute the frozen anchor (reference log-probs) of every master-bank row with the actor as currently
        configured (e.g. fsdp_config.dtype=fp32 + allow_tf32) and store it as ``anchors/<tag>/shard_XXXXX.pt``
        beside the shards (tag = ``fixed_dataset.master_anchor``).  Runs at the initial weights only; job split /
        resume as in ``_build_master_bank``.  A run reads the anchor with the same ``master_anchor`` tag, so its
        anchor and its training forward share one numerical precision."""
        from verl.trainer.ppo import kimi_master_bank as mb

        fd = self.config.trainer.fixed_dataset
        assert self.global_steps == 0, "the anchor must be computed at the reference weights (no checkpoint)"
        master_dir, tag = fd.master_dir, fd.master_anchor
        job, num_jobs = int(fd.get("master_job_index", 0)), int(fd.get("master_num_jobs", 1))
        man = mb.load_manifest(master_dir)
        n, ws = man["n"], self.actor_rollout_wg.world_size
        fsdp_cfg = self.config.actor_rollout_ref.actor.fsdp_config
        cfg = {
            "tag": tag,
            "model": self.config.actor_rollout_ref.model.path,
            "fsdp_dtype": str(fsdp_cfg.get("dtype")),
            "allow_tf32": bool(fsdp_cfg.get("allow_tf32", False)),
            # fp32 compute runs flash attention in fp16 (verl/models/transformers/monkey_patch.py)
            "attention": "flash fp16" if str(fsdp_cfg.get("dtype")) in ("fp32", "float32") else "flash",
            "temperature": float(self.config.actor_rollout_ref.rollout.temperature),
        }
        adir = os.path.join(master_dir, "anchors", tag)
        os.makedirs(adir, exist_ok=True)
        cfg_path = os.path.join(adir, "anchor_config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                old = json.load(f)
            diff = {k: (old.get(k), v) for k, v in cfg.items() if old.get(k) != v}
            assert not diff, f"anchor {adir} was started with different settings: {diff}"
        else:
            mb.atomic_write_json(cfg_path, cfg)
        n_sh = len(man["shards"])
        mine = [c for c in range(n_sh) if c % num_jobs == job and not mb.anchor_done(master_dir, tag, c)]
        print(f"master anchor {adir}: {cfg}; job {job}/{num_jobs} computes shards {mine}")
        for c in mine:
            t0 = time.time()
            s = man["shards"][c]
            prow = list(range(s["prompt_row_start"], s["prompt_row_end"]))
            pieces = mb.gather_rows(master_dir, [(p, list(range(n))) for p in prow])["pieces"]
            t = mb.unpack_rows(
                pieces,
                pad_token_id=self.tokenizer.pad_token_id,
                max_prompt_length=int(self.config.data.max_prompt_length),
                max_response_length=int(self.config.data.max_response_length),
            )
            keys = ("input_ids", "attention_mask", "position_ids", "responses", "response_mask")
            rows = DataProto.from_dict(tensors={k: t[k] for k in keys})
            rows, pad = pad_dataproto_to_divisor(rows, ws)
            out = self.actor_rollout_wg.compute_log_prob(rows)
            lp = out.batch["old_log_probs"][: len(rows) - pad].float()
            rmask = t["response_mask"].bool()
            flat = lp[rmask]  # row-major = the shard's flat response layout (row lp * n + s, real tokens only)
            old = torch.cat([r for p in pieces for r in p["ref_log_probs"]]).float()
            assert flat.numel() == old.numel(), f"shard {c}: {flat.numel()} anchor tokens vs {old.numel()} stored"
            seq_new = (lp * rmask).sum(-1).double()
            seq_old = (t["old_log_probs"].float() * rmask).sum(-1).double()
            st = {
                "num_rows": int(rmask.shape[0]),
                "num_tokens": int(flat.numel()),
                "seq_delta_vs_stored_rms": float(((seq_new - seq_old) ** 2).mean().sqrt()),
                "tok_delta_vs_stored_rms": float(((flat - old).double() ** 2).mean().sqrt()),
                "seconds": time.time() - t0,
            }
            mb.save_anchor(master_dir, tag, c, flat, st)
            print(f"master anchor shard {c}: {st}")
            logger.log(data={f"master_anchor/{k}": v for k, v in st.items()}, step=c)
        missing = [c for c in range(n_sh) if not mb.anchor_done(master_dir, tag, c)]
        print(f"master anchor {tag}: job {job} done; shards still missing: {missing}")

    def _master_dataproto(self, pieces: list, roles: list) -> DataProto:
        """Padded rows of master-bank pieces (kimi_master_bank.unpack_rows; one row per (piece, sample)) with the
        non-tensor fields of a fixed-bank row.  ``roles[i]`` = (kimi_holdout flag, uid, bank_prompt_uid) of
        piece i, shared by all its samples."""
        from verl.trainer.ppo import kimi_master_bank as mb

        tensors = mb.unpack_rows(
            pieces,
            pad_token_id=self.tokenizer.pad_token_id,
            max_prompt_length=int(self.config.data.max_prompt_length),
            max_response_length=int(self.config.data.max_response_length),
        )
        reps = [len(p["samples"]) for p in pieces]

        def rep(vals, dtype):
            return np.array([v for v, r in zip(vals, reps, strict=True) for _ in range(r)], dtype=dtype)

        non_tensors = {
            "uid": rep([r[1] for r in roles], object),
            "bank_prompt_uid": rep([r[2] for r in roles], object),
            "kimi_holdout": rep([r[0] for r in roles], np.int64),
            "index": rep([p["dapo_index"] for p in pieces], object),
            "data_source": rep([p["data_source"] for p in pieces], object),
            "master_sample": np.array([s for p in pieces for s in p["samples"]], dtype=np.int64),
            # prompt row in the master prompt file: key of the precomputed baselines (fixed_dataset.baseline_file)
            "master_prompt_row": rep([p["prompt_row"] for p in pieces], np.int64),
        }
        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)

    def _load_master_bank(self, role_names: tuple = ("train", "holdout_prompt", "holdout_sample")) -> tuple[DataProto, dict]:
        """Bank of one run from the master bank: rows chosen by ``fixed_dataset.master_selection`` (json written
        by examples/kimi_k2/fixed_ref/prepare_master_selection.py): training prompts x their sample indices
        (flag 0), fresh prompts (flag 1) and second sample sets of some training prompts (flag 2, own uids,
        ``bank_prompt_uid`` = the training prompt's uid).  Then the same finalisation as a built bank.
        ``role_names`` restricts the roles loaded (sub-sampled training loads its training rows per step)."""
        from verl.trainer.ppo import kimi_master_bank as mb

        fd = self.config.trainer.fixed_dataset
        with open(fd.master_selection) as f:
            sel = json.load(f)
        master_dir = sel.get("master_dir") or fd.master_dir
        assert master_dir, "master bank: set fixed_dataset.master_dir or master_dir in the selection file"
        n_roll = self.config.actor_rollout_ref.rollout.n
        k = int(sel["k"])
        assert k == n_roll, f"selection has k={k} samples per prompt; set actor_rollout_ref.rollout.n={k}"
        roles = [r for r in [(0, "train"), (1, "holdout_prompt"), (2, "holdout_sample")] if r[1] in role_names]
        requests, piece_roles = [], []
        for flag, name in roles:
            samples = [int(s) for s in sel["samples"][name]]
            assert len(samples) == k or not sel[name], f"{name}: {len(samples)} sample indices, expected {k}"
            for prow in sel[name]:
                requests.append((int(prow), samples))
                uid = f"{name}:{int(prow)}"
                piece_roles.append((flag, uid, f"train:{int(prow)}" if flag == 2 else uid))
        t0 = time.time()
        anchor = fd.get("master_anchor", None)
        pieces = mb.gather_rows(master_dir, requests, anchor=anchor)["pieces"]
        bank = self._master_dataproto(pieces, piece_roles)
        metrics: dict = {}
        ent = torch.cat([p["ref_entropy_sum"] for p in pieces]).double()
        train = torch.from_numpy(bank.non_tensor_batch["kimi_holdout"] == 0)
        if bool(train.any()):
            metrics["bank/entropy_ref_token_mean"] = float(
                ent[train].sum() / bank.batch["response_mask"][train].sum().clamp(min=1)
            )
        bank = self._finalize_fixed_bank(bank, n_roll, metrics)
        metrics["bank/load_s"] = time.time() - t0
        print(
            f"master bank {master_dir} (anchor {anchor or 'stored'}): loaded {len(bank)} rows for selection "
            f"{fd.master_selection} in "
            f"{metrics['bank/load_s']:.0f}s; summary: {metrics}"
        )
        return bank, metrics

    # ---- sub-sampled training (trainer.fixed_dataset.rotate_k) ---------------------------------------------------
    def _setup_rotation(self) -> tuple[dict, dict]:
        """Training pool of sub-sampled training: every training prompt of the selection x master samples
        0..rotate_pool-1 (default all n), minus the prompt's held-out-sample set (selection role holdout_sample),
        kept in memory in compact form (gather_rows pieces).  Per prompt: the precomputed baseline b(x) and
        N-sample pass rate (fixed_dataset.baseline_file; required, the targets must not depend on the step's
        sub-sample) and the probe flag (same draw as _finalize_fixed_bank).  Prompts are ordered non-probe first,
        probe last, so the step's last update chunk holds the probe rows.  Returns (state, step-0 metrics)."""
        from verl.trainer.ppo import kimi_master_bank as mb
        from verl.trainer.ppo.kimi_fixed_bank import lookup_master_baseline

        fd = self.config.trainer.fixed_dataset
        assert fd.get("master_selection", None), "fixed_dataset.rotate_k needs a master-bank run (master_selection)"
        assert fd.get("baseline_file", None), (
            "fixed_dataset.rotate_k needs fixed_dataset.baseline_file (per-prompt baselines that do not depend on "
            "the step's sub-sample)"
        )
        assert fd.get("anchor", "ref") == "ref", "fixed_dataset.rotate_k supports anchor=ref only"
        with open(fd.master_selection) as f:
            sel = json.load(f)
        master_dir = sel.get("master_dir") or fd.master_dir
        man = mb.load_manifest(master_dir)
        pool_n = int(fd.get("rotate_pool", None) or man["n"])
        assert 0 < pool_n <= man["n"], f"rotate_pool={pool_n} outside 1..{man['n']}"
        k = int(fd.rotate_k)
        hs_prompts = {int(p) for p in sel["holdout_sample"]}
        hs_samples = {int(s) for s in sel["samples"]["holdout_sample"]}
        train = np.array([int(p) for p in sel["train"]], dtype=np.int64)
        pools = [
            np.array([s for s in range(pool_n) if not (p in hs_prompts and s in hs_samples)], dtype=np.int64)
            for p in train
        ]
        for p in pools:
            assert k >= len(p) or len(p) % k == 0, f"pool of {len(p)} samples is not a multiple of rotate_k={k}"
        tau = float(self.config.actor_rollout_ref.actor.policy_loss.kimi_tau)
        b, rbar, info = lookup_master_baseline(
            fd.baseline_file, train, fd.get("baseline", "mean"), tau, master_dir=master_dir
        )
        n_probe = int(fd.get("probe_prompts", 0) or 0)
        uids = np.array([f"train:{p}" for p in train], dtype=object)
        probe = np.zeros(len(train), dtype=bool)
        if n_probe > 0:
            rng = np.random.default_rng(int(fd.get("seed", 0)))
            train_uids = np.unique(uids.astype(str))
            chosen = set(rng.choice(train_uids, size=min(n_probe, len(train_uids)), replace=False).tolist())
            probe = np.array([u in chosen for u in uids.astype(str)])
        order = np.concatenate([np.flatnonzero(~probe), np.flatnonzero(probe)])
        t0 = time.time()
        pieces = mb.gather_rows(
            master_dir, [(int(train[i]), pools[i].tolist()) for i in order], anchor=fd.get("master_anchor", None)
        )["pieces"]
        rot = {
            "k": k,
            "seed": int(fd.get("seed", 0)),
            "max_rows": int(fd.get("max_update_rows", 0) or 0),
            "pieces": pieces,
            "pools": [pools[i] for i in order],
            "b": b[order].astype(np.float64),
            "rbar": rbar[order].astype(np.float64),
            "probe": probe[order],
            "uids": uids[order],
            "n_nonprobe": int((~probe).sum()),
        }
        # step-0 summary of the pool (all rows any step can draw)
        pool_rows = np.array([len(p) for p in rot["pools"]])
        scores = np.concatenate([p["score"].double().numpy() for p in pieces])
        brow = np.repeat(rot["b"], pool_rows)
        tgt = (scores - brow) / tau
        resp_tok = np.array([sum(len(r) for r in p["responses"]) for p in pieces])
        per_step = [min(k, n) for n in pool_rows]
        m = {
            "rotation/k": k,
            "rotation/pool_samples": pool_n,
            "rotation/train_prompts": len(train),
            "rotation/pool_rows": int(pool_rows.sum()),
            "rotation/rows_per_step": int(sum(per_step)),
            "rotation/steps_per_epoch": int(max(n // min(k, n) for n in pool_rows)),
            "rotation/pool_reward_mean": float(scores.mean()),
            "rotation/pool_response_length_mean": float(resp_tok.sum() / pool_rows.sum()),
            "rotation/pool_target_rms": float(np.sqrt((tgt**2).mean())),
            "rotation/pool_loss_at_zero_delta": float(((scores - brow) ** 2).mean()),
            "rotation/pool_load_s": time.time() - t0,
            **{f"rotation/baseline_{kk}": v for kk, v in info.items()},
        }
        print(f"sub-sampled training pool: {m}")
        return rot, m

    def _rotation_chunks(self, rot: dict, step: int, tracked: DataProto):
        """Yield (rows, n_update_rows, is_last) for the update chunks of ``step``: the step's sub-sample of the
        pool (kimi_master_bank.rotation_samples), split by prompts into chunks of at most max_update_rows rows
        (non-probe prompts; the probe prompts and the tracked held-out rows join the last chunk).  Each chunk is
        unpacked only when requested, with A = r - b(x) per row."""
        from verl.trainer.ppo import kimi_master_bank as mb

        chosen = mb.rotation_samples(rot["pools"], rot["k"], step, rot["seed"])
        rows_per = np.array([len(c) for c in chosen], dtype=np.int64)
        total = int(rows_per.sum())
        n_np = rot["n_nonprobe"]
        # greedy split of the non-probe prompts into chunks of at most max_rows rows; the probe prompts join the
        # last chunk (so it can exceed max_rows by the probe rows)
        ranges, lo, acc = [], 0, 0
        for i in range(n_np):
            if rot["max_rows"] > 0 and acc > 0 and acc + rows_per[i] > rot["max_rows"]:
                ranges.append((lo, i))
                lo, acc = i, 0
            acc += int(rows_per[i])
        ranges.append((lo, len(chosen)))  # last chunk: remaining non-probe prompts + the probe prompts
        for ci, (lo, hi) in enumerate(ranges):
            idx = range(lo, hi)
            pieces = [mb.subset_piece(rot["pieces"][i], chosen[i]) for i in idx]
            roles = [(0, rot["uids"][i], rot["uids"][i]) for i in idx]
            rows = self._master_dataproto(pieces, roles)
            reps = rows_per[lo:hi]
            b = np.repeat(rot["b"][lo:hi], reps)
            seq_r = rows.batch["token_level_scores"].sum(-1).double().numpy()
            adv = torch.from_numpy(seq_r - b).float().unsqueeze(-1) * rows.batch["response_mask"].float()
            rows.batch["advantages"] = adv
            rows.batch["returns"] = adv.clone()
            rows.non_tensor_batch["ref_pass_rate"] = np.repeat(rot["rbar"][lo:hi], reps).astype(np.float32)
            rows.non_tensor_batch["kimi_probe"] = np.repeat(rot["probe"][lo:hi], reps).astype(np.int64)
            rows.non_tensor_batch["kimi_baseline"] = b.astype(np.float32)
            n_update = len(rows)
            last = ci == len(ranges) - 1
            if last and len(tracked):
                rows = DataProto.concat([rows, tracked])
            rows.meta_info = {}
            yield rows, n_update, total, last

    def _save_hf_weights(self):
        path = os.path.join(self.config.trainer.default_local_dir, f"hf_step_{self.global_steps}")
        self.actor_rollout_wg.save_hf_model(path)

    def _fit_fixed_dataset(self, logger):
        """Training loop of the fixed-dataset study: every step is one pass of ``update_actor`` over the same
        bank (built at step 0 from the reference policy or loaded from ``trainer.fixed_dataset.path``), with
        ``old_log_probs`` frozen at the reference log-probs (anchor=ref) or recomputed each step
        (anchor=current); then validation / checkpoints as in ``fit``."""
        fd = self.config.trainer.fixed_dataset
        path = fd.get("path", None)
        build_only = bool(fd.get("build_only", False))
        if fd.get("master_dir", None) and fd.get("master_anchor_build", False):
            self._build_master_anchor(logger)
            if hasattr(logger, "finish"):
                logger.finish()
            return
        if fd.get("master_dir", None) and build_only:
            self._build_master_bank(logger)
            if hasattr(logger, "finish"):
                logger.finish()
            return
        rot = None
        if fd.get("master_selection", None) and int(fd.get("rotate_k", 0) or 0) > 0:
            # sub-sampled training: the bank holds the fixed held-out rows; training rows are drawn per step
            bank, bank_metrics = self._load_master_bank(role_names=("holdout_prompt", "holdout_sample"))
            rot, rot_metrics = self._setup_rotation()
            bank_metrics.update(rot_metrics)
            if self.global_steps == 0:
                logger.log(data=bank_metrics, step=0)
        elif fd.get("master_selection", None):
            bank, bank_metrics = self._load_master_bank()
            if self.global_steps == 0:
                logger.log(data=bank_metrics, step=0)
        elif not path:
            raise ValueError("trainer.fixed_dataset needs path (pickled bank) or master_selection (master bank)")
        elif os.path.exists(path):
            print(f"Loading fixed bank from {path}")
            bank = DataProto.load_from_disk(path)
            if build_only:
                print("fixed bank exists and build_only=True: nothing to do")
                return
        else:
            bank = self._build_fixed_bank(path, logger)
            if build_only:
                if hasattr(logger, "finish"):
                    logger.finish()
                return
        anchor = fd.get("anchor", "ref")
        assert anchor in ("ref", "current"), f"unknown fixed_dataset.anchor={anchor}"
        holdout_prefixes = {1: "kimi_holdout_prompt", 2: "kimi_holdout_sample"}
        flags_all = bank.non_tensor_batch["kimi_holdout"].astype(np.int64)
        print(
            f"fixed bank: {len(bank)} rows, {int((flags_all == 0).sum())} training "
            f"({int(bank.non_tensor_batch['kimi_probe'].sum())} probe), "
            f"{int((flags_all == 1).sum())} held-out fresh-prompt, {int((flags_all == 2).sum())} held-out "
            f"fresh-sample rows; anchor={anchor}"
        )
        bank.meta_info = {}
        baseline_metrics = self._rebaseline_fixed_bank(bank, fd.get("baseline", "mean"))
        if baseline_metrics and self.global_steps == 0:
            logger.log(data=baseline_metrics, step=0)
        hf_freq = int(fd.get("hf_save_freq", 0) or 0)
        hf_dense = int(fd.get("hf_save_dense_until", 0) or 0)

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training (fixed bank)")
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0
        while self.global_steps <= self.total_training_steps:
            metrics: dict = {}
            timing_raw: dict = {}
            is_last_step = self.global_steps >= self.total_training_steps
            with marked_timer("step", timing_raw):
                if rot is not None:
                    batch, train_rows = self._rotation_update(rot, bank, holdout_prefixes, metrics, timing_raw)
                else:
                    batch, train_rows = self._fixed_bank_update(bank, anchor, holdout_prefixes, metrics, timing_raw)
                # global solve ratios: every rank holds an equal share of each tracked set, so the rank-mean
                # of reg_loss is the set's loss; the per-rank ratios averaged by reduce_metrics are not
                # (a rank whose rows all have zero advantage has loss_before = 0)
                for pfx in ("kimi_probe", *holdout_prefixes.values()):
                    lb = metrics.get(f"{pfx}/reg_loss_before")
                    la = metrics.get(f"{pfx}/reg_loss_after")
                    l0 = metrics.get(f"{pfx}/reg_loss_at_zero_delta_before")
                    if lb is None or la is None or l0 is None:
                        continue
                    metrics[f"{pfx}/solve_ratio"] = la / max(lb, 1e-12)  # this step
                    metrics[f"{pfx}/solve_ratio_vs_zero_delta"] = la / max(l0, 1e-12)  # vs theta_ref (anchor=ref)
                    metrics[f"{pfx}/recompute_floor_share"] = (lb - l0) / max(l0, 1e-12)

            if (
                self.val_reward_fn is not None
                and self.config.trainer.test_freq > 0
                and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
            ):
                with marked_timer("testing", timing_raw, color="green"):
                    val_metrics: dict = self._validate()
                if self._val_reached_stop_threshold(val_metrics):
                    is_last_step = True
                    metrics["training/stopped_at_val_threshold"] = 1
                if is_last_step:
                    last_val_metrics = val_metrics
                metrics.update(val_metrics)

            if self.config.trainer.save_freq > 0 and (
                is_last_step or self.global_steps % self.config.trainer.save_freq == 0
            ):
                with marked_timer("save_checkpoint", timing_raw, color="green"):
                    self._save_checkpoint()
            if (hf_freq > 0 and self.global_steps % hf_freq == 0) or self.global_steps <= hf_dense:
                with marked_timer("save_hf", timing_raw, color="green"):
                    self._save_hf_weights()

            self.max_steps_duration = max(self.max_steps_duration, timing_raw["step"])
            metrics.update({"training/global_step": self.global_steps, "training/epoch": self.global_steps})
            if rot is not None:  # data metrics of all chunks (the last chunk alone is a part of the step)
                metrics.update(self._rotation_data_metrics)
            else:
                metrics.update(compute_data_metrics(batch=train_rows, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=train_rows, timing_raw=timing_raw))
            n_gpus = self.resource_pool_manager.get_n_gpus()
            metrics.update(compute_throughout_metrics(batch=train_rows, timing_raw=timing_raw, n_gpus=n_gpus))
            logger.log(data=metrics, step=self.global_steps)

            progress_bar.update(1)
            self.global_steps += 1
            if is_last_step:
                pprint(f"Final validation metrics: {last_val_metrics}")
                progress_bar.close()
                if hasattr(logger, "finish"):
                    logger.finish()
                return

    def _update_meta(self, batch: DataProto, holdout_prefixes: dict) -> None:
        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
        batch.meta_info["kimi_holdout_prefixes"] = holdout_prefixes
        batch.meta_info["kimi_frozen_anchor"] = True  # stored old_log_probs, no on-policy shortcut
        batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
        # the main loop inherits these from generate_sequences' meta_info; the bank carries none
        batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

    def _rotation_update(self, rot: dict, tracked: DataProto, holdout_prefixes: dict, metrics: dict, timing_raw: dict):
        """One optimizer step of sub-sampled training: the step's chunks go through update_actor in turn,
        accumulating gradients (kimi_accum_*: loss scaled by the chunk's share of the step's training rows, zero_grad
        on the first chunk, optimizer step + tracked-row evaluation on the last).  Chunk metrics are merged: the
        exact loss is summed over chunks, other per-micro-batch means are row-weighted, and the tracked/grad-norm
        metrics come from the last chunk.  Returns (last chunk, all training rows' data-metric batch)."""
        step = self.global_steps
        parts: list[tuple[int, dict]] = []
        data_parts: list[tuple[int, dict]] = []
        tok_total: list[int] = []
        n_chunks = 0
        with marked_timer("layout", timing_raw):
            gen = self._rotation_chunks(rot, step, tracked)
        while True:
            with marked_timer("layout", timing_raw):
                nxt = next(gen, None)
                if nxt is None:
                    break
                rows, n_update, total, last = nxt
                mini_rows = int(self.config.actor_rollout_ref.actor.ppo_mini_batch_size) * int(
                    self.config.actor_rollout_ref.rollout.n
                )
                assert n_update <= mini_rows, (
                    f"update chunk of {n_update} rows exceeds ppo_mini_batch_size x rollout.n = {mini_rows}: it would "
                    "take several optimizer steps; lower fixed_dataset.max_update_rows or raise ppo_mini_batch_size"
                )
                batch = self._layout_fixed_bank(rows, metrics)
                del rows
                self._update_meta(batch, holdout_prefixes)
                batch.meta_info["kimi_accum_scale"] = n_update / total
                batch.meta_info["kimi_accum_zero_grad"] = n_chunks == 0
                batch.meta_info["kimi_accum_step"] = bool(last)
            with marked_timer("update_actor", timing_raw, color="red"):
                out = self.actor_rollout_wg.update_actor(batch)
            parts.append((n_update, reduce_metrics(out.meta_info["metrics"])))
            train_rows = batch.select_idxs(batch.non_tensor_batch["kimi_holdout"] == 0)
            tok_total += torch.sum(train_rows.batch["attention_mask"], dim=-1).tolist()
            data_parts.append((len(train_rows), compute_data_metrics(batch=train_rows, use_critic=self.use_critic)))
            n_chunks += 1
            if not last:
                del batch, train_rows
        metrics.update(_merge_chunk_metrics(parts, sum_keys=("actor/loss_exact",)))
        data_m = _merge_chunk_metrics(data_parts)
        metrics["rotation/chunks"] = n_chunks
        metrics["rotation/rows"] = int(sum(n for n, _ in parts))
        per_epoch = max(len(p) // min(rot["k"], len(p)) for p in rot["pools"])
        metrics["rotation/epoch"] = (step - 1) // per_epoch
        metrics["rotation/slot"] = (step - 1) % per_epoch
        # the caller computes data/timing metrics on train_rows: hand it the last chunk with the step's token count
        train_rows.meta_info = {"global_token_num": tok_total}
        self._rotation_data_metrics = data_m
        return batch, train_rows

    def _fixed_bank_update(self, bank: DataProto, anchor: str, holdout_prefixes: dict, metrics: dict, timing_raw: dict):
        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
        with marked_timer("layout", timing_raw):
            batch = self._layout_fixed_bank(bank, metrics)
        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
        if anchor == "current":
            with marked_timer("old_log_prob", timing_raw, color="blue"):
                old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                batch.batch["old_log_probs"] = old_log_prob.batch["old_log_probs"]
                keep = torch.from_numpy(batch.non_tensor_batch["kimi_holdout"] == 0)
                metrics["actor/entropy"] = agg_loss(
                    loss_mat=old_log_prob.batch["entropys"][keep],
                    loss_mask=batch.batch["response_mask"][keep],
                    loss_agg_mode=loss_agg_mode,
                ).item()
        self._update_meta(batch, holdout_prefixes)
        with marked_timer("update_actor", timing_raw, color="red"):
            actor_output = self.actor_rollout_wg.update_actor(batch)
        metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))
        train_rows = batch.select_idxs(batch.non_tensor_batch["kimi_holdout"] == 0)
        train_rows.meta_info = {"global_token_num": torch.sum(train_rows.batch["attention_mask"], dim=-1).tolist()}
        return batch, train_rows

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        fixed_cfg = self.config.trainer.get("fixed_dataset", None)
        if fixed_cfg is not None and fixed_cfg.get("enable", False):
            self._fit_fixed_dataset(logger)
            return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                rm_scores = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(rm_scores)
                            reward_baseline_tensor, _ = compute_reward(batch, self.reward_fn)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Kimi-K2 fresh-prompt hold-out: the dataloader drew data.kimi_holdout_prompts extra prompts;
                    # their rollouts are flagged and kept out of the update (see _split_holdout_rows,
                    # _layout_holdout_rows and dp_actor.update_policy). Split off before balancing so the
                    # training rows alone are balanced across ranks.
                    batch, holdout_batch = self._split_holdout_rows(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)
                    if holdout_batch is not None:
                        # appended after balancing: the training rows keep their per-rank blocks; the
                        # reward / log-prob / advantage stages below are row-wise (advantages group by uid)
                        batch = DataProto.concat([batch, holdout_batch])

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                data=batch, config=self.config, tokenizer=self.tokenizer
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_rollout_correction

                        apply_rollout_correction(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            if "kimi_holdout" in batch.non_tensor_batch:  # training rows only
                                keep = torch.from_numpy(batch.non_tensor_batch["kimi_holdout"] == 0)
                                entropys, response_masks = entropys[keep], response_masks[keep]
                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            entropy_agg = agg_loss(
                                loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode
                            )
                            old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                        # CBS downsample-rollouts study (Q: separate σ²_intra from
                        # advantage-estimation noise). Advantages above are computed
                        # over ALL rollout.n rollouts per prompt (high-quality, ~true
                        # baseline). When trainer.downsample_update_k > 0 we then keep
                        # only K of those rollouts per prompt for the actual gradient
                        # update, carrying their rollout.n-based advantages. This holds
                        # baseline quality fixed while varying the gradient sample count.
                        downsample_k = self.config.trainer.get("downsample_update_k", 0)
                        n_roll = self.config.actor_rollout_ref.rollout.n
                        if downsample_k and 0 < downsample_k < n_roll:
                            import numpy as _np

                            uids = batch.non_tensor_batch["uid"]
                            n_prompts = len(uids) // n_roll

                            # Sort by uid so same-prompt rollouts are contiguous, then
                            # reshape to (n_prompts, n_roll) for vectorised operations.
                            sort_order = _np.argsort(uids, kind="stable")
                            grouped = sort_order.reshape(n_prompts, n_roll)  # (n_prompts, n_roll)

                            # sequence-level scalar reward and advantage (one value per response)
                            _resp_mask = batch.batch["response_mask"]
                            _seq_rewards = batch.batch["token_level_rewards"].sum(dim=-1)
                            _seq_len = _resp_mask.sum(dim=-1).clamp(min=1)
                            _seq_adv = (batch.batch["advantages"] * _resp_mask).sum(dim=-1) / _seq_len

                            # pre-downsample metrics
                            _rewards_grouped = _seq_rewards.cpu().numpy()[sort_order].reshape(n_prompts, n_roll)
                            _prompt_stds = _rewards_grouped.std(axis=1)
                            metrics["downsample/pre/mean_reward"] = float(_seq_rewards.mean().item())
                            metrics["downsample/pre/adv_magnitude"] = float(_seq_adv.abs().mean().item())
                            metrics["downsample/pre/zero_sigma_frac"] = float((_prompt_stds < 1e-6).mean())
                            # Heaviest advantage in the batch. With N-sample normalisation the
                            # positive tail reaches sqrt((N-k)/k) for k successes -- ~7.9 at N=64
                            # vs ~3.9 at N=16 -- so this tracks the extra-weight "rare success"
                            # samples that plain GRPO-16 would never see.
                            metrics["downsample/pre/max_abs_adv"] = float(_seq_adv.abs().max().item())

                            # select K rollouts per prompt: shuffle each row, take first K columns
                            _rng = _np.random.default_rng(self.global_steps)
                            sel_grouped = _rng.permuted(grouped, axis=1)[:, :downsample_k]  # (n_prompts, K)
                            sel = _np.sort(sel_grouped.ravel())

                            # Lost-cancellation term. Full GRPO has sum_i A_i == 0 within every
                            # prompt; the kept subset does not, and its per-prompt mean advantage
                            # is a random prompt-wide push with variance ~ (1/K)(1-K/N) on
                            # non-degenerate prompts (0.047 -> std 0.217 at K=16, N=64). This is
                            # the sigma^2_intra component the study is after; log its realised std.
                            _prompt_mean_adv = _seq_adv.cpu().numpy()[sel_grouped].mean(axis=1)  # (n_prompts,)
                            _nondeg = _prompt_stds >= 1e-6
                            metrics["downsample/post/prompt_mean_adv_std"] = float(_prompt_mean_adv.std())
                            metrics["downsample/post/prompt_mean_adv_std_nondeg"] = (
                                float(_prompt_mean_adv[_nondeg].std()) if _nondeg.sum() > 1 else 0.0
                            )

                            batch = batch.select_idxs(sel)

                            # post-downsample metrics
                            _resp_mask_post = batch.batch["response_mask"]
                            _seq_rewards_post = batch.batch["token_level_rewards"].sum(dim=-1)
                            _seq_adv_post = (batch.batch["advantages"] * _resp_mask_post).sum(dim=-1) / _resp_mask_post.sum(dim=-1).clamp(min=1)
                            metrics["downsample/post/mean_reward"] = float(_seq_rewards_post.mean().item())
                            metrics["downsample/post/adv_magnitude"] = float(_seq_adv_post.abs().mean().item())
                            metrics["downsample/post/max_abs_adv"] = float(_seq_adv_post.abs().max().item())
                            metrics["downsample/k"] = downsample_k
                            metrics["downsample/n_rollouts_generated"] = n_roll
                            metrics["downsample/rows_after"] = len(sel)

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        if "kimi_holdout" in batch.non_tensor_batch:
                            self._layout_holdout_rows(batch)
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        # Gradient noise measurement for CBS study:
                        # Extract per-mini-batch grad norms BEFORE reduce_metrics averages them.
                        grad_noise_freq = self.config.trainer.get("grad_noise_measure_freq", 0)
                        if grad_noise_freq > 0 and self.global_steps % grad_noise_freq == 0:
                            raw_actor_metrics = actor_output.meta_info["metrics"]
                            grad_norms = raw_actor_metrics.get("actor/grad_norm", [])
                            if isinstance(grad_norms, list) and len(grad_norms) >= 2:
                                import numpy as _np
                                gn = _np.array(grad_norms)
                                gn_sq = gn ** 2
                                mean_gn_sq = _np.mean(gn_sq)
                                approx_full_gn_sq = _np.mean(gn) ** 2
                                mini_bs = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
                                if approx_full_gn_sq > 1e-12:
                                    b_noise = float(mini_bs * (mean_gn_sq / approx_full_gn_sq - 1))
                                    metrics["grad_noise/B_noise_estimate"] = max(b_noise, 0.0)
                                metrics["grad_noise/grad_norm_mean"] = float(_np.mean(gn))
                                metrics["grad_noise/grad_norm_std"] = float(_np.std(gn))
                                metrics["grad_noise/grad_norm_cv"] = float(_np.std(gn) / _np.mean(gn))
                                metrics["grad_noise/n_mini_batches"] = len(grad_norms)

                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                    # Kimi-K2 hold-out: log the held-out rollouts' reward and drop them, so the batch metrics
                    # below (train reward, response length, ...) describe the training rows only.
                    if "kimi_holdout" in batch.non_tensor_batch:
                        batch = self._drop_holdout_rows(batch, metrics)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                    if self._val_reached_stop_threshold(val_metrics):
                        is_last_step = True   # save the final checkpoint and return via the last-step path below
                        metrics["training/stopped_at_val_threshold"] = 1
                    if is_last_step:
                        last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    if hasattr(logger, "finish"):
                        logger.finish()   # flush the last steps before Ray tears the actor down
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
