"""
Gradient noise scale measurement for RLVR critical batch size studies.

This module provides a callback-style measurement that can be injected into
the verl training loop to estimate B_noise (the gradient noise scale) without
modifying the core verl codebase.

The measurement is done by computing per-mini-batch gradients from the SAME
model state (no optimizer step between them), then estimating:

    B_noise = b * trace(Cov(g_k)) / ||mean(g_k)||^2

where g_k is the gradient from mini-batch k of size b.

Usage:
    In the training loop (ray_trainer.py), after the batch is constructed:

        if global_steps % NOISE_MEASURE_FREQ == 0:
            noise_metrics = measure_gradient_noise(actor_wg, batch, config)
            metrics.update(noise_metrics)
"""

import torch
import numpy as np
from typing import Any


def estimate_b_noise_from_grad_norms(
    per_mb_grad_norms_sq: list[float],
    full_batch_grad_norm_sq: float,
    mini_batch_size: int,
    n_mini_batches: int,
) -> dict[str, float]:
    """
    Estimate B_noise from per-mini-batch and full-batch gradient norms.

    Uses the decomposition:
        E[||g_k||^2] = ||g||^2 + (1/b) * trace(Sigma)

    where b is the mini-batch size, g is the true gradient, and Sigma is the
    per-sample gradient covariance. Therefore:

        trace(Sigma) = b * (mean(||g_k||^2) - ||g||^2)
        B_noise = trace(Sigma) / ||g||^2
                = b * (mean(||g_k||^2) / ||g||^2 - 1)

    Args:
        per_mb_grad_norms_sq: List of ||g_k||^2 for each mini-batch.
        full_batch_grad_norm_sq: ||g_full||^2 from the full-batch gradient.
        mini_batch_size: Number of samples in each mini-batch.
        n_mini_batches: Number of mini-batches.

    Returns:
        Dictionary with B_noise estimate and diagnostic metrics.
    """
    mean_mb_norm_sq = np.mean(per_mb_grad_norms_sq)
    var_mb_norm = np.var(per_mb_grad_norms_sq)

    if full_batch_grad_norm_sq < 1e-12:
        return {
            "grad_noise/B_noise_estimate": float("nan"),
            "grad_noise/full_batch_grad_norm": 0.0,
            "grad_noise/mean_mb_grad_norm": np.sqrt(mean_mb_norm_sq),
            "grad_noise/n_mini_batches": n_mini_batches,
        }

    # B_noise estimate: b * (E[||g_k||^2] / ||g||^2 - 1)
    b_noise = mini_batch_size * (mean_mb_norm_sq / full_batch_grad_norm_sq - 1)
    b_noise = max(b_noise, 0.0)  # can be slightly negative due to noise

    return {
        "grad_noise/B_noise_estimate": b_noise,
        "grad_noise/full_batch_grad_norm": np.sqrt(full_batch_grad_norm_sq),
        "grad_noise/mean_mb_grad_norm": np.sqrt(mean_mb_norm_sq),
        "grad_noise/grad_norm_ratio": np.sqrt(mean_mb_norm_sq / full_batch_grad_norm_sq),
        "grad_noise/mb_grad_norm_std": np.sqrt(var_mb_norm),
        "grad_noise/n_mini_batches": n_mini_batches,
        "grad_noise/mini_batch_size": mini_batch_size,
    }


def estimate_b_noise_from_training_metrics(
    metrics: dict[str, list[Any]],
    mini_batch_size: int,
) -> dict[str, float]:
    """
    Estimate B_noise from already-collected training metrics.

    This uses the per-mini-batch gradient norms that verl already logs during
    update_policy. This is an APPROXIMATION since the model parameters change
    between mini-batches (optimizer steps happen). It is most accurate when
    ppo_epochs=1 and the learning rate is small.

    For a proper estimate, use measure_gradient_noise_on_batch() below.

    Args:
        metrics: The metrics dict from update_policy, containing "actor/grad_norm".
        mini_batch_size: The mini-batch size used.

    Returns:
        Dictionary with approximate B_noise estimate.
    """
    grad_norms = metrics.get("actor/grad_norm", [])
    if not isinstance(grad_norms, list):
        grad_norms = [grad_norms]

    if len(grad_norms) < 2:
        return {}

    grad_norms_sq = [g ** 2 for g in grad_norms]
    mean_norm_sq = np.mean(grad_norms_sq)
    var_norm = np.var(grad_norms_sq)

    # Approximate the full-batch gradient norm as the mean of mini-batch norms
    # (this assumes gradients are roughly aligned, which is approximate)
    # A better approximation: ||g_full||^2 ≈ (mean(||g_k||))^2
    full_norm_sq_approx = np.mean(grad_norms) ** 2

    if full_norm_sq_approx < 1e-12:
        return {"grad_noise/B_noise_approx": float("nan")}

    b_noise_approx = mini_batch_size * (mean_norm_sq / full_norm_sq_approx - 1)
    b_noise_approx = max(b_noise_approx, 0.0)

    return {
        "grad_noise/B_noise_approx": b_noise_approx,
        "grad_noise/K_mini_batches": len(grad_norms),
        "grad_noise/grad_norm_cv": np.std(grad_norms) / np.mean(grad_norms),
    }
