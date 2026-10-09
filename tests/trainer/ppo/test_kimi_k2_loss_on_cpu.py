# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""CPU checks for the Kimi-K2 squared-regression policy loss."""

from types import SimpleNamespace

import pytest
import torch

from verl.trainer.ppo.core_algos import (
    compute_policy_loss_kimi_k2,
    get_policy_loss_fn,
    kimi_k2_regression_stats,
)
from verl.workers.config import PolicyLossConfig


def _cfg(**kw):
    pl = PolicyLossConfig(loss_mode="kimi_k2", **kw)
    return SimpleNamespace(policy_loss=pl)


def _batch(B=6, T=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    old = -torch.rand(B, T, generator=g)
    mask = torch.ones(B, T)
    mask[0, 3:] = 0  # one short response
    mask[1, 4:] = 0
    adv_seq = torch.tensor([0.5, -0.5, 0.0, 0.25, -0.25, 1.0])  # A_i = r_i - rbar, constant over tokens
    adv = adv_seq[:, None].expand(B, T) * mask
    return old, adv, mask, adv_seq


def test_registered():
    assert get_policy_loss_fn("kimi_k2") is compute_policy_loss_kimi_k2


def test_loss_at_old_policy_equals_mean_sq_advantage():
    old, adv, mask, adv_seq = _batch()
    log_prob = old.clone().requires_grad_(True)
    loss, m = compute_policy_loss_kimi_k2(old, log_prob, adv, mask, config=_cfg(kimi_tau=0.1))
    assert torch.allclose(loss, adv_seq.pow(2).mean())
    assert m["kimi/delta_rms"] == pytest.approx(0.0)
    assert m["kimi/reg_loss"] == pytest.approx(adv_seq.pow(2).mean().item())


def test_gradient_at_old_policy_is_scaled_policy_gradient():
    """d/dlogp of (A - tau*sum d)^2 at d=0 is -2 tau A on every response token: a mean-baseline PG step."""
    old, adv, mask, adv_seq = _batch()
    tau = 0.1
    log_prob = old.clone().requires_grad_(True)
    loss, _ = compute_policy_loss_kimi_k2(old, log_prob, adv, mask, config=_cfg(kimi_tau=tau))
    loss.backward()
    B = old.shape[0]
    expected = (-2.0 * tau * adv_seq[:, None] * mask) / B  # mean over sequences
    assert torch.allclose(log_prob.grad, expected, atol=1e-6)

    # gradient form: same direction, rescaled by 1/(2 tau)
    log_prob2 = old.clone().requires_grad_(True)
    loss2, _ = compute_policy_loss_kimi_k2(old, log_prob2, adv, mask, config=_cfg(kimi_tau=tau, kimi_form="gradient"))
    loss2.backward()
    assert torch.allclose(log_prob2.grad, expected / (2 * tau), atol=1e-6)


def test_minimizer_is_mirror_descent_iterate():
    """Delta_i = A_i / tau gives zero loss, s = rho = 1."""
    old, adv, mask, adv_seq = _batch()
    tau = 0.05
    seq_len = mask.sum(-1)
    d = (adv_seq / tau / seq_len)[:, None] * mask  # spread the target uniformly over tokens
    log_prob = old + d
    loss, m = compute_policy_loss_kimi_k2(old, log_prob, adv, mask, config=_cfg(kimi_tau=tau))
    assert loss.item() == pytest.approx(0.0, abs=1e-10)
    assert m["kimi/s"] == pytest.approx(1.0, abs=1e-6)
    assert m["kimi/rho"] == pytest.approx(1.0, abs=1e-6)
    assert m["kimi/target_rms"] == pytest.approx((adv_seq / tau).pow(2).mean().sqrt().item(), rel=1e-6)


def test_seq_mean_level():
    old, adv, mask, adv_seq = _batch()
    tau = 0.5
    st = kimi_k2_regression_stats(old, old + 0.3 * mask, adv, mask, tau=tau, target_level="seq_mean")
    assert torch.allclose(st["delta"], torch.full_like(st["delta"], 0.3))
    st2 = kimi_k2_regression_stats(old, old + 0.3 * mask, adv, mask, tau=tau, target_level="seq_sum")
    assert torch.allclose(st2["delta"], 0.3 * mask.sum(-1))


def test_logratio_clip_detaches_large_tokens():
    old, adv, mask, adv_seq = _batch()
    d = 0.1 * mask
    d[2, 0] = 5.0  # one extreme token
    log_prob = (old + d).requires_grad_(True)
    loss, _ = compute_policy_loss_kimi_k2(old, log_prob, adv, mask, config=_cfg(kimi_tau=0.1, kimi_logratio_clip=1.0))
    loss.backward()
    assert log_prob.grad[2, 0].item() == 0.0
    assert log_prob.grad[2, 1].item() != 0.0


def test_overshoot_is_visible_in_metrics():
    """Moving 10x past the target must give solve_ratio-like signal: s ~ 10, resid rms ~ 9x target rms."""
    old, adv, mask, adv_seq = _batch()
    tau = 0.1
    seq_len = mask.sum(-1)
    d = (10 * adv_seq / tau / seq_len)[:, None] * mask
    _, m = compute_policy_loss_kimi_k2(old, old + d, adv, mask, config=_cfg(kimi_tau=tau))
    assert m["kimi/s"] == pytest.approx(10.0, rel=1e-5)
    assert m["kimi/rho"] == pytest.approx(1.0, abs=1e-6)
    assert m["kimi/reg_loss"] > m["kimi/reg_loss_at_zero_delta"]
