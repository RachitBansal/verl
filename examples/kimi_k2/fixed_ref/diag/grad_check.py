"""Gradient check of the fp32-compute flash-attention path (verl monkey_patch: fp16 kernel + gradient scaling)
against an fp32 sdpa reference, at the tiny loss scale of a full-batch mean (loss * 1e-6), on a few bank rows.
Also reports forward seq-logp agreement.  Variants: ref (fp32 sdpa), patched (tf32 + fp16 flash, scaled grads),
noscale (same with scale 1), bf16attn (tf32 + bf16 flash)."""

import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM
from transformers.integrations import flash_attention as fa

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
import verl.models.transformers.monkey_patch as mp  # noqa: E402
from verl.trainer.ppo import kimi_master_bank as mb  # noqa: E402

BASE = "/n/netscratch/sham_lab/Everyone/cmohri/rl_cbs/models/Qwen2.5-Math-1.5B-Instruct"
sel = json.load(open(sys.argv[1]))
pieces = mb.gather_rows(sel["master_dir"], [(p, [0]) for p in sel["train"][:6]])["pieces"]
rows = [(pc["prompt"].long(), pc["responses"][0].long()) for pc in pieces if len(pc["responses"][0]) < 1200][:4]
dev = "cuda"
orig_fa = fa._flash_attention_forward


def run(variant):
    torch.backends.cuda.matmul.allow_tf32 = variant != "ref"
    attn = "sdpa" if variant == "ref" else "flash_attention_2"
    fa._flash_attention_forward = orig_fa
    if variant in ("patched", "noscale"):
        fa._flash_attention_forward = mp._ulysses_flash_attention_forward
        mp._FP16_ATTN_GRAD_SCALE = float(2**16) if variant == "patched" else 1.0
    elif variant == "bf16attn":
        def bf(q, k, v, *a, **kw):
            kw["target_dtype"] = None
            return orig_fa(q.bfloat16(), k.bfloat16(), v.bfloat16(), *a, **kw).float()
        fa._flash_attention_forward = bf
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32, attn_implementation=attn).to(dev)
    model.gradient_checkpointing_enable()
    model.train()
    seqs = []
    for p, r in rows:
        ids = torch.cat([p, r]).to(dev)[None]
        logits = model(input_ids=ids, use_cache=False).logits[0, len(p) - 1 : -1]
        lp = torch.log_softmax(logits.float(), -1).gather(-1, r.to(dev)[:, None])[:, 0]
        s = lp.sum()
        (s * 1e-6).backward()  # full-batch mean loss scale
        seqs.append(s.item())
    g = {n: p.grad.detach().float().cpu().clone() for n, p in model.named_parameters() if p.grad is not None}
    del model
    torch.cuda.empty_cache()
    return seqs, g


ref_s, ref_g = run("ref")
for v in ("patched", "noscale", "bf16attn"):
    s, g = run(v)
    num = sum((g[k] * ref_g[k]).sum() for k in ref_g)
    den = (sum((g[k] ** 2).sum() for k in ref_g) * sum((ref_g[k] ** 2).sum() for k in ref_g)) ** 0.5
    attn_keys = [k for k in ref_g if "q_proj" in k or "k_proj" in k]
    na = sum((g[k] * ref_g[k]).sum() for k in attn_keys)
    da = (sum((g[k] ** 2).sum() for k in attn_keys) * sum((ref_g[k] ** 2).sum() for k in attn_keys)) ** 0.5
    rel = (sum((g[k] ** 2).sum() for k in ref_g) / sum((ref_g[k] ** 2).sum() for k in ref_g)) ** 0.5
    ds = [a - b for a, b in zip(s, ref_s)]
    print(f"{v:9s} grad cos {float(num / den):.6f}  q/k-proj grad cos {float(na / da):.6f}  |g|/|g_ref| {float(rel):.4f}"
          f"  seq-logp diff vs ref {[round(x, 4) for x in ds]}", flush=True)
