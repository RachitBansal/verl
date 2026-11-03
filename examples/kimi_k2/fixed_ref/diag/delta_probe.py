"""Offline check of the 'one Adam step moves Delta by ~0.5 nats at any lr' observation.

Computes per-sequence summed log-probs (HF, bf16, one unpadded sequence at a time) of a fixed set of master-bank
rows under several weight sets and reports rms_i(sum_t logp_X - sum_t logp_base) over rows, plus the base vs
stored-anchor recompute floor.  Weight sets: base, HF snapshots (--snap name=path), FSDP fp32 masters merged to
a state dict (--master name=path.pt), and synthetic perturbations base_fp32 + lr * random_sign cast to bf16
(--rand-lrs).
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
from verl.trainer.ppo import kimi_master_bank as mb  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="/n/netscratch/sham_lab/Everyone/cmohri/rl_cbs/models/Qwen2.5-Math-1.5B-Instruct")
ap.add_argument("--selection", required=True)
ap.add_argument("--prompts", type=int, default=24)
ap.add_argument("--samples", type=int, default=2)
ap.add_argument("--snap", action="append", default=[])
ap.add_argument("--master", action="append", default=[])
ap.add_argument("--rand-lrs", default="1e-7,1e-6,1e-5")
ap.add_argument("--max-resp", type=int, default=1500)
ap.add_argument("--compute", default="bf16", choices=["bf16", "fp32", "tf32"])
ap.add_argument("--attn", default="sdpa", choices=["sdpa", "flash", "flash16"])
args = ap.parse_args()

sel = json.load(open(args.selection))
reqs = [(p, list(range(args.samples))) for p in sel["train"][: args.prompts]]
pieces = mb.gather_rows(sel["master_dir"], reqs)["pieces"]
rows = []
for pc in pieces:
    for resp, ref in zip(pc["responses"], pc["ref_log_probs"]):
        if len(resp) <= args.max_resp:
            rows.append((pc["prompt"].long(), resp.long(), ref.float()))
print(f"{len(rows)} rows, mean response length {np.mean([len(r[1]) for r in rows]):.0f}", flush=True)
dev = "cuda"
CDT = torch.bfloat16 if args.compute == "bf16" else torch.float32
torch.backends.cuda.matmul.allow_tf32 = args.compute == "tf32"
torch.backends.cudnn.allow_tf32 = args.compute == "tf32"
print("compute", args.compute, "attn", args.attn, flush=True)
if args.attn in ("flash", "flash16"):
    KDT = torch.float16 if args.attn == "flash16" else torch.bfloat16
    # same cast as verl's patched _flash_attention_forward: fp32 q/k/v -> bf16 for the kernel, output back to fp32
    from transformers.integrations import flash_attention as _fa

    _orig_fa = _fa._flash_attention_forward

    def _fa_cast(q, k, v, *a, **kw):
        dt = q.dtype
        if dt == torch.float32:
            q, k, v = q.to(KDT), k.to(KDT), v.to(KDT)
            kw["target_dtype"] = None
        return _orig_fa(q, k, v, *a, **kw).to(dt)

    _fa._flash_attention_forward = _fa_cast


@torch.no_grad()
def seq_logps(model):
    import time

    torch.cuda.synchronize()
    t0 = time.time()
    out = []
    for p, r, _ in rows:
        ids = torch.cat([p, r]).to(dev)[None]
        logits = model(input_ids=ids).logits[0, len(p) - 1 : -1].float()
        lp = torch.log_softmax(logits, -1).gather(-1, r.to(dev)[:, None])[:, 0]
        out.append(lp.cpu())
    torch.cuda.synchronize()
    print(f"    [forward {1000 * (time.time() - t0) / len(rows):.1f} ms/row]", flush=True)
    return out


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=CDT, attn_implementation="flash_attention_2" if args.attn != "sdpa" else "sdpa").to(dev)
model.eval()
base_sd = {k: v.detach().float().cpu().clone() for k, v in model.state_dict().items()}
base = seq_logps(model)
base_sum = np.array([x.sum().item() for x in base])
ref_sum = np.array([r[2].sum().item() for r in rows])
tok = np.concatenate([(b - r[2]).numpy() for b, r in zip(base, rows)])
print(f"base vs stored anchor: seq-sum rms {rms(base_sum - ref_sum):.4f}  token rms {rms(tok):.2e}", flush=True)


def load_into(sd_fp32):
    model.load_state_dict({k: v.to(CDT) for k, v in sd_fp32.items()}, strict=False)


def report(name, sd_fp32):
    load_into(sd_fp32)
    nch = sum(int((sd_fp32[k].to(torch.bfloat16).float() != base_sd[k]).sum()) for k in base_sd if k in sd_fp32)
    ntot = sum(base_sd[k].numel() for k in base_sd if k in sd_fp32)
    s = np.array([x.sum().item() for x in seq_logps(model)])
    print(f"{name:40s} delta rms {rms(s - base_sum):.4f} mean {np.mean(s - base_sum):+.4f}  "
          f"bf16 params changed {nch / ntot:.4%}", flush=True)


g = torch.Generator().manual_seed(0)
for lr in [float(x) for x in args.rand_lrs.split(",") if x]:
    sd = {k: v + lr * (torch.randint(0, 2, v.shape, generator=g).float() * 2 - 1) for k, v in base_sd.items()}
    report(f"random sign lr={lr:g}", sd)
for spec in args.snap:
    name, path = spec.split("=", 1)
    m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32)
    sd = {k: v.float() for k, v in m.state_dict().items() if k in base_sd}
    del m
    report(f"snap {name}", sd)
for spec in args.master:
    name, path = spec.split("=", 1)
    sd = torch.load(path, map_location="cpu")
    sd = {k: v.float() for k, v in sd.items() if k in base_sd}
    d = {k: (sd[k] - base_sd[k]) for k in sd}
    absd = torch.cat([x.abs().flatten() for x in d.values()])
    print(f"master {name}: fp32 |w - w0| mean {absd.mean():.3e} median {absd.median():.3e} max {absd.max():.3e}")
    report(f"master {name} (bf16 cast)", sd)
model.load_state_dict({k: v.to(CDT) for k, v in base_sd.items()})
s = np.array([x.sum().item() for x in seq_logps(model)])
print(f"base reloaded (determinism check) delta rms {rms(s - base_sum):.2e}")
