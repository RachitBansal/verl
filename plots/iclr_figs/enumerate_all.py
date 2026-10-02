import wandb, json, sys
api = wandb.Api(timeout=120)
projects = ["grpo_on_policy_cbs", "grpo_on_policy_cbs_updated_scaling", "cbs_rlvr", "cbs_long_runs", "cbs"]
out = open(sys.argv[1], "w")
for p in projects:
    try:
        runs = list(api.runs(f"harvardml/{p}", per_page=500))
    except Exception as e:
        print("ERR", p, e); continue
    print(p, len(runs))
    for r in runs:
        cfg = r.config or {}
        def g(path, default=None):
            d = cfg
            for k in path.split("."):
                if isinstance(d, dict) and k in d: d = d[k]
                else: return default
            return d
        s = r.summary or {}
        rec = dict(project=p, name=r.name, id=r.id, state=r.state, steps=s.get("_step"),
                   created=str(r.created_at), n=g("actor_rollout_ref.rollout.n"),
                   bsz=g("data.train_batch_size"), lr=g("actor_rollout_ref.actor.optim.lr"),
                   kl=g("actor_rollout_ref.actor.kl_loss_coef"), mini=g("actor_rollout_ref.actor.ppo_mini_batch_size"),
                   clip=g("actor_rollout_ref.actor.clip_ratio"), clip_high=g("actor_rollout_ref.actor.clip_ratio_high"),
                   dsk=g("trainer.downsample_update_k"), max_resp=g("data.max_response_length"), max_prompt=g("data.max_prompt_length"),
                   epochs=g("trainer.total_epochs"), model=g("actor_rollout_ref.model.path"),
                   loss_agg=g("actor_rollout_ref.actor.loss_agg_mode"), loss_mode=g("actor_rollout_ref.actor.policy_loss.loss_mode"),
                   resp_len_mean=s.get("response_length/mean"), prompt_len_mean=s.get("prompt_length/mean"),
                   val_keys=[k for k in s.keys() if k.startswith("val")], group=r.group, tags=list(r.tags or []))
        out.write(json.dumps(rec) + "\n")
out.close()
