# ICLR Figure 2: total cost of reaching the target

`make_budget_frontier.py` re-plots every cell of the three steps-to-target sweeps (prompts at K=16,
prompts at K=64, rollouts at B=128) as the totals consumed before first reaching 50% AIME mean@1:
(a) sequences generated S·B·K against steps S, and (b) prompt draws S·B against steps S.
Dashed curves are the McCandlish fits S = S_min(1 + N*/N) in these coordinates
(E = S_min(N + N*); P = E/K for a prompt sweep, P = 128·S for the rollout sweep).

```
iclr_figs/
  enumerate_all.py          wandb run configs  -> all_runs.jsonl   (wandb-capable python)
  fetch_all.py              per-step AIME mean@1 trajectories -> traj_all.csv
  cbs_data.py               trajectories -> per-run crossings -> per-cell best LR; McCandlish fit
  make_budget_frontier.py   the figure (two-panel preview + the two paper panels _a/_b) and a per-cell table
```

Pull, then plot with the repo venv:

```bash
PULL=/n/home03/cmohri/venvs/verl_env/bin/python     # any python with wandb
PLOT=/n/home03/cmohri/team_verl/.venv/bin/python     # any python with numpy + matplotlib
cd plots
$PULL iclr_figs/enumerate_all.py csv/all_runs.jsonl
$PULL iclr_figs/fetch_all.py     csv/traj_all.csv
$PLOT iclr_figs/make_budget_frontier.py --traj csv/traj_all.csv --meta csv/all_runs.jsonl --out paper
```

Outputs: `paper/fig_budget_frontier_a.pdf`, `paper/fig_budget_frontier_b.pdf` (the panels used in the
paper), `fig_budget_frontier.{pdf,png}` (two-panel preview with titles) and `fig_budget_frontier_table.csv`
(series, B, K, sequences per step, steps, total sequences, total prompt draws, best LR).

Protocol, identical to `scripts/pull_steps_to_50*.py`: S = first crossing of 0.5, linearly interpolated
between the bracketing validation readings; resume segments stitched; per (B, K) cell the fastest run over
every learning rate and KL coefficient tried (KL 1e-3 and 1e-2 pooled). Series selection: prompts K=16 pools
pre-fix and `_updated_scaling` runs; rollouts at B=128 uses pre-fix runs only for K<=16; prompts K=64 uses
`_updated_scaling` runs, preferring the per-step-validated `_dense` reruns where they exist.
