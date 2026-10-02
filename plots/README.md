# CBS plots

```
scripts/   pull_*.py (wandb -> CSV) and plot_*.py (CSV -> PNG)
csv/       pulled data and the *_table.csv each plot script emits alongside its figure
png/       figures only
paper/     paper-ready PDFs (vector), written by the *_paper.py scripts
html/      interactive pages (built from csv/, published as artifacts)
```

Pull with the wandb-capable python, plot with the repo venv (has matplotlib):

```bash
PULL=/n/home03/cmohri/venvs/verl_env/bin/python
PLOT=/n/home03/cmohri/team_verl/.venv/bin/python
cd plots

# batch-size sweep (plain GRPO, n = 16)
$PULL scripts/pull_steps_to_50.py csv/steps_to_50_kl.csv "1e-3,1e-2"          # add --incremental to re-fetch only new/running runs
$PLOT scripts/plot_steps_to_50.py    csv/steps_to_50_kl.csv png/steps_to_50.png        # LR x batch grid, KL 1e-3
$PLOT scripts/plot_steps_to_50.py    csv/steps_to_50_kl.csv png/steps_to_50_n.png --by-n   # LR x rollout-count grid at batch 128
$PLOT scripts/plot_steps_to_50_kl.py csv/steps_to_50_kl.csv png/steps_to_50_kl.png     # ... KL 1e-3 vs 1e-2
$PLOT scripts/plot_steps_vs_bsz.py   csv/steps_to_50_kl.csv png/steps_vs_bsz.png       # steps to 50% vs batch
$PLOT scripts/plot_steps_vs_bsz_kl.py csv/steps_to_50_kl.csv png/steps_vs_bsz_kl.png

# downsampling vs plain GRPO on a sequences-per-step axis
$PULL scripts/pull_steps_to_50_seqs.py csv/steps_to_50_seqs.csv
$PLOT scripts/plot_steps_vs_seqs.py csv/steps_to_50_seqs.csv png/steps_vs_seqs.png # plain n=16 sweep vs downsampling; teal = plain n=64 batch sweep (fixed code)
$PLOT scripts/plot_steps_vs_seqs.py csv/steps_to_50_seqs.csv png/steps_vs_seqs_prefix.png --prefix   # + pre-fix downsample runs
$PLOT scripts/plot_steps_vs_seqs.py csv/steps_to_50_seqs.csv png/steps_vs_seqs_nsweep.png --nsweep   # + rollout sweep (bsz 128, n varied; solid = fixed-code at n=32/64, dashed = pre-fix only)
$PLOT scripts/plot_steps_vs_seqs_paper.py csv/steps_to_50_seqs.csv png/steps_vs_seqs_paper.png   # paper figure: n=16 batch sweep, rollout sweep, n=64 batch sweep (PNG + PDF)
$PLOT scripts/plot_steps_vs_seqs_paper.py csv/steps_to_50_seqs.csv png/steps_vs_seqs_fit.png --fit   # same, plus critical-batch fits S = S_min (1 + N*/N) per series, N* annotated (table: csv/steps_vs_seqs_fit_fit_table.csv)
$PLOT scripts/plot_lr_scaling_paper.py csv/steps_to_50_kl.csv png/lr_scaling_paper.png            # paper figure: best LR vs batch size (KL 1e-3/1e-2) and vs rollouts (PNG + PDF)
$PLOT scripts/plot_lr_slope_vs_target.py csv/val_curves_n16.json png/lr_slope_vs_target.png      # exponent of best LR vs batch size, per target accuracy and KL
$PLOT scripts/plot_lr_slope_vs_target.py csv/val_curves_n16.json png/lr_slope_paper.png --paper  # paper version (PNG + PDF in paper/)
$PLOT scripts/plot_rollouts_vs_downsample_paper.py csv/steps_to_50_seqs.csv png/rollouts_vs_downsample_paper.png   # paper: more rollouts vs downsampling at B=128, with LR-bracket status (PNG + PDF)

# critical batch size vs target accuracy (30%..55% in 0.5% steps), from the full val curves
$PLOT scripts/plot_cbs_vs_target.py csv/val_curves_n16.json png/cbs_vs_target.png                     # first batch whose 2-doubling average slope is shallower than -0.7; bsz < 8 and non-monotone outliers ignored
$PLOT scripts/plot_cbs_vs_target.py csv/val_curves_n16.json png/cbs_vs_target_sustained.png --rule sustained   # first doubling from which no later doubling pays off

# reward curves: downsample 64->K vs plain GRPO with n = K rollouts (bsz 128)
$PULL scripts/pull_curves_downsample_vs_n.py csv/curves_downsample_vs_n.csv
$PLOT scripts/plot_curves_downsample_vs_n.py csv/curves_downsample_vs_n.csv png/curves_downsample_vs_n.png
```

Conventions: n = 16 only and KL from config for the batch-sweep plots; runs named `*_updated_scaling`
are on the fixed dp_actor loss normalisation (2026-09-11) and are marked with a centre dot. Steps-to-threshold is
the crossing interpolated between the two bracketing validation readings (`steps_to_<pct>_interp`); the first
checkpoint at/above the threshold is kept alongside as `steps_to_<pct>`.
```bash

# interactive: pick a target accuracy, see steps-to-target vs batch size (KL 1e-3 vs 1e-2)
$PULL scripts/pull_val_curves.py csv/val_curves_n16.json                  # full val curve per n=16 run
$PLOT scripts/build_steps_vs_bsz_interactive.py csv/val_curves_n16.json html/steps_vs_bsz_interactive.html
# then republish html/steps_vs_bsz_interactive.html to https://claude.ai/artifact/GhSHS36AsHoVcctvwfPDwn
$PLOT scripts/build_lr_grid_interactive.py csv/val_curves_n16.json html/lr_grid_interactive.html   # (batch, LR) grid with target slider + bracket status
# then republish html/lr_grid_interactive.html to https://claude.ai/artifact/7E7Y5QAMBp9hJavaHhC9X2
```

# ICLR Figure 2 (total compute and data to target)
`iclr_figs/` holds a self-contained pipeline (wandb pull -> per-cell best-LR crossings -> figure) for the
paper's Figure 2, the sequences and prompt draws each sweep cell consumed before reaching 50%. See
`iclr_figs/README.md`. Outputs: `paper/fig_budget_frontier_{a,b}.pdf`, `png/fig_budget_frontier.png`,
`csv/fig_budget_frontier_table.csv`.
