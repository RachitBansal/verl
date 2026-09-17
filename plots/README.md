# CBS plots

```
scripts/   pull_*.py (wandb -> CSV) and plot_*.py (CSV -> PNG)
csv/       pulled data and the *_table.csv each plot script emits alongside its figure
png/       figures only
html/      interactive pages (built from csv/, published as artifacts)
```

Pull with the wandb-capable python, plot with the repo venv (has matplotlib):

```bash
PULL=/n/home03/cmohri/venvs/verl_env/bin/python
PLOT=/n/home03/cmohri/team_verl/.venv/bin/python
cd plots

# batch-size sweep (plain GRPO, n = 16)
$PULL scripts/pull_steps_to_50.py csv/steps_to_50_kl.csv "1e-3,1e-2"
$PLOT scripts/plot_steps_to_50.py    csv/steps_to_50_kl.csv png/steps_to_50.png        # LR x batch grid, KL 1e-3
$PLOT scripts/plot_steps_to_50_kl.py csv/steps_to_50_kl.csv png/steps_to_50_kl.png     # ... KL 1e-3 vs 1e-2
$PLOT scripts/plot_steps_vs_bsz.py   csv/steps_to_50_kl.csv png/steps_vs_bsz.png       # steps to 50% vs batch
$PLOT scripts/plot_steps_vs_bsz_kl.py csv/steps_to_50_kl.csv png/steps_vs_bsz_kl.png
# (append 60 as a 3rd arg to plot_steps_to_50*.py for the 60% threshold)

# downsampling vs plain GRPO on a sequences-per-step axis
$PULL scripts/pull_steps_to_50_seqs.py csv/steps_to_50_seqs.csv
$PLOT scripts/plot_steps_vs_seqs.py csv/steps_to_50_seqs.csv png/steps_vs_seqs.png
$PLOT scripts/plot_steps_vs_seqs.py csv/steps_to_50_seqs.csv png/steps_vs_seqs_prefix.png --prefix   # + pre-fix downsample runs

# reward curves: downsample 64->K vs plain GRPO with n = K rollouts (bsz 128)
$PULL scripts/pull_curves_downsample_vs_n.py csv/curves_downsample_vs_n.csv
$PLOT scripts/plot_curves_downsample_vs_n.py csv/curves_downsample_vs_n.csv png/curves_downsample_vs_n.png
```

Conventions: n = 16 only and KL from config for the batch-sweep plots; runs named `*_updated_scaling`
are on the fixed dp_actor loss normalisation (2026-09-11) and are marked with a centre dot.
```bash

# interactive: pick a target accuracy, see steps-to-target vs batch size (KL 1e-3 vs 1e-2)
$PULL scripts/pull_val_curves.py csv/val_curves_n16.json                  # full val curve per n=16 run
$PLOT scripts/build_steps_vs_bsz_interactive.py csv/val_curves_n16.json html/steps_vs_bsz_interactive.html
# then republish html/steps_vs_bsz_interactive.html to https://claude.ai/artifact/GhSHS36AsHoVcctvwfPDwn
```
