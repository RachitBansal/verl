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
$PLOT scripts/plot_steps_to_50.py    csv/steps_to_50_kl.csv png/steps_to_50_n64.png --n 64   # same for the K = 64 batch sweep
$PULL scripts/gpu_hours_sweeps.py csv/gpu_hours_sweeps.csv csv/val_curves_n16.json csv/val_curves_n64.json csv/val_curves_bsz128.json csv/val_curves_bsz1024.json   # GPU hours behind the four LR grids (wandb runtime x 4, resumed segments de-overlapped; failed attempts listed separately)
$PULL scripts/check_verdicts.py [--users cmohri,brachit] [--cancel] [--log logs/check_verdicts.log]                                # running CBS jobs vs their neighbours' 50% crossings: which already have their verdict (never / slower); --cancel stops mine
$PLOT scripts/plot_steps_to_50.py    csv/steps_to_50_kl.csv png/steps_to_50_n.png --by-n   # LR x rollout-count grid at batch 128
$PLOT scripts/plot_steps_to_50.py    csv/steps_to_50_kl.csv png/steps_to_50_n_bsz512.png --by-n 512     # same at batch 512
$PLOT scripts/plot_steps_to_50.py    csv/steps_to_50_kl.csv png/steps_to_50_n_bsz1024.png --by-n 1024   # same at batch 1024
$PLOT scripts/plot_steps_to_50_combined.py csv/steps_to_50_kl.csv png/steps_to_50_combined.png --lr-max 1e-4 --lr-min 5e-7 --nstar csv/steps_vs_seqs_b1024_fit_fit_table.csv   # the K=16 batch grid and the B=1024 rollout grid side by side: shared LR axis, one colour scale and legend, (--n 16 --bsz 1024; PNG + PDF; table csv/steps_to_50_combined_table.csv)
$PLOT scripts/plot_steps_to_50_combined.py csv/steps_to_50_kl.csv png/steps_to_50_n64_paper.png --n 64 --lr-min 5e-7 --fixed-only --batch-only --nstar csv/steps_vs_seqs_b1024_fit_fit_table.csv   # the K=64 batch grid alone, same paper styling (fixed-code runs only: its three pre-fix runs stepped at ~0.46x nominal LR); note the K=64 LR ladder was laid out at sqrt(2) per doubling, so its fastest-LR fit returns 0.50 by construction
$PLOT scripts/plot_steps_to_50_combined.py csv/steps_to_50_kl.csv png/steps_to_50_combined3.png --n2 64 --lr-max 1.2e-4 --lr-min 5e-7 --nstar csv/steps_vs_seqs_b1024_fit_fit_table.csv   # three panels: K=16 batch grid, B=1024 rollout grid, K=64 batch grid (fixed-code only), one LR axis, colour scale and legend; lr-max 1.2e-4 keeps the 1.1e-4 'never' that closes the K=64 B=4096 bracket
$PLOT scripts/plot_steps_to_50_combined.py csv/steps_to_50_kl.csv png/steps_to_50_split.png --n2 64 --lr-max 1.2e-4 --lr-min 5e-7 --split --nstar csv/steps_vs_seqs_b1024_fit_fit_table.csv   # markers in each sweep's Figure 1 hue (dark = fewer steps) with a grey colour bar (--mono: one blue scale), no fit on the K = 64 panel (--fit-n2 restores it); --nstar marks each sweep's fitted critical batch (B* / K*) from the Figure 1 fit table; the same three panels as separate files for the paper's subfigure layout: _a (K=16 and K=64 prompt sweeps, 0.655 linewidth), _b (B=1024 rollout sweep with the colour bar, 0.33 linewidth), _legend (two-row strip at full linewidth); PNG + paper/*.pdf
$PLOT scripts/plot_steps_to_50_kl.py csv/steps_to_50_kl.csv png/steps_to_50_kl.png     # ... KL 1e-3 vs 1e-2
$PULL scripts/gpu_hours_sweeps.py csv/gpu_hours_sweeps.csv csv/val_curves_n16.json csv/val_curves_n64.json csv/val_curves_bsz128.json csv/val_curves_bsz1024.json   # GPU hours behind the LR grids and the downsampling figures (wandb runtime x 4 GPUs; resumed segments de-overlapped; failed attempts listed separately)
$PLOT scripts/plot_steps_vs_bsz.py   csv/steps_to_50_kl.csv png/steps_vs_bsz.png       # steps to 50% vs batch
$PLOT scripts/plot_steps_vs_bsz_kl.py csv/steps_to_50_kl.csv png/steps_vs_bsz_kl.png

# downsampling vs plain GRPO on a sequences-per-step axis
$PULL scripts/pull_steps_to_50_seqs.py csv/steps_to_50_seqs.csv
$PLOT scripts/plot_steps_vs_seqs.py csv/steps_to_50_seqs.csv png/steps_vs_seqs.png # plain n=16 sweep vs downsampling; teal = plain n=64 batch sweep (fixed code)
$PLOT scripts/plot_steps_vs_seqs.py csv/steps_to_50_seqs.csv png/steps_vs_seqs_prefix.png --prefix   # + pre-fix downsample runs
$PLOT scripts/plot_steps_vs_seqs.py csv/steps_to_50_seqs.csv png/steps_vs_seqs_nsweep.png --nsweep   # + rollout sweep (bsz 128, n varied; solid = fixed-code at n=32/64, dashed = pre-fix only)
$PLOT scripts/plot_steps_vs_seqs_paper.py csv/steps_to_50_seqs.csv png/steps_vs_seqs_paper.png   # paper figure: n=16 batch sweep, rollout sweep, n=64 batch sweep (PNG + PDF)
$PLOT scripts/plot_steps_vs_seqs_paper.py csv/steps_to_50_seqs.csv png/steps_vs_seqs_fit.png --fit   # same, plus the fit S = S_min (1 + N_crit/N) per series with the fitted N_crit marked (--knee 1.25 marks the 25%-above-perfect-scaling point instead; table: csv/steps_vs_seqs_fit_fit_table.csv)
$PLOT scripts/plot_steps_vs_seqs_paper.py csv/steps_to_50_seqs.csv png/steps_vs_prompts_paper.png --x-prompts         # same with prompts per step B on the x-axis (rollout sweeps collapse to one column); add --fit for B* and B_crit
$PLOT scripts/plot_steps_vs_seqs_paper.py csv/steps_to_50_seqs.csv png/steps_vs_seqs_b1024.png --b1024   # + the B = 1024 rollout sweep (purple); combine with --fit
$PLOT scripts/plot_steps_vs_seqs_paper.py csv/steps_to_50_seqs.csv png/steps_vs_seqs_b1024_fit.png --b1024 --fit   # same with the fits (table: csv/steps_vs_seqs_b1024_fit_fit_table.csv)
$PLOT scripts/plot_steps_vs_seqs_paper.py csv/steps_to_50_seqs.csv png/steps_vs_seqs_b1024_kl2_fit.png --b1024 --fit --kl2   # same plus the K=16 batch sweep at KL 1e-2 as a fifth series (light-blue squares, own fit; never pooled with KL 1e-3)
$PLOT scripts/plot_lr_scaling_paper.py csv/steps_to_50_kl.csv png/lr_scaling_paper.png            # paper figure: best LR vs batch size (KL 1e-3/1e-2) and vs rollouts (PNG + PDF)
$PLOT scripts/plot_lr_slope_vs_target.py csv/val_curves_n16.json png/lr_slope_vs_target.png      # exponent of best LR vs batch size, per target accuracy and KL
$PLOT scripts/plot_lr_slope_vs_target.py csv/val_curves_n16.json png/lr_slope_paper.png --paper  # paper version (PNG + PDF in paper/)
$PLOT scripts/plot_theorem_fits.py csv/steps_to_50_seqs.csv png/theorem_fits.png                           # Theorem 1: B_crit(K) and K_crit(B) forms fitted to the per-sweep critical sizes (prompt sweeps K=16/64, rollout sweeps B=128/1024)
$PLOT scripts/plot_theorem_joint.py csv/steps_to_50_seqs.csv png/theorem_joint.png                         # Theorem 1 fitted once (S0, sigma_prompt^2, sigma_rollout^2) to all four sweeps: curves + residuals, and the implied B_crit(K), K_crit(B) vs measured (--drop K64,B128 to exclude sweeps)
$PLOT scripts/plot_theorem_staged.py csv/steps_to_50_seqs.csv png/theorem_staged.png                       # Theorem 1 in stages: S0 from the prompt sweeps, then (sigma_prompt^2, sigma_rollout^2) per rollout sweep from its floor and knee
$PLOT scripts/plot_doubling_corollary.py csv/steps_to_50_seqs.csv png/doubling_corollary.png --bracketed-only   # Corollary 2: from each (B,K) whose three cells all have a complete LR bracket, steps after doubling rollouts vs after doubling prompts, with the joint-fit prediction
$PULL scripts/pull_val_curves.py csv/val_curves_n64.json --n 64                                       # full val curves of the K=64 batch sweep
$PLOT scripts/plot_nstar_vs_target.py csv/val_curves_n16.json png/nstar_vs_target.png --n64 csv/val_curves_n64.json --b128 csv/val_curves_bsz128.json   # fitted N* vs target per sweep, from columns with a complete LR bracket (lower panel: how many); batch sizes whose crossing falls before the run's first eval are left out of the fits (--keep-first-interval to include them)
$PLOT scripts/plot_nstar_vs_target.py csv/val_curves_n16.json png/nstar_vs_target_from30.png --n64 csv/val_curves_n64.json --b128 csv/val_curves_bsz128.json --min-target 0.30   # same from 30%: shows where the fits stop being supported
$PLOT scripts/plot_rollouts_vs_downsample_paper.py csv/steps_to_50_seqs.csv png/rollouts_vs_downsample_paper.png   # paper: more rollouts vs downsampling at B=128, with LR-bracket status (PNG + PDF)
$PLOT scripts/plot_rollouts_vs_downsample_clean.py csv/steps_to_50_seqs.csv png/rollouts_vs_downsample_clean.png   # clean paper version in the steps_to_50 style ("upsampling" = the downsample runs: generate 64, train K): left = steps to target per K (fastest LR), right = every LR tried per K up to 5e-5 with the fastest ringed (PNG + PDF; table: csv/rollouts_vs_downsample_clean_table.csv)
$PLOT scripts/plot_rollouts_vs_downsample_clean.py csv/steps_to_50_seqs.csv png/rollouts_vs_downsample_clean_fit.png --fit   # same with the Figure 1 fit of the rollout sweep (S_min, N* from csv/steps_vs_seqs_fit_fit_table.csv, 7 points incl. K=128) drawn solid; downsampling points joined, not fitted (table: csv/rollouts_vs_downsample_clean_fit_fit_table.csv)
$PULL scripts/pull_val_curves_bsz128.py csv/val_curves_bsz128.json                                 # full val curves of every B=128 run (rollout sweep + downsampling); add --incremental
$PULL scripts/pull_val_curves_bsz1024.py csv/val_curves_bsz1024.json                               # full val curves of the B=1024 rollout sweep; add --incremental
$PLOT scripts/audit_crossings.py csv/val_curves_n16.json csv/val_curves_n64.json csv/val_curves_bsz128.json csv/val_curves_bsz1024.json   # audit: N* per sweep under interpolated / checkpoint / previous-reading / sustained crossings (--target 0.5); per-batch-size diagnostics in csv/crossing_sensitivity_cells.csv
$PLOT scripts/plot_kl_compare_clean.py csv/steps_to_50_kl.csv png/kl_compare_clean.png   # paper: KL 1e-3 vs 1e-2 on the K=16 prompt sweep in the same style: left steps to target per B (fastest LR per KL), right every LR tried per B for both KLs with the fastest ringed (PNG + PDF; table csv/kl_compare_clean_table.csv)
$PLOT scripts/plot_kl_compare_clean.py csv/steps_to_50_kl.csv png/kl_compare_clean_fit.png --fit   # same with both series' Figure 1 fits (S_min, N* from csv/steps_vs_seqs_b1024_kl2_fit_fit_table.csv)
$PULL scripts/pull_token_series.py csv/token_series.json csv/val_curves_n16.json csv/val_curves_n64.json csv/val_curves_bsz128.json   # per-step mean prompt/response length of every run in those files (add --incremental)
$PLOT scripts/plot_nstar_vs_tokens.py csv/val_curves_n16.json png/nstar_vs_tokens.png --tokens csv/token_series.json --n64 csv/val_curves_n64.json --b128 csv/val_curves_bsz128.json   # fitted N* vs the tokens actually generated to reach each target (cheapest column, measured lengths); KL 1e-3 and 1e-2 separate; --all-cols --target-step 0.01 ignores bracketing; --x fitted places points at S_min N* L like Rachit's Fig 5b
$PLOT scripts/plot_nstar_vs_tokens.py csv/val_curves_n16.json png/nstar_vs_tokens_fitted.png --tokens csv/token_series.json --n64 csv/val_curves_n64.json --b128 csv/val_curves_bsz128.json --x fitted   # x = S_min N* times the measured mean sequence length (Rachit's placement with real lengths); add --all-cols --target-step 0.01 for _allcols, plus --min-target 0.30 for _from30_allcols
$PLOT scripts/plot_nstar_vs_tokens.py csv/val_curves_n16.json png/nstar_vs_prompts.png --tokens csv/token_series.json --n64 csv/val_curves_n64.json --b128 csv/val_curves_bsz128.json --x prompts   # x = prompts trained on to reach the target (steps x B of the cheapest bracketed batch size)
$PLOT scripts/plot_nstar_vs_tokens.py csv/val_curves_n16.json png/nstar_vs_prompts_fit.png --tokens csv/token_series.json --x prompts --sweeps n16 --fit   # K=16 KL 1e-3 only, with the power-law fit N* = c P^alpha across the targets (exponent in the legend; csv/nstar_vs_prompts_fit_fit_table.csv); --sweeps n16,n16kl2,n64,b128 selects series, --fit works with any --x
$PLOT scripts/plot_nstar_vs_tokens.py csv/val_curves_n16.json png/nstar_vs_prompts_fitted_fit.png --tokens csv/token_series.json --x prompts-fitted --sweeps n16 --fit   # x = fitted prompts to target S_min x B* (the prompt-unit analogue of --x fitted), K=16 KL 1e-3, with the power-law fit (csv/nstar_vs_prompts_fitted_fit_fit_table.csv)
$PLOT scripts/plot_nstar_vs_tokens.py csv/val_curves_n16.json png/nstar_vs_prompts_fitted_from30.png --tokens csv/token_series.json --x prompts-fitted --sweeps n16 --fit --min-target 0.30   # same from 30% (default: at least 5 LR-bracketed batch sizes per target; the 33% point, the first with 5, fits N* = 441 and is an outlier)
$PLOT scripts/plot_fixed_budget_acc.py csv/val_curves_n16.json png/fixed_budget_acc.png --budgets 1600,3200,6400,12800,25600,51200,76800,102400   # accuracy after a fixed number of prompts trained on vs batch size (best LR per point; strict interpolation rule, no early-stopped runs) + _dropoff.png: batch size where accuracy drops off vs budget; --prompts N for one budget
$PLOT scripts/plot_fixed_budget_dropoff_all.py png/fixed_budget_dropoff_all.png --n16 csv/val_curves_n16.json --n64 csv/val_curves_n64.json --b128 csv/val_curves_bsz128.json --b1024 csv/val_curves_bsz1024.json   # same drop-off analysis for every sweep, budget in sequences (B x K x steps), drop-off in sequences per step; + _curves.png per sweep
$PLOT scripts/plot_fixed_budget_dropoff_all.py png/fixed_budget_dropoff_all_fit.png --n16 csv/val_curves_n16.json --n64 csv/val_curves_n64.json --b128 csv/val_curves_bsz128.json --b1024 csv/val_curves_bsz1024.json --fit   # same, plus power-law fits N_drop = c S^alpha per sweep (>= 3 points) and pooled over all sweeps; exponents in the legend, csv/fixed_budget_dropoff_all_fit_fit_table.csv
$PLOT scripts/plot_fixed_budget_dropoff_all.py png/fixed_budget_dropoff_prompts_fit.png --n16 csv/val_curves_n16.json --sweeps n16 --prompts --fit   # prompt units: budget = prompts trained on (B x steps), drop-off = prompts per step B, K=16 KL 1e-3 only, with the power-law fit (csv/fixed_budget_dropoff_prompts_fit_fit_table.csv); --sweeps n16,n16kl2,n64 (and --n64 csv/val_curves_n64.json) adds the other prompt sweeps and a pooled fit
$PLOT scripts/plot_rollouts_vs_downsample_targets.py csv/val_curves_bsz128.json png/rollouts_vs_downsample_targets.png   # steps to 35/45/50% vs rollouts trained per prompt, standard vs downsampling, LR-bracket rings (--targets 0.35,0.45,0.5)
$PLOT scripts/plot_downsample_matched_lr.py csv/val_curves_bsz128.json png/downsample_matched_lr.png              # standard vs downsampling at a matched LR (1e-5): val curves per K (_curves) and steps to 35/45/50% (_steps)
$PLOT scripts/plot_downsample_best_lr_steps.py csv/val_curves_bsz128.json png/downsample_best_lr_steps.png   # same comparison with the fastest LR per point instead of a matched LR (--targets 0.35,0.45,0.5)
$PLOT scripts/plot_budget_frontier.py csv/steps_to_50_seqs.csv png/budget_frontier.png              # paper Fig. 2 from our table: total sequences and prompt draws to 50% per cell, KL 1e-3 (add --pool-kl for Rachit's pooled-KL selection); PNG + paper/budget_frontier_{a,b}.pdf
$PLOT scripts/plot_compute_vs_data.py csv/steps_to_50_seqs.csv png/compute_vs_data.png                 # compute (sequences) vs data (prompt draws) to 50%, one point per (B,K); --rollouts adds the K sweeps
$PULL scripts/pull_token_costs.py csv/compute_vs_data_table.csv csv/token_costs.csv                   # mean prompt/response length up to the crossing for those runs
$PLOT scripts/plot_compute_vs_data.py csv/steps_to_50_seqs.csv png/compute_vs_data_flops.png --flops  # same with the y-axis in FLOPs (12 N tokens per sequence, see the script)

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

# ICLR figures 2 and 5b
`iclr_figs/` holds a self-contained pipeline (wandb pull -> per-cell best-LR crossings -> figure) for the
paper's Figure 2 (sequences and prompt draws each sweep cell consumed before reaching 50%) and Figure 5b
(critical batch size N* refit at each target accuracy, against training tokens). See `iclr_figs/README.md`.

His Figure 5b on our val-curve files (inputs rebuilt from csv/val_curves_*.json + csv/token_series.json into his long format):
$PLOT scripts/build_traj_from_curves.py csv/traj_from_curves.csv csv/meta_from_curves.jsonl csv/token_series.json csv/val_curves_n16.json csv/val_curves_n64.json csv/val_curves_bsz128.json   # rebuild his inputs after a val-curve pull
$PLOT iclr_figs/make_cbs_vs_tokens.py --traj csv/traj_from_curves.csv --meta csv/meta_from_curves.jsonl --out iclr_out --stem fig4_cbs_vs_tokens_ourdata   # his script unchanged (pools KL within a batch size)
$PLOT iclr_figs/make_cbs_vs_tokens_by_kl.py --traj csv/traj_from_curves.csv --meta csv/meta_from_curves.jsonl --out iclr_out                          # K=16 split into KL 1e-3 / 1e-2 lines; K=64 and rollouts at KL 1e-3
$PLOT iclr_figs/make_cbs_vs_tokens_by_kl.py --traj csv/traj_from_curves.csv --meta csv/meta_from_curves.jsonl --out iclr_out --kl3_only --stem fig4_cbs_vs_tokens_kl1e-3   # every line KL 1e-3 only   (iclr_out/: copy the PNGs to png/, PDFs to paper/, tables to csv/)
Outputs: `paper/fig_budget_frontier_{a,b}.pdf`, `paper/fig4_cbs_vs_tokens.pdf`, the PNGs under `png/`, and
`csv/fig_budget_frontier_table.csv`, `csv/cbs_vs_tokens_table.csv`.
$PLOT iclr_figs/make_kl_fig.py --traj csv/traj_from_curves.csv --meta csv/meta_from_curves.jsonl --out iclr_out   # KL figure, two panels in the kl_compare_clean style (started from Rachit's make_kl_fig.py in cbs-for-rl@onpol-theory-reeval-jun10; his panel (c) dropped): (a) steps vs B per KL with the Figure 1 fits and B* (--fit-table), (b) fastest LR vs the highest LR that reached the target, band shaded, non-reaching LRs above it as hollow dots. 'stable' = crossed the target (--stable reach, default); --stable hold also requires its last 10% of readings to still average >= the target; --stable peak is his original rule (last 10% >= 60% of peak). Also writes fig_kl_stability_{a,b} and paper/fig_kl_stability.pdf; prints the ceiling per B
$PLOT iclr_figs/make_kl_fig_horizon.py --traj csv/traj_from_curves.csv --meta csv/meta_from_curves.jsonl --out iclr_out --horizon 2.5   # its panel (b) as run vs with every run cut at H x its crossing (a common horizon), to show how much of 'stable' is the horizon; same --stable rules (non-default rules get a _<rule> suffix)

# Paper figures (2026-10-20): the exact commands behind the ICLR draft
```bash
$PLOT scripts/plot_steps_vs_seqs_paper.py csv/steps_to_50_seqs.csv png/steps_vs_seqs_fit.png --b1024 --fit --knee 1.25          # Figure 1 (-> iclr/figures/steps_vs_seqs_fit.pdf)
$PLOT scripts/plot_steps_vs_seqs_paper.py csv/steps_to_50_seqs.csv png/steps_vs_seqs_b1024_fit.png --b1024 --fit                # fitted N* table read by the LR figure
$PLOT scripts/plot_steps_to_50_combined.py csv/steps_to_50_kl.csv png/steps_to_50_split.png --n2 64 --lr-max 1.2e-4 --lr-min 5e-7 --split --nstar csv/steps_vs_seqs_b1024_fit_fit_table.csv --fig1-colors --no-nstar-n2 --no-fit-n2 --no-early --share-b   # LR figure: Figure 1 palette; K=64 panel without N* line or fit; no "too early to tell" runs
$PLOT scripts/plot_rollouts_vs_downsample_clean.py csv/steps_to_50_seqs.csv png/rollouts_vs_downsample_clean_fit.png --fit      # upsampling figure
```
Figure 3 (KL): `$PLOT iclr_figs/make_kl_fig.py --traj <traj.csv> --meta <runs.jsonl> --out iclr_out --fit-table csv/steps_vs_seqs_b1024_kl2_fit_fit_table.csv` (refresh that fit table first with `plot_steps_vs_seqs_paper.py ... --b1024 --fit --kl2`; the trajectories can be the cbs repo's `iclr_figs/data/traj_all.<date>.csv` / `all_runs.<date>.jsonl`), PDF in `paper/fig_kl_stability.pdf`.
All paper figures share `scripts/paperstyle.py` (canvas 2.45x the printed width; Figure 1 palette).
Budget frontier, CBS vs tokens and the appendix trajectories come from the cbs repo, `critical-batch-rlvr/analysis/iclr_figs` (README there; `fetch_incremental.py` refreshes the trajectories).
