# Adaptive trajectory diagnostics

The completed adaptive records retain total denoising steps and a commitment
histogram for each of the 100 paired tasks. These support two exact analyses
without model replay:

1. **Trajectory survival:** at step `s`, count tasks with
   `denoising_steps >= s`.
2. **Commitment-width distribution:** pool histogram entries over tasks to
   measure the share of decisions committing 1, 2, ..., 64 tokens.

Regenerate the report inputs from the project root:

```bash
conda run -n diffusion python report/make_available_trajectory_analysis.py
```

Generated files:

- `report/derived/adaptive_survival.csv`
- `report/derived/adaptive_commitment_widths.csv`
- `report/table_survival_<benchmark>.tex`
- `report/table_commitment_widths.tex`
- `report/figures/fig6_adaptive_trajectories.{pdf,png}`

The report appendix explains the scope precisely. Survival counts preserve
step position but contain no width information. Histograms preserve width
frequencies but discard their temporal order. Consequently, neither artifact
can recover the mean commitment width at a particular step, the cumulative
tokens committed by that step, or the step at which EOS was committed.

`scripts/replay_adaptive_trajectories.py` remains available if those ordered
quantities are later required. It validates each replay against the saved
completion and aggregate scheduler accounting and resumes at task granularity.
