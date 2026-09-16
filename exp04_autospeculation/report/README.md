# report/

`report.pdf` and its sources. Rebuild with:

```
./build.sh
```

Everything is read through `../outputs/final/`, the by-deliverable symlink
index, so the report can only describe the four deliverables -- repoint a
symlink there and the report follows it. No step needs a GPU.

| file | what it does |
|---|---|
| `common.py` | loading, labels, the validated palette |
| `make_figures.py` | `figures/frontier_<benchmark>.pdf` -- score against tokens/s, one figure per benchmark |
| `make_tables.py` | `tables/*.tex` (4 tables) and `tables/key_numbers.json` |
| `make_examples.py` | `examples.tex` -- 3 decoded tasks per benchmark, both schedulers |
| `make_trajectory.py` | `trajectory.tex` -- every step of Mbpp/56 at adaptive delta=1.5, from `data/trajectory_mbpp56/` |
| `report.tex` | the document; prose only, every number comes from the generated files |

Requires `tectonic` (in the `diffusion` conda env), matplotlib, and the SDAR
tokenizer via the `vendor_env/tf452` overlay (`build.sh` sets `PYTHONPATH`).
There is no `pdflatex` on this host.

`data/trajectory_mbpp56/` is the one GPU re-run the report depends on: Mbpp/56 at
adaptive delta=1.5 with `--trace-limit 1` and otherwise the launcher's exact
arguments. `make_trajectory.py` refuses to build if it stops matching the
recorded run.

## Selection rules, fixed in advance

- Examples are the **first three tasks in run order** per benchmark, the same
  task ids for both schedulers. Adaptive is shown at delta=0.5, mid-sweep.
- Figure colours are the dataviz reference palette, validated all-pairs in light
  mode (worst CVD dE 9.2, worst normal-vision dE 16.3). Aqua warns on contrast
  against the surface, so every series is also direct-labelled and every plotted
  value appears in Table 1.
- Ratios in the tables are aggregate (sum/sum over tasks), not means of per-task
  ratios.
