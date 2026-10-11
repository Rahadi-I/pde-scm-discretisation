# pde-scm-discretisation

Code and data for the paper

> **Predictively Equivalent, Causally Different: Discretisation and Learned Stencils in
> Causal Models of PDE Systems**

A discretised partial differential equation is a structural causal model whose parents are
fixed by the numerical stencil. The paper asks when the effect of an intervention computed on
such a model converges to the effect on the continuous system as the grid is refined, for
stencils derived from an equation and for stencils learned from data. This repository
reproduces every number and figure in the paper.

## Contents

```
experiments/   E1–E5: the five numerical experiments and the paper figures
pdebench/      solver verification against PDEBench and the data-quality scan
figures/       the figures as they appear in the paper
```

| Script | Section | What it shows | Output |
|---|---|---|---|
| `experiments/e1_convergence.py` | 3, 5 | Smooth forcing, derived stencils: the effect converges at the scheme's order (1.01 upwind, 2.00 centred, semi-discrete and exact in time; 0.99 for the explicit scheme) | `e1_convergence.csv` |
| `experiments/e2_singular.py` | 3, 5 | Point source: order 1 on a cell edge, with the first-order term of Proposition 1 predicted to 1.2%; order 2 at a cell centre; the 2D pointwise functional grows as log(1/h)/(2πD) | `e2_singular.csv` |
| `experiments/e3_clamping.py` | 3, 5 | State interventions: a 2D point clamp decays as A/(log(1/h)+C), with A predicted from capacity (to 0.03%) | `e3_clamping.csv`, `e3_capacity_fit.csv` |
| `experiments/e4_bistable.py` | 3, 5 | Bistable front: on coarse grids the front is pinned and the effect stays confined to the intervened interval; also checks the continuum front speed | `e4_bistable.csv` |
| `experiments/e5_learned_stencil.py` | 4, 5 | Latent correlated forcing: learned stencils, free or constrained to a consistent diffusion stencil, predict better and estimate the intervention effect wrongly by ~100%; fitting to recorded interventions recovers it | `e5_learned_stencil.csv` |
| `experiments/make_figures.py` | — | All figures except the PDEBench verification figure and the framework diagram | `figures/*.pdf` |
| `experiments/verify_theory.py` | 3, 4 | Independent checks of Lemma 1, Corollary 1 and Propositions 1–3 against the outputs above | printed |

The experiments generate their own data and need no downloads.

## Requirements

Python ≥ 3.10 and the packages in `requirements.txt`:

```
pip install -r requirements.txt
```

The results in the paper were produced with numpy 2.x, scipy 1.1x and matplotlib 3.10–3.11.
Figures produced with other matplotlib versions can differ slightly in font metrics.

## Reproducing the experiments

```
cd experiments
python e1_convergence.py        # < 1 s
python e2_singular.py           # ~ 1 s
python e3_clamping.py           # ~ 20 s
python e4_bistable.py           # ~ 1 s
python e5_learned_stencil.py    # ~ 10 s
python make_figures.py          # writes ../figures/*.pdf
python verify_theory.py         # independent checks of the formulas
```

Each script overwrites the CSV of the same name. The CSVs in this repository are the ones
reported in the paper; rerunning reproduces them to a relative difference below 1e-11.

In E5 the latent forcing takes the values of an Ornstein–Uhlenbeck process at the sampling
times and is held constant over each interval, so every sampled Fourier mode follows the
scalar model of Lemma 1 exactly; with continuously varying forcing the leading-order bias
is the same but the correction is O(Δt²) rather than O(Δt³).

## PDEBench verification and data-quality scan

PDEBench is used only to verify the solvers. It contains no interventions; all
counterfactuals in E1–E5 are produced by re-solving the equations. These scripts need the
PDEBench files, which are available from DaRUS (https://doi.org/10.18419/DARUS-2986, CC BY 4.0).
The files used are `1D_Advection_Sols_beta0.4.hdf5`, the sixteen `ReacDiff_Nu*_Rho*.hdf5`
training files, the twenty `ReacDiff_react_*` test files, and, for the scan only,
`1D_Burgers_Sols_Nu0.001.hdf5`, `1D_diff-sorp_NA_NA.h5`, `2D_diff-react_NA_NA.h5` and
`2D_rdb_NA_NA.h5`.

```
cd pdebench
python check_pdebench.py --root <data folder> --csv pdebench_data_urls.csv   # MD5 check
python make_verification_figure.py --root <data folder> --out ../figures/fig_pdebench_verification.pdf
python scan_pdebench_quality.py --root <data folder>
python check_reacdiff_stagnation.py --root <data folder>
python compare_reacdiff_files.py --root <data folder>
```

`check_pdebench.py` needs `pdebench_data_urls.csv` from the PDEBench repository
(https://github.com/pdebench/PDEBench). `pdebench_io.py` reads both PDEBench file layouts
(flat `tensor` files and per-sample groups) and is imported by the verification scripts.

| Script | What it does |
|---|---|
| `make_verification_figure.py` | Advection vs. exact translation; reaction–diffusion vs. an IMEX spectral solver over the evolving part of each trajectory |
| `verify_against_pdebench.py` | The same comparisons, printed as numbers |
| `inspect_pdebench.py` | Prints the layout, shapes and stored parameters of each file |
| `scan_pdebench_quality.py` | Frozen trajectories, repeated frames and NaN/Inf in any PDEBench file |
| `check_reacdiff_stagnation.py` | Per frozen sample: freeze index, spatial mean and range at the freeze, implied ν·Δt |
| `compare_reacdiff_files.py` | Checks that reaction–diffusion files sharing a freeze value contain different data |
| `diagnose_reacdiff_freeze.py` | Freeze times compared with the logistic solution |

`pdebench/results/` holds the outputs used in the paper:
- `pdebench_quality_scan.csv`: the scan of the nine time-dependent files.
- `reacdiff_stagnation.csv`: all frozen samples, with 1531 rows for the sixteen training files and the five frozen test trajectories.
- `reacdiff_stagnation_test.csv`: the test files on their own.

### Note on the 1D reaction–diffusion files

In all sixteen released 1D reaction–diffusion training files, most trajectories stop evolving
before t = 1. In 99.6% of frozen samples the frozen field is spatially uniform, at a value
common to all samples of a file. This is consistent with single-precision stagnation of the
time integrator, as reported in PDEBench issue #94
(https://github.com/pdebench/PDEBench/issues/94). The verification in the paper is therefore
restricted to the evolving part of each trajectory. If you use these files for training or
evaluation, check them with `scan_pdebench_quality.py` first.

## Citation

If you use this code, please cite the paper (see `CITATION.cff`).

## Licence

MIT; see `LICENSE`. The PDEBench data are distributed by their authors under CC BY 4.0 and are
not included here.
