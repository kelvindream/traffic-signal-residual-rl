# Traffic Signal Residual RL: Reproducibility Package

This package accompanies "Auditing Safety-Constrained Residual Reinforcement Learning for Multi-Intersection Traffic Signal Control: Fixed-Time Gains and Actuated-Control Limits". It contains compact code and derived data for implementation inspection, E11/E12/E13 statistical recomputation, and figure regeneration. It intentionally excludes checkpoints, replay buffers, large raw trajectories, manuscript source, and submission correspondence.

Creator: Rongtao Zhu (ORCID: 0009-0000-4528-5714), Yangtze University College of Arts and Sciences.

Repository: https://github.com/kelvindream/traffic-signal-residual-rl

## Licensing

- Original source code, supporting configurations, and documentation: MIT, see `LICENSE`.
- Original statistical data and generated figures under `results/`, and PDF/PNG figures under `paper/figures/`: CC BY 4.0, see `LICENSE-DATA`.
- Third-party materials and dependencies retain their own licenses; neither license above overrides separately applicable third-party terms.

The repository is not yet a published Zenodo archive. Do not cite the DOI reserved for the separate manual-upload draft as the DOI of this GitHub release.

## Contents

- `agents/`, `environment/`, `tools/`: core CTDE agent, SUMO environment, formal training/evaluation entry point, and E11/E12 analysis scripts.
- `nets/2x2-grid/`: network and five route scenarios used by the formal entry point.
- `configs/`: experiment configuration files retained for reference.
- `results/`: compact per-run evaluation CSV files required for exact statistical recomputation, plus aggregate JSON files required by figure generation.
- `paper/figures/make_figures.py`: script for regenerating the four paper figures.
- `scan_anonymity.py`: package-local anonymity and prohibited-artifact scan.
- `DEPENDENCY_AUDIT.txt`: audited dependency closure and known blocker.

## Installation

A Python 3.10 or 3.11 virtual environment is recommended because the pinned scientific stack predates Python 3.14 support in some packages.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For a CPU-only system, install the appropriate CPU PyTorch wheel for the local platform instead of a CUDA wheel.

## SUMO prerequisite

Install Eclipse SUMO separately (the Python packages do not install the SUMO simulator binary), verify that `sumo` is on `PATH`, and set `SUMO_HOME` to the SUMO installation directory:

```bash
sumo --version
export SUMO_HOME=/path/to/sumo
```

The environment imports `traci` and `sumolib` when `reset()` is called and launches the `sumo` binary. No SUMO binary is bundled here.

## Smoke checks

Checks that do not launch SUMO:

```bash
python -m py_compile agents/*.py environment/*.py tools/*.py paper/figures/make_figures.py
python tools/analyze_e11_n10.py --help
python tools/analyze_e12_n10.py --help
python tools/eval_enhanced_actuated_r1.py --help
python scan_anonymity.py
```

The formal training entry point can be inspected with:

```bash
python tools/train_eval_fixed_prior_ctde.py --help
```

The package includes the canonical `utils.metrics_schema` module used by the formal runners. Its constants and CSV writer are covered by the included schema tests and match the headers of the archived formal outputs. End-to-end SUMO execution still requires the dependencies and simulator described above.

## Statistical recomputation

E11 uses 10 model-level training seeds per scenario and exact common-random-number evaluation keys:

```bash
python tools/analyze_e11_n10.py
```

E12 recomputes the preregistered CoLight comparisons from the compact per-run CSV inputs:

```bash
python tools/analyze_e12_n10.py
```

E13 recomputes the explicitly post-hoc n=10 fixed-prior framework versus actuated-controller analysis, including the single 25-test Holm family and both 50,000-resample confidence intervals. Its independent audit separately recomputes the point estimates, signs, Wilcoxon tests, and Holm values:

```bash
python -m unittest tests.test_analyze_e13_n10_actuated
python tools/analyze_e13_n10_actuated.py
python tools/audit_e13_n10_actuated.py
```

These commands overwrite/create the corresponding aggregate JSON, Markdown, and CSV outputs under `results/formal_v2/`.

## Figure regeneration

After E11 recomputation (or using the included aggregate JSON), run:

```bash
python paper/figures/make_figures.py
```

This creates `fig_forest`, `fig_attribution`, `fig_ablation`, and `fig_convergence` as PDF and PNG files under `paper/figures/`.

## Training/evaluation command template

After installing the declared dependencies and SUMO, a minimal smoke run uses a short horizon and one episode, for example:

```bash
python tools/train_eval_fixed_prior_ctde.py \
  --train-scenarios low --eval-scenarios low \
  --train-seeds 2026 --eval-seeds 2031 \
  --train-episodes 1 --eval-episodes 1 \
  --simulation-time 60 --device cpu \
  --output results/smoke_ctde
```

The command interface and schema module are validated without launching SUMO; a simulator run was not executed in the packaging environment because SUMO and the full PyTorch stack were unavailable.

## Resource requirements

- Statistical recomputation: CPU only, typically under a few minutes, less than 2 GB RAM.
- Figure regeneration: CPU only, typically under one minute, less than 2 GB RAM.
- SUMO smoke evaluation: CPU only; runtime depends on SUMO and simulation horizon.
- Formal CTDE training: a CUDA-capable GPU is recommended; runtime and memory depend on episode count and evaluation schedule. Checkpoints are intentionally excluded.

## Exclusions and limitations

- No model checkpoints or optimizer/replay state are included.
- No large raw trajectory logs are included.
- No manuscript source (`paper/main.tex`) or submission correspondence is included.
- Per-run evaluation CSVs are included only where required to recompute E11/E12/E13 statistics; the 25,000-row E13 episode-difference export is omitted because it is deterministically regenerated.
- The package was assembled in an environment without the SUMO binary and full PyTorch stack, so simulator execution could not be validated there.
