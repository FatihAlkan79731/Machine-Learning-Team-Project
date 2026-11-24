# Contributing guide

This document describes how the team works on the repository. Please read it before your first
commit.

## Branches

| Branch | Purpose | Merges into |
|--------|---------|-------------|
| `main` | Stable, reviewed state of the project. Submitted versions are tagged here. | — |
| `develop` | Integration branch for finished work packages. | `main` |
| `feature/data-acquisition` | Download scripts and loaders for STATS19, traffic counts, weather and OSM. | `develop` |
| `feature/preprocessing` | Cleaning, H3 / time-window alignment and the join into one cell-window table. | `develop` |
| `feature/eda` | Exploratory data analysis notebooks and figures. | `develop` |
| `feature/feature-engineering` | Historical, spatial-lag, exposure, weather, time and road features; feature selection. | `develop` |
| `feature/modeling` | Baselines, LightGBM, hyperparameter optimisation, imbalance handling. | `develop` |
| `feature/evaluation` | Metrics, backtests, bootstrap confidence intervals, ablation experiments. | `develop` |
| `feature/explainability` | TreeSHAP analysis and risk maps. | `develop` |
| `paper/proposed-solution` | Changes to the LaTeX paper. | `develop` |

For a small task inside a work package, create a short-lived branch from that work-package
branch, for example `feature/data-acquisition-weather`, and open a pull request back into it.

## Daily workflow

```bash
git checkout develop
git pull
git checkout feature/preprocessing     # or the branch you work on
git merge develop                      # bring in the latest integrated work
# ... edit, test ...
git add <files>
git commit -m "preprocessing: map collisions to H3 cells"
git push
```

Then open a pull request on GitHub into `develop` and ask at least one teammate for a review.

## Commit messages

Use the form `<area>: <short description in the imperative>`, for example:

* `data: add Open-Meteo hourly download`
* `features: compute decayed collision intensity`
* `models: add LightGBM training script`
* `paper: update gap-to-solution table`
* `fix: handle missing coordinates in STATS19 2019`

Keep each commit focused on one change.

## Pull requests

* Fill in the pull request template.
* All tests must pass (`pytest`) and the code must pass `ruff check .`.
* At least one approving review from another team member.
* Link the related issue (for example `Closes #12`).

## Code style

* Python 3.11, formatted to a line length of 100 (`ruff format`).
* Put reusable code in `src/uarps/`; notebooks should import from it rather than copy code.
* Every new function in `src/uarps/` gets a docstring and, where practical, a unit test in `tests/`.
* Never fit scalers, encoders or calibrators on validation or test years.
* Never use information from inside or after a prediction window as a feature.

## Data and secrets

* Do not commit raw or processed data, trained models or large files. They are ignored by `.gitignore`.
* Do not commit API keys or passwords. None of the planned data sources needs a key.

## Notebooks

Name notebooks `NN_initials_topic.ipynb`, for example `01_fa_stats19_eda.ipynb`. Clear large
outputs before committing.
