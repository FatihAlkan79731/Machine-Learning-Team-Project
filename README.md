# Machine Learning Team Project — Urban Accident Risk Prediction System

A machine learning system that predicts the risk of road traffic accidents in urban areas using
**historical accident data, traffic conditions, weather, time and road characteristics**.

The system divides Greater London into a hexagonal grid (H3, resolution 8, about 0.74 km² per
cell) and time into three-hour windows. For every cell and window it estimates the probability
that at least one police-reported injury collision will occur, using only information that is
available before the window starts. The output is a calibrated, explainable risk score that can
be ranked into risk maps and priority lists.

> **Status:** Proposed solution defined and presented (see [Documents](#documents)). Implementation in progress.
> No experimental results have been produced yet.

---

## Team

| Name | Student ID |
|------|-----------|
| Fatih Alkan | 79731 |
| Denzel Blessing Sibanda | 79826 |
| Solomon Mbedzi | 79788 |
| Helin Agze Gulec | 79879 |
| Furkan Çokpınar | 79791 |

---

## Approach at a glance

| Stage | What we do |
|-------|-----------|
| Data acquisition | STATS19 collisions, DfT traffic counts, Open-Meteo weather, OpenStreetMap roads, calendar |
| Alignment | Map every record to an H3 cell and a 3-hour window |
| Preprocessing | Filter to London, clean, deduplicate, aggregate, handle missing values |
| Feature engineering | Accident history (decayed, spatial lags), traffic exposure, weather, time, road features |
| Validation | Chronological split: train 2018–2022, validation 2023, test 2024 |
| Models | Historical-rate baseline, logistic regression, random forest, MLP, XGBoost, **LightGBM (proposed)** |
| Imbalance | Negative subsampling + prior correction, class-weighted loss |
| Calibration | Isotonic regression on the validation year |
| Evaluation | PR-AUC (primary), Recall@k, ROC-AUC, F1, Brier score, block-bootstrap confidence intervals |
| Explainability | TreeSHAP global and local explanations, risk maps |

## Documents

| Document | Description |
|----------|-------------|
| [Paper (PDF)](paper/Urban_Accident_Risk_Paper.pdf) | Gap Analysis and Proposed Solution, with figures, tables, equations and references |
| [Paper on Overleaf](https://www.overleaf.com/project/6b060c04e2ca21afb91b3d88/share#e13afaa6995383fe55032fde3958ebc94222bf5a139b3d31) | The same paper in the team's Overleaf project (read-only link) |
| [Presentation (PDF)](presentations/proposed-solution/Proposed_Solution_Presentation.pdf) | Proposed Solution presentation, five presenters, with speaking notes in its [README](presentations/proposed-solution/README.md) |
| [PROJECT.md](PROJECT.md) | Complete project content in one Markdown document |
| [REPORT.md](REPORT.md) | Progress report for the current stage |
| [docs/project_plan.md](docs/project_plan.md) | Phases, tasks and compute plan |
| [docs/data_sources.md](docs/data_sources.md) | Data downloads, licences and notes |

---

## Data sources

All data are open and free. Raw data are **not** committed to this repository; see
[`docs/data_sources.md`](docs/data_sources.md) for download instructions and licences.

| Factor group | Source | Licence |
|--------------|--------|---------|
| Historical accidents | DfT Road Safety Data (STATS19) | Open Government Licence v3.0 |
| Traffic conditions | DfT road traffic statistics (AADF) and table TRA0307 | Open Government Licence v3.0 |
| Weather | Open-Meteo Historical Weather API (ECMWF IFS / ERA5-Land) | CC BY 4.0 |
| Road characteristics | OpenStreetMap via OSMnx | ODbL |
| Time and calendar | `holidays` package, solar position | — |

---

## Repository structure

```
.
├── configs/              # Experiment configuration (study area, years, grid, splits)
├── data/                 # Local data folders (ignored by git, except .gitkeep)
│   ├── raw/              # Original downloads, never edited
│   ├── interim/          # Cleaned, per-source intermediate files
│   ├── processed/        # Model-ready cell-window tables (Parquet)
│   └── external/         # Boundaries and other reference data
├── docs/                 # Project plan, data sources, meeting notes
├── models/               # Trained model files (ignored by git)
├── notebooks/            # Jupyter notebooks for EDA and experiments
├── paper/                # LaTeX source and PDF of the project paper (Overleaf)
├── presentations/        # Presentation slides (PDF and LaTeX source)
├── reports/figures/      # Figures exported for the paper and presentations
├── scripts/              # Command-line entry points (download, build, train, evaluate)
├── src/uarps/            # Python package: Urban Accident Risk Prediction System
│   ├── data/             # Loading and cleaning of each data source
│   ├── features/         # Temporal, historical and spatial feature engineering
│   ├── models/           # Baselines, calibration, training
│   └── evaluation/       # Metrics, temporal splits, bootstrap confidence intervals
└── tests/                # Unit tests (pytest)
```

---

## Getting started

```bash
git clone https://github.com/FatihAlkan79731/Machine-Learning-Team-Project.git
cd Machine-Learning-Team-Project

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .

pytest                               # run the unit tests
```

**Google Colab:** clone the repository in a notebook cell, then run
`pip install -r requirements.txt` and `pip install -e .`.

Download the collision data (example):

```bash
python scripts/download_data.py stats19 --years 2018 2019 2020 2021 2022 2023 2024
```

---

## How we work

* `main` always contains reviewed, working code. Nobody commits directly to `main`.
* `develop` is the integration branch. Feature branches are merged into `develop` through pull requests.
* Each work package has its own branch (`feature/...`), and the paper has its own branch (`paper/...`).

The full workflow, branch list, commit conventions and review rules are in
[`CONTRIBUTING.md`](CONTRIBUTING.md). Tasks and milestones are tracked in GitHub Issues; the
plan is summarised in [`docs/project_plan.md`](docs/project_plan.md).

---

## Paper

The paper is written in LaTeX. The source in [`paper/`](paper/) can be compiled locally with
`latexmk -pdf main.tex` or uploaded to Overleaf (see [`paper/README.md`](paper/README.md)).
