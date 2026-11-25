# Progress Report — Proposed Solution Stage

**Project:** Urban Accident Risk Prediction System
**Course work:** Machine Learning Team Project
**Report date:** 24 November 2026

| Name | Student ID |
|------|-----------|
| Fatih Alkan | 79731 |
| Denzel Blessing Sibanda | 79826 |
| Solomon Mbedzi | 79788 |
| Helin Agze Gulec | 79879 |
| Furkan Çokpınar | 79791 |

---

## 1. Summary

The team has completed the **Proposed Solution** stage. Building on the Gap Analysis, we
designed a machine learning system that predicts the probability of at least one
police-reported injury road collision for every H3 hexagon cell (about 0.74 km²) in Greater
London and every three-hour window. It combines historical accident data, traffic conditions,
weather, time and road characteristics from open sources. The main model is LightGBM with
imbalance handling, probability calibration and TreeSHAP explanations, and it is evaluated on a
future year.

The approach is documented in the LaTeX paper and in a presentation. A GitHub repository with
code, tests, documentation and a branch-based workflow is set up for the implementation phase.

> At this stage **no experiments have been run** and no results are reported. All improvements
> in the documents are stated as expectations.

---

## 2. Requirements of this stage and where they are met

| Requirement | Status | Where |
|-------------|--------|-------|
| 1. Introduction and overview (problem, approach, justification, differences, expected improvements) | Done | Paper §2.1; [PROJECT.md §3](PROJECT.md#3-overview-of-the-proposed-solution) |
| 2. Mapping of research gaps to proposed solutions (structured table, every gap covered) | Done | Paper Table 2 (gaps G1–G6); [PROJECT.md §4](PROJECT.md#4-mapping-research-gaps-to-solutions) |
| 3. Proposed system architecture diagram | Done | Paper Figure 1 (11 stages) |
| 4. ML methodology (dataset, preprocessing, features, algorithms, training, tuning, loss) | Done | Paper §2.4, Tables 3–7 |
| 5. Additional technical diagrams | Done | Paper Figures 2–4 (preprocessing pipeline, training workflow, prediction framework) |
| 6. Evaluation strategy (metrics, justification, baselines) | Done | Paper §2.6, Tables 8–9 |
| 7. Tools and technologies | Done | Paper §2.7, Table 10 |
| 8. Mathematical formulations (numbered, variables defined, referenced) | Done | Paper §2.8, Equations 1–25 |
| 9. Expected contributions, advantages, limitations | Done | Paper §2.9 |
| Academic references | Done | 55 references in `paper/references.bib` |
| Overleaf project | Done | [Overleaf project (read-only link)](https://www.overleaf.com/project/6b060c04e2ca21afb91b3d88/share#e13afaa6995383fe55032fde3958ebc94222bf5a139b3d31) |
| Presentation of the proposed solution | Done | [`presentations/proposed-solution/`](presentations/proposed-solution/) |

---

## 3. Work completed

### 3.1 Gap Analysis

Seven representative studies were reviewed, from statistical crash-frequency models to deep
spatio-temporal networks (Hetero-ConvLSTM, DAP, RiskOracle, GSNet). Six research gaps were
identified:

| ID | Gap |
|----|-----|
| G1 | Problem framing and information leakage |
| G2 | Limited integration of heterogeneous factors |
| G3 | Extreme class imbalance and misleading metrics |
| G4 | Optimistic validation of spatio-temporal models |
| G5 | Lack of interpretability |
| G6 | Uncalibrated outputs and limited reproducibility |

### 3.2 Proposed Solution

| Element | Decision |
|---------|----------|
| Study area | Greater London, STATS19 police forces 1 and 48 |
| Spatial unit | H3 resolution 8 (about 2,200 cells) |
| Temporal unit | 3-hour windows |
| Period | 2018–2024 (2025 as an additional out-of-time check) |
| Data | STATS19, DfT AADF + TRA0307, Open-Meteo, OpenStreetMap, calendar |
| Target | At least one collision in the cell-window (binary) |
| Main model | LightGBM, class-weighted loss, negative subsampling (β = 0.05) with prior correction |
| Baselines | Historical rate, logistic regression, random forest, MLP, XGBoost |
| Validation | Train 2018–2022, validate 2023, test 2024 once; backtests; spatial block check |
| Tuning | Optuna TPE, 100 trials, validation PR-AUC |
| Calibration | Isotonic regression on 2023 |
| Metrics | PR-AUC (primary), Recall@k, ROC-AUC, F1, Brier score, bootstrap CIs |
| Explainability | TreeSHAP, risk maps |
| Experiments | E1–E8, each linked to a gap |

### 3.3 Documents produced

| Document | Content | Link |
|----------|---------|------|
| Paper (LaTeX + PDF, ACM format) | Gap Analysis and Proposed Solution: 4 figures, 10 tables, 25 equations, 55 references | [`paper/`](paper/), [Overleaf](https://www.overleaf.com/project/6b060c04e2ca21afb91b3d88/share#e13afaa6995383fe55032fde3958ebc94222bf5a139b3d31) |
| Presentation (PDF) | 13 slides, five presenters, formulas and diagrams | [`presentations/proposed-solution/`](presentations/proposed-solution/) |
| Project documentation | Complete project content in Markdown | [`PROJECT.md`](PROJECT.md) |
| Project plan | Phases, tasks and compute plan | [`docs/project_plan.md`](docs/project_plan.md) |
| Data sources | Download steps, licences, column notes | [`docs/data_sources.md`](docs/data_sources.md) |

### 3.4 Repository and code

The repository uses a branch-based workflow (`main`, `develop`, seven `feature/...` branches and
`paper/proposed-solution`). Every push to `main` and `develop` runs continuous integration with
linting and unit tests. Every change to the paper is compiled automatically.

| Module | Content | Status |
|--------|---------|--------|
| `uarps.features.temporal` | Time windows, cyclical encoding (Eq. 4), calendar and COVID-19 flags | Implemented, unit-tested |
| `uarps.features.history` | Past counts and decayed intensity (Eq. 5), leakage-safe | Implemented, unit-tested |
| `uarps.features.spatial` | H3 mapping, k-ring neighbours, spatial lag (Eq. 6) | Implemented, unit-tested |
| `uarps.evaluation.metrics` | PR-AUC, Recall@k, thresholds, block-bootstrap confidence intervals | Implemented, unit-tested |
| `uarps.evaluation.splits` | Chronological split, backtests, negative subsampling | Implemented, unit-tested |
| `uarps.models.baselines` | B0 historical-rate baseline | Implemented, unit-tested |
| `uarps.models.calibration` | Prior correction (Eq. 16) and isotonic calibrator (Eq. 17) | Implemented, unit-tested |
| `uarps.data.stats19` | Download, loading, filtering, cleaning of STATS19 | Implemented; tested on synthetic files, not yet on real downloads |
| `uarps.data.weather` | Open-Meteo download and window aggregation | Implemented; tested on synthetic data, not yet on real downloads |
| `uarps.data.traffic` | AADF loading and exposure (Eq. 7) | Implemented; tested on synthetic data, not yet on real downloads |
| `uarps.data.osm` | OSM network and road features | Draft, not yet tested |
| Model training (LightGBM, B1–B4), SHAP | — | Planned (Phase 4–5) |

---

## 4. Presentation plan

| Part | Presenter | Topic | Slides |
|------|-----------|-------|--------|
| 1 | Fatih Alkan | The problem and our idea | 3–4 (+ title, agenda, summary) |
| 2 | Denzel Blessing Sibanda | Research gaps and our answers | 5–6 |
| 3 | Solomon Mbedzi | Data and system architecture | 7–8 |
| 4 | Helin Agze Gulec | Data preparation and features (formulas) | 9–10 |
| 5 | Furkan Çokpınar | Model, training and evaluation (formulas) | 11–12 |

Speaking notes and likely questions are in
[`presentations/proposed-solution/README.md`](presentations/proposed-solution/README.md).

---

## 5. Next steps

| Phase | Next tasks | Branch |
|-------|-----------|--------|
| 1. Data acquisition | Download STATS19 2018–2024, AADF and TRA0307, Open-Meteo weather, OSM network; build the London H3 cell list | `feature/data-acquisition` |
| 2. Preprocessing and EDA | Map collisions to cells and windows; build the full panel; EDA (temporal profiles, Moran's I, COVID-19 and 2024 effects) | `feature/preprocessing`, `feature/eda` |
| 3. Feature engineering | History, exposure, weather, calendar, road features; leakage audit | `feature/feature-engineering` |
| 4. Modelling | Baselines B0–B4; LightGBM with Optuna; calibration | `feature/modeling` |
| 5. Evaluation | Experiments E1–E8; SHAP; risk maps | `feature/evaluation`, `feature/explainability` |
| 6. Final paper | Results, discussion, final presentation | `paper/proposed-solution` |

---

## 6. Risks and mitigation

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Large data volume (about 6.4 million rows per year) | Memory limits in Colab | Parquet files, monthly chunks, negative subsampling for training |
| Changes in STATS19 column names between years | Loading errors | Column alias mapping in `uarps.data.stats19`; checks per year |
| Traffic counts are annual and less reliable on minor roads | Weaker exposure feature | Combine with OSM road length; flag estimated counts; ablation E3 |
| Data leakage through features | Over-optimistic results | Past-only features, unit tests, leakage audit, experiment E2 |
| Temporal drift (COVID-19, 2024 specification change) | Unstable performance | Backtests, sensitivity run, 2025 out-of-time check |
| Limited time for five work packages | Delays | Phased plan, branch per work package, short weekly meetings |
