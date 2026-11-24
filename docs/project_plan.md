# Project plan

The work is split into six phases. Each phase has a GitHub milestone, and each task is a GitHub
issue assigned to one or more team members. Owners are agreed in the team meeting and recorded
on the issues.

| Phase | Branch | Main deliverable | Status |
|-------|--------|------------------|--------|
| 0. Proposal | `paper/proposed-solution` | Gap Analysis and Proposed Solution in the paper | Done (draft) |
| 1. Data acquisition | `feature/data-acquisition` | Raw STATS19, AADF, TRA0307, weather and OSM files; download scripts | To do |
| 2. Preprocessing and EDA | `feature/preprocessing`, `feature/eda` | Clean cell-window panel; EDA notebook and figures | To do |
| 3. Feature engineering | `feature/feature-engineering` | Model-ready Parquet tables for 2018–2024 | To do |
| 4. Modelling | `feature/modeling` | Baselines B0–B4, tuned LightGBM, calibrator | To do |
| 5. Evaluation and explainability | `feature/evaluation`, `feature/explainability` | Experiments E1–E8, SHAP analysis, risk maps | To do |
| 6. Final paper and presentation | `paper/proposed-solution` | Results, discussion, final PDF and slides | To do |

## Phase 1 — Data acquisition

- [ ] Download STATS19 collision files 2018–2024 (`scripts/download_data.py stats19`).
- [ ] Download the DfT AADF count-point file and table TRA0307; convert TRA0307 into a
      day-of-week × window profile.
- [ ] Download Open-Meteo hourly weather for the London point grid, 2018–2024.
- [ ] Download the OSM drive network and POIs for Greater London; save to `data/external/`.
- [ ] Get the Greater London boundary polygon (London Datastore) and build the list of H3
      resolution-8 cells.

## Phase 2 — Preprocessing and EDA

- [ ] Filter, clean and map collisions to cells and 3-hour windows.
- [ ] Build the complete cell × window panel (about 6.4 million rows per year) and labels.
- [ ] EDA: temporal profiles, spatial clustering (Moran's I, LISA), positive rate per year,
      effect of COVID-19 years, comparison of 2024 with earlier years.

## Phase 3 — Feature engineering

- [ ] History features (past counts, decayed intensity) and spatial lags.
- [ ] Traffic exposure per cell and window.
- [ ] Weather features per window, including the previous-window precipitation.
- [ ] Time, calendar and darkness features.
- [ ] Road and POI features per cell.
- [ ] Leakage audit and feature selection.

## Phase 4 — Modelling

- [ ] B0 historical rate, B1 logistic regression, B2 random forest, B3 MLP, B4 XGBoost.
- [ ] LightGBM with class weighting; Optuna search on validation PR-AUC.
- [ ] Prior correction and isotonic calibration on 2023.

## Phase 5 — Evaluation and explainability

- [ ] Experiments E1–E8 (see Table 9 of the paper).
- [ ] Block-bootstrap confidence intervals.
- [ ] TreeSHAP global and local explanations; risk maps with Folium.

## Phase 6 — Final paper and presentation

- [ ] Results and discussion sections.
- [ ] Final figures exported to `reports/figures/`.
- [ ] Presentation slides and demonstration.

## Compute plan

* Training set: about 1.7 million rows after negative subsampling (β = 0.05), fits in Google
  Colab memory.
* Validation and test years: about 6.4 million rows each; predict in monthly chunks.
* Store intermediate tables as Parquet in `data/interim/` and `data/processed/`.
