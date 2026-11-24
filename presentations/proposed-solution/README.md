# Presentation — Proposed Solution

**File:** [`Proposed_Solution_Presentation.pdf`](Proposed_Solution_Presentation.pdf) (13 slides, 16:9)
**Source:** [`slides.tex`](slides.tex) (LaTeX Beamer; compile with `latexmk -pdf slides.tex`)

The presentation explains how our machine learning approach addresses the research gaps from
the Gap Analysis. It has five parts, one per team member, from the simplest topic to the most
technical one. Each presenter has a colour, which is shown in the title bar and in the progress
bar at the bottom of each slide.

| Part | Presenter | Colour | Slides | Topic |
|------|-----------|--------|--------|-------|
| — | Fatih Alkan | — | 1–2 | Title and agenda |
| 1 | Fatih Alkan (79731) | Blue | 3–4 | The problem and our idea |
| 2 | Denzel Blessing Sibanda (79826) | Teal | 5–6 | Research gaps and our answers |
| 3 | Solomon Mbedzi (79788) | Purple | 7–8 | Data and system architecture |
| 4 | Helin Agze Gulec (79879) | Orange | 9–10 | Data preparation and features (formulas) |
| 5 | Furkan Çokpınar (79791) | Red | 11–12 | Model, training and evaluation (formulas) |
| — | Fatih Alkan | — | 13 | Summary and questions |

Planned length: about 2 minutes per person, 12–14 minutes in total. Equation numbers on the
slides match the numbers in the paper.

---

## Speaking notes

Short, simple notes. Use them as a guide; you do not need to read them word for word.

### Part 1 — Fatih Alkan: the problem and our idea

**Slide 3 — The problem.**
"Every year the police in London record about twenty-three thousand road accidents with
injuries. Road safety teams cannot be everywhere, so they need to know where and when accidents
are most likely. Our question is simple: for a small area and a three-hour period, what is the
probability of at least one accident? We only use information that we know before the period
starts."

**Slide 4 — Our idea in one picture.**
"We divide London into small hexagon cells and the day into three-hour windows. For each cell
and window we collect five types of data. A machine learning model, LightGBM, learns the risk.
The result is a risk map with a probability and a reason for every cell. Our approach uses only
open data and gives real probabilities and explanations. These improvements are expected; we
will test them in the next phase."

### Part 2 — Denzel Blessing Sibanda: research gaps and our answers

**Slide 5 — What is missing.**
"In our Gap Analysis we found six gaps. Some studies use information that is written after the
accident. Many use only one or two data types. Accidents are rare, so accuracy can be misleading.
Random testing lets the model see the future. Deep models do not explain their results. And the
scores are often not real probabilities."

**Slide 6 — From gap to solution.**
"This table shows our answer to each gap. For example, for gap three we keep all accidents,
weight them more in training and use better metrics such as PR-AUC. For gap four we train on
2018 to 2022, tune on 2023 and test only once on 2024. Every answer is linked to an experiment,
so we can check it."

### Part 3 — Solomon Mbedzi: data and system architecture

**Slide 7 — Our data.**
"We use five free data sources: accident records from the Department for Transport, traffic
counts, hourly weather from Open-Meteo, time and calendar information, and road data from
OpenStreetMap. London has about 2,200 cells, so we get about 6.4 million rows per year. Only
about 0.3 to 0.4 percent of them have an accident. Our target is one if there is at least one
accident, otherwise zero."

**Slide 8 — System architecture.**
"This is the full system in eleven steps: data collection, alignment to cells and time windows,
cleaning, data analysis, feature engineering, a time-based split, training, tuning, calibration,
evaluation, and finally the risk maps with explanations."

### Part 4 — Helin Agze Gulec: data preparation and features

**Slide 9 — Preparing the data.**
"Each source is cleaned separately and then joined by cell and time window. We follow two golden
rules: features use only the past, and scalers or calibrators learn only from training or
validation years. We do not use police notes such as 'wet road', because they are written after
the accident. We use rain data and sun position instead."

**Slide 10 — Features and formulas.**
"Here are our main feature formulas. Standardisation puts features on the same scale. Cyclical
encoding shows the model that 23:00 is close to midnight. Accident history gives more weight to
recent accidents. The neighbour average looks at the six or eighteen cells around a cell. Traffic
exposure estimates vehicle-kilometres in each window. At the end, we remove leaking, constant and
duplicate features."

### Part 5 — Furkan Çokpınar: model, training and evaluation

**Slide 11 — The model.**
"Our main model is LightGBM. It adds many small decision trees together, as in equation twelve.
Missing an accident is worse than a false alarm, so we use a weighted loss, equation ten. We
train on a smaller sample of non-accident cases to save memory, and equation sixteen corrects
the probabilities afterwards. Then isotonic calibration makes them reliable, and SHAP explains
each prediction."

**Slide 12 — Training and evaluation.**
"We train on 2018 to 2022, tune on 2023 and test once on 2024. Accuracy is not useful here,
because saying 'no accident' is right 99.6 percent of the time. So we use precision, recall, F1,
Recall at k and the Brier score, with PR-AUC as our main metric. We compare our model with five
baselines and report 95 percent confidence intervals. Our main limits are yearly traffic data,
a weather grid of about ten kilometres and only police-reported accidents."

### Summary — Fatih Alkan

**Slide 13.**
"To summarise: we predict accident probability for every small area of London and every
three-hour period. Our solution answers all six gaps, uses open data, gives calibrated
probabilities and explains its predictions. Next, we will collect the data, build the features
and run the experiments. Thank you. We are happy to answer your questions."

---

## Possible questions and short answers

| Question | Answer |
|----------|--------|
| Why LightGBM and not deep learning? | Our data are tabular. Tree models work very well on this kind of data, train fast and can be explained exactly with SHAP. We still compare with a neural network (MLP). |
| Why three-hour windows? | Shorter windows make accidents even rarer and the data much bigger; longer windows hide daily patterns. Three hours is a balance. |
| Why London? | The UK publishes accident locations and times openly (STATS19), and London has enough accidents to learn from. |
| How do you avoid data leakage? | Features use only past data, the split is by year, and post-accident police notes are not used as inputs. |
| What if the model is wrong? | We report confidence intervals, compare with simple baselines and show explanations, so users can judge each prediction. |
