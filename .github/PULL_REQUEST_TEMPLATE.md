## What does this pull request change?

<!-- Short description. Link the issue, for example "Closes #12". -->

## Type of change

- [ ] Data acquisition / preprocessing
- [ ] Feature engineering
- [ ] Modelling / evaluation
- [ ] Paper
- [ ] Documentation / tooling

## Checklist

- [ ] `pytest` passes locally
- [ ] `ruff check .` passes
- [ ] No data files, model files or secrets are committed
- [ ] No feature uses information from inside or after the prediction window
- [ ] Scalers, encoders and calibrators are fitted only on training or validation years
- [ ] Notebooks were restarted and run from top to bottom

## How was it tested?

<!-- Commands, notebooks or figures used to check the change. -->
