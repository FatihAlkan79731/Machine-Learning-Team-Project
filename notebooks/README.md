# Notebooks

Notebooks are for exploration and for producing figures. Reusable code belongs in
`src/uarps/` and is imported here:

```python
import sys
sys.path.append("../src")   # not needed after `pip install -e .`
from uarps.config import load_config
```

## Naming

`NN_initials_topic.ipynb`, for example:

| Notebook | Purpose |
|----------|---------|
| `01_xx_stats19_eda.ipynb` | Collision counts by year, hour, weekday; positive rate; maps |
| `02_xx_spatial_clustering.ipynb` | Moran's I and LISA on H3 cells |
| `03_xx_weather_traffic_eda.ipynb` | Weather and exposure distributions |
| `04_xx_baselines.ipynb` | B0–B4 baseline results |
| `05_xx_lightgbm.ipynb` | LightGBM tuning and calibration |
| `06_xx_shap.ipynb` | Explanations and risk maps |

## Rules

* Restart and run all cells before committing.
* Clear very large outputs (maps, long tables) before committing.
* Save figures for the paper to `reports/figures/` with descriptive names.
