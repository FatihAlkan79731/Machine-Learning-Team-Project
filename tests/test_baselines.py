import numpy as np
import pandas as pd

from uarps.models.baselines import HistoricalRateBaseline


def test_historical_rate_ranks_hot_cells_higher():
    rng = np.random.default_rng(6)
    n = 20_000
    df = pd.DataFrame(
        {
            "cell": rng.choice(["hot", "cold"], size=n),
            "window_of_week": rng.integers(0, 56, size=n),
        }
    )
    y = np.where(df["cell"] == "hot", rng.random(n) < 0.05, rng.random(n) < 0.005).astype(int)
    model = HistoricalRateBaseline(smoothing=10).fit(df, y)
    test = pd.DataFrame({"cell": ["hot", "cold", "unseen"], "window_of_week": [3, 3, 3]})
    p = model.predict_proba(test)
    assert p[0] > p[1]
    assert p[2] == model.global_rate_  # unseen cell falls back to the global rate
