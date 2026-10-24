"""
baseline.py - what the seller does today: "average of the last 28 days" for every future day.
"""
import numpy as np
import pandas as pd


def forecast_baseline(data, origin_idx, horizon, quantiles):
    rows = []
    dates = data.dates[origin_idx + 1: origin_idx + 1 + horizon]
    for s in data.eligible(origin_idx):
        lo = max(origin_idx - 27, 0)
        window = data.y[s, lo: origin_idx + 1][data.valid[s, lo: origin_idx + 1]]
        mean = float(window.mean()) if len(window) else 0.0
        qs = np.quantile(window, quantiles) if len(window) else np.zeros(len(quantiles))
        rows.append(pd.DataFrame({
            "sku": data.skus[s], "date": dates, "h": np.arange(1, horizon + 1), "mean": mean,
            **{f"p{int(q * 100)}": float(v) for q, v in zip(quantiles, qs)}}))
    return pd.concat(rows, ignore_index=True)
