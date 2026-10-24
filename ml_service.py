"""JSON stdin/stdout forecasting service. No dependencies needed for fallback.

Trained inference is enabled only when the original feature metadata is present.
Missing observations are not silently converted into zero-sales observations.
"""
import json
import math
import statistics
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
MODEL_NAMES = ["tree_mean.txt", "tree_spread.json"] + [f"deep_seed{i}.pt" for i in range(3)]


def statistical(history, anchor, horizon):
    known = sorted((date.fromisoformat(r["date"]), max(0, r["unitsSold"]))
                   for r in history if r["date"] <= anchor.isoformat())[-90:]
    recent = known[-28:]
    values = [v for _, v in recent]
    mean = statistics.mean(values) if values else 0.0
    # A short, bounded trend, shrunk toward the longer-run mean.
    last7 = statistics.mean(values[-7:]) if values else 0.0
    level = .65 * mean + .35 * last7
    variance = statistics.variance(values) if len(values) > 1 else mean
    result = []
    for h in range(1, horizon + 1):
        target = anchor + timedelta(days=h)
        weekday = [v for d, v in recent if d.weekday() == target.weekday()]
        seasonal = statistics.mean(weekday) / mean if weekday and mean else 1
        mu = max(0, level * (.65 + .35 * min(1.8, max(.4, seasonal))))
        spread = 1.282 * math.sqrt(max(mu, variance)) * math.sqrt(1 + h / 28)
        result.append(dict(date=target.isoformat(), predicted=round(mu, 2),
                           lower=round(max(0, mu-spread), 2), upper=round(mu+spread, 2)))
    return result


def predict(request):
    anchor = date.fromisoformat(request["demoDate"])
    results = []
    missing = [f for f in MODEL_NAMES if not (MODELS / f).is_file()]
    metadata = MODELS / "training_data"
    metadata_missing = [f for f in ["products.csv", "calendar_events.csv", "calendar_daily.csv"]
                        if not (metadata / f).is_file()]
    for product in request["products"]:
        horizon = {"Daily":1, "Weekly":7, "Bi-weekly":14, "Monthly":30}[product["resupplyFrequency"]]
        history = [r for r in product["history"] if r["date"] <= anchor.isoformat()]
        reason = ""
        points = None
        if missing:
            reason = "Statistical fallback: missing model files: " + ", ".join(missing)
        elif metadata_missing:
            reason = ("Model weights detected. Statistical fallback is active because the original "
                      "products/category mapping and holiday-calendar CSV files are not supplied.")
        elif len(history) < 28:
            reason = "Statistical fallback: trained inference requires at least 28 daily sales records."
        else:
            try:
                from ml.trained_adapter import predict_ensemble
                points = predict_ensemble(product, history, anchor, horizon, MODELS)
            except (ImportError, ValueError, RuntimeError, OSError, KeyError) as exc:
                reason = f"Statistical fallback: trained model unavailable ({type(exc).__name__}: {exc})."
        mode = "Trained ensemble" if points is not None else "Statistical fallback"
        if points is None:
            points = statistical(history, anchor, horizon)
        if not history:
            reason += " No sales observations exist on or before the simulation date; forecast is zero."
        results.append(dict(id=product["id"], forecast=points, mode=mode,
                            modelMessage=reason or "Three PyTorch seeds and LightGBM, equally weighted.",
                            predictedSales=round(sum(p["predicted"] for p in points), 2),
                            forecastDays=horizon))
    return results


if __name__ == "__main__":
    json.dump(predict(json.load(sys.stdin)), sys.stdout, allow_nan=False)
