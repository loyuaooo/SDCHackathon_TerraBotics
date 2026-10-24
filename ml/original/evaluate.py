"""
evaluate.py - scores forecasts against what really happened, and draws the result charts.

Truth: data/_ground_truth/true_demand.csv (history) and future_demand.csv (the 90 days after
"today"). These files are ONLY read here, never for training. They contain true demand, including
the sales that were lost while a product was sold out. If they are missing (real data), observed
sales on non-sold-out days are used instead.

Metrics (lower is better unless noted):
  wape_daily   sum |forecast - actual| / sum actual, per product per day
  wape_weekly  the same on weekly totals per product (what restocking decisions use)
  error_28d    error of the 28-day total per product (roughly lead time + one week)
  bias         (sum forecast - sum actual) / sum actual  (+ = over-forecast, - = under-forecast)
  pinball_p90  quantile loss of P90 / sum actual (how good the "safe" forecast is)
  cover_p90    share of days with actual <= P90  (target: about 0.90, higher is not better)
  cover_80     share of days with P10 <= actual <= P90  (target: about 0.80)
"""
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_COLORS = {"deep": "#2a78d6", "tree": "#eb6834", "ensemble": "#1baf7a", "baseline": "#8a8985"}
MODEL_NAMES = {"deep": "Deep learning (LSTM)", "tree": "LightGBM", "ensemble": "Deep + LightGBM average",
               "baseline": "28-day average (today's method)"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"


def load_truth(data):
    gt = Path(data.data_dir) / "_ground_truth"
    parts = []
    for name in ("true_demand.csv", "future_demand.csv"):
        if (gt / name).exists():
            df = pd.read_csv(gt / name, parse_dates=["date"], usecols=["date", "sku", "true_demand"])
            parts.append(df)
    if parts:
        truth = pd.concat(parts, ignore_index=True).rename(columns={"true_demand": "actual"})
        truth["sku"] = truth["sku"].astype(str)
        return truth.drop_duplicates(["date", "sku"]), "true demand (answer key)"
    s, t = np.where(data.valid[:, :data.n_hist])
    truth = pd.DataFrame({"date": data.dates[t], "sku": np.array(data.skus)[s], "actual": data.y[s, t]})
    return truth, "observed sales on non-sold-out days"


def day_types(data):
    """Label each date: mega sale (+-1 day), holiday, festive run-up (28 days before a festival), normal."""
    ev = data.events
    label = pd.Series("normal day", index=data.dates)
    fest = pd.to_datetime(ev.loc[ev["event_type"] == "festival", "date"])
    for f in fest:
        label[(data.dates >= f - pd.Timedelta(days=28)) & (data.dates < f)] = "festive run-up"
    is_hol = ev["event_type"].isin(["festival", "public_holiday"])
    if "is_public_holiday" in ev:
        is_hol &= ev["is_public_holiday"].astype(str).str.lower().isin(["true", "1"])
    hol = pd.to_datetime(ev.loc[is_hol, "date"])
    label[data.dates.isin(hol)] = "public holiday"
    mega = pd.to_datetime(ev.loc[(ev["event_type"] == "mega_sale") & ev["importance"].isin(["medium", "major"]), "date"])
    for m in mega:
        label[(data.dates >= m - pd.Timedelta(days=1)) & (data.dates <= m + pd.Timedelta(days=1))] = "big sale (+-1 day)"
    return label.rename("day_type").rename_axis("date").reset_index()


def score(fc, truth, by=("run", "model")):
    """fc: run, model, sku, date, h, mean, p10, p50, p90 (+ any extra grouping columns)."""
    by = list(by)
    df = fc.merge(truth, on=["date", "sku"], how="inner")
    y, f = df["actual"].values, df["mean"].values
    df["_ae"] = np.abs(f - y)
    df["_err"] = f - y
    e = y - df["p90"].values
    df["_pin"] = np.maximum(0.9 * e, -0.1 * e)
    df["_c90"] = (y <= df["p90"].values + 1e-9).astype(float)
    df["_c80"] = ((y >= df["p10"].values - 1e-9) & (y <= df["p90"].values + 1e-9)).astype(float)
    g = df.groupby(by)
    tot = g["actual"].sum().clip(lower=1e-9)
    out = pd.DataFrame({
        "wape_daily": g["_ae"].sum() / tot,
        "bias": g["_err"].sum() / tot,
        "pinball_p90": g["_pin"].sum() / tot,
        "cover_p90": g["_c90"].mean(),
        "cover_80": g["_c80"].mean(),
        "actual_units": g["actual"].sum(),
    })
    df["_week"] = (df["h"] - 1) // 7
    w = df.groupby(by + ["sku", "_week"])[["mean", "actual"]].sum()
    w["_ae"] = (w["mean"] - w["actual"]).abs()
    wg = w.groupby(level=by)
    out.insert(0, "wape_weekly", wg["_ae"].sum() / wg["actual"].sum().clip(lower=1e-9))
    m = df[df["h"] <= 28].groupby(by + ["sku"])[["mean", "actual"]].sum()
    m["_ae"] = (m["mean"] - m["actual"]).abs()
    mg = m.groupby(level=by)
    out.insert(2, "error_28d", mg["_ae"].sum() / mg["actual"].sum().clip(lower=1e-9))
    return out.reset_index()


# --------------------------------------------------------------------------------------- charts
def _style(plt):
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
        "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
        "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.titlelocation": "left"})


def plot_scores(backtest, final, metric, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    models = [m for m in ["baseline", "tree", "deep", "ensemble"]
              if m in set(final["model"]) | set(backtest["model"] if len(backtest) else [])]
    fig, axes = plt.subplots(1, 2 if len(backtest) else 1, figsize=(12 if len(backtest) else 6.5, 4.2), squeeze=False)
    panels = ([("Backtest (average of folds)", backtest)] if len(backtest) else []) + [("Final test: next 90 days", final)]
    for ax, (title, df) in zip(axes[0], panels):
        vals = df.groupby("model")[metric].mean().reindex(models)
        y = np.arange(len(models))[::-1]
        ax.barh(y, vals.values * 100, color=[MODEL_COLORS[m] for m in models], height=0.6)
        for yi, v in zip(y, vals.values):
            ax.text(v * 100 + 0.6, yi, f"{v * 100:.1f}%", va="center", fontsize=10, color=INK)
        ax.set_yticks(y)
        ax.set_yticklabels([MODEL_NAMES[m] for m in models])
        ax.set_xlim(0, np.nanmax(vals.values) * 100 * 1.25)
        ax.grid(axis="y", visible=False)
        ax.set_title(title)
        ax.set_xlabel(f"{metric} (lower is better)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_examples(data, fc_final, truth, best, skus, path, history_days=120):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    _style(plt)
    n = len(skus)
    cols = 2
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.6 * rows), squeeze=False)
    name = data.products.set_index("sku")["product_name"].to_dict() if "product_name" in data.products else {}
    end = data.dates[data.n_hist - 1]
    for ax, sku in zip(axes.ravel(), skus):
        s = data.sku_index[sku]
        lo = max(data.n_hist - history_days, 0)
        hist_d = data.dates[lo:data.n_hist]
        ax.plot(hist_d, data.y[s, lo:data.n_hist], color=INK2, lw=1, label="Past sales")
        t = truth[(truth["sku"] == sku) & (truth["date"] > end)].sort_values("date")
        ax.plot(t["date"], t["actual"], color=INK, lw=1.4, label="What really happened (answer key)")
        f = fc_final[(fc_final["sku"] == sku)]
        fb = f[f["model"] == best].sort_values("date")
        if len(fb) and "p10" in fb:
            ax.fill_between(fb["date"], fb["p10"], fb["p90"], color=MODEL_COLORS[best], alpha=0.15, lw=0,
                            label=f"P10-P90 range ({MODEL_NAMES[best]})")
        for m in ["baseline", "tree", "deep"]:
            fm = f[f["model"] == m].sort_values("date")
            if len(fm):
                ax.plot(fm["date"], fm["mean"], color=MODEL_COLORS[m], lw=2 if m == best else 1.4,
                        ls="--" if m == "baseline" else "-", label=MODEL_NAMES[m])
        ax.axvline(end, color=INK2, lw=0.8, ls=":")
        ax.set_title(f"{name.get(sku, sku)} ({sku})")
        ax.set_ylabel("units / day")
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", ncol=3, frameon=False, bbox_to_anchor=(0.01, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)
