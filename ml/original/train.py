#!/usr/bin/env python3
"""
train.py - trains and compares the demand forecasting models. This is what you run on RunPod.

    python train.py --quick              # smoke test (a few minutes): 1 backtest fold, 1 seed, 3 epochs
    python train.py                      # full run: 4 backtest folds + final, 3 seeds each
    python train.py --seeds 5 --hidden 128   # bigger / more stable deep model (uses the GPU more)
    python train.py --skip-deep          # baseline + LightGBM only (no PyTorch needed)

What it does
  1. For each backtest "today" (config.BACKTEST_ORIGINS) and the final "today" (last day of data):
       - baseline: 28-day average (what the seller does now)
       - tree:     LightGBM
       - deep:     PyTorch LSTM, trained on the GPU, several seeds averaged
       - ensemble: average of tree and deep
     each trained only on data up to that day, then forecasting the next 90 days.
  2. Scores every forecast against the answer key (evaluate.py).
  3. Picks the model with the best average BACKTEST score (never the final test, to stay honest)
     and writes its forecast to outputs/forecast_next90.csv for the restocking step.

Outputs (in outputs/): see README.md.
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import config
from baseline import forecast_baseline
from data_utils import ShopData
from evaluate import MODEL_NAMES, day_types, load_truth, plot_examples, plot_scores, score
from tree_model import BACKEND as TREE_BACKEND
from tree_model import TreeForecaster

EXAMPLE_SKUS = ["PH04", "HK02", "FE05", "FE08", "SE01", "PH01"]
if TREE_BACKEND != "lightgbm":
    MODEL_NAMES["tree"] = "Gradient boosting (scikit-learn)"
SAFE_RUNPOD_VARS = ["RUNPOD_POD_ID", "RUNPOD_GPU_COUNT", "RUNPOD_CPU_COUNT", "RUNPOD_DC_ID", "RUNPOD_POD_HOSTNAME"]


class Logger:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, msg=""):
        line = f"[{datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def environment_info(device_arg, need_torch, out):
    info = dict(started=datetime.now().isoformat(timespec="seconds"), python=platform.python_version(),
                platform=platform.platform(), tree_backend=TREE_BACKEND)
    info["runpod"] = {k: os.environ[k] for k in SAFE_RUNPOD_VARS if k in os.environ}
    try:
        smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=20)
        if smi.returncode == 0:
            (out / "nvidia_smi.txt").write_text(smi.stdout)
            q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
            info["nvidia_smi"] = q.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        info["nvidia_smi"] = "not available"
    device = "cpu"
    if need_torch:
        try:
            import torch
        except ImportError:
            raise SystemExit("PyTorch is not installed. On RunPod use a PyTorch template (or run: bash setup.sh). "
                             "To run without the deep model: python train.py --skip-deep")
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
            torch.backends.cudnn.benchmark = True
        if device_arg == "cuda" and not torch.cuda.is_available():
            raise SystemExit("--device cuda was asked for, but PyTorch cannot see a GPU. Check nvidia-smi.")
        device = "cuda" if (device_arg in ("auto", "cuda") and torch.cuda.is_available()) else "cpu"
    try:
        import lightgbm
        info["lightgbm"] = lightgbm.__version__
    except ImportError:
        pass
    info["device"] = device
    return info, device


def ensemble(a, b, quantiles):
    cols = ["mean"] + [f"p{int(q * 100)}" for q in quantiles]
    m = a.merge(b, on=["sku", "date", "h"], suffixes=("_a", "_b"))
    out = m[["sku", "date", "h"]].copy()
    for c in cols:
        out[c] = (m[f"{c}_a"] + m[f"{c}_b"]) / 2
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=config.DATA_DIR)
    ap.add_argument("--out", default=config.OUT_DIR)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--quick", action="store_true", help="smoke test: 1 fold, 1 seed, 3 epochs")
    ap.add_argument("--folds", type=int, default=None, help="number of backtest folds (most recent first), 0 = none")
    ap.add_argument("--seeds", type=int, default=None, help="deep model copies to average")
    ap.add_argument("--epochs", type=int, default=None, help="max epochs for the deep model")
    ap.add_argument("--hidden", type=int, default=None, help="LSTM size")
    ap.add_argument("--skip-deep", action="store_true")
    ap.add_argument("--skip-tree", action="store_true")
    args = ap.parse_args()

    t_start = time.time()
    out = Path(args.out)
    (out / "models").mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    (out / "train_log.txt").write_text("")
    log = Logger(out / "train_log.txt")

    deep_cfg, tree_cfg = dict(config.DEEP), dict(config.TREE)
    n_folds = len(config.BACKTEST_ORIGINS)
    if args.quick:
        n_folds = config.QUICK["folds"]
        deep_cfg.update(seeds=config.QUICK["seeds"], max_epochs=config.QUICK["max_epochs"])
        tree_cfg.update(max_rounds=config.QUICK["max_rounds"])
    if args.folds is not None:
        n_folds = args.folds
    if args.seeds is not None:
        deep_cfg["seeds"] = args.seeds
    if args.epochs is not None:
        deep_cfg["max_epochs"] = args.epochs
    if args.hidden is not None:
        deep_cfg["hidden"] = args.hidden
    use_deep, use_tree = not args.skip_deep, not args.skip_tree

    info, device = environment_info(args.device, use_deep, out)
    log("=" * 72)
    log(f"Demand forecasting training  |  mode: {'QUICK TEST' if args.quick else 'full'}")
    log(f"Python {info['python']}  |  device: {device}  |  GPU: {info.get('gpu', 'none')}  |  "
        f"PyTorch {info.get('torch', '-')}  |  trees: {TREE_BACKEND}")
    if use_deep and device == "cpu":
        log("NOTE: no GPU found - the deep model will train on the CPU (much slower).")

    data = ShopData(args.data, config.HORIZON)
    truth, truth_source = load_truth(data)
    log(f"Data: {len(data.skus)} products, {data.start:%d %b %Y} to {data.end:%d %b %Y} ({data.n_hist} days), "
        f"{int(data.censored[:, :data.n_hist].sum()):,} sold-out product-days skipped as targets, "
        f"{len(data.cal_names)} calendar features")
    log(f"Scoring against: {truth_source}")

    runs = []
    for o in (config.BACKTEST_ORIGINS[len(config.BACKTEST_ORIGINS) - n_folds:] if n_folds > 0 else []):
        oi = data.idx(o)
        if oi < 180 or oi + config.HORIZON > data.n_hist - 1:
            log(f"  skipping backtest origin {o}: needs 180+ days before and {config.HORIZON} days after it")
            continue
        runs.append((f"backtest_{o}", oi))
    runs.append(("final", data.n_hist - 1))

    if use_deep:
        from deep_model import forecast_deep, save_model, train_deep
        from windows import WindowMaker
        wm = WindowMaker(data, deep_cfg["enc_len"], config.HORIZON)

    forecasts, histories, timings = [], [], []
    importance = None
    for run, oi in runs:
        origin = data.dates[oi]
        log("-" * 72)
        log(f"{run}: train on data up to {origin:%d %b %Y}, forecast {data.dates[oi + 1]:%d %b} - "
            f"{data.dates[oi + config.HORIZON]:%d %b %Y}")
        fcs = {}
        t0 = time.time()
        fcs["baseline"] = forecast_baseline(data, oi, config.HORIZON, config.QUANTILES)
        timings.append(dict(run=run, model="baseline", seconds=time.time() - t0))

        if use_tree:
            t0 = time.time()
            tf = TreeForecaster(tree_cfg, config.QUANTILES, log).fit(data, oi, config.HORIZON)
            fcs["tree"] = tf.forecast(data, oi, config.HORIZON)
            timings.append(dict(run=run, model="tree", seconds=time.time() - t0))
            log(f"    tree done in {time.time() - t0:.0f}s")
            if run == "final":
                tf.save(out / "models")
                importance = tf.feature_importance()

        if use_deep:
            t0 = time.time()
            models = []
            for seed in range(deep_cfg["seeds"]):
                model, hist = train_deep(data, wm, oi, deep_cfg, seed, device, log)
                models.append(model)
                histories += [dict(run=run, **h) for h in hist]
                if run == "final":
                    save_model(model, out / "models" / f"deep_seed{seed}.pt", data, wm, deep_cfg, seed)
            fcs["deep"] = forecast_deep(models, data, wm, oi, config.HORIZON, config.QUANTILES, device)
            timings.append(dict(run=run, model="deep", seconds=time.time() - t0))
            log(f"    deep done in {time.time() - t0:.0f}s ({deep_cfg['seeds']} seed(s))")

        if "tree" in fcs and "deep" in fcs:
            fcs["ensemble"] = ensemble(fcs["tree"], fcs["deep"], config.QUANTILES)

        run_fc = pd.concat([df.assign(run=run, model=m, origin=origin) for m, df in fcs.items()], ignore_index=True)
        forecasts.append(run_fc)
        quick = score(run_fc, truth)
        for _, r in quick.sort_values(config.PRIMARY_METRIC).iterrows():
            log(f"    {r['model']:9s} {config.PRIMARY_METRIC} {r[config.PRIMARY_METRIC]:.3f}  "
                f"wape_daily {r['wape_daily']:.3f}  bias {r['bias']:+.3f}  cover_p90 {r['cover_p90']:.2f}")

    # ------------------------------------------------------------------ scoring and model choice
    fc = pd.concat(forecasts, ignore_index=True)
    cat = data.products.set_index(data.products["sku"].astype(str))["category"]
    fc["category"] = fc["sku"].map(cat)
    metrics = score(fc, truth)
    backtest = metrics[metrics["run"] != "final"]
    final = metrics[metrics["run"] == "final"]
    if len(backtest):
        ranking = backtest.groupby("model")[config.PRIMARY_METRIC].mean().sort_values()
        chosen_on = "backtest average"
    else:
        ranking = final.set_index("model")[config.PRIMARY_METRIC].sort_values()
        chosen_on = "final test (no backtest folds were run)"
    best = ranking.index[0]
    log("=" * 72)
    log(f"Best model by {config.PRIMARY_METRIC} ({chosen_on}): {best} - {MODEL_NAMES[best]}")

    summary = pd.concat([
        backtest.groupby("model").mean(numeric_only=True).assign(stage="backtest average"),
        final.set_index("model").drop(columns="run").assign(stage="final test (next 90 days)"),
    ]).reset_index()
    summary = summary[["stage", "model"] + [c for c in summary.columns if c not in ("stage", "model")]]
    by_cat = score(fc, truth, by=("run", "model", "category"))
    labels = day_types(data)
    by_day = score(fc.merge(labels, on="date", how="left"), truth, by=("run", "model", "day_type"))

    # ------------------------------------------------------------------ write everything
    fc_final = fc[fc["run"] == "final"]
    names = data.products.set_index(data.products["sku"].astype(str))["product_name"] \
        if "product_name" in data.products else pd.Series(dtype=str)
    best_fc = fc_final[fc_final["model"] == best].drop(columns=["run", "origin"]).copy()
    best_fc.insert(1, "product_name", best_fc["sku"].map(names))
    best_fc["date"] = best_fc["date"].dt.date
    best_fc.round(3).to_csv(out / "forecast_next90.csv", index=False)
    fc_final.assign(date=fc_final["date"].dt.date, origin=fc_final["origin"].dt.date).round(3) \
        .to_csv(out / "forecast_next90_all_models.csv", index=False)
    bt = fc[fc["run"] != "final"]
    if len(bt):
        bt.assign(date=bt["date"].dt.date, origin=bt["origin"].dt.date).round(3) \
            .to_csv(out / "forecasts_backtest.csv", index=False)
    metrics.round(4).to_csv(out / "metrics_by_run.csv", index=False)
    summary.round(4).to_csv(out / "metrics_summary.csv", index=False)
    by_cat.round(4).to_csv(out / "metrics_by_category.csv", index=False)
    by_day.round(4).to_csv(out / "metrics_by_day_type.csv", index=False)
    if histories:
        pd.DataFrame(histories).round(5).to_csv(out / "deep_training_history.csv", index=False)
    if importance is not None:
        importance.to_csv(out / "tree_feature_importance.csv", index=False)
    pd.DataFrame(timings).round(1).to_csv(out / "timings.csv", index=False)

    plot_scores(backtest, final, config.PRIMARY_METRIC, out / "plots" / "model_comparison.png")
    examples = [s for s in EXAMPLE_SKUS if s in data.sku_index] or data.skus[:6]
    plot_examples(data, fc_final, truth, best, examples, out / "plots" / "forecast_examples.png")

    write_summary(out, summary, ranking, best, chosen_on, by_cat, by_day, info, timings, deep_cfg, args, truth_source)
    info.update(finished=datetime.now().isoformat(timespec="seconds"), total_minutes=round((time.time() - t_start) / 60, 1),
                best_model=best, chosen_on=chosen_on, deep_config=deep_cfg if use_deep else None,
                tree_config=tree_cfg if use_tree else None, runs=[r for r, _ in runs], quick=args.quick)
    (out / "run_info.json").write_text(json.dumps(info, indent=2, default=str))
    log(f"Done in {(time.time() - t_start) / 60:.1f} min. Results in {out.resolve()}")
    log("Start with outputs/summary.md and outputs/plots/.")


def _table(df, cols, pct=True):
    head = "| " + " | ".join(["model"] + cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    lines = [head, sep]
    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            vals.append(f"{v * 100:.1f}%" if pct and c not in ("cover_p90", "cover_80") else f"{v:.2f}")
        lines.append("| " + " | ".join([MODEL_NAMES.get(r["model"], r["model"])] + vals) + " |")
    return "\n".join(lines)


def write_summary(out, summary, ranking, best, chosen_on, by_cat, by_day, info, timings, deep_cfg, args, truth_source):
    cols = ["wape_weekly", "wape_daily", "error_28d", "bias", "pinball_p90", "cover_p90", "cover_80"]
    order = {m: i for i, m in enumerate(ranking.index)}
    lines = [f"# Forecast model comparison{' (QUICK TEST - not final numbers)' if args.quick else ''}", "",
             f"- Best model: **{MODEL_NAMES[best]}** (`{best}`), chosen on {config.PRIMARY_METRIC}, {chosen_on}",
             f"- Device: {info['device']}  |  GPU: {info.get('gpu', 'none')}  |  PyTorch {info.get('torch', '-')}"
             f"  |  trees: {info['tree_backend']}",
             f"- Scored against: {truth_source}", ""]
    for stage in summary["stage"].unique():
        s = summary[summary["stage"] == stage].copy()
        s["_o"] = s["model"].map(order)
        lines += [f"## {stage}", "", _table(s.sort_values("_o"), cols), ""]
    lines += ["How to read: wape = average error as a share of actual demand (lower is better). "
              "bias + means over-forecasting. cover_p90 should be close to 0.90 and cover_80 close to 0.80.", ""]
    fin = by_cat[by_cat["run"] == "final"].pivot(index="category", columns="model", values="wape_weekly")
    if len(fin):
        lines += ["## Final test, weekly error by category", "",
                  "| category | " + " | ".join(MODEL_NAMES.get(m, m) for m in fin.columns) + " |",
                  "|" + "---|" * (len(fin.columns) + 1)]
        for c, r in fin.iterrows():
            lines.append(f"| {c} | " + " | ".join(f"{v * 100:.1f}%" for v in r.values) + " |")
        lines.append("")
    dd = by_day.groupby(["day_type", "model"])["wape_daily"].mean().unstack()
    if len(dd):
        lines += ["## Daily error by type of day (all runs)", "",
                  "| day type | " + " | ".join(MODEL_NAMES.get(m, m) for m in dd.columns) + " |",
                  "|" + "---|" * (len(dd.columns) + 1)]
        for c, r in dd.iterrows():
            lines.append(f"| {c} | " + " | ".join(f"{v * 100:.1f}%" for v in r.values) + " |")
        lines.append("")
    t = pd.DataFrame(timings).groupby("model")["seconds"].sum()
    lines += ["## Training time", ""] + [f"- {MODEL_NAMES.get(m, m)}: {v / 60:.1f} min" for m, v in t.items()] + [""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
