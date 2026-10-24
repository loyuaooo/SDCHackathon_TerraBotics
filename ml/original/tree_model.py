"""
tree_model.py - gradient-boosted trees (LightGBM) that forecast each of the next 90 days directly.

One training row = (product, "today" = origin, days ahead = h). The features only use what was
known on the origin day (recent sales, same time last year) plus the calendar of the target day,
which is known in advance. The target is that day's sales; sold-out days are skipped.

LightGBM is used when installed (it is on RunPod after setup.sh). Without it, scikit-learn's
HistGradientBoostingRegressor is used instead, so the code still runs anywhere.
"""
import numpy as np
import pandas as pd

from config import MIN_HISTORY, VAL_DAYS

try:
    import lightgbm as lgb
    BACKEND = "lightgbm"
except ImportError:                       # pragma: no cover - depends on the machine
    lgb = None
    BACKEND = "sklearn"


def make_rows(data, origins, horizon, target_lo=-1, target_hi=None, training=True):
    """Feature table for every (eligible product, origin, h). Training rows need a trustworthy target
    with target_lo < target day <= target_hi."""
    origins = np.asarray(sorted(set(int(o) for o in origins)))
    S = len(data.skus)
    s, o, h = np.meshgrid(np.arange(S), origins, np.arange(1, horizon + 1), indexing="ij")
    s, o, h = s.ravel(), o.ravel(), h.ravel()
    d = o + h
    keep = (o - data.first_day[s] + 1 >= MIN_HISTORY) & (d < len(data.dates))
    if training:
        hi = len(data.dates) - 1 if target_hi is None else target_hi
        keep &= (d > target_lo) & (d <= hi)
        keep[keep] &= data.valid[s[keep], d[keep]]
    s, o, h, d = s[keep], o[keep], h[keep], d[keep]

    X = {"sku": s, "category": data.cat_id[s], "h": h}
    cal = data.cal[d]
    for j, name in enumerate(data.cal_names):
        X[name] = cal[:, j]
    for w in (7, 28, 91):
        X[f"mean_{w}d"] = data.window_mean(s, o, w)
    m28, sd28, zero28, sold_out28 = data.window_stats(s, o, 28)
    X["std_28d"], X["zero_share_28d"], X["sold_out_share_28d"] = sd28, zero28, sold_out28
    X["trend_7_vs_28"] = X["mean_7d"] / (m28 + 0.1)
    X["age_days"] = np.minimum(o - data.first_day[s] + 1, 730)
    ly_target = data.last_year[s, d]
    ly_before = data.window_mean(s, o - 364, 28)                     # same 4 weeks before "today", last year
    X["last_year_target"] = ly_target
    X["last_year_before"] = ly_before
    X["last_year_ratio"] = (ly_target + 0.1) / (ly_before + 0.1)
    X["last_year_guess"] = X["last_year_ratio"] * m28                 # "this year's level x last year's shape"
    for j, nm in enumerate(data.season_names):                       # same point of last year's festive season
        X[f"last_season_{nm}"] = data.season_aligned[j, s, d]
    X["last_season_any"] = data.season_any[s, d]
    X["last_season_ratio"] = (np.nan_to_num(X["last_season_any"]) + 0.1) / (ly_before + 0.1)
    X["log_price"] = np.log1p(data.price[s])
    # the same signals divided by the recent level, i.e. in the units the model predicts in
    level = np.maximum(np.nan_to_num(np.where(np.isnan(m28), X["mean_91d"], m28), nan=0.0), 1.0)
    for c in ("last_year_target", "last_season_any", "last_year_guess", "mean_7d", "mean_91d"):
        X[f"{c}_rel"] = X[c] / level
    X = pd.DataFrame(X)
    # recent level (floored at 1 unit/day): the model learns "how many times the recent average", which
    # works across big and small sellers
    meta = pd.DataFrame({"sku_idx": s, "origin_idx": o, "h": h, "date_idx": d, "level": level})
    y = data.y[s, d] if training else None
    return X, y, meta


def _alpha(y, mu):
    """Method-of-moments negative binomial spread: variance = mu + alpha * mu^2."""
    if len(y) == 0:
        return 0.3
    a = (np.sum((y - mu) ** 2 - mu)) / max(np.sum(mu ** 2), 1e-9)
    return float(np.clip(a, 0.02, 5.0))


def _params(cfg, objective, alpha=None, seed=0):
    p = dict(objective=objective, learning_rate=cfg["learning_rate"], num_leaves=cfg["num_leaves"],
             min_data_in_leaf=cfg["min_data_in_leaf"], feature_fraction=cfg["feature_fraction"],
             bagging_fraction=cfg["bagging_fraction"], bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=seed)
    if alpha is not None:
        p["alpha"] = alpha
    return p


class TreeForecaster:
    """Poisson-loss mean model; P10 / P50 / P90 from a negative binomial spread measured out of sample."""

    def __init__(self, cfg, quantiles, log=print, seed=0):
        self.cfg, self.quantiles, self.log, self.seed = cfg, quantiles, log, seed
        self.models = {}

    def fit(self, data, origin_idx, horizon):
        cfg = self.cfg
        cut = origin_idx - VAL_DAYS
        first = int(data.first_day.min()) + MIN_HISTORY - 1
        train_origins = np.arange(first, cut, cfg["origin_step"])
        val_origins = [cut - 21, cut - 14, cut - 7, cut]
        Xtr, ytr, mtr = make_rows(data, train_origins, horizon, target_hi=cut)
        Xva, yva, mva = make_rows(data, val_origins, horizon, target_lo=cut, target_hi=origin_idx)
        if cfg.get("relative_target", True):
            # predict y / level with weight = level. For the Poisson and quantile losses this is the same as
            # predicting y with log(level) as a fixed offset, so sale-day and festival multipliers are shared
            # across products of every size.
            wtr, wva = mtr["level"].values, mva["level"].values
            ytr, yva = ytr / wtr, yva / wva
        else:
            wtr, wva = None, None
        # columns with no data yet (e.g. "last year" in the first year) are dropped
        self.columns = [c for c in Xtr.columns if Xtr[c].notna().any()]
        Xtr, Xva = Xtr[self.columns], Xva[self.columns]
        self.log(f"    tree [{BACKEND}]: {len(Xtr):,} training rows, {len(Xva):,} check rows, {Xtr.shape[1]} features")
        if BACKEND == "lightgbm":
            dtr = lgb.Dataset(Xtr, ytr, weight=wtr, categorical_feature=["sku", "category"], free_raw_data=False)
            dva = lgb.Dataset(Xva, yva, weight=wva, reference=dtr)
            booster = lgb.train(_params(cfg, "poisson", None, self.seed), dtr, num_boost_round=cfg["max_rounds"],
                                valid_sets=[dva], callbacks=[lgb.early_stopping(cfg["early_stopping"], verbose=False),
                                                             lgb.log_evaluation(0)])
            self.models["mean"] = booster
            self.log(f"      best round {booster.best_iteration}")
        else:
            from sklearn.ensemble import HistGradientBoostingRegressor
            model = HistGradientBoostingRegressor(
                loss="poisson", learning_rate=cfg["learning_rate"], max_iter=min(cfg["max_rounds"], 500),
                max_leaf_nodes=cfg["num_leaves"], min_samples_leaf=cfg["min_data_in_leaf"],
                categorical_features=[0, 1], early_stopping=True, validation_fraction=0.1,
                n_iter_no_change=20, random_state=self.seed)
            model.fit(Xtr, ytr, sample_weight=wtr)
            self.models["mean"] = model
            self.log(f"      rounds {model.n_iter_}")

        # Spread for P10 / P50 / P90: negative binomial around the mean forecast, with the spread
        # (variance = mean + alpha * mean^2) measured per category on the check period (out of sample).
        mu = self._predict("mean", Xva) * (mva["level"].values if wva is not None else 1.0)
        y_true = data.y[mva["sku_idx"].values, mva["date_idx"].values]
        cats = data.cat_id[mva["sku_idx"].values]
        self.alpha_all = _alpha(y_true, mu)
        self.alpha = {c: (_alpha(y_true[cats == c], mu[cats == c]) if (cats == c).sum() >= 200 else self.alpha_all)
                      for c in range(len(data.categories))}
        self.log("      spread (alpha) by category: " + ", ".join(
            f"{data.categories[c]} {a:.2f}" for c, a in self.alpha.items()))
        return self

    def _predict(self, name, X):
        X = X[self.columns]
        m = self.models[name]
        if BACKEND == "lightgbm":
            return m.predict(X, num_iteration=m.best_iteration)
        return m.predict(X)

    def forecast(self, data, origin_idx, horizon):
        from scipy.stats import nbinom
        X, _, meta = make_rows(data, [origin_idx], horizon, training=False)
        mult = meta["level"].values if self.cfg.get("relative_target", True) else 1.0
        out = pd.DataFrame({"sku": np.array(data.skus)[meta["sku_idx"].values],
                            "date": data.dates[meta["date_idx"].values], "h": meta["h"].values})
        # safety cap: never more than 3x the product's best day so far (+5)
        cap = 3 * data.y[meta["sku_idx"].values, :origin_idx + 1].max(1) + 5
        mean = np.clip(self._predict("mean", X) * mult, 1e-6, cap)
        out["mean"] = mean
        alpha = np.array([self.alpha[c] for c in data.cat_id[meta["sku_idx"].values]])
        r = 1.0 / alpha
        for q in self.quantiles:
            out[f"p{int(q * 100)}"] = nbinom.ppf(q, r, r / (r + mean))
        return out

    def feature_importance(self):
        m = self.models.get("mean")
        if BACKEND != "lightgbm" or m is None:
            return None
        return pd.DataFrame({"feature": m.feature_name(), "gain": m.feature_importance("gain")}) \
            .sort_values("gain", ascending=False).reset_index(drop=True)

    def save(self, folder):
        import json
        m = self.models["mean"]
        if BACKEND == "lightgbm":
            m.save_model(str(folder / "tree_mean.txt"), num_iteration=m.best_iteration)
        else:
            import joblib
            joblib.dump(m, folder / "tree_mean.joblib")
        (folder / "tree_spread.json").write_text(json.dumps(
            dict(columns=self.columns, alpha_by_category=self.alpha, alpha_all=self.alpha_all), indent=2))
