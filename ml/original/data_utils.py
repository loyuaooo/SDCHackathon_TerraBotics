"""
data_utils.py - loads the shop data into (product x day) arrays and builds calendar features.

Input files (in the data folder):
    sales_daily.csv      date, sku, units_sold, stock_end
    products.csv         sku, category, price_rm
    calendar_events.csv  date, event, event_type, importance   (mega sales, festivals, holidays)
    calendar_daily.csv   one row per day: holiday flags (optional but recommended)

Stockouts: on a day that ends with stock_end = 0 the product sold out, so sales understate
demand. Those days are flagged as "censored" and are never used as training targets.
"""
from pathlib import Path

import numpy as np
import pandas as pd

CLIP_FESTIVAL = 120     # "days to / since festival" features are capped at this
CLIP_SALE = 30          # "days to / since big sale" features are capped at this
IMPORTANCE = {"minor": 1, "medium": 2, "major": 3}
HOLIDAY_KINDS = ["family", "festival", "rest", "regional"]


def slug(text):
    return "".join(ch if ch.isalnum() else "_" for ch in str(text).lower()).strip("_")


def _to_and_since(dates, event_dates, clip):
    """Days until the next event (on/after each day) and since the last event (before it), capped."""
    day = dates.values.astype("datetime64[D]").astype(np.int64)
    ev = np.sort(pd.to_datetime(pd.Series(event_dates)).values.astype("datetime64[D]").astype(np.int64))
    if len(ev) == 0:
        full = np.full(len(day), float(clip))
        return full, full
    i = np.searchsorted(ev, day, side="left")
    nxt = np.where(i < len(ev), ev[np.minimum(i, len(ev) - 1)] - day, clip)
    j = i - 1
    prv = np.where(j >= 0, day - ev[np.maximum(j, 0)], clip)
    return np.minimum(nxt, clip).astype(float), np.minimum(prv, clip).astype(float)


def calendar_features(dates, events, daily=None):
    """One row per date, all values roughly in 0..1. Everything here is known in advance."""
    f = {}
    dow = dates.dayofweek.values
    for k in range(7):
        f[f"dow_{k}"] = (dow == k).astype(float)
    doy = dates.dayofyear.values
    f["year_sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["year_cos"] = np.cos(2 * np.pi * doy / 365.25)
    f["day_of_month"] = (dates.day.values - 1) / 30.0
    f["payday_window"] = ((dates.day.values >= 25) & (dates.day.values <= 28)).astype(float)

    # public holidays / days off / long weekends
    if daily is not None:
        d = daily.set_index("date").reindex(dates)
        if d["is_day_off"].isna().any():
            missing = dates[d["is_day_off"].isna().values]
            raise ValueError(f"calendar_daily.csv does not cover {missing.min().date()} .. {missing.max().date()}. "
                             "Regenerate the data with a longer calendar.")
        kind = d["holiday_kind"].fillna("").astype(str).values
        for k in HOLIDAY_KINDS:
            f[f"holiday_{k}"] = (kind == k).astype(float)
        f["day_off"] = d["is_day_off"].astype(bool).values.astype(float)
        f["long_weekend"] = d["in_long_weekend"].astype(bool).values.astype(float)
    else:
        hol = events[events["event_type"].isin(["festival", "public_holiday"])]
        is_hol = dates.isin(pd.to_datetime(hol["date"]))
        f["holiday_any"] = is_hol.astype(float)
        f["day_off"] = ((dow >= 5) | is_hol).astype(float)

    # Shopee / Lazada double-day sales
    mega = events[events["event_type"] == "mega_sale"]
    level = pd.Series(mega["importance"].map(IMPORTANCE).values, index=pd.to_datetime(mega["date"]))
    level = level[~level.index.duplicated()]
    f["mega_level"] = level.reindex(dates).fillna(0).values / 3.0
    big = mega.loc[mega["importance"].isin(["medium", "major"]), "date"]
    to_big, since_big = _to_and_since(dates, big, CLIP_SALE)
    f["days_to_big_sale"], f["days_since_big_sale"] = to_big / CLIP_SALE, since_big / CLIP_SALE
    major = mega.loc[mega["importance"] == "major", "date"]
    to_major, _ = _to_and_since(dates, major, 60)
    f["days_to_major_sale"] = to_major / 60.0

    # festive seasons and school start: countdown and "days since"
    fests = events[events["event_type"].isin(["festival", "school"])]
    for name, grp in fests.groupby("event"):
        k = slug(name)
        to_f, since_f = _to_and_since(dates, grp["date"], CLIP_FESTIVAL)
        f[f"days_to_{k}"] = to_f / CLIP_FESTIVAL
        f[f"days_since_{k}"] = since_f / CLIP_FESTIVAL

    names = list(f)
    return np.column_stack([f[n] for n in names]).astype(np.float32), names


class ShopData:
    """Sales history as dense (product x day) arrays, plus calendar features up to the forecast horizon."""

    def __init__(self, data_dir, horizon):
        d = Path(data_dir)
        sales = pd.read_csv(d / "sales_daily.csv", parse_dates=["date"])
        products = pd.read_csv(d / "products.csv")
        events = pd.read_csv(d / "calendar_events.csv", parse_dates=["date"])
        daily = pd.read_csv(d / "calendar_daily.csv", parse_dates=["date"]) if (d / "calendar_daily.csv").exists() else None

        self.data_dir = d
        self.horizon = horizon
        self.products = products.reset_index(drop=True)
        self.skus = self.products["sku"].astype(str).tolist()
        self.sku_index = {s: i for i, s in enumerate(self.skus)}
        self.categories = sorted(self.products["category"].astype(str).unique())
        self.cat_id = self.products["category"].astype(str).map({c: i for i, c in enumerate(self.categories)}).values
        self.price = self.products["price_rm"].astype(float).values if "price_rm" in self.products else np.ones(len(self.skus))
        self.events = events

        sales["sku"] = sales["sku"].astype(str)
        unknown = set(sales["sku"]) - set(self.skus)
        if unknown:
            raise ValueError(f"SKUs in sales_daily.csv but not in products.csv: {sorted(unknown)[:5]}")
        self.start, self.end = sales["date"].min(), sales["date"].max()
        self.n_hist = (self.end - self.start).days + 1                  # days of history
        self.dates = pd.date_range(self.start, self.end + pd.Timedelta(days=horizon), freq="D")
        S, T = len(self.skus), len(self.dates)

        y = np.full((S, T), np.nan)
        stock = np.full((S, T), np.nan)
        si = sales["sku"].map(self.sku_index).values
        ti = (sales["date"] - self.start).dt.days.values
        y[si, ti] = sales["units_sold"].clip(lower=0).values
        if "stock_end" in sales:
            stock[si, ti] = sales["stock_end"].values
        else:
            stock[si, ti] = 1.0                                          # no stock data: trust every day

        self.listed = ~np.isnan(y)                                       # product was on sale that day
        self.censored = self.listed & (stock <= 0)                       # sold out: true demand unknown
        self.valid = self.listed & ~self.censored                        # trustworthy days
        self.y = np.nan_to_num(y).astype(np.float32)
        first = np.where(self.listed.any(1), self.listed.argmax(1), T)
        self.first_day = first

        self.cal, self.cal_names = calendar_features(self.dates, events, daily)

        # running sums of trustworthy sales -> fast window averages
        clean = np.where(self.valid, self.y, 0.0)
        z = np.zeros((S, 1))
        self._cs = np.concatenate([z, np.cumsum(clean, 1)], 1)
        self._cs2 = np.concatenate([z, np.cumsum(clean ** 2, 1)], 1)
        self._cn = np.concatenate([z, np.cumsum(self.valid, 1)], 1)
        self._cz = np.concatenate([z, np.cumsum(self.valid & (self.y == 0), 1)], 1)
        self._ccen = np.concatenate([z, np.cumsum(self.censored, 1)], 1)
        self._clist = np.concatenate([z, np.cumsum(self.listed, 1)], 1)

        # "same time last year": average of trustworthy sales over 7 days centred on t - 364
        t = np.arange(T)
        a, b = t - 367, t - 361
        self.last_year = self.window_mean(np.arange(S)[:, None], b[None, :], 7, a_override=a[None, :])

        # "same point in last year's festive season": if a festival is within 60 days, the sales at the
        # same number of days before LAST year's festival (moving festivals like Raya and Deepavali shift
        # every year, so "same date last year" would miss them)
        self.season_names, aligned = [], []
        fests = events[events["event_type"].isin(["festival", "school"])]
        tt = np.arange(T)
        for name, grp in fests.groupby("event"):
            fi = np.array(sorted({(pd.Timestamp(x) - self.start).days for x in grp["date"]}))
            j = np.searchsorted(fi, tt, side="left")
            has_next = j < len(fi)
            k = np.where(has_next, fi[np.minimum(j, len(fi) - 1)] - tt, 999)
            ok = has_next & (j >= 1) & (k <= 60)
            centre = np.where(ok, fi[np.maximum(j - 1, 0)] - k, -999)
            vals = self.window_mean(np.arange(S)[:, None], (centre + 3)[None, :], 7, a_override=(centre - 3)[None, :])
            vals = np.where(ok[None, :], vals, np.nan)
            self.season_names.append(slug(name))
            aligned.append(vals)
        self.season_aligned = np.stack(aligned) if aligned else np.full((0, S, T), np.nan)
        with np.errstate(all="ignore"):
            self.season_any = np.fmax.reduce(self.season_aligned, axis=0) if aligned else np.full((S, T), np.nan)

    def idx(self, date):
        return int((pd.Timestamp(date) - self.start).days)

    def window_mean(self, s, end, w, a_override=None):
        """Mean of trustworthy sales over [end - w + 1, end] (NaN if none). s, end broadcast together."""
        end = np.asarray(end)
        a = (end - w + 1) if a_override is None else np.asarray(a_override)
        lo = np.clip(a, 0, self._cs.shape[1] - 1)
        hi = np.clip(end + 1, 0, self._cs.shape[1] - 1)
        n = self._cn[s, hi] - self._cn[s, lo]
        tot = self._cs[s, hi] - self._cs[s, lo]
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(n > 0, tot / np.maximum(n, 1), np.nan)
        return np.where(end < 0, np.nan, out)

    def window_stats(self, s, end, w):
        """mean, std, share of zero days, share of sold-out days over the last w days up to `end`."""
        end = np.asarray(end)
        lo = np.clip(end - w + 1, 0, None)
        hi = end + 1
        n = self._cn[s, hi] - self._cn[s, lo]
        tot = self._cs[s, hi] - self._cs[s, lo]
        tot2 = self._cs2[s, hi] - self._cs2[s, lo]
        zeros = self._cz[s, hi] - self._cz[s, lo]
        cen = self._ccen[s, hi] - self._ccen[s, lo]
        lst = self._clist[s, hi] - self._clist[s, lo]
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(n > 0, tot / np.maximum(n, 1), np.nan)
            var = np.where(n > 1, tot2 / np.maximum(n, 1) - mean ** 2, np.nan)
            std = np.sqrt(np.clip(var, 0, None))
            zero_share = np.where(n > 0, zeros / np.maximum(n, 1), np.nan)
            soldout_share = np.where(lst > 0, cen / np.maximum(lst, 1), 0.0)
        return mean, std, zero_share, soldout_share

    def eligible(self, origin_idx):
        """Products with at least MIN_HISTORY days on sale by the origin day."""
        from config import MIN_HISTORY
        return np.where(origin_idx - self.first_day + 1 >= MIN_HISTORY)[0]
