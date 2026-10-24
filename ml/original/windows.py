"""
windows.py - cuts the sales history into (look-back, look-ahead) windows for the deep model.

Pure numpy (no PyTorch), so it can be tested anywhere. One window for product s with
split day t:
    encoder (look-back)  days t-L+1 .. t   : past sales (scaled, log1p), sold-out flags, calendar
    decoder (look-ahead) days t+1 .. t+H   : calendar + "same time last year" + "same point of last
                                             year's festive season" (all known in advance)
    target                days t+1 .. t+H   : sales on trustworthy days inside the allowed range, plus
                                             a "demand was at least this" mark on sold-out days

Each window is scaled by its own average (1 + mean of the look-back), so a cable selling 25/day
and a projector selling 0.2/day look alike to the network.
"""
import numpy as np

N_ENC_OWN = 5       # scaled sales, listed, sold out, last-year level, last-year available
N_DEC_OWN = 5       # last-year level + available, last festive season level + available, position
N_STATIC = 3        # log(scale), log(price), age


class WindowMaker:
    def __init__(self, data, enc_len, horizon):
        self.L, self.H, self.pad = enc_len, horizon, enc_len
        S, T = data.y.shape
        P = self.pad
        zf = np.zeros((S, P), dtype=np.float32)
        zb = np.zeros((S, P), dtype=bool)
        self.y = np.concatenate([zf, data.y], 1).astype(np.float32)
        self.valid = np.concatenate([zb, data.valid], 1)
        self.listed = np.concatenate([zb, data.listed], 1)
        self.censored = np.concatenate([zb, data.censored], 1)
        ly = np.concatenate([np.full((S, P), np.nan, dtype=np.float32), data.last_year.astype(np.float32)], 1)
        self.ly_avail = ~np.isnan(ly)
        self.ly = np.nan_to_num(ly).astype(np.float32)
        ls = np.concatenate([np.full((S, P), np.nan, dtype=np.float32), data.season_any.astype(np.float32)], 1)
        self.ls_avail = ~np.isnan(ls)
        self.ls = np.nan_to_num(ls).astype(np.float32)
        self.cal = np.concatenate([np.zeros((P, data.cal.shape[1]), dtype=np.float32), data.cal], 0)
        self.first_day = data.first_day
        self.log_price = np.log1p(data.price).astype(np.float32)
        self.cat_id = data.cat_id
        self.n_cal = data.cal.shape[1]
        self.n_enc = N_ENC_OWN + self.n_cal
        self.n_dec = N_DEC_OWN + self.n_cal
        self.n_static = N_STATIC

    def batch(self, sku, split, target_lo, target_hi):
        """sku, split: int arrays of equal length (split = last look-back day, unpadded index).
        Targets are kept only where target_lo < day <= target_hi (unpadded indices)."""
        sku = np.asarray(sku)
        split = np.asarray(split)
        L, H, P = self.L, self.H, self.pad
        s = sku[:, None]
        enc_t = split[:, None] + P + np.arange(-L + 1, 1)[None, :]          # [B, L]
        dec_t = split[:, None] + P + np.arange(1, H + 1)[None, :]           # [B, H]

        y_enc = self.y[s, enc_t]
        v_enc = self.valid[s, enc_t]
        n = v_enc.sum(1)
        scale = 1.0 + np.where(n > 0, (y_enc * v_enc).sum(1) / np.maximum(n, 1), 0.0)
        sc = scale[:, None].astype(np.float32)

        # log1p keeps big spikes (11.11, festive peaks of 50x normal) in a range the LSTM handles well
        enc_own = np.stack([np.log1p(y_enc / sc), self.listed[s, enc_t], self.censored[s, enc_t],
                            np.log1p(self.ly[s, enc_t] / sc), self.ly_avail[s, enc_t]], -1).astype(np.float32)
        enc_x = np.concatenate([enc_own, self.cal[enc_t]], -1)

        pos = np.broadcast_to(np.arange(1, H + 1, dtype=np.float32)[None, :] / H, dec_t.shape)
        dec_own = np.stack([np.log1p(self.ly[s, dec_t] / sc), self.ly_avail[s, dec_t],
                            np.log1p(self.ls[s, dec_t] / sc), self.ls_avail[s, dec_t], pos], -1).astype(np.float32)
        dec_x = np.concatenate([dec_own, self.cal[dec_t]], -1)

        day = dec_t - P
        in_range = (day > np.asarray(target_lo)) & (day <= np.asarray(target_hi))
        mask = self.valid[s, dec_t] & in_range
        # sold-out days with some sales: true demand was AT LEAST the units sold (used by the censored loss)
        y_dec = self.y[s, dec_t]
        cmask = self.censored[s, dec_t] & in_range & (y_dec > 0)
        age = np.clip(split - self.first_day[sku] + 1, 0, 730) / 365.0
        static = np.stack([np.log(scale), self.log_price[sku], age], -1).astype(np.float32)
        return dict(enc_x=enc_x, dec_x=dec_x, y=y_dec, mask=mask.astype(np.float32), cmask=cmask.astype(np.float32),
                    scale=scale.astype(np.float32), sku=sku.astype(np.int64),
                    cat=self.cat_id[sku].astype(np.int64), static=static)

    def train_windows(self, cut, min_history):
        """All (product, split) pairs with at least min_history days on sale and a target day <= cut."""
        pairs = []
        for s, first in enumerate(self.first_day):
            lo = first + min_history - 1
            if lo <= cut - 1:
                splits = np.arange(lo, cut)
                pairs.append(np.column_stack([np.full(len(splits), s), splits]))
        return np.concatenate(pairs) if pairs else np.zeros((0, 2), dtype=int)

    def check_windows(self, cut, min_history):
        """Windows whose targets fall in the check period (cut, origin]."""
        pairs = []
        for s, first in enumerate(self.first_day):
            for split in (cut - 21, cut - 14, cut - 7, cut):
                if split - first + 1 >= min_history:
                    pairs.append((s, split))
        return np.array(pairs, dtype=int).reshape(-1, 2)
