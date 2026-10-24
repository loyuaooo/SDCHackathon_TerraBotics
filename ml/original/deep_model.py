"""
deep_model.py - the deep learning model (PyTorch, trains on the GPU).

Architecture: sequence-to-sequence LSTM, DeepAR-style.
  - One model for all products ("global model"): each product has a learned embedding vector,
    so festive items can learn from each other and from the calendar.
  - Encoder LSTM reads the last 16 weeks (scaled sales, sold-out flags, calendar).
  - Decoder LSTM walks through the next 90 days using only things known in advance
    (calendar, holidays, sales events, same time last year).
  - Output for every future day: a negative binomial distribution (mean + spread), which suits
    daily unit counts with many zeros. From it we get the mean forecast and P10 / P50 / P90.
  - Loss: negative binomial log-likelihood on normal days. On sold-out days the true demand is
    unknown but at least the units sold, so those days use a censored likelihood, -log P(demand >= sold).
    This stops the model learning "festive peaks are small" just because stock ran out at the peak.
"""
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import nbinom

from config import MIN_HISTORY, VAL_DAYS


class Seq2SeqNB(nn.Module):
    def __init__(self, n_sku, n_cat, n_enc, n_dec, n_static, hidden=64, layers=2, dropout=0.15,
                 emb_sku=8, emb_cat=4):
        super().__init__()
        self.sku_emb = nn.Embedding(n_sku, emb_sku)
        self.cat_emb = nn.Embedding(n_cat, emb_cat)
        st = emb_sku + emb_cat + n_static
        drop = dropout if layers > 1 else 0.0
        self.encoder = nn.LSTM(n_enc + st, hidden, layers, batch_first=True, dropout=drop)
        self.decoder = nn.LSTM(n_dec + st, hidden, layers, batch_first=True, dropout=drop)
        self.head = nn.Sequential(nn.Linear(hidden + n_dec + st, hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, 2))

    def forward(self, enc_x, dec_x, sku, cat, static):
        st = torch.cat([self.sku_emb(sku), self.cat_emb(cat), static], dim=-1)            # [B, st]
        e = torch.cat([enc_x, st[:, None, :].expand(-1, enc_x.shape[1], -1)], dim=-1)
        _, state = self.encoder(e)
        d = torch.cat([dec_x, st[:, None, :].expand(-1, dec_x.shape[1], -1)], dim=-1)
        out, _ = self.decoder(d, state)
        z = self.head(torch.cat([out, d], dim=-1))
        mu = F.softplus(z[..., 0]) + 1e-4          # mean, in units of the window's scale
        alpha = F.softplus(z[..., 1]) + 1e-3       # spread: variance = mean + alpha * mean^2
        return mu, alpha


def nb_logpmf(y, m, alpha):
    """log P(Y = y) for a negative binomial with mean m and variance m + alpha * m^2."""
    r = 1.0 / alpha
    return (torch.lgamma(y + r) - torch.lgamma(r) - torch.lgamma(y + 1.0)
            + r * (torch.log(r) - torch.log(r + m)) + y * (torch.log(m) - torch.log(r + m)))


def censored_term(y, m, alpha, cmask):
    """Sum of -log P(Y >= y) over sold-out days: we only know demand was at least the units sold.
    P(Y >= y) = 1 - sum_{k < y} P(Y = k), computed exactly for every sold-out day in the batch."""
    sel = cmask > 0
    n = sel.sum()
    if int(n.item()) == 0:
        return y.new_zeros(()), y.new_zeros(())
    yc, mc, ac = y[sel], m[sel], alpha[sel]
    k = torch.arange(int(yc.max().item()), device=y.device, dtype=y.dtype)[None, :]      # 0 .. max(y)-1
    pmf = torch.exp(nb_logpmf(k, mc[:, None], ac[:, None])) * (k < yc[:, None]).to(y.dtype)
    survival = (1.0 - pmf.sum(1)).clamp(min=0.0) + 1e-6
    return -torch.log(survival).sum(), n.to(y.dtype)


def nb_loss(b, m, alpha):
    """Average loss per target day: exact likelihood on normal days + censored likelihood on sold-out days."""
    normal = -(nb_logpmf(b["y"], m, alpha) * b["mask"]).sum()
    cens, n_cens = censored_term(b["y"], m, alpha, b["cmask"])
    return (normal + cens) / (b["mask"].sum() + n_cens).clamp(min=1.0), b["mask"].sum() + n_cens


def to_device(batch, device):
    return {k: torch.from_numpy(np.ascontiguousarray(v)).to(device, non_blocking=True) for k, v in batch.items()}


def _forward(model, b):
    mu, alpha = model(b["enc_x"], b["dec_x"], b["sku"], b["cat"], b["static"])
    return mu * b["scale"][:, None], alpha


def build_model(data, wm, cfg):
    return Seq2SeqNB(len(data.skus), len(data.categories), wm.n_enc, wm.n_dec, wm.n_static,
                     hidden=cfg["hidden"], layers=cfg["layers"], dropout=cfg["dropout"],
                     emb_sku=cfg["emb_sku"], emb_cat=cfg["emb_cat"])


def train_deep(data, wm, origin_idx, cfg, seed, device, log=print):
    """Train one model on data up to origin_idx. Early stopping on the last VAL_DAYS days."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cut = origin_idx - VAL_DAYS
    train_w = wm.train_windows(cut, MIN_HISTORY)
    check_w = wm.check_windows(cut, MIN_HISTORY)
    model = build_model(data, wm, cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    bs = cfg["batch_size"]
    best, best_state, bad, history = float("inf"), None, 0, []
    n_params = sum(p.numel() for p in model.parameters())
    log(f"    deep seed {seed}: {len(train_w):,} training windows, {len(check_w)} check windows, "
        f"{n_params:,} parameters, device {device}")

    for epoch in range(1, cfg["max_epochs"] + 1):
        t0 = time.time()
        model.train()
        order = rng.permutation(len(train_w))
        total, steps = 0.0, 0
        for i in range(0, len(order), bs):
            w = train_w[order[i:i + bs]]
            b = to_device(wm.batch(w[:, 0], w[:, 1], -1, cut), device)
            m, alpha = _forward(model, b)
            loss, _ = nb_loss(b, m, alpha)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Training loss became {loss.item()} at epoch {epoch}. "
                                   "Try a lower learning rate: change DEEP['lr'] in config.py (e.g. 1e-3).")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
            steps += 1

        model.eval()
        check_loss, check_n = 0.0, 0.0
        with torch.no_grad():
            for i in range(0, len(check_w), 1024):
                w = check_w[i:i + 1024]
                b = to_device(wm.batch(w[:, 0], w[:, 1], cut, origin_idx), device)
                m, alpha = _forward(model, b)
                loss, n = nb_loss(b, m, alpha)
                n = float(n.item())
                if n == 0:
                    continue
                check_loss += loss.item() * n
                check_n += n
        check = check_loss / max(check_n, 1.0)
        sched.step(check)
        secs = time.time() - t0
        history.append(dict(seed=seed, epoch=epoch, train_loss=total / max(steps, 1), check_loss=check,
                            lr=opt.param_groups[0]["lr"], seconds=secs))
        improved = check < best - 1e-4
        if improved:
            best, bad = check, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        log(f"      epoch {epoch:3d}  train {total / max(steps, 1):.4f}  check {check:.4f}"
            f"{'  *' if improved else ''}  ({secs:.1f}s)")
        if bad >= cfg["patience"]:
            log(f"      early stop: no improvement for {cfg['patience']} epochs, best check loss {best:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history


@torch.no_grad()
def forecast_deep(models, data, wm, origin_idx, horizon, quantiles, device):
    """Average the seeds: mean of the means, mean of each quantile."""
    skus = data.eligible(origin_idx)
    b = to_device(wm.batch(skus, np.full(len(skus), origin_idx), origin_idx, origin_idx + horizon), device)
    means, qs = [], []
    for model in models:
        model.eval()
        m, alpha = _forward(model, b)
        m = m.double().cpu().numpy()
        a = alpha.double().cpu().numpy()
        r = 1.0 / a
        p = r / (r + m)
        means.append(m)
        qs.append(np.stack([nbinom.ppf(q, r, p) for q in quantiles], -1))
    mean = np.mean(means, 0)                       # [n_sku, H]
    q = np.sort(np.mean(qs, 0), axis=-1)           # [n_sku, H, n_q]
    dates = data.dates[origin_idx + 1: origin_idx + 1 + horizon]
    rows = []
    for i, s in enumerate(skus):
        df = pd.DataFrame({"sku": data.skus[s], "date": dates, "h": np.arange(1, horizon + 1), "mean": mean[i]})
        for j, qq in enumerate(quantiles):
            df[f"p{int(qq * 100)}"] = q[i, :, j]
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def save_model(model, path, data, wm, cfg, seed):
    torch.save(dict(state_dict=model.state_dict(), config=cfg, seed=seed, skus=data.skus,
                    categories=data.categories, calendar_features=data.cal_names,
                    n_enc=wm.n_enc, n_dec=wm.n_dec, n_static=wm.n_static), path)
