"""
config.py - every setting in one place.

Change numbers here, or override the common ones on the command line
(see `python train.py --help`).
"""

DATA_DIR = "data"
OUT_DIR = "outputs"

# --- forecasting problem ------------------------------------------------------------
HORIZON = 90            # forecast the next 90 days (covers lead time + safety + festive pre-orders)
MIN_HISTORY = 28        # a product needs 28 days of sales before we forecast or train on it
VAL_DAYS = 28           # the last 28 days before each origin are the early-stopping check set
QUANTILES = [0.1, 0.5, 0.9]   # P10 / P50 / P90: P90 is what safety stock is based on

# Backtest: pretend "today" is each of these dates, train only on data before it,
# forecast the next 90 days and compare with what really happened.
# The final run uses all history ("today" = last day of data) and is scored on the
# hidden future in data/_ground_truth/future_demand.csv.
BACKTEST_ORIGINS = ["2025-10-01", "2026-01-01", "2026-04-01", "2026-07-01"]

# The model with the best average backtest score on this metric is used for the final forecast.
# wape_weekly = error on weekly totals per product (restocking decisions are made weekly).
PRIMARY_METRIC = "wape_weekly"

# --- deep learning model (PyTorch, runs on the GPU) ------------------------------------
DEEP = dict(
    enc_len=112,         # looks back 16 weeks
    hidden=64,           # LSTM size
    layers=2,
    dropout=0.15,
    emb_sku=8,           # each product gets its own learned "personality" vector
    emb_cat=4,
    lr=2e-3,
    weight_decay=1e-4,
    batch_size=256,
    max_epochs=60,
    patience=8,          # stop when the check-set loss hasn't improved for 8 epochs
    seeds=3,             # train 3 copies with different random seeds and average them
)

# --- tree model (LightGBM; falls back to scikit-learn if LightGBM is missing) ---------
# Learns "future sales / recent 28-day level" with a Poisson loss; P10/P50/P90 come from a
# negative binomial spread measured on the check period.
TREE = dict(
    relative_target=True,
    origin_step=7,       # one training "snapshot" per week of history
    learning_rate=0.05,
    num_leaves=63,
    min_data_in_leaf=40,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    max_rounds=3000,
    early_stopping=100,
)

# --- quick smoke test (python train.py --quick) ---------------------------------------
QUICK = dict(folds=1, seeds=1, max_epochs=3, max_rounds=150)
