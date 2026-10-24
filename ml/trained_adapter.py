"""Adapter for the user's unchanged training modules and checkpoint format."""
import json
import shutil
import sys
import tempfile
from pathlib import Path


def predict_ensemble(product, history, anchor, horizon, model_dir):
    # These optional dependencies are needed only once training metadata is supplied.
    import numpy as np
    import pandas as pd
    import torch
    import lightgbm as lgb
    from scipy.stats import nbinom
    sys.path.insert(0, str(Path(__file__).parent / "original"))
    from data_utils import ShopData
    from windows import WindowMaker
    from deep_model import Seq2SeqNB, forecast_deep
    from tree_model import TreeForecaster
    from config import TREE

    torch.set_num_threads(1)
    checkpoints = [torch.load(model_dir / f"deep_seed{i}.pt", map_location="cpu", weights_only=True) for i in range(3)]
    meta = checkpoints[0]
    sku = str(product["barcode"])
    if sku not in meta["skus"]:
        raise ValueError("barcode is not an original trained SKU; retrain or provide the original SKU")
    # Never assign an unseen SKU to another product's learned embedding.
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for filename in ("products.csv", "calendar_events.csv", "calendar_daily.csv"):
            shutil.copyfile(model_dir / "training_data" / filename, tmp / filename)
        products = pd.read_csv(tmp / "products.csv")
        if products["sku"].astype(str).tolist() != meta["skus"]:
            raise ValueError("products.csv must preserve the checkpoint's original SKU ordering")
        # The live application's dated observations are the only sales input.
        rows = [{"date":r["date"], "sku":sku, "units_sold":r["unitsSold"]} for r in history]
        pd.DataFrame(rows).to_csv(tmp / "sales_daily.csv", index=False)
        extra_days = (pd.Timestamp(anchor) - pd.Timestamp(max(r["date"] for r in history))).days
        data = ShopData(tmp, 90 + extra_days)
        if data.categories != meta["categories"] or data.cal_names != meta["calendar_features"]:
            raise ValueError("calendar features or category ordering differ from training")
        origin = data.idx(anchor)
        models = []
        for checkpoint in checkpoints:
            if checkpoint["skus"] != meta["skus"] or checkpoint["calendar_features"] != meta["calendar_features"]:
                raise ValueError("checkpoint metadata differ between seeds")
            cfg = checkpoint["config"]
            model = Seq2SeqNB(len(meta["skus"]), len(meta["categories"]), checkpoint["n_enc"],
                             checkpoint["n_dec"], checkpoint["n_static"],
                             **{k:cfg[k] for k in ("hidden","layers","dropout","emb_sku","emb_cat")})
            model.load_state_dict(checkpoint["state_dict"])
            models.append(model.eval())
        # Keep the training horizon (90) for the decoder-position feature.
        wm = WindowMaker(data, meta["config"]["enc_len"], 90)
        deep = forecast_deep(models, data, wm, origin, 90, [.1,.5,.9], "cpu")
        spread = json.loads((model_dir / "tree_spread.json").read_text())
        tree = TreeForecaster(TREE,[.1,.5,.9],log=lambda *args:None)
        tree.columns = spread["columns"]
        tree.alpha = {int(k):v for k,v in spread["alpha_by_category"].items()}
        tree.alpha_all = spread["alpha_all"]
        tree.models["mean"] = lgb.Booster(model_file=str(model_dir / "tree_mean.txt"))
        trees = tree.forecast(data,origin,90)
        deep = deep[deep.sku==sku].sort_values("h").head(horizon)
        trees = trees[trees.sku==sku].sort_values("h").head(horizon)
        if len(deep)!=horizon or len(trees)!=horizon:
            raise ValueError("not enough eligible history for the requested forecast")
        result = []
        for d,t in zip(deep.to_dict("records"),trees.to_dict("records")):
            values = [(d[k]+t[k])/2 for k in ("mean","p10","p90")]
            if not all(np.isfinite(v) for v in values):
                raise ValueError("model produced a non-finite result")
            result.append(dict(date=d["date"].strftime("%Y-%m-%d"),predicted=round(values[0],2),
                               lower=round(values[1],2),upper=round(values[2],2)))
        return result
