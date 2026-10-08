"""Runtime evaluation harness for the saved models.

This module NEVER retrains, modifies or reimplements the models. It only:

  1. downloads historical OHLCV data from Yahoo Finance,
  2. rebuilds features with the existing feature_engineering.py,
  3. replays the SAVED GA-XGBoost and LSTM artefacts over a held-out slice,
  4. computes standard classification metrics against next-day direction.

Two evaluation windows are supported:
  * "test_split" – reproduces the project split from universal_config.json
    (70/15/15 per stock with a 1-row embargo) on data ending at the training
    cut-off date, i.e. the held-out tail the models never saw.
  * "forward"    – rows dated AFTER the training cut-off, i.e. true
    out-of-sample inference on unseen market history.

Metrics are computed at runtime and are clearly labelled as such.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from feature_engineering import build_features_for_stock
from prediction_backend import normalise_market_df


EVAL_START = "2015-01-01"          # matches supported_stocks.json first_date
FORWARD_WARMUP_START = "2023-01-01"  # ~1 year of rows before the cut-off for warm-up
CHUNK = 20                          # tickers per yfinance request


@st.cache_data(ttl=4 * 3600, show_spinner=False)
def download_history(tickers, start: str, end: str):
    """Batch-download OHLCV for the universe. Returns (frames, failed_tickers)."""
    tickers = list(tickers)
    frames: dict[str, pd.DataFrame] = {}
    failed: list[str] = []

    for i in range(0, len(tickers), CHUNK):
        chunk = tickers[i : i + CHUNK]
        ns_symbols = [f"{t}.NS" for t in chunk]
        try:
            data = yf.download(
                ns_symbols,
                start=start,
                end=end,
                interval="1d",
                auto_adjust=False,
                progress=False,
                actions=False,
                threads=True,
                group_by="column",
            )
        except Exception:
            failed.extend(chunk)
            continue

        if data is None or len(data) == 0:
            failed.extend(chunk)
            continue

        multi = isinstance(data.columns, pd.MultiIndex)
        level = set(data.columns.get_level_values(-1)) if multi else set()

        for t, sym in zip(chunk, ns_symbols):
            try:
                if multi:
                    if sym not in level:
                        failed.append(t)
                        continue
                    sub = data.xs(sym, level=-1, axis=1)
                else:
                    if len(chunk) != 1:
                        failed.append(t)
                        continue
                    sub = data
                if sub is None or sub.empty:
                    failed.append(t)
                    continue
                frames[t] = normalise_market_df(sub.reset_index(), t)
            except Exception:
                failed.append(t)

    return frames, failed


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, score) -> dict:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": None,
    }
    if score is not None and len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, score))
    return out


def _downsample(fpr, tpr, points=450):
    if len(fpr) <= points:
        return list(map(float, fpr)), list(map(float, tpr))
    idx = np.unique(np.linspace(0, len(fpr) - 1, points).astype(int))
    return [float(fpr[i]) for i in idx], [float(tpr[i]) for i in idx]


def evaluate_models(models, frames, mode: str, train_end: str, progress=None) -> dict:
    """Replay the saved models over downloaded history and compute metrics.

    models: the tuple returned by prediction_backend.load_models()
    frames: {ticker: normalised OHLCV DataFrame} from download_history()
    mode:   "test_split" | "forward"
    """
    xgb_model, lstm_model, scaler, features, config, _ = models

    all_features = features["all_features"]
    ga_features = features["ga_selected_features"]
    lstm_features = features["lstm_features"]
    window = int(features["lstm_window"])
    clip_z = float(config["lstm"]["clip_z"])
    xgb_thr = float(config["thresholds"].get("GA-XGBoost", 0.5))
    lstm_thr = float(config["thresholds"].get("Universal LSTM", 0.5))
    split = config.get("split", {})
    train_frac = float(split.get("train", 0.7))
    val_frac = float(split.get("val", 0.15))
    embargo = int(split.get("embargo_rows", 1))
    min_rows = int(config["feature_engineering"]["min_rows_needed_for_lstm_inference"])

    cutoff = pd.Timestamp(train_end)
    y_all, px_all, pl_all = [], [], []
    ref_y = []           # rows the model was allowed to train on -> majority baseline
    per_stock = []
    skipped: list[str] = []
    dates_all = []
    n_total = len(frames)

    for i, (ticker, df) in enumerate(sorted(frames.items())):
        if progress is not None and n_total:
            progress(min(i / n_total, 1.0), ticker)
        try:
            g = df.sort_values("Date").reset_index(drop=True)
            if len(g) < min_rows:
                skipped.append(ticker)
                continue

            feat = build_features_for_stock(g)
            usable = feat.replace([np.inf, -np.inf], np.nan).dropna(subset=all_features)

            # Ground truth: next-day direction, exactly as in universal_config.json
            #   UP = 1 if Close[t+1]/Close[t] - 1 > 0 else DOWN = 0
            nxt = g["Close"].shift(-1) / g["Close"] - 1.0
            label_ok = nxt.notna().loc[usable.index]
            usable = usable[label_ok]
            if usable.empty:
                skipped.append(ticker)
                continue
            y = (nxt.loc[usable.index] > 0).astype(int)
            dates = g.loc[usable.index, "Date"]

            if mode == "test_split":
                n = len(usable)
                n_train = int(n * train_frac)
                n_val = int(n * val_frac)
                test_start = n_train + embargo + n_val + embargo
                if test_start >= n:
                    skipped.append(ticker)
                    continue
                eval_pos = np.arange(test_start, n)
                ref = y.iloc[:n_train]
            else:
                eval_mask = (dates > cutoff).to_numpy()
                if eval_mask.sum() < 20:
                    skipped.append(ticker)
                    continue
                eval_pos = np.flatnonzero(eval_mask)
                ref = y[dates <= cutoff]
                if len(ref) < 20:
                    skipped.append(ticker)
                    continue

            # Each evaluated row needs `window` earlier rows for the LSTM sequence.
            eval_pos = eval_pos[eval_pos >= window]
            if len(eval_pos) == 0:
                skipped.append(ticker)
                continue

            y_eval = y.to_numpy()[eval_pos]

            # --- GA-XGBoost: GA-selected features of each evaluated row ---
            X = usable.iloc[eval_pos][ga_features].to_numpy(dtype=np.float32)
            p_xgb = xgb_model.predict_proba(X)[:, 1]

            # --- LSTM: 60-day sequence -> scaler -> clip -> model ---
            S = scaler.transform(usable[lstm_features].to_numpy(dtype=np.float32))
            S = np.clip(S, -clip_z, clip_z).astype(np.float32)
            seqs = np.stack([S[p - window : p] for p in eval_pos])
            p_lstm = lstm_model.predict(seqs, batch_size=512, verbose=0).ravel().astype(float)

            y_all.append(y_eval)
            px_all.append(p_xgb)
            pl_all.append(p_lstm)
            ref_y.append(ref.to_numpy())
            dates_all.append(dates.to_numpy()[eval_pos])
            per_stock.append(
                {
                    "stock": ticker,
                    "n": int(len(y_eval)),
                    "xgb_acc": float(accuracy_score(y_eval, p_xgb >= xgb_thr)),
                    "lstm_acc": float(accuracy_score(y_eval, p_lstm >= lstm_thr)),
                }
            )
        except Exception:
            skipped.append(ticker)
            continue

    if progress is not None:
        progress(1.0, "aggregating")

    if not y_all:
        raise ValueError(
            "No stocks could be evaluated — the market data download appears to have failed. "
            "Check the connection and try again."
        )

    y = np.concatenate(y_all)
    p_xgb = np.concatenate(px_all)
    p_lstm = np.concatenate(pl_all)
    ref = np.concatenate(ref_y)
    dates_cat = np.concatenate(dates_all)

    pred_xgb = (p_xgb >= xgb_thr).astype(int)
    pred_lstm = (p_lstm >= lstm_thr).astype(int)

    # Majority baseline: always predict the class most frequent in the data the
    # model was allowed to train on (per-stock train slice / pre-cut-off rows).
    majority_class = int(np.bincount(ref.astype(int), minlength=2).argmax())
    pred_base = np.full_like(y, majority_class)

    fpr_x, tpr_x = roc_curve(y, p_xgb)[:2] if len(np.unique(y)) > 1 else (np.array([0, 1]), np.array([0, 1]))
    fpr_l, tpr_l = roc_curve(y, p_lstm)[:2] if len(np.unique(y)) > 1 else (np.array([0, 1]), np.array([0, 1]))
    rx, ry = _downsample(fpr_x, tpr_x)
    lx, ly = _downsample(fpr_l, tpr_l)

    return {
        "mode": mode,
        "stocks_requested": int(n_total),
        "stocks_evaluated": int(len(per_stock)),
        "stocks_skipped": skipped,
        "rows": int(len(y)),
        "up_rate": float(y.mean()),
        "date_min": str(pd.Timestamp(dates_cat.min()).date()),
        "date_max": str(pd.Timestamp(dates_cat.max()).date()),
        "thresholds": {
            "xgb": xgb_thr,
            "lstm": lstm_thr,
            "baseline_class": "UP" if majority_class == 1 else "DOWN",
            "reference_rows": int(len(ref)),
        },
        "metrics": {
            "xgb": _metrics(y, pred_xgb, p_xgb),
            "lstm": _metrics(y, pred_lstm, p_lstm),
            "baseline": _metrics(y, pred_base, None),
        },
        "confusion": {
            "xgb": confusion_matrix(y, pred_xgb, labels=[0, 1]).tolist(),
            "lstm": confusion_matrix(y, pred_lstm, labels=[0, 1]).tolist(),
        },
        "roc": {
            "xgb": {"fpr": rx, "tpr": ry},
            "lstm": {"fpr": lx, "tpr": ly},
        },
        "per_stock": per_stock,
    }


def run_evaluation(models, mode: str, train_end: str, n_stocks: int, tickers, progress=None) -> dict:
    """Download + evaluate convenience wrapper used by the UI."""
    if mode == "test_split":
        start, end = EVAL_START, "2024-01-01"  # end is exclusive; includes 2023-12-29
    else:
        start, end = FORWARD_WARMUP_START, None

    subset = list(tickers)[: int(n_stocks)]
    if end:
        frames, failed = download_history(tuple(subset), start, end)
    else:
        frames, failed = download_history(tuple(subset), start, "2099-01-01")

    if not frames:
        raise ValueError(
            "Unable to retrieve market data for the evaluation universe. "
            "Try again later or reduce the number of stocks."
        )

    result = evaluate_models(models, frames, mode, train_end, progress=progress)
    result["download_failed"] = failed
    return result
